# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What only a Trino source has: catalogs, a server's own access control,
and time columns with and without a zone.

The contract every source keeps is in ``test_tables_stay_where_they_are.py``,
which runs on Trino too.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
backend = pytest.importorskip('sdm_connectors.backends.trino')

from sdm_connectors import Source, TrinoSource  # noqa: E402

from sdm_client import (  # noqa: E402
    PredictiveQuery,
    Relationship,
    Schema,
)

QUERY = 'PREDICT SUM(orders.amount, 0, 30, days) FOR EACH users.user_id'
USERS = pd.DataFrame({'user_id': range(1, 11), 'age': range(20, 30)})
ORDERS = pd.DataFrame(
    {
        'order_id': range(1, 101),
        'user_id': [1 + index % 10 for index in range(100)],
        'amount': [float(index) for index in range(100)],
        'ts': pd.Timestamp('2024-01-01')
        + pd.to_timedelta(range(0, 100 * 36, 36), unit='h'),
    }
)
KEYS = {'users': 'user_id', 'orders': 'order_id'}
LINKS = [Relationship('orders', 'user_id', 'users', 'user_id')]


@pytest.fixture
def url() -> str:
    url = os.getenv('SDM_TRINO_TEST_URL')
    if url:
        return url
    if os.getenv('SDM_REQUIRE_TRINO'):
        pytest.fail('SDM_REQUIRE_TRINO is set but SDM_TRINO_TEST_URL is not')
    pytest.skip('SDM_TRINO_TEST_URL is not set')


def _run(connection: Any, sql: str, parameters: list | None = None) -> None:
    connection.cursor().execute(sql, parameters).fetchall()


def _fill(
    connection: Any, name: str, frame: pd.DataFrame, columns: str
) -> None:
    _run(connection, f'CREATE TABLE {name} ({columns})')
    rows = [
        tuple(
            value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
            for value in row
        )
        for row in frame.itertuples(index=False, name=None)
    ]
    marks = '(' + ', '.join(['?'] * len(frame.columns)) + ')'
    _run(
        connection,
        f'INSERT INTO {name} VALUES {", ".join([marks] * len(rows))}',
        [value for row in rows for value in row],
    )


@pytest.fixture
def schemas(url: str) -> Iterator[tuple[Any, list[str]]]:
    r"""A connection, and two fresh ``memory`` schemas dropped afterwards."""
    connection = backend.connect(url, user='sdm_test')
    names = [f'sdm_{uuid.uuid4().hex[:8]}' for _ in range(2)]
    for name in names:
        _run(connection, f'CREATE SCHEMA memory.{name}')
    try:
        yield connection, names
    finally:
        for name in names:
            _run(connection, f'DROP SCHEMA memory.{name} CASCADE')
        connection.close()


def _orders_columns(time_type: str = 'TIMESTAMP(6)') -> str:
    return f'order_id BIGINT, user_id BIGINT, amount DOUBLE, ts {time_type}'


def _payload(schema: Schema) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], random_seed=7
        )
    (payload,) = request._payloads
    return json.loads(json.dumps(payload, default=str))


def test_source_recognises_a_trino_connection(
    schemas: tuple[Any, list[str]],
) -> None:
    connection, _ = schemas
    source = Source(connection, catalog='memory', schema='default')

    assert isinstance(source, TrinoSource)
    assert source.backend == 'trino'
    assert source.namespace == {'catalog': 'memory', 'schema': 'default'}


def test_fully_qualified_names_span_catalogs(
    schemas: tuple[Any, list[str]],
) -> None:
    connection, (first, second) = schemas
    _fill(
        connection,
        f'memory.{first}.users',
        USERS,
        'user_id BIGINT, age BIGINT',
    )
    _fill(connection, f'memory.{second}.orders', ORDERS, _orders_columns())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(
            TrinoSource(connection, catalog='memory', schema=first),
            {'users': 'users', 'orders': f'memory.{second}.orders'},
            primary_keys=KEYS,
            time_columns={'orders': 'ts'},
            relationships=LINKS,
        )

    assert schema['orders'].source_name == f'memory.{second}.orders'
    assert _payload(schema)['predict']['instance_table']['rows']


def test_time_columns_with_and_without_zone_sample_alike(
    schemas: tuple[Any, list[str]],
) -> None:
    connection, names = schemas
    payloads = []
    for name, time_type in zip(
        names, ('TIMESTAMP(3)', 'TIMESTAMP(3) WITH TIME ZONE'), strict=True
    ):
        _fill(
            connection,
            f'memory.{name}.users',
            USERS,
            'user_id BIGINT, age BIGINT',
        )
        _fill(
            connection,
            f'memory.{name}.orders',
            ORDERS,
            _orders_columns(time_type),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            schema = Schema.from_source(
                TrinoSource(connection, catalog='memory', schema=name),
                {'users': 'users', 'orders': 'orders'},
                primary_keys=KEYS,
                time_columns={'orders': 'ts'},
                relationships=LINKS,
            )
        payloads.append(_payload(schema))

    def rows(payload: dict[str, Any], part: str) -> list[Any]:
        table = payload[part]['related_tables']['orders']
        at = {name: i for i, name in enumerate(table['columns'])}
        return sorted(
            (
                row[at['user_id']],
                pd.Timestamp(row[at['ts']]).tz_localize(None),
                row[at['amount']],
            )
            for row in table['rows']
        )

    without_zone, with_zone = payloads
    assert rows(without_zone, 'context') == rows(with_zone, 'context')
    assert rows(without_zone, 'predict') == rows(with_zone, 'predict')
    labels = [
        [label for *_, label in payload['context']['instance_table']['rows']]
        for payload in payloads
    ]
    assert labels[0] == labels[1]


def _refuses(url: str, user: str, sql: str) -> bool:
    connection = backend.connect(url, user=user)
    try:
        connection.cursor().execute(sql).fetchall()
    except Exception as error:
        if 'Access Denied' in str(error):
            return True
        raise
    finally:
        connection.close()
    return False


@pytest.mark.parametrize(
    'user, refusal',
    [
        # A table the user may not read is left out of information_schema,
        # so it reads as missing, never as empty.
        ('nobody', 'does not exist in the Trino catalog, or cannot be read'),
        # A catalog the user may not reach refuses the metadata read itself.
        ('outsider', 'Access Denied'),
    ],
)
def test_access_denied_is_not_an_empty_sample(
    url: str, schemas: tuple[Any, list[str]], user: str, refusal: str
) -> None:
    r"""Needs the server started with ``scripts/trino_test_server.py
    --access-control packages/structured-data-models-client/tests/
    trino_access_rules.json``; skipped against one that grants everything.
    """
    connection, (name, _) = schemas
    _fill(
        connection, f'memory.{name}.users', USERS, 'user_id BIGINT, age BIGINT'
    )
    _fill(connection, f'memory.{name}.orders', ORDERS, _orders_columns())
    if not _refuses(url, user, f'SELECT 1 FROM memory.{name}.orders'):
        pytest.skip('the test server grants every user every table')

    refused = backend.connect(url, user=user)
    try:
        with pytest.raises(Exception, match=refusal):
            Schema.from_source(
                TrinoSource(refused, catalog='memory', schema=name),
                {'users': 'users', 'orders': 'orders'},
                primary_keys=KEYS,
                time_columns={'orders': 'ts'},
                relationships=LINKS,
                check_keys=False,
            )
    finally:
        refused.close()


def _relational_weights() -> bool:
    r"""Whether Kumo-Relational's weights are already on this machine.

    Never downloads them: the suite stays off the network, and this test runs
    only where they were fetched before.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            'nvidia/Kumo-Relational', revision='v1.0.1', local_files_only=True
        )
    except Exception:
        return False
    return True


def test_from_source_predicts_end_to_end(
    schemas: tuple[Any, list[str]],
) -> None:
    if not _relational_weights():
        pytest.skip("Kumo-Relational's weights are not cached locally")
    from sdm_client import KumoRelational, SDMClient

    connection, (name, _) = schemas
    _fill(
        connection, f'memory.{name}.users', USERS, 'user_id BIGINT, age BIGINT'
    )
    _fill(connection, f'memory.{name}.orders', ORDERS, _orders_columns())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(
            TrinoSource(connection, catalog='memory', schema=name),
            {'users': 'users', 'orders': 'orders'},
            primary_keys=KEYS,
            time_columns={'orders': 'ts'},
            relationships=LINKS,
            check_keys=False,
        )
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2, 3], model=KumoRelational()
        )
    with SDMClient.in_process() as client:
        frame = client.predict(request, model=KumoRelational()).to_pandas()

    assert frame['id'].tolist() == [1, 2, 3]
    assert frame['prediction'].notna().all()
