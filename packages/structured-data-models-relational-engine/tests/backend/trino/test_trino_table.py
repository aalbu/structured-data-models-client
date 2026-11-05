# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import uuid

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.trino import TrinoTable

from .conftest import CATALOG, trino_connection, write_trino


@pytest.mark.parametrize(
    'trino_type, expected',
    [
        ('tinyint', Dtype.int),
        ('smallint', Dtype.int),
        ('integer', Dtype.int),
        ('bigint', Dtype.int),
        ('real', Dtype.float),
        ('double', Dtype.float),
        ('decimal(10,0)', Dtype.int),
        ('decimal(10,2)', Dtype.float),
        ('varchar', Dtype.string),
        ('varchar(20)', Dtype.string),
        ('char(3)', Dtype.string),
        ('uuid', Dtype.string),
        ('json', Dtype.string),
        ('varbinary', Dtype.binary),
        ('boolean', Dtype.bool),
        ('date', Dtype.date),
        ('timestamp(3)', Dtype.date),
        ('timestamp(6) with time zone', Dtype.date),
        ('time(3)', Dtype.time),
        ('array(bigint)', Dtype.intlist),
        ('array(double)', Dtype.floatlist),
        ('array(varchar)', Dtype.stringlist),
        ('map(varchar, bigint)', None),
        ('row(a bigint)', None),
    ],
)
def test_trino_type_mapping(trino_type: str, expected: Dtype | None) -> None:
    assert TrinoTable._to_dtype(trino_type) == expected


def test_columns_and_primary_key_from_arguments(
    trino_url: str, trino_schema: str
) -> None:
    users = pd.DataFrame({'user_id': [1, 2, 3], 'age': [20, 30, 40]})
    write_trino(
        trino_url,
        trino_schema,
        {'users': users},
        types={'users': {'age': 'integer'}},
    )
    connection = trino_connection(trino_url)
    try:
        table = TrinoTable(
            connection, 'users', schema=trino_schema, primary_key='user_id'
        )
        columns = table._source_column_dict
        assert list(columns) == ['user_id', 'age']
        assert columns['user_id'].dtype == Dtype.int
        assert columns['age'].dtype == Dtype.int
        assert table.primary_key is not None
        assert table.primary_key.name == 'user_id'
        assert table._get_source_foreign_keys() == []
        assert table.source_name == f'{CATALOG}.{trino_schema}.users'
    finally:
        connection.close()


def test_names_that_need_quoting_are_read(
    trino_url: str, trino_schema: str
) -> None:
    frame = pd.DataFrame({'select': [1, 2], 'Mixed Case': ['a', 'b']})
    write_trino(trino_url, trino_schema, {'Order': frame})
    connection = trino_connection(trino_url)
    try:
        # Trino folds unquoted names to lower case, and so stores this one.
        table = TrinoTable(
            connection, 'order', schema=trino_schema, primary_key='select'
        )
        preview = table._get_source_sample_df()
        assert sorted(preview.columns) == ['mixed case', 'select']
        assert sorted(preview['select'].tolist()) == [1, 2]
    finally:
        connection.close()


def test_missing_table_raises_table_not_found(
    trino_url: str, trino_schema: str
) -> None:
    connection = trino_connection(trino_url)
    try:
        with pytest.raises(TableNotFoundError, match=f'{trino_schema}.nope'):
            TrinoTable(connection, 'nope', schema=trino_schema)
    finally:
        connection.close()


def test_row_count_and_time_range_come_from_stats(
    trino_url: str, trino_schema: str
) -> None:
    frame = pd.DataFrame({'user_id': range(25)})
    write_trino(trino_url, trino_schema, {'users': frame})
    source = f'sdm-stats-{trino_schema}'
    connection = trino_connection(trino_url, source=source)
    try:
        table = TrinoTable(
            connection, 'users', schema=trino_schema, primary_key='user_id'
        )
        assert table._get_num_rows() == 25
    finally:
        connection.close()

    audit = trino_connection(trino_url).cursor()
    audit.execute(
        'SELECT query FROM system.runtime.queries WHERE source = ?', [source]
    )
    statements = [' '.join(row[0].split()).upper() for row in audit.fetchall()]
    assert statements
    assert not any('SELECT COUNT(*)' in sql for sql in statements)


def test_row_count_is_unknown_without_stats(
    trino_url: str, trino_schema: str
) -> None:
    connection = trino_connection(trino_url)
    try:
        cursor = connection.cursor()
        cursor.execute(
            f'CREATE VIEW {CATALOG}.{trino_schema}.users AS '
            'SELECT * FROM UNNEST(sequence(1, 25)) AS t(user_id)'
        )
        cursor.fetchall()
        table = TrinoTable(
            connection, 'users', schema=trino_schema, primary_key='user_id'
        )
        assert table._get_num_rows() is None
    finally:
        connection.close()


def test_a_truncated_preview_confirms_the_time_column_from_stats(
    trino_url: str,
) -> None:
    r"""A preview capped at its row limit asks for the time column's range
    over the whole table, which read as one COUNT, MIN and MAX across a
    production table; the connector's statistics hold all of it.
    """
    source = f'sdm-range-{uuid.uuid4().hex}'
    connection = trino_connection(trino_url, source=source)
    try:
        table = TrinoTable(
            connection,
            'orders',
            catalog='tpch',
            schema='tiny',
            primary_key='orderkey',
        )
        assert table._infer_time_column_among(['orderdate']) == 'orderdate'
        whole = table._full_range(['orderdate'])
    finally:
        connection.close()

    assert whole is not None
    assert whole.rows == 15_000
    assert whole.columns['orderdate'].held == 15_000
    # The values the query would read, of the column's own type.
    assert whole.columns['orderdate'].low == datetime.date(1992, 1, 1)
    assert whole.columns['orderdate'].high == datetime.date(1998, 8, 2)
    audit = trino_connection(trino_url).cursor()
    audit.execute(
        'SELECT query FROM system.runtime.queries WHERE source = ?', [source]
    )
    statements = [' '.join(row[0].split()).upper() for row in audit.fetchall()]
    assert not any('MIN(' in sql or 'COUNT(*)' in sql for sql in statements)


def test_the_range_falls_back_to_a_query_without_stats(
    trino_url: str, trino_schema: str
) -> None:
    connection = trino_connection(trino_url)
    try:
        cursor = connection.cursor()
        cursor.execute(
            f'CREATE VIEW {CATALOG}.{trino_schema}.events AS '
            "SELECT x AS event_id, DATE '2024-01-01' + x * INTERVAL '1' DAY "
            'AS at FROM UNNEST(sequence(0, 9)) AS t(x)'
        )
        cursor.fetchall()
        table = TrinoTable(
            connection, 'events', schema=trino_schema, primary_key='event_id'
        )
        whole = table._full_range(['at'])
    finally:
        connection.close()
    assert whole is not None
    assert whole.rows == 10
    assert whole.columns['at'].held == 10
