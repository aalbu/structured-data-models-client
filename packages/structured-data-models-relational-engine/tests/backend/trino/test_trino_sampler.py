# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The Trino sampler reaches the rows the in-memory sampler reaches.

Ids travel to Trino as one JSON parameter, which the client inlines into the
statement's text, so the tests here also hold every statement under Trino's
query length.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest
import sdm_relational_engine.rfm as rfm
from sdm_relational_engine.rfm.backend.trino import TrinoSampler, TrinoTable

from ..test_sql_sampler_parity import (
    _KEYS,
    _SHAPES,
    _frames,
    _sampled,
    _shape_frames,
    _shape_graph,
)
from .conftest import trino_connection, write_trino

pytest.importorskip('sdm_relational_engine.relationallib')

_MAX_QUERY_LENGTH = 1_000_000


def _graph(
    url: str,
    schema: str,
    users: pd.DataFrame,
    orders: pd.DataFrame,
    types: Mapping[str, Mapping[str, str]] | None = None,
    source: str = 'sdm-tests',
) -> rfm.Graph:
    write_trino(url, schema, {'users': users, 'orders': orders}, types=types)
    connection = trino_connection(url, source=source)
    graph = rfm.Graph(
        tables=[
            TrinoTable(
                connection, 'users', schema=schema, primary_key='user_id'
            ),
            TrinoTable(
                connection,
                'orders',
                schema=schema,
                primary_key='order_id',
                time_column='date',
            ),
        ],
        edges=[],
    )
    graph.link('orders', 'user_id', 'users')
    graph._connection = connection
    return graph


def _close(graph: rfm.Graph) -> None:
    if graph._connection is not None:
        graph._connection.close()
        graph._connection = None


def _statements(url: str, source: str) -> list[str]:
    cursor = trino_connection(url).cursor()
    cursor.execute(
        'SELECT query FROM system.runtime.queries WHERE source = ?', [source]
    )
    return [row[0] for row in cursor.fetchall()]


def _trino_shape_graph(url: str, schema: str, edges: list) -> rfm.Graph:
    frames = _shape_frames()
    names = sorted({table for edge in edges for table in (edge[0], edge[2])})
    write_trino(url, schema, {name: frames[name] for name in names})
    connection = trino_connection(url)
    graph = rfm.Graph(
        tables=[
            TrinoTable(
                connection,
                name,
                schema=schema,
                primary_key=_KEYS[name],
                **({'time_column': 'ts'} if 'ts' in frames[name] else {}),
            )
            for name in names
        ],
        edges=edges,
    )
    graph._connection = connection
    return graph


@pytest.mark.parametrize(
    'shape, entity',
    [
        (shape, entity)
        for shape, edges in _SHAPES.items()
        for entity in sorted({e[0] for e in edges} | {e[2] for e in edges})[:3]
    ],
)
def test_every_shape_samples_as_in_memory(
    trino_url: str, trino_schema: str, shape: str, entity: str
) -> None:
    edges = _SHAPES[shape]
    graph = _trino_shape_graph(trino_url, trino_schema, edges)
    try:
        sampled = _sampled(graph, entity)
    finally:
        _close(graph)
    assert sampled == _sampled(_shape_graph('local', '', edges), entity)


def test_big_and_quoted_ids_round_trip(
    trino_url: str, trino_schema: str
) -> None:
    connection = trino_connection(trino_url)
    try:
        for name, ids, trino_type in (
            ('words', ["o'brien", 'a"b', 'plain'], 'varchar'),
            ('numbers', [9007199254740993, 1, 2], 'bigint'),
        ):
            write_trino(
                trino_url,
                trino_schema,
                {name: pd.DataFrame({'id': ids})},
                types={name: {'id': trino_type}},
            )
            graph = rfm.Graph(
                tables=[
                    TrinoTable(
                        connection, name, schema=trino_schema, primary_key='id'
                    )
                ],
                edges=[],
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                sampler = TrinoSampler(graph, verbose=False)
            frame = sampler._sample_entity_table(
                name, {'id'}, len(ids), entity_ids=list(ids)
            )
            assert sorted(frame['id'].tolist()) == sorted(ids)
    finally:
        connection.close()


def test_ten_thousand_ids_are_all_sampled(
    trino_url: str, trino_schema: str
) -> None:
    ids = [str(uuid.UUID(int=index)) for index in range(10_000)]
    users = pd.DataFrame({'user_id': ids})
    orders = pd.DataFrame(
        {
            'order_id': range(10_000),
            'user_id': ids,
            'date': pd.Timestamp('2025-01-01'),
            'amount': 1.0,
        }
    )
    source = f'sdm-many-{trino_schema}'
    graph = _graph(trino_url, trino_schema, users, orders, source=source)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sampler = TrinoSampler(graph, verbose=False)
        anchor = pd.Series(pd.to_datetime(['2025-02-01'] * len(ids)))
        frame, batch = sampler._by_fkey(
            'orders', 'user_id', pd.Series(ids), 1, anchor, {'order_id'}
        )
        rows = [
            {'id': value, 'e': '2025-02-01T00:00:00+00:00'} for value in ids
        ]
        chunks = sampler._chunk_rows(rows)
    finally:
        _close(graph)

    assert len(frame) == len(ids)
    assert len(np.unique(batch)) == len(ids)
    assert len(chunks) > 1
    assert all(len(sampler._dumps(chunk)) < 400_000 for chunk in chunks)
    statements = _statements(trino_url, source)
    assert statements
    assert max(len(sql) for sql in statements) < _MAX_QUERY_LENGTH


def test_seeded_sampling_repeats_across_connections(
    trino_url: str, trino_schema: str
) -> None:
    users, orders = _frames()
    write_trino(trino_url, trino_schema, {'users': users})

    def sample(seed: int) -> list[int]:
        connection = trino_connection(trino_url)
        try:
            graph = rfm.Graph(
                tables=[
                    TrinoTable(
                        connection,
                        'users',
                        schema=trino_schema,
                        primary_key='user_id',
                    )
                ],
                edges=[],
            )
            sampler = TrinoSampler(graph, verbose=False)
            frame = sampler._sample_entity_table(
                'users', {'user_id'}, 10, random_seed=seed
            )
            return frame['user_id'].tolist()
        finally:
            connection.close()

    assert sample(42) == sample(42)
    assert sample(42) != sample(2_000_043)


def test_time_range_comes_from_stats(trino_url: str) -> None:
    source = f'sdm-range-{uuid.uuid4().hex}'
    connection = trino_connection(trino_url, source=source)
    try:
        graph = rfm.Graph(
            tables=[
                TrinoTable(
                    connection,
                    'orders',
                    catalog='tpch',
                    schema='tiny',
                    primary_key='orderkey',
                    time_column='orderdate',
                )
            ],
            edges=[],
        )
        sampler = TrinoSampler(graph, verbose=False)
        low, high = sampler._get_min_max_time_dict(['orders'])['orders']
    finally:
        connection.close()
    assert (low, high) == (
        pd.Timestamp('1992-01-01'),
        pd.Timestamp('1998-08-02'),
    )
    statements = [
        ' '.join(sql.split()).upper() for sql in _statements(trino_url, source)
    ]
    assert not any('MIN(' in sql for sql in statements)


def test_rows_sharing_a_timestamp_come_back_in_one_order(
    trino_url: str, trino_schema: str
) -> None:
    r"""A target sums the rows ``_by_time`` returns, and a float sum depends
    on their order. Trino returns rows that tie on the time column in
    whatever order its splits finish, so the same seed gave a label that
    differed in its last bits between two runs. The primary key breaks the
    tie.
    """
    rng = np.random.default_rng(0)
    users = pd.DataFrame({'user_id': [1]})
    orders = pd.DataFrame(
        {
            'order_id': rng.permutation(400),
            'user_id': 1,
            'date': pd.Timestamp('2025-01-01'),
            'amount': rng.uniform(0, 1e6, 400),
        }
    )
    graph = _graph(trino_url, trino_schema, users, orders)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            sampler = TrinoSampler(graph, verbose=False)
        anchor = pd.Series(pd.to_datetime(['2024-12-31']))
        orders_seen = [
            sampler._by_time(
                'orders',
                'user_id',
                pd.Series([1]),
                anchor,
                None,
                pd.DateOffset(days=30),
                {'order_id', 'amount', 'date'},
            )[0]['order_id'].tolist()
            for _ in range(5)
        ]
    finally:
        _close(graph)
    assert orders_seen[0] == sorted(orders_seen[0])
    assert all(seen == orders_seen[0] for seen in orders_seen)
