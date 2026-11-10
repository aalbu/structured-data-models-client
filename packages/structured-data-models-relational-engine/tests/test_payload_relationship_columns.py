# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How the request builder rebuilds foreign key columns from sampled links.

Both directions of a relationship rebuild the same column, and only the first
one met is used. The SQL samplers insert the reverse edge of the entity table's
own foreign keys first, since the entity is visited on hop 0, and that edge
holds a pair only for the rows reached outward from the parent. Built from it,
nearly every row of the entity table lost its foreign key.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from predictive_query.schema.typing import Stype
from sdm_relational_engine.rfm.inference.context import (
    REVERSE_RELATION_PREFIX,
    SampledEdgeLayout,
    SampledLink,
)
from sdm_relational_engine.rfm.inference.payload import (
    _populate_relationship_columns,
)

_FORWARD = ('orders', 'customer_id', 'customers')
_REVERSE = ('customers', f'{REVERSE_RELATION_PREFIX}customer_id', 'orders')


def _coo(pairs: list[tuple[int, int]]) -> SampledLink:
    row, col = zip(*pairs) if pairs else ((), ())
    return SampledLink(
        layout=SampledEdgeLayout.COO,
        row=np.asarray(row, dtype=np.int64),
        col=np.asarray(col, dtype=np.int64),
        num_sampled_edges=[len(pairs)],
    )


def _build(links: dict) -> tuple[pd.DataFrame, list[dict]]:
    r"""Rebuild ``orders.customer_id`` for six orders of three customers."""
    orders = pd.DataFrame({'order_id': [f'o{i}' for i in range(6)]})
    customers = pd.DataFrame({'customer_id': ['c0', 'c1', 'c2']})
    context = SimpleNamespace(
        entity_table_names=(),
        sampled_subgraph=SimpleNamespace(links=links),
    )
    frames = {'orders': orders, 'customers': customers}
    relationships = _populate_relationship_columns(
        context,
        frames,
        {'orders': 'order_id', 'customers': 'customer_id'},
        {'orders': {}, 'customers': {}},
        {},
    )
    return frames['orders'], relationships


# Every order and its customer, as the forward edge holds them.
_EVERY_ORDER = [(i, i % 3) for i in range(6)]


def test_forward_edge_is_used_when_the_reverse_edge_comes_first() -> None:
    # The reverse edge reaches two orders from their customers; the forward
    # edge holds all six. The SQL samplers insert the reverse edge first.
    reverse = _coo([(0, 0), (1, 1)])  # (customer, order) pairs
    orders, relationships = _build(
        {_REVERSE: reverse, _FORWARD: _coo(_EVERY_ORDER)}
    )
    assert orders['customer_id'].tolist() == [
        'c0',
        'c1',
        'c2',
        'c0',
        'c1',
        'c2',
    ]
    assert len(relationships) == 1


def test_rows_only_the_reverse_edge_reaches_are_left_without_a_key() -> None:
    # Measured: filling these from the reverse edge cost as much accuracy as
    # the reordering gained, so they are deliberately left empty.
    forward = _coo([(i, i % 3) for i in range(4)])
    reverse = _coo([(1, 4), (2, 5)])
    orders, relationships = _build({_FORWARD: forward, _REVERSE: reverse})
    assert orders['customer_id'].tolist()[:4] == ['c0', 'c1', 'c2', 'c0']
    assert orders['customer_id'].iloc[4:].isna().all()
    assert len(relationships) == 1


def test_the_forward_edge_wins_where_both_directions_name_a_row() -> None:
    reverse = _coo([(2, 0)])  # claims order 0 belongs to c2
    orders, _ = _build({_REVERSE: reverse, _FORWARD: _coo(_EVERY_ORDER)})
    assert orders['customer_id'].iloc[0] == 'c0'


def test_the_column_is_declared_an_id_once() -> None:
    overrides: dict[str, dict] = {'orders': {}, 'customers': {}}
    context = SimpleNamespace(
        entity_table_names=(),
        sampled_subgraph=SimpleNamespace(
            links={_REVERSE: _coo([(0, 0)]), _FORWARD: _coo(_EVERY_ORDER)}
        ),
    )
    relationships = _populate_relationship_columns(
        context,
        {
            'orders': pd.DataFrame({'order_id': [f'o{i}' for i in range(6)]}),
            'customers': pd.DataFrame({'customer_id': ['c0', 'c1', 'c2']}),
        },
        {'orders': 'order_id', 'customers': 'customer_id'},
        overrides,
        {},
    )
    assert overrides['orders'] == {'customer_id': Stype.ID}
    assert [r['source_columns'] for r in relationships] == [['customer_id']]
