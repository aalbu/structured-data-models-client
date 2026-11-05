# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Tables for the Trino backend's tests, in Trino's ``memory`` catalog.

Each test gets a schema of its own, dropped afterwards, so tests share a
server and nothing else.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator, Mapping
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest

try:
    import trino
except ImportError:
    if os.getenv('SDM_REQUIRE_TRINO'):
        raise
    pytest.skip("'trino' extension not installed", allow_module_level=True)

CATALOG = 'memory'
TEST_USER = 'sdm_test'


def trino_connection(
    url: str, user: str = TEST_USER, source: str = 'sdm-tests'
) -> Any:
    r"""A connection to the test server, as ``user``, in UTC.

    ``source`` tags every statement, so a test can find its own in
    ``system.runtime.queries``.
    """
    parts = urlsplit(url)
    return trino.dbapi.connect(
        host=parts.hostname,
        port=parts.port,
        http_scheme=parts.scheme or 'http',
        user=user,
        catalog=CATALOG,
        timezone='UTC',
        source=source,
    )


def _trino_type(series: pd.Series) -> str:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'
    if pd.api.types.is_integer_dtype(dtype):
        return 'bigint'
    if pd.api.types.is_float_dtype(dtype):
        return 'double'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp(6)'
    return 'varchar'


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_trino(
    url: str,
    schema: str,
    frames: Mapping[str, pd.DataFrame],
    types: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    r"""Create each frame as a table in ``memory.<schema>``, and fill it.

    Column types follow the frame's dtypes unless ``types`` names them, so a
    test can ask for a type pandas has no spelling of, such as
    ``timestamp(3) with time zone``.
    """
    connection = trino_connection(url)
    try:
        cursor = connection.cursor()
        for table, frame in frames.items():
            given = (types or {}).get(table, {})
            columns = ', '.join(
                f'{_quote(column)} {given.get(column, _trino_type(frame[column]))}'
                for column in frame.columns
            )
            name = f'{_quote(schema)}.{_quote(table)}'
            cursor.execute(f'CREATE TABLE {name} ({columns})')
            cursor.fetchall()
            if frame.empty:
                continue
            casts = ', '.join(
                f'CAST(? AS {given.get(column, _trino_type(frame[column]))})'
                for column in frame.columns
            )
            rows = [
                [_value(value) for value in row]
                for row in frame.itertuples(index=False, name=None)
            ]
            # One statement per batch: the client inlines parameters, so a
            # single statement for a large frame outgrows the query length.
            for start in range(0, len(rows), 500):
                batch = rows[start : start + 500]
                values = ', '.join(f'({casts})' for _ in batch)
                cursor.execute(
                    f'INSERT INTO {name} VALUES {values}',
                    [value for row in batch for value in row],
                )
                cursor.fetchall()
    finally:
        connection.close()


@pytest.fixture()
def trino_schema(trino_url: str) -> Generator[str, None, None]:
    r"""A fresh, empty ``memory`` schema, dropped with its tables after."""
    schema = f'sdm_{uuid.uuid4().hex}'
    connection = trino_connection(trino_url)
    try:
        cursor = connection.cursor()
        cursor.execute(f'CREATE SCHEMA {CATALOG}.{schema}')
        cursor.fetchall()
        yield schema
    finally:
        cursor = connection.cursor()
        cursor.execute(f'DROP SCHEMA IF EXISTS {CATALOG}.{schema} CASCADE')
        cursor.fetchall()
        connection.close()
