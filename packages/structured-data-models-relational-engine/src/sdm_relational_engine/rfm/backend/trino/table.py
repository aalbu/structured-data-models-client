# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from predictive_query.schema.typing import Dtype, Stype
from predictive_query.time_columns import ColumnRange, TableRange
from sdm_connectors import quote_ident
from sdm_connectors.backends.trino import reading

from sdm_relational_engine.exceptions import TableNotFoundError
from sdm_relational_engine.rfm.backend.trino import Connection
from sdm_relational_engine.rfm.base import (
    Column,
    ColumnSpec,
    ColumnSpecType,
    DataBackend,
    SourceColumn,
    SourceForeignKey,
    Table,
)
from sdm_relational_engine.rfm.base.table import MissingType

_INTEGERS = frozenset({'tinyint', 'smallint', 'integer', 'bigint'})
_FLOATS = frozenset({'real', 'double'})
_STRINGS = ('varchar', 'char', 'uuid', 'json', 'ipaddress')
_DECIMAL = re.compile(r'decimal\(\s*\d+\s*,\s*(\d+)\s*\)')
_ARRAY = re.compile(r'array\((.*)\)')


def _stat_value(text: str, trino_type: str) -> Any:
    r"""A ``SHOW STATS`` bound, which is text, as the value a query reads.

    ``MIN`` and ``MAX`` come back typed, as a ``date``, a ``datetime`` or a
    number, and what compares the two must not see a difference between the
    statistics and the query.
    """
    trino_type = trino_type.strip().lower()
    if trino_type == 'date':
        return datetime.date.fromisoformat(text)
    if trino_type.startswith('timestamp'):
        stamp = pd.Timestamp(text)
        return stamp.to_pydatetime()
    if trino_type in _INTEGERS:
        return int(text)
    if trino_type in _FLOATS or trino_type.startswith('decimal'):
        return float(text)
    return text


def _frame(cursor: Any) -> pd.DataFrame:
    rows = cursor.fetchall()
    names = [column[0] for column in cursor.description]
    helper_names = {
        '__kumo_batch__': '__KUMO_BATCH__',
        '__kumo_row__': '__KUMO_ROW__',
        '__kumo_matches__': '__KUMO_MATCHES__',
    }
    names = [helper_names.get(name, name) for name in names]
    # Kept as Python scalars until `_sanitize_trino_frame` applies the
    # declared types: pandas inference would widen an integer column holding
    # NULL to float64 and round values above 2**53.
    return pd.DataFrame(rows, columns=names, dtype=object)


def _sanitize_trino_frame(
    df: pd.DataFrame,
    dtype_dict: Mapping[str, Dtype | None],
    stype_dict: Mapping[str, Stype | None] | None,
) -> pd.DataFrame:
    r"""Hold nullable Trino scalars in pandas' nullable dtypes."""

    def stringify(value: Any) -> str | None:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, separators=(',', ':'))
        return str(value)

    for name, dtype in dtype_dict.items():
        if name not in df:
            continue
        if dtype is not None and dtype.is_int():
            df[name] = pd.array(df[name], dtype='Int64')
        elif dtype is not None and dtype.is_float():
            df[name] = pd.array(
                [None if value is None else float(value) for value in df[name]],
                dtype='Float64',
            )
        elif dtype == Dtype.bool:
            df[name] = pd.array(df[name], dtype='boolean')
        elif dtype == Dtype.string:
            df[name] = pd.array(df[name].map(stringify), dtype='string')
    return Table._sanitize(df=df, dtype_dict=dtype_dict, stype_dict=stype_dict)


class TrinoTable(Table):
    r"""A table read through a Trino server.

    Named by catalog, schema and table, since one connection reaches every
    catalog the server holds. Trino declares no keys, so the primary key comes
    from the caller or from inference.
    """

    _SQL_TEXT_TYPE = 'VARCHAR'
    _SQL_CHR_FUNCTION = 'CHR'

    def __init__(
        self,
        connection: Connection,
        name: str,
        source_name: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        columns: Sequence[ColumnSpecType] | None = None,
        primary_key: MissingType | str | None = MissingType.VALUE,
        time_column: str | None = None,
        end_time_column: str | None = None,
    ) -> None:
        catalog = catalog or connection.catalog
        schema = schema or connection.schema
        if not catalog or not schema:
            raise ValueError(
                f"Unspecified 'catalog' or 'schema' for table "
                f"'{source_name or name}', and the connection names none"
            )
        self._connection = connection
        self._catalog = catalog
        self._schema = schema
        self._source_trino_type_dict: dict[str, str] = {}
        super().__init__(
            name=name,
            source_name=source_name,
            columns=columns,
            primary_key=primary_key,
            time_column=time_column,
            end_time_column=end_time_column,
        )

    @property
    def source_name(self) -> str:
        return f'{self._catalog}.{self._schema}.{self._source_name}'

    @property
    def _quoted_source_name(self) -> str:
        return '.'.join(
            quote_ident(name)
            for name in (self._catalog, self._schema, self._source_name)
        )

    @property
    def backend(self) -> DataBackend:
        return DataBackend.TRINO

    @staticmethod
    def _to_dtype(trino_type: str | None) -> Dtype | None:
        if trino_type is None:
            return None
        trino_type = trino_type.strip().lower()
        if trino_type in _INTEGERS:
            return Dtype.int
        if trino_type in _FLOATS:
            return Dtype.float
        decimal = _DECIMAL.fullmatch(trino_type)
        if decimal is not None:
            return Dtype.int if decimal.group(1) == '0' else Dtype.float
        if trino_type == 'decimal':
            return Dtype.float
        if trino_type.startswith(_STRINGS):
            return Dtype.string
        if trino_type == 'varbinary':
            return Dtype.binary
        if trino_type == 'boolean':
            return Dtype.bool
        if trino_type == 'date' or trino_type.startswith('timestamp'):
            return Dtype.date
        if trino_type.startswith('time'):
            return Dtype.time
        array = _ARRAY.fullmatch(trino_type)
        if array is not None:
            element = TrinoTable._to_dtype(array.group(1))
            if element is not None and element.is_int():
                return Dtype.intlist
            if element is not None and element.is_float():
                return Dtype.floatlist
            if element == Dtype.string:
                return Dtype.stringlist
        return None

    def _get_source_columns(self) -> list[SourceColumn]:
        with reading(self._connection) as cursor:
            cursor.execute(
                'SELECT column_name, data_type, is_nullable\n'
                f'FROM {quote_ident(self._catalog)}.information_schema.columns\n'
                'WHERE table_schema = ? AND table_name = ?\n'
                'ORDER BY ordinal_position',
                [self._schema, self._source_name],
            )
            rows = cursor.fetchall()
        if not rows:
            raise TableNotFoundError(
                f"Table '{self.source_name}' does not exist in the Trino "
                f'catalog, or cannot be read'
            )
        self._source_trino_type_dict = {row[0]: row[1] for row in rows}
        return [
            SourceColumn(
                name=column,
                dtype=self._to_dtype(trino_type),
                is_primary_key=False,
                is_unique_key=False,
                is_nullable=is_nullable != 'NO',
            )
            for column, trino_type, is_nullable in rows
        ]

    def _get_source_foreign_keys(self) -> list[SourceForeignKey]:
        return []

    def _trino_type(self, column: str) -> str | None:
        r"""Trino's own spelling of a source column's type, for casts."""
        return self._source_trino_type_dict.get(column)

    def _get_source_sample_df(self) -> pd.DataFrame:
        columns = [quote_ident(column) for column in self._source_column_dict]
        with reading(self._connection) as cursor:
            cursor.execute(
                f'SELECT {", ".join(columns)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            df = _frame(cursor)
        if df.empty:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return _sanitize_trino_frame(
            df=df,
            dtype_dict={
                column.name: column.dtype
                for column in self._source_column_dict.values()
            },
            stype_dict=None,
        )

    def _fetch_rows(self, sql: str) -> list[tuple]:
        with reading(self._connection) as cursor:
            cursor.execute(sql)
            return [tuple(row) for row in cursor.fetchall()]

    def _count_distinct(self, quoted: Sequence[str]) -> str:
        if len(quoted) == 1:
            return super()._count_distinct(quoted)
        # Trino counts distinct values of one argument only; a row of several
        # is not null when one of them is, so a row missing part of the key
        # is left out by hand.
        present = ' AND '.join(f'{column} IS NOT NULL' for column in quoted)
        return (
            f'COUNT(DISTINCT ROW({", ".join(quoted)})) FILTER (WHERE {present})'
        )

    def _table_stats(self) -> list[tuple]:
        r"""``SHOW STATS`` for this table: one row per column, then one with
        a null column name holding the row count.
        """
        return self._fetch_rows(f'SHOW STATS FOR {self._quoted_source_name}')

    def _get_num_rows(self) -> int | None:
        # A full COUNT(*) reads a production table end to end. The connector's
        # statistics answer for free when it keeps them, and without them the
        # row count is simply not known.
        try:
            stats = self._table_stats()
        except Exception:
            return None
        for row in stats:
            if row[0] is None and row[4] is not None:
                num_rows = int(row[4])
                if num_rows == 0:
                    raise RuntimeError(f"Table '{self.source_name}' is empty")
                return num_rows
        return None

    def _full_range(self, names: Sequence[str]) -> TableRange | None:
        r"""The row count, and each column's range, from ``SHOW STATS``.

        The base reads them with one ``COUNT``, ``MIN`` and ``MAX`` over
        every row, which reads a production table end to end. The connector's
        statistics hold the same numbers when it keeps them; a column they
        leave out falls back to that query.
        """
        try:
            stats = {row[0]: row for row in self._table_stats()}
        except Exception:
            return super()._full_range(names)
        total = stats.get(None)
        if total is None or total[4] is None:
            return super()._full_range(names)
        rows = int(total[4])
        columns: dict[str, ColumnRange] = {}
        for name in names:
            row = stats.get(name)
            held = self[name] if name in self else None
            if (
                row is None
                or held is None
                or not held.is_source
                or row[3] is None
                or row[5] is None
                or row[6] is None
            ):
                return super()._full_range(names)
            trino_type = self._trino_type(name) or ''
            columns[name] = ColumnRange(
                _stat_value(row[5], trino_type),
                _stat_value(row[6], trino_type),
                round(rows * (1.0 - float(row[3]))),
            )
        return TableRange(rows, columns)

    def _get_expr_sample_df(
        self,
        columns: Sequence[ColumnSpec | Column],
    ) -> pd.DataFrame:
        projections = [
            f'{column.expr} AS {quote_ident(column.name)}' for column in columns
        ]
        with reading(self._connection) as cursor:
            cursor.execute(
                f'SELECT {", ".join(projections)} '
                f'FROM {self._quoted_source_name} '
                f'LIMIT {self._NUM_SAMPLE_ROWS}'
            )
            df = _frame(cursor)
        if df.empty:
            raise RuntimeError(f"Table '{self.source_name}' is empty")
        return _sanitize_trino_frame(
            df=df,
            dtype_dict={column.name: column.dtype for column in columns},
            stype_dict=None,
        )
