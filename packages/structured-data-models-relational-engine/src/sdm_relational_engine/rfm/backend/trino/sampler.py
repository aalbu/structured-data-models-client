# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.typing import Dtype
from sdm_connectors.backends.trino import reading

from sdm_relational_engine.rfm.backend.trino.table import (
    TrinoTable,
    _frame,
    _sanitize_trino_frame,
)
from sdm_relational_engine.rfm.base import DataBackend, SQLSampler
from sdm_relational_engine.rfm.base.utils import Timestamp
from sdm_relational_engine.rfm.query.pandas_executor import PQueryPandasExecutor
from sdm_relational_engine.utils.progress_logger import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph

#: The most JSON one statement carries. The client inlines parameters into
#: the statement's text (``EXECUTE IMMEDIATE '...' USING ...``), quoting
#: them on the way, and Trino refuses a statement over ``query.max-length``,
#: 1,000,000 characters by default. This leaves room for the quoting and the
#: query around it.
MAX_PARAMETER_BYTES = 400_000


def _json_default(obj: Any) -> Any:
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


class TrinoSampler(SQLSampler):
    r"""Relational sampler that runs its lookups in Trino.

    Ids reach Trino as one JSON array per statement, unnested with their
    position, so a lookup for many entities is one statement rather than one
    per entity. Filtering, temporal predicates, ordering and neighbour limits
    run in Trino; pandas holds only the rows those statements return.
    """

    def __init__(
        self,
        graph: Graph,
        verbose: bool | ProgressLogger = True,
    ) -> None:
        super().__init__(graph=graph, verbose=verbose)
        tables: list[TrinoTable] = []
        for table in graph.tables.values():
            assert isinstance(table, TrinoTable)
            tables.append(table)
            self._connection = table._connection
        self._tables = {table.name: table for table in tables}
        self._trino_type_dict: dict[str, dict[str, str]] = {
            table.name: {
                column.name: table._trino_type(column.name)
                or self._elem_type(column.dtype)
                for column in table.columns
            }
            for table in tables
        }
        self._num_rows_dict: dict[str, int] = {
            table.name: cast(int, table._num_rows)
            for table in graph.tables.values()
        }

    @property
    def num_rows_dict(self) -> dict[str, int]:
        return self._num_rows_dict

    @property
    def backend(self) -> DataBackend:
        return DataBackend.TRINO

    # Helpers #################################################################

    def _projections(self, table_name: str, columns: Any) -> list[str]:
        proj_dict = self.table_column_proj_dict[table_name]
        wanted = set(columns)
        missing = wanted.difference(proj_dict)
        if missing:
            raise KeyError(
                f'Unknown columns requested for table {table_name!r}: '
                f'{", ".join(sorted(missing))}'
            )
        return [
            proj_dict[column]
            for column in self._ordered_columns(table_name, wanted)
        ]

    @staticmethod
    def _elem_type(dtype: Dtype) -> str:
        if dtype.is_int():
            return 'bigint'
        if dtype.is_float():
            return 'double'
        return 'varchar'

    def _trino_type(self, table_name: str, column: str) -> str:
        return self._trino_type_dict[table_name][column]

    @staticmethod
    def _coerce_id(value: Any, dtype: Dtype) -> Any:
        if hasattr(value, 'item'):
            value = value.item()
        if dtype.is_int():
            return int(value)
        # Decimals and floats travel as text, so no digit is lost to a JSON
        # double on the way.
        return str(value)

    @staticmethod
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(',', ':'), default=_json_default)

    def _chunk_rows(
        self,
        rows: list,
        max_bytes: int = MAX_PARAMETER_BYTES,
    ) -> list[list]:
        chunks: list[list] = []
        current: list = []
        current_bytes = 2
        for row in rows:
            size = len(self._dumps(row)) + 1
            if current and current_bytes + size > max_bytes:
                chunks.append(current)
                current, current_bytes = [], 2
            current.append(row)
            current_bytes += size
        if current:
            chunks.append(current)
        return chunks or [[]]

    def _flatten(self, sql: str, rows: list) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        for chunk in self._chunk_rows(rows):
            with reading(self._connection) as cursor:
                cursor.execute(sql, [self._dumps(chunk)])
                frame = _frame(cursor)
            if '__KUMO_BATCH__' in frame:
                frame['__KUMO_BATCH__'] = (
                    frame['__KUMO_BATCH__'].astype(np.int64) + offset
                )
            frames.append(frame)
            offset += len(chunk)
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    @staticmethod
    def _time_expr(field: str, time_type: str) -> str:
        r"""Read an ISO-8601 time from the JSON row as ``time_type``.

        Times travel with their UTC offset. A column with a zone compares
        against the instant itself; one without compares against the instant's
        UTC wall-clock time, which is what the UTC session gives it on read.
        """
        value = f"from_iso8601_timestamp(json_extract_scalar(t.v, '$.{field}'))"
        if 'with time zone' in time_type.lower():
            return value
        return f"CAST({value} AT TIME ZONE 'UTC' AS {time_type})"

    def _json_cte(
        self,
        trino_type: str,
        fields: tuple[str, ...] = (),
        time_type: str = 'timestamp(6)',
    ) -> str:
        select = ['CAST(t.ord - 1 AS bigint) AS __KUMO_BATCH__']
        if fields:
            select.append(
                f"CAST(json_extract_scalar(t.v, '$.id') AS {trino_type}) "
                'AS __KUMO_ID__'
            )
            if 'e' in fields:
                select.append(
                    f'{self._time_expr("e", time_type)} AS __KUMO_END_TIME__'
                )
            if 's' in fields:
                select.append(
                    f'{self._time_expr("s", time_type)} AS __KUMO_START_TIME__'
                )
        else:
            select.append(
                f"CAST(json_extract_scalar(t.v, '$') AS {trino_type}) "
                'AS __KUMO_ID__'
            )
        return (
            'WITH TMP AS (\n'
            f'  SELECT {", ".join(select)}\n'
            '  FROM UNNEST(CAST(json_parse(?) AS array(json)))\n'
            '       WITH ORDINALITY AS t(v, ord)\n'
            ')\n'
        )

    @staticmethod
    def _serialize_time(value: Any) -> str | None:
        if pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        return timestamp.isoformat()

    def _sanitize_output(
        self, table_name: str, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if frame.empty and '__KUMO_BATCH__' not in frame:
            return self._no_rows(
                table_name, self.table_column_proj_dict[table_name]
            )
        batch = frame.pop('__KUMO_BATCH__').to_numpy(dtype=np.int64)
        if '__KUMO_ROW__' in frame:
            frame.pop('__KUMO_ROW__')
        return _sanitize_trino_frame(
            df=frame,
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        ), batch

    # Abstract Methods ########################################################

    def _get_min_max_time_dict(
        self,
        table_names: list[str],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        unknown: list[str] = []
        for table_name in table_names:
            # A whole-table MIN and MAX reads a production table end to end;
            # the connector's statistics hold both when it keeps them.
            time_range = self._stats_time_range(table_name)
            if time_range is None:
                unknown.append(table_name)
            else:
                out[table_name] = time_range
        if not unknown:
            return out

        selects: list[str] = []
        for index, table_name in enumerate(unknown):
            column = self.time_column_dict[table_name]
            column_ref = self.table_column_ref_dict[table_name][column]
            selects.append(
                f'SELECT {index} AS table_index, MIN({column_ref}) AS '
                f'min_date, MAX({column_ref}) AS max_date\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
        with reading(self._connection) as cursor:
            cursor.execute('\nUNION ALL\n'.join(selects))
            for index, minimum, maximum in cursor.fetchall():
                out[unknown[int(index)]] = (
                    pd.Timestamp.max if minimum is None else Timestamp(minimum),
                    pd.Timestamp.min if maximum is None else Timestamp(maximum),
                )
        return out

    def _stats_time_range(
        self, table_name: str
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        column = self.time_column_dict[table_name]
        table = self._tables[table_name]
        if column not in table._source_column_dict:
            return None  # An expression, which has no statistics.
        try:
            stats = table._table_stats()
        except Exception:
            return None
        for row in stats:
            if row[0] == column and row[5] is not None and row[6] is not None:
                try:
                    return Timestamp(row[5]), Timestamp(row[6])
                except (TypeError, ValueError):
                    return None
        return None

    def _sample_entity_table(
        self,
        table_name: str,
        columns: set[str],
        num_rows: int,
        random_seed: int | None = None,
        entity_ids: list | None = None,
    ) -> pd.DataFrame:
        source_table = self.source_table_dict[table_name]
        filters: list[str] = []
        key = self.primary_key_dict[table_name]
        key_ref = self.table_column_ref_dict[table_name][key]
        if key not in source_table or source_table[key].is_nullable:
            filters.append(f'{key_ref} IS NOT NULL')
        time_column = self.time_column_dict.get(table_name)
        if time_column is not None and (
            time_column not in source_table
            or source_table[time_column].is_nullable
        ):
            filters.append(
                f'{self.table_column_ref_dict[table_name][time_column]} '
                'IS NOT NULL'
            )

        projections = self._projections(table_name, columns)
        if entity_ids is None:
            sql = (
                f'SELECT {", ".join(projections)}\n'
                f'FROM {self.source_name_dict[table_name]}'
            )
            if filters:
                sql += f'\nWHERE {" AND ".join(filters)}'
            # Trino's random() takes no seed, so a seeded sample orders by a
            # hash of the seed and the key: the same seed picks the same rows
            # on any connection, and another seed picks others.
            seed = 0 if random_seed is None else random_seed
            sql += (
                '\nORDER BY xxhash64(to_utf8(CAST(? AS varchar) || '
                f"'|' || CAST({key_ref} AS varchar)))\nLIMIT {int(num_rows)}"
            )
            with reading(self._connection) as cursor:
                cursor.execute(sql, [str(seed)])
                frame = _frame(cursor)
        else:
            dtype = self.table_dtype_dict[table_name][key]
            sql = self._json_cte(self._trino_type(table_name, key))
            sql += (
                f'SELECT {", ".join(projections)}\n'
                'FROM TMP\n'
                f'JOIN {self.source_name_dict[table_name]}\n'
                f'  ON {key_ref} = TMP.__KUMO_ID__'
            )
            if filters:
                sql += f'\nWHERE {" AND ".join(filters)}'
            frame = self._flatten(
                sql, [self._coerce_id(value, dtype) for value in entity_ids]
            )

        return _sanitize_trino_frame(
            df=frame,
            dtype_dict=self.table_dtype_dict[table_name],
            stype_dict=self.table_stype_dict[table_name],
        )

    def _sample_target(
        self,
        query: ValidatedPredictiveQuery,
        entity_df: pd.DataFrame,
        train_index: np.ndarray,
        train_time: pd.Series,
        num_train_examples: int,
        test_index: np.ndarray,
        test_time: pd.Series,
        num_test_examples: int,
        columns_dict: dict[str, set[str]],
        time_offset_dict: dict[
            tuple[str, str, str],
            tuple[pd.DateOffset | None, pd.DateOffset],
        ],
    ) -> tuple[pd.Series, np.ndarray, pd.Series, np.ndarray]:
        del num_train_examples, num_test_examples
        index = np.concatenate([train_index, test_index])
        entity_df = entity_df.iloc[index].reset_index(drop=True)
        time = pd.concat([train_time, test_time], axis=0, ignore_index=True)
        feat_dict, time_dict, batch_dict = self._sample_query_data(
            entity_table_name=query.entity_table,
            entity_pkey=entity_df[self.primary_key_dict[query.entity_table]],
            anchor_time=time,
            columns_dict={
                key: columns
                for key, columns in columns_dict.items()
                if key != query.entity_table
            },
            time_offset_dict=time_offset_dict,
        )
        feat_dict[query.entity_table] = entity_df
        time_column = self.time_column_dict.get(query.entity_table)
        if time_column in columns_dict[query.entity_table]:
            time_dict[query.entity_table] = entity_df[time_column]
        y, mask = PQueryPandasExecutor().execute(
            query=query,
            feat_dict=feat_dict,
            time_dict=time_dict,
            batch_dict=batch_dict,
            anchor_time=time,
        )
        train_mask = mask[: len(train_index)]
        test_mask = mask[len(train_index) :]
        boundary = int(train_mask.sum())
        train_y = y.iloc[:boundary]
        test_y = y.iloc[boundary:].reset_index(drop=True)
        return train_y, train_mask, test_y, test_mask

    def _by_pkey(
        self,
        table_name: str,
        index: pd.Series,
        columns: set[str],
        scored: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        if len(index) == 0:
            return pd.DataFrame(), np.empty(0, dtype=int)
        key = self.primary_key_dict[table_name]
        dtype = self.table_dtype_dict[table_name][key]
        key_ref = self.table_column_ref_dict[table_name][key]
        projections = self._projections(table_name, columns)
        sql = self._json_cte(self._trino_type(table_name, key))
        sql += (
            'SELECT * FROM (\n'
            '  SELECT TMP.__KUMO_BATCH__, '
            f'{", ".join(projections)},\n'
            '         ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__\n'
            f'           ORDER BY {key_ref}) AS __KUMO_ROW__,\n'
            '         COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) '
            'AS __KUMO_MATCHES__\n'
            '  FROM TMP\n'
            f'  JOIN {self.source_name_dict[table_name]}\n'
            f'    ON {key_ref} = TMP.__KUMO_ID__\n'
            ') AS sampled\n'
            'WHERE __KUMO_ROW__ = 1\n'
            'ORDER BY __KUMO_BATCH__'
        )
        frame = self._flatten(
            sql, [self._coerce_id(value, dtype) for value in index]
        )
        if '__KUMO_MATCHES__' in frame:
            matches = frame.pop('__KUMO_MATCHES__').to_numpy(dtype=np.int64)
            self._refuse_an_id_scored_twice(
                table_name,
                index,
                frame['__KUMO_BATCH__'].to_numpy(dtype=np.int64),
                matches,
                scored,
            )
        return self._sanitize_output(table_name, frame)

    def _by_fkey(
        self,
        table_name: str,
        foreign_key: str,
        index: pd.Series,
        num_neighbors: int,
        anchor_time: pd.Series | None,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        time_column = self.time_column_dict.get(table_name)
        dtype = self.table_dtype_dict[table_name][foreign_key]
        fields: tuple[str, ...] = ()
        rows: list
        if time_column is not None and anchor_time is not None:
            rows = [
                {'id': self._coerce_id(i, dtype), 'e': self._serialize_time(e)}
                for i, e in zip(index, anchor_time)
            ]
            fields = ('e',)
        else:
            rows = [self._coerce_id(i, dtype) for i in index]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        projections = self._projections(table_name, columns)
        time_ref = (
            None
            if time_column is None
            else self.table_column_ref_dict[table_name][time_column]
        )
        order_by = self._neighbor_order_by(table_name, time_ref, key_ref)
        time_type = (
            'timestamp(6)'
            if time_column is None
            else self._trino_type(table_name, time_column)
        )
        sql = self._json_cte(
            self._trino_type(table_name, foreign_key), fields, time_type
        )
        # Trino has no LIMIT inside a correlated LATERAL join, so the newest
        # `num_neighbors` rows of each entity are kept by rank instead.
        sql += (
            'SELECT * FROM (\n'
            f'  SELECT TMP.__KUMO_BATCH__, {", ".join(projections)},\n'
            '         ROW_NUMBER() OVER (PARTITION BY TMP.__KUMO_BATCH__\n'
            f'           ORDER BY {order_by}) AS __KUMO_ROW__\n'
            '  FROM TMP\n'
            f'  JOIN {self.source_name_dict[table_name]}\n'
            f'    ON {key_ref} = TMP.__KUMO_ID__\n'
        )
        if fields:
            assert time_ref is not None
            sql += f'   AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
        sql += (
            ') AS sampled\n'
            f'WHERE __KUMO_ROW__ <= {int(num_neighbors)}\n'
            'ORDER BY __KUMO_BATCH__, __KUMO_ROW__'
        )
        return self._sanitize_output(table_name, self._flatten(sql, rows))

    def _by_time(
        self,
        table_name: str,
        foreign_key: str,
        index: pd.Series,
        anchor_time: pd.Series,
        min_offset: pd.DateOffset | None,
        max_offset: pd.DateOffset,
        columns: set[str],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        time_column = self.time_column_dict[table_name]
        dtype = self.table_dtype_dict[table_name][foreign_key]
        time_type = self._trino_type(table_name, time_column)
        end_time = [
            self._serialize_time(value) for value in anchor_time + max_offset
        ]
        fields: tuple[str, ...]
        rows: list
        if min_offset is None:
            fields = ('e',)
            rows = [
                {'id': self._coerce_id(i, dtype), 'e': e}
                for i, e in zip(index, end_time)
            ]
        else:
            fields = ('e', 's')
            start_time = [
                self._serialize_time(value)
                for value in anchor_time + min_offset
            ]
            rows = [
                {'id': self._coerce_id(i, dtype), 'e': e, 's': s}
                for i, e, s in zip(index, end_time, start_time)
            ]

        key_ref = self.table_column_ref_dict[table_name][foreign_key]
        time_ref = self.table_column_ref_dict[table_name][time_column]
        projections = self._projections(table_name, columns)
        sql = self._json_cte(
            self._trino_type(table_name, foreign_key), fields, time_type
        )
        sql += (
            'SELECT TMP.__KUMO_BATCH__, '
            f'{", ".join(projections)}\n'
            'FROM TMP\n'
            f'JOIN {self.source_name_dict[table_name]}\n'
            f'  ON {key_ref} = TMP.__KUMO_ID__\n'
            f' AND {time_ref} <= TMP.__KUMO_END_TIME__\n'
        )
        if min_offset is not None:
            sql += f' AND {time_ref} > TMP.__KUMO_START_TIME__\n'
        # Trino returns rows that tie on the time column in whatever order
        # its splits finish, and a target sums these rows: a float sum taken
        # in another order differs in its last bits, so the same seed gave
        # another label. The tie-break the neighbours use makes it total.
        order_by = self._neighbor_order_by(table_name, None, key_ref)
        sql += f'ORDER BY TMP.__KUMO_BATCH__, {time_ref}, {order_by}'
        return self._sanitize_output(table_name, self._flatten(sql, rows))
