# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from .conftest import trino_connection, write_trino


def test_written_frames_read_back(trino_url: str, trino_schema: str) -> None:
    users = pd.DataFrame({'user_id': [1, 2, 3], 'name': ['a', 'b', 'c']})
    write_trino(trino_url, trino_schema, {'users': users})

    connection = trino_connection(trino_url)
    try:
        cursor = connection.cursor()
        cursor.execute(f'SELECT count(*) FROM {trino_schema}.users')
        assert cursor.fetchall() == [[3]]
    finally:
        connection.close()
