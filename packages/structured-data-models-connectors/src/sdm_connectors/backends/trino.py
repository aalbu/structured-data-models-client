# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Trino, and engines that speak its protocol, through the ``trino`` client."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import urlsplit

from sdm_connectors.sql import require_driver

if TYPE_CHECKING:
    import trino.dbapi

    Connection: TypeAlias = trino.dbapi.Connection
else:
    _dbapi = require_driver('trino', 'trino', 'trino.dbapi', absent=('trino',))
    Connection = _dbapi.Connection

#: What every connection this module opens tells the server about itself.
SOURCE = 'structured-data-models-client'


def connect(
    url: str,
    *,
    user: str,
    access_token: str | None = None,
    source: str = SOURCE,
    **kwargs: Any,
) -> Connection:
    r"""Connect to a Trino server, in UTC.

    The session's time zone is UTC whatever the server's is, so a timestamp
    without one compares the same way on every connection. ``access_token``
    is sent as a bearer token; otherwise only ``user`` is sent, which a
    server without authentication accepts.

    Args:
        url: The server, as ``http://host:port`` or ``https://host:port``.
        user: Who the statements run as.
        access_token: A JWT to authenticate with.
        source: What the server records as each statement's source.
        **kwargs: Passed to :func:`trino.dbapi.connect`; ``timezone`` cannot
            be among them.
    """
    if 'timezone' in kwargs:
        raise ValueError(
            'connect() always opens a UTC session; the time zone cannot be '
            'given'
        )
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError(f'url must be http(s)://host:port, got {url!r}')
    if access_token is not None:
        from trino.auth import JWTAuthentication

        kwargs['auth'] = JWTAuthentication(access_token)
    from trino.dbapi import connect as trino_connect

    return trino_connect(
        host=parts.hostname,
        port=parts.port or (443 if parts.scheme == 'https' else 80),
        http_scheme=parts.scheme,
        user=user,
        source=source,
        timezone='UTC',
        **kwargs,
    )


@contextmanager
def reading(connection: Connection) -> Iterator[Any]:
    r"""A cursor for one read.

    Trino runs each statement on its own unless a transaction is opened, so a
    read leaves nothing behind; the cursor is closed afterwards all the same.
    """
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
