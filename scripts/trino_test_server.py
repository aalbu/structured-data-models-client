#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start a Trino server for the Trino backend's tests, and print its URL.

    python scripts/trino_test_server.py            # start, wait, print the export line
    python scripts/trino_test_server.py --stop     # stop it

The server is a throwaway container with Trino's ``memory`` catalog, the one
the tests write their tables to. ``--access-control FILE`` mounts a
file-based access control rules file, which the access tests use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

IMAGE = 'trinodb/trino:483'
NAME = 'sdm-trino'
PORT = 8089
TIMEOUT_SECONDS = 120


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['docker', *args], check=True, capture_output=True, text=True
    )


def _started(url: str) -> bool:
    try:
        with urllib.request.urlopen(f'{url}/v1/info', timeout=2) as response:
            return json.load(response).get('starting') is False
    except (urllib.error.URLError, ConnectionError, TimeoutError, ValueError):
        return False


def start(access_control: Path | None) -> str:
    url = f'http://localhost:{PORT}'
    args = ['run', '-d', '--rm', '--name', NAME, '-p', f'{PORT}:8080']
    if access_control is not None:
        rules = access_control.resolve(strict=True)
        # Outside the repository, so starting a server leaves no file behind.
        properties = Path(tempfile.mkdtemp(prefix='sdm-trino-')) / (
            'access-control.properties'
        )
        properties.write_text(
            'access-control.name=file\n'
            'security.config-file=/etc/trino/rules.json\n'
        )
        args += [
            '-v',
            f'{rules}:/etc/trino/rules.json:ro',
            '-v',
            f'{properties}:/etc/trino/access-control.properties:ro',
        ]
    _docker(*args, IMAGE)
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while not _started(url):
        if time.monotonic() > deadline:
            raise SystemExit(f'{NAME} did not start within {TIMEOUT_SECONDS}s')
        time.sleep(2)
    return url


def stop() -> None:
    subprocess.run(
        ['docker', 'rm', '-f', NAME], check=False, capture_output=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stop', action='store_true', help='stop the server')
    parser.add_argument(
        '--access-control', type=Path, help='file-based access control rules'
    )
    options = parser.parse_args()
    if options.stop:
        stop()
        return
    url = start(options.access_control)
    print(f'export SDM_TRINO_TEST_URL={url}')


if __name__ == '__main__':
    sys.exit(main())
