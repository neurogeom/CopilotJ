# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Appose task script adapter for CopilotJ server.

This module is invoked as an Appose task script. It starts the aiohttp
server on a port, returns the assigned port via ``task.outputs``, and keeps
the server running in a background event loop thread.

In managed mode, the server attempts to reuse the last-used port from
``$COPILOTJ_HOME/config.json`` for a stable URL across restarts.

The Appose worker's main loop keeps the process alive. When Appose closes
stdin (graceful shutdown), the worker exits and the daemon thread is killed.
"""

import asyncio
import threading
from urllib.parse import urlparse

from copilotj.core import load_env
from copilotj.core.config import load_managed_config, save_managed_config
from copilotj.server import Server

__all__ = ["start_server"]


def _extract_port(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return urlparse(url).port
    except ValueError:
        return None


def start_server() -> dict:
    """Start the aiohttp server and return the assigned port.

    In managed mode, reuses the previously saved port if available.
    Falls back to port 0 (OS-assigned) if the saved port is unavailable.

    Returns:
        dict with ``"port"`` key containing the assigned port number.
    """
    load_env()
    saved_port = _extract_port(load_managed_config().get("server_url"))

    server = Server()
    loop = asyncio.new_event_loop()

    async def _start():
        try:
            port = await server.start("127.0.0.1", saved_port or 0)
        except OSError:
            port = await server.start("127.0.0.1", 0)
        return port

    port = loop.run_until_complete(_start())

    # Keep the event loop alive in a daemon thread so the server continues
    # processing requests while the Appose worker main loop handles stdin.
    threading.Thread(target=loop.run_forever, daemon=True).start()

    # Persist the URL for next start
    save_managed_config({"server_url": f"http://127.0.0.1:{port}"})

    result = {"port": port}
    if saved_port and saved_port != port:
        result["port_changed"] = True
        result["previous_port"] = saved_port
    return result
