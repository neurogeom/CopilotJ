# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Appose task script adapter for CopilotJ server.

This module is invoked as an Appose task script. It starts the aiohttp
server on port 0 (OS assigns a free port), returns the assigned port via
``task.outputs``, and keeps the server running in a background event loop
thread.

The Appose worker's main loop keeps the process alive. When Appose closes
stdin (graceful shutdown), the worker exits and the daemon thread is killed.
"""

import asyncio
import threading

from copilotj.core import load_env
from copilotj.server import Server

__all__ = ["start_server"]


def start_server() -> dict:
    """Start the aiohttp server on port 0 and return the assigned port.

    Returns:
        dict with ``"port"`` key containing the assigned port number.
    """
    load_env()
    server = Server()
    loop = asyncio.new_event_loop()

    async def _start():
        port = await server.start("127.0.0.1", 0)
        return port

    port = loop.run_until_complete(_start())

    # Keep the event loop alive in a daemon thread so the server continues
    # processing requests while the Appose worker main loop handles stdin.
    threading.Thread(target=loop.run_forever, daemon=True).start()

    return {"port": port}
