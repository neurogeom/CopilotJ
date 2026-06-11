# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Process lifecycle management -- cleanup registry for managed mode.

In Appose managed mode the aiohttp server runs in a daemon thread while
the Appose worker stdin loop occupies the main thread.  Resources created
on the daemon thread (e.g. JupyterClient) cannot register Python signal
handlers because ``signal.signal()`` requires the main thread.

This module provides a lightweight **cleanup registry** so that any
resource can register a cleanup callback from any thread.  The callbacks
are executed by:

* Signal handlers registered on the main thread (``appose_worker.py``).
* The aiohttp server ``on_shutdown`` hook (``server.py``).
* ``atexit`` as a final safety net.

Usage::

    from copilotj.core.lifecycle import register_cleanup


    def my_cleanup(): ...


    register_cleanup("my_resource", my_cleanup)
"""

import atexit
import logging
import threading
from typing import Callable

__all__ = ["register_cleanup", "run_cleanup"]

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_callbacks: list[tuple[str, Callable]] = []


def register_cleanup(name: str, callback: Callable) -> None:
    """Register a cleanup callback to run on shutdown or signal.

    Thread-safe: can be called from any thread (main, daemon, worker).
    Callbacks run in LIFO order (last registered runs first).
    """
    with _lock:
        _callbacks.append((name, callback))


def run_cleanup() -> None:
    """Run all registered cleanup callbacks in LIFO order.

    Called by:
    - Signal handler on main thread (managed mode, ``appose_worker.py``)
    - Server ``on_shutdown`` hook (``server.py``)
    """
    with _lock:
        callbacks = list(reversed(_callbacks))
        _callbacks.clear()
    for name, callback in callbacks:
        try:
            _log.info("Running cleanup: %s", name)
            callback()
        except Exception:
            _log.exception("Cleanup %s failed", name)


# Safety net: run all registered cleanups on process exit.
atexit.register(run_cleanup)
