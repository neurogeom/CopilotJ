# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Appose init script adapter for CopilotJ server.

This module is designed to be called from an Appose **init script**, not a
task script.  Using the init model ensures that all heavy library imports
(numpy, scikit-image, stardist, etc.) happen on the **main thread** before
the Appose worker enters its stdin I/O loop.  This avoids the stdin/thread
deadlock on Windows (see numpy/numpy#24290, apposed/appose#23).

Usage from the Java side::

    Service python = env.python().init(
        "from copilotj.appose_worker import start_server; start_server()"
    );
    // The server is now running.  Query the port with a lightweight task:
    Task t = python.task(
        "from copilotj.appose_worker import query_port; task.outputs.update(query_port())"
    );
    t.waitFor();
    int port = ((Number) t.outputs().get("port")).intValue();
    // ...
    // When done, shut down the server:
    python.task("from copilotj.appose_worker import stop_server").waitFor();

In managed mode, the server attempts to reuse the last-used port from
``$COPILOTJ_HOME/config.json`` for a stable URL across restarts.

The Appose worker's main loop keeps the process alive.  When Appose closes
stdin (graceful shutdown), the worker exits and the daemon thread is killed.
"""

import asyncio
import logging
import signal as _signal
import sys
import threading
import traceback
from urllib.parse import urlparse

from copilotj.core import load_config
from copilotj.core.config import bootstrap_assets, load_managed_config, save_managed_config
from copilotj.core.lifecycle import run_cleanup as _run_cleanup
from copilotj.server import Server

__all__ = ["start_server", "query_port", "query_status", "stop_server"]

_log = logging.getLogger(__name__)

# Module-level references to the running server and its event loop.
# Set by start_server(); read by query_port() / query_status() / stop_server().
_server: Server | None = None
_loop: asyncio.AbstractEventLoop | None = None
_previous_port: int | None = None
# Full traceback captured when start_server() fails. Appose swallows init-script
# exceptions (printing only str(e) to stderr), so the module records the failure
# itself and query_status() surfaces it to Java via the task/stdout channel that
# the bridge already inspects. See upstream appose issue #25 pitfalls #3/#8.
_startup_error: str | None = None


def _extract_port(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return urlparse(url).port
    except ValueError:
        return None


def start_server() -> None:
    """Start the aiohttp server (call from Appose init script).

    Starts the server on the main thread before the Appose worker enters its
    stdin I/O loop.  This ensures all heavy imports (numpy, skimage, stardist,
    etc.) complete on the main thread, avoiding stdin/thread deadlocks on
    Windows.

    The server runs in a background daemon thread so it keeps processing
    requests while the Appose worker main loop handles stdin.

    The assigned port is persisted to ``$COPILOTJ_HOME/config.json`` and can
    be retrieved later via :func:`query_port`.

    On failure the exception is captured into the module global
    ``_startup_error`` (full traceback) and logged via ``_log.exception``.  It
    is deliberately NOT re-raised: Appose wraps init-script execution in a
    ``try/except BaseException`` that only prints a one-line ``[WARNING]`` to
    stderr (no traceback), so re-raising would lose the cause entirely.  The
    next :func:`query_status` call surfaces the captured error to the Java
    bridge over the reliable task/stdout channel (upstream appose issue #25
    pitfalls #3 and #8).
    """
    global _server, _loop, _previous_port, _startup_error

    try:
        _ensure_utf8_stdout()
        bootstrap_assets()
        saved_port = _extract_port(load_managed_config().get("server_url"))

        cfg = load_config()
        _server = Server(cfg)
        _loop = asyncio.new_event_loop()

        async def _start():
            assert _server is not None
            try:
                return await _server.start("127.0.0.1", saved_port or 0)
            except OSError:
                return await _server.start("127.0.0.1", 0)

        _loop.run_until_complete(_start())

        # Keep the event loop alive in a daemon thread so the server continues
        # processing requests while the Appose worker main loop handles stdin.
        threading.Thread(target=_loop.run_forever, daemon=True).start()

        # Persist the URL for next start
        port = _server.port
        save_managed_config({"server_url": f"http://127.0.0.1:{port}"})

        # Remember if the port changed from the saved value.
        _previous_port = saved_port if (saved_port and saved_port != port) else None

        # Register signal handlers on the main thread (managed mode).
        # The daemon thread cannot register signal handlers, so we do it here
        # before returning control to Appose's stdin loop.  When a signal
        # arrives, Python interrupts the blocking input() call and invokes
        # the handler, which runs all registered lifecycle cleanups.
        def _managed_signal_handler(signum: int, _frame) -> None:
            _log.warning("Received signal %d in managed mode, running cleanup", signum)
            _run_cleanup()
            raise SystemExit(0)

        _signal.signal(_signal.SIGTERM, _managed_signal_handler)
        _signal.signal(_signal.SIGINT, _managed_signal_handler)

        # Success: clear any previously captured startup error (re-entrancy).
        _startup_error = None
    except BaseException:
        # Appose swallows init-script exceptions into a one-line stderr warning,
        # so capture the full traceback ourselves and drop any partially
        # constructed state; query_status() then surfaces it to Java.
        _startup_error = traceback.format_exc()
        _server = None
        _loop = None
        _log.exception("Managed server failed to start")


def query_port() -> dict:
    """Return the port of the running managed server.

    Lightweight helper meant to be called from an Appose **task** (not init).
    Reads the port directly from the live server instance.

    Returns:
        dict with ``"port"`` key.  If the saved port was unavailable,
        also includes ``"port_changed"`` (``True``) and ``"previous_port"``
        (the originally requested port).

    Raises:
        RuntimeError: If the server has not been started.
    """
    if _server is None:
        raise RuntimeError("Server has not been started")
    result: dict = {"port": _server.port}
    if _previous_port is not None:
        result["port_changed"] = True
        result["previous_port"] = _previous_port
    return result


def query_status() -> dict:
    """Return managed-server startup status (call from an Appose task).

    Separate from :func:`query_port`: verifies whether :func:`start_server`
    succeeded and returns the captured traceback if it failed.  Appose swallows
    init-script exceptions (printing only ``str(e)`` to stderr), so
    :func:`start_server` stores any failure in ``_startup_error``; this task
    surfaces it to the Java bridge over the reliable task/stdout channel so the
    real cause reaches the user-facing dialog instead of the opaque
    "Server has not been started" symptom (upstream appose issue #25 pitfalls
    #3 and #8).

    Returns:
        dict with ``"started"`` (``bool``) and ``"startup_error"`` (the full
        traceback ``str`` captured on failure, or ``None`` on success).
    """
    return {
        "started": _server is not None,
        "startup_error": _startup_error,
    }


def stop_server() -> dict:
    """Stop the managed server gracefully.

    Delegates to :meth:`copilotj.server.Server.stop` which cleans up the
    aiohttp runner, triggers on_shutdown hooks, and tears down connections.
    Can be called from an Appose task to shut down the server cleanly.

    Returns:
        dict with ``"stopped"`` key set to ``True``.

    Raises:
        RuntimeError: If the server has not been started.
    """
    global _server, _loop, _previous_port

    if _server is None or _loop is None:
        raise RuntimeError("Server has not been started")

    _log.info("Stopping managed server on port %d", _server.port)

    future = asyncio.run_coroutine_threadsafe(_server.stop(), _loop)
    future.result(timeout=10)

    _loop.call_soon_threadsafe(_loop.stop)

    _server = None
    _loop = None
    _previous_port = None

    _log.info("Managed server stopped")
    return {"stopped": True}


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    When running under Appose on Windows, stdout may be a pipe or socket
    with GBK encoding (the system OEM code page).  Emoji and other
    non-GBK characters in print() calls would raise UnicodeEncodeError.
    Reconfigure to UTF-8 with replacement so the process never crashes.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            reconfigure = getattr(stream, "reconfigure")
            if reconfigure is not None:
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (AttributeError, OSError):
                    pass
