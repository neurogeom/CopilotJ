# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging

import aiohttp.web as web
import aiohttp_cors

from copilotj.core.config import Config
from copilotj.server.bridge import Bridge
from copilotj.server.threads import Threads

__all__ = ["Server"]

_log = logging.getLogger("copilotj.server")


class Server:
    def __init__(self, cfg: Config):
        super().__init__()
        self._cfg = cfg
        self._bridge = Bridge()
        self._threads = Threads(cfg, bridge=self._bridge)
        self._app = self._create_app()

    def add_background_task(self, task: asyncio.Task) -> None:
        async def _run(_app: web.Application):
            yield

            nonlocal task
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task  # Ensure any exceptions etc. are raised.

        self._app.cleanup_ctx.append(_run)

    def run(self, host: str, port: int):
        _log.info(f"Listening on {host}:{port}")
        web.run_app(self._app, host=host, port=port)

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        """Start the server and return the actual bound port.

        Unlike ``run()``, this does not block. The caller is responsible for
        keeping the event loop alive (e.g. via ``loop.run_forever()``).
        """
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, port)
        await site.start()
        self._port = site._server.sockets[0].getsockname()[1]
        return self._port

    @property
    def port(self) -> int | None:
        """The bound port number, or ``None`` if the server is not running."""
        return self._port

    async def stop(self) -> None:
        """Shut down the server gracefully.

        ``AppRunner.cleanup()`` handles the full shutdown sequence: it stops
        listening sites, fires ``on_shutdown`` hooks (closing threads and
        WebSocket connections), and then runs ``on_cleanup`` hooks.
        """
        if hasattr(self, "_runner") and self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._port = None

    def _create_app(self) -> web.Application:
        app = web.Application()

        r = app.router
        r.add_get("/api/ping", _on_ping)
        r.add_get("/api/config", self._on_config)
        r.add_get("/api/plugins", self._bridge.on_plugin_connect)
        r.add_post("/api/plugins/events", self._bridge.on_forward_event)
        r.add_post("/api/threads", self._threads.new_thread)
        r.add_delete("/api/threads/{thread_id}", self._threads.del_thread)
        r.add_post("/api/threads/{thread_id}/posts", self._threads.new_thread_post)
        r.add_get("/api/threads/{thread_id}/config", self._threads.get_thread_config)
        r.add_post("/api/threads/{thread_id}/config", self._threads.update_thread_config)
        r.add_post("/api/threads/{thread_id}/optimize-prompt", self._threads.optimize_prompt_endpoint)
        r.add_post("/api/optimize-prompt", self._threads.optimize_prompt_standalone)

        cors = aiohttp_cors.setup(  # TODO: configure CORS
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods=["POST", "GET", "OPTIONS"],
                )
            },
        )
        for route in app.router.routes():
            cors.add(route)

        async def on_shutdown(app: web.Application) -> None:
            await asyncio.gather(self._threads.close(), self._bridge.close())

        app.on_shutdown.append(on_shutdown)

        async def _ensure_kb(app: web.Application):
            from copilotj.core.kb import ensure_faiss_index_async

            asyncio.ensure_future(ensure_faiss_index_async())
            yield

        app.cleanup_ctx.append(_ensure_kb)
        return app

    async def _on_config(self, request: web.Request) -> web.Response:
        """Return the server's default configuration so the frontend can show the
        correct initial state (e.g. suppress the 'no model configured' warning when
        a model is already set via environment variables / .env.local)."""
        model_name = self._cfg.llm_model
        if model_name:
            return web.json_response({"model": {"name": model_name, "api_key": None, "base_url": None}})
        return web.json_response({"model": None})


async def _on_ping(request: web.Request) -> web.Response:
    return web.Response(text="pong")
