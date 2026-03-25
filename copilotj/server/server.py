# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging

import aiohttp.web as web
import aiohttp_cors

from copilotj.server.bridge import Bridge
from copilotj.server.threads import Threads
from copilotj.server.oauth_handler import OAuthHandler

__all__ = ["Server"]

_log = logging.getLogger("copilotj.server")


class Server:
    def __init__(self):
        super().__init__()
        self._bridge = Bridge()
        self._threads = Threads()
        self._oauth = OAuthHandler()
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

    def _create_app(self) -> web.Application:
        app = web.Application()

        r = app.router
        r.add_get("/api/ping", _on_ping)
        r.add_get("/api/plugins", self._bridge.on_plugin_connect)
        r.add_post("/api/plugins/events", self._bridge.on_forward_event)
        r.add_post("/api/threads", self._threads.new_thread)
        r.add_delete("/api/threads/{thread_id}", self._threads.del_thread)
        r.add_post("/api/threads/{thread_id}/posts", self._threads.new_thread_post)
        r.add_get("/api/threads/{thread_id}/config", self._threads.get_thread_config)
        r.add_post("/api/threads/{thread_id}/config", self._threads.update_thread_config)
        r.add_post("/api/threads/{thread_id}/optimize-prompt", self._threads.optimize_prompt_endpoint)
        r.add_post("/api/optimize-prompt", self._threads.optimize_prompt_standalone)

        # OAuth authentication routes
        r.add_get("/api/auth/status", self._oauth.status)
        r.add_post("/api/auth/login", self._oauth.login)
        r.add_post("/api/auth/logout", self._oauth.logout)
        r.add_get("/api/auth/token", self._oauth.get_token)

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
        return app


async def _on_ping(request: web.Request) -> web.Response:
    return web.Response(text="pong")
