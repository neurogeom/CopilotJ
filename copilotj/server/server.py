# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import logging

import aiohttp.web as web
import aiohttp_cors

from copilotj.core.config import Config
from copilotj.core.kb import ensure_faiss_index_async
from copilotj.core.lifecycle import run_cleanup
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
        r.add_get("/api/model/capabilities", self._on_model_capabilities)
        r.add_get("/api/models", self._on_list_models)
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
            # Run all registered resource cleanups (Jupyter kernels, etc.).
            run_cleanup()

        app.on_shutdown.append(on_shutdown)

        async def _ensure_kb(app: web.Application):
            asyncio.ensure_future(ensure_faiss_index_async())
            yield

        app.cleanup_ctx.append(_ensure_kb)

        async def _prepare_model_db(app: web.Application):
            from copilotj.core.config import resolve_vision_config
            from copilotj.core.model_info import ensure_model_db_async

            async def _download_and_resolve():
                await ensure_model_db_async()
                new_cfg = resolve_vision_config(self._cfg)
                self._cfg = new_cfg
                self._threads.update_cfg(new_cfg)

            asyncio.ensure_future(_download_and_resolve())
            yield

        app.cleanup_ctx.append(_prepare_model_db)
        return app

    async def _on_config(self, request: web.Request) -> web.Response:
        """Return the server's default configuration so the frontend can show the
        correct initial state (e.g. suppress the 'no model configured' warning when
        a model is already set via environment variables / .env.local).

        IMPORTANT: API keys are never exposed to the frontend.
        """
        cfg = self._cfg

        model = (
            {"name": cfg.llm_model, "api_key": None, "base_url": cfg.llm_base_url, "provider": cfg.llm_provider}
            if cfg.llm_model
            else None
        )
        vlm = (
            {
                "name": cfg.vlm_model,
                "api_key": None,
                "base_url": cfg.vlm_base_url,
                "provider": cfg.vlm_provider or cfg.llm_provider,
            }
            if cfg.vision_enabled and cfg.vlm_model
            else None
        )

        return web.json_response(
            {
                "model": model,
                "vlm": vlm,
                "proxy": cfg.llm_proxy,
                "kb_autosave": cfg.kb_autosave,
                "vision_enabled": cfg.vision_enabled,
                "llm_supports_vision": cfg.llm_supports_vision,
                "vlm_configured": cfg.vlm_configured,
            }
        )

    async def _on_model_capabilities(self, request: web.Request) -> web.Response:
        """Return capability information for a model.

        Uses the LiteLLM model database to check features like vision support.
        Accepts an optional ``?model=`` query parameter to check an arbitrary
        model name.  Without the parameter, returns info for the server's
        configured LLM and VLM.
        """
        from copilotj.core.model_info import get_model_capabilities

        # Check arbitrary model via query parameter
        model_param = request.query.get("model")
        if model_param:
            caps = await asyncio.to_thread(get_model_capabilities, model_param)
            return web.json_response(
                {
                    "model": model_param,
                    "supports_vision": caps.supports_vision,
                    "supports_function_calling": caps.supports_function_calling,
                    "source": caps.source,
                }
            )

        cfg = self._cfg

        def _caps_dict(model: str) -> dict | None:
            if not model:
                return None
            caps = get_model_capabilities(model)
            return {
                "supports_vision": caps.supports_vision,
                "supports_function_calling": caps.supports_function_calling,
                "context_window": caps.context_window,
                "max_output_tokens": caps.max_output_tokens,
                "source": caps.source,
            }

        llm_caps, vlm_caps = await asyncio.gather(
            asyncio.to_thread(_caps_dict, cfg.llm_model),
            asyncio.to_thread(_caps_dict, cfg.vlm_model),
        )

        return web.json_response(
            {
                "llm": {"model": cfg.llm_model, "capabilities": llm_caps} if cfg.llm_model else None,
                "vlm": {"model": cfg.vlm_model, "capabilities": vlm_caps} if cfg.vlm_model else None,
                "vision_enabled": cfg.vision_enabled,
                "llm_supports_vision": cfg.llm_supports_vision,
                "vlm_configured": cfg.vlm_configured,
            }
        )

    async def _on_list_models(self, request: web.Request) -> web.Response:
        """List available models for one or all providers.

        Query params:

        - ``provider`` (optional): restrict to a single provider.
        - ``base_url`` (optional): Ollama host (default ``http://localhost:11434``);
          ignored for catalog providers.

        Cloud providers come from the cached LiteLLM catalog; Ollama is queried
        live.  Never returns 5xx — Ollama being unreachable surfaces as
        ``source: "unreachable"`` with an empty model list.
        """
        from copilotj.core.model_listing import list_provider_models

        provider = request.query.get("provider")
        base_url = request.query.get("base_url")

        if provider:
            result = await list_provider_models(provider, base_url=base_url)
            return web.json_response(result)

        # Grouped: resolve all supported providers concurrently. Catalog-backed
        # cloud providers (incl. DeepSeek / OpenRouter) are listed so the model
        # picker can offer autocomplete for them; Ollama is queried live.
        providers = ["openai", "anthropic", "gemini", "deepseek", "openrouter", "ollama"]
        results = await asyncio.gather(*(list_provider_models(p, base_url=base_url) for p in providers))
        return web.json_response({"providers": {r["provider"]: r for r in results}})


async def _on_ping(request: web.Request) -> web.Response:
    return web.Response(text="pong")
