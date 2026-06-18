# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import traceback
import uuid
from asyncio import Future
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, AsyncGenerator, Literal, override

import aiohttp.web as web
import langfuse
import pydantic
from langfuse import propagate_attributes

from copilotj.core import UI, UIEvent, UIEventPost, UIEventState
from copilotj.core.config import Config, resolve_vision_config
from copilotj.core.ui import UIEventContentMarkdown
from copilotj.multiagent.leader_multiagent import LeaderDriven
from copilotj.plugin.api import BridgePluginAPI, PluginAPI

if TYPE_CHECKING:
    from copilotj.server.bridge import Bridge

__all__ = ["Threads"]

# Timeout for thread locks in seconds (300ms)
THREAD_LOCK_TIMEOUT = 0.3

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTEL instrumentation for Langfuse tracing
# ---------------------------------------------------------------------------
# After `langfuse.Langfuse()` creates the global TracerProvider, calling
# `.instrument()` on these instrumentors makes all Anthropic / Gemini SDK
# calls automatically emit OTEL spans that Langfuse picks up.
# ---------------------------------------------------------------------------

_otel_instrumented = False


def _setup_otel_instrumentation() -> None:
    """Initialize OTEL instrumentors for Anthropic and Gemini SDKs.

    Safe to call multiple times (idempotent).  Each instrumentor is wrapped
    in ``try/except`` so the server starts even if the OTEL packages or their
    runtime dependencies are unavailable.

    Note: the instrumentors' ``.instrument()`` methods can raise *any*
    exception, not just ``ImportError`` — e.g. ``openinference`` raises a
    plain ``Exception`` ("Could not import google-genai. ...") when the
    installed ``google-genai`` is a version that lacks the submodule the
    instrumentor wraps.  Tracing is best-effort, so we never let a failure
    here abort server startup.
    """
    global _otel_instrumented
    if _otel_instrumented:
        return
    _otel_instrumented = True

    try:
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

        AnthropicInstrumentor().instrument()
        _log.debug("OTEL Anthropic instrumentor enabled")
    except Exception:
        _log.debug("Anthropic OTEL instrumentor unavailable, skipping", exc_info=True)

    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        GoogleGenAIInstrumentor().instrument()
        _log.debug("OTEL Google GenAI instrumentor enabled")
    except Exception:
        _log.debug("Google GenAI OTEL instrumentor unavailable, skipping", exc_info=True)


dumpable = str | int | float | bool | pydantic.BaseModel

ROLE_SYSTEM = "system"


class _UseServerModel(pydantic.BaseModel):
    """Sent by the client to mean "use the server's env-configured model"."""

    use_server: Literal[True]


class _ExplicitModel(pydantic.BaseModel):
    """An explicit, user-configured model. Also the resolved shape returned to the UI."""

    name: str
    api_key: str | None
    base_url: str | None = None
    provider: str | None = None


# A model slot in a config query: either explicit, or "use the server's".
_ConfigModel = _UseServerModel | _ExplicitModel


class _Config(pydantic.BaseModel):
    model: _ExplicitModel


class _ConfigQuery(pydantic.BaseModel):
    model: _ConfigModel | None = None
    vlm: _ConfigModel | None = None
    vision_enabled: bool | None = None
    proxy: str | None = None
    tavily_api_key: str | None = None
    kb_autosave: bool = False


class _ThreadConfigUpdate(pydantic.BaseModel):
    """Body of POST /threads/{id}/config — a single model slot (union)."""

    model: _ConfigModel | None = None


class _NewThread(pydantic.BaseModel):
    id: str
    config: _Config


class _Signal(pydantic.BaseModel):
    signal: Literal["end"]


class _OptimizePrompt(pydantic.BaseModel):
    prompt: str


def _resolve_config(cfg: Config, config: _ConfigQuery | None) -> Config:
    """Merge runtime overrides from the web UI into server-wide Config.

    A ``use_server`` slot leaves the server's env config untouched; an explicit
    model is applied VERBATIM (a null ``api_key`` means "no key", e.g. Ollama — it
    no longer silently borrows the server's key).
    """
    if config is None:
        return cfg

    overrides: dict = {}

    def _apply(m: _ConfigModel, *, prefix: str) -> None:
        if isinstance(m, _UseServerModel):
            return  # use the server's env config for this slot → no overrides
        overrides[f"{prefix}_model"] = m.name
        overrides[f"{prefix}_api_key"] = m.api_key
        overrides[f"{prefix}_base_url"] = m.base_url
        if m.provider is not None:
            overrides[f"{prefix}_provider"] = m.provider

    if config.model is not None:
        _apply(config.model, prefix="llm")
    if config.vlm is not None:
        _apply(config.vlm, prefix="vlm")
    if config.proxy is not None:
        overrides["llm_proxy"] = config.proxy
    if config.tavily_api_key is not None:
        overrides["tavily_api_key"] = config.tavily_api_key
    if config.kb_autosave:
        overrides["kb_autosave"] = config.kb_autosave
    if config.vision_enabled is not None:
        overrides["vision_enabled"] = config.vision_enabled

    return resolve_vision_config(replace(cfg, **overrides)) if overrides else cfg


def _check_llm_config(cfg: Config) -> str | None:
    """Return a user-facing configuration error, or None when usable."""
    if not cfg.llm_model:
        return "No model configured. Please click the Settings gear icon in the sidebar to set up a model."
    if not cfg.llm_model.startswith("ollama/") and not cfg.llm_api_key:
        return f"No API key configured for model {cfg.llm_model}. Please set an API key in Settings."
    return None


class _Thread(UI):
    def __init__(
        self,
        thread_id: str,
        cfg: Config,
        *,
        config: _ConfigQuery | None = None,
        trace_context: langfuse.Langfuse | None,
        bridge: "Bridge",
    ):
        self.thread_id = thread_id
        self._trace_ctx = trace_context

        self._mailbox = asyncio.Queue[UIEvent | _Signal]()

        self._apis: PluginAPI = BridgePluginAPI(bridge)
        client_apis = self._apis.attach_single_client()  # TODO: should from frontend

        # Merge runtime config override into server-wide Config
        resolved = _resolve_config(cfg, config)

        self._agent = LeaderDriven(
            apis=client_apis,
            ui=self,
            cfg=resolved,
        )
        self._post_task: asyncio.Task[None] | None = None
        self._post_done: asyncio.Event | None = None
        self._task_future: Future[str | None] | None = None
        self._confirmation_future: Future[bool] | None = None

        self._config = _Config(model=_ExplicitModel(name=self._agent.model_client.get_model(), api_key=None))

    async def on_post(self, prompt: str | bool) -> AsyncGenerator[UIEvent, None]:
        """Handle incoming chat messages."""
        if isinstance(prompt, bool):
            assert (
                (self._confirmation_future is not None and not self._confirmation_future.done())
                and (self._post_task is not None and not self._post_task.done())
                and (self._post_done is not None and not self._post_done.is_set())
            ), "Attempted to resolve a confirmation, but none is pending."
            self._confirmation_future.set_result(prompt)
            self._confirmation_future = None

        elif self._task_future is not None:
            assert (
                not self._task_future.done()
                and (self._post_task is not None and not self._post_task.done())
                and (self._post_done is not None and not self._post_done.is_set())
            ), "Attempted to continue a thread, but none is pending."
            self._task_future.set_result(prompt)
            self._task_future = None

        else:
            assert self._confirmation_future is None and self._task_future is None, (
                "Attempted to start a new post while a post is pending."
            )
            self._post_done = asyncio.Event()
            self._post_task = asyncio.create_task(self._run_agent(prompt, self._post_done))

        cleanup = False
        try:
            flag = True
            while flag:
                chunk = asyncio.create_task(self._mailbox.get())
                done_task = asyncio.create_task(self._post_done.wait())
                done, pending = await asyncio.wait([chunk, done_task], return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    message = await task
                    if message is True:
                        cleanup = True
                        flag = False
                        continue  # done

                    match message:
                        case _Signal(signal="end"):
                            flag = False  # wait for the user to respond, do not clean up yet

                        case _:
                            yield message

            # consume the rest chunk
            while True:
                try:
                    message = self._mailbox.get_nowait()
                    match message:
                        case _Signal(signal="end"):
                            pass

                        case _:
                            yield message

                except asyncio.QueueEmpty:
                    break

        except asyncio.CancelledError:
            cleanup = True

        finally:
            if cleanup:
                if self._post_task and not self._post_task.done():
                    self._post_task.cancel()
                    # Wait for the agent task to finish cancellation
                    await asyncio.gather(self._post_task, return_exceptions=True)

                self._post_task = None
                self._post_done = None

    def get_config(self) -> _Config:
        return self._config

    def update_config(
        self, *, model: str | None, api_key: str | None, base_url: str | None = None, provider: str | None = None
    ) -> None:
        self._agent.update_config(model=model, api_key=api_key, base_url=base_url, provider=provider)
        self._config = _Config(
            model=_ExplicitModel(
                name=self._agent.model_client.get_model(),
                api_key=self._agent.model_client.get_api_key(),
                base_url=base_url,
                provider=provider,
            )
        )

    def reset_config(self, base_cfg: Config) -> None:
        """Reset this thread's model to the server's env-configured model ("use server")."""
        self._agent.update_config(
            model=base_cfg.llm_model,
            api_key=base_cfg.llm_api_key,
            base_url=base_cfg.llm_base_url,
            provider=base_cfg.llm_provider,
        )
        self._config = _Config(
            model=_ExplicitModel(
                name=self._agent.model_client.get_model(),
                api_key=None,
                base_url=base_cfg.llm_base_url,
                provider=base_cfg.llm_provider,
            )
        )

    async def _run_agent(self, prompt: str, done_event: asyncio.Event) -> None:
        """Run the chat with the agent."""
        try:
            if self._trace_ctx is None:
                # Tracing disabled (e.g. langfuse failed to initialize): run the
                # agent directly, skipping the Langfuse/OTEL context managers.
                await self._agent.run(prompt, trace_ctx=None)
            else:
                with propagate_attributes(session_id=self.thread_id):
                    with self._trace_ctx.start_as_current_observation(
                        as_type="span", name="thread", metadata={"thread_id": self.thread_id}, input=prompt
                    ):
                        await self._agent.run(prompt, trace_ctx=self._trace_ctx)

        except Exception:
            _log.exception("Agent run failed for thread %s", self.thread_id)
            self._agent.log_error(f"Agent run failed for thread {self.thread_id}:\n{traceback.format_exc()}")
        finally:
            done_event.set()  # Signal that the chat is done

    def abort(self) -> None:
        """Abort any in-flight agent run promptly.

        Lightweight vs :meth:`close`: sets the abort event only — does not tear
        down plugin APIs or the mailbox. The abort endpoint uses this so a user
        "stop" interrupts a retry backoff immediately; the normal fetch-abort →
        disconnect → :meth:`close` path still runs for teardown.
        """
        self._agent.abort()

    async def close(self) -> None:
        self._agent.abort()
        self._mailbox.task_done()
        await self._apis.close()

    # UI
    #
    # NOTE: the following part will be called in the agent thread, not in the web server thread

    @override
    async def send(self, event: UIEvent) -> None:
        await self._mailbox.put(event)

    @override
    async def request_user_confirm(self, role: str, message: str | None = None) -> bool:
        assert (self._confirmation_future is None or self._confirmation_future.done()) and (
            self._task_future is None or self._task_future.done()
        ), "Another user interaction is already in progress."
        self._confirmation_future = Future[bool]()

        if message is not None:
            await self._mailbox.put(UIEventPost(role=role, data=[UIEventContentMarkdown(markdown=message)]))

        await self._mailbox.put(UIEventState(role=role, data="confirmation_request"))
        await self._mailbox.put(_Signal(signal="end"))  # Signal that the agent has finished

        try:
            # Pause and wait for the future to be resolved by the confirm endpoint
            return await self._confirmation_future
        finally:
            self._confirmation_future = None  # Clean up

    @override
    async def request_user_manipulate(self, role: str, message: str | None = None) -> str | None:
        assert (self._confirmation_future is None or self._confirmation_future.done()) and (
            self._task_future is None or self._task_future.done()
        ), "Another user interaction is already in progress."
        self._task_future = Future[str | None]()

        # Dont send message since the user already has the context
        await self._mailbox.put(_Signal(signal="end"))  # Signal that the agent has finished

        try:
            # Pause and wait for the future to be resolved
            return await self._task_future
        finally:
            self._task_future = None  # Clean up


class Threads:
    def __init__(self, cfg: Config, *, bridge: "Bridge"):
        super().__init__()
        self._cfg = cfg
        self._bridge = bridge
        self._threads: dict[str, tuple[_Thread, threading.Lock]] = {}
        self._threads_lock = threading.Lock()
        # langfuse.Langfuse() validates keys/network at construction time and is
        # the most likely single point to take down Server(cfg) on a misconfigured
        # machine. Degrade to running without tracing instead of crashing the whole
        # server; _run_agent skips the context managers when _trace_ctx is None.
        try:
            self._trace_ctx = langfuse.Langfuse()
            _setup_otel_instrumentation()
        except Exception:
            _log.exception("Langfuse tracing initialization failed; continuing without tracing")
            self._trace_ctx = None

    def update_cfg(self, cfg: Config) -> None:
        """Update the stored config (e.g. after async vision resolution)."""
        self._cfg = cfg

    async def new_thread(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception as e:
            return web.Response(status=500, text=f"Error processing request: {e}")

        if (config := data.get("config")) is not None:
            try:
                config = _ConfigQuery.model_validate(config)
            except pydantic.ValidationError as e:
                return web.Response(status=400, text=f"Invalid configuration: {e}")

        resolved = _resolve_config(self._cfg, config)
        if message := _check_llm_config(resolved):
            return web.Response(status=400, text=message)

        thread_id = str(uuid.uuid4())
        thread = _Thread(thread_id, resolved, config=None, trace_context=self._trace_ctx, bridge=self._bridge)
        thread_lock = threading.Lock()
        with self._threads_lock:
            self._threads[thread_id] = (thread, thread_lock)

        return web.Response(status=200, text=_NewThread(id=thread_id, config=thread.get_config()).model_dump_json())

    async def new_thread_post(self, request: web.Request) -> web.StreamResponse | web.Response:
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)
        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")
        thread, thread_lock = thread_tuple

        # Acquire the thread lock before processing the request with a timeout
        if not thread_lock.acquire(timeout=THREAD_LOCK_TIMEOUT):
            return web.Response(
                status=408, text=f"Request timeout waiting for thread lock after {THREAD_LOCK_TIMEOUT * 1000:.0f}ms."
            )

        try:
            data = await request.json()
            prompt = data.get("prompt")
            if prompt is None or not isinstance(prompt, str):
                return web.Response(status=400, text="'prompt' field must be a string.")

            response = await self._new_response(request)
            return await self._on_post(response, thread, prompt)
        except Exception as e:
            return web.Response(status=500, text=f"Error processing request: {e}")
        finally:
            thread_lock.release()

    async def get_thread_config(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)
        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")
        thread, thread_lock = thread_tuple

        # Acquire the thread lock before processing the request with a timeout
        if not thread_lock.acquire(timeout=THREAD_LOCK_TIMEOUT):
            return web.Response(
                status=408, text=f"Request timeout waiting for thread lock after {THREAD_LOCK_TIMEOUT * 1000:.0f}ms."
            )

        try:
            return web.json_response(thread.get_config().model_dump())
        except Exception as e:
            return web.Response(status=500, text=f"Error retrieving configuration: {e}")
        finally:
            thread_lock.release()

    async def update_thread_config(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)
        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")
        thread, thread_lock = thread_tuple

        # Acquire the thread lock before processing the request with a timeout
        if not thread_lock.acquire(timeout=THREAD_LOCK_TIMEOUT):
            return web.Response(
                status=408, text=f"Request timeout waiting for thread lock after {THREAD_LOCK_TIMEOUT * 1000:.0f}ms."
            )

        try:
            try:
                data = await request.json()
            except Exception as e:
                return web.Response(status=500, text=f"Error processing request: {e}")

            config = _ThreadConfigUpdate.model_validate(data)
            if (model := config.model) is not None:
                if isinstance(model, _UseServerModel):
                    thread.reset_config(self._cfg)
                else:
                    thread.update_config(
                        model=model.name, api_key=model.api_key, base_url=model.base_url, provider=model.provider
                    )

            return web.Response(status=200, text=thread.get_config().model_dump_json())

        except pydantic.ValidationError as e:
            return web.Response(status=400, text=f"Invalid data: {e}")

        except Exception as e:
            return web.Response(status=500, text=f"Error updating configuration: {e}")

        finally:
            thread_lock.release()

    async def del_thread(self, request: web.Request) -> web.Response:
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)

        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")

        thread, _ = thread_tuple
        try:
            await thread.close()
            with self._threads_lock:
                self._threads.pop(thread_id, None)
            return web.Response(status=200, text=f"thread {thread_id} deleted")

        except Exception as e:
            return web.Response(status=500, text=f"Error deleting thread: {e}")

    async def abort_endpoint(self, request: web.Request) -> web.Response:
        """Abort any in-flight agent run for a thread.

        Used by the frontend stop button so a retry backoff is interrupted
        promptly (the fetch abort alone is only detected on the next NDJSON
        write). Does NOT acquire the thread lock — that lock is held by the
        in-flight post we are trying to abort.
        """
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)
        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")
        thread, _ = thread_tuple
        try:
            thread.abort()
            return web.Response(status=204)
        except Exception as e:
            return web.Response(status=500, text=f"Error aborting thread: {e}")

    async def close(self) -> None:
        """Close all threads and clean up resources."""
        with self._threads_lock:
            threads = list(self._threads.values())
            self._threads.clear()

        await asyncio.gather(*(thread.close() for thread, _ in threads), return_exceptions=True)

    async def optimize_prompt_endpoint(self, request: web.Request) -> web.Response:
        """API endpoint to optimize user prompt."""
        thread_id = request.match_info["thread_id"]
        with self._threads_lock:
            thread_tuple = self._threads.get(thread_id)
        if thread_tuple is None:
            return web.Response(status=404, text=f"thread {thread_id} not found")
        thread, thread_lock = thread_tuple

        if not thread_lock.acquire(timeout=THREAD_LOCK_TIMEOUT):
            return web.Response(status=408, text="Request timeout")

        try:
            data = await request.json()
            prompt_data = _OptimizePrompt.model_validate(data)

            # Delegate to LeaderAgent for optimization
            optimized = await thread._agent.optimize_prompt(prompt_data.prompt)

            return web.json_response({"original": prompt_data.prompt, "optimized": optimized})
        except pydantic.ValidationError as e:
            return web.Response(status=400, text=f"Invalid data: {e}")
        except Exception as e:
            return web.Response(status=500, text=f"Error optimizing prompt: {e}")
        finally:
            thread_lock.release()

    async def optimize_prompt_standalone(self, request: web.Request) -> web.Response:
        """Optimize prompt without requiring an existing thread.

        This endpoint is used for optimizing prompts before a thread is created.
        It uses ImageJ window info for context but doesn't use chat history.
        """
        try:
            data = await request.json()
            prompt_data = _OptimizePrompt.model_validate(data)

            # Create a temporary agent instance for optimization
            # Use default model from settings

            plugin_apis: PluginAPI = BridgePluginAPI(self._bridge)
            apis = plugin_apis.attach_single_client()  # TODO: should from frontend
            temp_agent = LeaderDriven(apis=apis, cfg=self._cfg)
            # Optimize the prompt (without chat history context)
            optimized = await temp_agent.optimize_prompt(prompt_data.prompt)

            # Clean up temporary agent
            temp_agent.abort()
            await plugin_apis.close()

            return web.json_response({"original": prompt_data.prompt, "optimized": optimized})
        except pydantic.ValidationError as e:
            return web.Response(status=400, text=f"Invalid data: {e}")
        except Exception as e:
            return web.Response(status=500, text=f"Error optimizing prompt: {e}")

    async def _new_response(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "application/x-ndjson",  # Newline Delimited JSON
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(request)
        return response

    async def _on_post(self, response: web.StreamResponse, thread: _Thread, prompt: str | bool) -> web.StreamResponse:
        disconnected = False
        try:
            async for message_data in thread.on_post(prompt):
                if not response.prepared:
                    # This can happen if the client disconnects very early
                    _log.warning("Response not prepared, client might have disconnected.")
                    disconnected = True
                    break

                try:
                    await _send_ndjson(response, message_data)

                except (ConnectionResetError, BrokenPipeError):
                    _log.info("Client connection reset during stream.")
                    disconnected = True
                    break  # Stop streaming if client disconnects

            # Finalize the stream if not disconnected
            with suppress(ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
                await response.write_eof()

        except asyncio.CancelledError:
            _log.info("Client disconnected from agent chat stream.")
            disconnected = True

        except Exception:
            _log.exception("Error during agent chat stream")

        finally:
            if disconnected:
                await asyncio.shield(thread.close())

        return response


async def _send_ndjson(response: web.StreamResponse, data: pydantic.BaseModel) -> None:
    payload = data.model_dump_json()
    await response.write(payload.encode("utf-8"))
    await response.write(b"\n")
    await response.drain()  # Ensure data is sent
