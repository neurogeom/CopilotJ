# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import threading
import uuid
from asyncio import Future
from contextlib import suppress
from typing import AsyncGenerator, Literal, override

import aiohttp.web as web
import langfuse
import pydantic

from copilotj.core import UI, UIEvent, UIEventPost, UIEventState
from copilotj.core.config import get_llm_and_key
from copilotj.core.ui import UIEventContentMarkdown
from copilotj.multiagent.leader_multiagent import LeaderDriven
from copilotj.plugin.api import HTTPPluginAPI, PluginAPI

__all__ = ["Threads"]

# Timeout for thread locks in seconds (300ms)
THREAD_LOCK_TIMEOUT = 0.3

_log = logging.getLogger(__name__)


dumpable = str | int | float | bool | pydantic.BaseModel

ROLE_SYSTEM = "system"


class _ConfigModel(pydantic.BaseModel):
    name: str
    api_key: str | None
    base_url: str | None = None


class _Config(pydantic.BaseModel):
    model: _ConfigModel


class _ConfigQuery(pydantic.BaseModel):
    model: _ConfigModel | None = None


class _NewThread(pydantic.BaseModel):
    id: str
    config: _Config


class _Signal(pydantic.BaseModel):
    signal: Literal["end"]


class _OptimizePrompt(pydantic.BaseModel):
    prompt: str


class _Thread(UI):
    def __init__(self, thread_id: str, *, config: _ConfigQuery | None = None, trace_context: langfuse.Langfuse):
        self.thread_id = thread_id
        self._trace_ctx = trace_context

        self._mailbox = asyncio.Queue[UIEvent | _Signal]()

        self._apis: PluginAPI = HTTPPluginAPI("http://127.0.0.1:8786")  # TODO: make configurable
        client_apis = self._apis.attach_dev_client()  # TODO: should be removed

        config_model = config and config.model
        self._agent = LeaderDriven(
            apis=client_apis,
            ui=self,
            model=config_model.name if config_model else None,
            api_key=config_model.api_key if config_model else None,
            base_url=config_model.base_url if config_model else None,
        )
        self._post_task: asyncio.Task[None] | None = None
        self._post_done: asyncio.Event | None = None
        self._task_future: Future[str | None] | None = None
        self._confirmation_future: Future[bool] | None = None

        self._config = (
            _Config(model=config_model or _ConfigModel(name=self._agent.model_client.get_model(), api_key=None))
            if config
            else _Config(model=_ConfigModel(name=self._agent.model_client.get_model(), api_key=None))
        )

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

    def update_config(self, *, model: str | None, api_key: str | None, base_url: str | None = None) -> None:
        self._agent.update_config(model=model, api_key=api_key, base_url=base_url)
        self._config = _Config(
            model=_ConfigModel(
                name=self._agent.model_client.get_model(),
                api_key=self._agent.model_client.get_api_key(),
                base_url=base_url,
            )
        )

    async def _run_agent(self, prompt: str, done_event: asyncio.Event) -> None:
        """Run the chat with the agent."""
        try:
            with self._trace_ctx.start_as_current_observation(
                as_type="span", name="thread", metadata={"thread_id": self.thread_id}, input=prompt
            ):
                await self._agent.run(prompt, trace_ctx=self._trace_ctx)

        finally:
            done_event.set()  # Signal that the chat is done

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
    def __init__(self):
        super().__init__()
        self._threads: dict[str, tuple[_Thread, threading.Lock]] = {}
        self._threads_lock = threading.Lock()
        self._trace_ctx = langfuse.Langfuse()

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

        # If no model was explicitly provided, check that the server has one configured
        config_model = config.model if config else None
        if config_model is None:
            resolved_model, _ = get_llm_and_key()
            if not resolved_model:
                return web.Response(
                    status=400,
                    text="No model configured. Please click the Settings gear icon in the sidebar to set up a model and API key.",
                )

        thread_id = str(uuid.uuid4())
        thread = _Thread(thread_id, config=config, trace_context=self._trace_ctx)
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

            config = _Config.model_validate(data)
            if (model := config.model) is not None:
                thread.update_config(model=model.name, api_key=model.api_key, base_url=model.base_url)

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

            # Import here to avoid circular dependency
            from copilotj.multiagent.leader_multiagent import LeaderDriven
            from copilotj.plugin.api import HTTPPluginAPI

            # Create a temporary agent instance for optimization
            # Use default model from settings
            apis = HTTPPluginAPI("http://127.0.0.1:8786")
            client = apis.attach_dev_client()  # TODO: should be removed or made configurable
            temp_agent = LeaderDriven(apis=client)
            # Optimize the prompt (without chat history context)
            optimized = await temp_agent.optimize_prompt(prompt_data.prompt)

            # Clean up temporary agent
            temp_agent.abort()
            await apis.close()

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
