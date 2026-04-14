# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, override

import langfuse
import pydantic

from copilotj.core.message import HandoffMessage, ImageMessage, TextMessage
from copilotj.core.model_client import (
    FinishReasons,
    ModelClient,
    ModelProviderError,
    ModelResponse,
    ModelResponseChunk,
    ModelSyntaxError,
    ToolCall,
)
from copilotj.core.runtime import Runtime
from copilotj.core.tool import FunctionTool, Tool
from copilotj.core.ui import Handoff

__all__ = ["Agent", "ChatAgent", "message_handler", "HandoffFunctionTool"]


class Agent:
    """Base class for all agents."""

    _runtime_: Runtime | None = None

    def __init__(self, name: str, description: str):
        super().__init__()
        self.name = name
        self.description = description
        self._message_handlers: dict[type[Any], MessageHandler] = {}
        self._register_message_handlers()

    def __str__(self):
        return f"{self.name}: {self.description}"

    # Runtime

    def _set_runtime(self, runtime: Runtime) -> None:
        assert self._runtime_ is None or self._runtime_ is not runtime, (
            "Runtime already set, try to put the agent in a pattern"
        )
        self._runtime_ = runtime

    @property
    def _runtime(self) -> Runtime:
        assert self._runtime_ is not None, "Runtime not set"
        return self._runtime_

    async def request_user_confirm(self, message: str) -> bool:
        return await self._runtime.request_user_confirm(self.name, message)

    async def request_user_manipulate(self, message: str | None = None) -> str | None:
        return await self._runtime.request_user_manipulate(self.name, message)

    async def print_info(self, message: str) -> None:
        await self._runtime.print_info(self.name, message)

    async def print_error(self, message: str) -> None:
        await self._runtime.print_error(self.name, message)

    def log_info(self, message: str) -> None:
        self._runtime.log_info(message)

    def log_error(self, message: str) -> None:
        self._runtime.log_error(message)

    # Messages

    async def on_message(self, message: Any) -> HandoffMessage:
        message_type = type(message)
        if message_type not in self._message_handlers:
            raise ValueError(f"No handler registered for message type: {message_type}")

        return await self._message_handlers[message_type](message)

    def _register_message_handlers(self):
        """Register message handlers decorated with @message_handler.

        Scans the class for methods decorated with @message_handler and registers them in the _message_handlers
        dictionary.
        """
        for name in dir(self):
            if name.startswith("_"):
                continue

            value = getattr(self, name)
            if hasattr(value, "__message_handler__"):
                message_type, func = value.__message_handler__
                self._message_handlers[message_type] = func


class ChatAgent(Agent):
    def __init__(self, name: str, description: str, *, model_client: ModelClient):
        super().__init__(name, description)
        self._client = model_client
        self._abort_event = asyncio.Event()

    async def _create(
        self,
        *messages: TextMessage | ImageMessage,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
        trace_ctx: langfuse.Langfuse | None = None,
    ) -> ModelResponse:
        trace_ctx = trace_ctx or langfuse.get_client()
        with trace_ctx.start_as_current_observation(
            as_type="generation",
            name="agent_create",
            metadata={
                "agent": self.name,
                "agent_description": self.description,
                "agent_tools": {t.name: t.description for t in tools} if tools else None,
            },
        ):
            # Reset abort event before starting new stream
            self._abort_event.clear()
            await self._runtime.update_current_agent(self.name)
            stream = self._client.create_stream(messages, tools=tools, extra_args=extra_args)

            content = ""
            reasoning_content = ""
            tool_calls: list[ToolCall] = []
            finish_reason: FinishReasons = "unknown"

            abort_waiter = asyncio.create_task(self._abort_event.wait())  # create a task to wait for abort signal
            try:
                while True:
                    next_chunk = asyncio.create_task(stream.__anext__())
                    done, pending = await asyncio.wait(
                        {next_chunk, abort_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Check for abort signal before processing each chunk
                    if abort_waiter in done:
                        for t in pending:
                            t.cancel()

                        # Close the underlying stream to ensure connections are returned
                        if hasattr(stream, "aclose"):
                            try:
                                await asyncio.shield(stream.aclose())
                            except Exception:
                                pass
                        raise asyncio.CancelledError("Stream was aborted by user request")

                    # Process the next chunk
                    assert next_chunk in done
                    try:
                        chunk = next_chunk.result()
                    except StopAsyncIteration:
                        break  # End of stream

                    if isinstance(chunk, ToolCall):
                        # Dont send handoff tool calls, they are handled separately
                        if not isinstance(chunk.tool, HandoffTool):
                            await self._runtime.print_chat(self.name, chunk)

                        tool_calls.append(chunk)

                    else:
                        await self._runtime.print_chat(self.name, chunk)
                        if chunk.content is not None:
                            content += chunk.content

                        if chunk.reasoning_content is not None:
                            reasoning_content += chunk.reasoning_content

                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason

            except ModelSyntaxError as e:
                # send a finish message to the runtime
                await self._runtime.print_chat(
                    self.name, ModelResponseChunk(reasoning_content="", content="", finish_reason="stop")
                )
                e.chat_completion = ModelResponse(
                    content=content if len(content) else None,
                    reasoning_content=reasoning_content if len(reasoning_content) else None,
                    tool_calls=tool_calls if len(tool_calls) else None,
                    finish_reason=finish_reason,
                )
                raise e

            except ModelProviderError as e:
                await self._runtime.print_error("system", f"LlmProviderError: {e}")

            except Exception as e:
                # Log it — otherwise unexpected stream errors silently yield an empty response.
                self.log_error(f"Unexpected error during model stream: {type(e).__name__}: {e}")
                await self._runtime.print_chat(
                    self.name, ModelResponseChunk(reasoning_content="", content="", finish_reason="unknown")
                )

        completion = ModelResponse(
            content=content if len(content) else None,
            reasoning_content=reasoning_content if len(reasoning_content) else None,
            tool_calls=tool_calls if len(tool_calls) else None,
            finish_reason=finish_reason,
        )
        self._runtime.log_info(str(completion))
        return completion

    async def _call_tool(self, tool_call: ToolCall) -> pydantic.BaseModel:
        if isinstance(tool_call.tool, HandoffTool):
            handoff = tool_call.tool.get_handoff(tool_call.id, tool_call.args)
            await self._runtime.print_handoff(self.name, handoff)
        else:
            await self._runtime.print_tool_called(self.name, tool_call.id)

        try:
            result = await tool_call.run()
            await self._runtime.print_tool_call_result(self.name, tool_call.id, "success", str(result))
        except Exception as e:
            await self._runtime.print_tool_call_result(self.name, tool_call.id, "error", str(e))
            raise e

        return result

    def set_model_client(self, model_client: ModelClient) -> None:
        self._client = model_client

    def abort(self) -> None:
        """Abort the current stream operation.

        Sets the abort event to signal the streaming loop to stop.
        This method can be called from any thread to abort an ongoing stream.
        """
        # TODO: how to abort tool calls?
        self.log_info("Abort signal sent to ChatAgent")
        self._abort_event.set()


type MessageHandler = Callable[[Any], Awaitable[HandoffMessage]]


def message_handler(func: Callable[[Any], Awaitable[HandoffMessage]]) -> MessageHandler:
    """Decorator to register a method as a handler for a specific message type.

    Args:
        func: The message handler function to decorate

    Returns:
        The decorated message handler

    Raises:
        TypeError:
            If the function name starts with '_'
            If the function does not have exactly one parameter named 'message'
            If the 'message' parameter does not have a type annotation
            If the 'message' parameter has a type annotation of Any or None
    """
    # Check function name is not start with '_'
    if func.__name__.startswith("_"):
        raise TypeError("Message handler function name must not start with '_'")

    # Get the function signature
    sig = inspect.signature(func)
    parameters = list(sig.parameters.values())

    # Verify exactly one parameter named 'message'
    if len(parameters) != 1:
        raise TypeError("Message handler must have exactly one parameter named 'message'")

    # Get type annotation and verify it's specific
    param = parameters[0]
    if param.annotation is inspect.Parameter.empty:
        raise TypeError("Message handler parameter 'message' must have a type annotation")
    elif param.annotation in (Any, None, type(None)):
        raise TypeError("Message handler parameter 'message' must have a specific type annotation, not Any/None")

    @functools.wraps(func)
    def wrapper(message: Any) -> Any:
        return func(message)

    wrapper.__message_handler__ = (param.annotation, func)  # type: ignore
    return wrapper


class HandoffTool(abc.ABC):
    """Print a handoff event to the UI when called."""

    @abc.abstractmethod
    def get_handoff(self, id: str, args: pydantic.BaseModel) -> Handoff: ...


class HandoffFunctionTool(FunctionTool, HandoffTool):
    def __init__(
        self,
        func: Callable[..., Any],
        description: str,
        name: str | None = None,
        *,
        strict: bool = False,
        get_handoff: Callable[[str, pydantic.BaseModel], Handoff],
    ) -> None:
        super().__init__(func, description, name, strict=strict)
        self._get_handoff = get_handoff

    @override
    def get_handoff(self, id: str, args: pydantic.BaseModel) -> Handoff:
        return self._get_handoff(id, args)
