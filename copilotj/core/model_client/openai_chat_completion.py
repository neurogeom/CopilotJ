# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, AsyncGenerator, Literal, Sequence, overload, override

import langfuse.openai
import openai.types.chat

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client._retry import parse_retry_after
from copilotj.core.model_client._types import (
    FinishReasons,
    ModelClient,
    ModelProviderError,
    ModelResponse,
    ModelResponseChunk,
    ToolCall,
)
from copilotj.core.tool import Tool

logger = logging.getLogger(__name__)

__all__ = ["OpenAIChatCompletionClient"]


def _to_openai_provider_error(e: Exception, provider: str = "openai") -> ModelProviderError:
    """Convert an OpenAI SDK exception into a ModelProviderError.

    Preserves the HTTP status code and ``Retry-After`` header so the agent's
    retry loop can decide retryability and backoff. Shared by the OpenAI-family
    clients (chat completions, Responses API, and Ollama's OpenAI-compatible
    API).
    """
    status_code = getattr(e, "status_code", None)
    retry_after = None
    response = getattr(e, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = parse_retry_after(headers.get("retry-after"))
    message = getattr(e, "message", None) or str(e)
    return ModelProviderError(message, provider, status_code=status_code, retry_after=retry_after)


class OpenAIChatCompletionClient(ModelClient):
    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__()
        self._model = model
        self._api_key = api_key
        http_client = openai.DefaultAsyncHttpxClient(proxy=proxy) if proxy is not None else None

        # Langfuse support can be safely ignored if LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set
        # max_retries=0: own the retry loop in ChatAgent so 429 retries are
        # VISIBLE (the SDK retries silently and exposes no per-attempt hook).
        self._client = langfuse.openai.AsyncOpenAI(
            api_key=self._api_key, http_client=http_client, base_url=base_url, max_retries=0
        )

    @override
    def get_model(self) -> str:
        return self._model

    @override
    def get_api_key(self) -> str | None:
        return self._api_key

    @override
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse:
        try:
            completion = await self._create(messages, stream=False, tools=tools, extra_args=extra_args)
            if completion.usage is not None:
                _log_cache_usage(self._model, completion.usage)
            choice = completion.choices[0]
            tool_calls = []
            if choice.message.tool_calls:
                for tool_call in choice.message.tool_calls:
                    function = tool_call.function
                    tool = next((t for t in (tools or []) if t.name == function.name), None)
                    if tool is None:
                        raise ValueError(f"Tool '{function.name}' not found in tools")

                    args = tool.args_type().model_validate_json(function.arguments)
                    tool_calls.append(ToolCall(id=tool_call.id, tool=tool, args=args))

            return ModelResponse(
                content=choice.message.content,
                reasoning_content=None,
                tool_calls=tool_calls if len(tool_calls) > 0 else None,
                finish_reason=_openai_parse_finish_reason(choice.finish_reason),
            )
        except openai.APIError as e:
            raise _to_openai_provider_error(e) from e
        except openai.OpenAIError as e:
            raise _to_openai_provider_error(e) from e

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        stream = None
        try:
            stream = await self._create(messages, stream=True, tools=tools, extra_args=extra_args)
            tool_calls: dict[int, openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall] = {}

            stream_usage = None
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    stream_usage = usage

                if chunk.choices is None or len(chunk.choices) == 0:
                    # skip this chunk. gemini sometime send a None when he does not want to say anything :)
                    continue

                choice = chunk.choices[0]
                if choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        index = tool_call.index
                        if index not in tool_calls:
                            tool_calls[index] = tool_call
                            continue

                        if (func := tool_call.function) is not None and func.arguments is not None:
                            saved = tool_calls[index]
                            if saved.function is None:
                                saved.function = func
                            elif saved.function.arguments is None:
                                saved.function.arguments = func.arguments
                            else:
                                saved.function.arguments += func.arguments

                    if choice.finish_reason is None and choice.delta.content is None:
                        continue  # skip this chunk

                if choice.finish_reason is not None:
                    for tool_call in tool_calls.values():
                        f = tool_call.function
                        if f is None or f.name is None:
                            continue  #  Tool call function is missing name or arguments

                        tool = next((t for t in (tools or []) if f.name is not None and t.name == f.name), None)
                        if tool is None:
                            raise ValueError(f"Tool '{f.name}' not found in tools")

                        args = tool.args_type().model_validate_json(f.arguments or "{}")
                        tool_call.id = tool_call.id or f"tool_call_{len(tool_calls)}"
                        yield ToolCall(id=tool_call.id, tool=tool, args=args)

                yield ModelResponseChunk(
                    content=choice.delta.content,
                    reasoning_content=None,
                    finish_reason=_openai_parse_finish_reason(choice.finish_reason),
                )

            if stream_usage is not None:
                _log_cache_usage(self._model, stream_usage)

        except openai.APIError as e:
            raise _to_openai_provider_error(e) from e

        except openai.OpenAIError as e:
            raise _to_openai_provider_error(e) from e

        finally:
            if stream is not None:
                try:
                    await asyncio.shield(stream.aclose())  # type: ignore
                except Exception:
                    pass

    @overload
    async def _create(
        self, messages, *, tools, extra_args, stream: Literal[False]
    ) -> openai.types.chat.ChatCompletion: ...
    @overload
    async def _create(
        self, messages, *, tools, extra_args, stream: Literal[True]
    ) -> openai.AsyncStream[openai.types.chat.ChatCompletionChunk]: ...
    async def _create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None,
        extra_args: dict[str, Any] | None,
        stream: bool,
    ):
        try:
            openai_messages = self._format_messages(messages)
            openai_tools = openai.NOT_GIVEN
            if tools is not None:
                openai_tools = [
                    openai.types.chat.ChatCompletionToolParam(
                        type="function",
                        function=tool.json_schema,  # type: ignore
                    )
                    for tool in tools
                ]

            extra = dict(extra_args or {})
            if stream:
                # Ask OpenAI to include a final usage chunk so we can observe
                # whether prompt caching is actually discounting token spend.
                stream_options = dict(extra.get("stream_options") or {})
                stream_options.setdefault("include_usage", True)
                extra["stream_options"] = stream_options

            return await self._client.chat.completions.create(
                model=self._model, messages=openai_messages, tools=openai_tools, **extra, stream=stream
            )
        except openai.APIError as e:
            raise _to_openai_provider_error(e) from e
        except openai.OpenAIError as e:
            raise _to_openai_provider_error(e) from e

    @classmethod
    def _format_messages(cls, messages: Sequence[TextMessage | ImageMessage]):
        openai_messages: list[openai.types.chat.ChatCompletionMessageParam] = []
        group: list[TextMessage | ImageMessage] = []
        for message in messages:
            if len(group) > 0 and group[0].role != message.role:
                openai_messages.append(cls._merge_messages(group))
                group.clear()

            group.append(message)

        if len(group) > 0:
            openai_messages.append(cls._merge_messages(group))

        return openai_messages

    @staticmethod
    def _merge_messages(
        messages: Sequence[TextMessage | ImageMessage],
    ) -> openai.types.chat.ChatCompletionMessageParam:
        """Format a sequence of messages into OpenAI's chat completion format."""
        content = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                content.append({"type": "text", "text": msg.text})

            elif isinstance(msg, ImageMessage):
                content.append({"type": "image_url", "image_url": {"url": msg.image}})

            else:
                raise ValueError(f"Unsupported message type: {msg}")

        return {"role": _openai_convert_role(messages[0].role), "content": content}  # type: ignore


def _log_cache_usage(model: str, usage: Any) -> None:
    """Log prompt token usage with cache hit info.

    Works for both Chat Completions usage (``prompt_tokens`` +
    ``prompt_tokens_details.cached_tokens``) and Responses API usage
    (``input_tokens`` + ``input_tokens_details.cached_tokens``). Used to
    verify OpenAI's automatic prompt-prefix caching is actually hitting.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(usage, "input_tokens", None)

    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        details = getattr(usage, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details is not None else None

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(usage, "output_tokens", None)

    logger.info(
        "[CACHE] model=%s prompt=%s cached=%s completion=%s",
        model,
        prompt_tokens,
        cached if cached is not None else "n/a",
        completion_tokens,
    )


def _openai_convert_role(role: str) -> Literal["user", "system", "assistant"]:
    """Convert 'system' role to 'developer' role."""
    match role:
        case "system":
            # NOTE: Open AI is moving away from using 'system' role in favor of
            # 'developer' role. See [Model Spec](https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions)
            # for more details.
            #
            # However, the OpenAI API still compatibly uses 'system' role for
            # now, and some LLM providers (e.g., GLM) still expect 'system'
            # role. So we keep using 'system' role here.
            return "system"
        case "assistant":
            return "assistant"
        case _:
            return "user"


def _openai_parse_finish_reason(finish_reason: str | None) -> FinishReasons | None:
    match finish_reason:
        case "stop":
            return "stop"
        case "tool_calls":
            return "tool_calls"
        case None:
            return None
        case _:
            return "unknown"
