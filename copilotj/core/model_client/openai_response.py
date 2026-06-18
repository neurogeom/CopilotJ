# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, AsyncGenerator, Literal, Sequence, cast, overload, override

import langfuse.openai
import openai.types.responses

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client._types import (
    FinishReasons,
    ModelClient,
    ModelResponse,
    ModelResponseChunk,
    ToolCall,
)
from copilotj.core.model_client.openai_chat_completion import _to_openai_provider_error
from copilotj.core.tool import Tool

logger = logging.getLogger(__name__)

__all__ = ["OpenAIResponseClient"]


class OpenAIResponseClient(ModelClient):
    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__()
        self._model = model
        self._api_key = api_key
        http_client = openai.DefaultAsyncHttpxClient(proxy=proxy) if proxy is not None else None
        # max_retries=0: own the retry loop in ChatAgent so 429 retries are VISIBLE.
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
            response = await self._create(messages, stream=False, tools=tools, extra_args=extra_args)
            if response.usage is not None:
                _log_cache_usage(self._model, response.usage)
            content = None
            tool_calls = []
            for item in response.output:
                match item.type:
                    case "message":
                        if content is None:
                            content = ""

                        for i in item.content:
                            match i.type:
                                case "output_text":
                                    content += i.text
                                case "refusal":
                                    raise ValueError(f"Model refused to answer: {i.refusal}")
                                case _:
                                    raise ValueError(f"Unsupported message content type: {i}")

                    case "function_call":
                        tool = next((t for t in (tools or []) if t.name == item.name), None)
                        if tool is None:
                            raise ValueError(f"Tool '{item.name}' not found in tools")

                        args = tool.args_type().model_validate_json(item.arguments)
                        tool_calls.append(ToolCall(id=item.id or "unknown_id", tool=tool, args=args))

                    case "reasoning":
                        # gpt-5 / o-series reasoning trace — no actionable content to extract here.
                        pass

                    case _:
                        logger.warning("Unsupported Responses API output item, ignoring: %s", item.type)

            return ModelResponse(
                reasoning_content=None,
                content=response.output_text,
                tool_calls=tool_calls if len(tool_calls) > 0 else None,
                finish_reason="tool_calls" if len(tool_calls) > 0 else "stop",
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
        try:
            stream = await self._create(messages, stream=True, tools=tools, extra_args=extra_args)
            current_function_call: tuple[str, str] | None = None
            last_chunk = None
            async for chunk in stream:
                # print("\n", chunk.type, chunk, "\n\n")
                match chunk.type:
                    case "response.completed":
                        usage = getattr(chunk.response, "usage", None)
                        if usage is not None:
                            _log_cache_usage(self._model, usage)

                    case (
                        # response life cycle event
                        "response.created"  # first of all
                        | "response.in_progress"  # after created
                        # output item
                        | "response.output_item.done"
                        # output item - content
                        | "response.content_part.added"
                        | "response.output_text.done"  # after response.content_part.added
                        | "response.content_part.done"
                        # output item - function call
                        | "response.function_call_arguments.delta"
                        # output item - reasoning (gpt-5, o-series)
                        | "response.reasoning_summary_part.added"
                        | "response.reasoning_summary_part.done"
                        | "response.reasoning_summary_text.done"
                        | "response.reasoning_text.done"
                    ):
                        pass

                    case "response.reasoning_summary_text.delta" | "response.reasoning_text.delta":
                        # Surface reasoning trace as reasoning_content so the UI can show it.
                        last_chunk = ModelResponseChunk(content=None, reasoning_content=chunk.delta, finish_reason=None)
                        yield last_chunk

                    case "response.output_item.added":
                        match chunk.item.type:
                            case "message" | "reasoning":
                                pass

                            case "function_call":
                                item = cast(openai.types.responses.ResponseFunctionToolCall, chunk.item)
                                if current_function_call is not None and current_function_call[0] != item.id:
                                    raise NotImplementedError("Nested function calls are not supported")

                                current_function_call = (item.id or "unknown_id", item.name)

                            case _:
                                logger.warning("Unsupported output item type, ignoring: %s", chunk.item.type)

                    case "response.output_text.delta":
                        last_chunk = ModelResponseChunk(content=chunk.delta, reasoning_content=None, finish_reason=None)
                        yield last_chunk

                    case "response.function_call_arguments.done":
                        if current_function_call is None:
                            raise ValueError("Function call done without a current function call")

                        id, name = current_function_call
                        current_function_call = None
                        tool = next((t for t in (tools or []) if t.name == name), None)
                        if tool is None:
                            raise ValueError(f"Tool '{name}' not found in tools")

                        args = tool.args_type().model_validate_json(chunk.arguments)
                        last_chunk = ToolCall(id=id, tool=tool, args=args)
                        yield last_chunk

                    case "error":
                        raise ValueError(f"Error from model: {chunk.message}")

                    case _:
                        # Fail open on unfamiliar events so new model features don't silently blank the response.
                        logger.warning("Unsupported Responses API chunk type, ignoring: %s", chunk.type)

                finish_reason: FinishReasons = "unknown"
                match last_chunk:
                    case ModelResponseChunk():
                        finish_reason = "stop"

                    case ToolCall():
                        finish_reason = "tool_calls"

                    case _:
                        pass

                yield ModelResponseChunk(content=None, reasoning_content=None, finish_reason=finish_reason)
        except openai.APIError as e:
            raise _to_openai_provider_error(e) from e
        except openai.OpenAIError as e:
            raise _to_openai_provider_error(e) from e

    @overload
    async def _create(
        self, messages, *, tools, extra_args, stream: Literal[False]
    ) -> openai.types.responses.Response: ...
    @overload
    async def _create(
        self, messages, *, tools, extra_args, stream: Literal[True]
    ) -> openai.AsyncStream[openai.types.responses.ResponseStreamEvent]: ...
    async def _create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None,
        extra_args: dict[str, Any] | None,
        stream: bool,
    ):
        try:
            inputs: list[openai.types.responses.ResponseInputItemParam] = []
            group: list[TextMessage | ImageMessage] = []
            for message in messages:
                if len(group) > 0 and group[0].role != message.role:
                    inputs.append(self._merge_messages(group))
                    group.clear()

                group.append(message)

            if len(group) > 0:
                inputs.append(self._merge_messages(group))

            openai_tools = openai.NOT_GIVEN
            if tools is not None:
                openai_tools = []
                for tool in tools:
                    schema = tool.json_schema
                    tool_parama = openai.types.responses.FunctionToolParam(
                        type="function",
                        name=schema.get("name"),
                        description=schema.get("description", None),
                        parameters=schema.get("parameters", None),  # type: ignore
                        strict=schema.get("strict", True),
                    )
                    openai_tools.append(tool_parama)

            return await self._client.responses.create(
                model=self._model,  # type: ignore
                input=inputs,
                tools=openai_tools,
                **(extra_args or {}),
                stream=stream,
            )
        except openai.APIError as e:
            raise _to_openai_provider_error(e) from e
        except openai.OpenAIError as e:
            raise _to_openai_provider_error(e) from e

    @staticmethod
    def _merge_messages(
        messages: Sequence[TextMessage | ImageMessage],
    ) -> openai.types.responses.ResponseInputItemParam:
        """Format a sequence of messages into the Responses API input format."""
        role = _openai_convert_role(messages[0].role)

        # Assistant history turns must use the EasyInputMessage form with a
        # plain-string (or output_text) payload; the structured input_text /
        # input_image types are only valid for user/system/developer input.
        if role == "assistant":
            text_parts: list[str] = []
            for msg in messages:
                if isinstance(msg, TextMessage):
                    text_parts.append(msg.text)
                else:
                    raise ValueError(f"Assistant messages must be text-only, got: {msg}")
            return openai.types.responses.EasyInputMessageParam(
                role="assistant",
                content="\n".join(text_parts),
                type="message",
            )

        content: openai.types.responses.ResponseInputMessageContentListParam = []
        for msg in messages:
            match msg:
                case TextMessage():
                    content.append(openai.types.responses.ResponseInputTextParam(type="input_text", text=msg.text))

                case ImageMessage():
                    content.append(
                        openai.types.responses.ResponseInputImageParam(
                            type="input_image", detail="auto", image_url=msg.image
                        )
                    )

                case _:
                    raise ValueError(f"Unsupported message type: {msg}")

        return {"role": role, "content": content}


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
