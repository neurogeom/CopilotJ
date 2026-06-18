# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Any, AsyncGenerator, Sequence, override

import anthropic
import httpx

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

__all__ = ["AnthropicChatCompletionClient"]

logger = logging.getLogger(__name__)

_DATA_URL_RE = re.compile(r"^data:(image/[^;]+);base64,(.+)$", re.DOTALL)

# Ephemeral prompt-cache breakpoint (5-minute TTL). Tagged onto the last system
# block (caches tools+system) and the last message content block (caches the
# growing conversation prefix) in _format_messages. Two breakpoints, within the
# Anthropic 4-breakpoint-per-request limit.
_CACHE_CONTROL = {"type": "ephemeral"}


def _to_anthropic_provider_error(e: Exception) -> ModelProviderError:
    """Convert an Anthropic SDK exception into a ModelProviderError.

    Preserves the status code and ``Retry-After`` header so the agent retry loop
    can detect 429s and back off correctly.
    """
    status_code = getattr(e, "status_code", None)
    retry_after = None
    response = getattr(e, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = parse_retry_after(headers.get("retry-after"))
    return ModelProviderError(str(e), "anthropic", status_code=status_code, retry_after=retry_after)


class AnthropicChatCompletionClient(ModelClient):
    """Anthropic/Claude client using the native ``anthropic`` SDK.

    Uses the Messages API with streaming support, tool use, vision,
    and prompt caching.
    """

    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__()
        self._model = model
        self._api_key = api_key
        http_client = httpx.AsyncClient(proxy=proxy) if proxy else None
        # max_retries=0: own the retry loop in ChatAgent so 429 retries are VISIBLE
        # (the SDK retries silently and exposes no per-attempt hook).
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, http_client=http_client, max_retries=0
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
        extra = dict(extra_args or {})
        max_tokens = extra.pop("max_tokens", 8192)

        system, api_messages = self._format_messages(messages)
        api_tools = self._format_tools(tools)

        try:
            response = await self._client.messages.create(
                model=self._model, max_tokens=max_tokens, system=system, messages=api_messages, tools=api_tools, **extra
            )
            _log_cache_usage(self._model, response.usage)
            return _parse_response(response, tools)
        except anthropic.APIError as e:
            raise _to_anthropic_provider_error(e) from e

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        extra = dict(extra_args or {})
        max_tokens = extra.pop("max_tokens", 8192)

        system, api_messages = self._format_messages(messages)
        api_tools = self._format_tools(tools)

        try:
            async with self._client.messages.stream(
                model=self._model, max_tokens=max_tokens, system=system, messages=api_messages, tools=api_tools, **extra
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield ModelResponseChunk(content=event.text, reasoning_content=None, finish_reason=None)

                    elif event.type == "thinking":
                        yield ModelResponseChunk(content=None, reasoning_content=event.thinking, finish_reason=None)

                    elif event.type == "content_block_stop":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool = next((t for t in (tools or []) if t.name == block.name), None)
                            if tool is None:
                                raise ValueError(f"Tool '{block.name}' not found in tools")
                            args = tool.args_type().model_validate(block.input)
                            yield ToolCall(id=block.id, tool=tool, args=args)

                    elif event.type == "message_stop":
                        msg = await stream.get_final_message()
                        finish = _parse_stop_reason(msg.stop_reason)
                        yield ModelResponseChunk(content=None, reasoning_content=None, finish_reason=finish)

                # Log cache usage from the final accumulated message.
                final = await stream.get_final_message()
                _log_cache_usage(self._model, final.usage)

        except anthropic.APIError as e:
            raise _to_anthropic_provider_error(e) from e

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @classmethod
    def _format_messages(
        cls,
        messages: Sequence[TextMessage | ImageMessage],
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Returns ``(system, anthropic_messages)`` where *system* is either
        a list of ``TextBlockParam`` dicts or ``anthropic.NOT_GIVEN``.

        Leading ``system`` messages are extracted into the top-level ``system``
        parameter.  Any ``system`` message that appears *after* a non-system
        message is mapped to the ``assistant`` role (with a warning) so that
        message order is preserved.
        """
        system_parts: list[dict[str, Any]] = []
        anthropic_messages: list[dict[str, Any]] = []
        seen_non_system = False

        for msg in messages:
            if msg.role == "system" and not seen_non_system:
                system_parts.append({"type": "text", "text": msg.text})
                continue

            if msg.role == "system":
                # Mid-conversation system message — map to assistant.
                logger.warning("Mid-conversation system message mapped to assistant role: %.80s…", msg.text)
                msg = TextMessage(role="assistant", text=msg.text)

            seen_non_system = True
            content_block = _to_content_block(msg)

            # Merge consecutive same-role messages into one message object.
            if anthropic_messages and anthropic_messages[-1]["role"] == msg.role:
                anthropic_messages[-1]["content"].append(content_block)
            else:
                anthropic_messages.append({"role": msg.role, "content": [content_block]})

        # Tag the last system block and the last message content block as cache
        # breakpoints. The system breakpoint caches tools+system (stable across a
        # run); the message breakpoint caches the growing conversation prefix.
        # Two breakpoints, within the 4-breakpoint limit.
        # PERF: It should be added at the AI level, not at the API level.
        if system_parts:
            system_parts[-1]["cache_control"] = _CACHE_CONTROL
        if anthropic_messages:
            anthropic_messages[-1]["content"][-1]["cache_control"] = _CACHE_CONTROL

        return system_parts or anthropic.NOT_GIVEN, anthropic_messages

    # ------------------------------------------------------------------
    # Tool formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools(tools: list[Tool] | None) -> Any:
        """Convert internal tools to Anthropic tool definitions."""
        if not tools:
            return anthropic.NOT_GIVEN

        result: list[dict[str, Any]] = []
        for tool in tools:
            schema = tool.json_schema
            entry: dict[str, Any] = {"name": schema["name"]}
            if "description" in schema:
                entry["description"] = schema["description"]
            if "parameters" in schema:
                entry["input_schema"] = schema["parameters"]
            result.append(entry)
        return result


# ------------------------------------------------------------------
# Helpers (module-private)
# ------------------------------------------------------------------


def _to_content_block(msg: TextMessage | ImageMessage) -> dict[str, Any]:
    """Convert a single message to an Anthropic content block."""
    if isinstance(msg, TextMessage):
        return {"type": "text", "text": msg.text}

    # ImageMessage — detect URL vs data-URL
    m = _DATA_URL_RE.match(msg.image)
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}

    # Plain URL
    return {"type": "image", "source": {"type": "url", "url": msg.image}}


def _parse_response(response: Any, tools: list[Tool] | None) -> ModelResponse:
    """Convert an Anthropic ``Message`` to a ``ModelResponse``."""
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = []

    for block in response.content:
        match block.type:
            case "text":
                content = (content or "") + block.text
            case "thinking":
                reasoning = (reasoning or "") + block.thinking
            case "tool_use":
                tool = next((t for t in (tools or []) if t.name == block.name), None)
                if tool is None:
                    raise ValueError(f"Tool '{block.name}' not found in tools")
                args = tool.args_type().model_validate(block.input)
                tool_calls.append(ToolCall(id=block.id, tool=tool, args=args))

    return ModelResponse(
        reasoning_content=reasoning,
        content=content,
        tool_calls=tool_calls or None,
        finish_reason=_parse_stop_reason(response.stop_reason),
    )


def _parse_stop_reason(reason: str | None) -> FinishReasons | None:
    match reason:
        case "end_turn" | "stop":
            return "stop"
        case "tool_use" | "tool_calls":
            return "tool_calls"
        case None:
            return None
        case _:
            return "unknown"


def _log_cache_usage(model: str, usage: Any) -> None:
    """Log Anthropic token usage with cache details."""
    if usage is None:
        return
    logger.info(
        "[CACHE] model=%s input=%s cache_creation=%s cache_read=%s output=%s",
        model,
        usage.input_tokens,
        getattr(usage, "cache_creation_input_tokens", 0),
        getattr(usage, "cache_read_input_tokens", 0),
        usage.output_tokens,
    )
