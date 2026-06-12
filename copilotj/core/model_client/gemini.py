# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import base64
import logging
from typing import Any, AsyncGenerator, Sequence, override

from google import genai
from google.genai import errors, types

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client._types import (
    FinishReasons,
    ModelClient,
    ModelProviderError,
    ModelResponse,
    ModelResponseChunk,
    ToolCall,
)
from copilotj.core.tool import Tool

__all__ = ["GeminiChatCompletionClient"]

logger = logging.getLogger(__name__)

# Gemini finish reason → CopilotJ finish reason mapping.
_GEMINI_FINISH_REASON_MAP: dict[str, FinishReasons] = {
    "STOP": "stop",
    "stop": "stop",
    "FinishReason.STOP": "stop",
    "TOOL_CALLS": "tool_calls",
    "tool_calls": "tool_calls",
    "FinishReason.TOOL_CALLS": "tool_calls",
    "MAX_TOKENS": "stop",
    "FinishReason.MAX_TOKENS": "stop",
    "SAFETY": "unknown",
    "RECITATION": "unknown",
    "OTHER": "unknown",
    "BLOCKLIST": "unknown",
    "PROHIBITED_CONTENT": "unknown",
    "SPII": "unknown",
    "MALFORMED_FUNCTION_CALL": "unknown",
}

# MIME type lookup by URL extension.
_EXTENSION_MIME_MAP: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class GeminiChatCompletionClient(ModelClient):
    """ModelClient implementation using the native ``google-genai`` SDK.

    Supports text messages, image messages (URL or data-URI), system
    instructions, and function/tool calling with both streaming and
    non-streaming modes.
    """

    def __init__(self, model: str, api_key: str, *, proxy: str | None = None, base_url: str | None = None):
        super().__init__()
        self._model = model
        self._api_key = api_key

        http_options: types.HttpOptions | None = None
        if proxy or base_url:
            http_options = types.HttpOptions()
            if proxy:
                http_options.client_args = {"proxy": proxy}
                http_options.async_client_args = {"proxy": proxy}
            if base_url:
                http_options.base_url = base_url

        self._client = genai.Client(api_key=api_key, http_options=http_options)

    # ------------------------------------------------------------------
    # ModelClient interface
    # ------------------------------------------------------------------

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
        contents, system_instruction = self._convert_messages(messages)
        config = self._build_config(system_instruction, tools, extra_args)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except errors.APIError as e:  # type: ignore[misc]
            raise ModelProviderError(f"Gemini API error: {e.message}", "gemini") from e
        except Exception as e:
            raise ModelProviderError(f"Gemini error: {e!s}", "gemini") from e

        _log_cache_usage(self._model, response.usage_metadata)
        return self._parse_response(response, tools)

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        contents, system_instruction = self._convert_messages(messages)
        config = self._build_config(system_instruction, tools, extra_args)

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model, contents=contents, config=config
            )

            pending_function_calls: list[types.FunctionCall] = []
            stream_usage = None

            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    stream_usage = chunk.usage_metadata
                # Accumulate function call parts.
                # NOTE: a non-None ``content`` is always truthy (pydantic model),
                # yet ``content.parts`` can be ``None`` in streamed chunks (common
                # with thinking models), so guard it explicitly — mirroring the
                # SDK's own ``parts``/``text`` properties.
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            pending_function_calls.append(part.function_call)

                # Yield text content.
                text = _safe_chunk_text(chunk)
                if text:
                    yield ModelResponseChunk(reasoning_content=None, content=text, finish_reason=None)

                # Check for finish.
                finish_reason = None
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish_reason = self._parse_finish_reason(chunk.candidates[0].finish_reason)

                if finish_reason is not None:
                    # Yield accumulated tool calls before the final chunk.
                    for fc in pending_function_calls:
                        tool = next((t for t in (tools or []) if t.name == fc.name), None)
                        if tool is None:
                            raise ValueError(f"Tool '{fc.name}' not found in tools")

                        args = tool.args_type().model_validate(fc.args)
                        yield ToolCall(id=fc.id or f"tc_{len(pending_function_calls)}", tool=tool, args=args)
                    pending_function_calls.clear()

                    yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason=finish_reason)

            if stream_usage is not None:
                _log_cache_usage(self._model, stream_usage)

        except errors.APIError as e:  # type: ignore[misc]
            raise ModelProviderError(f"Gemini API error: {e.message}", "gemini") from e
        except Exception as e:
            raise ModelProviderError(f"Gemini error: {e!s}", "gemini") from e

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @classmethod
    def _convert_messages(
        cls,
        messages: Sequence[TextMessage | ImageMessage],
    ) -> tuple[list[types.Content], str | None]:
        """Convert CopilotJ messages to Gemini Content list + system instruction.

        Only **leading** system messages (from the start of the sequence) are
        extracted as ``system_instruction``.  Any system message that appears
        *after* a non-system message is demoted to ``user`` role with a
        warning, so that its position in the conversation is preserved.

        Returns:
            (contents, system_instruction) where *system_instruction* may be
            ``None`` when no leading system-role messages are present.
        """
        system_parts: list[str] = []
        converted: list[TextMessage | ImageMessage] = []
        seen_non_system = False

        for msg in messages:
            if isinstance(msg, TextMessage) and msg.role == "system":
                if seen_non_system:
                    # Mid-conversation system message — demote to user role.
                    logger.warning(
                        "Gemini does not support mid-conversation system messages; demoting to user role: %s",
                        msg.text[:80],
                    )
                    converted.append(TextMessage(role="user", text=msg.text))
                else:
                    system_parts.append(msg.text)
            else:
                seen_non_system = True
                converted.append(msg)

        system_instruction = "\n\n".join(system_parts) if system_parts else None

        # Group consecutive messages with the same role.
        contents: list[types.Content] = []
        group: list[TextMessage | ImageMessage] = []

        for msg in converted:
            if group and group[0].role != msg.role:
                contents.append(cls._merge_group(group))
                group.clear()
            group.append(msg)

        if group:
            contents.append(cls._merge_group(group))

        return contents, system_instruction

    @classmethod
    def _merge_group(
        cls,
        messages: Sequence[TextMessage | ImageMessage],
    ) -> types.Content:
        """Merge consecutive same-role messages into one ``types.Content``."""
        gemini_role = "model" if messages[0].role == "assistant" else "user"
        parts: list[types.Part] = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                parts.append(types.Part.from_text(text=msg.text))
            elif isinstance(msg, ImageMessage):
                parts.append(cls._convert_image(msg.image))
            else:
                raise ValueError(f"Unsupported message type: {msg!r}")
        return types.Content(role=gemini_role, parts=parts)

    @staticmethod
    def _convert_image(image: str) -> types.Part:
        """Convert an image URL or data-URI string to a ``types.Part``."""
        if image.startswith("data:"):
            # data:<mime_type>;base64,<base64-data>
            header, _, data = image.partition(",")
            mime_type = header.split(";")[0].split(":", 1)[1]
            return types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime_type)

        # URL – infer mime type from extension, default to image/jpeg.
        mime_type = "image/jpeg"
        lower = image.lower()
        for ext, mt in _EXTENSION_MIME_MAP.items():
            if lower.endswith(ext):
                mime_type = mt
                break
        return types.Part.from_uri(file_uri=image, mime_type=mime_type)

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[Tool] | None) -> list[types.Tool] | None:
        """Convert CopilotJ tools to Gemini ``types.Tool``."""
        if not tools:
            return None

        declarations = []
        for tool in tools:
            schema = tool.json_schema
            params = schema.get("parameters")
            declarations.append(
                types.FunctionDeclaration(
                    name=schema["name"],
                    description=schema.get("description", ""),
                    parameters_json_schema=params if params else {"type": "object", "properties": {}},
                )
            )

        return [types.Tool(function_declarations=declarations)]

    # ------------------------------------------------------------------
    # Config builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(
        system_instruction: str | None,
        tools: list[Tool] | None,
        extra_args: dict[str, Any] | None,
    ) -> types.GenerateContentConfig:
        """Build a ``GenerateContentConfig`` from the individual pieces."""
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if system_instruction:
            config.system_instruction = system_instruction

        gemini_tools = GeminiChatCompletionClient._convert_tools(tools)
        if gemini_tools:
            config.tools = gemini_tools

        # Merge extra_args (temperature, max_output_tokens, etc.) onto config.
        if extra_args:
            for key, value in extra_args.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return config

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(
        response: Any,
        tools: list[Tool] | None,
    ) -> ModelResponse:
        """Parse a non-streaming ``GenerateContentResponse``."""
        content = None
        tool_calls: list[ToolCall] = []

        # Text content.
        if response.text:
            content = response.text

        # Function calls.
        if response.function_calls:
            for fc in response.function_calls:
                tool = next((t for t in (tools or []) if t.name == fc.name), None)
                if tool is None:
                    raise ValueError(f"Tool '{fc.name}' not found in tools")
                args = tool.args_type().model_validate(fc.args)
                tool_calls.append(ToolCall(id=fc.id or f"tc_{len(tool_calls)}", tool=tool, args=args))

        # Finish reason.
        finish_reason: FinishReasons = "unknown"
        if response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason is not None:
                finish_reason = GeminiChatCompletionClient._parse_finish_reason(candidate.finish_reason)

        if tool_calls:
            finish_reason = "tool_calls"

        return ModelResponse(
            reasoning_content=None,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _parse_finish_reason(reason: Any) -> FinishReasons:
        """Map a Gemini finish reason to CopilotJ's ``FinishReasons``."""
        reason_str = str(reason)
        return _GEMINI_FINISH_REASON_MAP.get(reason_str, "unknown")


def _safe_chunk_text(chunk: Any) -> str | None:
    """Extract text from a streaming chunk, returning None on failure."""
    try:
        text = chunk.text
        return text if text else None
    except Exception:
        return None


def _log_cache_usage(model: str, usage_metadata: Any) -> None:
    """Log Gemini token usage with cache details."""
    if usage_metadata is None:
        return
    logger.info(
        "[CACHE] model=%s prompt=%s cached=%s candidates=%s total=%s",
        model,
        getattr(usage_metadata, "prompt_token_count", None),
        getattr(usage_metadata, "cached_content_token_count", 0),
        getattr(usage_metadata, "candidates_token_count", None),
        getattr(usage_metadata, "total_token_count", None),
    )
