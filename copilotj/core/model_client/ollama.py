# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, AsyncGenerator, Sequence, override

import openai.types.chat

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

logger = logging.getLogger(__name__)

__all__ = [
    "OllamaChatCompletionClient",
]

try:
    import ollama
except ImportError:
    ollama = None


class OllamaChatCompletionClient(ModelClient):
    def __init__(self, model: str, base_url: str | None = None):
        assert ollama is not None, "Ollama client requires the 'ollama' package."

        super().__init__()
        # Base URL for Ollama server, e.g., http://localhost:11434
        self._host = base_url or "http://localhost:11434"
        self._client = ollama.AsyncClient(host=self._host)
        self._model = model

    @override
    def get_model(self) -> str:
        return self._model

    @override
    def get_api_key(self) -> str | None:
        return None

    def _format_messages(self, messages: Sequence[TextMessage | ImageMessage]) -> list[dict]:
        """Formats messages for the Ollama API."""
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, TextMessage):
                ollama_messages.append({"role": msg.role, "content": msg.text})
            elif isinstance(msg, ImageMessage):
                logger.warning("Image messages not fully supported by Ollama client yet. Skipping image.")
            else:
                raise ValueError(f"Unsupported message type: {msg}")
        return ollama_messages

    @override
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse:
        ollama_messages = self._format_messages(messages)
        ollama_tools = None
        if tools is not None:
            ollama_tools = [
                openai.types.chat.ChatCompletionToolParam(
                    type="function",
                    function=tool.json_schema,  # type: ignore
                )
                for tool in tools
            ]

        try:
            response = await self._client.chat(
                model=self._model,
                messages=ollama_messages,
                tools=ollama_tools,
                options=extra_args,  # Pass extra args as options if applicable
            )
            content = response.get("message", {}).get("content")
            finish_reason: FinishReasons = "stop" if response.get("done") else "unknown"

            return ModelResponse(reasoning_content=None, content=content, tool_calls=None, finish_reason=finish_reason)
        except openai.APIError as e:
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e
        except Exception as e:
            logger.error("Error during Ollama API call: %s", e)
            return ModelResponse(
                reasoning_content=None, content=f"Error: {e}", tool_calls=None, finish_reason="unknown"
            )

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        if tools:
            logger.warning("Tool usage not implemented for Ollama client yet. Ignoring tools.")

        ollama_messages = self._format_messages(messages)
        try:
            stream = await self._client.chat(
                model=self._model, messages=ollama_messages, stream=True, options=extra_args
            )

            async for chunk in stream:
                content_chunk = chunk.get("message", {}).get("content")
                is_done = chunk.get("done", False)
                finish_reason: FinishReasons | None = "stop" if is_done else None

                if content_chunk:
                    yield ModelResponseChunk(reasoning_content=None, content=content_chunk, finish_reason=finish_reason)
                elif is_done:
                    yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop")

        except Exception as e:
            raise ModelProviderError(f"Ollama error: {str(e)}", "ollama") from e
