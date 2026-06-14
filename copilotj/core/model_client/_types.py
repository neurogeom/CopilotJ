# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
from typing import Any, AsyncGenerator, Literal, Sequence

import pydantic

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.tool import Tool

logger = logging.getLogger(__name__)

__all__ = [
    "FinishReasons",
    "LLMProvider",
    "_VALID_PROVIDERS",
    "ToolCall",
    "ModelResponse",
    "ModelResponseChunk",
    "ModelSyntaxError",
    "ModelProviderError",
    "ModelClient",
]


type FinishReasons = Literal["stop", "tool_calls", "unknown"]

type LLMProvider = Literal[
    "openai",
    "openai-responses",
    "anthropic",
    "gemini",
    "ollama",
    "deepseek",
    "siliconflow",
    "openrouter",
    "openai-compatible",
]

_VALID_PROVIDERS = (
    "openai",
    "openai-responses",
    "anthropic",
    "gemini",
    "ollama",
    "deepseek",
    "siliconflow",
    "openrouter",
    "openai-compatible",
)


class ToolCall(pydantic.BaseModel):
    id: str
    tool: Tool
    args: pydantic.BaseModel

    def __str__(self) -> str:
        args_type = self.tool.args_type()
        if args_type is None or len(args_type.model_fields) == 0:
            return self.tool.name

        return self.tool.name + f" with args {str(self.args)}"

    async def run(self) -> pydantic.BaseModel:
        """Run the tool with the provided arguments."""
        return await self.tool.run(self.args)

    @pydantic.field_serializer("args")
    def _serialize_args(self, v: pydantic.BaseModel, info: pydantic.FieldSerializationInfo) -> Any:
        if isinstance(v, pydantic.BaseModel):
            return v.model_dump()

        return v


class ModelResponse(pydantic.BaseModel):
    reasoning_content: str | None
    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: FinishReasons | None


class ModelResponseChunk(pydantic.BaseModel):
    reasoning_content: str | None
    content: str | None
    finish_reason: FinishReasons | None


class ModelSyntaxError(ValueError):
    chat_completion: ModelResponse | None

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.chat_completion = None


class ModelProviderError(Exception):
    """Exception raised for errors related to LLM providers.

    Attributes:
        message -- explanation of the error
        provider -- the provider that caused the error (optional)
    """

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message)
        self.message = message
        self.provider = provider


class ModelClient(abc.ABC):
    @abc.abstractmethod
    def get_model(self) -> str:
        """Get the model name used by this client."""
        ...

    @abc.abstractmethod
    def get_api_key(self) -> str | None:
        """Get the API key used by this client, if applicable."""
        ...

    @abc.abstractmethod
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse: ...

    @abc.abstractmethod
    def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]: ...
