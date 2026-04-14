# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import os
from typing import Any, AsyncGenerator, Literal, Sequence, cast, overload, override

import langfuse.openai
import openai.types.chat
import openai.types.responses
import pydantic
from langchain_openai import OpenAIEmbeddings

from copilotj.core.config import get_llm_and_key, get_llm_base_url, get_proxy, get_vlm_and_key
from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.tool import Tool

__all__ = [
    "FinishReasons",
    "ToolCall",
    "ModelResponse",
    "ModelResponseChunk",
    "ModelClient",
    "ModelSyntaxError",
    "ModelProviderError",
    "OpenAIChatCompletionClient",
    "GeminiChatCompletionClient",
    "OllamaChatCompletionClient",
    "new_model_client",
    "new_vlm_model_client",
    "new_langchain_openai_embeddings",
]


type FinishReasons = Literal["stop", "tool_calls", "unknown"]


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


def new_model_client(model: str | None = None, api_key: str | None = None, *, proxy: str | None = None, base_url: str | None = None) -> ModelClient:
    model, api_key = get_llm_and_key(model, api_key)
    return _new_model_client(model, api_key, proxy=proxy, base_url=base_url)


def new_vlm_model_client(
    model: str | None = None, api_key: str | None = None, *, proxy: str | None = None
) -> ModelClient:
    model, api_key = get_vlm_and_key(model, api_key)
    return _new_model_client(model, api_key, proxy=proxy)


def _new_model_client(model: str, api_key: str, *, proxy: str | None, base_url: str | None = None) -> ModelClient:
    proxy = get_proxy(proxy)
    # base_url arg (from per-thread settings) takes precedence over the env var
    effective_base_url = base_url or get_llm_base_url()

    if model.startswith("ollama/"):
        model_name = model.split("/", 1)[1]
        return OllamaChatCompletionClient(model=model_name)

    elif model.startswith("deepseek/"):
        model_name = model.split("/", 1)[1]
        url = effective_base_url or "https://api.deepseek.com/v1"
        return OpenAIChatCompletionClient(model=model_name, api_key=api_key, base_url=url, proxy=proxy)

    elif model.startswith("gemini-"):
        return GeminiChatCompletionClient(model, api_key, proxy=proxy)

    elif model.startswith("claude-"):
        url = effective_base_url or "https://api.anthropic.com/v1"
        return OpenAIChatCompletionClient(model, api_key, base_url=url, proxy=proxy)

    elif model.startswith("gpt-"):
        if effective_base_url:
            return OpenAIChatCompletionClient(model, api_key, base_url=effective_base_url, proxy=proxy)
        else:
            return OpenAIResponseClient(model, api_key, proxy=proxy)

    elif (
        model.startswith("zai-org/")
        or model.startswith("Pro/")
        or model.startswith("deepseek")
        or model.startswith("moonshotai/")
        or model.startswith("Qwen/")
    ):
        url = effective_base_url or "https://api.siliconflow.cn/v1"
        return OpenAIChatCompletionClient(model, api_key, base_url=url, proxy=proxy)

    return OpenAIChatCompletionClient(model, api_key, base_url=effective_base_url, proxy=proxy)


# TODO: this is tricky
def new_langchain_openai_embeddings(
    *, api_key: str | None = None, proxy: str | None = None, **kwargs
) -> OpenAIEmbeddings:
    api_key = api_key or os.getenv("COPILOTJ_API_KEY", None)
    assert api_key is not None, "API key is required"
    kwargs.update(api_key=api_key)

    proxy = get_proxy(proxy)
    return OpenAIEmbeddings(api_key=api_key, openai_proxy=proxy)  # type: ignore


#############################
#          OpenAI           #
#############################


class OpenAIChatCompletionClient(ModelClient):
    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__()
        self._model, self._api_key = get_llm_and_key(model, api_key)
        http_client = openai.DefaultAsyncHttpxClient(proxy=proxy) if proxy is not None else None

        # Langfuse support can be safely ignored if LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is not set
        self._client = langfuse.openai.AsyncOpenAI(api_key=self._api_key, http_client=http_client, base_url=base_url)

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
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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
            async for chunk in stream:
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

        except openai.APIError as e:
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e

        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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

            return await self._client.chat.completions.create(
                model=self._model, messages=openai_messages, tools=openai_tools, **(extra_args or {}), stream=stream
            )
        except openai.APIError as e:
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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


class OpenAIResponseClient(ModelClient):
    def __init__(self, model: str, api_key: str, *, base_url: str | None = None, proxy: str | None = None):
        super().__init__()
        self._model, self._api_key = get_llm_and_key(model, api_key)
        http_client = openai.DefaultAsyncHttpxClient(proxy=proxy) if proxy is not None else None
        self._client = langfuse.openai.AsyncOpenAI(api_key=self._api_key, http_client=http_client, base_url=base_url)

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

                    case _:
                        raise NotImplementedError(f"Unsupported content type {item.type}: {item}")

            return ModelResponse(
                reasoning_content=None,
                content=response.output_text,
                tool_calls=tool_calls if len(tool_calls) > 0 else None,
                finish_reason="tool_calls" if len(tool_calls) > 0 else "stop",
            )
        except openai.APIError as e:
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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
                    case (
                        # response life cycle event
                        "response.created"  # first of all
                        | "response.in_progress"  # after created
                        | "response.completed"  # all done
                        # output item
                        | "response.output_item.done"
                        # output item - content
                        | "response.content_part.added"
                        | "response.output_text.done"  # after response.content_part.added
                        | "response.content_part.done"
                        # output item - function call
                        | "response.function_call_arguments.delta"
                    ):
                        pass

                    case "response.output_item.added":
                        match chunk.item.type:
                            case "message":
                                pass

                            case "function_call":
                                item = cast(openai.types.responses.ResponseFunctionToolCall, chunk.item)
                                if current_function_call is not None and current_function_call[0] != item.id:
                                    raise NotImplementedError("Nested function calls are not supported")

                                current_function_call = (item.id or "unknown_id", item.name)

                            case _:
                                raise NotImplementedError(f"Unsupported output item: {chunk.item.type}")

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
                        raise NotImplementedError(f"Unsupported chunk type: {chunk.type}")

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
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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
            raise ModelProviderError(f"OpenAI API error: {e.message}", "openai") from e
        except openai.OpenAIError as e:
            raise ModelProviderError(f"OpenAI error: {str(e)}", "openai") from e

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


#############################
#          Gemini           #
#############################


class GeminiChatCompletionClient(OpenAIChatCompletionClient):
    def __init__(self, model: str, api_key: str, *, proxy: str | None = None):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        super().__init__(model, api_key, proxy=proxy, base_url=base_url)


class GeminiResponseClient(OpenAIResponseClient):
    def __init__(self, model: str, api_key: str, *, proxy: str | None = None):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        super().__init__(model, api_key, proxy=proxy, base_url=base_url)


#############################
#          Ollama           #
#############################

try:
    import ollama
except ImportError:
    ollama = None


class OllamaChatCompletionClient(ModelClient):
    def __init__(self, model: str, base_url: str | None = None):
        assert ollama is not None, "Ollama client requires the 'ollama' package."

        super().__init__()
        # Base URL for Ollama server, e.g., http://localhost:11434
        self._host = base_url or os.getenv("COPILOTJ_BASE_URL", "http://localhost:11434")
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
                print("Warning: Image messages not fully supported by Ollama client yet. Skipping image.")
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
            print(f"Error during Ollama API call: {e}")
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
            print("Warning: Tool usage not implemented for Ollama client yet. Ignoring tools.")

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


#############################
#           CLI             #
#############################

if __name__ == "__main__":
    # python -m copilotj.core.model_client
    import asyncio
    import os

    import click
    from rich.console import Console
    from rich.prompt import Prompt

    from copilotj.core.config import load_env
    from copilotj.core.tool import FunctionTool

    @click.command()
    @click.option("--model", default=None, help="The OpenAI model to use.")
    @click.option("--api-key", default=None, help="The OpenAI API key.")
    @click.option("--proxy", default=None, help="The proxy to use.")
    @click.option("--stream", is_flag=True, help="Whether to stream the response.")
    def cli(model, api_key, proxy, stream):
        load_env()
        model, api_key = get_llm_and_key(model, api_key)
        proxy = get_proxy(proxy)

        console = Console()

        async def run():
            client = new_model_client()

            def get_temperature(city: str):
                return 15

            tools = []
            tools.append(FunctionTool(get_temperature, "Get current temperature"))

            messages = []

            while True:
                role = Prompt.ask(
                    "Select a role (or type 'exit' to quit)",
                    choices=["user", "system", "assistant", "exit"],
                    console=console,
                )
                if role.lower() == "exit":
                    break

                content = Prompt.ask("Enter your message", console=console)
                messages.append(TextMessage(role=role, text=content))  # type: ignore
                if role != "user":
                    continue  # Only allow user to send messages

                if stream:
                    model_stream = client.create_stream(messages=messages, tools=tools)
                    async for chunk in model_stream:
                        if isinstance(chunk, ToolCall):
                            console.print(f"Tool Call: {chunk.tool.name}, args: {chunk.args}")
                            result = await chunk.run()
                            console.print(f"Tool Call Result: {result}")
                            messages.append(TextMessage(role="assistant", text=str(result)))
                        else:
                            console.print(chunk.content or "", end="")

                    console.print()  # Add a newline after the stream

                else:
                    completion = await client.create(messages=messages, tools=tools)
                    print(completion)
                    console.print(completion.content)

                    if completion.tool_calls:
                        for tool_call in completion.tool_calls:
                            console.print(f"Tool Call: {tool_call.tool.name}, args: {tool_call.args}")
                            result = await tool_call.run()
                            console.print(f"Tool Call Result: {result}")
                            messages.append(TextMessage(role="assistant", text=str(result)))

        asyncio.run(run())

    cli()
