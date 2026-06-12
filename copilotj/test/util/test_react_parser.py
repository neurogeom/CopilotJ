# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

from copilotj.core.message import (
    ImageMessage,
    TextMessage,
    ToolCallMessage,
    ToolCallRecord,
    ToolResultMessage,
)
from copilotj.core.model_client import ModelClient, ModelResponse, ModelResponseChunk, ModelSyntaxError, ToolCall
from copilotj.core.tool import FunctionTool, Tool
from copilotj.util.react_parser import ReActChatCompletionClient, _build_last_line_prefix_regex


def lookup(query: str) -> str:
    return query


class _StubModelClient(ModelClient):
    def __init__(
        self,
        *,
        response: ModelResponse | None = None,
        stream_chunks: list[ModelResponseChunk | ToolCall] | None = None,
    ) -> None:
        self._response = response or ModelResponse(
            reasoning_content=None,
            content=None,
            tool_calls=None,
            finish_reason="unknown",
        )
        self._stream_chunks = stream_chunks or []

    @override
    def get_model(self) -> str:
        return "stub"

    @override
    def get_api_key(self) -> str | None:
        return None

    @override
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse:
        return self._response

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        for chunk in self._stream_chunks:
            yield chunk


def test_create_parses_standard_react_response():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="""\
Thought: need more information
Action:
{"name": "lookup", "args": {"query": "cells"}}
Final Answer: found it
""",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.reasoning_content == "need more information"
    assert response.content == "found it"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"
    assert response.finish_reason == "stop"


def test_create_parses_bolded_react_keywords():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="""\
**Thought**: need more information
**Action**:
{"name": "lookup", "args": {"query": "cells"}}
**Final Answer**: found it
""",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.reasoning_content == "need more information"
    assert response.content == "found it"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"
    assert response.finish_reason == "stop"


def test_create_parses_bolded_keywords_with_colon_inside_bold():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="""\
**Thought:** need more information
**Action:**
{"name": "lookup", "args": {"query": "cells"}}
**Final Answer:** found it
""",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[tool]))

    assert response.reasoning_content == "need more information"
    assert response.content == "found it"
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool.name == "lookup"
    assert response.tool_calls[0].args.query == "cells"
    assert response.finish_reason == "stop"


def test_create_returns_raw_content_without_react_keywords():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="Just answer directly.",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))

    assert response.reasoning_content is None
    assert response.content == "Just answer directly."
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


def test_create_parses_final_answer_without_thought_or_action():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content="Final Answer: Just answer directly.",
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    response = asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))

    assert response.reasoning_content is None
    assert response.content == "Just answer directly."
    assert response.tool_calls is None
    assert response.finish_reason == "stop"


def test_create_raises_for_unknown_tool():
    client = ReActChatCompletionClient(
        _StubModelClient(
            response=ModelResponse(
                reasoning_content=None,
                content='Action: {"name": "missing_tool", "args": {}}',
                tool_calls=None,
                finish_reason="stop",
            )
        )
    )

    try:
        asyncio.run(client.create([TextMessage(role="user", text="help")], tools=[]))
    except ModelSyntaxError as exc:
        assert "Tool 'missing_tool' not found" in exc.message
    else:
        raise AssertionError("Expected ModelSyntaxError for unknown tool")


def test_create_stream_parses_standard_react_response():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(reasoning_content=None, content="Thought: inspect\n", finish_reason=None),
                ModelResponseChunk(
                    reasoning_content=None,
                    content='Action: {"name": "lookup", "args": {"query": "cells"}}\nFinal Answer: done',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = [
        item
        for item in asyncio.run(_collect_stream(client, [tool]))
        if not (isinstance(item, ModelResponseChunk) and item.reasoning_content == "")
    ]

    assert len(items) == 4
    assert isinstance(items[0], ModelResponseChunk)
    assert items[0].reasoning_content == "inspect\n"
    assert isinstance(items[1], ToolCall)
    assert items[1].tool.name == "lookup"
    assert items[1].args.query == "cells"
    assert isinstance(items[2], ModelResponseChunk)
    assert items[2].content == "done"
    assert isinstance(items[3], ModelResponseChunk)
    assert items[3].finish_reason == "stop"


def test_create_stream_parses_bolded_react_keywords():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(reasoning_content=None, content="**Thought**: inspect\n", finish_reason=None),
                ModelResponseChunk(
                    reasoning_content=None,
                    content='**Action**: {"name": "lookup", "args": {"query": "cells"}}\n**Final Answer**: done',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = [
        item
        for item in asyncio.run(_collect_stream(client, [tool]))
        if not (isinstance(item, ModelResponseChunk) and item.reasoning_content == "")
    ]

    assert len(items) == 4
    assert isinstance(items[0], ModelResponseChunk)
    assert items[0].reasoning_content == "inspect\n"
    assert isinstance(items[1], ToolCall)
    assert items[1].tool.name == "lookup"
    assert items[1].args.query == "cells"
    assert isinstance(items[2], ModelResponseChunk)
    assert items[2].content == "done"
    assert isinstance(items[3], ModelResponseChunk)
    assert items[3].finish_reason == "stop"


def test_create_stream_parses_bolded_keywords_with_colon_inside_bold():
    tool = FunctionTool(lookup, "Look up a query.", name="lookup")
    client = ReActChatCompletionClient(
        _StubModelClient(
            stream_chunks=[
                ModelResponseChunk(reasoning_content=None, content="**Thought:** inspect\n", finish_reason=None),
                ModelResponseChunk(
                    reasoning_content=None,
                    content='**Action:** {"name": "lookup", "args": {"query": "cells"}}\n**Final Answer:** done',
                    finish_reason=None,
                ),
                ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop"),
            ]
        )
    )

    items = [
        item
        for item in asyncio.run(_collect_stream(client, [tool]))
        if not (isinstance(item, ModelResponseChunk) and item.reasoning_content == "")
    ]

    assert len(items) == 4
    assert isinstance(items[0], ModelResponseChunk)
    assert items[0].reasoning_content == "inspect\n"
    assert isinstance(items[1], ToolCall)
    assert items[1].tool.name == "lookup"
    assert items[1].args.query == "cells"
    assert isinstance(items[2], ModelResponseChunk)
    assert items[2].content == "done"
    assert isinstance(items[3], ModelResponseChunk)
    assert items[3].finish_reason == "stop"


def test_build_last_line_prefix_regex_matches_action_and_final_prefixes():
    pattern = _build_last_line_prefix_regex("Action", "Final Answer")

    assert pattern.search("A")
    assert pattern.search("Actio")
    assert pattern.search("**Act")
    assert pattern.search("**Action**")
    assert pattern.search("**Action:")
    assert pattern.search("**Action:**")
    assert pattern.search("Final")
    assert pattern.search("**Final Answe")
    assert pattern.search("Final Answe")
    assert not pattern.search("Observation")


async def _collect_stream(client: ReActChatCompletionClient, tools: list[Tool]) -> list[ModelResponseChunk | ToolCall]:
    items: list[ModelResponseChunk | ToolCall] = []
    async for item in client.create_stream([TextMessage(role="user", text="help")], tools=tools):
        items.append(item)
    return items


# --- _convert_messages tests ---


def test_convert_tool_call_message_with_thought():
    messages = [
        ToolCallMessage(
            reasoning_content="I need to look something up",
            tool_calls=[ToolCallRecord(id="tc1", name="lookup", arguments={"query": "cells"})],
        )
    ]
    result = ReActChatCompletionClient._convert_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], TextMessage)
    assert result[0].role == "assistant"
    assert "Thought: I need to look something up" in result[0].text
    assert 'Action: {"name": "lookup"' in result[0].text


def test_convert_tool_call_message_without_thought():
    messages = [ToolCallMessage(tool_calls=[ToolCallRecord(id="tc1", name="lookup", arguments={"query": "cells"})])]
    result = ReActChatCompletionClient._convert_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], TextMessage)
    assert "Thought:" not in result[0].text
    assert 'Action: {"name": "lookup"' in result[0].text


def test_convert_tool_result_message():
    messages = [ToolResultMessage(tool_call_id="tc1", content="found 42 items")]
    result = ReActChatCompletionClient._convert_messages(messages)

    assert len(result) == 1
    assert isinstance(result[0], TextMessage)
    assert result[0].role == "user"
    assert result[0].text == "Observation:\nfound 42 items"


def test_convert_mixed_messages():
    messages = [
        TextMessage(role="user", text="hello"),
        ToolCallMessage(
            reasoning_content="thinking",
            tool_calls=[ToolCallRecord(id="tc1", name="lookup", arguments={"query": "x"})],
        ),
        ToolResultMessage(tool_call_id="tc1", content="result text"),
        TextMessage(role="assistant", text="done"),
    ]
    result = ReActChatCompletionClient._convert_messages(messages)

    assert len(result) == 4
    # First: TextMessage passed through
    assert isinstance(result[0], TextMessage)
    assert result[0].text == "hello"
    # Second: ToolCallMessage → TextMessage with Thought + Action
    assert isinstance(result[1], TextMessage)
    assert "Thought: thinking" in result[1].text
    assert "Action:" in result[1].text
    # Third: ToolResultMessage → TextMessage with Observation
    assert isinstance(result[2], TextMessage)
    assert result[2].text == "Observation:\nresult text"
    # Fourth: TextMessage passed through
    assert isinstance(result[3], TextMessage)
    assert result[3].text == "done"


def test_convert_empty_tool_calls_list():
    messages = [ToolCallMessage(tool_calls=[])]
    result = ReActChatCompletionClient._convert_messages(messages)

    # Empty tool_calls with no reasoning_content → empty parts → no message
    assert len(result) == 0


def test_convert_preserves_non_tool_messages():
    messages = [
        TextMessage(role="user", text="question"),
        ImageMessage(role="user", image="data:image/png;base64,abc"),
    ]
    result = ReActChatCompletionClient._convert_messages(messages)

    assert len(result) == 2
    assert isinstance(result[0], TextMessage)
    assert result[0].text == "question"
    assert isinstance(result[1], ImageMessage)
    assert result[1].image == "data:image/png;base64,abc"


class _CapturingStubClient(ModelClient):
    """Stub that records the messages it receives."""

    def __init__(self):
        self.received_messages = None

    @override
    def get_model(self) -> str:
        return "stub"

    @override
    def get_api_key(self) -> str | None:
        return None

    @override
    async def create(
        self,
        messages,
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.received_messages = messages
        return ModelResponse(
            reasoning_content=None, content="Final Answer: done", tool_calls=None, finish_reason="stop"
        )

    @override
    async def create_stream(self, messages, *, tools=None, extra_args=None):
        return
        yield  # make this an async generator  # noqa: RET503


def test_create_converts_messages_before_forwarding():
    stub = _CapturingStubClient()
    client = ReActChatCompletionClient(stub)

    tool_messages = [
        TextMessage(role="user", text="help"),
        ToolCallMessage(
            reasoning_content="thinking",
            tool_calls=[ToolCallRecord(id="tc1", name="lookup", arguments={"query": "x"})],
        ),
        ToolResultMessage(tool_call_id="tc1", content="observed"),
    ]

    asyncio.run(client.create(tool_messages, tools=[]))

    # The stub should have received only TextMessage/ImageMessage
    assert stub.received_messages is not None
    for msg in stub.received_messages:
        assert isinstance(msg, (TextMessage, ImageMessage)), f"Unexpected type: {type(msg)}"

    # Verify reconstructed content
    # The ToolCallMessage should have been converted to a TextMessage with Action:
    assert any("Action:" in m.text for m in stub.received_messages if isinstance(m, TextMessage))
    # The ToolResultMessage should have been converted to a TextMessage with Observation:
    assert any("Observation:" in m.text for m in stub.received_messages if isinstance(m, TextMessage))
