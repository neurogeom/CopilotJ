# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from copilotj.core.message import (
    ImageMessage,
    TextMessage,
    ToolCallMessage,
    ToolCallRecord,
    ToolResultMessage,
)
from copilotj.core.model_client import OpenAIChatCompletionClient, OpenAIResponseClient, detect_tool_call_mode


def test_format_single_text_message():
    text_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._format_messages([text_msg])

    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["type"] == "text"  # type: ignore
    assert result[0]["content"][0]["text"] == "Hello"  # type: ignore


def test_format_single_image_message():
    img_msg = ImageMessage(role="user", image="data:image/jpeg;base64,abc123")
    result = OpenAIChatCompletionClient._format_messages([img_msg])

    assert result[0]["content"][0]["type"] == "image_url"  # type: ignore
    assert (
        result[0]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"  # type: ignore
    )


def test_format_mixed_messages_same_role():
    text_msg = TextMessage(role="user", text="Hello")
    img_msg = ImageMessage(role="user", image="abc123")
    result = OpenAIChatCompletionClient._format_messages([text_msg, img_msg])

    assert len(result[0]["content"]) == 2  # type: ignore
    assert result[0]["content"][0]["type"] == "text"  # type: ignore
    assert result[0]["content"][1]["type"] == "image_url"  # type: ignore


def test_format_messages_different_roles():
    system_msg = TextMessage(role="system", text="Be helpful")
    user_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._format_messages([system_msg, user_msg])

    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"


def test_merge_single_text_message():
    text_msg = TextMessage(role="user", text="Hello")
    result = OpenAIChatCompletionClient._merge_messages([text_msg])

    assert result["role"] == "user"
    assert result["content"][0]["type"] == "text"  # type: ignore
    assert result["content"][0]["text"] == "Hello"  # type: ignore


def test_merge_single_image_message():
    img_msg = ImageMessage(role="user", image="data:image/jpeg;base64,abc123")
    result = OpenAIChatCompletionClient._merge_messages([img_msg])

    assert result["content"][0]["type"] == "image_url"  # type: ignore
    assert result["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"  # type: ignore


def test_merge_mixed_messages():
    text_msg = TextMessage(role="user", text="Hello")
    img_msg = ImageMessage(role="user", image="abc123")
    result = OpenAIChatCompletionClient._merge_messages([text_msg, img_msg])

    assert len(result["content"]) == 2  # type: ignore
    assert result["content"][0]["type"] == "text"  # type: ignore
    assert result["content"][1]["type"] == "image_url"  # type: ignore


# --- _format_tool_message tests ---


def test_format_tool_call_message():
    msg = ToolCallMessage(
        reasoning_content="I need to search",
        tool_calls=[ToolCallRecord(id="call_1", name="search", arguments={"query": "cells"})],
    )
    result = OpenAIChatCompletionClient._format_tool_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] == "I need to search"  # type: ignore
    tool_calls = result["tool_calls"]  # type: ignore
    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "search"  # type: ignore
    assert '"query"' in tool_calls[0]["function"]["arguments"]  # type: ignore


def test_format_tool_call_message_without_thought():
    msg = ToolCallMessage(
        tool_calls=[ToolCallRecord(id="call_2", name="lookup", arguments={})],
    )
    result = OpenAIChatCompletionClient._format_tool_message(msg)

    assert result["role"] == "assistant"
    assert result["content"] is None  # type: ignore


def test_format_tool_result_message():
    msg = ToolResultMessage(tool_call_id="call_1", content="found 42 items")
    result = OpenAIChatCompletionClient._format_tool_message(msg)

    assert result["role"] == "tool"  # type: ignore
    assert result["tool_call_id"] == "call_1"  # type: ignore
    assert result["content"] == "found 42 items"  # type: ignore


def test_format_tool_call_message_multiple_calls():
    msg = ToolCallMessage(
        reasoning_content="parallel calls",
        tool_calls=[
            ToolCallRecord(id="c1", name="search", arguments={"q": "a"}),
            ToolCallRecord(id="c2", name="lookup", arguments={"q": "b"}),
        ],
    )
    result = OpenAIChatCompletionClient._format_tool_message(msg)

    tool_calls = result["tool_calls"]  # type: ignore
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "search"  # type: ignore
    assert tool_calls[1]["function"]["name"] == "lookup"  # type: ignore


# --- _format_tool_input tests (Responses API) ---


def test_format_tool_input_call_message():
    msg = ToolCallMessage(
        tool_calls=[ToolCallRecord(id="fc_1", name="search", arguments={"query": "test"})],
    )
    result = OpenAIResponseClient._format_tool_input(msg)

    assert result["type"] == "function_call"  # type: ignore
    assert result["id"] == "fc_1"  # type: ignore
    assert result["name"] == "search"  # type: ignore
    assert '"query"' in result["arguments"]  # type: ignore


def test_format_tool_input_result_message():
    msg = ToolResultMessage(tool_call_id="fc_1", content="result data")
    result = OpenAIResponseClient._format_tool_input(msg)

    assert result["type"] == "function_call_output"  # type: ignore
    assert result["call_id"] == "fc_1"  # type: ignore
    assert result["output"] == "result data"  # type: ignore


# --- detect_tool_call_mode tests ---


@patch("copilotj.core.model_info.get_model_capabilities")
def test_detect_native_for_known_model(mock_caps):
    from copilotj.core.model_info import ModelCapabilities

    mock_caps.return_value = ModelCapabilities(
        model="gpt-4o",
        supports_vision=True,
        supports_function_calling=True,
        context_window=128000,
        max_output_tokens=16384,
        source="litellm_db",
    )

    # Create a minimal ModelClient subclass for testing
    class _TestClient(OpenAIChatCompletionClient):
        pass

    client = _TestClient.__new__(_TestClient)
    client._model = "gpt-4o"  # noqa: SLF001

    assert detect_tool_call_mode(client) == "native"


@patch("copilotj.core.model_info.get_model_capabilities")
def test_detect_react_for_unsupported_model(mock_caps):
    from copilotj.core.model_info import ModelCapabilities

    mock_caps.return_value = ModelCapabilities(
        model="some-local-model",
        supports_vision=False,
        supports_function_calling=False,
        context_window=None,
        max_output_tokens=None,
        source="unknown",
    )

    class _TestClient(OpenAIChatCompletionClient):
        pass

    client = _TestClient.__new__(_TestClient)
    client._model = "some-local-model"  # noqa: SLF001

    assert detect_tool_call_mode(client) == "react"
