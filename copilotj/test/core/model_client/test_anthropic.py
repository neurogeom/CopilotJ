# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import anthropic

from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client.anthropic import AnthropicChatCompletionClient


def test_anthropic_format_system_message_extracted():
    """System messages are extracted into the separate system parameter."""
    msgs = [TextMessage(role="system", text="Be helpful")]
    system, messages = AnthropicChatCompletionClient._format_messages(msgs)
    assert messages == []
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "Be helpful"


def test_anthropic_format_no_system():
    """When no system messages exist, system is NOT_GIVEN."""
    msgs = [TextMessage(role="user", text="Hello")]
    system, messages = AnthropicChatCompletionClient._format_messages(msgs)
    assert system is anthropic.NOT_GIVEN
    assert len(messages) == 1


def test_anthropic_format_mixed_messages():
    """System, user, and assistant messages are properly separated."""
    system, messages = AnthropicChatCompletionClient._format_messages(
        [
            TextMessage(role="system", text="Be helpful"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="assistant", text="Hi there"),
            TextMessage(role="user", text="How are you?"),
        ]
    )
    assert len(system) == 1
    assert len(messages) == 3  # user + assistant + user
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


def test_anthropic_format_consecutive_same_role_merged():
    """Consecutive same-role messages are merged into one message object."""
    system, messages = AnthropicChatCompletionClient._format_messages(
        [
            TextMessage(role="user", text="Hello"),
            TextMessage(role="user", text="World"),
        ]
    )
    assert len(messages) == 1
    assert len(messages[0]["content"]) == 2
    assert messages[0]["content"][0]["text"] == "Hello"
    assert messages[0]["content"][1]["text"] == "World"


def test_anthropic_format_mid_conversation_system_mapped_to_assistant():
    """System messages after non-system messages are mapped to assistant role with a warning."""
    system, messages = AnthropicChatCompletionClient._format_messages(
        [
            TextMessage(role="system", text="Be helpful"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="assistant", text="Hi there"),
            TextMessage(role="system", text="Now speak French"),
            TextMessage(role="user", text="How are you?"),
        ]
    )
    # Leading system extracted to system parameter
    assert len(system) == 1
    assert system[0]["text"] == "Be helpful"
    # Mid-conversation system mapped to assistant and merged with preceding assistant
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"][0]["text"] == "Hi there"
    assert messages[1]["content"][1]["text"] == "Now speak French"  # merged
    assert messages[2]["role"] == "user"
    assert messages[2]["content"][0]["text"] == "How are you?"


def test_anthropic_format_image_url():
    """Plain URL images use the url source type."""
    msgs = [ImageMessage(role="user", image="https://example.com/img.png")]
    system, messages = AnthropicChatCompletionClient._format_messages(msgs)
    block = messages[0]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "url"
    assert block["source"]["url"] == "https://example.com/img.png"


def test_anthropic_format_image_data_url():
    """Data-URL images are parsed into base64 source with media_type."""
    msgs = [ImageMessage(role="user", image="data:image/png;base64,abc123")]
    system, messages = AnthropicChatCompletionClient._format_messages(msgs)
    block = messages[0]["content"][0]
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert block["source"]["data"] == "abc123"


def test_anthropic_format_text_message():
    """Text messages produce text content blocks."""
    msgs = [TextMessage(role="user", text="Hello")]
    system, messages = AnthropicChatCompletionClient._format_messages(msgs)
    block = messages[0]["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "Hello"


def test_anthropic_format_tools():
    """Tool schemas are converted to Anthropic format (input_schema)."""
    from unittest.mock import MagicMock

    from copilotj.core.tool import ToolSchema

    tool = MagicMock()
    tool.json_schema = ToolSchema(
        name="get_weather",
        description="Get the weather",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
            "additionalProperties": False,
        },
    )

    result = AnthropicChatCompletionClient._format_tools([tool])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get the weather"
    assert "input_schema" in result[0]
    assert result[0]["input_schema"]["properties"]["location"]["type"] == "string"


def test_anthropic_format_tools_none():
    """None tools returns NOT_GIVEN."""
    result = AnthropicChatCompletionClient._format_tools(None)
    assert result is anthropic.NOT_GIVEN


def test_anthropic_format_tools_empty():
    """Empty tool list returns NOT_GIVEN."""
    result = AnthropicChatCompletionClient._format_tools([])
    assert result is anthropic.NOT_GIVEN
