# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from copilotj.core.message import TextMessage
from copilotj.core.model_client.openrouter import OpenRouterChatCompletionClient, _supports_explicit_cache

_CC = {"type": "ephemeral"}


# --------------------------------------------------------------------------- #
# _supports_explicit_cache
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-haiku",
        "anthropic/claude-sonnet-4",
        "qwen/qwen3-max",
        "qwen/qwen-plus",
        "~anthropic/claude-3-haiku",  # provider-pin prefix still counts
        "~qwen/qwen3-max",
    ],
)
def test_supports_explicit_cache_true(model):
    assert _supports_explicit_cache(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o-mini",
        "deepseek/deepseek-chat",
        "google/gemini-2.5-pro",
        "meta-llama/llama-3.1-8b-instruct",
        "claude-3-haiku",  # bare name -> not OpenRouter-routed to Anthropic
    ],
)
def test_supports_explicit_cache_false(model):
    assert _supports_explicit_cache(model) is False


# --------------------------------------------------------------------------- #
# _format_messages breakpoint placement
# --------------------------------------------------------------------------- #


def _format(model: str, messages):
    """Build a client (no network) and run its message formatter."""
    return OpenRouterChatCompletionClient(model, "k")._format_messages(messages)


def test_anthropic_tags_system_and_last_message():
    msgs = _format(
        "anthropic/claude-3-haiku",
        [
            TextMessage(role="system", text="Be helpful"),
            TextMessage(role="user", text="Hello"),
            TextMessage(role="assistant", text="Hi"),
            TextMessage(role="user", text="How are you?"),
        ],
    )
    # last system block tagged
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"][-1]["cache_control"] == _CC
    # last message block tagged
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"][-1]["cache_control"] == _CC
    # middle messages not tagged
    assert "cache_control" not in msgs[1]["content"][-1]
    assert "cache_control" not in msgs[2]["content"][-1]


def test_qwen_tags_system_and_last_message():
    """Qwen uses the same explicit-cache syntax as Anthropic."""
    msgs = _format(
        "qwen/qwen3-max",
        [
            TextMessage(role="system", text="Be helpful"),
            TextMessage(role="user", text="Hello"),
        ],
    )
    assert msgs[0]["content"][-1]["cache_control"] == _CC
    assert msgs[-1]["content"][-1]["cache_control"] == _CC


def test_qwen_provider_pin_prefix_still_tags():
    msgs = _format(
        "~qwen/qwen3-max",
        [
            TextMessage(role="system", text="Be helpful"),
            TextMessage(role="user", text="Hello"),
        ],
    )
    assert msgs[0]["content"][-1]["cache_control"] == _CC
    assert msgs[-1]["content"][-1]["cache_control"] == _CC


def test_implicit_cache_models_not_tagged():
    for model in ("openai/gpt-4o-mini", "deepseek/deepseek-chat"):
        msgs = _format(
            model,
            [
                TextMessage(role="system", text="Be helpful"),
                TextMessage(role="user", text="Hello"),
            ],
        )
        assert "cache_control" not in msgs[0]["content"][-1]
        assert "cache_control" not in msgs[-1]["content"][-1]


def test_system_only_tagged():
    msgs = _format("anthropic/claude-3-haiku", [TextMessage(role="system", text="Be helpful")])
    assert len(msgs) == 1
    assert msgs[0]["content"][-1]["cache_control"] == _CC


def test_no_system_tags_last_message():
    msgs = _format("anthropic/claude-3-haiku", [TextMessage(role="user", text="Hello")])
    assert len(msgs) == 1
    assert msgs[0]["content"][-1]["cache_control"] == _CC


def test_empty_messages_returns_empty():
    msgs = _format("anthropic/claude-3-haiku", [])
    assert msgs == []
