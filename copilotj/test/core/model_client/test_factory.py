# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model client factory (_resolve_client / new_model_client)."""

from __future__ import annotations

import pytest

from copilotj.core.model_client import OpenAIChatCompletionClient, OpenRouterChatCompletionClient, _resolve_client
from copilotj.core.model_client._types import _VALID_PROVIDERS


def _base_url(client) -> str:
    """Read the resolved base URL off the underlying OpenAI client (trailing-slash agnostic)."""
    return str(client._client.base_url).rstrip("/")


class TestResolveClient:
    def test_openrouter_registered(self):
        assert "openrouter" in _VALID_PROVIDERS

    def test_openrouter_default_base_url(self):
        client = _resolve_client("openrouter", "anthropic/claude-3-haiku", "k", proxy=None, base_url=None)
        assert isinstance(client, OpenRouterChatCompletionClient)
        assert isinstance(client, OpenAIChatCompletionClient)  # subclass relationship
        assert _base_url(client) == "https://openrouter.ai/api/v1"
        assert client.get_model() == "anthropic/claude-3-haiku"

    def test_openrouter_custom_base_url(self):
        client = _resolve_client("openrouter", "m", "k", proxy=None, base_url="https://custom.example/v1")
        assert _base_url(client) == "https://custom.example/v1"

    def test_deepseek_default_base_url(self):
        # Regression guard: DeepSeek keeps working alongside the new provider.
        client = _resolve_client("deepseek", "deepseek-chat", "k", proxy=None, base_url=None)
        assert _base_url(client) == "https://api.deepseek.com"

    def test_siliconflow_default_base_url(self):
        client = _resolve_client("siliconflow", "Qwen/Qwen2.5-7B-Instruct", "k", proxy=None, base_url=None)
        assert _base_url(client) == "https://api.siliconflow.cn/v1"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            _resolve_client("not-a-provider", "m", "k", proxy=None, base_url=None)
