# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import openai

from copilotj.core import langfuse_compat


def test_langfuse_disabled_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert langfuse_compat.is_langfuse_enabled() is False


def test_new_langfuse_client_falls_back_to_noop(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")

    client = langfuse_compat.new_langfuse_client()

    assert isinstance(client, langfuse_compat.NoOpLangfuse)
    with client.start_as_current_observation(name="test"):
        pass


def test_new_langfuse_client_falls_back_to_noop_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    client = langfuse_compat.new_langfuse_client()

    assert isinstance(client, langfuse_compat.NoOpLangfuse)
    with client.start_as_current_observation(name="test"):
        pass


def test_new_async_openai_client_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "secret-key")

    client = langfuse_compat.new_async_openai_client(api_key="test-key")

    assert isinstance(client, openai.AsyncOpenAI)


def test_new_async_openai_client_falls_back_to_openai_without_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "1")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    client = langfuse_compat.new_async_openai_client(api_key="test-key")

    assert isinstance(client, openai.AsyncOpenAI)
