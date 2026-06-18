# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that each provider's ``_to_*_provider_error`` helper preserves the
HTTP status code + Retry-After so the agent retry loop can decide retryability.
"""

from types import SimpleNamespace

from copilotj.core.model_client.anthropic import _to_anthropic_provider_error
from copilotj.core.model_client.gemini import _to_gemini_provider_error
from copilotj.core.model_client.openai_chat_completion import _to_openai_provider_error


def _openai_like(message: str, status_code: int, retry_after: str | None = None) -> SimpleNamespace:
    """Build an object shaped like an openai/anthropic SDK status error."""
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return SimpleNamespace(message=message, status_code=status_code, response=SimpleNamespace(headers=headers))


def test_openai_helper_extracts_status_and_retry_after():
    err = _to_openai_provider_error(_openai_like("rate limited", 429, retry_after="5"))
    assert err.status_code == 429
    assert err.retry_after == 5.0
    assert err.is_retryable()
    assert err.provider == "openai"


def test_openai_helper_provider_override():
    # Ollama uses the same helper with a different provider label.
    err = _to_openai_provider_error(_openai_like("rate limited", 429), provider="ollama")
    assert err.provider == "ollama"
    assert err.is_retryable()


def test_openai_helper_non_retryable_status():
    err = _to_openai_provider_error(_openai_like("bad request", 400))
    assert err.status_code == 400
    assert not err.is_retryable()


def test_openai_helper_plain_exception_has_no_status():
    err = _to_openai_provider_error(ValueError("no sdk attrs"))
    assert err.status_code is None
    assert err.retry_after is None
    assert not err.is_retryable()


def test_anthropic_helper_extracts_status_and_retry_after():
    err = _to_anthropic_provider_error(_openai_like("rate limited", 429, retry_after="2"))
    assert err.status_code == 429
    assert err.retry_after == 2.0
    assert err.is_retryable()
    assert err.provider == "anthropic"


def test_gemini_helper_uses_code_as_status():
    # google-genai errors carry the HTTP status as ``.code``.
    err = _to_gemini_provider_error(SimpleNamespace(message="resource exhausted", code=429))
    assert err.status_code == 429
    assert err.is_retryable()
    assert err.provider == "gemini"


def test_gemini_helper_without_code_not_retryable():
    err = _to_gemini_provider_error(RuntimeError("something broke"))
    assert err.status_code is None
    assert not err.is_retryable()
