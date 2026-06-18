# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the 429 auto-retry behaviour added to ChatAgent._create (#96).

Drives async code with ``asyncio.run`` (no pytest-asyncio), matching the rest of
the suite. A flaky stub ModelClient raises ModelProviderError(429) on demand; a
recording stub Runtime captures the print_retry / print_error / print_chat calls
so we can assert what the user would see.
"""

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Sequence
from typing import Any, override

import pytest

from copilotj.core.agent import ChatAgent
from copilotj.core.message import ImageMessage, TextMessage
from copilotj.core.model_client import ModelClient, ModelProviderError, ModelResponseChunk
from copilotj.core.model_client._retry import DEFAULT_RETRY_CONFIG
from copilotj.core.tool import Tool
from copilotj.core.ui import RetryInfo


class _NullTrace:
    """No-op Langfuse stand-in so tests don't depend on Langfuse being configured."""

    def start_as_current_observation(self, **_kwargs: Any) -> Any:
        return contextlib.nullcontext()


class _RecordingRuntime:
    """Duck-typed Runtime that records retry/error/chat emissions."""

    def __init__(self) -> None:
        self.retries: list[tuple[str, RetryInfo]] = []
        self.errors: list[tuple[str, str]] = []
        self.chunks: list[Any] = []

    async def update_current_agent(self, agent: str) -> None:  # noqa: ARG002
        pass

    async def print_chat(self, agent: str, message: Any) -> None:  # noqa: ARG002
        self.chunks.append(message)

    async def print_retry(self, role: str, info: RetryInfo) -> None:
        self.retries.append((role, info))

    async def print_error(self, role: str, message: str) -> None:
        self.errors.append((role, message))

    def log_info(self, message: str) -> None:  # noqa: ARG002
        pass

    def log_error(self, message: str) -> None:  # noqa: ARG002
        pass


class _FlakyStreamClient(ModelClient):
    """ModelClient that simulates 429 rate limiting.

    - ``fail_before`` attempts raise a pre-stream ModelProviderError (nothing
      yielded yet → safe to retry).
    - When ``raise_after_chunk`` is set, the next successful attempt yields one
      chunk and THEN raises (mid-stream → must NOT be retried).
    - ``retry_after`` is stamped on the raised error; 0.0 keeps backoff instant.
    """

    def __init__(
        self,
        *,
        fail_before: int = 0,
        raise_after_chunk: bool = False,
        status_code: int = 429,
        retry_after: float = 0.0,
    ) -> None:
        self._fail_before = fail_before
        self._raise_after_chunk = raise_after_chunk
        self._status_code = status_code
        self._retry_after = retry_after
        self.calls = 0

    @override
    def get_model(self) -> str:
        return "flaky"

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
    ) -> Any:
        raise NotImplementedError

    @override
    def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk, None]:
        return self._stream()

    async def _stream(self) -> AsyncGenerator[ModelResponseChunk, None]:
        self.calls += 1
        if self.calls <= self._fail_before:
            raise ModelProviderError(
                f"rate limited (attempt {self.calls})",
                "flaky",
                status_code=self._status_code,
                retry_after=self._retry_after,
            )
        if self._raise_after_chunk:
            yield ModelResponseChunk(reasoning_content=None, content="partial", finish_reason=None)
            raise ModelProviderError("rate limited mid-stream", "flaky", status_code=self._status_code)
        yield ModelResponseChunk(reasoning_content=None, content="hello", finish_reason=None)
        yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason="stop")


def _make_agent(client: ModelClient) -> tuple[ChatAgent, _RecordingRuntime]:
    agent = ChatAgent("leader", "desc", model_client=client)
    runtime = _RecordingRuntime()
    agent._set_runtime(runtime)  # noqa: SLF001
    return agent, runtime


def test_retry_then_success():
    client = _FlakyStreamClient(fail_before=2)
    agent, runtime = _make_agent(client)

    completion = asyncio.run(agent._create(TextMessage(role="user", text="hi"), trace_ctx=_NullTrace()))

    assert client.calls == 3  # 2 failed + 1 success
    assert len(runtime.retries) == 2
    assert [info.attempt for _, info in runtime.retries] == [1, 2]
    assert all(info.max_attempts == DEFAULT_RETRY_CONFIG.max_attempts for _, info in runtime.retries)
    assert runtime.errors == []
    assert completion.content == "hello"
    assert completion.finish_reason == "stop"


def test_retry_exhaustion_surfaces_error():
    # More failures than max_attempts → never succeeds.
    client = _FlakyStreamClient(fail_before=DEFAULT_RETRY_CONFIG.max_attempts + 5)
    agent, runtime = _make_agent(client)

    completion = asyncio.run(agent._create(TextMessage(role="user", text="hi"), trace_ctx=_NullTrace()))

    assert client.calls == DEFAULT_RETRY_CONFIG.max_attempts
    # Retries signalled for every attempt except the last (which gives up).
    assert len(runtime.retries) == DEFAULT_RETRY_CONFIG.max_attempts - 1
    assert len(runtime.errors) == 1
    # Empty completion on exhaustion (pre-existing behaviour, preserved).
    assert completion.content is None
    assert completion.finish_reason == "unknown"


def test_non_retryable_error_is_not_retried():
    client = _FlakyStreamClient(fail_before=1, status_code=400)
    agent, runtime = _make_agent(client)

    asyncio.run(agent._create(TextMessage(role="user", text="hi"), trace_ctx=_NullTrace()))

    assert client.calls == 1  # no retries
    assert runtime.retries == []
    assert len(runtime.errors) == 1


def test_midstream_error_is_not_retried():
    # A 429 AFTER a chunk has been streamed must not retry (would duplicate).
    client = _FlakyStreamClient(raise_after_chunk=True)
    agent, runtime = _make_agent(client)

    completion = asyncio.run(agent._create(TextMessage(role="user", text="hi"), trace_ctx=_NullTrace()))

    assert client.calls == 1  # mid-stream error → no retry
    assert runtime.retries == []
    assert len(runtime.errors) == 1
    # The partial content that streamed before the error is preserved.
    assert completion.content == "partial"


def test_abort_during_backoff_raises_cancelled():
    # retry_after=10s gives a real backoff window for abort to interrupt.
    client = _FlakyStreamClient(fail_before=99, retry_after=10.0)
    agent, runtime = _make_agent(client)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, agent.abort)
        await agent._create(TextMessage(role="user", text="hi"), trace_ctx=_NullTrace())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    # At least one retry was signalled before the abort landed.
    assert len(runtime.retries) >= 1
