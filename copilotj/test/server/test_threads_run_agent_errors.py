# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _Thread._run_agent crash surfacing (decision B + Codex Q3 sanitization).

- PluginNotConnectedError -> a UIEventError carrying the curated DEFAULT_MESSAGE.
- Any other exception -> a UIEventError with the generic message only; the exception
  text (which may contain provider payloads / file paths / script content) must NOT
  leak to the UI. Full detail still goes to logs via log_error.
"""

import asyncio

from copilotj.core.ui import UIEventError
from copilotj.plugin.api import PluginNotConnectedError
from copilotj.server.threads import GENERIC_AGENT_ERROR_MSG, _Thread


class _FakeAgent:
    def __init__(self, exc: BaseException | None) -> None:
        self._exc = exc
        self.log_errors: list[str] = []

    async def run(self, _prompt: str, *, trace_ctx: object = None) -> None:  # noqa: ARG002
        if self._exc is not None:
            raise self._exc

    def log_error(self, message: str) -> None:
        self.log_errors.append(message)


def _make_thread(agent: _FakeAgent) -> tuple[_Thread, list[UIEventError]]:
    thread = _Thread.__new__(_Thread)
    thread.thread_id = "t-1"
    thread._trace_ctx = None  # type: ignore[attr-defined]
    thread._agent = agent  # type: ignore[attr-defined]

    sent: list[UIEventError] = []

    async def _send(event: UIEventError) -> None:
        sent.append(event)

    thread.send = _send  # type: ignore[method-assign]
    return thread, sent


def test_run_agent_surfaces_plugin_not_connected_with_curated_message():
    thread, sent = _make_thread(_FakeAgent(PluginNotConnectedError()))
    done = asyncio.Event()

    asyncio.run(thread._run_agent("hi", done))

    assert done.is_set()
    assert len(sent) == 1
    assert sent[0].data == PluginNotConnectedError.DEFAULT_MESSAGE
    assert sent[0].role == "system"


def test_run_agent_sanitizes_unknown_crash():
    secret = ValueError("boom at /tmp/secret with token xyz and ValueError")
    thread, sent = _make_thread(_FakeAgent(secret))
    agent = thread._agent  # type: ignore[attr-defined]
    done = asyncio.Event()

    asyncio.run(thread._run_agent("hi", done))

    assert done.is_set()
    assert len(sent) == 1
    # Generic message only — no leakage.
    assert sent[0].data == GENERIC_AGENT_ERROR_MSG
    assert "secret" not in sent[0].data
    assert "ValueError" not in sent[0].data
    assert "token" not in sent[0].data
    # Full detail preserved in logs.
    assert len(agent.log_errors) == 1
    assert "secret" in agent.log_errors[0]


def test_run_agent_no_event_when_run_succeeds():
    thread, sent = _make_thread(_FakeAgent(None))
    done = asyncio.Event()

    asyncio.run(thread._run_agent("hi", done))

    assert done.is_set()
    assert sent == []
