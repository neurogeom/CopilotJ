# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for reusing the leader's cached conversation as a prompt-cache prefix.

Covers the snapshot primitive and the ``_generate_dialog_summary`` rewrite (#2):
when a dialog is cached, the summary call appends a short tail to a snapshot of
``_dialog_messages`` (so the conversation prefix is a cache hit) instead of
sending a brand-new standalone prompt. When no dialog is cached yet, it falls
back to the legacy re-serialized-steps prompt.
"""

import asyncio
from types import SimpleNamespace

from copilotj.core.message import TextMessage
from copilotj.core.model_client._types import ModelResponse
from copilotj.multiagent.leader_multiagent import LeaderDriven


def _stub_leaderdriven(dialog_messages, model_response_content="summary text"):
    """Build a ``LeaderDriven`` with only the attributes that
    ``_leader_dialog_snapshot`` / ``_generate_dialog_summary`` touch.

    Bypasses the heavy ``__init__`` (which loads agent configs, model client,
    plugin APIs, etc.) — we only need ``leader_agent``, ``model_client``, and
    the two log helpers.
    """
    ld = LeaderDriven.__new__(LeaderDriven)
    ld.leader_agent = SimpleNamespace(dialog_messages=list(dialog_messages))

    captured: dict = {}

    async def fake_create(messages, **_kwargs):
        captured["messages"] = list(messages)
        return ModelResponse(
            reasoning_content=None,
            content=model_response_content,
            tool_calls=None,
            finish_reason="stop",
        )

    ld.model_client = SimpleNamespace(create=fake_create)
    ld.log_info = lambda *_a, **_k: None
    ld.log_error = lambda *_a, **_k: None
    return ld, captured


# --------------------------------------------------------------------------- #
# _leader_dialog_snapshot
# --------------------------------------------------------------------------- #


def test_snapshot_is_a_shallow_copy_not_an_alias():
    ld, _ = _stub_leaderdriven([TextMessage(role="user", text="hi")])
    live = ld.leader_agent.dialog_messages
    snap = ld._leader_dialog_snapshot()
    assert snap == live
    assert snap is not live  # new list, so callers can't mutate the leader's state
    snap.append(TextMessage(role="user", text="mutate"))
    assert ld.leader_agent.dialog_messages == [TextMessage(role="user", text="hi")]


def test_snapshot_empty_before_any_dialog():
    ld, _ = _stub_leaderdriven([])
    assert ld._leader_dialog_snapshot() == []


# --------------------------------------------------------------------------- #
# _generate_dialog_summary (#2)
# --------------------------------------------------------------------------- #


def _user_ended_dialog() -> list[TextMessage]:
    """A dialog that ends in a user turn, like a real ReAct trace."""
    return [
        TextMessage(role="system", text="sys"),
        TextMessage(role="user", text="count cells"),
        TextMessage(role="assistant", text="Thought: ..."),
        TextMessage(role="user", text="Observation: 42 cells"),
    ]


def test_summary_uses_snapshot_prefix_when_dialog_cached():
    dialog = _user_ended_dialog()
    ld, captured = _stub_leaderdriven(dialog)
    steps = [{"thought": "do thing", "name": "run_macro", "response": "ok"}]

    result = asyncio.run(ld._generate_dialog_summary({"task": "count cells", "steps": steps}))

    msgs = captured["messages"]
    # The cached conversation is reused verbatim as the prefix ...
    assert msgs[: len(dialog)] == dialog
    # ... followed by exactly one appended user tail.
    assert len(msgs) == len(dialog) + 1
    assert msgs[-1].role == "user"
    assert "count cells" in msgs[-1].text
    # The tail must NOT re-serialize the steps JSON (that's the whole point).
    assert "run_macro" not in msgs[-1].text
    assert '"thought"' not in msgs[-1].text
    assert result == "summary text"


def test_summary_falls_back_to_standalone_prompt_when_snapshot_empty():
    ld, captured = _stub_leaderdriven([])
    steps = [{"thought": "do thing", "name": "run_macro", "response": "ok"}]

    asyncio.run(ld._generate_dialog_summary({"task": "count cells", "steps": steps}))

    msgs = captured["messages"]
    # Legacy path: a single user message whose body embeds the steps JSON.
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert "count cells" in msgs[0].text
    assert "run_macro" in msgs[0].text  # steps_text present in fallback


def test_summary_returns_none_on_empty_model_response():
    ld, _ = _stub_leaderdriven(_user_ended_dialog(), model_response_content=None)
    result = asyncio.run(ld._generate_dialog_summary({"task": "t", "steps": []}))
    assert result is None
