# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeaderDriven.run() syntax-error counter reset.

`syntax_error_counter` must track CONSECUTIVE ReAct syntax errors, not
cumulative ones: a model that recovers (produces a valid response) between
errors must have its counter reset, mirroring `tool_retry_counter` at L850.

Control-flow fact these tests rely on (verified by trace + outside-voice
review): every ModelSyntaxError increment in run() leaves `agent_resp = None`,
which routes recovery through the single `send_correction` at ~L782. That site
is the only place a reset is load-bearing.
"""

import asyncio

from copilotj.core import ModelSyntaxError
from copilotj.multiagent.leader_multiagent import LeaderDriven

_MAX_SYNTAX_ERRORS = 3


class _Args:
    def model_dump(self) -> dict:
        return {}


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _ToolCall:
    def __init__(self, name: str) -> None:
        self.tool = _Tool(name)
        self.args = _Args()


class _Resp:
    """Minimal stand-in for ModelResponse exercised by run()."""

    def __init__(self, *, content=None, reasoning_content=None, tool_calls=None) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


def _syntax_err() -> ModelSyntaxError:
    return ModelSyntaxError("bad ReAct")


def _action() -> _Resp:
    """A valid parsed response that carries a tool call (no final answer)."""
    return _Resp(reasoning_content="thought", tool_calls=[_ToolCall("some_tool")])


def _final() -> _Resp:
    return _Resp(content="final answer")


class _ScriptedLeader:
    """LeaderAgent stub whose dialog methods pop a scripted queue.

    Each queue holds either a _Resp (returned) or a ModelSyntaxError (raised).
    """

    def __init__(self) -> None:
        self.begin_dialog_q: list = []
        self.send_correction_q: list = []
        self.continue_dialog_q: list = []
        self.call_tool_q: list = []

    @staticmethod
    def _pop(queue: list):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def begin_dialog(self, task, trace_ctx=None):
        return self._pop(self.begin_dialog_q)

    async def send_correction(self, message, trace_ctx=None):
        return self._pop(self.send_correction_q)

    async def continue_dialog(self, prior_response, observation, trace_ctx=None):
        return self._pop(self.continue_dialog_q)

    async def _call_tool(self, tool_call):
        return self._pop(self.call_tool_q)


async def _noop_dialog_changed(*args, **kwargs) -> None:
    pass


async def _drive(leader: _ScriptedLeader) -> str:
    """Run one LeaderDriven.run() dialog and return its final status.

    Bypasses LeaderDriven.__init__ (which needs cfg/apis/model_client); run()
    only touches the attributes wired up here.
    """
    pattern = LeaderDriven.__new__(LeaderDriven)
    pattern.dialog_counter = 1
    pattern._summarize_task = None  # type: ignore[attr-defined]
    pattern.log_info = lambda *a, **k: None  # type: ignore[assignment]
    pattern.log_error = lambda *a, **k: None  # type: ignore[assignment]
    pattern.dialog_changed = _noop_dialog_changed  # type: ignore[assignment]
    pattern.leader_agent = leader  # type: ignore[assignment]

    captured: dict = {}

    async def _record(*args) -> None:
        # _background_summarize_and_store(dialog_id, task, dialog_context)
        captured["ctx"] = args[-1]

    pattern._background_summarize_and_store = _record  # type: ignore[assignment]

    await pattern.run("do the thing")
    await pattern._summarize_task  # type: ignore[attr-defined]
    return captured["ctx"]["status"]


def test_cumulative_errors_with_recovery_complete():
    """>=3 total syntax errors, each recovered from, must NOT abort.

    Under the old cumulative counting this sequence aborted at the third error
    (status "failed"); with the reset-on-recovery it completes.
    """
    leader = _ScriptedLeader()
    leader.begin_dialog_q = [_syntax_err()]
    # Three recoveries at site (a), each producing a valid tool call:
    leader.send_correction_q = [_action(), _action(), _action()]
    leader.call_tool_q = ["ok", "ok", "ok"]
    # Two errors interleaved, then a final answer on the third continue:
    leader.continue_dialog_q = [_syntax_err(), _syntax_err(), _final()]

    status = asyncio.run(_drive(leader))

    assert status == "completed"


def test_consecutive_errors_still_abort():
    """Three consecutive syntax errors (no recovery) must still abort."""
    leader = _ScriptedLeader()
    leader.begin_dialog_q = [_syntax_err()]
    leader.send_correction_q = [_syntax_err(), _syntax_err()]

    status = asyncio.run(_drive(leader))

    assert status == "failed"
    # Sanity: we exercised the abort threshold, not some other failure path.
    assert len(leader.send_correction_q) == 0
