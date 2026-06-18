# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeaderDriven.abort() propagation.

abort() must reach the leader AND every specialized agent, because the agent
currently mid-stream (and possibly mid-retry-backoff) may be an executor, not
the leader — each holds its own abort event.
"""

from copilotj.multiagent.leader_multiagent import LeaderDriven


class _AbortRecorder:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class _NoAbortAgent:
    """Agent-like object without an abort() method — exercises the hasattr guard."""


def test_abort_propagates_to_leader_and_executors():
    # Bypass LeaderDriven.__init__ (needs cfg/apis/model_client); abort() only
    # touches leader_agent and _agents.
    pattern = LeaderDriven.__new__(LeaderDriven)
    leader = _AbortRecorder()
    exec_a = _AbortRecorder()
    exec_b = _AbortRecorder()
    pattern.leader_agent = leader
    pattern._agents = {"a": exec_a, "b": exec_b, "c": _NoAbortAgent()}  # type: ignore[attr-defined]

    pattern.abort()

    assert leader.aborted
    assert exec_a.aborted
    assert exec_b.aborted
