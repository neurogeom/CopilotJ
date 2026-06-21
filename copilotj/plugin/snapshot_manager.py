# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from copilotj.plugin._base import Request
from copilotj.plugin.awt import SnapshotDifference
from copilotj.plugin.awt.action import TypedActionResponse

__all__ = ["CompareSnapshotRequest"]


class CompareSnapshotRequest(Request[SnapshotDifference], event="compare_snapshots", response_type=SnapshotDifference):
    id_early: int
    id_later: int | None


class ActionRequest(Request[Any], event="run_action", response_type=TypedActionResponse):
    ref: str
    action: str
    parameters: list[Any] | None
