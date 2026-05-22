# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, override

from copilotj.plugin._base import Request, Response, Verbosity
from copilotj.plugin.awt import SnapshotDifference

__all__ = ["ScriptResult", "ScriptRequest"]

TIP_RESULT_NONE = """\
No result was returned, but the script likely ran successfully.\
"""

TIP_NO_OUTPUTS = """\
But due to the limitation of our ImageJ plugin, some outputs and errors may not be detected. If you are not sure, you \
can kindly ask user check the 'Debug' panel.\
"""


class ScriptResult(Response):
    outputs: dict[str, Any] | None
    err: str | None
    snapshot: SnapshotDifference | None = None

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        heading = "#" * level

        lines = []
        lines.append(f"{heading} Results")

        if self.outputs is not None and len(self.outputs) > 0:
            if len(self.outputs) == 1 and "result" in self.outputs:
                lines.append(self.outputs["result"] or TIP_RESULT_NONE)
            else:
                lines.append("Outputs:")
                for k, v in self.outputs.items():
                    lines.append(f"- {k}: {v}")

        elif self.err:
            lines.append(f"Error: {self.err}")

        else:
            tip = "No output detected."
            if (self.outputs is None or len(self.outputs) == 0) or (
                self.snapshot is None or not self.snapshot.any_changed
            ):
                tip += " " + TIP_NO_OUTPUTS

            lines.append(tip)

        return lines


class ScriptRequest(Request[ScriptResult], event="run_script", response_type=ScriptResult, timeout=16):
    language: str
    script: str
    timeout: int
