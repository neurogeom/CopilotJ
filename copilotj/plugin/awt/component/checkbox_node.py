# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ActionResponse, ComponentBase

__all__ = ["CheckboxNode", "CheckboxSetStateResponse"]


class CheckboxNode(ComponentBase[Literal["java.awt.Checkbox"]]):
    label: str | None
    state: bool

    @override
    def role(self) -> str:
        return "checkbox"

    @override
    def _node_name(self) -> str | None:
        return self.label

    @override
    def _state_inline(self) -> str | None:
        return f"checked={str(self.state).lower()}"


type CheckboxSetStateResponse = ActionResponse[Literal["java.awt.Checkbox.setState"], None]
