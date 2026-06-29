# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ActionResponse, ComponentBase

__all__ = ["ButtonNode", "ButtonClickResponse"]


class ButtonNode(ComponentBase[Literal["java.awt.Button"]]):
    label: str

    @override
    def role(self) -> str:
        return "button"

    @override
    def _node_name(self) -> str | None:
        return self.label


ButtonClickResponse = ActionResponse[Literal["java.awt.Button.click"], None]
