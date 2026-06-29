# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ComponentBase

__all__ = ["CanvasNode"]


class CanvasNode(ComponentBase[Literal["java.awt.Canvas"]]):
    @override
    def role(self) -> str:
        return "canvas"
