# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ComponentBase

__all__ = ["ScrollbarNode"]


class ScrollbarNode(ComponentBase[Literal["java.awt.Scrollbar"]]):
    value: int
    orientation: Literal["horizontal", "vertical"]
    minimum: int = 0
    maximum: int = 0

    @override
    def role(self) -> str:
        return "scrollbar"

    @override
    def _state_inline(self) -> str | None:
        return f"value={self.value}"
