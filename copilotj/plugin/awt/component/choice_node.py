# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ActionResponse, ComponentBase, format_items, str_or_empty

__all__ = ["ChoiceNode", "ChoiceSelectItemResponse"]


class ChoiceNode(ComponentBase[Literal["java.awt.Choice"]]):
    items: list[str]
    selected_item: str

    @override
    def role(self) -> str:
        return "choice"

    @override
    def _state_inline(self) -> str | None:
        return f"selected={str_or_empty(self.selected_item)} items={format_items(self.items)}"


type ChoiceSelectItemResponse = ActionResponse[Literal["java.awt.Choice.selectItem"], None]
