# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ComponentBase, format_items, str_or_empty

__all__ = ["ListNode"]


class ListNode(ComponentBase[Literal["java.awt.List"]]):
    items: list[str]
    selected_item: str | None  # can be null in Java AWT List

    @override
    def role(self) -> str:
        return "list"

    @override
    def _state_inline(self) -> str | None:
        return f"selected={str_or_empty(self.selected_item)} items={format_items(self.items)}"
