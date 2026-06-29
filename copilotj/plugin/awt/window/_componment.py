# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import override

from copilotj.plugin._base import Verbosity
from copilotj.plugin.awt.component.button_node import ButtonNode
from copilotj.plugin.awt.component.canvas_node import CanvasNode
from copilotj.plugin.awt.component.choice_node import ChoiceNode
from copilotj.plugin.awt.component.scrollbar_node import ScrollbarNode
from copilotj.plugin.awt.component.text_area_node import TextAreaNode
from copilotj.plugin.awt.component.text_field_node import TextFieldNode
from copilotj.plugin.awt.container.container_node import ContainerNodeBase

__all__ = [
    "Buttons",
    "CanvasWithLabel",
    "ChoiceWithLabel",
    "ScrollbarWithLabel",
    "TextAreaWithLabel",
    "TextFieldWithLabel",
]


class Buttons(ContainerNodeBase[str]):
    """A transparent group of buttons: renders its children inline (no group head line)."""

    def __init__(self, *children: ButtonNode):
        assert all(isinstance(a, ButtonNode) for a in children), "All children must be ButtonNode instances"
        super().__init__(type="copilotj.Buttons", name="Buttons", is_container=True, children=list(children))

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        # Transparent: emit each button's line directly so they appear at the
        # parent's indent level (the parent adds the 2-space indent).
        lines: list[str] = []
        for child in self.children or []:
            lines.extend(child._describe(level=level, verbosity=verbosity))
        return lines


class CanvasWithLabel(CanvasNode):
    label: str

    @override
    def _node_name(self) -> str | None:
        return self.label


class ChoiceWithLabel(ChoiceNode):
    label: str

    @override
    def _node_name(self) -> str | None:
        return self.label


class ScrollbarWithLabel(ScrollbarNode):
    label: str

    @override
    def _node_name(self) -> str | None:
        return self.label


class TextAreaWithLabel(TextAreaNode):
    label: str

    @override
    def _node_name(self) -> str | None:
        return self.label


class TextFieldWithLabel(TextFieldNode):
    label: str

    @override
    def _node_name(self) -> str | None:
        return self.label
