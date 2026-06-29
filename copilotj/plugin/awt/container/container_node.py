# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Literal, override

from copilotj.plugin._base import Verbosity
from copilotj.plugin.awt._base import ComponentBase
from copilotj.plugin.awt.component import (
    ButtonNode,
    CanvasNode,
    CheckboxNode,
    ChoiceNode,
    LabelNode,
    ListNode,
    ScrollbarNode,
    TextAreaNode,
    TextFieldNode,
    UnknownNode,
)

__all__ = ["TypedComponentNode", "ContainerNodeBase", "ContainerNode"]

# NOTE: put typed node here to avoid cycle imports
type TypedComponentNode = "ButtonNode| CanvasNode | CheckboxNode| ChoiceNode| ContainerNode | LabelNode| ListNode | ScrollbarNode | TextAreaNode | TextFieldNode | UnknownNode"


class ContainerNodeBase[T: str](ComponentBase[T], ABC):
    is_container: Literal[True]  # Mark this class as a container, used for pydantic validation
    children: list[TypedComponentNode] | None

    @override
    def role(self) -> str:
        return "container"

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        """Render this container and its children as a 2-space-indented YAML tree."""
        has_children = bool(self.children) and verbosity >= Verbosity.NORMAL
        if not has_children:
            # No children to show: render like a leaf (own head + optional state).
            return [self._yaml_line()]

        # Container with children: head + ":" then children indented 2 spaces.
        lines = [self._yaml_head() + ":"]
        lines.extend(self._describe_children(self.children, level=level, verbosity=verbosity))
        return lines

    @classmethod
    def _describe_children(
        cls, children: list[TypedComponentNode] | None, *, level: int, verbosity: Verbosity
    ) -> list[str]:
        """Render each child's subtree and indent it one level (2 spaces).

        Windows override this to flatten/merge children (e.g. merge a Label into
        the following widget) before delegating to ``super()``.
        """
        lines: list[str] = []
        for child in children or []:
            for child_line in child._describe(level=level + 1, verbosity=verbosity):
                lines.append("  " + child_line)
        return lines


class ContainerNode(ContainerNodeBase[str]):
    pass
