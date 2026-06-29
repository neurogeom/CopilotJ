# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast, override

from copilotj.plugin._base import Verbosity
from copilotj.plugin.awt.component import ButtonNode, CanvasNode, CheckboxNode, ChoiceNode, ScrollbarNode, TextFieldNode
from copilotj.plugin.awt.container import ContainerNode, TypedComponentNode
from copilotj.plugin.awt.window._componment import Buttons, CanvasWithLabel, ChoiceWithLabel, ScrollbarWithLabel
from copilotj.plugin.awt.window.awt_window import AwtWindowBase

__all__ = ["IjThresholdAdjuster"]


class IjThresholdAdjuster(AwtWindowBase[Literal["ij.plugin.frame.ThresholdAdjuster"]]):
    """Window of Image > Adjust > Threshold...

    Renders as a playwright-mcp-style YAML window: each widget carries a
    ``[ref=eN]`` handle and its short action id inline (e.g. ``(selectItem)``).
    See ``copilotj/test/plugin/test_awt_snapshot.py`` for the exact grammar.

    The raw AWT tree is flattened for readability: the histogram canvas and each
    scrollbar/choice get a descriptive label merged from the adjacent ``Label``,
    the redundant text fields mirroring the scrollbars are dropped, the two
    ``Choice`` widgets and the checkboxes are pulled out of their panels, and the
    four action buttons are inlined as a transparent ``Buttons`` group.
    """

    @override
    @classmethod
    def _describe_children(
        cls, children: list[TypedComponentNode] | None, *, level: int, verbosity: Verbosity
    ) -> list[str]:
        try:
            new_children = cls._convert_children(children)
            return super()._describe_children(new_children, level=level, verbosity=verbosity)
        except (AssertionError, IndexError):
            return super()._describe_children(children, level=level, verbosity=verbosity)

    @classmethod
    def _convert_children(cls, children: list[TypedComponentNode] | None) -> list[TypedComponentNode]:
        assert children is not None and len(children) == 9

        new_children = []

        # canvas
        canvas = children[0]
        assert isinstance(canvas, CanvasNode)
        new_children.append(CanvasWithLabel(label="Histogram Canvas", **canvas.model_dump()))

        # label
        new_children.append(children[1])

        # scrollbars
        # The text fields next to the scrollbars are redundant, so we'll discard them
        # and give the scrollbars descriptive label.
        # TODO: remove action of text field
        scrollbar1 = children[2]
        textfield1 = children[3]
        assert isinstance(scrollbar1, ScrollbarNode)
        assert isinstance(textfield1, TextFieldNode)
        new_children.append(ScrollbarWithLabel(label="Lower threshold", **scrollbar1.model_dump()))

        scrollbar2 = children[4]
        textfield2 = children[5]
        assert isinstance(scrollbar2, ScrollbarNode)
        assert isinstance(textfield2, TextFieldNode)
        new_children.append(ScrollbarWithLabel(label="Upper threshold", **scrollbar2.model_dump()))

        # Panel with choices
        choices = children[6]
        assert (
            isinstance(choices, ContainerNode)
            and choices.children is not None
            and len(choices.children) == 2
            and all(isinstance(c, ChoiceNode) for c in choices.children)
        )
        new_children.append(ChoiceWithLabel(label="Method", **choices.children[0].model_dump()))
        new_children.append(ChoiceWithLabel(label="Preview mode", **choices.children[1].model_dump()))

        # Panel with checkboxes
        checkboxes = children[7]
        assert (
            isinstance(checkboxes, ContainerNode)
            and checkboxes.children is not None
            and all(isinstance(c, CheckboxNode) for c in checkboxes.children)
        )
        new_children.extend(checkboxes.children)

        # Panel with buttons
        buttons = children[8]
        assert (
            isinstance(buttons, ContainerNode)
            and buttons.children is not None
            and all(isinstance(c, ButtonNode) for c in buttons.children)
        )
        new_children.append(Buttons(*cast(list[ButtonNode], buttons.children)))
        return new_children
