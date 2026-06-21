# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the playwright-mcp-style AWT snapshot YAML format.

Covers: per-element ref handles, inline short action ids, the `- role "name"
[ref=eN] (actions): state` line grammar, 2-space indentation of children,
labels carrying no ref, and the transparent `Buttons` group.
"""

from typing import Literal, override

from copilotj.plugin._base import Verbosity
from copilotj.plugin.awt.component import ButtonNode, CheckboxNode, ChoiceNode, LabelNode, ScrollbarNode
from copilotj.plugin.awt.window._componment import Buttons
from copilotj.plugin.awt.window.awt_window import AwtWindowBase

CLICK = [{"type": "java.awt.Button.click", "name": "Click", "description": "d", "parameters": []}]
SELECT = [{"type": "java.awt.Choice.selectItem", "name": "Select", "description": "d", "parameters": []}]
SET_STATE = [{"type": "java.awt.Checkbox.setState", "name": "Set", "description": "d", "parameters": []}]
SET_VALUE = [{"type": "java.awt.Scrollbar.setValue", "name": "Set", "description": "d", "parameters": []}]


def _ref(ref: str) -> str:
    """Mirror ComponentBase._yaml_head's ref token: f'[ref={ref}]'."""
    return f"[ref={ref}]"


class _TestWindow(AwtWindowBase[Literal["test.Window"]]):
    """Minimal concrete window for exercising the inherited container rendering."""

    @override
    def _node_name(self) -> str | None:
        return "Test"


def test_button_line():
    b = ButtonNode(type="java.awt.Button", name="ap", ref="e9", label="Apply", actions=CLICK)
    assert b._yaml_line() == f'- button "Apply" {_ref("e9")} (click)'


def test_choice_line_with_items():
    c = ChoiceNode(
        type="java.awt.Choice",
        name=None,
        ref="e5",
        items=["Default", "Huang", "RenyiEntropy"],
        selected_item="RenyiEntropy",
        actions=SELECT,
    )
    assert c._yaml_line() == (
        f'- choice {_ref("e5")} (selectItem): selected="RenyiEntropy" items=["Default", "Huang", "RenyiEntropy"]'
    )


def test_label_has_no_ref_and_raw_text():
    label = LabelNode(type="java.awt.Label", name=None, text="99.33 %")
    assert label.ref is None
    assert label._yaml_line() == "- label: 99.33 %"


def test_checkbox_and_scrollbar_lines():
    ch = CheckboxNode(
        type="java.awt.Checkbox", name=None, ref="e7", label="Dark background", state=False, actions=SET_STATE
    )
    assert ch._yaml_line() == f'- checkbox "Dark background" {_ref("e7")} (setState): checked=false'

    sb = ScrollbarNode(
        type="java.awt.Scrollbar",
        name=None,
        ref="e3",
        value=0,
        orientation="horizontal",
        minimum=0,
        maximum=255,
        actions=SET_VALUE,
    )
    assert sb._yaml_line() == f"- scrollbar {_ref('e3')} (setValue): value=0"


def test_buttons_are_transparent():
    """Buttons emits each button inline at the parent level (no group head line)."""
    btns = Buttons(
        ButtonNode(type="java.awt.Button", name="a", ref="e9", label="Auto", actions=CLICK),
        ButtonNode(type="java.awt.Button", name="p", ref="e10", label="Apply", actions=CLICK),
    )
    out = "\n".join(btns._describe(level=1, verbosity=Verbosity.NORMAL))
    assert out == f'- button "Auto" {_ref("e9")} (click)\n- button "Apply" {_ref("e10")} (click)'


def test_window_tree_two_space_indent():
    """A window renders a `- window ... (id=N):` header and nests children 2 spaces."""
    win = _TestWindow(
        type="test.Window",
        name=None,
        ref="e1",
        id=1,
        is_container=True,
        children=[
            ScrollbarNode(
                type="java.awt.Scrollbar",
                name=None,
                ref="e3",
                value=0,
                orientation="horizontal",
                minimum=0,
                maximum=255,
                actions=SET_VALUE,
            ),
            LabelNode(type="java.awt.Label", name=None, text="99.33 %"),
            ButtonNode(type="java.awt.Button", name="a", ref="e9", label="Apply", actions=CLICK),
        ],
    )
    out = "\n".join(win._describe(level=1, verbosity=Verbosity.NORMAL))
    assert out == "\n".join(
        [
            f'- window "Test" {_ref("e1")} (id=1):',
            f"  - scrollbar {_ref('e3')} (setValue): value=0",
            "  - label: 99.33 %",
            f'  - button "Apply" {_ref("e9")} (click)',
        ]
    )


def test_low_verbosity_hides_children():
    win = _TestWindow(
        type="test.Window",
        name=None,
        ref="e1",
        id=1,
        is_container=True,
        children=[ButtonNode(type="java.awt.Button", name="a", ref="e9", label="Apply", actions=CLICK)],
    )
    out = "\n".join(win._describe(level=1, verbosity=Verbosity.LOW))
    assert out == f'- window "Test" {_ref("e1")} (id=1)'
