# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ActionResponse, ComponentBase, str_or_empty

__all__ = ["TextAreaNode", "TextAreaSetTextResponse"]


class TextAreaNode(ComponentBase[Literal["java.awt.TextArea"]]):
    text: str | None  # text can be null if component.getText() is null

    @override
    def role(self) -> str:
        return "textarea"

    @override
    def _state_inline(self) -> str | None:
        return f"text={str_or_empty(self.text)}"


type TextAreaSetTextResponse = ActionResponse[Literal["java.awt.TextArea.setText"], None]
