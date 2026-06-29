# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin.awt._base import ComponentBase

__all__ = ["LabelNode"]


class LabelNode(ComponentBase[Literal["java.awt.Label"]]):
    text: str

    @override
    def role(self) -> str:
        return "label"

    @override
    def _state_inline(self) -> str | None:
        # Raw, unquoted text (sanitized to one line, truncated); renders as
        # `- label: {text}`. Labels carry no ref and no actions.
        text = self.text if self.text is not None else "<empty>"
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        if len(text) > 300:
            text = text[:300] + "..."
        return text
