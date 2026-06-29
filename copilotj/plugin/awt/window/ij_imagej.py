# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin._base import Verbosity
from copilotj.plugin.awt.window.awt_window import AwtWindowBase

__all__ = ["IjImageJ"]


class IjImageJ(AwtWindowBase[Literal["ij.ImageJ"]]):
    @override
    def _node_name(self) -> str | None:
        return "ImageJ"

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        # The main toolbar's individual buttons aren't useful to enumerate, so
        # render just the window header line.
        return [self._yaml_head()]
