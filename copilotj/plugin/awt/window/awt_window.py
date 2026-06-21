# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import override

from copilotj.plugin._base import Response, Verbosity
from copilotj.plugin.awt.container import ContainerNodeBase

__all__ = ["AwtWindowBase", "AwtWindowDifferenceBase", "AwtWindowDifference"]


class AwtWindowBase[T: str](ContainerNodeBase[T]):
    id: int

    @override
    def role(self) -> str:
        return "window"

    @override
    def _node_name(self) -> str | None:
        # Windows usually expose their title via a `title` field (e.g. IjImage,
        # IjTextWindow); those override _node_name. Others fall back to the AWT
        # component name, which may be None.
        return self.name

    @override
    def _head_extras(self) -> str | None:
        return f"(id={self.id})"


class AwtWindowDifferenceBase[T: str](Response):
    id: int
    type: T


class AwtWindowDifference(AwtWindowDifferenceBase[str]):
    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        return [f"id={self.id}, type={self.type}"]
