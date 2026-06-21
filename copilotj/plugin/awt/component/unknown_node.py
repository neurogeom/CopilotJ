# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import override

from copilotj.plugin.awt._base import ComponentBase

__all__ = ["UnknownNode"]


class UnknownNode(ComponentBase[str]):
    @override
    def role(self) -> str:
        return "unknown"

    @override
    def _node_name(self) -> str | None:
        return self.name
