# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from copilotj.mcp.server import mcp

__all__ = ["mcp", "run"]


def run() -> None:
    """Entry point for the copilotj-mcp CLI command."""
    mcp.run()
