# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from copilotj.mcp.server import mcp

__all__ = ["mcp", "run"]


def run() -> None:
    """Entry point for the copilotj-mcp CLI command."""
    parser = argparse.ArgumentParser(description="CopilotJ MCP server")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Listen on 0.0.0.0 (accessible from other machines)",
    )
    args, remaining = parser.parse_known_args()

    kwargs: dict = {}
    if args.public:
        kwargs["host"] = "0.0.0.0"
        kwargs["transport"] = "http"

    mcp.run(**kwargs)
