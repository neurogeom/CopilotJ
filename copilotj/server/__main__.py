# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the server from the command line."""

import logging

import click

from copilotj.core import load_config
from copilotj.server import Server


@click.command()
@click.option("--host", default="127.0.0.1", help="Host address to bind to")
@click.option("--port", default=8786, help="Port number to listen on")
@click.option(
    "--log-level",
    default="DEBUG",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    help="Set the logging level",
)
def cli(host: str, port: int, log_level: str):
    """Start the CopilotJ server."""
    cfg = load_config()
    # resolve_vision_config deferred to async startup (avoids blocking sync download)
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    server = Server(cfg)
    server.run(host, port)


if __name__ == "__main__":
    cli()
