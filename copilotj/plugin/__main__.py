# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from collections.abc import Callable
from io import BytesIO

import click
import rich
import textual_image.renderable
from PIL import Image as PILImage

from copilotj.core import load_config
from copilotj.plugin import HTTPPluginAPI, Verbosity
from copilotj.plugin.api import ClientPluginAPI
from copilotj.util import extract_base64_image


@click.group()
@click.option("--server", default="http://127.0.0.1:8786", help="ImageJ plugin server address")
@click.option("--verbosity", type=click.Choice(["low", "normal", "high"]), default="normal")
@click.pass_context
def cli(ctx, server, verbosity):
    """CLI for interacting with the ImageJ plugin API."""
    load_config()
    ctx.ensure_object(dict)
    ctx.obj["server"] = server
    match verbosity:
        case "low":
            ctx.obj["verbosity"] = Verbosity.NORMAL
        case "normal":
            ctx.obj["verbosity"] = Verbosity.NORMAL
        case "high":
            ctx.obj["verbosity"] = Verbosity.HIGH
        case _:
            ctx.obj["verbosity"] = Verbosity.NORMAL


def run(get_api: Callable[[ClientPluginAPI], Callable], ctx, *args, **kwargs):
    async def runner():
        apis = HTTPPluginAPI(ctx.obj["server"])
        client_apis = apis.attach_single_client()
        api = get_api(client_apis)
        try:
            response = await api(*args, **kwargs)
        finally:
            await apis.close()
        return response

    loop = asyncio.new_event_loop()
    response = loop.run_until_complete(runner())
    loop.close()

    if callable(getattr(response, "describe")):
        resp = response.describe(level=1, verbosity=ctx.obj["verbosity"])
    else:
        resp = str(response)

    console = rich.console.Console()
    console.print(resp)


@cli.command()
@click.option("--ref", type=str, required=True, help="Component ref handle, e.g. e9")
@click.option("--action", type=str, required=True, help="Action short id, e.g. click")
@click.option("--parameters", type=str, default=None, help="JSON-encoded parameter list")
@click.pass_context
def call_action(ctx, ref, action, parameters):
    """Run an action on a component identified by its ref handle."""
    parsed = json.loads(parameters) if parameters else None
    run(lambda a: a.call_action, ctx, ref, action, parsed)


@cli.command()
@click.option("--title", type=str, default=None, help="Title of the image")
@click.pass_context
def capture_image(ctx, title):
    """Capture the image from ImageJ."""

    def capture_image(client: ClientPluginAPI):
        async def wrapper(title: str):
            result = await client.capture_image(title)
            b64 = extract_base64_image(result.image)
            image = PILImage.open(BytesIO(base64.b64decode(b64)))

            console = rich.console.Console()
            console.print(textual_image.renderable.Image(image))
            return result

        return wrapper

    run(capture_image, ctx, title)


@cli.command()
@click.pass_context
def capture_screen(ctx):
    """Capture the current screen from ImageJ."""

    def capture_screen(client: ClientPluginAPI):
        async def wrapper():
            result = await client.capture_screen()

            console = rich.console.Console()
            for screen in result.screenshots:
                b64 = extract_base64_image(screen.image)
                image = PILImage.open(BytesIO(base64.b64decode(b64)))
                console.print(textual_image.renderable.Image(image))

            return result

        return wrapper

    run(capture_screen, ctx)


@cli.command()
@click.pass_context
@click.argument("early", type=int, required=True)  # ID of the early snapshot
@click.argument("later", type=int, required=False, default=None)  # ID of the later snapshot
def compare_snapshots(ctx, early, later):
    """Compare two snapshots from ImageJ."""
    run(lambda a: a.compare_snapshots, ctx, early, later)


@cli.command()
@click.pass_context
@click.argument("since", type=click.DateTime(), required=False, default=None)
def get_operation_history(ctx, since):
    """Get the operation history from ImageJ."""
    run(lambda a: a.get_operation_history, ctx, since)


@cli.command()
@click.pass_context
def take_snapshot(ctx):
    """Take a snapshot of the current ImageJ state."""
    run(lambda a: a.take_snapshot, ctx)


@cli.command()
@click.option("--language", default="macro", help="Scripting language")
@click.option("--script", type=str, default=None, help="Path to the script file")
@click.option("--script-file", type=str, default=None, help="Path to the script file")
@click.pass_context
def run_script(ctx, language, script, script_file):
    """Run a script in ImageJ."""
    # If a script file is provided, read its content
    if script_file:
        with open(script_file, "r") as f:
            script = f.read()

    assert script or script_file, "Either --script or --script-file must be provided"
    run(lambda a: a.run_script, ctx, language, script)


@cli.command()
@click.pass_context
def summarise_environment(ctx):
    """Get a summary of the current ImageJ environment."""
    run(lambda a: a.summarise_environment, ctx)


cli()
