# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import click
from rich.console import Console
from rich.prompt import Prompt

from copilotj.core.config import Config, load_config
from copilotj.core.message import TextMessage
from copilotj.core.model_client import ToolCall, new_model_client
from copilotj.core.tool import FunctionTool


@click.command()
@click.option("--model", default=None, help="The OpenAI model to use.")
@click.option("--api-key", default=None, help="The OpenAI API key.")
@click.option("--proxy", default=None, help="The proxy to use.")
@click.option("--stream", is_flag=True, help="Whether to stream the response.")
def cli(model, api_key, proxy, stream):
    cfg: Config = load_config()

    console = Console()

    async def run():
        client = new_model_client(cfg)

        def get_temperature(city: str):
            return 15

        tools = []
        tools.append(FunctionTool(get_temperature, "Get current temperature"))

        messages = []

        while True:
            role = Prompt.ask(
                "Select a role (or type 'exit' to quit)",
                choices=["user", "system", "assistant", "exit"],
                console=console,
            )
            if role.lower() == "exit":
                break

            content = Prompt.ask("Enter your message", console=console)
            messages.append(TextMessage(role=role, text=content))  # type: ignore
            if role != "user":
                continue  # Only allow user to send messages

            if stream:
                model_stream = client.create_stream(messages=messages, tools=tools)
                async for chunk in model_stream:
                    if isinstance(chunk, ToolCall):
                        console.print(f"Tool Call: {chunk.tool.name}, args: {chunk.args}")
                        result = await chunk.run()
                        console.print(f"Tool Call Result: {result}")
                        messages.append(TextMessage(role="assistant", text=str(result)))
                    else:
                        console.print(chunk.content or "", end="")

                console.print()  # Add a newline after the stream

            else:
                completion = await client.create(messages=messages, tools=tools)
                print(completion)
                console.print(completion.content)

                if completion.tool_calls:
                    for tool_call in completion.tool_calls:
                        console.print(f"Tool Call: {tool_call.tool.name}, args: {tool_call.args}")
                        result = await tool_call.run()
                        console.print(f"Tool Call Result: {result}")
                        messages.append(TextMessage(role="assistant", text=str(result)))

    asyncio.run(run())


cli()
