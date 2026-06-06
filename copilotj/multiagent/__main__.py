# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from copilotj.core import load_config
from copilotj.multiagent.leader_multiagent import LeaderDriven
from copilotj.plugin.api import HTTPPluginAPI, PluginAPI


async def main():
    load_config()

    apis: PluginAPI = HTTPPluginAPI("http://127.0.0.1:8786")
    client_apis = apis.attach_single_client()

    # Initialize the main orchestrator
    orchestrator = LeaderDriven(client_apis)
    print("CopilotJ Multi-Agent System Initialized. Enter your task or 'exit'.")
    while True:
        try:
            task = input("> ")
            if task.lower() == "exit":
                print("Catch you later!")
                break

            if not task:
                continue

            # Run the task using the LeaderDriven orchestrator
            await orchestrator.run(task)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            # Optionally add more robust error handling or logging

    await apis.close()


asyncio.run(main())
