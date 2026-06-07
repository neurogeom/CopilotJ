# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

from copilotj.core import Pattern, load_config
from copilotj.core.model_client import new_model_client
from copilotj.multiagent.agent_loader import load_agent_configs


class LeaderDriven(Pattern):
    def __init__(self) -> None:
        super().__init__("copilotj.leader_driven")
        self.dialog_counter = 1

        # Load all agents from agent_configs and store in a dictionary
        cfg = load_config()
        self.agents = load_agent_configs(model_client=new_model_client(cfg=cfg), cfg=cfg)
        for agent in self.agents.values():
            self.register(agent)

    async def run_agent(self, agent_name: str, task: str):
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
        return await agent.run(task)

    async def test_all_agents(self):
        results = {}
        for name, agent in self.agents.items():
            results[name] = await agent.run("test")
        return results


if __name__ == "__main__":
    from copilotj.core import load_config

    async def main():
        load_config()

        group = LeaderDriven()

        while True:
            agent_name = input("Enter agent name (or 'all' for testing all agents, 'exit' to quit): ")
            if agent_name.lower() == "exit":
                print("Catch you later!")
                break
            task = input("Task: ")
            if agent_name.lower() == "all":
                results = await group.test_all_agents()
                for name, output in results.items():
                    print(f"Output from {name}: {output}")
            else:
                result = await group.run_agent(agent_name, task)
                print(f"Output from {agent_name}: {result}")

    asyncio.run(main())
