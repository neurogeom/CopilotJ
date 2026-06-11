# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from copilotj.core.agent import Agent
from copilotj.core.runtime import Runtime
from copilotj.core.ui import UI, DialogChange

__all__ = ["Pattern"]


class Pattern:
    def __init__(self, name: str, *, ui: UI | None = None) -> None:
        super().__init__()
        self._runtime = Runtime(name, ui=ui)
        self._agents: dict[str, Agent] = {}

    #########################
    #      Rigistration     #
    #########################

    def register(self, agent: Agent) -> None:
        """Register an agent with the pattern.

        Args:
            agent: The agent to register

        Raises:
            ValueError: If an agent with the same name is already registered
        """
        name = agent.name
        if name in self._agents and self._agents[name] is not agent:
            raise ValueError(f"Agent '{name}' already registered")

        agent._set_runtime(self._runtime)
        self._agents[name] = agent

    def _agent(self, name: str) -> Agent:
        """Get a registered agent by name."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' not registered")

        return self._agents[name]

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, Agent):
            self.register(value)

        super().__setattr__(name, value)

    #########################
    #          UI           #
    #########################

    async def request_user_confirm(self, message: str) -> bool:
        """Request user confirmation through the UI.

        Args:
            message: The confirmation message to display

        Returns:
            approval: True if user confirms, False otherwise
        """
        return await self._runtime.request_user_confirm(message)

    async def dialog_changed(self, dialog_id: int, state: Literal["completed"]) -> None:
        await self._runtime.print_dialog_change(DialogChange(id=str(dialog_id), state=state))

    def log_info(self, message: str) -> None:
        self._runtime.log_info(message)

    def log_error(self, message: str) -> None:
        self._runtime.log_error(message)

    #########################
    #       Messaging       #
    #########################

    async def _send_message(self, name: str, message: Any) -> None:
        """Send a message to a specific agent by name.

        Args:
            name: The name of the agent to send the message to
            message: The message to send (type should match what the agent expects)

        Raise:
            ValueError: If no agent with the given name is registered
        """
        if name not in self._agents:
            raise ValueError(f"No agent registered with name: {name}")

        agent = self._agents[name]
        await agent.on_message(message)
        raise NotImplementedError("Not implemented")
