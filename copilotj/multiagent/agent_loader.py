# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import importlib
import logging
import os
import tomllib

from copilotj.core import FunctionTool, ModelClient, Tool
from copilotj.multiagent.Executor import Executor

__all__ = ["load_agent_configs"]

_log = logging.getLogger(__name__)

GLOB_PATTERN = os.path.join(os.path.dirname(__file__), "agent_configs", "*_agent.toml")


def load_agent_configs(*, model_client: ModelClient):
    return _load_agent_configs(GLOB_PATTERN, model_client=model_client)


def _load_agent_configs(glob_pattern: str, *, model_client: ModelClient):
    agents = {}
    configs = glob.glob(glob_pattern)
    _log.info("Found %d configs in %s.", len(configs), glob_pattern)
    for file in configs:
        try:
            with open(file, "rb") as f:
                config = tomllib.load(f)

        except Exception as e:
            _log.error("Failed to load %s: %s", file, e)
            continue

        # Skip files that are commented out or empty
        if not config or config == "pass":
            _log.info("Skipping %s: empty or commented out", file)
            continue

        if "name" not in config or "class" not in config:
            _log.warning("Configuration file %s is missing required 'name' or 'class' field", file)
            continue

        module_class_str = config["class"]
        module_name, class_name = module_class_str.rsplit(".", 1)
        _log.info("Loading agent class %s from module %s...", class_name, module_name)
        try:
            module = importlib.import_module(module_name)
            agent_class = getattr(module, class_name)
            assert issubclass(agent_class, Executor), f"{agent_class} is not a subclass of Executor"

            name = config["name"]
            description = config.get("description", "")
            prompt = config.get("prompt", "")
            agent_tools_config = config.get("tools", [])

            # Load and prepare tools with descriptions
            agent_tools: list[Tool] = []

            for tool_conf in agent_tools_config:
                tool_name = tool_conf.get("name")
                tool_display_name = tool_conf.get("display_name")
                tool_description = tool_conf.get("description", f"Tool for {tool_name}")

                if "function" in tool_conf:
                    func_full = tool_conf["function"]
                    mod_name, func_name = func_full.rsplit(".", 1)
                    mod = importlib.import_module(mod_name)
                    fn = getattr(mod, func_name)
                    agent_tools.append(
                        FunctionTool(fn, tool_description, name=tool_name, display_name=tool_display_name)
                    )

                elif "class" in tool_conf:
                    class_full = tool_conf["class"]
                    mod_name, tool_class_name = class_full.rsplit(".", 1)
                    mod = importlib.import_module(mod_name)
                    tool_class = getattr(mod, tool_class_name)
                    assert issubclass(tool_class, Tool), f"{tool_class} is not a subclass of Tool"
                    agent_tools.append(tool_class())

                else:
                    _log.warning("Tool configuration for %s is missing 'function' or 'class' field", tool_name)

            _log.info("Loaded tools for %s: %s", name, list(agent_tools))

            # Create agent instance with tool descriptions
            agents[name] = agent_class(
                name=name, description=description, prompt=prompt, tools=agent_tools, model_client=model_client
            )
            _log.info("Successfully loaded agent: %s", name)

        except Exception as e:
            _log.error("Error loading agent from %s: %s", file, e, exc_info=True)

    return agents


if __name__ == "__main__":
    from copilotj.core import load_env, new_model_client

    load_env()

    # Test: Load agent configurations and print each agent's tool list
    print("Loading agent configurations...")
    agent_configs = os.path.join(os.path.dirname(__file__), "agent_configs")
    agents = load_agent_configs(model_client=new_model_client())
    for name, agent in agents.items():
        print(f"Loaded agent: {name}")
        print(
            "Agent tools:",
            getattr(agent, "tools", "No tools attribute"),
            getattr(agent, "tool_descriptions", "No tool descriptions"),
        )
