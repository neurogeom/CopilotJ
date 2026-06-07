# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from typing import Annotated
from unittest.mock import MagicMock

import pytest

from copilotj.core import FunctionTool
from copilotj.core.config import Config
from copilotj.multiagent import agent_loader as _loader_module
from copilotj.multiagent.agent_loader import _load_agent_configs


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes so we don't import the real Executor / tools
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Minimal stand-in for Executor accepted by the loader.

    Patched in via the ``_patch_executor`` fixture so the ``issubclass``
    check in the loader accepts it.
    """

    def __init__(self, *, name, description, prompt, tools, model_client):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.tools = tools
        self.model_client = model_client


@pytest.fixture(autouse=True)
def _patch_executor(monkeypatch):
    """Replace ``Executor`` in the loader module with our fake."""
    monkeypatch.setattr(_loader_module, "Executor", _FakeExecutor)


def _write_toml(tmp_path, filename: str, content: str):
    """Write a TOML config file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def _make_model_client():
    return MagicMock()


def _make_cfg(**overrides):
    return Config(**overrides)


def _write_toml(tmp_path, filename: str, content: str):
    """Write a TOML config file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(p)


def _make_model_client():
    return MagicMock()


def _make_cfg(**overrides):
    return Config(**overrides)


# ---------------------------------------------------------------------------
# Tests — function field (existing behaviour)
# ---------------------------------------------------------------------------


def test_load_function_tool(tmp_path, monkeypatch):
    """Tools with ``function`` are imported directly and wrapped in FunctionTool."""
    # Use a real, importable function as the tool target
    toml = """
    name = "Test Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = "test"
    prompt = ""

    [[tools]]
    function = "copilotj.test.multiagent.test_agent_loader._sample_tool"
    description = "A sample tool."
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "test_agent.toml", toml)

    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert "Test Agent" in agents
    agent = agents["Test Agent"]
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], FunctionTool)


def test_function_tool_schema_exposes_params(tmp_path):
    """The JSON schema for a function tool must contain its parameters."""
    toml = """
    name = "Schema Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""

    [[tools]]
    function = "copilotj.test.multiagent.test_agent_loader._sample_tool"
    description = "desc"
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "schema_agent.toml", toml)

    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    tool = agents["Schema Agent"].tools[0]
    schema = tool.json_schema
    assert "query" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Tests — factory field (new behaviour)
# ---------------------------------------------------------------------------


def test_load_factory_tool(tmp_path):
    """Tools with ``factory`` are called with cfg; the returned callable is wrapped."""
    toml = """
    name = "Factory Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""

    [[tools]]
    factory = "copilotj.test.multiagent.test_agent_loader.make_sample_tool"
    description = "A factory-made tool."
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "factory_agent.toml", toml)

    cfg = _make_cfg(tavily_api_key="secret-123")
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=cfg)
    assert "Factory Agent" in agents
    agent = agents["Factory Agent"]
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], FunctionTool)


def test_factory_tool_schema_hides_config(tmp_path):
    """A factory-produced tool must NOT expose cfg / api_key in its schema."""
    toml = """
    name = "Clean Schema Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""

    [[tools]]
    factory = "copilotj.test.multiagent.test_agent_loader.make_sample_tool"
    description = "desc"
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "clean_agent.toml", toml)

    cfg = _make_cfg(tavily_api_key="secret-123")
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=cfg)
    tool = agents["Clean Schema Agent"].tools[0]
    schema = tool.json_schema
    props = schema["parameters"]["properties"]
    assert "query" in props  # LLM-facing param present
    assert "cfg" not in props  # config param hidden
    assert "tavily_api_key" not in props  # config param hidden


def test_factory_tool_receives_config(tmp_path):
    """The closure returned by a factory must have access to the injected config."""

    async def _run():
        toml = """
        name = "Config Check Agent"
        class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
        description = ""
        prompt = ""

        [[tools]]
        factory = "copilotj.test.multiagent.test_agent_loader.make_echo_api_key"
        description = "Echoes the API key."
        """
        pattern = str(tmp_path / "*.toml")
        _write_toml(tmp_path, "cfg_check_agent.toml", toml)

        cfg = _make_cfg(tavily_api_key="my-secret-key")
        agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=cfg)
        tool = agents["Config Check Agent"].tools[0]

        args = tool.args_type()(query="test")
        result = await tool.run(args)
        assert result == "my-secret-key"

    import asyncio

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Tests — error handling & edge cases
# ---------------------------------------------------------------------------


def test_skip_empty_config(tmp_path):
    """Empty TOML files are skipped gracefully."""
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "empty_agent.toml", "")
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert agents == {}


def test_skip_missing_name_or_class(tmp_path):
    """Configs missing required fields are skipped."""
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "no_class.toml", 'name = "X"\n')
    _write_toml(tmp_path, "no_name.toml", 'class = "X.Y"\n')
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert agents == {}


def test_skip_invalid_toml(tmp_path):
    """Malformed TOML files are skipped, not crashing the loader."""
    pattern = str(tmp_path / "*.toml")
    p = tmp_path / "bad.toml"
    p.write_text("{{{{invalid", encoding="utf-8")
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert agents == {}


def test_skip_tool_missing_fields(tmp_path):
    """A tool entry with neither function, factory, nor class is skipped."""
    toml = """
    name = "Bad Tool Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""

    [[tools]]
    name = "mystery"
    description = "Has no function/factory/class"
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "bad_tool.toml", toml)

    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert "Bad Tool Agent" in agents
    assert agents["Bad Tool Agent"].tools == []


def test_multiple_tools_mixed_types(tmp_path):
    """An agent can have a mix of function, factory, and class tools."""
    toml = """
    name = "Mixed Agent"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""

    [[tools]]
    function = "copilotj.test.multiagent.test_agent_loader._sample_tool"
    description = "plain function"

    [[tools]]
    factory = "copilotj.test.multiagent.test_agent_loader.make_sample_tool"
    description = "factory function"
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "mixed.toml", toml)

    cfg = _make_cfg(tavily_api_key="key")
    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=cfg)
    agent = agents["Mixed Agent"]
    assert len(agent.tools) == 2
    assert all(isinstance(t, FunctionTool) for t in agent.tools)


def test_multiple_agents_loaded(tmp_path):
    """Multiple TOML files produce multiple agents."""
    toml_a = """
    name = "Agent A"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""
    """
    toml_b = """
    name = "Agent B"
    class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
    description = ""
    prompt = ""
    """
    pattern = str(tmp_path / "*.toml")
    _write_toml(tmp_path, "a_agent.toml", toml_a)
    _write_toml(tmp_path, "b_agent.toml", toml_b)

    agents = _load_agent_configs(pattern, model_client=_make_model_client(), cfg=_make_cfg())
    assert set(agents.keys()) == {"Agent A", "Agent B"}


# ---------------------------------------------------------------------------
# Dummy tool definitions used by the tests above
# ---------------------------------------------------------------------------


async def _sample_tool(query: Annotated[str, "Search query"]) -> str:
    """A trivial tool used for testing function loading."""
    return f"result: {query}"


def make_sample_tool(cfg: Config):
    """Factory that returns a closure with access to cfg."""

    async def _tool(query: Annotated[str, "Search query"]) -> str:
        return f"result: {query}, key={cfg.tavily_api_key}"

    return _tool


def make_echo_api_key(cfg: Config):
    """Factory that echoes the tavily_api_key — for verifying config binding."""

    async def _echo(query: Annotated[str, "Ignored"]) -> str:
        return cfg.tavily_api_key or ""

    return _echo
