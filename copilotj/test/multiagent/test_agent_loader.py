# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
from pathlib import Path
from typing import Annotated
from unittest.mock import MagicMock

import pytest

from copilotj.core import FunctionTool
from copilotj.core.config import Config
from copilotj.multiagent import agent_loader as _loader_module
from copilotj.multiagent.agent_loader import _load_agent_configs, load_agent_configs, sync_agent_configs


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


# ---------------------------------------------------------------------------
# Tests — sync_agent_configs (dpkg-style refresh into $COPILOTJ_HOME/agents)
# ---------------------------------------------------------------------------

_AGENT_TOML = """
name = "Sync Agent"
class = "copilotj.test.multiagent.test_agent_loader._FakeExecutor"
description = ""
prompt = ""
"""


def _seed(src: Path, name: str, body: str) -> None:
    src.mkdir(parents=True, exist_ok=True)
    (src / name).write_text(textwrap.dedent(body), encoding="utf-8")


def test_sync_first_run_copies_seed(tmp_path):
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    assert (dst / "sync_agent.toml").read_text().strip() != ""
    assert (dst / ".seed-versions.json").exists()


def test_sync_dev_guard_when_source_is_target(tmp_path):
    src = tmp_path / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    # source and target resolve to the same path -> dev mode, no marker, no copy churn
    sync_agent_configs(source=src, target=src)
    assert not (src / ".seed-versions.json").exists()


def test_sync_unrelated_seed_change_preserves_unrelated_customization(tmp_path):
    """Codex P1: a seed bump to ONE file must not back up + revert a DIFFERENT file the
    user customized, when that other file's own default did not change."""
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "a_agent.toml", _AGENT_TOML)
    _seed(src, "b_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    # User customizes a_agent; then upstream ships a change to b_agent ONLY.
    (dst / "a_agent.toml").write_text("USER A", encoding="utf-8")
    (src / "b_agent.toml").write_text("# new b default\n", encoding="utf-8")
    sync_agent_configs(source=src, target=dst)
    assert (dst / "a_agent.toml").read_text() == "USER A"  # untouched: its seed didn't change
    assert not list(dst.glob("a_agent.toml.bak.*"))  # ...and not backed up
    assert (dst / "b_agent.toml").read_text() == "# new b default\n"  # b refreshed


def test_sync_noop_when_source_missing(tmp_path):
    dst = tmp_path / "home" / "agents"
    sync_agent_configs(source=tmp_path / "does_not_exist", target=dst)
    assert dst.exists()  # target is still ensured
    assert not (dst / ".seed-versions.json").exists()


def test_sync_marker_match_preserves_user_edit(tmp_path):
    """When the shipped seed is unchanged, a user's live edit is NOT touched."""
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)  # initial seed + marker
    # User now edits the live file.
    (dst / "sync_agent.toml").write_text("USER CUSTOMIZATION", encoding="utf-8")
    sync_agent_configs(source=src, target=dst)  # marker still matches -> no-op
    assert (dst / "sync_agent.toml").read_text() == "USER CUSTOMIZATION"
    assert not list(dst.glob("*.bak.*"))  # no backup created


def test_sync_seed_changed_backs_up_user_edit(tmp_path):
    """On upgrade, a customized file is backed up (.bak.YYYYMMDD) then replaced."""
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    # User customizes; then upstream ships a new default.
    (dst / "sync_agent.toml").write_text("USER CUSTOMIZATION", encoding="utf-8")
    (src / "sync_agent.toml").write_text("# new default\n", encoding="utf-8")
    sync_agent_configs(source=src, target=dst)
    backups = list(dst.glob("sync_agent.toml.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "USER CUSTOMIZATION"  # edit preserved
    assert (dst / "sync_agent.toml").read_text() == "# new default\n"  # new default live


def test_sync_same_day_second_refresh_preserves_second_edit(tmp_path):
    """A second seed bump the same day must back up a NEW user edit, not silently
    overwrite it just because a .bak from the first refresh already exists."""
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    # First customization + first same-day seed bump -> .bak captures it.
    (dst / "sync_agent.toml").write_text("USER EDIT 1", encoding="utf-8")
    (src / "sync_agent.toml").write_text("# default v2\n", encoding="utf-8")
    sync_agent_configs(source=src, target=dst)
    # Second customization + second same-day seed bump -> must NOT be lost.
    (dst / "sync_agent.toml").write_text("USER EDIT 2", encoding="utf-8")
    (src / "sync_agent.toml").write_text("# default v3\n", encoding="utf-8")
    sync_agent_configs(source=src, target=dst)
    backup_contents = sorted(p.read_text() for p in dst.glob("sync_agent.toml.bak.*"))
    assert "USER EDIT 1" in backup_contents
    assert "USER EDIT 2" in backup_contents  # second edit preserved, not overwritten
    assert (dst / "sync_agent.toml").read_text() == "# default v3\n"


def test_sync_identical_file_not_backed_up(tmp_path):
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    # Second sync with identical seed -> no backup, no change.
    sync_agent_configs(source=src, target=dst)
    assert not list(dst.glob("*.bak.*"))


def test_sync_new_seed_file_copied(tmp_path):
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    _seed(src, "extra_agent.toml", _AGENT_TOML)
    sync_agent_configs(source=src, target=dst)
    assert (dst / "extra_agent.toml").exists()


def test_sync_orphan_logged_at_debug(tmp_path, caplog):
    src, dst = tmp_path / "seed", tmp_path / "home" / "agents"
    _seed(src, "sync_agent.toml", _AGENT_TOML)
    dst.mkdir(parents=True)
    (dst / "ghost_agent.toml").write_text("orphan", encoding="utf-8")  # not in seed
    import logging as _logging

    with caplog.at_level(_logging.DEBUG, logger=_loader_module._log.name):
        sync_agent_configs(source=src, target=dst)
    assert any("ghost_agent.toml" in r.message for r in caplog.records)


def test_load_agent_configs_reads_from_home(tmp_path, monkeypatch):
    """Regression: the public loader reads from $COPILOTJ_HOME/agents, not __file__."""
    monkeypatch.setenv("COPILOTJ_HOME", str(tmp_path))
    monkeypatch.setattr(_loader_module, "_synced", False)
    monkeypatch.setattr(_loader_module, "sync_agent_configs", lambda *a, **k: None)  # don't copy real seed
    cfgs = tmp_path / "agents"
    cfgs.mkdir()
    _seed(cfgs, "home_agent.toml", _AGENT_TOML.replace("Sync Agent", "Home Agent"))
    agents = load_agent_configs(model_client=_make_model_client(), cfg=_make_cfg())
    assert "Home Agent" in agents
