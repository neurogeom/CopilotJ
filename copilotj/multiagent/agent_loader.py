# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import filecmp
import glob
import hashlib
import importlib
import json
import logging
import shutil
import tomllib
from datetime import date
from pathlib import Path

from copilotj.core import FunctionTool, ModelClient, Tool
from copilotj.core.config import Config, get_home
from copilotj.multiagent.Executor import Executor

__all__ = ["load_agent_configs", "sync_agent_configs"]

_log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_TOML_GLOB = "*_agent.toml"
# Per-file baseline: maps seed filename -> sha256 of the shipped content at last sync.
# A whole-dir marker would refresh EVERY customized file on any seed bump; per-file
# tracking means a release that changes only one config only touches that one config.
_SEED_BASELINE_FILE = ".seed-versions.json"
_synced = False


def load_agent_configs(*, model_client: ModelClient, cfg: Config):
    _sync_agent_configs_once()
    home_configs = get_home() / "agents"
    return _load_agent_configs(str(home_configs / _AGENT_TOML_GLOB), model_client=model_client, cfg=cfg)


def _sync_agent_configs_once() -> None:
    """Run :func:`sync_agent_configs` at most once per process.

    ``load_agent_configs`` is called per LeaderDriven pattern (once per chat thread);
    the sync is idempotent but we avoid re-hashing the seed on every call.
    """
    global _synced
    if _synced:
        return
    sync_agent_configs()
    _synced = True


def sync_agent_configs(source: Path | None = None, target: Path | None = None) -> None:
    """Seed and refresh ``$COPILOTJ_HOME/agents`` from the bundled defaults.

    Per-file dpkg-style refresh. The seed baseline (``.seed-versions.json``) records
    the shipped sha256 of each config at last sync. On each run a config is touched
    ONLY if its shipped content changed since last sync (``baseline[name] != new``):
    the user's current copy is backed up as ``<name>.bak.YYYYMMDD`` and the new default
    applied. Configs whose seed is unchanged are left entirely alone — so a release
    that edits one agent does not revert a user's customization of a different agent.
    New seed files are copied in; user-added configs are never touched.
    ``.bak.YYYYMMDD`` files don't match the ``*_agent.toml`` glob, so backups are never
    loaded as agents.

    In dev mode (``source`` and ``target`` resolve to the same path, i.e.
    ``COPILOTJ_HOME`` is the repo root) this is a no-op beyond ensuring the directory
    exists — developers edit the source seed directly.
    """
    home = get_home()
    source = _PROJECT_ROOT / "agents" if source is None else source
    target = home / "agents" if target is None else target
    target.mkdir(parents=True, exist_ok=True)
    # Dev mode (home == repo root) or no seed available: nothing to sync.
    if source.resolve() == target.resolve() or not source.exists():
        return
    baseline = _load_seed_baseline(target / _SEED_BASELINE_FILE)
    today = date.today().strftime("%Y%m%d")
    seed_names: set[str] = set()
    dirty = False
    for src_file in _seed_files(source):
        seed_names.add(src_file.name)
        new_cs = hashlib.sha256(src_file.read_bytes()).hexdigest()
        if baseline.get(src_file.name) == new_cs:
            continue  # This file's default is unchanged -> never touch the user's copy.
        dst_file = target / src_file.name
        if dst_file.exists() and not filecmp.cmp(src_file, dst_file, shallow=False):
            _backup_before_overwrite(dst_file, target, today)
        if not dst_file.exists() or not filecmp.cmp(src_file, dst_file, shallow=False):
            shutil.copy2(src_file, dst_file)
        baseline[src_file.name] = new_cs
        dirty = True
    # Surface configs that no longer ship (removed upstream) or are user-added.
    for dst_file in target.glob(_AGENT_TOML_GLOB):
        if dst_file.name not in seed_names:
            _log.debug(
                "Agent config %s is not in the shipped defaults (user-added or removed upstream).",
                dst_file.name,
            )
    if dirty:
        _save_seed_baseline(target / _SEED_BASELINE_FILE, baseline)


def _seed_files(source: Path):
    """Yield seed files (sorted, files only) — single source of truth for refresh + glob."""
    for f in sorted(source.glob("*")):
        if f.is_file():
            yield f


def _load_seed_baseline(marker: Path) -> dict[str, str]:
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_seed_baseline(marker: Path, baseline: dict[str, str]) -> None:
    marker.write_text(json.dumps(baseline, sort_keys=True, indent=2), encoding="utf-8")


def _backup_before_overwrite(live: Path, target: Path, today: str) -> None:
    """Save ``live`` to ``<name>.bak.YYYYMMDD`` before it is overwritten.

    Never loses a distinct state: if a same-date backup already holds the exact
    current content, skip (already preserved); otherwise write a collision-avoiding
    name (``.bak.YYYYMMDD``, ``.bak.YYYYMMDD.2``, ``.bak.YYYYMMDD.3``, ...).
    """
    candidate = target / f"{live.name}.bak.{today}"
    n = 1
    while candidate.exists():
        if filecmp.cmp(candidate, live, shallow=False):
            return  # this exact content is already backed up today
        n += 1
        candidate = target / f"{live.name}.bak.{today}.{n}"
    shutil.copy2(live, candidate)
    _log.info(
        "Backed up customized %s to %s; applied new default. Re-apply your edits from the .bak if needed.",
        live.name,
        candidate.name,
    )


def _load_agent_configs(glob_pattern: str, *, model_client: ModelClient, cfg: Config):
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

        # Skip files that are empty
        if not config:
            _log.info("Skipping %s: empty", file)
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

                elif "factory" in tool_conf:
                    factory_full = tool_conf["factory"]
                    mod_name, factory_name = factory_full.rsplit(".", 1)
                    mod = importlib.import_module(mod_name)
                    factory_fn = getattr(mod, factory_name)
                    fn = factory_fn(cfg)
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
                    _log.warning(
                        "Tool configuration for %s is missing 'function', 'factory', or 'class' field", tool_name
                    )

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
    from copilotj.core import load_config, new_model_client

    cfg = load_config()

    # Test: Load agent configurations and print each agent's tool list
    print("Loading agent configurations...")
    agents = load_agent_configs(model_client=new_model_client(cfg), cfg=cfg)
    for name, agent in agents.items():
        print(f"Loaded agent: {name}")
        print(
            "Agent tools:",
            getattr(agent, "tools", "No tools attribute"),
            getattr(agent, "tool_descriptions", "No tool descriptions"),
        )
