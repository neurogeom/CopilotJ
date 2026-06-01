#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copilotj.multiagent.tools as tools  # noqa: E402
from copilotj.core import FunctionTool, ToolCall, load_env  # noqa: E402
from copilotj.multiagent.leader_prompts import (  # noqa: E402
    PROMPT_TOOL_EXECUTE_PYTHON_SCRIPT,
    PROMPT_TOOL_FOLDER_SUMMARY,
    PROMPT_TOOL_IMAGEJ_PERCEPTION,
    PROMPT_TOOL_LABEL_IMAGE,
    PROMPT_TOOL_RUN_MACRO,
)
from copilotj.plugin.api import HTTPPluginAPI  # noqa: E402
from copilotj.workflow.contract import parse_interface  # noqa: E402
from copilotj.workflow.executor import WorkflowExecutor  # noqa: E402
from copilotj.workflow.manager import BASE_DIR, Workflow, WorkflowMeta, WorkflowStep, read_json  # noqa: E402

DEFAULT_SERVER = "http://127.0.0.1:8786"


def main() -> int:
    try:
        workflow = prompt_workflow()
        inputs = prompt_inputs(workflow)
        server = prompt_text("CopilotJ server", DEFAULT_SERVER)
        results = asyncio.run(execute_workflow(workflow, inputs, server))
        print(format_results(results))
        return 1 if any(not result["ok"] for result in results) else 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def prompt_workflow() -> Workflow:
    workflows = list_workflows()
    if workflows:
        print("Available workflows:")
        for index, workflow_path in enumerate(workflows, 1):
            workflow = load_workflow_path(workflow_path)
            print(f"  {index}. {workflow.meta.name} ({workflow.meta.id})")
    answer = prompt_text("Choose workflow number, id, directory, or workflow.json")
    workflow_ref = resolve_workflow_ref(answer, workflows)
    return load_workflow(workflow_ref)


def list_workflows() -> list[Path]:
    if not BASE_DIR.exists():
        return []
    return sorted(path / "workflow.json" for path in BASE_DIR.iterdir() if (path / "workflow.json").exists())


def resolve_workflow_ref(answer: str, workflows: list[Path]) -> str:
    if answer.isdigit() and workflows:
        index = int(answer)
        if 1 <= index <= len(workflows):
            return str(workflows[index - 1])
        raise ValueError(f"Workflow selection {index} is out of range.")
    return answer


def load_workflow(workflow_ref: str) -> Workflow:
    path = Path(workflow_ref).expanduser()
    if path.is_dir():
        return load_workflow_path(path / "workflow.json")
    if path.exists():
        return load_workflow_path(path)
    return load_workflow_path(BASE_DIR / workflow_ref / "workflow.json")


def load_workflow_path(path: Path) -> Workflow:
    raw = read_json(path)
    return Workflow(
        meta=meta_from_raw(raw),
        steps=[WorkflowStep(id=step["id"], action=step["action"]) for step in raw.get("steps", [])],
        schema_version=raw.get("schema_version", "2.0"),
        interface=parse_interface(raw),
        dataset_pattern=raw.get("dataset_pattern"),
        outputs=raw.get("outputs"),
    )


def meta_from_raw(raw: dict[str, Any]) -> WorkflowMeta:
    meta = raw.get("meta", {})
    return WorkflowMeta(
        id=meta.get("id"),
        name=meta.get("name"),
        version=meta.get("version", "1.0"),
        about=meta.get("about"),
        tags=meta.get("tags"),
        source=meta.get("source", "derived"),
        created_at=meta.get("created_at"),
        updated_at=meta.get("updated_at"),
    )


def prompt_inputs(workflow: Workflow) -> dict[str, Any]:
    inputs = {}
    specs = workflow.interface.inputs if workflow.interface else {}
    inputs.update(prompt_file_inputs(specs))
    if "output_dir" in specs:
        default_output = str(BASE_DIR / "runs" / workflow.meta.id)
        inputs["output_dir"] = prompt_text("Output directory", default_output)
    print("Optional overrides: enter name=value, blank line to continue.")
    while override := input("> ").strip():
        name, value = parse_override(override)
        inputs[name] = value
    return inputs


def prompt_file_inputs(specs: dict[str, Any]) -> dict[str, str]:
    names = [name for name, spec in specs.items() if isinstance(spec, dict) and spec.get("type") == "file"]
    return {name: prompt_text(f"Input file/folder for '{name}'") for name in names}


def parse_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected name=value.")
    name, value = override.split("=", 1)
    return name.strip(), value.strip()


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{label}{suffix}: ").strip()
    if answer:
        return answer
    if default is not None:
        return default
    raise ValueError(f"{label} is required.")


async def execute_workflow(workflow: Workflow, inputs: dict[str, Any], server: str) -> list[dict[str, Any]]:
    load_env()
    api = HTTPPluginAPI(server)
    try:
        executor = WorkflowExecutor(ScriptWorkflowAgent(api.attach_dev_client()))
        return await executor.execute_workflow(workflow, stop_on_error=True, inputs=inputs)
    finally:
        await api.close()


def format_results(results: list[dict[str, Any]]) -> str:
    lines = ["Workflow execution results:"]
    for result in results:
        lines.append(format_result(result))
    return "\n".join(lines)


def format_result(result: dict[str, Any]) -> str:
    if not result["ok"]:
        return f"FAIL step {result['step_id']}: {result['action'].get('name', 'unknown')} - {result['error']}"
    if result["step_id"] == "outputs":
        return f"OK outputs: {json.dumps(result['result'], ensure_ascii=False)}"
    return f"OK step {result['step_id']}: {result['action'].get('name', 'unknown')}"


class ScriptWorkflowAgent:
    def __init__(self, api):
        self.plugin_tools = tools.PluginTools(api)
        self.tools = [
            FunctionTool(self.plugin_tools.imagej_perception, PROMPT_TOOL_IMAGEJ_PERCEPTION),
            FunctionTool(self.plugin_tools.run_macro, PROMPT_TOOL_RUN_MACRO),
            FunctionTool(tools.folder_summary, PROMPT_TOOL_FOLDER_SUMMARY),
            FunctionTool(tools.execute_python_script, PROMPT_TOOL_EXECUTE_PYTHON_SCRIPT),
            FunctionTool(self.plugin_tools.label_image, PROMPT_TOOL_LABEL_IMAGE),
        ]

    async def _call_tool(self, tool_call: ToolCall) -> Any:
        return await tool_call.run()


if __name__ == "__main__":
    raise SystemExit(main())
