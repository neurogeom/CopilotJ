# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Optional

from copilotj.workflow.contract import parse_interface
from copilotj.workflow.manager import Workflow, WorkflowMeta, WorkflowStep, slugify

ALLOWED_TEMPLATE_RE = re.compile(r"{{\s*(inputs\.[A-Za-z_][\w.]*|outputs\.[A-Za-z_]\w*\.path|run_dir)\s*}}")
ANY_TEMPLATE_RE = re.compile(r"{{\s*([^{}]+?)\s*}}")
ABSOLUTE_PATH_RE = re.compile(r"(^|[\s\"'=])(/Users/|/home/|/mnt/|[A-Za-z]:[\\/])")
ALLOWED_ACTION_NAMES = {
    "run_macro",
    "execute_python_script",
    "folder_summary",
    "imagej_perception",
    "label_image",
}


class DialogToWorkflowConverter:
    @staticmethod
    def extract_definition(steps_text: str) -> dict[str, Any] | list[dict[str, Any]]:
        if not isinstance(steps_text, str) or not steps_text.strip():
            raise ValueError("workflow steps must be a non-empty JSON string")

        json_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        match = re.search(json_pattern, steps_text, re.DOTALL)
        json_content = match.group(1).strip() if match else steps_text.strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            raise ValueError("steps is not a valid JSON string, please try to generate again")

    @staticmethod
    def extract_steps(steps_text: str) -> list[dict[str, Any]]:
        definition = DialogToWorkflowConverter.extract_definition(steps_text)
        steps = definition.get("steps") if isinstance(definition, dict) else definition
        if not isinstance(steps, list):
            raise ValueError("workflow steps must be a JSON array")
        return steps

    @staticmethod
    def create_workflow(
        workflow_name: str,
        steps_text: str,
        workflow_summary: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Workflow:
        definition = DialogToWorkflowConverter.extract_definition(steps_text)
        _validate_definition(definition)
        steps = definition["steps"]
        workflow_steps = [WorkflowStep(id=step["id"], action=step["action"]) for step in steps]
        meta = WorkflowMeta(
            id=slugify(workflow_name),
            name=workflow_name,
            about=workflow_summary or "Workflow created from summarized steps",
            tags=tags or "no tags",
            source="summarized",
        )
        return Workflow(
            meta=meta,
            steps=workflow_steps,
            schema_version=_schema_version(definition),
            interface=parse_interface(definition) if isinstance(definition, dict) else None,
            outputs=definition.get("outputs") if isinstance(definition, dict) else None,
        )


def _schema_version(definition: dict[str, Any] | list[dict[str, Any]]) -> str:
    if isinstance(definition, dict):
        return str(definition.get("schema_version", "2.0"))
    return "1.0"


def _validate_definition(definition: Any) -> None:
    if not isinstance(definition, dict):
        raise ValueError("workflow definition must be a JSON object with schema_version, interface, and steps")
    if definition.get("schema_version") != "2.0":
        raise ValueError('workflow schema_version must be exactly "2.0"')
    _validate_interface(definition.get("interface"))
    _validate_steps(definition.get("steps"))


def _validate_interface(interface: Any) -> None:
    if not isinstance(interface, dict):
        raise ValueError("workflow interface must be an object")
    if not isinstance(interface.get("inputs"), dict):
        raise ValueError("workflow interface.inputs must be an object")
    if not isinstance(interface.get("outputs"), dict):
        raise ValueError("workflow interface.outputs must be an object")
    _validate_templates(interface, "interface", allow_absolute_paths=False)


def _validate_steps(steps: Any) -> None:
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow steps must be a non-empty JSON array")
    seen_ids: set[int] = set()
    for index, step in enumerate(steps, 1):
        _validate_step(step, index, seen_ids)


def _validate_step(step: Any, index: int, seen_ids: set[int]) -> None:
    if not isinstance(step, dict):
        raise ValueError(f"workflow steps[{index}] must be an object with id and action")
    step_id = step.get("id")
    if not isinstance(step_id, int) or step_id <= 0:
        raise ValueError(f"workflow steps[{index}].id must be a positive integer")
    if step_id in seen_ids:
        raise ValueError(f"workflow step id {step_id} is duplicated")
    seen_ids.add(step_id)
    _validate_action(step.get("action"), step_id)


def _validate_action(action: Any, step_id: int) -> None:
    if not isinstance(action, dict):
        raise ValueError(f"workflow step {step_id}.action must be an object")
    name = action.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"workflow step {step_id}.action.name must be a non-empty string")
    if name not in ALLOWED_ACTION_NAMES:
        allowed = ", ".join(sorted(ALLOWED_ACTION_NAMES))
        raise ValueError(f"workflow step {step_id}.action.name has unknown tool '{name}'. Allowed tools: {allowed}")
    args = action.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"workflow step {step_id}.action.args must be an object")
    _validate_templates(args, f"workflow step {step_id}.action.args", allow_absolute_paths=False)


def _validate_templates(value: Any, location: str, allow_absolute_paths: bool) -> None:
    if isinstance(value, str):
        _validate_string(value, location, allow_absolute_paths)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_templates(item, f"{location}[{index}]", allow_absolute_paths)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_templates(item, f"{location}.{key}", allow_absolute_paths)


def _validate_string(value: str, location: str, allow_absolute_paths: bool) -> None:
    for match in ANY_TEMPLATE_RE.finditer(value):
        if ALLOWED_TEMPLATE_RE.fullmatch(match.group(0)) is None:
            raise ValueError(f"workflow {location} uses unsupported template variable '{match.group(0)}'")
    if not allow_absolute_paths and ABSOLUTE_PATH_RE.search(value):
        raise ValueError(f"workflow {location} contains a hardcoded absolute path; use interface inputs or outputs")
