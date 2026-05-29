# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Optional

from copilotj.workflow.contract import parse_interface
from copilotj.workflow.manager import Workflow, WorkflowMeta, WorkflowStep, slugify


class DialogToWorkflowConverter:
    @staticmethod
    def extract_definition(steps_text: str) -> dict[str, Any] | list[dict[str, Any]]:
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
        steps = definition.get("steps") if isinstance(definition, dict) else definition
        if not isinstance(steps, list):
            raise ValueError("workflow steps must be a JSON array")
        workflow_steps = [WorkflowStep(id=index, action=step) for index, step in enumerate(steps, 1)]
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
