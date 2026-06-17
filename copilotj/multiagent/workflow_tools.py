# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated, Any, Optional

from copilotj.core import ModelClient, TextMessage
from copilotj.multiagent.leader_prompts import make_workflow_definition_prompt
from copilotj.workflow.converter import DialogToWorkflowConverter
from copilotj.workflow.executor import WorkflowExecutor
from copilotj.workflow.manager import WorkflowManager

WORKFLOW_ABOUT_PREVIEW_CHARS = 200


def _preview_text(text: str, limit: int) -> str:
    return f"{text[:limit]}..." if len(text) > limit else text


class WorkflowSaveService:
    def __init__(
        self,
        *,
        model_client: ModelClient,
        chat_history: list[dict[str, Any]],
        workflow_contexts: dict[int, dict[str, Any]],
    ) -> None:
        self.model_client = model_client
        self.chat_history = chat_history
        self.workflow_contexts = workflow_contexts

    async def save(
        self,
        workflow_name: str,
        tags: str | None,
        dialog_id: int | None,
        dialog_ids: list[int] | None,
    ) -> str:
        if not self.chat_history:
            return "No workflow in chat history to save."
        selected_ids = self._select_dialog_ids(dialog_id, dialog_ids)
        if isinstance(selected_ids, str):
            return selected_ids
        missing = [dialog_id for dialog_id in selected_ids if dialog_id not in self.workflow_contexts]
        if missing:
            missing_text = ", ".join(str(dialog_id) for dialog_id in missing)
            return f"Raw workflow context is unavailable for dialog(s) {missing_text}."
        summary = self._combined_summary(selected_ids)
        try:
            steps = await self._generate_workflow_steps(self._combined_context(selected_ids), summary)
        except Exception as e:
            return f"Error generating reusable workflow steps: {e}"
        if steps is None:
            return "Failed to generate reusable workflow steps from the selected dialogs."
        save_status = await save_workflow_from_steps(workflow_name, steps, summary, tags)
        return f"Workflow saved for dialog(s) {selected_ids}: {save_status}"

    def _select_dialog_ids(self, dialog_id: int | None, dialog_ids: list[int] | None) -> list[int] | str:
        if dialog_id is not None and dialog_ids is not None:
            return "Use either dialog_id or dialog_ids, not both."
        if dialog_ids is not None:
            if not dialog_ids:
                return "dialog_ids must not be empty."
            if len(set(dialog_ids)) != len(dialog_ids):
                return "dialog_ids must not contain duplicates."
            return dialog_ids
        if dialog_id is not None:
            return [dialog_id]
        latest_dialog = self.chat_history[-1].get("dialog")
        return [latest_dialog] if isinstance(latest_dialog, int) else "Latest dialog does not have a dialog id."

    def _combined_context(self, dialog_ids: list[int]) -> dict[str, Any]:
        tasks = [str(self.workflow_contexts[dialog_id].get("task", "")) for dialog_id in dialog_ids]
        return {
            "task": "\n".join(f"Dialog {dialog_id}: {task}" for dialog_id, task in zip(dialog_ids, tasks)),
            "steps": [
                {"dialog": dialog_id, "task": task, "steps": self.workflow_contexts[dialog_id].get("steps", [])}
                for dialog_id, task in zip(dialog_ids, tasks)
            ],
        }

    def _combined_summary(self, dialog_ids: list[int]) -> str:
        return "\n\n".join(
            self._format_history_item(item)
            for dialog_id in dialog_ids
            if (item := self._find_chat_history_dialog(dialog_id)) is not None
        )

    def _find_chat_history_dialog(self, dialog_id: int) -> dict[str, Any] | None:
        for item in reversed(self.chat_history):
            if item.get("dialog") == dialog_id:
                return item
        return None

    def _format_history_item(self, item: dict[str, Any]) -> str:
        text = f"Dialog {item.get('dialog')}: {item.get('assistant', '')}"
        return (
            text if not item.get("context") else text + "\n" + json.dumps(item["context"], ensure_ascii=False, indent=2)
        )

    async def _generate_workflow_steps(self, dialog_context: dict, summary: object | None = None) -> str | None:
        steps_text = json.dumps(dialog_context["steps"], indent=2, ensure_ascii=False)
        steps_prompt = make_workflow_definition_prompt(dialog_context["task"], steps_text, summary)
        response = await self.model_client.create([TextMessage(role="user", text=steps_prompt)])
        return response.content.strip() if response.content is not None else None


async def save_workflow_from_steps(
    workflow_name: Annotated[str, "Descriptive name for the workflow"],
    steps: Annotated[str, "The steps of the workflow"],
    summary: Annotated[str, "The summary of the workflow"],
    tags: Annotated[Optional[str], "Optional string of tags for categorization"] = None,
) -> str:
    if not isinstance(steps, str) or not steps.strip():
        return "Failed to auto-save workflow: dialog has no generated reusable workflow steps"

    try:
        workflow = DialogToWorkflowConverter.create_workflow(
            steps_text=steps,
            workflow_name=workflow_name,
            workflow_summary=summary,
            tags=tags,
        )

        workflow_id = WorkflowManager.save_workflow(workflow)
        workflow_about = workflow.meta.about or ""
        return f"""
✅ Workflow saved successfully:
workflow_id: {workflow_id}
workflow_name: {workflow_name}
workflow_about: {_preview_text(workflow_about, WORKFLOW_ABOUT_PREVIEW_CHARS)}
tags: {tags}
"""
    except Exception as e:
        return f"Failed to auto-save workflow: {e}"


async def list_workflows(dummy: Annotated[Optional[str], "dummy"] = None) -> str:
    try:
        workflows = WorkflowManager.list_workflows()

        if not workflows:
            return "📋 No workflows found in the library."

        result = ["📋 Available Workflows:"]
        for i, wf in enumerate(workflows, 1):
            name = wf.get("name", "Unknown")
            wf_id = wf.get("id", "Unknown")
            line = f"{i}. **{name}** (ID: `{wf_id}`)"
            result.append(line)
            if wf.get("about"):
                result.append(f"   Description: {_preview_text(wf['about'], WORKFLOW_ABOUT_PREVIEW_CHARS)}")

        return "\n".join(result)

    except json.JSONDecodeError as e:
        return f"❌ Error parsing workflow data: {str(e)}. The workflow storage may be corrupted or empty."
    except Exception as e:
        return f"❌ Error listing workflows: {str(e)}"


async def get_workflow(workflow_id: Annotated[str, "The ID of the workflow to retrieve"]) -> str:
    """Get detailed information about a specific workflow"""
    try:
        workflow = WorkflowManager.load_workflow(workflow_id)

        result = [
            f"📄 **Workflow Details: {workflow.meta.name}**",
            f"ID: {workflow.meta.id}",
            f"Version: {workflow.meta.version}",
            f"Source: {workflow.meta.source}",
            "",
        ]

        if workflow.meta.about:
            result.extend(["**Description:**", workflow.meta.about, ""])

        if workflow.meta.tags:
            result.extend(["**Tags:**", workflow.meta.tags, ""])

        if workflow.interface:
            result.extend(
                [
                    "**Interface:**",
                    "```json",
                    json.dumps(workflow.interface.to_dict(), ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

        result.extend(["**Steps:**", ""])

        for step in workflow.steps:
            result.extend(
                [
                    f"### Step {step.id}",
                    "**Action:**",
                    "```json",
                    json.dumps(step.action, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

        if workflow.dataset_pattern:
            result.extend(["**Dataset Pattern:**", f"`{workflow.dataset_pattern}`", ""])

        if workflow.outputs:
            result.extend(
                ["**Outputs:**", "```json", json.dumps(workflow.outputs, ensure_ascii=False, indent=2), "```", ""]
            )

        return "\n".join(result)

    except Exception as e:
        return f"❌ Error loading workflow: {str(e)}"


async def delete_workflow(workflow_id: Annotated[str, "The ID of the workflow to delete"]) -> str:
    """Delete a workflow"""
    try:
        success = WorkflowManager.delete_workflow(workflow_id)
        if success:
            return f"✅ Workflow '{workflow_id}' deleted successfully"
        else:
            return f"❌ Workflow '{workflow_id}' not found"

    except Exception as e:
        return f"❌ Error deleting workflow: {str(e)}"


async def export_workflow(
    workflow_id: Annotated[str, "The ID of the workflow to export"],
    format: Annotated[str, "Export format: json, markdown, actions, or zip"] = "json",
) -> str:
    """Export a workflow in various formats"""
    try:
        if format not in ["json", "markdown", "actions", "zip"]:
            return "❌ Invalid format. Supported formats: json, markdown, actions, zip"

        content = WorkflowManager.export_workflow(workflow_id, format)

        if format == "zip":
            return f"📦 **Workflow Export (ZIP):**\n\nZip file created at: {content}"
        elif format == "markdown":
            return f"📄 **Workflow Export ({format.upper()}):**\n\n{content}"
        else:
            return f"📄 **Workflow Export ({format.upper()}):**\n\n```{format}\n{content}\n```"

    except Exception as e:
        return f"❌ Error exporting workflow: {str(e)}"


async def execute_workflow(
    leader_agent: Any,
    workflow_id: Annotated[str, "The ID of the workflow to execute"],
    inputs: Annotated[Optional[dict[str, Any]], "Runtime inputs declared by workflow.interface.inputs"] = None,
    stop_on_error: Annotated[bool, "Whether to stop execution on first error"] = True,
) -> str:
    """Execute a workflow with the provided leader agent"""
    try:
        executor = WorkflowExecutor(leader_agent)
        results = await executor.execute_workflow_by_id(workflow_id, stop_on_error, inputs)

        # Format results for display
        result_lines = ["📋 **Workflow Execution Results:**"]

        for result in results:
            step_id = result["step_id"]
            action = result["action"]
            if result["ok"]:
                if step_id == "batch":
                    result_lines.append(f"✅ Batch expanded: {result['result']['count']} input files")
                    continue
                if str(step_id).endswith("outputs"):
                    result_lines.append(f"✅ Outputs verified: {json.dumps(result['result'], ensure_ascii=False)}")
                    continue
                result_lines.append(
                    f"✅ Step {step_id}:  {action.get('name', 'unknown')} - {action.get('args', 'unknown')} executed successfully."
                )
            else:
                result_lines.append(
                    f"❌ Step {step_id}:  {action.get('name', 'unknown')} - {action.get('args', 'unknown')} executed failed."
                )
                result_lines.append(f"   Error: {result['error']}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"❌ Error executing workflow: {str(e)}"
