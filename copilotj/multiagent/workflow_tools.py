# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Annotated, Optional

import copilotj.multiagent.leader_multiagent as leader_multiagent
from copilotj.core import load_env
from copilotj.multiagent.workflow_manager import (
    DialogToWorkflowConverter,
    WorkflowExecutor,
    WorkflowManager,
)


async def save_workflow_from_steps(
    workflow_name: Annotated[str, "Descriptive name for the workflow"],
    steps: Annotated[str, "The steps of the workflow"],
    summary: Annotated[str, "The summary of the workflow"],
    tags: Annotated[Optional[str], "Optional string of tags for categorization"] = None,
) -> str:
    try:
        workflow = DialogToWorkflowConverter.create_workflow(
            steps_text=steps,
            workflow_name=workflow_name,
            workflow_summary=summary,
            tags=tags,
        )

        workflow_id = WorkflowManager.save_workflow(workflow)
        return f"""
✅ Workflow saved successfully:
workflow_id: {workflow_id}
workflow_name: {workflow_name}
workflow_about: {summary[:100]}...
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
            result.append(f"\n{i}. **{wf.get('name', 'Unknown')}** (ID: {wf.get('id', 'Unknown')})")
            result.append(f"   Version: {wf.get('version', '1.0')}")
            if wf.get("about"):
                result.append(f"   Description: {wf['about']}")  # TODO: be concise
            if wf.get("tags"):
                result.append(f"   Tags: {wf['tags']}")

            # Format timestamps for better readability
            import time

            created_at = wf.get("created_at")
            updated_at = wf.get("updated_at")

            if created_at and isinstance(created_at, (int, float)):
                created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
                result.append(f"   Created: {created_str}")
            else:
                result.append(f"   Created: {created_at or 'Unknown'}")

            if updated_at and isinstance(updated_at, (int, float)):
                updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at))
                result.append(f"   Updated: {updated_str}")
            else:
                result.append(f"   Updated: {updated_at or 'Unknown'}")

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
    workflow_id: Annotated[str, "The ID of the workflow to execute"],
    stop_on_error: Annotated[bool, "Whether to stop execution on first error"] = True,
) -> str:
    """Execute a workflow with the provided leader agent"""
    try:
        from copilotj.plugin.api import HTTPPluginAPI

        load_env()
        apis = HTTPPluginAPI("http://127.0.0.1:8786")
        client_apis = apis.attach_dev_client()
        leader = leader_multiagent.LeaderDriven(apis=client_apis)
        executor = WorkflowExecutor(leader.leader_agent)
        results = await executor.execute_workflow_by_id(workflow_id, stop_on_error)

        # Format results for display
        result_lines = ["📋 **Workflow Execution Results:**"]

        for result in results:
            step_id = result["step_id"]
            action = result["action"]
            if result["ok"]:
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
