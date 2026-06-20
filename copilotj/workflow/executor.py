# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import uuid
from typing import Any, Optional

from copilotj.core import ToolCall
from copilotj.workflow.contract import (
    RUNS_DIR,
    WorkflowExecutionContext,
    bind_workflow_context,
    expand_batch_inputs,
    render_templates,
    template_uses_run_dir,
    verify_declared_outputs,
)
from copilotj.workflow.manager import Workflow, WorkflowManager, WorkflowStep


class WorkflowExecutor:
    """Execute a workflow after binding its runtime interface."""

    def __init__(self, leader_agent):
        self.leader_agent = leader_agent

    async def _exec_tool(self, action: dict):
        tool_obj = next((t for t in self.leader_agent.tools if getattr(t, "name", None) == action.get("name")), None)
        if not tool_obj:
            names = [getattr(t, "name", None) for t in self.leader_agent.tools]
            raise RuntimeError(f"Tool '{action.get('name')}' not found. Registered: {names}")
        if not isinstance(action.get("args"), dict):
            args = action.get("args")
            raise TypeError(f"Tool '{action.get('name')}' expects args as dict, got {type(args)}: {args}")

        args = tool_obj.args_type().model_validate(action.get("args"))
        tool_call = ToolCall(id=str(uuid.uuid4()), tool=tool_obj, args=args)
        return await self.leader_agent._call_tool(tool_call)

    async def execute_workflow(
        self,
        workflow: Workflow,
        stop_on_error: bool = True,
        inputs: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        batch_inputs = expand_batch_inputs(workflow.interface, inputs)
        if len(batch_inputs) == 1:
            return await self._execute_bound_workflow(workflow, stop_on_error, batch_inputs[0])
        return await self._execute_batch(workflow, stop_on_error, batch_inputs)

    async def _execute_batch(
        self,
        workflow: Workflow,
        stop_on_error: bool,
        batch_inputs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        results = [self._batch_start_result(batch_inputs)]
        for index, item_inputs in enumerate(batch_inputs, 1):
            item_results = await self._execute_bound_workflow(workflow, stop_on_error, item_inputs)
            results.extend(self._with_batch_metadata(index, item_inputs, item_results))
            if stop_on_error and any(not result["ok"] for result in item_results):
                break
        return results

    async def _execute_bound_workflow(
        self,
        workflow: Workflow,
        stop_on_error: bool,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        context = self._bind_context(workflow, inputs)
        results = []
        for step in workflow.steps:
            action = self._render_action(step, context)
            self._validate_args(step, action)
            try:
                res = await self._exec_tool(action)
                results.append({"step_id": step.id, "action": action, "ok": True, "result": res})
            except Exception as e:
                results.append({"step_id": step.id, "action": action, "ok": False, "error": str(e)})
                if stop_on_error:
                    break

        self._append_output_verification(results, context)
        return results

    async def execute_workflow_by_id(
        self,
        wf_id: str,
        stop_on_error: bool = True,
        inputs: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        workflow = WorkflowManager.load_workflow(wf_id)
        return await self.execute_workflow(workflow, stop_on_error, inputs)

    @staticmethod
    def _render_action(step: WorkflowStep, context: WorkflowExecutionContext) -> dict:
        action = render_templates(step.action, context.to_template_scope())
        if not isinstance(action, dict):
            raise TypeError(f"Workflow step {step.id} action must be dict, got {type(action)}")
        return action

    @staticmethod
    def _workflow_steps_use_run_dir(workflow: Workflow) -> bool:
        return any(template_uses_run_dir(step.action) for step in workflow.steps)

    @staticmethod
    def _bind_context(workflow: Workflow, inputs: dict[str, Any]) -> WorkflowExecutionContext:
        return bind_workflow_context(
            workflow.interface,
            inputs,
            RUNS_DIR,
            require_run_dir=WorkflowExecutor._workflow_steps_use_run_dir(workflow),
            run_name=workflow.meta.id,
        )

    @staticmethod
    def _validate_args(step: WorkflowStep, action: dict):
        args = action.get("args")
        if isinstance(args, dict):
            return
        raise TypeError(f"Workflow step {step.id} action.args must be dict, got {type(args)}: {args}")

    @staticmethod
    def _batch_start_result(batch_inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "step_id": "batch",
            "action": {"name": "expand_batch_inputs"},
            "ok": True,
            "result": {"count": len(batch_inputs), "items": batch_inputs},
        }

    @staticmethod
    def _with_batch_metadata(
        index: int, item_inputs: dict[str, Any], results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tagged = []
        for result in results:
            item_result = dict(result)
            item_result["batch_index"] = index
            item_result["batch_inputs"] = item_inputs
            item_result["step_id"] = f"{index}.{result['step_id']}"
            tagged.append(item_result)
        return tagged

    @staticmethod
    def _append_output_verification(results: list[dict[str, Any]], context: WorkflowExecutionContext):
        failed = any(not result["ok"] for result in results)
        if failed or not context.outputs:
            return
        try:
            outputs = verify_declared_outputs(context.outputs)
            results.append({"step_id": "outputs", "action": {"name": "verify_outputs"}, "ok": True, "result": outputs})
        except Exception as e:
            results.append({"step_id": "outputs", "action": {"name": "verify_outputs"}, "ok": False, "error": str(e)})
