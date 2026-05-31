# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from copilotj.core import ToolCall
from copilotj.multiagent.py_tools import get_project_temp_dir

BASE_DIR = get_project_temp_dir("workflows")
SHARE_DIR = BASE_DIR / "shared"
BASE_DIR.mkdir(parents=True, exist_ok=True)
SHARE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WorkflowStep:
    id: int
    action: Dict[str, Any]  # {"name": str, "args": dict}

    def to_dict(self):
        # If action is a string, parse as JSON
        if isinstance(self.action, str):
            try:
                action = json.loads(self.action)
            except Exception:
                action = self.action
        else:
            action = self.action
        return {"id": self.id, "action": action}


@dataclass
class WorkflowMeta:
    id: str
    name: str
    version: str = "1.0"
    about: Optional[str] = None
    tags: str = None
    source: str = "derived"
    created_at: float = None
    updated_at: float = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ""
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()


@dataclass
class Workflow:
    meta: WorkflowMeta
    steps: List[WorkflowStep]
    dataset_pattern: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "meta": asdict(self.meta),
            "steps": [step.to_dict() for step in self.steps],
            "dataset_pattern": self.dataset_pattern,
            "outputs": self.outputs,
        }


_slug_re = re.compile(r"[^a-z0-9\-]+")


def slugify(name: str) -> str:
    """Convert name to URL-friendly slug"""
    s = name.lower().replace(" ", "-")
    s = _slug_re.sub("", s)
    return s or f"wf-{int(time.time())}"


def wf_dir(wf_id: str) -> Path:
    """Get workflow directory"""
    p = BASE_DIR / wf_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any):
    """Write JSON file"""
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON file with robust encoding handling"""
    # Try UTF-8 first (for new files), then fall back to system default encoding
    encodings_to_try = ["utf-8", "gbk", "cp936", "latin1"]

    for encoding in encodings_to_try:
        try:
            content = path.read_text(encoding=encoding)
            if not content.strip():
                # Return empty dict for empty files to avoid JSON decode error
                return {}
            return json.loads(content)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except json.JSONDecodeError:
            # If JSON is invalid, that's a different issue - re-raise
            raise

    # If all encodings fail, raise an error with helpful message
    raise ValueError(f"Could not decode file {path} with any of the attempted encodings: {encodings_to_try}")


class DialogToWorkflowConverter:
    @staticmethod
    def extract_steps(steps_text: str) -> List[Dict[str, Any]]:
        json_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        match = re.search(json_pattern, steps_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            json_content = steps_text.strip()
        try:
            steps = json.loads(json_content)
            return steps
        except json.JSONDecodeError:
            return steps_text + "\nsteps is not a valid JSON string, please try to generate again"

    @staticmethod
    def create_workflow(
        workflow_name: str,
        steps_text: str,
        workflow_summary: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Workflow:
        steps = DialogToWorkflowConverter.extract_steps(steps_text)

        # Create workflow steps directly from the steps
        workflow_steps = []
        for i, step in enumerate(steps, 1):
            workflow_step = WorkflowStep(id=i, action=step)
            workflow_steps.append(workflow_step)

        # Create metadata
        meta = WorkflowMeta(
            id=slugify(workflow_name),
            name=workflow_name,
            about=workflow_summary or "Workflow created from summarized steps",
            tags=tags or "no tags",
            source="summarized",
        )

        # Create Workflow
        workflow = Workflow(meta=meta, steps=workflow_steps)

        return workflow


META_FILE = "workflow.json"


class WorkflowManager:
    """Workflow Manager"""

    @staticmethod
    def save_workflow(workflow: Workflow) -> str:
        """Save Workflow"""
        d = wf_dir(workflow.meta.id)
        write_json(d / META_FILE, workflow.to_dict())
        return workflow.meta.id

    @staticmethod
    def load_workflow(wf_id: str) -> Workflow:  # TODO add profiler load and JIpipe
        """Load Workflow"""
        p = wf_dir(wf_id) / META_FILE
        raw = read_json(p)

        # Rebuild metadata
        meta_data = raw.get("meta", {})
        meta = WorkflowMeta(
            id=meta_data.get("id"),
            name=meta_data.get("name"),
            version=meta_data.get("version", "1.0"),
            about=meta_data.get("about"),
            tags=meta_data.get("tags"),
            source=meta_data.get("source", "derived"),
            created_at=meta_data.get("created_at"),
            updated_at=meta_data.get("updated_at"),
        )

        # Rebuild steps
        steps = []
        for step_data in raw.get("steps", []):
            step = WorkflowStep(id=step_data["id"], action=step_data["action"])
            steps.append(step)

        return Workflow(
            meta=meta,
            steps=steps,
            dataset_pattern=raw.get("dataset_pattern"),
            outputs=raw.get("outputs"),
        )

    @staticmethod
    def list_workflows() -> List[Dict[str, Any]]:
        """List all Workflows"""
        items = []
        for d in sorted(BASE_DIR.iterdir()):
            if not d.is_dir():
                continue
            f = d / META_FILE
            if f.exists():
                raw = read_json(f)
                meta = raw.get("meta", {})
                # Skip workflows with empty or invalid metadata
                if meta and meta.get("name"):
                    items.append(meta)
        return items

    @staticmethod
    def delete_workflow(wf_id: str) -> str:
        """Delete Workflow and its folder"""
        import shutil
        from pathlib import Path

        d = Path(BASE_DIR) / wf_id
        if d.exists() and d.is_dir():
            try:
                shutil.rmtree(d)
                return f"The workflow {wf_id} and its folder have been deleted successfully."
            except Exception:
                return f"Failed to delete the workflow {wf_id} and its folder."
        return f"Workflow {wf_id} folder not found."

    @staticmethod
    def export_workflow(wf_id: str, fmt: str = "json") -> str:
        """Export Workflow"""
        workflow = WorkflowManager.load_workflow(wf_id)

        if fmt == "json":
            return json.dumps(workflow.to_dict(), ensure_ascii=False, indent=2)
        elif fmt == "actions":
            actions = [step.action for step in workflow.steps]
            return json.dumps(actions, ensure_ascii=False, indent=2)
        elif fmt == "markdown":
            markdown_content = WorkflowManager._generate_markdown(workflow)
            wf_path = wf_dir(workflow.meta.id) / (workflow.meta.id + ".md")
            wf_path.write_text(markdown_content, encoding="utf-8")
            return f"Markdown documentation exported to: {wf_path} successfully"
        elif fmt == "zip":
            return WorkflowManager._create_zip_export(wf_id)
        else:
            raise ValueError(f"Unsupported export format: {fmt}. Supported: json, actions, markdown, zip")

    @staticmethod
    def _generate_markdown(workflow: Workflow) -> str:
        """Generate Markdown format documentation"""
        lines = [
            f"# {workflow.meta.name}",
            "",
            f"**Version:** {workflow.meta.version}",
            f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow.meta.created_at))}",
            f"**Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(workflow.meta.updated_at))}",
            "",
        ]

        if workflow.meta.about:
            lines.extend(["## About", workflow.meta.about, ""])

        if workflow.meta.tags:
            lines.extend(["## Tags", workflow.meta.tags, ""])

        if workflow.dataset_pattern:
            lines.extend(["## Dataset Pattern", f"`{workflow.dataset_pattern}`", ""])

        lines.extend(["## Steps", ""])

        for step in workflow.steps:
            lines.extend(
                [
                    f"### Step {step.id}",
                    "",
                    "**Action:**",
                    "```json",
                    json.dumps(step.action, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

        if workflow.outputs:
            lines.extend(
                ["## Outputs", "", "```json", json.dumps(workflow.outputs, ensure_ascii=False, indent=2), "```", ""]
            )

        return "\n".join(lines)

    @staticmethod
    def _create_zip_export(wf_id: str) -> str:
        """Create zip export of workflow"""
        d = wf_dir(wf_id)
        out = SHARE_DIR / f"{wf_id}.zip"

        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in d.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(d))

        return f"Files in {wf_id} exported to: {out} successfully"


class WorkflowExecutor:
    """Workflow Executor"""

    def __init__(self, leader_agent):
        self.leader_agent = leader_agent

    async def _exec_tool(self, action: dict):
        tool_obj = next((t for t in self.leader_agent.tools if getattr(t, "name", None) == action.get("name")), None)
        if not tool_obj:
            raise RuntimeError(
                f"Tool '{action.get('name')}' not found. Registered: {[getattr(t, 'name', None) for t in self.leader_agent.tools]}"
            )
        if not isinstance(action.get("args"), dict):
            raise TypeError(
                f"Tool '{action.get('name')}' expects args as dict, got {type(action.get('args'))}: {action.get('args')}"
            )

        args = tool_obj.args_type().model_validate(action.get("args"))
        tool_call = ToolCall(id=str(uuid.uuid4()), tool=tool_obj, args=args)
        try:
            results = await self.leader_agent._call_tool(tool_call)
            return results
        except Exception as e:
            return f"Error occurred while executing tool '{tool_obj.name}': {e}"

    async def execute_workflow(self, workflow: Workflow, stop_on_error: bool = True) -> List[Dict[str, Any]]:
        results = []
        for step in workflow.steps:
            action = step.action
            a_args = action.get("args")
            if not isinstance(a_args, dict):
                raise TypeError(
                    f"Workflow step {step.id} action.args must be dict, got {type(a_args)}: {a_args}, please check this step in manuscript"
                )

            try:
                res = await self._exec_tool(action)
                results.append({"step_id": step.id, "action": action, "ok": True, "result": res})
            except Exception as e:
                results.append({"step_id": step.id, "action": action, "ok": False, "error": str(e)})
                if stop_on_error:
                    break

        return results

    async def execute_workflow_by_id(self, wf_id: str, stop_on_error: bool = True) -> List[Dict[str, Any]]:
        """Execute Workflow by ID"""
        workflow = WorkflowManager.load_workflow(wf_id)
        return await self.execute_workflow(workflow, stop_on_error)
