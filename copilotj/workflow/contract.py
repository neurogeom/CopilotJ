# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from copilotj.multiagent.py_tools import get_project_temp_dir

SCHEMA_VERSION = "2.0"
RUN_DIR_PREFIX = "run-"
RUN_DIR_REF = "run_dir"
TEMPLATE_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_.-]+)\s*\}\}")
DEFAULT_FILE_INPUT_NAME = "image"
OUTPUT_DIR_INPUT_NAME = "output_dir"

RUNS_DIR = get_project_temp_dir("workflow_runs")


class WorkflowContractError(ValueError):
    """Raised when a workflow contract cannot be bound or verified."""


@dataclass(frozen=True)
class WorkflowInterface:
    inputs: dict[str, Any]
    outputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "outputs": self.outputs}


@dataclass(frozen=True)
class WorkflowExecutionContext:
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    run_dir: str

    def to_template_scope(self) -> dict[str, Any]:
        return {"inputs": self.inputs, "outputs": self.outputs, "run_dir": self.run_dir}


def parse_interface(raw: dict[str, Any]) -> WorkflowInterface | None:
    interface = raw.get("interface")
    if isinstance(interface, dict):
        return WorkflowInterface(
            inputs=_dict_or_empty(interface.get("inputs")),
            outputs=_dict_or_empty(interface.get("outputs")),
        )
    outputs = raw.get("outputs")
    if isinstance(outputs, dict):
        return WorkflowInterface(inputs={}, outputs=outputs)
    return None


def bind_workflow_context(
    interface: WorkflowInterface | None,
    provided_inputs: dict[str, Any] | None,
    base_dir: Path,
    require_run_dir: bool = False,
) -> WorkflowExecutionContext:
    provided = provided_inputs or {}
    run_dir = _resolve_run_dir(interface, provided, base_dir, require_run_dir)
    if interface is None:
        return WorkflowExecutionContext(inputs=provided, outputs={}, run_dir=run_dir)
    inputs = _bind_inputs(interface.inputs, provided, run_dir)
    _ensure_directory_inputs(interface.inputs, inputs)
    scope = {"inputs": inputs, "run_dir": run_dir}
    outputs = render_templates(interface.outputs, scope)
    return WorkflowExecutionContext(inputs=inputs, outputs=outputs, run_dir=run_dir)


def expand_batch_inputs(
    interface: WorkflowInterface | None, provided_inputs: dict[str, Any] | None
) -> list[dict[str, Any]]:
    inputs = dict(provided_inputs or {})
    if interface is None:
        return [inputs]
    file_input = _folder_file_input(interface.inputs, inputs)
    if file_input is None:
        return [inputs]
    name, folder = file_input
    files = _matching_files(folder, interface.inputs[name])
    output_dir = _provided_output_dir(interface.inputs, inputs)
    return [_batch_item_inputs(inputs, name, file_path, output_dir) for file_path in files]


def render_templates(value: Any, scope: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _render_string(value, scope)
    if isinstance(value, list):
        return [render_templates(item, scope) for item in value]
    if isinstance(value, dict):
        return {key: render_templates(item, scope) for key, item in value.items()}
    return value


def verify_declared_outputs(outputs: dict[str, Any]) -> dict[str, str]:
    missing = {}
    for name, spec in outputs.items():
        path = _output_path(spec)
        if path and not Path(path).exists():
            missing[name] = path
    if missing:
        detail = ", ".join(f"{name}: {path}" for name, path in missing.items())
        raise WorkflowContractError(f"Declared workflow outputs were not created: {detail}")
    return {name: path for name, spec in outputs.items() if (path := _output_path(spec))}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise WorkflowContractError(f"Workflow interface section must be an object, got {type(value)}")


def _make_run_dir() -> Path:
    run_dir = RUNS_DIR / f"{RUN_DIR_PREFIX}{int(time.time() * 1000)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_run_dir(
    interface: WorkflowInterface | None,
    provided: dict[str, Any],
    base_dir: Path,
    require_run_dir: bool,
) -> str:
    if require_run_dir or _interface_needs_run_dir(interface, provided):
        return str(_make_run_dir())
    return ""


def _interface_needs_run_dir(interface: WorkflowInterface | None, provided: dict[str, Any]) -> bool:
    if interface is None:
        return False
    return _input_defaults_need_run_dir(interface.inputs, provided) or template_uses_run_dir(interface.outputs)


def _input_defaults_need_run_dir(specs: dict[str, Any], provided: dict[str, Any]) -> bool:
    for name, spec in specs.items():
        if name in provided or not isinstance(spec, dict):
            continue
        if template_uses_run_dir(spec.get("default")):
            return True
    return False


def template_uses_run_dir(value: Any) -> bool:
    if isinstance(value, str):
        return any(match.group(1) == RUN_DIR_REF for match in TEMPLATE_RE.finditer(value))
    if isinstance(value, list):
        return any(template_uses_run_dir(item) for item in value)
    if isinstance(value, dict):
        return any(template_uses_run_dir(item) for item in value.values())
    return False


def _bind_inputs(specs: dict[str, Any], provided: dict[str, Any], run_dir: str) -> dict[str, Any]:
    bound = {}
    for name, spec in specs.items():
        bound[name] = _resolve_input(name, spec, provided, bound, run_dir)
    unknown = sorted(set(provided) - set(specs))
    if unknown:
        raise WorkflowContractError(f"Unknown workflow inputs: {', '.join(unknown)}")
    return bound


def _resolve_input(name: str, spec: Any, provided: dict[str, Any], bound: dict[str, Any], run_dir: str) -> Any:
    if name in provided:
        return _normalize_input_value(spec, provided[name])
    if not isinstance(spec, dict):
        raise WorkflowContractError(f"Workflow input '{name}' spec must be an object")
    if "default" in spec:
        scope = {"inputs": bound, "run_dir": run_dir}
        return _normalize_input_value(spec, render_templates(spec["default"], scope))
    if spec.get("required", False):
        raise WorkflowContractError(f"Missing required workflow input: {name}")
    return None


def _ensure_directory_inputs(specs: dict[str, Any], inputs: dict[str, Any]):
    for name, spec in specs.items():
        if not isinstance(spec, dict) or spec.get("type") != "directory":
            continue
        path = inputs.get(name)
        if isinstance(path, str) and path:
            Path(path).mkdir(parents=True, exist_ok=True)


def _folder_file_input(specs: dict[str, Any], inputs: dict[str, Any]) -> tuple[str, Path] | None:
    for name in _file_input_names(specs):
        value = inputs.get(name)
        if isinstance(value, str) and Path(value).is_dir():
            return name, Path(value)
    return None


def _file_input_names(specs: dict[str, Any]) -> list[str]:
    names = [name for name, spec in specs.items() if isinstance(spec, dict) and spec.get("type") == "file"]
    if DEFAULT_FILE_INPUT_NAME in names:
        return [DEFAULT_FILE_INPUT_NAME, *[name for name in names if name != DEFAULT_FILE_INPUT_NAME]]
    return names


def _matching_files(folder: Path, spec: Any) -> list[Path]:
    formats = _input_formats(spec)
    files = [path for path in sorted(folder.iterdir()) if path.is_file() and _matches_format(path, formats)]
    if not files:
        suffixes = ", ".join(sorted(formats)) if formats else "any file"
        raise WorkflowContractError(f"No workflow input files found in folder {folder} matching {suffixes}")
    return files


def _input_formats(spec: Any) -> set[str]:
    if not isinstance(spec, dict):
        return set()
    formats = spec.get("formats") or spec.get("extensions")
    if not isinstance(formats, list):
        return set()
    return {str(item).lower().lstrip(".") for item in formats}


def _matches_format(path: Path, formats: set[str]) -> bool:
    return not formats or path.suffix.lower().lstrip(".") in formats


def _provided_output_dir(specs: dict[str, Any], inputs: dict[str, Any]) -> Path | None:
    output_spec = specs.get(OUTPUT_DIR_INPUT_NAME)
    if not isinstance(output_spec, dict) or output_spec.get("type") != "directory":
        return None
    output_dir = inputs.get(OUTPUT_DIR_INPUT_NAME)
    return Path(output_dir) if isinstance(output_dir, str) and output_dir else None


def _batch_item_inputs(
    inputs: dict[str, Any], input_name: str, file_path: Path, output_dir: Path | None
) -> dict[str, Any]:
    item_inputs = dict(inputs)
    item_inputs[input_name] = str(file_path)
    if output_dir is not None:
        item_inputs[OUTPUT_DIR_INPUT_NAME] = str(output_dir / file_path.stem)
    return item_inputs


def _render_string(template: str, scope: dict[str, Any]) -> str:
    return TEMPLATE_RE.sub(lambda match: str(_resolve_reference(match.group(1), scope)), template)


def _resolve_reference(reference: str, scope: dict[str, Any]) -> Any:
    value: Any = scope
    for part in reference.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        raise WorkflowContractError(f"Unknown workflow template reference: {reference}")
    return value


def _output_path(spec: Any) -> str | None:
    if isinstance(spec, dict):
        path = spec.get("path")
    else:
        path = spec
    if not isinstance(path, str) or not path:
        return None
    if "://" in path:
        return None
    return path


def _normalize_input_value(spec: Any, value: Any) -> Any:
    if not isinstance(spec, dict) or spec.get("type") not in {"file", "directory"}:
        return value
    if not isinstance(value, str):
        return value
    return value.replace("\\", "/")
