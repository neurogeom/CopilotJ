# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

WORKFLOW_DEFINITION_PROMPT = """
You are an expert workflow author. Your job is to convert an execution trace into a reusable Workflow JSON definition.
Now the task is finished. According to the Original Task and Execution History, create a minimal, correct, reproducible workflow.

#Original Task: {{TASK}}

#Execution History:
{{STEPS}}

#Existing Summary:
{{SUMMARY}}

Return a JSON object ONLY, not markdown. Follow the exact workflow step shape shown in the examples.

<Example: single-image workflow>
{
  "schema_version": "2.0",
  "interface": {
    "inputs": {
      "image": {"type": "file", "required": true, "description": "Input image"},
      "output_dir": {"type": "directory", "required": false, "default": "{{run_dir}}"}
    },
    "outputs": {
      "measurements": {"type": "table", "path": "{{inputs.output_dir}}/measurements.csv"}
    }
  },
  "steps": [
    {
      "id": 1,
      "action": {
        "name": "run_macro",
        "args": {
          "script": "open(\"{{inputs.image}}\");\nrun(\"8-bit\");\nsaveAs(\"Results\", \"{{outputs.measurements.path}}\");"
        }
      }
    }
  ]
}
</Example>

<Example: two-image workflow>
{
  "schema_version": "2.0",
  "interface": {
    "inputs": {
      "t0_image": {"type": "file", "required": true, "description": "Initial timepoint image"},
      "t24_image": {"type": "file", "required": true, "description": "Later timepoint image"},
      "output_dir": {"type": "directory", "required": false, "default": "{{run_dir}}"},
      "threshold": {"type": "number", "required": false, "default": 100}
    },
    "outputs": {
      "t0_mask": {"type": "file", "path": "{{inputs.output_dir}}/t0_mask.tif"},
      "t24_mask": {"type": "file", "path": "{{inputs.output_dir}}/t24_mask.tif"},
      "measurements": {"type": "table", "path": "{{inputs.output_dir}}/measurements.csv"}
    }
  },
  "steps": [
    {
      "id": 1,
      "action": {
        "name": "run_macro",
        "args": {
          "script": "open(\"{{inputs.t0_image}}\");\nsetThreshold(0, {{inputs.threshold}});\nrun(\"Convert to Mask\");\nsaveAs(\"Tiff\", \"{{outputs.t0_mask.path}}\");"
        }
      }
    },
    {
      "id": 2,
      "action": {
        "name": "run_macro",
        "args": {
          "script": "open(\"{{inputs.t24_image}}\");\nsetThreshold(0, {{inputs.threshold}});\nrun(\"Convert to Mask\");\nsaveAs(\"Tiff\", \"{{outputs.t24_mask.path}}\");"
        }
      }
    },
    {
      "id": 3,
      "action": {
        "name": "execute_python_script",
        "args": {
          "script": "from pathlib import Path\nPath(\"{{outputs.measurements.path}}\").write_text(\"timepoint,mask\\nt0,{{outputs.t0_mask.path}}\\nt24,{{outputs.t24_mask.path}}\\n\")"
        }
      }
    }
  ]
}
</Example>

Rules:
1. Put values that change between runs in `interface.inputs`: image paths, folders, output_dir, thresholds, radii, channel indices, timepoints, and model choices.
2. Put every file artifact promised by the workflow in `interface.outputs`.
3. Replace hardcoded input paths, output paths, and reusable parameters inside step args with template variables. Step args must not retain absolute paths from the original run.
4. Use `{{inputs.<name>}}`, `{{outputs.<name>.path}}`, and `{{run_dir}}` only.
5. For multi-image workflows, use separate named file inputs such as `t0_image` and `t24_image`; do not depend on file ordering.
6. Each step must be an object with `id` and `action`. `action` must be an object with `name` and `args`.
7. Preserve the original tool names and the minimal step order required for execution.
8. Do not add mock outputs, fallback branches, or silent error handling.
**Encoding**: Avoid non-ASCII subscript/superscript characters (e.g. \u2080\u2013\u2089, \u2070\u2013\u2079) in generated scripts and file paths; they cause encoding errors on Windows.
"""


def make_workflow_definition_prompt(task: str, steps_text: str, summary: object | None = None) -> str:
    return (
        WORKFLOW_DEFINITION_PROMPT.replace("{{TASK}}", task)
        .replace("{{STEPS}}", steps_text)
        .replace("{{SUMMARY}}", "" if summary is None else str(summary))
    )
