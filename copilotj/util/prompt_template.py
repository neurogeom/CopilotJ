# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal Jinja2-like prompt template renderer.

Supports ``{% if variable == "value" %}`` / ``{% elif %}`` / ``{% else %}`` /
``{% endif %}`` conditional blocks with simple equality checks.  No external
dependency — just regex and a small state machine.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["render_prompt"]

# Matches: {% if var == "val" %}, {% elif var == "val" %}, {% else %}, {% endif %}
_TAG_RE = re.compile(
    r"^\s*\{%"  # opening delimiter {%
    r"\s*(if|elif|else|endif)"  # tag name
    r"(?:\s+(\w+)\s*==\s*\"([^\"]*)\")?"  # optional: variable == "value"
    r"\s*%\}\s*$",  # closing delimiter %}
    re.MULTILINE,
)

# Matches template variables: {variable_name}
_VAR_RE = re.compile(r"\{(\w+)\}")


def render_prompt(template: str, **variables: str) -> str:
    """Render a prompt template with conditional sections.

    Supports:
        {% if variable == "value" %}
        {% elif variable == "other" %}
        {% else %}
        {% endif %}

    Conditional tags must appear on their own lines.  Lines containing only a
    conditional tag are removed from the output.  ``{% else %}`` and
    ``{% endif %}`` do not take a condition.

    Template variables ``{name}`` are substituted with *variables[name]*.
    Missing variables are left as-is.

    Args:
        template: The prompt template string.
        **variables: Variable values for condition evaluation and substitution.

    Returns:
        The rendered prompt string.
    """
    # --- Phase 1: evaluate conditionals ---
    lines = template.split("\n")
    result_lines: list[str] = []

    # Stack of (branch_taken, active, in_else).  Tracks nesting.
    # branch_taken: whether any branch in this if-block has been taken so far.
    # active: whether lines in the *current* section should be emitted.
    # in_else: whether we are past an {% else %} for this level.
    stack: list[tuple[bool, bool, bool]] = []

    for line in lines:
        m = _TAG_RE.match(line.strip())
        if m is None:
            # Regular line — emit only if all enclosing conditions are true.
            if all(active for _, active, _ in stack):
                result_lines.append(line)
            continue

        tag, var_name, var_value = m.group(1), m.group(2), m.group(3)

        if tag == "if":
            cond = _eval_condition(var_name, var_value, variables)
            stack.append((cond, cond, False))

        elif tag == "elif":
            if not stack:
                raise SyntaxError("{% elif %} without matching {% if %}")
            branch_taken, _, in_else = stack[-1]
            if in_else:
                raise SyntaxError("{% elif %} after {% else %}")
            # Only evaluate if no previous branch was taken.
            new_cond = _eval_condition(var_name, var_value, variables) if not branch_taken else False
            stack[-1] = (branch_taken or new_cond, new_cond, False)

        elif tag == "else":
            if not stack:
                raise SyntaxError("{% else %} without matching {% if %}")
            branch_taken, _, _ = stack[-1]
            stack[-1] = (branch_taken, not branch_taken, True)

        elif tag == "endif":
            if not stack:
                raise SyntaxError("{% endif %} without matching {% if %}")
            stack.pop()

    if stack:
        raise SyntaxError("Unclosed {% if %} block in prompt template")

    # --- Phase 2: substitute variables ---
    text = "\n".join(result_lines)

    def _replace_var(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in variables:
            return str(variables[name])
        return m.group(0)  # leave as-is if not provided

    text = _VAR_RE.sub(_replace_var, text)

    # Clean up blank lines left by removed conditional tags
    text = _clean_blank_lines(text)

    return text


def _eval_condition(var_name: str | None, var_value: str | None, variables: dict[str, Any]) -> bool:
    """Evaluate ``variable == "value"``."""
    if var_name is None or var_value is None:
        raise SyntaxError("Missing condition in {% if/elif %} tag")
    actual = variables.get(var_name)
    return str(actual) == var_value


def _clean_blank_lines(text: str) -> str:
    """Remove excessive blank lines (3+ consecutive → 2) caused by stripped tags."""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text
