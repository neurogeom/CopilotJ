# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared ReAct text (de)serialization helpers.

Used by both :class:`copilotj.multiagent.leader_multiagent.LeaderAgent` and
:class:`copilotj.multiagent.Executor.Executor` to rebuild an assistant-side
ReAct text block from a parsed ``ModelResponse`` so the next turn's context
matches the ReAct-formatted examples in the system prompt.

Lives in its own module so neither agent has to import the other (Executor is
imported by ``leader_multiagent``, which would create a circular import).
"""

import json

from copilotj.core import ModelResponse

__all__ = ["reconstruct_react_text"]


def reconstruct_react_text(response: ModelResponse) -> str:
    """Rebuild an assistant-side ReAct text block from a parsed ModelResponse.

    The ReAct wrapper parses the model's raw text output into ``reasoning_content``
    (Thought), ``tool_calls`` (Action), and ``content`` (Final Answer). For
    multi-turn conversation we re-assemble those parts into the same shape the
    model originally produced, so the next turn's context matches the
    ReAct-formatted examples in the system prompt.
    """
    parts: list[str] = []
    if response.reasoning_content:
        parts.append(f"Thought: {response.reasoning_content.strip()}")
    if response.tool_calls:
        tc = response.tool_calls[0]
        args_json = json.dumps(tc.args.model_dump(), ensure_ascii=False)
        parts.append(f'Action: {{"name": "{tc.tool.name}", "args": {args_json}}}')
    if response.content:
        parts.append(f"Final Answer: {response.content.strip()}")
    return "\n".join(parts)
