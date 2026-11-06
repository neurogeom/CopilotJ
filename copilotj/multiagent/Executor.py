# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from copilotj.core import ChatAgent, ModelClient, ModelSyntaxError, TextMessage, Tool
from copilotj.multiagent.leader_prompts import build_tool_prompt
from copilotj.multiagent.react_format import reconstruct_react_text

__all__ = ["Executor"]


class Executor(ChatAgent):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        prompt: str,
        tools: list[Tool] | None = None,
        model_client: ModelClient,
    ):
        super().__init__(name, description, model_client=model_client)

        self.tools = tools or []
        self.max_iterations = 15
        self.tool_retry_counter = 0
        self.max_tool_retry = 3
        # Mirror LeaderAgent's malformed-ReAct budget so a stuck model can't
        # spam the prompt with correction turns (which would bloat the very
        # prefix we want cache-stable).
        self.max_syntax_errors = 3

        self.system_prompt = self._build_enhanced_system_prompt(prompt)

    def _build_enhanced_system_prompt(self, base_prompt: str) -> str:
        """Build system prompt that includes available tools information from config.

        The tool descriptions embedded here are the ONLY way the model sees the
        tools: the ReAct wrapper strips the ``tools=`` API param before the
        provider call (see ``react_parser.py`` — "dont send tools since we are
        parsing the response manually"). Do NOT remove this text expecting the
        ``tools=`` param to cover it; it won't reach the provider.
        """
        if not self.tools:
            return base_prompt

        tools_info = build_tool_prompt(self.tools)

        tools_usage = """\
## Tool Usage Format:
When you need to use a tool, format your action as json format:
Action: {"name": "<tool_name>", "args": <tool_args_in_json_format>}

## Tool Selection Guidelines:
- Choose the most appropriate tool based on the task requirements
- Provide clear and specific parameters for the tool
- If unsure about parameters, describe what you want to accomplish
"""

        return "\n".join((base_prompt, tools_info, tools_usage))

    async def run(self, task: str) -> str:
        """Execute the agent task with tool usage and reflection.

        Uses an append-only dialog — the static system prompt plus the initial
        user task, then one assistant turn (reconstructed ReAct text) plus one
        user observation turn per step — so the prompt prefix stays
        byte-stable and provider-side prefix caching hits on every iteration
        (only the newest turn is uncached). ``steps`` is a side-channel used
        solely to build the fallback summary on loop exhaustion; it never
        enters the prompt.
        """
        self.log_info(f"🟢 {self.name} is executing: {task}")
        self.log_info(f"📋 Available tools: {[t.name for t in self.tools]}")

        # Reset per-call counters. tool_retry_counter is instance state and the
        # executor instance is reused across delegated tasks
        # (LeaderAgent.delegate_task), so it must be reset here or one failed
        # task can poison the next.
        self.tool_retry_counter = 0
        syntax_error_counter = 0

        steps: list[dict[str, Any]] = []
        messages: list[TextMessage] = [
            TextMessage(role="system", text=self.system_prompt),
            TextMessage(role="user", text=task),
        ]

        try:
            for iteration in range(self.max_iterations):
                self.log_info(f"Iteration {iteration + 1}/{self.max_iterations}")

                try:
                    response = await self._create(*messages, tools=self.tools)
                except ModelSyntaxError as e:
                    self.log_error("Agent generated invalid ReAct syntax. Retrying...")
                    syntax_error_counter += 1
                    if syntax_error_counter >= self.max_syntax_errors:
                        return (
                            f"❌ {self.name} aborted: too many invalid ReAct responses "
                            f"({syntax_error_counter}/{self.max_syntax_errors})."
                        )
                    steps.append(
                        {
                            "thought": getattr(e.chat_completion, "reasoning_content", None),
                            "error": e.message,
                            "iteration": iteration + 1,
                        }
                    )
                    # NOTE(cache trade-off, deferred fix): appending this correction
                    # as a second consecutive `user` turn means the provider formatter
                    # (Anthropic/Gemini) merges same-role messages — it tacks this block
                    # onto the previous user message, mutating that message's content. So
                    # the prefix cached up to the previous observation no longer matches
                    # and THIS retry eats one cache miss (re-cached at the new trailing
                    # block; subsequent iterations resume cache hits). Rare (only on
                    # malformed ReAct) and self-healing, so not fixed here. The fix would
                    # be to first append the model's malformed output as an `assistant`
                    # turn (via reconstruct_react_text(e.chat_completion)) so roles
                    # alternate and no merge occurs.
                    messages.append(TextMessage(role="user", text=self._correction_text(e, iteration)))
                    continue

                # Turn parsed cleanly — reset the malformed-ReAct budget so only a
                # truly stuck model (3 CONSECUTIVE bad turns) aborts, not 3 slips
                # scattered across a long run.
                syntax_error_counter = 0

                if not response.content and not response.tool_calls and not response.reasoning_content:
                    await self.print_error("No response from agent")
                    break

                # Final answer?
                if response.content or self._is_task_complete(response.reasoning_content or ""):
                    steps.append(
                        {
                            "thought": response.reasoning_content or "Task completed",
                            "final_answer": response.content,
                            "iteration": iteration + 1,
                        }
                    )
                    return response.content or response.reasoning_content or ""

                # Record + append the assistant turn (frozen from here on).
                assistant_text = reconstruct_react_text(response)
                if assistant_text:
                    messages.append(TextMessage(role="assistant", text=assistant_text))

                # No action produced → append a reflection/suggestion user turn.
                if not response.tool_calls:
                    suggestion = self._suggest_tool_based_on_context(response.reasoning_content or "", task)
                    steps.append(
                        {
                            "thought": response.reasoning_content or "No clear thought",
                            "reflection_needed": True,
                            "tool_suggestion": suggestion,
                            "iteration": iteration + 1,
                        }
                    )
                    messages.append(TextMessage(role="user", text=self._reflection_text(suggestion, iteration)))
                    continue

                # Execute the tool, then append a user observation turn.
                tool_call = response.tool_calls[0]  # TODO: handle multiple tool calls
                action_summary = f"{tool_call.tool.name} with args: {str(tool_call.args)[:100]}"
                self.log_info(f"🔧 Executing tool: {action_summary}...")
                try:
                    tool_response = await self._call_tool(tool_call)
                    self.tool_retry_counter = 0  # Reset on success

                    steps.append(
                        {
                            "thought": response.reasoning_content or "No thought provided",
                            "action": action_summary,
                            "response": tool_response,
                            "iteration": iteration + 1,
                        }
                    )
                    messages.append(TextMessage(role="user", text=self._observation_text(tool_response, iteration)))

                except Exception as e:
                    error_msg = f"❌ Error executing action: {str(e)}"
                    self.log_error(error_msg)
                    self.tool_retry_counter += 1

                    if self.tool_retry_counter >= self.max_tool_retry:
                        return f"❌ {self.name} failed after {self.max_tool_retry} attempts: {error_msg}"

                    steps.append(
                        {
                            "thought": response.reasoning_content or "No thought provided",
                            "action": action_summary,
                            "error": error_msg,
                            "iteration": iteration + 1,
                        }
                    )
                    messages.append(TextMessage(role="user", text=self._observation_text(error_msg, iteration)))

            return self._generate_final_summary({"task": task, "status": "in_progress", "steps": steps})

        except Exception as e:
            self.log_error(f"Executor error: {e}")
            return f"❌ {self.name} encountered an error: {str(e)}"

    def _observation_text(self, tool_response: str, iteration: int) -> str:
        """User-turn text for a tool observation.

        Volatile content (the response, the step counter) lives only in this
        newest turn, so the cached prefix (system + prior turns) is never
        invalidated.
        """
        return f"Observation:\n{tool_response}\n\nProgress: step {iteration + 1}/{self.max_iterations}"

    def _reflection_text(self, suggestion: str, iteration: int) -> str:
        """User-turn text when the model produced a thought but no action."""
        return (
            f"{self._generate_reflection_prompt(iteration)}\n\n{suggestion}\n\n"
            f"Progress: step {iteration + 1}/{self.max_iterations}"
        )

    def _correction_text(self, e: ModelSyntaxError, iteration: int) -> str:
        """User-turn corrective prompt appended after a malformed ReAct response."""
        return (
            f"Your previous response was not valid ReAct format. Error: {e.message}\n"
            f'Reply with Thought: then Action: {{"name": ..., "args": ...}} (or Final Answer:).\n'
            f"Progress: step {iteration + 1}/{self.max_iterations}"
        )

    def _suggest_tool_based_on_context(self, thought: str, task: str) -> str:
        """Suggest appropriate tool based on context and descriptions"""
        context = f"{thought} {task}".lower()

        # Use the tool descriptions from config to make suggestions
        suggested_tools = []
        for tool in self.tools:
            # FIXME(pre-existing): `tool.name` is a str tested against self.tools (a
            # list[Tool]), so this is always False — the keyword-matching body below
            # never runs, and this method always returns the generic "choose from all
            # tools" fallback. Unrelated to caching; intentionally left unchanged in
            # the append-only refactor. Likely meant `if tool in self.tools:` (a
            # tautology) or to drop the guard entirely so matching actually fires.
            if tool.name in self.tools:
                # Check if context matches tool description keywords
                desc_words = tool.description.lower().split()
                context_words = context.split()

                # Simple keyword matching
                matches = len(set(desc_words) & set(context_words))
                if matches > 0:
                    suggested_tools.append((tool.name, matches))

        # Sort by number of matches
        suggested_tools.sort(key=lambda x: x[1], reverse=True)

        if suggested_tools:
            top_suggestions = [tool[0] for tool in suggested_tools[:3]]
            TEMPLATE = 'Consider using one of these tools: {{TOOLS}}. Format: {"name": "tool_name", "args": <tool_args_in_json_format>}'
            return TEMPLATE.replace("{{TOOLS}}", ", ".join(top_suggestions))
        else:
            TEMPLATE = 'Please choose from available tools: {{TOOLS}}. Format: {"name": "tool_name", "args": <tool_args_in_json_format>}'
            return TEMPLATE.replace("{{TOOLS}}", self._tool_names())

    def _is_task_complete(self, response: str) -> bool:
        """Check if the task appears to be complete"""
        completion_indicators = [
            "task completed",
            "finished",
            "done",
            "successfully executed",
            "analysis complete",
            "result:",
            "summary:",
            "conclusion:",
            "final answer",
            "task accomplished",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    REFLECTION_TEMPLATE = """\
Please reflect on your progress so far:

1. **Status Check**: What have you accomplished toward the task goal?
2. **Next Action**: What specific action should you take next?
3. **Tool Usage**: Which tool would be most appropriate for the next step?
4. **Expected Outcome**: What do you expect to achieve with this action?

Available tools: {{TOOL_NAMES}}

Use the format:
Thought: [Your analysis and reasoning]
Action: {"name": "<tool_name>", "args": <tool_args_in_json_format>}

Iteration: {{CURRENT_ITERATION}}/{{MAX_ITERATIONS}}
"""

    def _generate_reflection_prompt(self, iteration: int) -> str:
        """Generate a reflection prompt for the agent"""
        return (
            self.REFLECTION_TEMPLATE.replace("{{TOOL_NAMES}}", self._tool_names())
            .replace("{{CURRENT_ITERATION}}", str(iteration + 1))
            .replace("{{MAX_ITERATIONS}}", str(self.max_iterations))
        )

    def _generate_final_summary(self, conversation_context: dict[str, Any]) -> str:
        """Generate a final summary of the task execution"""
        steps = conversation_context.get("steps", [])
        if not steps:
            return "Task execution completed without specific steps recorded."

        summary = f"**{self.name} Task Summary:**\n\n"
        summary += f"**Original Task**: {conversation_context['task']}\n\n"
        summary += f"**Available Tools**: {self._tool_names()}\n\n"
        summary += f"**Execution Steps** ({len(steps)} total):\n"

        for i, step in enumerate(steps, 1):
            summary += f"\n{i}. "
            if step.get("thought"):
                summary += f"Thought: {step['thought']}\n   "

            if step.get("action"):
                # action is the string action_summary ("tool with args: ...")
                summary += f"Action: {step['action']}\n   "

            if step.get("response"):
                response = step["response"][:200] + "..." if len(step["response"]) > 200 else step["response"]
                summary += f"**Result**: {response}\n"

            if step.get("error"):
                summary += f"**Error**: {step['error']}\n"

        if conversation_context.get("status") == "completed":
            summary += "\n✅ **Status**: Task completed successfully"
        else:
            summary += f"\n⚠️ **Status**: Task execution incomplete after {self.max_iterations} iterations"

        return summary

    def _tool_names(self) -> str:
        """Get the names of the available tools"""
        return ", ".join(tool.name for tool in self.tools)
