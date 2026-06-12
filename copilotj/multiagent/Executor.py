# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from copilotj.core import ChatAgent, ModelClient, ModelSyntaxError, TextMessage, Tool
from copilotj.multiagent.leader_prompts import build_tool_prompt
from copilotj.util import ReActChatCompletionClient

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

        self.system_prompt = self._build_enhanced_system_prompt(prompt)

    def _build_enhanced_system_prompt(self, base_prompt: str) -> str:
        """Build system prompt that includes available tools information from config"""
        if not self.tools:
            return base_prompt

        tools_info = build_tool_prompt(self.tools)

        if isinstance(self._client, ReActChatCompletionClient):
            # ReAct mode: include Action format instructions
            tools_usage = """\
## Tool Usage Format:
When you need to use a tool, format your action as json format:
Action: {"name": "<tool_name>", "args": <tool_args_in_json_format>}

## Tool Selection Guidelines:
- Choose the most appropriate tool based on the task requirements
- Provide clear and specific parameters for the tool
- If unsure about parameters, describe what you want to accomplish
"""
        else:
            # Native mode: tools are called directly, just list them
            tools_usage = """\
## Tool Selection Guidelines:
- Choose the most appropriate tool based on the task requirements
- Provide clear and specific parameters for the tool
- If unsure about parameters, describe what you want to accomplish
"""

        return "\n".join((base_prompt, tools_info, tools_usage))

    async def run(self, task: str) -> str:
        """Execute the agent task with tool usage and reflection"""
        self.log_info(f"🟢 {self.name} is executing: {task}")
        self.log_info(f"📋 Available tools: {[t.name for t in self.tools]}")

        try:
            # Initialize conversation with system prompt and task
            conversation_context = {
                "task": task,
                "status": "in_progress",
                "last_tool_response": None,
                "steps": [],
            }

            for iteration in range(self.max_iterations):
                self.log_info(f"Iteration {iteration + 1}/{self.max_iterations}")

                # Build prompt with context
                execution_context = self._build_execution_context(conversation_context, iteration)

                # Get agent response
                try:
                    response = await self._create(
                        TextMessage(role="system", text=self.system_prompt),
                        TextMessage(role="user", text=execution_context),
                        tools=self.tools,
                    )
                except ModelSyntaxError as e:
                    self.log_error("Agent generated invalid ReAct syntax. Retrying...")
                    conversation_context["last_tool_response"] = e.message

                    thought = ((e.chat_completion and e.chat_completion.reasoning_content) or "No thought provided",)
                    conversation_context["steps"].append(
                        {
                            "thought": thought,
                            "action": "",
                            "error": e.message,
                            "iteration": iteration + 1,
                        }
                    )
                    continue

                if not response.content and not response.tool_calls and not response.reasoning_content:
                    await self.print_error("No response from agent")
                    break

                # Check for Final Answer
                if response.content or self._is_task_complete(response.reasoning_content or ""):
                    conversation_context["steps"].append(
                        {
                            "thought": response.reasoning_content or "Task completed",
                            "final_answer": response.content,
                            "iteration": iteration + 1,
                        }
                    )
                    conversation_context["status"] = "completed"
                    return response.content or response.reasoning_content or ""

                # Parse and execute action if present
                if not response.tool_calls:
                    # No valid action found, add reflection prompt
                    suggestion = self._suggest_tool_based_on_context(response.reasoning_content or "", task)
                    conversation_context["last_tool_response"] = f"No valid action identified. {suggestion}"
                    conversation_context["steps"].append(
                        {
                            "thought": response.reasoning_content or "No clear thought provided",
                            "reflection_needed": True,
                            "tool_suggestion": suggestion,
                            "iteration": iteration + 1,
                        }
                    )
                    continue

                tool_call = response.tool_calls[0]  # TODO: handle multiple tool calls
                action_summary = f"{tool_call.tool.name} with args: {str(tool_call.args)[:100]}"
                self.log_info(f"🔧 Executing tool: {action_summary}...")
                try:
                    tool_response = await self._call_tool(tool_call)
                    self.tool_retry_counter = 0  # Reset on success

                    conversation_context["last_tool_response"] = tool_response
                    conversation_context["steps"].append(
                        {
                            "thought": response.reasoning_content or "No thought provided",
                            "action": action_summary,
                            "response": tool_response,
                            "iteration": iteration + 1,
                        }
                    )

                except Exception as e:
                    error_msg = f"❌ Error executing action: {str(e)}"
                    self.log_error(error_msg)
                    self.tool_retry_counter += 1

                    if self.tool_retry_counter >= self.max_tool_retry:
                        return f"❌ {self.name} failed after {self.max_tool_retry} attempts: {error_msg}"

                    conversation_context["last_tool_response"] = error_msg
                    conversation_context["steps"].append(
                        {
                            "thought": response.reasoning_content or "No thought provided",
                            "action": action_summary,
                            "error": error_msg,
                            "iteration": iteration + 1,
                        }
                    )

            return self._generate_final_summary(conversation_context)

        except Exception as e:
            self.log_error(f"Executor error: {e}")
            return f"❌ {self.name} encountered an error: {str(e)}"

    def _suggest_tool_based_on_context(self, thought: str, task: str) -> str:
        """Suggest appropriate tool based on context and descriptions"""
        context = f"{thought} {task}".lower()

        # Use the tool descriptions from config to make suggestions
        suggested_tools = []
        for tool in self.tools:
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
            if isinstance(self._client, ReActChatCompletionClient):
                template = 'Consider using one of these tools: {{TOOLS}}. Format: {"name": "tool_name", "args": <tool_args_in_json_format>}'
            else:
                template = "Consider using one of these tools: {{TOOLS}}."
            return template.replace("{{TOOLS}}", ", ".join(top_suggestions))
        else:
            if isinstance(self._client, ReActChatCompletionClient):
                template = 'Please choose from available tools: {{TOOLS}}. Format: {"name": "tool_name", "args": <tool_args_in_json_format>}'
            else:
                template = "Please choose from available tools: {{TOOLS}}."
            return template.replace("{{TOOLS}}", self._tool_names())

    def _build_execution_context(self, conversation_context: dict[str, Any], iteration: int) -> str:
        """Build the execution context for the current iteration"""
        context = f"""
Task: {conversation_context["task"]}

Current Status: {conversation_context["status"]}
Iteration: {iteration + 1}/{self.max_iterations}

Available Tools: {self._tool_names()}

Previous Steps:
{json.dumps(conversation_context["steps"], indent=2) if conversation_context["steps"] else "No previous steps"}
"""

        if conversation_context["last_tool_response"]:
            context += f"\n\nLast Tool Response:\n{conversation_context['last_tool_response']}"

        if iteration > 0:
            context += f"\n\n{self._generate_reflection_prompt(iteration)}"

        return context

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

    REFLECTION_TEMPLATE_REACT = """\
Please reflect on your progress so far:

1. **Status Check**: What have you accomplished toward the task goal?
2. **Next Action**: What specific action should you take next?
3. **Tool Usage**: Which tool would be most appropriate for the next step?
4. **Expected Outcome**: What do you expect to achieve with this action?

Available tools: {{TOOL_NAMES}}

Use the format:
Thought: [Your analysis and reasoning]
Action: {"name": "<tool_name>", "args": <tool_args_in_json_format>}"""

    REFLECTION_TEMPLATE_NATIVE = """\
Please reflect on your progress so far:

1. **Status Check**: What have you accomplished toward the task goal?
2. **Next Action**: What specific action should you take next?
3. **Tool Usage**: Which tool would be most appropriate for the next step?
4. **Expected Outcome**: What do you expect to achieve with this action?

Available tools: {{TOOL_NAMES}}

Use the format:
Thought: [Your analysis and reasoning]
Then call the appropriate tool directly."""

    def _generate_reflection_prompt(self, iteration: int) -> str:
        """Generate a reflection prompt for the agent"""
        template = (
            self.REFLECTION_TEMPLATE_REACT
            if isinstance(self._client, ReActChatCompletionClient)
            else self.REFLECTION_TEMPLATE_NATIVE
        )
        return template.replace("{{TOOL_NAMES}}", self._tool_names())

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
                summary += f"Action: Used {step['action'].get('name', 'unknown')} tool\n   "

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
