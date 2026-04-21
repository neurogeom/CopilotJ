# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from typing import Annotated

import pydantic
from langfuse import Langfuse

import copilotj.multiagent.py_tools as py_tools
import copilotj.multiagent.tools as tools
import copilotj.multiagent.workflow_tools as workflow_tools
from copilotj.core import (
    UI,
    ChatAgent,
    FunctionTool,
    Handoff,
    HandoffFunctionTool,
    ModelClient,
    ModelResponse,
    ModelSyntaxError,
    Pattern,
    TextMessage,
    Tool,
    new_model_client,
)
from copilotj.multiagent.agent_loader import load_agent_configs
from copilotj.multiagent.Executor import Executor
from copilotj.multiagent.kb_tools import _load_macro_plugin_names, kb_build, kb_retrieve, rebuild_registry
from copilotj.multiagent.leader_prompts import (
    PROMPT_LEADER,
    PROMPT_TOOL_DELETE_WORKFLOW,
    PROMPT_TOOL_EXECUTE_PYTHON_SCRIPT,
    PROMPT_TOOL_EXECUTE_WORKFLOW,
    PROMPT_TOOL_EXPORT_WORKFLOW,
    PROMPT_TOOL_FOLDER_SUMMARY,
    PROMPT_TOOL_GET_WORKFLOW,
    PROMPT_TOOL_IMAGEJ_PERCEPTION,
    PROMPT_TOOL_KB_RETRIEVE,
    PROMPT_TOOL_LABEL_IMAGE,
    PROMPT_TOOL_LIST_WORKFLOWS,
    PROMPT_TOOL_RUN_MACRO,
    PROMPT_TOOL_SAVE_WORKFLOW,
    PROMPT_TOOL_USER_MANIPULATION,
    build_available_specialized_agents_prompt,
    build_tool_prompt,
    make_steps_prompt,
    make_summary_prompt,
)
from copilotj.multiagent.tools import system_info
from copilotj.plugin import ClientPluginAPI
from copilotj.util import ReActChatCompletionClient

__all__ = ["LeaderDriven"]


class LeaderAgent(ChatAgent):
    def __init__(
        self,
        name: str,
        description: str,
        *,
        agents: dict[str, Executor] | None = None,
        model_client: ModelClient,
        apis: ClientPluginAPI,
    ):
        super().__init__(name, description, model_client=model_client)

        self.chat_history: list[dict[str, str | int]] = []

        self.plugin_tools = tools.PluginTools(apis)

        # Store the combined tools and agents
        self.tools: list[Tool] = [
            FunctionTool(
                self.plugin_tools.imagej_perception, PROMPT_TOOL_IMAGEJ_PERCEPTION, display_name="ImageJ Perception"
            ),
            FunctionTool(self.plugin_tools.run_macro, PROMPT_TOOL_RUN_MACRO),
            FunctionTool(tools.folder_summary, PROMPT_TOOL_FOLDER_SUMMARY),
            FunctionTool(tools.execute_python_script, PROMPT_TOOL_EXECUTE_PYTHON_SCRIPT),
            FunctionTool(self.plugin_tools.label_image, PROMPT_TOOL_LABEL_IMAGE),
            # Knowledge bank tools
            FunctionTool(kb_retrieve, PROMPT_TOOL_KB_RETRIEVE, display_name="Knowledge Bank Retrieve"),
            self._mk_tool_user_manipulate(),
            self._mk_tool_delegate(),
            # Workflow management tools
            FunctionTool(self.save_workflow, PROMPT_TOOL_SAVE_WORKFLOW),
            FunctionTool(workflow_tools.list_workflows, PROMPT_TOOL_LIST_WORKFLOWS),
            FunctionTool(workflow_tools.get_workflow, PROMPT_TOOL_GET_WORKFLOW),
            FunctionTool(workflow_tools.delete_workflow, PROMPT_TOOL_DELETE_WORKFLOW),
            FunctionTool(workflow_tools.export_workflow, PROMPT_TOOL_EXPORT_WORKFLOW),
            FunctionTool(workflow_tools.execute_workflow, PROMPT_TOOL_EXECUTE_WORKFLOW),
        ]
        self.agents = agents if agents else {}

    async def handle_request(self, main_task: str, *, trace_ctx: Langfuse | None = None) -> ModelResponse:
        self.imagej_windowInfo_text = await self.plugin_tools.imagej_windowInfo()
        # Generate the list of available tools and agents for the prompt
        mytools = self.tools
        tool_list = build_tool_prompt(mytools) + "\n" + build_available_specialized_agents_prompt(self.agents)

        # Generate dynamic plugin list for the leader prompt
        macro_plugins = _load_macro_plugin_names() or ["StarDist", "TrackMate", "CLIJ2"]
        plugins_text = ", ".join(macro_plugins)

        system_info_text = await system_info()

        leader_prompt = (
            PROMPT_LEADER.replace("{MAIN_TASK}", main_task)
            .replace("{CHAT_HISTORY}", json.dumps(self.chat_history))
            .replace("{TOOL_LIST}", tool_list)
            .replace("{DEFAULT_IMAGE_PATH}", str(py_tools.get_project_temp_dir()))
            .replace("{SPECIAL_PLUGIN}", plugins_text)
            .replace("{SYSTEM_INFO}", system_info_text)
            .replace("{IMAGEJ_WINDOWINFO}", self.imagej_windowInfo_text)
        )
        # self._runtime.log_info(f"Prompt for {self.name}:\n{leader_prompt}")
        return await self._create(TextMessage(role="user", text=leader_prompt), tools=mytools, trace_ctx=trace_ctx)

    async def user_manipulate(
        self, instructions: Annotated[str, "Clear, step-by-step instructions for the user."]
    ) -> str:
        """Pauses the process and asks the user to perform a manual action in ImageJ.

        The user can press Enter to confirm completion or type feedback before pressing Enter.

        Returns:
            Confirmation message, potentially including user feedback if provided.
        """
        user_input = await self.request_user_manipulate(instructions)

        if not user_input:
            return "User confirmed completion of the manual action."

        return f"Feedback from user: '{user_input}'."

    def _mk_tool_user_manipulate(self) -> Tool:
        def get_handoff(id: str, args: pydantic.BaseModel) -> Handoff:
            return Handoff(id=id, name="user", message=getattr(args, "instructions"))

        return HandoffFunctionTool(self.user_manipulate, PROMPT_TOOL_USER_MANIPULATION, get_handoff=get_handoff)

    async def save_workflow(self, workflow_name: str, tags: str | None = None, dialog_id: int | None = None) -> str:
        if not self.chat_history:
            return "No workflow in chat history to save."

        target = None
        if dialog_id is None:
            target = self.chat_history[-1]
        else:
            for item in reversed(self.chat_history):
                if item.get("dialog") == dialog_id:
                    target = item
                    break

        if target is None:
            ids = ", ".join(str(x.get("dialog")) for x in self.chat_history if x.get("dialog") is not None)
            return f"Dialog {dialog_id} not found. Available dialogs: {ids}"

        save_status = await workflow_tools.save_workflow_from_steps(
            workflow_name, target.get("steps"), target.get("assistant"), tags
        )
        return f"Workflow saved for dialog {target.get('dialog')}: {save_status}"

    async def delegate_task(self, agent: str, task: str) -> str:
        if agent not in self.agents:
            raise ValueError(f"Agent '{agent}' not found in the available list.")

        agent_instance = self.agents[agent]
        self.log_info(f"[CALL] Calling Agent: {agent} | Params/Task: {task}")
        self.imagej_windowInfo_text = await self.plugin_tools.imagej_windowInfo()
        
        return await agent_instance.run(task + self.imagej_windowInfo_text + "Previous Chat History: \n" + json.dumps(self.chat_history))

    def _mk_tool_delegate(self) -> Tool:
        def get_handoff(id: str, args: pydantic.BaseModel) -> Handoff:
            return Handoff(id=id, name=getattr(args, "agent"), message=getattr(args, "task"))

        return HandoffFunctionTool(self.delegate_task, "Delegating task to specialized agents", get_handoff=get_handoff)

    async def optimize_prompt(self, user_prompt: str) -> str:
        """Optimize user prompt using LLM.

        This is a read-only operation with no side effects - it does NOT modify
        chat_history or any agent state.

        Args:
            user_prompt: The original user prompt to optimize

        Returns:
            Optimized prompt string, or original if optimization fails
        """
        # Get ImageJ window info for context (only if available)
        # This is read-only, no side effects
        window_info = ""
        try:
            if hasattr(self, "plugin_tools"):
                window_info = await self.plugin_tools.imagej_windowInfo()
        except Exception:
            window_info = ""

        # Build filtered chat history (only final answers, no intermediate steps)
        history_context = ""
        if self.chat_history:
            # Filter: only include entries with 'assistant' field (final answers)
            # Exclude: entries with only 'thought', 'steps', 'response', etc.
            filtered_history = [
                {k: v for k, v in item.items() if k == "assistant" or k == "user"}
                for item in self.chat_history[-3:]  # Only last 3 for brevity
                if "assistant" in item and item["assistant"]
            ]

            if filtered_history:
                history_text = "\n".join([
                    f"User: {h.get('user', '')}\nAssistant: {h['assistant'][:100]}..."
                    for h in filtered_history
                ])
                history_context = f"\n## Recent Conversation\n{history_text}"
            else:
                history_context = ""

        # Build context section
        context_parts = []
        if window_info:
            context_parts.append(f"## ImageJ Window\n{window_info}")
        if history_context:
            context_parts.append(history_context)

        context_section = "\n".join(context_parts) if context_parts else ""

        system_prompt = f"""You are a prompt optimization assistant for an AI that specializes in image analysis and ImageJ processing.

## Your Task
Improve the user's prompt to make it more clear, specific, and actionable.

{context_section}

## Optimization Rules
1. **Be Specific**: Add relevant technical details (e.g., image type, analysis method, output format)
2. **Maintain Intent**: Keep the user's original goal and language
3. **Add Technical Context**: Include terms like "segmentation", "threshold", "ROI", "measurement" when relevant
4. **Be Concise**: Remove fluff while ensuring clarity
5. **Output Format**: Reply with ONLY the optimized prompt text, no explanations, no preamble

## Examples
- "count cells" → "Count nuclei cells in DAPI fluorescence image using automated segmentation"
- "make it red" → "Apply a red color lookup table (LUT) to display the image in red color"
- "blur" → "Apply Gaussian blur with sigma=1.5 to reduce noise"

User prompt to optimize:
{user_prompt}
"""

        from copilotj.core.message import TextMessage

        try:
            response = await self._client.create([TextMessage(role="user", text=system_prompt)])
            optimized = response.content if response.content else user_prompt
            self.log_info(f"Prompt optimized: '{user_prompt[:50]}...' -> '{optimized[:50]}...'")
            return optimized
        except Exception as e:
            self.log_error(f"Failed to optimize prompt: {e}")
            return user_prompt


class LeaderDriven(Pattern):
    def __init__(
        self,
        apis: ClientPluginAPI,
        *,
        ui: UI | None = None,
        max_steps_before_summary: int = 8,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__("copilotj.leader_driven", ui=ui)

        self.model_client = new_model_client(model=model, api_key=api_key, base_url=base_url)
        # wrap the model client to handle ReAct-style responses
        wrapped_model_client = ReActChatCompletionClient(self.model_client)

        self.dialog_counter = 1
        self.max_steps_before_summary = max_steps_before_summary

        self._summarize_task: asyncio.Task[None] | None = None

        # --- Load Agents and Define Tools ---
        rebuild_registry()  # Rebuild registry to include any new workflow files
        self.log_info("Loading agents from agent_configs...")

        # Load agents defined in configurations
        try:
            self.specialized_agents = load_agent_configs(model_client=wrapped_model_client)
            for agent in self.specialized_agents.values():
                self.register(agent)

            self.log_info(f"Loaded agents: {list(self.specialized_agents.keys())}")

            for name, agent_instance in self.specialized_agents.items():
                if not (
                    hasattr(agent_instance, "run")
                    and hasattr(agent_instance, "description")
                    and hasattr(agent_instance, "name")
                ):
                    self.log_error(
                        f"Loaded agent '{name}' might be missing required attributes (run, description, name)."
                    )

        except Exception as e:
            self.log_error(f"Error loading agents from 'agent_configs': {e}")
            self.specialized_agents = {}

        # --- Initialize Leader Agent ---
        self.leader_agent = LeaderAgent(
            name="Leader Agent",
            description="Orchestrates tasks by reasoning and calling tools or specialized agents.",
            model_client=wrapped_model_client,
            agents=self.specialized_agents,
            apis=apis,
        )
        self.log_info("Leader Agent initialized.")

    def update_config(self, *, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        if model is not None or api_key is not None or base_url is not None:
            model = model or self.model_client.get_model()
            api_key = api_key or self.model_client.get_api_key()
            self.model_client = new_model_client(model=model, api_key=api_key, base_url=base_url)
            self.leader_agent.set_model_client(self.model_client)
            for agent in self.specialized_agents.values():
                agent.set_model_client(self.model_client)

    def abort(self) -> None:
        """Abort any ongoing dialog or task.

        NOTE: This will may make the pattern unstable.
        """
        self.leader_agent.abort()

    async def optimize_prompt(self, user_prompt: str) -> str:
        """Optimize user prompt using the raw model client (not ReAct-wrapped).

        This is a read-only operation with no side effects.

        Args:
            user_prompt: The original user prompt to optimize

        Returns:
            Optimized prompt string, or original if optimization fails
        """
        from copilotj.core.message import TextMessage

        # Get ImageJ window info for context (only if available)
        window_info = ""
        try:
            window_info = await self.leader_agent.plugin_tools.imagej_windowInfo()
        except Exception:
            window_info = ""

        # Build filtered chat history (only final answers, no intermediate steps)
        history_context = ""
        if self.leader_agent.chat_history:
            filtered_history = [
                {k: v for k, v in item.items() if k == "assistant" or k == "user"}
                for item in self.leader_agent.chat_history[-3:]
                if "assistant" in item and item["assistant"]
            ]
            if filtered_history:
                history_text = "\n".join([
                    f"User: {h.get('user', '')}\nAssistant: {h['assistant'][:100]}..."
                    for h in filtered_history
                ])
                history_context = f"\n## Recent Conversation\n{history_text}"

        # Build context section
        context_parts = []
        if window_info:
            context_parts.append(f"## ImageJ Window\n{window_info}")
        if history_context:
            context_parts.append(history_context)
        context_section = "\n".join(context_parts) if context_parts else ""

        system_prompt = f"""You are a prompt optimization assistant for an AI that specializes in image analysis and ImageJ processing.

## Your Task
Improve the user's prompt to make it more clear, specific, and actionable.

{context_section}

## Optimization Rules
1. **Be Specific**: Add relevant technical details (e.g., image type, analysis method, output format)
2. **Maintain Intent**: Keep the user's original goal and language
3. **Add Technical Context**: Include terms like "segmentation", "threshold", "ROI", "measurement" when relevant
4. **Be Concise**: Remove fluff while ensuring clarity
5. **Output Format**: Reply with ONLY the optimized prompt text, no explanations, no preamble

## Examples
- "count cells" → "Count nuclei cells in DAPI fluorescence image using automated segmentation"
- "make it red" → "Apply a red color lookup table (LUT) to display the image in red color"
- "blur" → "Apply Gaussian blur with sigma=1.5 to reduce noise"

User prompt to optimize:
{user_prompt}
"""
        try:
            # Use raw model_client (not ReAct-wrapped) to get plain text response
            response = await self.model_client.create([TextMessage(role="user", text=system_prompt)])
            optimized = response.content if response.content else user_prompt
            self.log_info(f"Prompt optimized: '{user_prompt[:50]}...' -> '{optimized[:50]}...'")
            return optimized
        except Exception as e:
            self.log_error(f"Failed to optimize prompt: {e}")
            return user_prompt

    async def summarize_dialog_context(self, dialog_context: dict, dialog_id: int | None = None):
        steps = dialog_context["steps"]
        if len(steps) <= self.max_steps_before_summary:
            return dialog_context, None

        self.log_info(f"[SUMMARY] Summarizing {len(steps)} steps into a detailed summary...")

        steps_text = json.dumps(steps, indent=2, ensure_ascii=False)

        summary_prompt = make_summary_prompt(dialog_context["task"], steps_text)
        steps_prompt = make_steps_prompt(dialog_context["task"], steps_text)

        try:
            response = await self.model_client.create([TextMessage(role="user", text=summary_prompt)])
            steps_response = await self.model_client.create([TextMessage(role="user", text=steps_prompt)])
            if response.content is None or steps_response.content is None:
                self.log_error("[ERROR] Failed to generate summary - empty response")
                return dialog_context, None

            summary = response.content.strip()
            steps = steps_response.content.strip()
            self.log_info("[SUCCESS] Successfully generated dialog context summary")

        except Exception as e:
            self.log_error(f"[ERROR] Error generating dialog context summary: {e}")
            return dialog_context, None

        if str(os.getenv("COPILOTJ_KB_AUTOSAVE")) != "1":
            return summary, steps

        # Persist to knowledge bank (non-blocking failure)
        try:
            kb_result = await kb_build(
                dialog=dialog_context,
                summary=summary,
                steps=steps,
                question=dialog_context["task"] if dialog_context.get("task") else None,
            )
            result_data = json.loads(kb_result)
            if result_data.get("status") != "ok":
                error_msg = result_data.get("error", "Unknown error")
                self.log_info(f"[WARNING] Dialog {dialog_id} knowledge processing failed: {error_msg}")
                return summary, steps

            created_files = []
            skipped_items = []

            for item in result_data.get("created", []):
                if "path" in item:
                    # Extract filename from path (works for both / and \ separators)
                    filename = os.path.basename(item["path"])
                    created_files.append(f"{item.get('type', 'file')}: {filename}")

            # Check for skipped items
            for item in result_data.get("skipped", []):
                reason = item.get("reason", "unknown")
                name = item.get("name", "unknown")
                skipped_items.append(f"{item.get('type', 'item')}: {name} ({reason})")

            if created_files:
                files_str = ", ".join(created_files)
                self.log_info(f"[SUCCESS] Successfully persisted dialog {dialog_id} to knowledge bank: {files_str}")

            elif skipped_items:
                skipped_str = ", ".join(skipped_items)
                self.log_info(f"[INFO] Dialog {dialog_id} processed - no new files created (skipped: {skipped_str})")

            else:
                self.log_info(f"[INFO] Dialog {dialog_id} processed - no new knowledge generated")

        except Exception as kb_e:
            self.log_error(f"[WARNING] KB build failed for dialog {dialog_id}: {kb_e}")

        finally:
            return summary, steps

    async def _background_summarize_and_store(self, dialog_id: int, task: str, dialog_context: dict) -> None:
        try:
            summary, steps = await self.summarize_dialog_context(dialog_context.copy(), dialog_id)
            self.leader_agent.chat_history.append(
                {"dialog": dialog_id, "user": task, "assistant": str(summary), "steps": steps}
            )
        except Exception as e:
            self.log_error(f"Background summarization failed: {e}")

    async def run(self, task: str, trace_ctx: Langfuse | None = None) -> None:
        topic_intro = f"[DIALOG #{self.dialog_counter}] User Task: {task}"
        self.log_info(topic_intro)

        dialog_context = {
            "task": task,
            "status": "in_progress",
            "last_tool_response": None,
            "steps": [],
        }

        tool_retry_counter = 0
        max_tool_retry = 3
        syntax_error_counter = 0
        max_syntax_errors = 3

        # TODO: add supervise back
        # parallel delegation to multiple agents
        #
        # if action.type == "supervise":
        #     assert isinstance(action.arguments, str), "Supervise params must be a JSON string"
        #
        #     try:
        #         tasks = json.loads(action.arguments)
        #     except json.JSONDecodeError:
        #         self.log_error("supervise params must be valid JSON")
        #         dialog_context["status"] = "failed"
        #         break
        #
        #     # Dispatch subtasks in parallel
        #     coros = {
        #         name: self.leader_agent.agents[name].run(sub)
        #         for name, sub in tasks.items()
        #         if name in self.leader_agent.agents
        #     }
        #     results = await asyncio.gather(*coros.values(), return_exceptions=True)
        #     team_responses = {
        #         name: (res if not isinstance(res, Exception) else f"Error: {res}")
        #         for name, res in zip(coros.keys(), results)
        #     }
        #     dialog_context["steps"].append({"supervise_results": team_responses})
        #     # Consolidate with LeaderAgent
        #     summary_prompt = (
        #         "Please consolidate the following team responses into a single reply to the user:\n"
        #         + json.dumps(team_responses, indent=2)
        #     )
        #     final = await self.leader_agent.handle_request(summary_prompt)
        #     self.log_info(f"🎯 FINAL RESULT: {final}")
        #     dialog_context["steps"].append({"final_answer": final})
        #     dialog_context["status"] = "completed"
        #     break

        while True:
            if self._summarize_task is not None:
                await self._summarize_task

            planning_context = f"""
User: "{task}".
So far, your progress includes the following steps:
{json.dumps(dialog_context["steps"], indent=2)}
"""

            if dialog_context["last_tool_response"]:
                planning_context += f"""\n
Last Observation:
{dialog_context["last_tool_response"]}
"""

            # Let LeaderAgent decide the next step
            try:
                agent_resp = await self.leader_agent.handle_request(planning_context, trace_ctx=trace_ctx)
            except ModelSyntaxError as e:
                syntax_error_counter += 1
                self.log_error(f"LeaderAgent generated invalid ReAct syntax ({syntax_error_counter}/{max_syntax_errors}). Retrying...: {e.message}")
                dialog_context["steps"].append({"agent": str(e)})
                if syntax_error_counter >= max_syntax_errors:
                    self.log_error("Too many ReAct syntax errors. Aborting task.")
                    dialog_context["status"] = "failed"
                    break
                continue

            if not (agent_resp.content or agent_resp.reasoning_content or agent_resp.tool_calls):
                # Handle empty LLM response
                self.log_error("LeaderAgent returned empty response. Aborting task.")
                dialog_context["status"] = "failed"
                dialog_context["steps"].append({"agent": "Error: LLM failed to respond."})
                break

            # Check for Final Answer
            if agent_resp.content:
                final_answer = agent_resp.content
                dialog_context["status"] = "completed"
                dialog_context["steps"].append(
                    {
                        # Include thought if present "final_answer": final_answer,
                        "thought": agent_resp.reasoning_content or "N/A",
                    }
                )
                self.log_info(f"[FINAL RESULT] Dialog {self.dialog_counter} Finished!\n{final_answer}")
                await self.dialog_changed(self.dialog_counter, "completed")

                dialog_context["status"] = "completed"
                break

            if not agent_resp.tool_calls:
                self.log_error(f"No valid action found in agent response: {agent_resp.reasoning_content}")
                error = "Agent did not provide a valid tool/agent call."
                tool_retry_counter += 1
                dialog_context["steps"].append({"agent_response": agent_resp.reasoning_content, "error": error})
                continue

            # Handle supervise in one branch before tool/agent
            tool_call = agent_resp.tool_calls[0]  # TODO: handle multiple tool calls
            self.log_info(f"[CALL] Calling Tool: {tool_call.tool.name} | Params/Task: {tool_call.args}")
            try:
                resp = await self.leader_agent._call_tool(tool_call)
                tool_retry_counter = 0  # Reset retry counter on success

            except Exception as e:
                self.log_error(f"[ERROR] Error executing '{tool_call.tool.name}': {e}")
                resp = f"""
You tried solving the task: "{task}" but encountered several issues.
Tool/Agent execution failed for '{tool_call.tool.name}' with parameters '{tool_call.args}'.
Error: {str(e)}
Please reflect on what went wrong. 
- Identify any mistakes you made.
- Suggest how you might improve your next plan.
- Then generate a new Thought + Action to try again.
"""
                tool_retry_counter += 1

            dialog_context["last_tool_response"] = str(resp)
            dialog_context["steps"].append(
                {
                    "thought": agent_resp.reasoning_content,
                    "name": tool_call.tool.name,
                    "args": tool_call.args.model_dump(),
                    "response": resp,
                }
            )

            if tool_retry_counter >= max_tool_retry:
                dialog_context["status"] = "failed"
                dialog_context["steps"].append(
                    {"agent": "[WARNING] Too many failed tool calls. Task will be handed to the user."}
                )
                break

        # Offload summarization to background to avoid blocking the frontend
        current_dialog_id = self.dialog_counter
        self.dialog_counter += 1
        self._summarize_task = asyncio.create_task(
            self._background_summarize_and_store(current_dialog_id, task, dialog_context)
        )
