# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import itertools
import json
import re
from typing import Any, AsyncGenerator, Iterable, Sequence, override
from uuid import uuid4

import pydantic

from copilotj.core import (
    FinishReasons,
    ImageMessage,
    ModelClient,
    ModelResponse,
    ModelResponseChunk,
    ModelSyntaxError,
    TextMessage,
    Tool,
    ToolCall,
)

__all__ = ["ReActChatCompletionClient", "ModelSyntaxError"]


class _StreamingState(enum.Enum):
    Init = enum.auto()
    BeforeThought = enum.auto()  # Trim leading space
    Thought = enum.auto()
    BeforeAction = enum.auto()  # Trim leading space
    Action = enum.auto()
    BeforeFinal = enum.auto()  # Trim leading space
    Final = enum.auto()


class ReActChatCompletionClient(ModelClient):
    """Wrapped client to handle ReAct-style responses."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        kw_thought: str = "Thought",
        kw_action: str = "Action",
        kw_final: str = "Final Answer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_client = model_client

        self._kw_thought = kw_thought
        self._kw_action = kw_action
        self._kw_final = kw_final

        # compile regex patterns
        ends = "".join((":", r"\n"))
        re_thought = rf"{kw_thought}[{ends}]"
        re_action = rf"{kw_action}[{ends}]"
        re_final = rf"{kw_final}[{ends}]"
        flags = re.IGNORECASE | re.DOTALL

        # create
        self._pattern_thought = re.compile(rf"{re_thought}(.*?)(?={re_action}|{re_final}|$)", flags)
        self._pattern_action = re.compile(rf"{re_action}(.*?)(?={re_final}|$)", flags)
        self._pattern_final = re.compile(rf"{re_final}\s*(.+)$", flags)

        # create_stream
        self._pattern_thought_kw = re.compile(rf"{re_thought}\s*", flags)
        self._pattern_action_kw = re.compile(rf"{re_action}\s*", flags)
        self._pattern_final_kw = re.compile(rf"{re_final}\s*", flags)
        self._pattern_action_final_kw_prefix = _build_last_line_prefix_regex(kw_action, kw_final)

    @override
    def get_api_key(self) -> str | None:
        return self._model_client.get_api_key()

    @override
    def get_model(self) -> str:
        return self._model_client.get_model()

    @override
    async def create(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ModelResponse:
        # NOTE: dont send tools since we are parsing the response manually
        response = await self._model_client.create(messages, extra_args=extra_args)
        if response.reasoning_content is not None or response.content is None:
            return response

        # Use compiled regex patterns
        text = response.content.strip()

        thought_match = self._pattern_thought.search(text)
        action_match = self._pattern_action.search(text)
        final_match = self._pattern_final.search(text)
        thought = thought_match and thought_match.group(1).strip()

        # Fallback: if no structured reasoning or action found, return raw content as final answer
        if thought_match is None and action_match is None and final_match is None:
            return ModelResponse(
                content=text,
                reasoning_content=None,
                tool_calls=[],
                finish_reason=response.finish_reason,
            )

        try:
            tool_call = self._parse_action(text, tools=tools or []) if action_match is not None else None
        except ModelSyntaxError as e:
            e.chat_completion = ModelResponse(
                content=text,
                reasoning_content=thought,
                tool_calls=[],
                finish_reason=response.finish_reason,
            )
            raise e

        return ModelResponse(
            content=final_match.group(1).strip() if final_match else None,
            reasoning_content=thought,
            tool_calls=[tool_call] if tool_call else None,
            finish_reason=response.finish_reason,
        )

    @override
    async def create_stream(
        self,
        messages: Sequence[TextMessage | ImageMessage],
        *,
        tools: list[Tool] | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> AsyncGenerator[ModelResponseChunk | ToolCall, None]:
        tools = tools or []

        # NOTE: dont send tools since we are parsing the response manually
        stream = self._model_client.create_stream(messages, tools=None, extra_args=extra_args)

        buffer = ""
        state = _StreamingState.Init
        last: ModelResponseChunk | ToolCall | None = None
        finish_reason: FinishReasons = "unknown"
        async for chunk in stream:
            if isinstance(chunk, ToolCall):
                last = chunk
                yield chunk
                continue

            if chunk.reasoning_content is not None:
                last = ModelResponseChunk(reasoning_content=chunk.reasoning_content, content=None, finish_reason=None)
                yield last

            if chunk.content is not None:
                buffer += chunk.content
                match state:
                    case _StreamingState.Init:
                        state, resp, buffer = self._fsm_init(buffer, tools=tools)

                    case _StreamingState.BeforeThought:
                        state, resp, buffer = self._fsm_before_status(state, buffer, tools=tools)

                    case _StreamingState.Thought:
                        state, resp, buffer = self._fsm_thought(buffer, tools=tools)

                    case _StreamingState.BeforeAction:
                        state, resp, buffer = self._fsm_before_status(state, buffer, tools=tools)

                    case _StreamingState.Action:
                        state, resp, buffer = self._fsm_action(buffer, tools=tools)

                    case _StreamingState.BeforeFinal:
                        state, resp, buffer = self._fsm_before_status(state, buffer, tools=tools)

                    case _StreamingState.Final:
                        state, resp, buffer = self._fsm_final(buffer)

                    case _:
                        raise ValueError(f"Unexpected state: {state}")

                for item in resp:
                    last = item
                    yield last

            # dont send finish_reason since we may have buffer left
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        # flush buffer
        if buffer.strip():
            match state:
                case _StreamingState.Init:
                    # If we are still in Init state, we can yield the buffer as a final answer
                    last = ModelResponseChunk(reasoning_content=None, content=buffer, finish_reason=None)
                    yield last

                case _StreamingState.Thought:
                    last = ModelResponseChunk(reasoning_content=buffer, content=None, finish_reason=None)
                    yield last

                case _StreamingState.Action:
                    last = self._parse_action(buffer, tools=tools)
                    if last is None:
                        raise ModelSyntaxError(f"Failed to parse action from text: {buffer}")

                    yield last

                case _StreamingState.Final:
                    last = ModelResponseChunk(reasoning_content=None, content=buffer, finish_reason=None)
                    yield last

                case _:
                    raise ValueError(f"Unexpected state: {state}")

        # generate a finish reason
        match last:
            case ModelResponseChunk():
                yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason=finish_reason)

            case ToolCall():
                yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason="tool_calls")

            case None:
                yield ModelResponseChunk(reasoning_content=None, content=None, finish_reason="unknown")

            case _:
                raise ValueError(f"Unexpected last item type: {type(last)}")

    def _fsm_init(
        self, buffer: str, *, tools: list[Tool]
    ) -> tuple[_StreamingState, Iterable[ModelResponseChunk | ToolCall], str]:
        thought_match = self._pattern_thought_kw.search(buffer)
        if thought_match:
            if thought_match.start() > 0 and buffer[: thought_match.start()].strip():
                raise ModelSyntaxError(
                    f"Unexpected content before '{self._kw_thought}' keyword: " + buffer[: thought_match.start()]
                )

            return self._fsm_before_status(_StreamingState.BeforeThought, buffer[thought_match.end() :], tools=tools)

        for pattern_kw, next in [
            (self._pattern_action_kw, _StreamingState.BeforeAction),
            (self._pattern_final_kw, _StreamingState.BeforeFinal),
        ]:
            match = pattern_kw.search(buffer)
            if not match:
                continue

            head = buffer[: match.start()]
            if head.strip():
                # through these content does not start with thought keyword, we still treat it as a thought
                return self._fsm_before_status(_StreamingState.BeforeThought, buffer, tools=tools)

            return self._fsm_before_status(next, buffer[match.end() :], tools=tools)

        # Buffer is likely incomplete, hold off
        return _StreamingState.Init, (), buffer

    def _fsm_thought(
        self, buffer: str, *, tools: list[Tool]
    ) -> tuple[_StreamingState, Iterable[ModelResponseChunk | ToolCall], str]:
        for pattern_kw, next in [
            (self._pattern_action_kw, _StreamingState.BeforeAction),
            (self._pattern_final_kw, _StreamingState.BeforeFinal),
        ]:
            match = pattern_kw.search(buffer)
            if not match:
                continue

            head = buffer[: match.start()]
            state, rest_resp, rest = self._fsm_before_status(next, buffer[match.end() :], tools=tools)
            resp = ModelResponseChunk(reasoning_content=head, content=None, finish_reason=None)
            return state, itertools.chain([resp], rest_resp), rest

        # check if buffer ends with the start of a keyword, hold off
        end_with_kw = self._pattern_action_final_kw_prefix.search(buffer)
        if end_with_kw:
            rest = buffer[end_with_kw.start() :]
            buffer = buffer[: end_with_kw.start()]
        else:
            rest = ""

        chunks = (
            (ModelResponseChunk(reasoning_content=buffer, content=None, finish_reason=None),) if len(buffer) > 0 else ()
        )
        return _StreamingState.Thought, chunks, rest

    def _fsm_action(
        self, buffer: str, *, tools: list[Tool]
    ) -> tuple[_StreamingState, Iterable[ModelResponseChunk | ToolCall], str]:
        final_match = self._pattern_final_kw.search(buffer)
        if final_match is None:
            # Buffer is likely incomplete, hold off
            return _StreamingState.Action, [], buffer

        head = buffer[: final_match.start()]
        action = self._parse_action(head, tools=tools)
        if action is None:
            raise ModelSyntaxError(f"Failed to parse action from text: {head}")

        rest = buffer[final_match.end() :]
        if not rest:
            return _StreamingState.Final, (action,), ""

        state, rest_resp, rest = self._fsm_before_status(_StreamingState.BeforeFinal, rest, tools=tools)
        return state, itertools.chain((action,), rest_resp), rest

    def _fsm_final(self, buffer: str) -> tuple[_StreamingState, Iterable[ModelResponseChunk | ToolCall], str]:
        # yield all content
        if buffer:
            msgs = [ModelResponseChunk(reasoning_content=None, content=buffer, finish_reason=None)]
        else:
            msgs = []

        return _StreamingState.Final, msgs, ""

    def _fsm_before_status(
        self, current: _StreamingState, buffer: str, *, tools: list[Tool]
    ) -> tuple[_StreamingState, Iterable[ModelResponseChunk | ToolCall], str]:
        buffer = buffer.lstrip(" \n")  # remove leading space or newline
        if len(buffer) == 0:
            return current, (), ""  # nothing to process, hold off

        match current:
            case _StreamingState.BeforeThought:
                return self._fsm_thought(buffer, tools=tools)

            case _StreamingState.BeforeAction:
                return self._fsm_action(buffer, tools=tools)

            case _StreamingState.BeforeFinal:
                return self._fsm_final(buffer)

            case _:
                raise ValueError(f"Unexpected state: {current}. Cannot process before status.")

    def _parse_action(self, text: str, *, tools: list[Tool]) -> ToolCall | None:
        matches = self._pattern_action.search(text)
        if matches is not None:
            text = matches.group(1).strip()

        try:
            tool_call = _extract_json_tool_call(text)

        except ModelSyntaxError as e:
            raise ModelSyntaxError(f"Failed to extract action from text: {text}. Error: {e.message}")

        tool = next((t for t in tools if t.name == tool_call.name), None)
        if tool is None:
            # Try case-insensitive match
            name = tool_call.name.lower().replace(" ", "_")
            tool = next((t for t in tools if t.name.lower().replace(" ", "_") == name), None)

        if tool is None:
            names = ", ".join(t.name for t in tools) if tools else "no tools provided"
            raise ModelSyntaxError(f"Tool '{tool_call.name}' not found in provided tools: {names}")

        args_type = tool.args_type()
        try:
            args = args_type.model_validate(tool_call.args)
            return ToolCall(id=str(uuid4()), tool=tool, args=args)
        except pydantic.ValidationError as e:
            raise ModelSyntaxError(f"Failed to validate action arguments for tool '{tool_call.name}': {e}")


@dataclasses.dataclass
class ReActToolCall:
    name: str
    args: dict[str, Any]


def _extract_json_tool_call(text: str) -> ReActToolCall:
    code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    code_block_match = code_block_pattern.search(text.strip())
    if code_block_match:
        text = code_block_match.group(1).strip()

    try:
        parsed = json.loads(text.strip())
        if not isinstance(parsed, dict):
            raise ModelSyntaxError(f"Parsed Action is not a dictionary: {parsed}")

    except json.JSONDecodeError as e1:
        try:
            # Attempt basic fixes: trailing commas, single quotes
            text_fixed = re.sub(r",\s*([\}\]])", r"\1", text)
            text_fixed = text_fixed.replace("'", '"')
            parsed = json.loads(text_fixed.strip())

        except json.JSONDecodeError as e2:
            # Try to extract the action using regex patterns as fallback
            try:
                return _extract_action_with_regex(text.strip())

            except ValueError as e3:
                raise ModelSyntaxError(f"Failed to parse Action JSON: {text}. Errors: {e1}, {e2}, {e3}") from e3

        except Exception as e3:
            raise ModelSyntaxError(f"Unexpected error parsing Action JSON: {text}. Error: {e3}") from e3

    tool_name = parsed.get("name")
    if tool_name is None or not isinstance(tool_name, str):
        raise ModelSyntaxError(f"Action JSON must contain a 'name' field: {parsed}")

    tool_args = parsed.get("args", {})
    return ReActToolCall(name=tool_name, args=tool_args)


def _extract_action_with_regex(text: str) -> ReActToolCall:
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)

    if not name_match:
        raise ModelSyntaxError("Could not extract required 'name' fields from malformed JSON")

    action_name = name_match.group(1)

    args = None
    if args is None:
        params_pattern = r'"args"\s*:\s*"(.*?)"\s*}'
        params_match = re.search(params_pattern, text, re.DOTALL)
        if params_match:
            args_content = params_match.group(1)
            args_content = args_content.replace('\\"', '"')
            args = args_content

    if args is None:
        params_pattern_alt = r'"args"\s*:\s*"(.*)"'
        params_match_alt = re.search(params_pattern_alt, text, re.DOTALL)
        if params_match_alt:
            args_content = params_match_alt.group(1)
            # Remove any trailing "}
            args_content = re.sub(r'"\s*}?\s*$', "", args_content)
            args_content = args_content.replace('\\"', '"')
            args = args_content

    if args is not None:
        try:
            args = json.loads(args)

        except json.JSONDecodeError as e:
            raise ModelSyntaxError(f"Failed to parse 'args' JSON: {args}. Error: {e}")

        if not isinstance(args, dict):
            raise ModelSyntaxError(f"Parsed 'args' is not a dictionary: {args}")

    else:
        args = {}

    return ReActToolCall(name=action_name, args=args)


def _build_last_line_prefix_regex(*words: str) -> re.Pattern:
    """Build a regex pattern to match any prefix of multiple keywords

    Args:
        words: ["Action", "Final Answer"] => match A / Actio / Fin / Final Ans ...

    Return:
        Compiled regex pattern that matches any prefix of the given words,
    """

    def word_to_prefix_regex(word: str) -> str:
        assert word, "Word must not be empty"
        # Action -> A(?:c(?:t(?:i(?:o(?:n)?)?)?)?)?
        return word[0] + "".join(f"(?:{char}?" for i, char in enumerate(word[1:])) + ")?" * (len(word) - 1)

    pattern_parts = [word_to_prefix_regex(w) for w in words]
    # Combine patterns with non-capturing group and ignore case: ^(ActionPrefix|FinalAnswerPrefix)\Z
    pattern = r"(?i)^(" + "|".join(pattern_parts) + r")\Z"
    return re.compile(pattern)
