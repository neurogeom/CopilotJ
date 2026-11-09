# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import asyncio
from typing import Any, ClassVar, List, Literal, Sequence, override

import pydantic
import rich
import rich.live
import rich.panel

from copilotj.core.model_client import ToolCall

__all__ = [
    "UI",
    "CLI",
    "UIEventState",
    "UIEventPost",
    "UIEventError",
    "UIEventPostReasoningChunk",
    "UIEventPostContentChunk",
    "UIEventToolCall",
    "UIEventToolCalled",
    "UIEventToolCallResult",
    "UIEventHandoff",
    "UIEventRetry",
    "RetryInfo",
    "UIEvent",
    "Handoff",
]


type dumpable = str | int | float | bool | pydantic.BaseModel

ROLE_SYSTEM = "system"


class _Event[T: dumpable | Sequence[dumpable]](pydantic.BaseModel):
    _type: ClassVar[str | None]
    role: str  # "system" | "user" | "agent:foo"
    data: T

    def __init_subclass__(cls, *, type: str | None = None, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._type = type

    @pydantic.computed_field
    @property
    def type(self) -> str:
        assert self._type is not None
        return self._type


class UIEventContentMarkdown(pydantic.BaseModel):
    type: Literal["markdown"] = "markdown"
    markdown: str


type UIEventContent = UIEventContentMarkdown


class UIEventState(_Event[Literal["agent_speaking", "agent_finish", "confirmation_request"]], type="update:state"): ...


class UIEventPost(_Event[List[UIEventContentMarkdown]], type="new:post"): ...


class UIEventPostReasoningChunk(_Event[UIEventContentMarkdown], type="new:post_reasoning_chunk"): ...


class UIEventPostContentChunk(_Event[UIEventContentMarkdown], type="new:post_content_chunk"): ...


class UIEventError(_Event[str], type="new:error"): ...


class RetryInfo(pydantic.BaseModel):
    """Payload for an in-progress 429 auto-retry, surfaced to the UI (#96)."""

    attempt: int
    max_attempts: int
    wait_seconds: float
    reason: str


class UIEventRetry(_Event[RetryInfo], type="update:retry"): ...


class UIEventToolCall(_Event[ToolCall], type="new:tool_call"): ...


class ToolCallId(pydantic.BaseModel):
    id: str


class UIEventToolCalled(_Event[ToolCallId], type="new:tool_called"): ...


class ToolCallResult(pydantic.BaseModel):
    id: str
    type: Literal["success", "error"]
    result: list[UIEventContent]


class UIEventToolCallResult(_Event[ToolCallResult], type="new:tool_call_result"): ...


class Handoff(pydantic.BaseModel):
    id: str
    name: str
    message: str


class UIEventHandoff(_Event[Handoff], type="new:handoff"): ...


class DialogChange(pydantic.BaseModel):
    id: str
    state: Literal["completed"]


class UIEventDialog(_Event[DialogChange], type="new:dialog_changed"): ...


type UIEvent = (
    UIEventPost
    | UIEventState
    | UIEventPostReasoningChunk
    | UIEventPostContentChunk
    | UIEventError
    | UIEventRetry
    | UIEventToolCall
    | UIEventToolCalled
    | UIEventToolCallResult
    | UIEventHandoff
    | UIEventDialog
)


class UI(abc.ABC):
    @abc.abstractmethod
    async def send(self, event: UIEvent) -> None: ...

    @abc.abstractmethod
    async def request_user_confirm(self, role: str, message: str | None = ...) -> bool: ...

    @abc.abstractmethod
    async def request_user_manipulate(self, role: str, message: str | None = ...) -> str | None: ...


class CLI(UI):
    def __init__(self) -> None:
        super().__init__()
        self._console = rich.console.Console()
        self._current_role: str | None = None

    @override
    async def send(self, event: UIEvent) -> None:
        match event:
            case UIEventPost():
                self._update_current_role(event.role)
                self._console.print(rich.panel.Panel(event.data, title="ℹ️ Info", style="cyan"))

            case UIEventPostReasoningChunk():
                self._update_current_role(event.role)
                self._console.print(event.data, end="")  # TODO: identify reasoning content

            case UIEventPostContentChunk():
                self._update_current_role(event.role)
                self._console.print(event.data, end="")  # TODO: change to Live

            case UIEventToolCall():
                self._update_current_role(event.role)
                self._console.print(f"Tool call: {str(event)}")

            case UIEventToolCalled():
                self._update_current_role(event.role)
                self._console.print(f"Tool called: {event.data.id}")

            case UIEventToolCallResult():
                self._update_current_role(event.role)
                self._console.print(f"Tool call result: {event.data.id} -> {event.data.result}")

            case UIEventError():
                self._console.print(rich.panel.Panel(event.data, title="❌ Error", style="red"))

            case UIEventRetry():
                info = event.data
                msg = (
                    f"⏳ Rate limited — retrying {info.attempt}/{info.max_attempts} "
                    f"in {info.wait_seconds:.1f}s: {info.reason}"
                )
                self._console.print(rich.panel.Panel(msg, title="⚠️ Retry", style="yellow"))

            case UIEventState() if event.data == "confirmation_request":
                pass  # Placeholder for confirmation request handling

            case UIEventState() if event.data == "agent_finish":
                if self._current_role is not None:
                    self._console.print("")  # Clear the last line
                    self._current_role = None

            case UIEventState() if event.data == "agent_speaking":
                msg = f"🤖 {event.role} is planning the next step..."
                self._console.print(rich.panel.Panel(msg, title="ℹ️ Info", style="cyan"))

            case _:
                raise TypeError(f"Unsupported message type: {type(event)}")

    @override
    async def request_user_confirm(self, role: str, message: str | None = None) -> bool:
        max_tries = 3
        for _ in range(max_tries):
            with rich.live.Live(
                rich.panel.Panel(
                    f"[green]{message} ([bold bright_green]Y[/]es/[bold red]N[/]o)?",
                    title="Confirmation",
                ),
                console=self._console,
                screen=False,
                auto_refresh=False,
            ) as live:
                live.refresh()
                # Use asyncio.get_event_loop().run_in_executor to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, input)
                resp = resp.strip().lower()

            if resp in ("y", "yes"):
                return True

            if resp in ("n", "no"):
                return False

            self._console.print("[green]Invalid input. Please enter '[bold]Y[/]es' or '[bold]N[/]o'.[/]")

        self._console.print("[green]Too many invalid inputs.[/]")
        return False

    @override
    async def request_user_manipulate(self, role: str, message: str | None = None) -> str | None:
        # Use asyncio.to_thread to avoid blocking the event loop with input()
        message = message or "Press <Enter> after completing the action, or type your feedback and press <Enter>:"
        return await asyncio.to_thread(input, f"{role}: {message}\n> ")

    def _update_current_role(self, role: str) -> None:
        if self._current_role != role:
            num_prefix = (50 - 2 - len(role)) // 2
            num_suffix = 50 - 2 - len(role) - num_prefix
            self._console.print("-" * num_prefix + f" {role} " + "-" * num_suffix)
            self._current_role = role
