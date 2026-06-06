# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
import datetime
import logging
import os
from typing import Literal

from copilotj.core.model_client import ModelResponse, ModelResponseChunk, ToolCall
from copilotj.core.ui import (
    CLI,
    UI,
    DialogChange,
    Handoff,
    ToolCallId,
    ToolCallResult,
    UIEventContentMarkdown,
    UIEventDialog,
    UIEventError,
    UIEventHandoff,
    UIEventPost,
    UIEventPostContentChunk,
    UIEventPostReasoningChunk,
    UIEventState,
    UIEventToolCall,
    UIEventToolCalled,
    UIEventToolCallResult,
)

__all__ = ["Runtime"]


class Runtime(abc.ABC):
    _ui: UI
    _logger: logging.Logger

    def __init__(self, name: str, *, ui: UI | None = None) -> None:
        super().__init__()
        self._ui = ui or CLI()
        self._logger = _setup_logger(name)

    async def request_user_confirm(self, role: str, message: str | None = None) -> bool:
        return await self._ui.request_user_confirm(role, message)

    # TODO: redesign following methods

    async def request_user_manipulate(self, role: str, message: str | None = None) -> str | None:
        return await self._ui.request_user_manipulate(role, message)

    async def print_chat(self, agent: str, message: str | ModelResponse | ModelResponseChunk | ToolCall) -> None:
        self._logger.debug(f"{agent}: {message}")
        match message:
            case str():
                await self._ui.send(UIEventPost(role=agent, data=[UIEventContentMarkdown(markdown=message)]))

            case ModelResponse():
                # TODO: identify if this is a reasoning content
                if message.reasoning_content is not None:
                    await self._ui.send(
                        UIEventPost(role=agent, data=[UIEventContentMarkdown(markdown=message.reasoning_content)])
                    )

                if message.content is not None:
                    await self._ui.send(
                        UIEventPost(role=agent, data=[UIEventContentMarkdown(markdown=message.content)])
                    )

            case ModelResponseChunk():
                if message.reasoning_content is not None:
                    await self._ui.send(
                        UIEventPostReasoningChunk(
                            role=agent, data=UIEventContentMarkdown(markdown=message.reasoning_content)
                        )
                    )

                if message.content is not None:
                    await self._ui.send(
                        UIEventPostContentChunk(role=agent, data=UIEventContentMarkdown(markdown=message.content))
                    )

                if message.finish_reason is not None and message.finish_reason in ["stop", "tool_calls"]:
                    await self._ui.send(UIEventState(role=agent, data="agent_finish"))

            case ToolCall():
                await self._ui.send(UIEventToolCall(role=agent, data=message))

            case _:
                # Should not be reached due to type hints, but good for robustness
                self._logger.error(f"Unhandled message type in on_chat: {type(message)}")

    async def print_info(self, agent: str, message: str) -> None:
        self._logger.info(message)
        await self._ui.send(UIEventPost(role=agent, data=[UIEventContentMarkdown(markdown=message)]))

    async def print_error(self, role: str, message: str) -> None:
        self._logger.error(message)
        await self._ui.send(UIEventError(role=role, data=message))  # TODO: log level: info / warn / error?

    async def print_tool_called(self, role: str, id: str) -> None:
        self._logger.info("Tool called #%s", id)
        await self._ui.send(UIEventToolCalled(role=role, data=ToolCallId(id=id)))

    async def print_tool_call_result(self, role: str, id: str, typee: Literal["success", "error"], result: str) -> None:
        self._logger.info("Tool call #%s %s, result: %s", id, typee, result)
        await self._ui.send(
            UIEventToolCallResult(
                role=role, data=ToolCallResult(id=id, type=typee, result=[UIEventContentMarkdown(markdown=result)])
            )
        )

    async def print_handoff(self, role: str, handoff: Handoff) -> None:
        self._logger.info("Handoff: %s", handoff)
        await self._ui.send(UIEventHandoff(role=role, data=handoff))

    async def print_dialog_change(self, dialog: DialogChange) -> None:
        self._logger.info("Dialog: %s", dialog)
        await self._ui.send(UIEventDialog(role="system", data=dialog))

    async def update_current_agent(self, agent: str) -> None:
        await self._ui.send(UIEventState(role=agent, data="agent_speaking"))

    def log_info(self, message: str) -> None:
        self._logger.info(message)
        # print(message)  # TODO: re-enable with UTF-8 stdout support

    def log_error(self, message: str) -> None:
        self._logger.error(message)
        # print(message)  # TODO: re-enable with UTF-8 stdout support


def _setup_logger(name: str, *, log_folder: str = "logs") -> logging.Logger:
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    now = datetime.datetime.now()
    log_filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    log_filepath = os.path.join(log_folder, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filepath, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    return logger
