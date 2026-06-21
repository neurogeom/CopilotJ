# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Literal, override

import pydantic

from copilotj.plugin._base import Response, Verbosity
from copilotj.plugin.awt._base import str_or_empty
from copilotj.plugin.awt.component.button_node import ButtonClickResponse
from copilotj.plugin.awt.component.checkbox_node import CheckboxSetStateResponse
from copilotj.plugin.awt.component.choice_node import ChoiceSelectItemResponse
from copilotj.plugin.awt.component.text_area_node import TextAreaSetTextResponse
from copilotj.plugin.awt.component.text_field_node import TextFieldSetTextResponse
from copilotj.plugin.awt.window.ij_image import IjImageCaptureResponse
from copilotj.plugin.awt.window.ij_text_window import ResultsTableChunkResponse

__all__ = ["Action", "AnyParameterSchema", "TypedActionResponse"]


class ParameterSchemaBase[T](pydantic.BaseModel, abc.ABC):
    type: T
    name: str
    description: str | None = None

    def describe(self) -> str:
        base = f"{self.name}: {self.type}"
        raw_parts = self._constrains()
        if not raw_parts:
            return base

        return f"{base}, {', '.join(raw_parts)}"

    @abc.abstractmethod
    def _constrains(self) -> list[str]: ...


class StringParameterSchema(ParameterSchemaBase[Literal["string"]]):
    min_length: int | None = None
    max_length: int | None = None
    pattern: int | None = None
    enum_values: list[str] | None = None

    @override
    def _constrains(self) -> list[str]:
        parts = []

        match (self.min_length, self.max_length):
            case (None, None):
                pass

            case (min_length, None):
                parts.append(f"min length: {min_length}")

            case (None, max_length):
                parts.append(f"max length: {max_length}")

            case (min_length, max_length):
                parts.append(f"length: {min_length} - {max_length}")

        if self.pattern is not None:  # Changed from `if self.pattern else []` to handle pattern being 0
            parts.append(f"pattern: {self.pattern}")

        if self.enum_values:
            parts.append(f"enum values: [{', '.join(str_or_empty(a) for a in self.enum_values)}]")

        return parts


class _NumberParameterSchema[T, K](ParameterSchemaBase[T]):
    minimum: K | None = None
    maximum: K | None = None

    def _constrains(self) -> list[str]:
        parts = []
        match (self.minimum, self.maximum):
            case (None, None):
                pass

            case (min_value, None):
                parts.append(f"min: {min_value}")

            case (None, max_value):
                parts.append(f"max: {max_value}")

            case (min_value, max_value):
                parts.append(f"range: {min_value} - {max_value}")

        return parts


class IntegerParameterSchema(_NumberParameterSchema[Literal["integer"], int]):
    pass


class NumberParameterSchema(_NumberParameterSchema[Literal["number"], float]):  # Java's Number can be float/double
    pass


class BooleanParameterSchema(ParameterSchemaBase[Literal["boolean"]]):
    def _constrains(self) -> list[str]:
        return []  # No additional constrains for boolean parameters


type AnyParameterSchema = (
    StringParameterSchema | IntegerParameterSchema | NumberParameterSchema | BooleanParameterSchema
)


class Action(Response):
    type: str  # identifies the action
    name: str
    description: str
    parameters: list[AnyParameterSchema]

    @property
    def short_id(self) -> str:
        """Short action id: the substring after the last '.' of type.

        e.g. ``java.awt.Button.click`` -> ``click``. This is the identifier a
        caller passes to ``call_action(ref, action, params)``.
        """
        return self.type.rsplit(".", 1)[-1]

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        description = self.description or self.name

        if verbosity > Verbosity.LOW:
            if self.parameters:
                parameters = [f"{i}. {param.describe()}" for i, param in enumerate(self.parameters)]
                description += f" (Parameters: {'; '.join(parameters)})"
            else:
                description += " (No parameters needed)"

        return [description]


type TypedActionResponse = (
    ButtonClickResponse
    | CheckboxSetStateResponse
    | ChoiceSelectItemResponse
    | TextAreaSetTextResponse
    | TextFieldSetTextResponse
    | IjImageCaptureResponse
    | ResultsTableChunkResponse
)
