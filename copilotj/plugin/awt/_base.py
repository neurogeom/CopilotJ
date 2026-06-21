# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Any, override

from copilotj.plugin._base import Response, Verbosity

__all__ = ["ComponentBase", "ActionResponse", "str_or_empty"]


class ComponentBase[T: str](Response, ABC):
    type: T
    name: str | None

    # Playwright-mcp-style ref handle ("e" + int) assigned on the Java side, or
    # None for non-ref-eligible nodes (labels, intermediate containers).
    ref: str | None = None

    # Per-component action metadata (Java-built list of Action-like objects with
    # at least a fully-qualified "type"). Typed loosely to avoid a circular
    # import with copilotj.plugin.awt.action; only the short id is read here.
    actions: list[Any] | None = None

    # ------------------------------------------------------------------
    # YAML rendering (playwright-mcp-style snapshot format)
    #
    # Each node renders as a single YAML list item, e.g.:
    #   - button "Apply" [ref=e10] (click)
    #   - choice "Method" [ref=e5] (selectItem): selected="Huang" items=[...]
    # Containers append a trailing ":" and nest children with 2-space indent.
    # ------------------------------------------------------------------

    def role(self) -> str:
        """Lowercase short role name (e.g. 'button', 'window'). Override per subclass."""
        return "component"

    def _node_name(self) -> str | None:
        """The quoted "name" segment, or None to omit it."""
        return None

    def _state_inline(self) -> str | None:
        """The inline `key=value` tail after the colon, or None to omit."""
        return None

    def _head_extras(self) -> str | None:
        """Extra bracketed segments after the ref (e.g. '(id=1)' for windows)."""
        return None

    def _action_segment(self) -> str | None:
        """The '(actionId,...)' segment listing short action ids, or None."""
        if not self.actions:
            return None
        ids = [_action_short_id(a) for a in self.actions]
        return "(" + ",".join(ids) + ")"

    def _yaml_head(self) -> str:
        """The line head: `- role "name" [ref=eN] (extras) (actions)`."""
        parts: list[str] = ["-", self.role()]
        name = self._node_name()
        if name:
            parts.append(f'"{name}"')
        if self.ref:
            parts.append(f"[ref={self.ref}]")
        extras = self._head_extras()
        if extras:
            parts.append(extras)
        actions = self._action_segment()
        if actions:
            parts.append(actions)
        return " ".join(parts)

    def _yaml_line(self) -> str:
        """Full leaf line: head, plus `: state` when there is inline state."""
        head = self._yaml_head()
        state = self._state_inline()
        if state is not None:
            return f"{head}: {state}"
        return head

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        return [self._yaml_line()]


class ActionResponse[T: str, K: Response | None](Response):
    type: T
    result: K

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        if self.result is None:
            return ["success"]

        return self.result._describe(level=level, verbosity=verbosity)


def str_or_empty(value: str | None, *, max_length: int = 300) -> str:
    if value is None or len(value) == 0:
        return "<empty>"

    value = value.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(value) > max_length:
        return f'"{value[:max_length]}..."'

    return f'"{value}"'


def _action_short_id(action: Any) -> str:
    """Short action id: the substring after the last '.' of the action type."""
    action_type = getattr(action, "type", None)
    if action_type is None and isinstance(action, dict):
        action_type = action.get("type")
    return str(action_type).rsplit(".", 1)[-1]


def format_items(items: list[str], *, max_items: int = 8) -> str:
    """Render option strings as `["a","b",...]`, truncating long lists."""
    shown = items[:max_items]
    rendered = ", ".join(str_or_empty(i) for i in shown)
    if len(items) > max_items:
        rendered += ", ..."
    return f"[{rendered}]"
