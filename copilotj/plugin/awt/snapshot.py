# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from datetime import datetime
from typing import override

from copilotj.plugin._base import FromTo, Request, Response, Verbosity
from copilotj.plugin.awt.window import TypedWindow, TypedWindowDifference

__all__ = ["SnapshotSummary", "TakeSnapshotRequest", "SnapshotDifference"]


class SnapshotSummary(Response):
    id: int
    current_image: str | None
    windows: list[TypedWindow]

    screen_width: int
    screen_height: int
    gui_scale: str
    timestamp: str

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = [
            f"Snapshot #{self.id} (image: {self.current_image or 'N/A'}, "
            f"screen: {self.screen_width}x{self.screen_height}, scale: {self.gui_scale})"
        ]

        if len(self.windows) == 0:
            lines.append("no window opened")
            return lines

        for window in self.windows:
            lines.extend(window._describe(level=level + 1, verbosity=verbosity))

        return lines


class TakeSnapshotRequest(Request[SnapshotSummary], event="take_snapshot", response_type=SnapshotSummary):
    pass


class _WindowAndDifference(Response):
    later: TypedWindow
    difference: TypedWindowDifference | None

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []
        if verbosity >= Verbosity.NORMAL:
            verb = Verbosity.LOW if verbosity <= Verbosity.NORMAL else verbosity.NORMAL
            lines.extend(self.later._describe(level=level, verbosity=verb))
        else:
            title = getattr(self.later, "title", None)
            if title is None:
                lines.append(f"Window: id={self.later.id}, type={self.later.type}")
            else:
                lines.append(f"Window: {title} (id={self.later.id})")

        diff_lines = self.difference._describe(level=level + 1, verbosity=verbosity) if self.difference else None
        if diff_lines is None or len(diff_lines) == 0:
            # This case should ideally not happen if difference is not None
            warnings.warn("No changes detected but difference is not None.")
            lines.append("No changes detected.")
            return lines

        lines.append("Changes:")
        lines.extend("  " + line for line in diff_lines)
        return lines


class _WindowSnapshotDifference(Response):
    added: list[TypedWindow]
    changed: list[_WindowAndDifference]
    removed: list[TypedWindow]
    unchanged: list[TypedWindow]

    def any_changed(self) -> bool:
        return len(self.added) > 0 or len(self.changed) > 0 or len(self.removed) > 0

    def any_opened(self) -> bool:
        return self.any_changed() or len(self.unchanged) > 0

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = ["Windows difference:"]

        if not self.any_opened():
            lines.append("  no window opened")
            return lines

        groups = (
            (self.added, "added", Verbosity.HIGH if verbosity >= Verbosity.NORMAL else verbosity),
            (self.changed, "changed", verbosity),
            (self.removed, "removed", Verbosity.LOW if verbosity <= Verbosity.NORMAL else verbosity),
            (self.unchanged, "unchanged", Verbosity.LOW if verbosity <= Verbosity.NORMAL else verbosity),
        )

        for xs, group, verb in groups:
            if len(xs) == 0:
                continue

            lines.append(f"  {group}:")
            for x in xs:
                for line in x._describe(level=level + 1, verbosity=verb):
                    lines.append("  " + line)

        no_actions = "/".join(action for xs, action, _ in groups if len(xs) == 0)
        if no_actions:
            lines.append(f"  no windows {no_actions}.")

        return lines


class SnapshotDifference(Response):
    timestamp_early: datetime
    timestamp_later: datetime
    current_image: FromTo[str | None] | None
    windows: _WindowSnapshotDifference

    def any_changed(self) -> bool:
        return self.current_image is not None or self.windows.any_changed()

    def any_opened(self) -> bool:
        return self.windows.any_opened()

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = ["Snapshot difference:"]

        if self.current_image is not None:
            lines.append(f"Current image changed from {self.current_image.from_} to {self.current_image.to}")

        lines.extend(self.windows._describe(level=level + 1, verbosity=verbosity))
        return lines
