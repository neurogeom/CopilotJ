# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin._base import FromTo, Response, Verbosity
from copilotj.plugin.awt._base import ActionResponse
from copilotj.plugin.awt.window.awt_window import AwtWindowBase, AwtWindowDifferenceBase

__all__ = [
    "IjTextWindow",
    "IjTextWindowDifference",
    "ResultsTableSummary",
    "ResultsTableSummaryDifference",
    "ResultsTableChunk",
    "ResultsTableChunkResponse",
]


# TODO: preview some rows
class ResultsTableSummary(Response):
    title: str
    headings: list[str]
    size: int

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        return [f"Results Table: {self.title} (size: {self.size}, headings: {', '.join(self.headings)})"]


class ResultsTableSummaryDifference(Response):
    title: FromTo[str] | None
    headings: FromTo[list[str]] | None
    size: FromTo[int] | None

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []
        if self.title is not None:
            lines.append(f"Title: {self.title.from_} -> {self.title.to}")

        if self.headings is not None:
            lines.append(f"Headings: {self.headings.from_} -> {self.headings.to}")

        if self.size is not None:
            lines.append(f"Size: {self.size.from_} -> {self.size.to}")

        return lines


type TYPE = Literal["ij.text.TextWindow"]


class IjTextWindow(AwtWindowBase[TYPE]):
    title: str
    results_table: ResultsTableSummary | None

    @override
    def _node_name(self) -> str | None:
        return self.title

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = [self._yaml_head() + ":"]
        if self.results_table is not None:
            for line in self.results_table._describe(level=level + 1, verbosity=verbosity):
                lines.append("  " + line)
        else:
            lines.append("  no results table")
        return lines


class IjTextWindowDifference(AwtWindowDifferenceBase[TYPE]):
    title: FromTo[str] | None
    results_table_added: ResultsTableSummary | None
    results_table_changed: ResultsTableSummaryDifference | None
    results_table_removed: bool

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []

        if self.title is not None:
            lines.append(f"Title: {self.title}")

        if self.results_table_added is not None:
            if verbosity <= Verbosity.LOW:
                lines.append("Results table added.")
            else:
                lines.append("Results table added:")
                lines.extend(self.results_table_added._describe(level=level + 1, verbosity=verbosity))

        elif self.results_table_removed:
            lines.append("Results table removed.")

        elif self.results_table_changed is not None:
            if verbosity <= Verbosity.LOW:
                lines.append("Results table changed.")
            else:
                lines.append("Results table changes:")
                lines.extend(self.results_table_changed._describe(level=level + 1, verbosity=verbosity))

        return lines


class ResultsTableChunk(Response):
    title: str
    total: int
    offset: int
    limit: int
    data: dict[str, list[float]]

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        heading = "#" * level
        lines = []
        lines.append(f"{heading} Result Table")
        lines.append(f"Title: {self.title}")

        if self.total == 0 or len(self.data) == 0:
            lines.append("No data")
            return lines

        if self.total > self.limit:
            s, e = self.offset, self.offset + len(self.data)
            lines.append(f"Showing from {s} to {e} of {self.total} data")
        else:
            lines.append(f"Showing all {self.total} data")

        lines.append(f"{heading}# Data")
        lines.append(f"| {' | '.join(self.data.keys())} |")
        for i in range(self.limit):
            row = " | ".join(str(self.data[k][i]) for k in self.data.keys())
            lines.append(f"| {row} |")

        return lines


type ResultsTableChunkResponse = ActionResponse[Literal["ij.text.TextWindow.getResultsTable"], ResultsTableChunk]
