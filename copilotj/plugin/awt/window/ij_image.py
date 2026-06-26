# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, override

from copilotj.plugin._base import FromTo, Response, Verbosity
from copilotj.plugin.awt._base import ActionResponse
from copilotj.plugin.awt.window.awt_window import AwtWindowBase, AwtWindowDifferenceBase

__all__ = [
    "IjImage",
    "IjImageDifference",
    "IjImagePreview",
    "IjImagePreviewWithInfoResponse",
    "IjImageCaptureResponse",
    "RectangleRoi",
    "DescribedRoi",
]

type TYPE = Literal["ij.Image"]


class RectangleRoi(Response):
    x: int
    y: int
    width: int
    height: int

    def _describe_one_line(self, *, level: int, verbosity: Verbosity) -> str:
        return f"Rectangle ROI at [{self.x}, {self.y}] with size {self.width}x{self.height}"

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        return [self._describe_one_line(level=level, verbosity=verbosity)]


class DescribedRoi(Response):
    description: str

    def _describe_one_line(self, *, level: int, verbosity: Verbosity) -> str:
        return f"ROI: {self.description}"

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        return [self._describe_one_line(level=level, verbosity=verbosity)]


class IjImage(AwtWindowBase[TYPE]):
    title: str
    image_type: str
    size: str
    path: str | None

    bit_depth: int
    stack_size: int
    channels: int
    slices: int
    frames: int

    width: int
    height: int
    depth: int

    calibrated: bool
    calibrated_width: float
    calibrated_height: float
    calibrated_depth: float

    x_unit: str
    y_unit: str
    z_unit: str

    x_resolution: float
    y_resolution: float

    zoom_factor: float
    roi: RectangleRoi | DescribedRoi | None

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []
        typee = (
            f"{self.bit_depth}-bit {self.image_type}"
            if f"{self.bit_depth}-bit" not in self.image_type
            else self.image_type
        )
        lines.append(
            f"Image: {self.title} (id: {self.id}, type: {typee}, size: {self.size}, Path: {self.path or 'N/A'})"
        )

        if verbosity < Verbosity.NORMAL:
            return lines

        dimension = (
            f"{self.width}x{self.height} pixels"
            if self.depth <= 0
            else f"{self.width}x{self.height}x{self.depth} voxels"
        )
        ss, ch, sl, fr = self.stack_size, self.channels, self.slices, self.frames
        lines.append(f"Image dimension: {dimension}, stack size: {ss} ({ch} channels, {sl} slices, {fr} frames)")

        if self.calibrated:
            lines.append(
                f"Width: {self.calibrated_width:.2f} {self.x_unit}, "
                f"Height: {self.calibrated_height:.2f} {self.y_unit}, "
                f"Depth: {self.calibrated_depth:.2f} {self.z_unit}"
            )

        if self.x_resolution == self.y_resolution and self.x_unit == self.y_unit:
            resolution = f"Resolution: {self.x_resolution} pixels per {self.x_unit}"
        else:
            rx, ry = self.x_resolution, self.y_resolution
            resolution = f"X Resolution: {rx} pixels per {self.x_unit}, Y Resolution: {ry} pixels per {self.y_unit}"

        if self.zoom_factor != 1:
            lines.append(f"{resolution}, Zoom factor: {self.zoom_factor:.2f}")
        else:
            lines.append(resolution)

        if self.roi:
            lines.append(self.roi._describe_one_line(level=level, verbosity=verbosity))

        return lines


class IjImageDifference(AwtWindowDifferenceBase[TYPE]):
    title: FromTo[str] | None = None
    image_type: FromTo[str] | None = None
    size: FromTo[str] | None = None
    path: FromTo[str | None] | None = None

    bit_depth: FromTo[int] | None = None
    stack_size: FromTo[int] | None = None
    channels: FromTo[int] | None = None
    slices: FromTo[int] | None = None
    frames: FromTo[int] | None = None

    width: FromTo[int] | None = None
    height: FromTo[int] | None = None
    depth: FromTo[int] | None = None

    calibrated: FromTo[bool] | None = None
    calibrated_width: FromTo[float] | None = None
    calibrated_height: FromTo[float] | None = None
    calibrated_depth: FromTo[float] | None = None

    x_unit: FromTo[str] | None = None
    y_unit: FromTo[str] | None = None
    z_unit: FromTo[str] | None = None

    x_resolution: FromTo[float] | None = None
    y_resolution: FromTo[float] | None = None

    zoom_factor: FromTo[float] | None = None
    roi: FromTo[RectangleRoi | DescribedRoi | None] | None = None

    image_updated: bool

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []

        if verbosity <= Verbosity.LOW:
            count = sum(1 for field_value in self if field_value is not None)
            msg = "Image content updated (details not tracked)" if self.image_updated else "Image content not changed"
            if count == 0:
                lines.append(msg + ", and no other changes detected")
            elif count == 1:
                lines.append(msg + ", and one other change detected")
            else:
                changes = "changes" if count > 1 else "change"
                lines.append(f"{msg}, and other {count} {changes} detected")

            return lines

        if self.image_updated:
            lines.append("Image content updated (details not tracked).")

        for field_name, field_value in self:
            if field_name in ("id", "type", "image_updated"):
                continue  # Skip this field as it's already handled above

            # Replace underscores with spaces and capitalize for readability
            readable_name = field_name.replace("_", " ").capitalize()
            if verbosity >= Verbosity.HIGH and field_value is None:
                lines.append(f"{readable_name}: No change")

            elif field_value is not None:
                lines.append(f"{readable_name}: {field_value}")

        return lines


class IjImagePreview(Response):
    image: str
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale_factor: float

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        lines = []
        ow, oh = self.original_width, self.original_height
        rw, rh, sf = self.resized_width, self.resized_height, self.scale_factor
        lines.append(f"Dimensions: {ow}x{oh} -> {rw}x{rh} (scale: {sf:.2f})")
        lines.append(f"Image data: {self._truncated_base64()}")
        return lines

    def _truncated_base64(self, max_length: int = 32) -> str:
        if len(self.image) > max_length:
            return self.image[: max_length - 3] + "..."

        return self.image


class _Histogram(Response):
    bin_count: int
    bins: list[int]
    min_gray: float
    max_gray: float

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        """Generate histogram description with bin ranges and counts."""
        if not self.bins:
            return ["Empty histogram"]

        heading = "#" * level
        lines = [f"{heading} Histogram"]

        # Create markdown table header
        lines.append("| Bin | Range | Count |")
        lines.append("|-----|-------|-------|")

        bin_size = (self.max_gray - self.min_gray) / self.bin_count
        for i in range(self.bin_count):
            lower_bound = self.min_gray + i * bin_size
            upper_bound = self.min_gray + (i + 1) * bin_size
            if i == self.bin_count - 1:
                upper_bound = self.max_gray  # Last bin covers max value

            lines.append(f"| {i + 1} | {lower_bound:.1f}-{upper_bound:.1f} | {self.bins[i]} |")

        return lines


class IjImagePreviewWithInfoResponse(IjImagePreview):
    info: IjImage
    histogram: _Histogram

    @override
    def _describe(self, *, level: int, verbosity: Verbosity) -> list[str]:
        """Describe both the image and its info."""
        lines = self.info._describe(level=level, verbosity=verbosity)
        ow, oh = self.original_width, self.original_height
        rw, rh, sf = self.resized_width, self.resized_height, self.scale_factor
        lines.append(f"Dimensions: {ow}x{oh} -> {rw}x{rh} (scale: {sf:.2f})")
        # lines.append(f"Image data: {self._truncated_base64()}")

        if verbosity >= Verbosity.LOW:
            lines.extend(self.histogram._describe(level=level + 1, verbosity=verbosity))

        return lines


type IjImageCaptureResponse = ActionResponse[Literal["ij.Image.capture"], IjImagePreviewWithInfoResponse]
