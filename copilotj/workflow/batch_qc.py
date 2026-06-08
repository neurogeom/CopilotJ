# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import logging
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from skimage import io, transform

from copilotj.core import ImageMessage, TextMessage, new_vlm_model_client
from copilotj.core.config import Config
from copilotj.multiagent.py_tools import get_project_temp_dir

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ome.tif", ".ome.tiff", ".czi", ".nd2"}
OME_TIFF_EXTENSIONS = {".ome.tif", ".ome.tiff"}
TIFF_EXTENSIONS = {".tif", ".tiff", *OME_TIFF_EXTENSIONS}
THUMBNAIL_SIZE = (256, 256)
PADDING = 4
BACKGROUND = (40, 40, 40)
LOW_PCT = 1.0
HIGH_PCT = 99.8
UINT8_MAX = 255
MAX_IMAGE_METADATA_LINES = 20


async def batch_precheck(
    folder_path: Annotated[str, "Path to the image folder to check before batch processing"],
    *,
    sample_size: Annotated[int, "Number of images to sample for QC (default: 9)"] = 9,
    analysis_focus: Annotated[str | None, "Optional focus, e.g. 'background consistency'"] = None,
    output_dir: Annotated[str | None, "Optional output directory for montage and report"] = None,
    skip_analysis: Annotated[bool, "Skip VLM analysis, only generate montage"] = False,
    montage_count: Annotated[int, "Number of montages to sample/analyze (default: 1)"] = 1,
    cfg: Config | None = None,
) -> str:
    try:
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")
        if montage_count <= 0:
            raise ValueError("montage_count must be greater than 0")
        sampled_paths, folder_meta = sample_images_from_folder(folder_path, sample_size * montage_count)
        montage_items = _create_montages(sampled_paths, sample_size, montage_count, output_dir)
        analysis = await _run_qc_analyses(montage_items, folder_meta, analysis_focus, skip_analysis, cfg)
        report = _format_qc_report(analysis, folder_meta | _combine_montage_metadata(montage_items))
        saved = "\n".join(f"- `{item['path']}`" for item in montage_items)
        return f"{report}\n\n**Montages saved to**:\n{saved}"
    except FileNotFoundError as error:
        return f"[ERROR] {error}"
    except ValueError as error:
        return f"[INVALID] {error}"
    except Exception as error:
        logger.error("Batch QC failed: %s", error)
        return f"[FAILED] Batch QC Failed: {error}\n\nPlease check the folder path and Vision configuration."


def bind_batch_precheck(cfg: Config | Callable[[], Config]):
    def current_cfg() -> Config:
        return cfg() if callable(cfg) else cfg

    async def configured_batch_precheck(
        folder_path: Annotated[str, "Path to the image folder to check before batch processing"],
        *,
        sample_size: Annotated[int, "Number of images to sample for QC (default: 9)"] = 9,
        analysis_focus: Annotated[str | None, "Optional focus, e.g. 'background consistency'"] = None,
        output_dir: Annotated[str | None, "Optional output directory for montage and report"] = None,
        skip_analysis: Annotated[bool, "Skip VLM analysis, only generate montage"] = False,
        montage_count: Annotated[int, "Number of montages to sample/analyze (default: 1)"] = 1,
    ) -> str:
        return await batch_precheck(
            folder_path,
            sample_size=sample_size,
            analysis_focus=analysis_focus,
            output_dir=output_dir,
            skip_analysis=skip_analysis,
            montage_count=montage_count,
            cfg=current_cfg(),
        )

    return configured_batch_precheck


def sample_images_from_folder(
    folder_path: str, sample_size: int = 9, max_depth: int = 3
) -> tuple[list[Path], dict[str, Any]]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    images = sorted({path for path in folder.rglob("*") if _is_image(path, folder, max_depth)})
    if not images:
        raise FileNotFoundError(f"No image files found in: {folder_path}")
    sampled = random.sample(images, min(sample_size, len(images)))
    return sampled, {
        "total_images": len(images),
        "sampled_count": len(sampled),
        "folder_path": str(folder_path),
        "extensions_found": sorted({_image_suffix(path) for path in images}),
    }


def _chunks(paths: list[Path], size: int) -> list[list[Path]]:
    return [paths[index : index + size] for index in range(0, len(paths), size)]


def _create_montages(
    sampled_paths: list[Path], sample_size: int, montage_count: int, output_dir: str | None
) -> list[dict[str, Any]]:
    output_base = Path(output_dir) if output_dir else get_project_temp_dir("batch_qc")
    montage_items: list[dict[str, Any]] = []
    for index, chunk in enumerate(_chunks(sampled_paths, sample_size), start=1):
        name = "batch_montage.png" if montage_count == 1 else f"batch_montage_{index}.png"
        montage_path, montage_meta = create_montage(chunk, output_path=output_base / name)
        montage_items.append({"path": montage_path, "meta": montage_meta})
    return montage_items


def create_montage(
    image_paths: list[Path],
    *,
    output_path: Path | None = None,
    grid_size: tuple[int, int] | None = None,
    thumbnail_size: tuple[int, int] = THUMBNAIL_SIZE,
    padding: int = PADDING,
    background: tuple[int, int, int] = BACKGROUND,
) -> tuple[Path, dict[str, Any]]:
    if not image_paths:
        raise ValueError("No images provided for montage")
    rows, cols = _grid_size(len(image_paths), grid_size)
    montage = _create_canvas(rows, cols, thumbnail_size, padding, background)
    loaded_images: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths):
        try:
            thumbnail, original_shape = _load_thumbnail(image_path, thumbnail_size)
            _paste_thumbnail(montage, thumbnail, index=index, cols=cols, size=thumbnail_size, padding=padding)
            loaded_images.append(_image_metadata(index, image_path, original_shape))
        except Exception as error:
            errors.append({"index": index, "path": str(image_path), "error": str(error)})
            logger.warning("Failed to load image %s: %s", image_path, error)
    if not loaded_images:
        raise RuntimeError(f"Failed to load all sampled images: {errors}")
    saved_path = _save_montage(montage, output_path)
    return saved_path, _montage_metadata(rows, cols, thumbnail_size, loaded_images, errors, montage.shape)


async def _run_qc_analyses(
    montage_items: list[dict[str, Any]],
    folder_meta: dict[str, Any],
    analysis_focus: str | None,
    skip_analysis: bool,
    cfg: Config | None,
) -> dict[str, Any]:
    if skip_analysis:
        return _manual_review_analysis("VLM analysis was skipped by caller.")
    if reason := _vlm_qc_disabled_reason(cfg):
        return _manual_review_analysis(reason)
    analyses = await asyncio.gather(
        *(
            analyze_image_heterogeneity(
                montage_path=item["path"],
                image_metadata=item["meta"]["loaded_images"],
                folder_metadata=folder_meta,
                analysis_focus=analysis_focus,
                cfg=cfg,
            )
            for item in montage_items
        )
    )
    return _aggregate_analyses(analyses)


def _vlm_qc_disabled_reason(cfg: Config | None) -> str | None:
    if cfg is None:
        return "Vision configuration is unavailable for Batch QC."
    if not cfg.vision_enabled:
        return "Vision is disabled in Settings."
    if not cfg.vision_available:
        return "Vision is enabled, but no vision-capable model is configured."
    return None


async def analyze_image_heterogeneity(
    *,
    montage_path: Path,
    image_metadata: list[dict[str, Any]],
    folder_metadata: dict[str, Any],
    analysis_focus: str | None = None,
    cfg: Config | None = None,
) -> dict[str, Any]:
    messages = [
        ImageMessage(role="user", image=_image_data_url(montage_path)),
        TextMessage(role="user", text=_analysis_prompt(image_metadata, folder_metadata, analysis_focus)),
    ]
    try:
        response = await new_vlm_model_client(cfg=cfg).create(messages)
        return _parse_analysis_response(response.content or "", montage_path)
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse VLM response as JSON: %s", error)
        return _manual_review_analysis(
            "VLM JSON parse failed. Manual inspection is required.",
            raw_response=response.content or "",
            error=str(error),
        )
    except Exception as error:
        logger.error("VLM analysis failed: %s", error)
        raise


def _is_image(path: Path, folder: Path, max_depth: int) -> bool:
    depth = len(path.relative_to(folder).parts) - 1
    return path.is_file() and depth <= max_depth and _image_suffix(path) in IMAGE_EXTENSIONS


def _image_suffix(path: Path) -> str:
    suffixes = "".join(path.suffixes[-2:]).lower()
    return suffixes if suffixes in OME_TIFF_EXTENSIONS else path.suffix.lower()


def _grid_size(n_images: int, grid_size: tuple[int, int] | None) -> tuple[int, int]:
    if grid_size:
        return grid_size
    cols = int(np.ceil(np.sqrt(n_images)))
    return int(np.ceil(n_images / cols)), cols


def _create_canvas(
    rows: int, cols: int, size: tuple[int, int], padding: int, background: tuple[int, int, int]
) -> np.ndarray:
    canvas = np.zeros((rows * size[0] + (rows + 1) * padding, cols * size[1] + (cols + 1) * padding, 3), dtype=np.uint8)
    canvas[:] = background
    return canvas


def _load_thumbnail(img_path: Path, thumbnail_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, ...]]:
    image = _read_image(img_path)
    thumbnail = transform.resize(_to_display_rgb(image), thumbnail_size, preserve_range=True, anti_aliasing=True)
    return thumbnail.astype(np.uint8), tuple(image.shape)


def _read_image(img_path: Path) -> np.ndarray:
    try:
        return io.imread(str(img_path))
    except Exception as error:
        if _image_suffix(img_path) not in TIFF_EXTENSIONS:
            raise error
        import tifffile

        return tifffile.imread(str(img_path))


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    plane = _select_display_plane(np.asarray(image))
    if plane.ndim == 2:
        gray = _scale_to_uint8(plane)
        return np.stack([gray, gray, gray], axis=-1)
    return np.stack([_scale_to_uint8(plane[..., channel]) for channel in range(3)], axis=-1)


def _select_display_plane(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] in (3, 4)):
        return image
    if image.ndim < 2:
        raise ValueError(f"Unsupported image shape for montage: {image.shape}")
    if image.ndim > 3 and image.shape[-1] in (3, 4):
        planes = list(image.reshape((-1, *image.shape[-3:])))
    else:
        planes = list(image.reshape((-1, *image.shape[-2:])))
    return max(planes, key=_plane_score)


def _plane_score(plane: np.ndarray) -> float:
    gray = plane[..., :3].mean(axis=-1) if plane.ndim == 3 else plane
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return float("-inf")
    lo, hi = np.percentile(finite.astype(np.float32), [LOW_PCT, HIGH_PCT])
    return float(hi - lo) if hi > lo else float(finite.max() - finite.min())


def _scale_to_uint8(channel: np.ndarray) -> np.ndarray:
    if channel.dtype == np.uint8:
        return channel
    if channel.dtype == bool:
        return channel.astype(np.uint8) * UINT8_MAX
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    lo, hi = np.percentile(finite.astype(np.float32), [LOW_PCT, HIGH_PCT])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(channel.shape, dtype=np.uint8)
    scaled = np.clip((channel.astype(np.float32) - lo) / max(hi - lo, 1e-8), 0, 1)
    return (scaled * UINT8_MAX).astype(np.uint8)


def _paste_thumbnail(
    montage: np.ndarray, thumbnail: np.ndarray, *, index: int, cols: int, size: tuple[int, int], padding: int
) -> None:
    row, col = divmod(index, cols)
    y_start = padding + row * (size[0] + padding)
    x_start = padding + col * (size[1] + padding)
    montage[y_start : y_start + size[0], x_start : x_start + size[1]] = thumbnail


def _image_metadata(index: int, image_path: Path, original_shape: tuple[int, ...]) -> dict[str, Any]:
    return {"index": index, "path": str(image_path), "original_shape": original_shape, "filename": image_path.name}


def _montage_metadata(
    rows: int,
    cols: int,
    size: tuple[int, int],
    loaded: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "grid_size": (rows, cols),
        "thumbnail_size": size,
        "n_images": len(loaded),
        "n_errors": len(errors),
        "loaded_images": loaded,
        "errors": errors,
        "montage_shape": shape,
    }


def _combine_montage_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    metas = [item["meta"] for item in items]
    return {
        "montage_count": len(items),
        "n_images": sum(meta["n_images"] for meta in metas),
        "n_errors": sum(meta["n_errors"] for meta in metas),
        "loaded_images": [image for meta in metas for image in meta["loaded_images"]],
        "errors": [error for meta in metas for error in meta["errors"]],
    }


def _save_montage(montage: np.ndarray, output_path: Path | None) -> Path:
    output_path = output_path or get_project_temp_dir("batch_qc") / "batch_montage.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(output_path), montage)
    return output_path


def _analysis_prompt(
    image_metadata: list[dict[str, Any]], folder_metadata: dict[str, Any], analysis_focus: str | None
) -> str:
    sampled = "\n".join(f"- [{img['index']}] {img['filename']}" for img in image_metadata[:MAX_IMAGE_METADATA_LINES])
    focus = f"\n\nSpecial focus: {analysis_focus}" if analysis_focus else ""
    return f"""Inspect this montage of {len(image_metadata)} sample images for batch-processing heterogeneity.
Dataset: {folder_metadata.get("total_images", "unknown")} total; {folder_metadata.get("sampled_count", "unknown")} sampled.
Extensions: {", ".join(folder_metadata.get("extensions_found", []))}
Sampled images:
{sampled}
Check background, quality, content, technical, and structural consistency issues.{focus}
Return only JSON with keys: heterogeneity_detected, severity, confidence, issues, recommendations,
batch_processing_suitable, suggested_workflow_modifications, summary."""


def _image_data_url(montage_path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(montage_path.read_bytes()).decode("utf-8")


def _parse_analysis_response(result_text: str, montage_path: Path) -> dict[str, Any]:
    result = _load_json_object(result_text)
    result["qc_mode"] = "vlm"
    result["montage_path"] = str(montage_path)
    result["analysis_timestamp"] = time.time()
    return result


def _load_json_object(text: str) -> dict[str, Any]:
    # Strip markdown code fence if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = match.group(1) if match else text
    result = json.loads(candidate.strip())
    if not isinstance(result, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return result


def _manual_review_analysis(
    reason: str, *, raw_response: str | None = None, error: str | None = None
) -> dict[str, Any]:
    unavailable = "disabled or not configured" in reason
    result: dict[str, Any] = {
        "qc_mode": "manual",
        "heterogeneity_detected": None,
        "severity": "unknown",
        "confidence": 0.0,
        "issues": [],
        "recommendations": [{"action": "Inspect montage before batch execution", "reason": reason}],
        "batch_processing_suitable": None,
        "summary": "[WARNING] VLM QC is unavailable. Ask the user to inspect the montage: reply 'yes' to continue batch execution, or 'no' to stop."
        if unavailable
        else f"[WARNING] {reason} Ask the user to inspect the montage before continuing.",
    }
    if raw_response is not None:
        result["raw_response"] = raw_response
    if error is not None:
        result["error"] = error
    return result


def _aggregate_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    if len(analyses) == 1:
        return analyses[0]
    heterogeneity_values = [analysis.get("heterogeneity_detected") for analysis in analyses]
    issues = [issue for analysis in analyses for issue in analysis.get("issues", [])]
    recommendations = [rec for analysis in analyses for rec in analysis.get("recommendations", [])]
    suitability_values = [analysis.get("batch_processing_suitable") for analysis in analyses]
    qc_modes = {analysis.get("qc_mode", "unknown") for analysis in analyses}
    return {
        "qc_mode": "mixed" if len(qc_modes) > 1 else next(iter(qc_modes), "unknown"),
        "heterogeneity_detected": True
        if any(heterogeneity_values)
        else None
        if None in heterogeneity_values
        else False,
        "confidence": max((_confidence_value(analysis.get("confidence")) for analysis in analyses), default=0.0),
        "issues": issues,
        "recommendations": recommendations,
        "batch_processing_suitable": False
        if False in suitability_values
        else None
        if None in suitability_values
        else True,
        "summary": "Multiple montages analyzed. Review combined findings above.",
    }


def _confidence_value(confidence: Any) -> float:
    """Normalize VLM confidence output to a 0–1 float."""
    if isinstance(confidence, bool) or confidence is None:
        return 0.0
    if isinstance(confidence, int | float):
        return float(confidence) / 100 if confidence > 1 else float(confidence)
    return 0.0


def _format_confidence(confidence: Any) -> str:
    if isinstance(confidence, str) and confidence.strip():
        return confidence
    return f"{_confidence_value(confidence):.0%}"


def _format_qc_report(analysis: dict[str, Any], metadata: dict[str, Any]) -> str:
    title, gate = _qc_gate(analysis)
    lines = [
        "## Batch QC Report",
        "",
        f"**QC Mode**: {analysis.get('qc_mode', 'unknown')}",
        f"**Dataset**: {metadata.get('folder_path', 'unknown')}",
        f"**Total Images**: {metadata.get('total_images', 'unknown')}",
        f"**Sample Size**: {metadata.get('sampled_count', 'unknown')}",
        f"**Loaded Images**: {metadata.get('n_images', 0)}",
        f"**Montage Count**: {metadata.get('montage_count', 1)}",
        f"**Load Errors**: {metadata.get('n_errors', 0)}",
        "",
        f"### {title}",
        f"**Confidence**: {_format_confidence(analysis.get('confidence', 0))}",
    ]
    lines.extend(
        f"- {issue.get('category', 'unknown')}: {issue.get('description', '')}"
        if isinstance(issue, dict)
        else f"- {issue}"
        for issue in analysis.get("issues", [])
    )
    lines.extend(
        f"- Recommendation: {rec.get('action', '')} ({rec.get('reason', '')})"
        if isinstance(rec, dict)
        else f"- Recommendation: {rec}"
        for rec in analysis.get("recommendations", [])
    )
    lines.extend(["", "**Batch Gate**:", gate])
    if analysis.get("summary"):
        lines.extend(["", "### Summary", analysis["summary"]])
    if metadata.get("errors"):
        lines.extend(["", "### Load Errors", *[f"- {err['path']}: {err['error']}" for err in metadata["errors"][:5]]])
    return "\n".join(lines)


def _qc_gate(analysis: dict[str, Any]) -> tuple[str, str]:
    if analysis.get("heterogeneity_detected") is None:
        first_rec = (analysis.get("recommendations") or [{"reason": "Manual review is required."}])[0]
        reason = (
            first_rec.get("reason", "Manual review is required.") if isinstance(first_rec, dict) else str(first_rec)
        )
        gate = "VLM is unavailable." if "disabled or not configured" in reason else reason
        return (
            "Manual Review Required",
            f"{gate} Wait for the user's yes/no decision: yes continues the workflow, no stops it.",
        )
    if analysis.get("heterogeneity_detected"):
        return (
            "[WARNING] Heterogeneity Detected",
            "VLM found a warning. Surface it to the user; this does not force-stop batch execution.",
        )
    return "VLM QC Passed", "VLM check passed. Continue to workflow execution."
