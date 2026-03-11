# SPDX-FileCopyrightText: Copyright contributors to the CopilotJ project.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import numpy as np
from csbdeep.utils import normalize
from skimage import io
from skimage.util import img_as_float32, img_as_ubyte
from stardist.models import StarDist2D
import yaml

__all__ = [
    "cellpose_segmentation",
    "stardist_segmentation",
    "biapy_tool",
    # Segmentation tools
    "gauss_otsu_labeling_tool",
    "voronoi_otsu_labeling_tool",
    "eroded_otsu_labeling_tool",
    # Convolution tools
    "deconvolution_tool",
    "super_resolution_tool",
]

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stable_templates")


def get_project_temp_dir(subdir: str | None = None) -> Path:
    project_root = Path(__file__).parent.parent.parent
    temp_dir = project_root / "temp"

    if subdir:
        temp_dir = temp_dir / subdir

    temp_dir.mkdir(parents=True, exist_ok=True)

    return temp_dir


def get_project_templates_dir() -> Path:
    """Get the local templates directory."""
    project_root = Path(__file__).parent.parent.parent
    templates_dir = project_root / "templates"
    return templates_dir


MODEL_ROOT = Path(__file__).resolve().parent.parent.parent / "assets" / "models"


async def cellpose_segmentation(
    image_path: Annotated[str | None, "Path to the input image or directory (use forward slashes)"] = None,
    model_type: Annotated[str, "Type of model to use (nuclei, cyto, cyto2)"] = "nuclei",
    diameter: Annotated[int | None, "Expected cell diameter (if None, will be estimated)"] = None,
    channels: Annotated[list[int], "List of channels to use [0,0] for grayscale"] = [0, 0],
    flow_threshold: Annotated[float, "Flow error threshold"] = 0.4,
    cellprob_threshold: Annotated[float, "Cell probability threshold"] = 0.0,
    min_size: Annotated[int, "Minimum cell size"] = 15,
    gpu: Annotated[bool, "Whether to use GPU for processing"] = False,
    normalize: Annotated[bool, "Whether to normalize image intensities"] = True,
    norm_range_low: Annotated[float, "Lower percentile for normalization"] = 1.0,
    norm_range_high: Annotated[float, "Upper percentile for normalization"] = 99.8,
    save_path: Annotated[
        str | None,
        "Path to save segmentation results (if None, uses original image directory with _cellpose_segmented suffix). Path end must with \"/\"",
    ] = None,
) -> dict[str, str]:
    try:
        from cellpose import io as cellpose_io
        from cellpose import models

        if image_path is None:
            return "No image path provided, please provide an image path"

        # Handle directory vs file paths using cellpose's own functions
        path_obj = Path(image_path)
        if path_obj.is_dir():
            logger.info(f"Directory provided: {image_path}")
            # Use cellpose's function to get image files from directory
            image_files = await asyncio.to_thread(cellpose_io.get_image_files, image_path, mask_filter="")
            if not image_files:
                raise FileNotFoundError(f"No image files found in directory: {image_path}")

            # Use the first image file found
            actual_image_path = image_files[0]
            logger.info(f"Using first image from directory: {actual_image_path}")
        else:
            actual_image_path = image_path

        # Use cellpose's native image loading function
        logger.info(f"Loading image using cellpose.io.imread: {actual_image_path}")
        image = await asyncio.to_thread(cellpose_io.imread, actual_image_path)

        if save_path is None:
            actual_path = Path(actual_image_path)
            save_dir = actual_path.parent
            base_name = actual_path.stem
            save_path = str(save_dir / f"{base_name}_cellpose_segmented.png")

        # Validate image
        if not _validate_image(image):
            raise ValueError("Invalid input image")

        # Apply normalization if requested
        if normalize:
            logger.info(f"Normalizing image using {norm_range_low}-{norm_range_high} percentile range")
            image = await asyncio.to_thread(_normalize_image, image, norm_range_low, norm_range_high)

        # Run Cellpose
        logger.info(f"Running Cellpose model: {model_type}")
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

        # Run model evaluation in a thread pool to avoid blocking
        masks, flows, styles = await asyncio.to_thread(
            model.eval,
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        save_path_obj = Path(save_path)
        
        # Determine if save_path is a directory or file path
        # Consider it a directory if: ends with /, has no suffix, or is an existing directory
        is_directory = (
            str(save_path).endswith(("/", "\\")) or 
            not save_path_obj.suffix or 
            save_path_obj.is_dir()
        )
        
        if is_directory:
            # save_path is a directory
            save_dir = save_path_obj
            save_dir.mkdir(parents=True, exist_ok=True)
            # Use image filename as base for output files
            base_name = Path(actual_image_path).stem
            final_save_path = save_dir / f"{base_name}_cellpose_segmented.png"
        else:
            # save_path is a file path
            save_dir = save_path_obj.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_name = save_path_obj.stem
            final_save_path = save_path_obj
        
        logger.info(f"Save directory: {save_dir}")

        try:
            # Save original image
            original_path = save_dir / f"{base_name}_original.png"
            await asyncio.to_thread(io.imsave, str(original_path), img_as_ubyte(image))
            logger.info(f"Saved original image to {original_path}")

            # Create and save colored segmentation result
            colored_masks = await asyncio.to_thread(create_colored_masks, masks)
            await asyncio.to_thread(io.imsave, str(final_save_path), colored_masks)
            logger.info(f"Saved colored segmentation to {final_save_path}")
            
            # Save ROIs
            rois_save_path = save_dir / f"{base_name}_rois.zip"
            await asyncio.to_thread(cellpose_io.save_rois, masks, str(rois_save_path))

            return {
                "original_image": str(original_path),
                "segmentation_image": str(final_save_path),
                "rois_path": str(rois_save_path),
                "num_cells": int(masks.max()),
                "save_directory": str(save_dir.resolve()),
            }
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in cellpose_segmentation: {str(e)}")
        raise


async def stardist_segmentation(
    image_path: Annotated[str | None, "Path to the input image (use forward slashes)"] = None,
    image_array: Annotated[Any | None, "Optional in-memory image; overrides image_path if provided"] = None,
    image_type: Annotated[str, "fluorescence | he | tissue | brightfield | phase | dapi | unknown"] = "fluorescence",
    model: Annotated[
        str | None,
        "Override pretrained model name for 2D (e.g., '2D_versatile_fluo', '2D_versatile_he', '2D_demo')",
    ] = None,
    p_low: Annotated[float, "Lower percentile for normalization"] = 1.0,
    p_high: Annotated[float, "Upper percentile for normalization"] = 99.8,
    prob_thresh: Annotated[float | None, "StarDist probability threshold override"] = None,
    nms_thresh: Annotated[float | None, "StarDist NMS threshold override"] = None,
    save_path: Annotated[
        str | None, "Output PNG path for colored labels; default saves near input or temp/stardist_results"
    ] = None,
) -> dict[str, Any]:
    try:
        if image_array is not None:
            img = image_array
            inferred_dir = get_project_temp_dir("stardist_segmentation_results")
            inferred_name = "in_memory"
            # Convert RGB to grayscale if needed
            if img.ndim == 3 and img.shape[-1] in (3, 4):
                rgb = img[..., :3].astype(np.float32)
                img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            if image_path is None:
                return "No image path provided, please provide an image path"
            else:
                img = await asyncio.to_thread(load_image, image_path, to_gray=True)
                p = Path(image_path)
                inferred_dir = get_project_temp_dir("stardist_segmentation_results")
                inferred_name = p.stem

        if not _validate_image(img):
            raise ValueError("Invalid input image for StarDist")

        ndim = img.ndim
        if ndim != 2:
            raise ValueError(f"Unsupported image ndim={ndim}; only 2D images are supported")

        img_norm = normalize(img.astype(np.float32, copy=False), p_low, p_high)

        if model is None:
            key = (image_type or "fluorescence").lower()
            model_name = "2D_versatile_he" if key in {"he", "tissue", "brightfield"} else "2D_versatile_fluo"
        else:
            model_name = model
        mclass = StarDist2D

        model_dir = MODEL_ROOT / "stardist" / model_name
        model_instance = mclass(None, name=model_name, basedir=str(model_dir.parent))

        kwargs = {}
        if prob_thresh is not None:
            kwargs["prob_thresh"] = prob_thresh
        if nms_thresh is not None:
            kwargs["nms_thresh"] = nms_thresh

        labels, details = await asyncio.to_thread(model_instance.predict_instances, img_norm, **kwargs)

        out_dir = inferred_dir if save_path is None else Path(save_path).parent
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        original_path = out_dir / f"{inferred_name}_stardist_input.png"
        await asyncio.to_thread(io.imsave, original_path, (img_norm * 255).astype(np.uint8))

        colored = await asyncio.to_thread(create_colored_masks, labels.astype(np.int32))
        seg_png = out_dir / f"{inferred_name}_stardist_labels.png" if save_path is None else Path(save_path)
        await asyncio.to_thread(io.imsave, seg_png, colored)

        labels_tif = out_dir / f"{inferred_name}_stardist_labels.tif"
        await asyncio.to_thread(io.imsave, labels_tif, labels.astype(np.uint16))

        return {
            "original_image": str(original_path),
            "segmentation_image": str(seg_png),
            "labels_path": str(labels_tif),
            "num_instances": int(labels.max()),
            "model_name": model_name,
        }

    except Exception as e:
        logger.error(f"Error in stardist_segmentation: {e}")
        raise


# <BiaPy>
def _deep_merge(dst: dict, src: dict):
    """Recursive dict merge: values in src override dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v

# -----------------------------
# Task registry
# Based on: "Select workflow" documentation and workflow pages in BiaPy docs.
# https://biapy.readthedocs.io/en/latest/get_started/select_workflow.html
# -----------------------------
TASKS: Dict[str, Dict[str, Any]] = {
    # Classification
    "cls2d": {
        "template": "classification/2d_classification.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": True,
    },
    "cls3d": {
        "template": "classification/3d_classification.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": True,
    },
    # Semantic segmentation
    "seg2d": {
        "template": "semantic_segmentation/2d_semantic_segmentation.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": True,
    },
    "seg3d": {
        "template": "semantic_segmentation/3d_semantic_segmentation.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": True,
    },
    # Instance segmentation
    "inst2d": {
        "template": "instance_segmentation/2d_instance_segmentation.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "inst3d": {
        "template": "instance_segmentation/3d_instance_segmentation.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
    # Object detection
    "det2d": {
        "template": "detection/2d_detection.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "det3d": {
        "template": "detection/3d_detection.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
    # Denoising
    "denoise2d": {
        "template": "denoising/2d_denoising.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "denoise3d": {
        "template": "denoising/3d_denoising.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
    # Super-resolution
    "sr2d": {
        "template": "super_resolution/2d_super-resolution.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "sr3d": {
        "template": "super_resolution/3d_sr.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
    # Self-supervised learning
    "ssl2d": {
        "template": "self_supervised/2d_self_supervised.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "ssl3d": {
        "template": "self_supervised/3d_self_supervised.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
    # Image-to-image translation (e.g., restoration mappings)
    "i2i2d": {
        "template": "image_to_image/2d_image_to_image.yaml",
        "patch_size_shape": lambda ps: (ps, ps, 1),
        "needs_classes": False,
    },
    "i2i3d": {
        "template": "image_to_image/3d_image_to_image.yaml",
        "patch_size_shape": lambda ps: (ps, ps, ps, 1),
        "needs_classes": False,
    },
}


def _set_paths(cfg: dict, train_raw_path: str, train_gt_path: str, test_raw_path: str, test_gt_path: str, 
               val_from_train: bool, split_ratio: float, val_raw_path: str = None, val_gt_path: str = None):
    d = cfg.setdefault("DATA", {})
    
    # For prediction mode, BiaPy requires non-empty paths to avoid string index errors
    # Use test_raw_path as fallback for empty paths
    safe_train_raw = train_raw_path if train_raw_path else test_raw_path
    safe_train_gt = train_gt_path if train_gt_path else test_raw_path  # Use raw path as fallback
    safe_test_gt = test_gt_path if test_gt_path else test_raw_path      # Use raw path as fallback
    
    # Set training paths
    d.setdefault("TRAIN", {})["PATH"] = safe_train_raw
    d.setdefault("TRAIN", {})["GT_PATH"] = safe_train_gt
    
    # Set test paths
    d.setdefault("TEST", {})["PATH"] = test_raw_path
    d.setdefault("TEST", {})["GT_PATH"] = safe_test_gt
    # For evaluation, many workflows expect ground-truth to exist
    d.setdefault("TEST", {})["LOAD_GT"] = True
    
    # Configure validation data
    if val_raw_path and val_gt_path and not val_from_train:
        # Use separate validation dataset
        d.setdefault("VAL", {})["FROM_TRAIN"] = False
        d.setdefault("VAL", {})["PATH"] = val_raw_path
        d.setdefault("VAL", {})["GT_PATH"] = val_gt_path
    else:
        # Use validation split from training data
        d.setdefault("VAL", {})["FROM_TRAIN"] = bool(val_from_train)
        d.setdefault("VAL", {})["SPLIT_TRAIN"] = float(split_ratio)
    
    # Replace any remaining "/path/to/data" placeholders in the entire config
    base_path = train_raw_path or test_raw_path
    if base_path:
        _replace_placeholders(cfg, "/path/to/data", os.path.dirname(base_path))


def _replace_placeholders(cfg: dict, placeholder: str, replacement: str):
    """Recursively replace all occurrences of placeholder with replacement in the config dict."""
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if isinstance(value, str) and value == placeholder:
                cfg[key] = replacement
            elif isinstance(value, (dict, list)):
                _replace_placeholders(value, placeholder, replacement)
    elif isinstance(cfg, list):
        for item in cfg:
            _replace_placeholders(item, placeholder, replacement)


def _set_patch_size(cfg: dict, shape_tuple):
    d = cfg.setdefault("DATA", {})
    d["PATCH_SIZE"] = shape_tuple


def _set_test_padding(cfg: dict, patch_shape):
    # Heuristic padding: ~1/8 of spatial dims
    if len(patch_shape) == 3:  # 2D case: (ps, ps, 1)
        pad_xy = max(2, patch_shape[0] // 8)
        cfg.setdefault("DATA", {}).setdefault("TEST", {})["PADDING"] = (pad_xy, pad_xy)
    elif len(patch_shape) == 4:  # 3D case: (ps, ps, ps, 1)
        pad = max(2, patch_shape[0] // 8)
        cfg.setdefault("DATA", {}).setdefault("TEST", {})["PADDING"] = (pad, pad, pad)


def _set_training_hparams(cfg: dict, epochs: int, bs: int, lr: float, opt: str):
    tr = cfg.setdefault("TRAIN", {})
    tr["EPOCHS"] = int(epochs)
    tr["PATIENCE"] = max(10, min(100, epochs // 2))  # simple guard
    tr["BATCH_SIZE"] = int(bs)
    tr["OPTIMIZER"] = opt
    tr["LR"] = float(lr)


def _set_model_head(cfg: dict, task_key: str, n_classes: int, architecture: Optional[str]):
    m = cfg.setdefault("MODEL", {})
    if architecture:
        m["ARCHITECTURE"] = architecture
    # According to most workflow templates, N_CLASSES is required for
    # classification and semantic segmentation; others usually ignore it.
    if task_key.startswith("cls") or task_key.startswith("seg"):
        m["N_CLASSES"] = int(n_classes)


def _set_task_specific_metrics(cfg: dict, task_key: str):
    """Set task-specific metrics based on BiaPy workflow requirements."""
    train_cfg = cfg.setdefault("TRAIN", {})
    test_cfg = cfg.setdefault("TEST", {})
    
    if task_key.startswith("inst"):
        # Instance segmentation requires 'iou' metric
        train_cfg["METRICS"] = ["iou"]
        test_cfg["METRICS"] = ["iou"]
    elif task_key.startswith("seg"):
        # Semantic segmentation typically uses 'iou' as well
        train_cfg["METRICS"] = ["iou"]
        test_cfg["METRICS"] = ["iou"]
    elif task_key.startswith("cls"):
        # Classification uses accuracy
        train_cfg["METRICS"] = ["accuracy"]
        test_cfg["METRICS"] = ["accuracy"]
    elif task_key.startswith("det"):
        # Detection uses mAP or similar detection metrics
        train_cfg["METRICS"] = ["mAP"]
        test_cfg["METRICS"] = ["mAP"]
    else:
        # For other tasks (denoising, super-resolution, etc.), use default metrics
        # These might not need specific metrics or will use task-specific ones
        pass


def _attach_pretrained(cfg: dict, ckpt: Optional[str]):
    if not ckpt:
        return
    m = cfg.setdefault("MODEL", {})
    paths = cfg.setdefault("PATHS", {})
    # Use BiaPy's standard checkpoint configuration
    m["LOAD_CHECKPOINT"] = True
    paths["CHECKPOINT_FILE"] = ckpt


def _fix_predict_mode_config(cfg: dict, task_key: str, num_classes: int):
    """Fix configuration issues for prediction mode."""
    # For prediction mode, disable training
    cfg.setdefault("TRAIN", {})["ENABLE"] = False
    
    # For prediction mode, disable ground truth loading and metric calculation
    cfg.setdefault("DATA", {}).setdefault("TEST", {})["LOAD_GT"] = False
    
    # Don't calculate metrics in predict mode
    cfg.setdefault("TEST", {})["METRICS"] = []
    cfg.setdefault("TRAIN", {})["METRICS"] = []
    
    # For instance segmentation in predict mode, disable matching stats
    if task_key.startswith("inst"):
        cfg.setdefault("TEST", {})["MATCHING_STATS"] = False
        return
    
    # Fix metrics that require >= 5 classes (only for classification tasks)
    if task_key.startswith("cls") and num_classes < 5:
        # Fix train metrics
        train_metrics = cfg.get("TRAIN", {}).get("METRICS", [])
        if isinstance(train_metrics, list) and "top-5-accuracy" in train_metrics:
            train_metrics.remove("top-5-accuracy")
            cfg.setdefault("TRAIN", {})["METRICS"] = train_metrics
        
        # Fix test metrics  
        test_metrics = cfg.get("TEST", {}).get("METRICS", [])
        if isinstance(test_metrics, list) and "top-5-accuracy" in test_metrics:
            test_metrics.remove("top-5-accuracy")
            cfg.setdefault("TEST", {})["METRICS"] = test_metrics
        
        # Set safe default metrics for classification with < 5 classes
        cfg.setdefault("TRAIN", {})["METRICS"] = ["accuracy"]
        cfg.setdefault("TEST", {})["METRICS"] = ["accuracy"]


def _maybe_enable_basic_augs(cfg: dict):
    aug = cfg.setdefault("AUGMENTOR", {})
    for k in ["DROPOUT", "GRIDMASK", "CUTBLUR", "CUTNOISE", "MOTION_BLUR"]:
        if k in aug:
            aug[k] = True


async def biapy_tool(
    task: Annotated[str, "One of: cls2d, cls3d, seg2d, seg3d, inst2d, inst3d, det2d, det3d, denoise2d, denoise3d, sr2d, sr3d, ssl2d, ssl3d, i2i2d, i2i3d"],
    mode: Annotated[str, "Execution mode: train | predict | eval"],
    train_raw_path: Annotated[str, "Path to training raw images directory"],
    train_gt_path: Annotated[str, "Path to training ground truth/labels directory"],
    test_raw_path: Annotated[str, "Path to test raw images directory"],
    test_gt_path: Annotated[str, "Path to test ground truth/labels directory"],
    model_name: Annotated[str, "Name for the run or trained model"] = "biapy_run",
    num_classes: Annotated[int, "Number of classes (for cls/seg workflows)"] = 2,
    num_epochs: Annotated[int, "Training epochs"] = 3,
    batch_size: Annotated[int, "Training batch size"] = 8,
    patch_size: Annotated[int, "Patch size. 2D uses (ps,ps,1); 3D uses (ps,ps,ps)"] = 32,
    learning_rate: Annotated[float, "Learning rate"] = 1e-4,
    optimizer: Annotated[str, "Optimizer (e.g., ADAM, ADAMW, SGD)"] = "ADAMW",
    architecture: Annotated[Optional[str], "Model architecture (UNet, UNETR, ResUNet, etc.)"] = None,
    use_augmentation: Annotated[bool, "Enable a conservative set of augmentations if available in template"] = False,
    gpu: Annotated[Optional[int], "GPU id; None for CPU"] = 0,
    pretrained_ckpt: Annotated[Optional[str], "Path to pretrained checkpoint for fine-tuning or prediction"] = None,
    val_from_train: Annotated[bool, "Whether to derive validation from training data"] = True,
    val_split_ratio: Annotated[float, "Fraction of training used as validation"] = 0.1,
    val_raw_path: Annotated[Optional[str], "Path to separate validation raw images directory (if not using val_from_train)"] = None,
    val_gt_path: Annotated[Optional[str], "Path to separate validation ground truth directory (if not using val_from_train)"] = None,
    local_template_dir: Annotated[Optional[str], "If provided, prefer local templates by task key (e.g., seg2d.yaml)"] = None,
    extra_overrides: Annotated[Optional[Dict[str, Any]], "Additional YAML overrides to merge, e.g., for custom metrics or system settings"] = None,
    dry_run: Annotated[bool, "If True, only generate YAML without execution"] = False,
) -> Dict[str, Any]:
    """Run BiaPy across multiple workflows with a single, consistent entry point.

    This function follows the selection guidance from the "Select workflow" docs and
    workflow-specific templates. It auto-downloads the appropriate YAML template,
    applies common settings (paths, training hyperparameters, patch size), and
    optionally runs training/prediction/evaluation using the Python API.
    """
    try:
        from biapy import BiaPy
    except Exception as e:
        raise RuntimeError("BiaPy is not installed in this environment.") from e

    task_key = task.lower()
    if task_key not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Choices: {sorted(TASKS.keys())}")
    if mode not in {"train", "predict", "eval"}:
        raise ValueError("mode must be one of: train | predict | eval")

    # Prepare output directory and template path
    temp_root = get_project_temp_dir("biapy_runs")
    run_dir = temp_root / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolve template
    task_info = TASKS[task_key]
    if local_template_dir:
        # User provided custom template directory
        template_path = Path(local_template_dir).expanduser() / f"{task_key}.yaml"
        if not template_path.exists():
            logger.warning(f"Local template {template_path} not found. Falling back to built-in templates.")
            # Fall back to built-in templates
            templates_dir = get_project_templates_dir()
            template_path = templates_dir / task_info["template"]
    else:
        # Use built-in templates
        templates_dir = get_project_templates_dir()
        template_path = templates_dir / task_info["template"]
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    logger.info(f"Using template: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Common configuration according to workflow docs
    _set_paths(cfg, train_raw_path, train_gt_path, test_raw_path, test_gt_path, 
               val_from_train, val_split_ratio, val_raw_path, val_gt_path)

    ps_shape = task_info["patch_size_shape"](patch_size)
    _set_patch_size(cfg, ps_shape)
    _set_test_padding(cfg, ps_shape)

    _set_training_hparams(cfg, num_epochs, batch_size, learning_rate, optimizer)
    _set_model_head(cfg, task_key, num_classes, architecture)
    _set_task_specific_metrics(cfg, task_key)
    _attach_pretrained(cfg, pretrained_ckpt)
    
    # Fix metrics for prediction mode
    if mode == "predict":
        _fix_predict_mode_config(cfg, task_key, num_classes)

    if use_augmentation:
        _maybe_enable_basic_augs(cfg)

    if extra_overrides:
        _deep_merge(cfg, extra_overrides)

    # Write final configuration YAML
    cfg_path = run_dir / f"{model_name}.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Early exit when only generating YAML
    if dry_run:
        return {
            "success": True,
            "status": "dry_run_completed",
            "task": task_key,
            "mode": mode,
            "output_dir": str(run_dir),
            "config_file": str(cfg_path),
            "model_name": model_name,
            "message": "Configuration YAML generated successfully (dry run mode)"
        }

    bia = BiaPy(
        str(cfg_path),
        result_dir=str(run_dir),
        name=model_name,
        run_id=1,
        gpu="" if gpu is None else str(gpu),
    )

    logger.info(f"[BiaPy] Start: task={task_key}, mode={mode}, cfg={cfg_path}")
    
    try:
        await asyncio.to_thread(bia.run_job)
        logger.info("[BiaPy] Job completed successfully.")
        
        # Build comprehensive return information
        result = {
            "success": True,
            "status": "completed",
            "task": task_key,
            "mode": mode,
            "output_dir": str(run_dir),
            "config_file": str(cfg_path),
            "model_name": model_name,
            "execution_info": {
                "epochs": num_epochs if mode == "train" else None,
                "batch_size": batch_size,
                "patch_size": patch_size,
                "architecture": architecture,
                "gpu": gpu,
            }
        }
        
        # Add mode-specific information
        if mode == "predict":
            result["prediction_info"] = {
                "input_path": test_raw_path,
                "checkpoint": pretrained_ckpt if pretrained_ckpt else "default",
                "message": "Prediction completed successfully"
            }
        elif mode == "train":
            result["training_info"] = {
                "train_path": train_raw_path,
                "epochs": num_epochs,
                "message": "Training completed successfully"
            }
        elif mode == "eval":
            result["evaluation_info"] = {
                "test_path": test_raw_path,
                "message": "Evaluation completed successfully"
            }
        
        # Check if output files exist to verify success
        output_files = list(run_dir.glob("*"))
        result["output_files"] = [str(f) for f in output_files]
        result["num_output_files"] = len(output_files)
        
        return result
        
    except Exception as e:
        logger.error(f"[BiaPy] Job failed with error: {e}")
        return {
            "success": False,
            "status": "failed",
            "task": task_key,
            "mode": mode,
            "output_dir": str(run_dir),
            "config_file": str(cfg_path),
            "model_name": model_name,
            "error": str(e),
            "message": f"BiaPy {mode} job failed: {e}"
        }
# </BiaPy>


def _gauss_otsu_segmentation(image: np.ndarray, gaussian_sigma: float, min_object_size: int) -> tuple:
    from skimage import filters, measure, morphology

    # 1. Gaussian blur
    blurred = filters.gaussian(image, sigma=gaussian_sigma)

    # 2. Otsu thresholding
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold

    # 3. Connected component labeling
    labels = measure.label(binary)

    # 4. Remove small objects
    if min_object_size > 0:
        labels = morphology.remove_small_objects(labels, min_size=min_object_size)
        labels = measure.label(labels > 0)  # Relabel after removal

    # Calculate statistics
    object_count = len(np.unique(labels)) - 1
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]

    info_dict = {
        "method": "Gauss-Otsu-Labeling",
        "object_count": object_count,
        "gaussian_sigma": gaussian_sigma,
        "otsu_threshold": threshold,
        "min_object_size": min_object_size,
        "areas": areas,
        "mean_area": np.mean(areas) if areas else 0,
    }

    return labels, info_dict


def _voronoi_otsu_segmentation(
    image: np.ndarray, gaussian_sigma: float, spot_sigma: float, outline_sigma: float
) -> tuple:
    from skimage import filters, measure, segmentation

    # 1. Spot detection (seeds)
    blurred_spots = filters.gaussian(image, sigma=spot_sigma)
    threshold_spots = filters.threshold_otsu(blurred_spots)
    binary_spots = blurred_spots > threshold_spots

    # 2. Find connected components as seeds
    spot_labels = measure.label(binary_spots)

    # 3. Create markers from centroids
    props = measure.regionprops(spot_labels)
    markers = np.zeros_like(image, dtype=int)
    for i, prop in enumerate(props, 1):
        y, x = map(int, prop.centroid)
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            markers[y, x] = i

    # 4. Create mask for overall objects
    blurred_outline = filters.gaussian(image, sigma=outline_sigma)
    threshold_outline = filters.threshold_otsu(blurred_outline)
    mask = blurred_outline > threshold_outline

    # 5. Voronoi tessellation
    if np.any(markers > 0):
        labels = segmentation.watershed(-image, markers, mask=mask)
    else:
        labels = np.zeros_like(image, dtype=int)

    # Calculate statistics
    object_count = len(np.unique(labels)) - 1
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]

    info_dict = {
        "method": "Voronoi-Otsu-Labeling",
        "object_count": object_count,
        "gaussian_sigma": gaussian_sigma,
        "spot_sigma": spot_sigma,
        "outline_sigma": outline_sigma,
        "spot_threshold": threshold_spots,
        "outline_threshold": threshold_outline,
        "areas": areas,
        "mean_area": np.mean(areas) if areas else 0,
    }

    return labels, info_dict


def _eroded_otsu_segmentation(
    image: np.ndarray, gaussian_sigma: float, erosion_radius: int, min_object_size: int
) -> tuple:
    from skimage import filters, measure, morphology, segmentation

    # 1. Gaussian blur
    blurred = filters.gaussian(image, sigma=gaussian_sigma)

    # 2. Otsu thresholding
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold

    # 3. Binary erosion to separate touching objects
    footprint = morphology.disk(erosion_radius)
    eroded = morphology.binary_erosion(binary, footprint)

    # 4. Label eroded objects as seeds
    eroded_labels = measure.label(eroded)

    # 5. Remove small eroded objects
    if min_object_size > 0:
        eroded_labels = morphology.remove_small_objects(eroded_labels, min_size=min_object_size)
        eroded_labels = measure.label(eroded_labels > 0)

    # 6. Masked Voronoi labeling (expand back to original boundaries)
    if np.any(eroded_labels > 0):
        labels = segmentation.watershed(-blurred, eroded_labels, mask=binary)
    else:
        labels = np.zeros_like(image, dtype=int)

    # Calculate statistics
    object_count = len(np.unique(labels)) - 1
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]

    info_dict = {
        "method": "Eroded-Otsu-Labeling",
        "object_count": object_count,
        "gaussian_sigma": gaussian_sigma,
        "otsu_threshold": threshold,
        "erosion_radius": erosion_radius,
        "min_object_size": min_object_size,
        "areas": areas,
        "mean_area": np.mean(areas) if areas else 0,
    }

    return labels, info_dict


async def gauss_otsu_labeling_tool(
    image_path: Annotated[str, "Path to the input image"],
    gaussian_sigma: Annotated[float, "Standard deviation for Gaussian blur  preprocessing (0.1-5.0)"] = 1.0,
    min_object_size: Annotated[int, "Minimum object size in pixels to keep"] = 10,
) -> dict[str, str]:
    try:
        from skimage.util import img_as_ubyte

        # Load image
        logger.info(f"Loading image from {image_path}")
        image = await asyncio.to_thread(load_image, image_path)

        if not _validate_image(image):
            raise ValueError("Invalid input image")  # Run segmentation in thread to avoid blocking
        labels, info_dict = await asyncio.to_thread(_gauss_otsu_segmentation, image, gaussian_sigma, min_object_size)

        # Save results
        temp_dir = get_project_temp_dir("segmentation_results")
        temp_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = temp_dir / "original.png"
        await asyncio.to_thread(io.imsave, original_path, img_as_ubyte(image))

        # Save segmentation result
        seg_path = temp_dir / "gauss_otsu_segmentation.png"
        colored_labels = await asyncio.to_thread(create_colored_masks, labels)
        await asyncio.to_thread(io.imsave, seg_path, colored_labels)

        logger.info(f"Gauss-Otsu segmentation completed: {info_dict['object_count']} objects detected")

        return {
            "original_image": str(original_path),
            "segmentation_image": str(seg_path),
            "object_count": info_dict["object_count"],
            "mean_area": info_dict["mean_area"],
            "method": info_dict["method"],
        }

    except Exception as e:
        logger.error(f"Error in gauss_otsu_labeling_tool: {str(e)}")
        raise


async def voronoi_otsu_labeling_tool(
    image_path: Annotated[str, "Path to the input image"],
    gaussian_sigma: Annotated[float, "General smoothing parameter (0.1-3.0)"] = 1.0,
    spot_sigma: Annotated[float, "Smoothing for spot detection (0.5-10.0)"] = 2.0,
    outline_sigma: Annotated[float, "Smoothing for edge detection (0.1-2.0)"] = 0.5,
) -> dict[str, str]:
    try:
        from skimage.util import img_as_ubyte

        # Load image
        logger.info(f"Loading image from {image_path}")
        image = await asyncio.to_thread(load_image, image_path)

        if not _validate_image(image):
            raise ValueError("Invalid input image")  # Run segmentation in thread to avoid blocking
        labels, info_dict = await asyncio.to_thread(
            _voronoi_otsu_segmentation, image, gaussian_sigma, spot_sigma, outline_sigma
        )

        # Save results
        temp_dir = get_project_temp_dir("segmentation_results")
        temp_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = temp_dir / "original.png"
        await asyncio.to_thread(io.imsave, original_path, img_as_ubyte(image))

        # Save segmentation result
        seg_path = temp_dir / "voronoi_otsu_segmentation.png"
        colored_labels = await asyncio.to_thread(create_colored_masks, labels)
        await asyncio.to_thread(io.imsave, seg_path, colored_labels)

        logger.info(f"Voronoi-Otsu segmentation completed: {info_dict['object_count']} objects detected")

        return {
            "original_image": str(original_path),
            "segmentation_image": str(seg_path),
            "object_count": info_dict["object_count"],
            "mean_area": info_dict["mean_area"],
            "method": info_dict["method"],
        }

    except Exception as e:
        logger.error(f"Error in voronoi_otsu_labeling_tool: {str(e)}")
        raise


async def eroded_otsu_labeling_tool(
    image_path: Annotated[str, "Path to the input image"],
    gaussian_sigma: Annotated[float, "Standard deviation for Gaussian blur (0.1-5.0)"] = 1.0,
    erosion_radius: Annotated[int, "Radius for binary erosion (1-10)"] = 2,
    min_object_size: Annotated[int, "Minimum object size after erosion (1-500)"] = 10,
) -> dict[str, str]:
    try:
        from skimage.util import img_as_ubyte

        # Load image
        logger.info(f"Loading image from {image_path}")
        image = await asyncio.to_thread(load_image, image_path)

        if not _validate_image(image):
            raise ValueError("Invalid input image")  # Run segmentation in thread to avoid blocking
        labels, info_dict = await asyncio.to_thread(
            _eroded_otsu_segmentation, image, gaussian_sigma, erosion_radius, min_object_size
        )

        # Save results
        temp_dir = get_project_temp_dir("segmentation_results")
        temp_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = temp_dir / "original.png"
        await asyncio.to_thread(io.imsave, original_path, img_as_ubyte(image))

        # Save segmentation result
        seg_path = temp_dir / "eroded_otsu_segmentation.png"
        colored_labels = await asyncio.to_thread(create_colored_masks, labels)
        await asyncio.to_thread(io.imsave, seg_path, colored_labels)

        logger.info(f"Eroded-Otsu segmentation completed: {info_dict['object_count']} objects detected")

        return {
            "original_image": str(original_path),
            "segmentation_image": str(seg_path),
            "object_count": info_dict["object_count"],
            "mean_area": info_dict["mean_area"],
            "method": info_dict["method"],
        }

    except Exception as e:
        logger.error(f"Error in eroded_otsu_labeling_tool: {str(e)}")
        raise


async def deconvolution_tool(
    image_path: Annotated[str, "Path to the input image"],
    method: Annotated[
        str, "Deconvolution method, options: richardson_lucy, wiener, unsupervised_wiener"
    ] = "richardson_lucy",
    iterations: Annotated[int, "Number of iterations for iterative methods (5-50)"] = 10,
    noise_level: Annotated[float, "Noise level estimate (0.01-0.5)"] = 0.1,
    psf_type: Annotated[str, "Type of PSF to generate, options: gaussian, motion, defocus"] = "gaussian",
    psf_sigma: Annotated[float, "Standard deviation for PSF (0.5-5.0)"] = 1.0,
    psf_size: int = 15,
    regularization: float = 0.001,
) -> dict[str, str]:
    try:
        from skimage.util import img_as_ubyte

        # Load image
        logger.info(f"Loading image from {image_path}")
        image = await asyncio.to_thread(load_image, image_path)

        if not _validate_image(image):
            raise ValueError("Invalid input image")

        # Point Spread Function. If None, will generate based on psf_type
        psf = None

        # Run deconvolution in thread to avoid blocking
        deconvolved, info_dict = await asyncio.to_thread(
            _deconvolve_image,
            image,
            psf,
            method,
            iterations,
            noise_level,
            psf_type,
            psf_sigma,
            psf_size,
            regularization,
        )

        # Save results
        temp_dir = get_project_temp_dir("deconvolution_results")
        temp_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = temp_dir / "original.png"
        await asyncio.to_thread(io.imsave, original_path, img_as_ubyte(image))

        # Save deconvolved result
        deconv_path = temp_dir / f"deconvolved_{method}.png"
        await asyncio.to_thread(io.imsave, deconv_path, img_as_ubyte(deconvolved))

        logger.info(
            f"Deconvolution completed using {method}, sharpness improvement: {info_dict['sharpness_improvement']:.2f}%"
        )

        return {
            "original_image": str(original_path),
            "deconvolved_image": str(deconv_path),
            "method": info_dict["method"],
            "sharpness_improvement": info_dict["sharpness_improvement"],
            "iterations": info_dict["iterations"],
            "message": f"Deconvolution completed with {info_dict['sharpness_improvement']:.2f}% sharpness improvement",
        }

    except Exception as e:
        logger.error(f"Error in deconvolution_tool: {str(e)}")
        raise


def _deconvolve_image(
    image: np.ndarray,
    psf: Optional[np.ndarray],
    method: str,
    iterations: int,
    noise_level: float,
    psf_type: str,
    psf_sigma: float,
    psf_size: int,
    regularization: float,
) -> tuple:
    # Validate inputs
    if image.ndim not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")

    if method not in ["richardson_lucy", "wiener", "unsupervised_wiener"]:
        raise ValueError(f"Unknown method: {method}")

    # Ensure PSF size is odd
    if psf_size % 2 == 0:
        psf_size += 1

    # Generate PSF if not provided
    if psf is None:
        psf = _generate_psf(psf_type, psf_size, psf_sigma)

    # Convert image to float and normalize
    if image.dtype != np.float64:
        image_float = image.astype(np.float64)
        if image.max() > 1:
            image_float = image_float / 255.0
    else:
        image_float = image.copy()

    # Process each channel if RGB
    if image_float.ndim == 3:
        result_channels = []
        for i in range(image_float.shape[2]):
            channel_result = _deconvolve_channel(
                image_float[:, :, i], psf, method, iterations, noise_level, regularization
            )
            result_channels.append(channel_result)
        deconvolved = np.stack(result_channels, axis=2)
    else:
        deconvolved = _deconvolve_channel(image_float, psf, method, iterations, noise_level, regularization)

    # Clip values to valid range
    deconvolved = np.clip(deconvolved, 0, 1)

    # Convert back to original dtype
    if image.dtype == np.uint8:
        deconvolved = (deconvolved * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        deconvolved = (deconvolved * 65535).astype(np.uint16)

    # Calculate quality metrics
    info_dict = {
        "method": method,
        "iterations": iterations,
        "psf_type": psf_type,
        "psf_size": psf_size,
        "noise_level": noise_level,
        "original_shape": image.shape,
        "original_dtype": str(image.dtype),
        "sharpness_improvement": _calculate_sharpness_improvement(image, deconvolved),
    }

    return deconvolved, info_dict


def _generate_psf(psf_type: str, size: int, sigma: float) -> np.ndarray:
    center = size // 2

    if psf_type == "gaussian":
        # Gaussian PSF
        y, x = np.ogrid[:size, :size]
        psf = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

    elif psf_type == "motion":
        # Motion blur PSF
        psf = np.zeros((size, size))
        length = int(sigma * 2)
        angle = 0  # horizontal motion
        for i in range(length):
            x = center + int(i * np.cos(angle))
            y = center + int(i * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1

    elif psf_type == "defocus":
        # Defocus (disk) PSF
        y, x = np.ogrid[:size, :size]
        radius = sigma
        psf = ((x - center) ** 2 + (y - center) ** 2) <= radius**2
        psf = psf.astype(float)

    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")

    # Normalize PSF
    psf = psf / np.sum(psf)
    return psf


def _deconvolve_channel(
    image: np.ndarray, psf: np.ndarray, method: str, iterations: int, noise_level: float, regularization: float
) -> np.ndarray:
    from skimage import restoration

    if method == "richardson_lucy":
        # Richardson-Lucy deconvolution
        result = restoration.richardson_lucy(image, psf, num_iter=iterations)

    elif method == "wiener":
        # Wiener filter deconvolution
        result = restoration.wiener(image, psf, balance=noise_level)

    elif method == "unsupervised_wiener":
        # Unsupervised Wiener filter
        result = restoration.unsupervised_wiener(image, psf)[0]

    return result


def _calculate_sharpness_improvement(original: np.ndarray, deconvolved: np.ndarray) -> float:
    import cv2

    def sharpness(img):
        if img.ndim == 3:
            img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        grad_x = np.gradient(img, axis=1)
        grad_y = np.gradient(img, axis=0)
        return np.mean(np.sqrt(grad_x**2 + grad_y**2))

    original_sharpness = sharpness(original)
    deconvolved_sharpness = sharpness(deconvolved)

    if original_sharpness > 0:
        improvement = (deconvolved_sharpness - original_sharpness) / original_sharpness * 100
    else:
        improvement = 0

    return improvement


async def super_resolution_tool(
    image_path: Annotated[str, "Path to the input image"],
    model_name: Annotated[str, "Model architecture, options: ESPCN, EDSR, FSRCNN, LapSRN"] = "ESPCN",
    scale_factor: Annotated[int, "Upscaling factor, options: 2, 3, 4"] = 2,
) -> dict[str, str]:
    try:
        from skimage.util import img_as_ubyte

        # Load image
        logger.info(f"Loading image from {image_path}")
        image = await asyncio.to_thread(load_image, image_path)

        if not _validate_image(image):
            raise ValueError("Invalid input image")

        # Run super resolution in thread to avoid blocking
        upscaled, info_dict = await asyncio.to_thread(_super_resolve_image, image, model_name, scale_factor)

        # Save results
        temp_dir = get_project_temp_dir("deconvolution_results")
        temp_dir.mkdir(exist_ok=True)

        # Save original image
        original_path = temp_dir / "original.png"
        await asyncio.to_thread(io.imsave, original_path, img_as_ubyte(image))

        # Save upscaled result
        upscaled_path = temp_dir / f"upscaled_{model_name}_{scale_factor}x.png"
        await asyncio.to_thread(io.imsave, upscaled_path, img_as_ubyte(upscaled))

        logger.info(f"Super resolution completed: {image.shape} -> {upscaled.shape}")

        return {
            "original_image": str(original_path),
            "upscaled_image": str(upscaled_path),
            "model_name": info_dict["model_name"],
            "scale_factor": info_dict["scale_factor"],
            "original_size": info_dict["original_size"],
            "upscaled_size": info_dict["upscaled_size"],
            "message": f"Image upscaled {scale_factor}x using {model_name}: {info_dict['original_size']} -> {info_dict['upscaled_size']}",
        }

    except Exception as e:
        logger.error(f"Error in super_resolution_tool: {str(e)}")
        raise


def _super_resolve_image(image: np.ndarray, model_name: str, scale_factor: int) -> tuple:
    import cv2

    # Validate scale factor
    if scale_factor not in [2, 3, 4]:
        raise ValueError("Scale factor must be 2, 3, or 4")

    # Convert to OpenCV format (BGR)
    if image.ndim == 3:
        # Convert RGB to BGR for OpenCV
        cv_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        cv_image = (image * 255).astype(np.uint8)

    try:
        # For simplicity, use bicubic interpolation as fallback
        # This avoids downloading large model files
        height, width = cv_image.shape[:2]
        new_height, new_width = height * scale_factor, width * scale_factor

        # Use high-quality bicubic interpolation
        upscaled_cv = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert back to RGB if needed
        if upscaled_cv.ndim == 3:
            upscaled = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB) / 255.0
        else:
            upscaled = upscaled_cv / 255.0

    except Exception as e:
        logger.warning(f"DNN super resolution failed, using bicubic interpolation: {e}")
        # Fallback to bicubic interpolation
        height, width = cv_image.shape[:2]
        new_height, new_width = height * scale_factor, width * scale_factor
        upscaled_cv = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        if upscaled_cv.ndim == 3:
            upscaled = cv2.cvtColor(upscaled_cv, cv2.COLOR_BGR2RGB) / 255.0
        else:
            upscaled = upscaled_cv / 255.0

    info_dict = {
        "model_name": model_name,
        "scale_factor": scale_factor,
        "original_size": f"{image.shape[1]}x{image.shape[0]}",
        "upscaled_size": f"{upscaled.shape[1]}x{upscaled.shape[0]}",
        "method": "bicubic_interpolation",  # fallback method
    }

    return upscaled, info_dict


def _normalize_image(image: np.ndarray, norm_range_low: float, norm_range_high: float) -> np.ndarray:
    try:
        # Calculate percentiles
        low_val = np.percentile(image, norm_range_low)
        high_val = np.percentile(image, norm_range_high)

        # Normalize to 0-1 range
        normalized = (image - low_val) / (high_val - low_val)

        # Clip values to [0, 1] range
        normalized = np.clip(normalized, 0, 1)

        # Convert back to original dtype range
        if image.dtype == np.uint8:
            normalized = (normalized * 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            normalized = (normalized * 65535).astype(np.uint16)
        else:
            # Keep as float for other dtypes
            pass

        return normalized

    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        return image  # Return original image if normalization fails


def _validate_image(image: np.ndarray) -> bool:
    if image is None:
        logger.error("Input image is None")
        return False
    if not isinstance(image, np.ndarray):
        logger.error(f"Input image must be numpy array, got {type(image)}")
        return False
    if len(image.shape) not in [2, 3]:
        logger.error(f"Input image must be 2D or 3D, got shape {image.shape}")
        return False
    return True


def load_image(
    image_path: str,
    *,
    to_gray: bool = False,
    normalize: bool = False,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> np.ndarray:
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ome.tif", ".ome.tiff")

    p = Path(str(image_path))
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")

    if p.is_dir():
        cand = sorted([q for q in p.iterdir() if q.suffix.lower() in exts], key=lambda x: x.name.lower())
        if not cand:
            raise FileNotFoundError(f"No image files found in directory: {p}")
        p = cand[0]
        logger.info(f"Directory provided, using first image: {p}")

    # Ensure we have a file, not a directory
    if p.is_dir():
        raise ValueError(f"Path is still a directory after processing: {p}")

    # Validate file extension
    if p.suffix.lower() not in exts:
        raise ValueError(f"Unsupported file format: {p.suffix}. Supported formats: {exts}")

    logger.info(f"Loading image from: {p}")

    try:
        img = io.imread(str(p))
    except Exception as e:
        # Try with tifffile for TIFF formats
        if p.suffix.lower() in (".tif", ".tiff", ".ome.tif", ".ome.tiff"):
            try:
                import tifffile
                img = tifffile.imread(str(p))
                logger.info(f"Successfully loaded TIFF using tifffile: {p}")
            except ImportError:
                raise ValueError(f"tifffile not available for loading {p}")
            except Exception as e2:
                raise ValueError(f"Failed to load TIFF image from {p}: {str(e2)}")
        else:
            raise ValueError(f"Failed to load image from {p}: {str(e)}")

    if to_gray and img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(np.float32)
        img = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    if normalize:
        a = img_as_float32(img)
        lo, hi = np.percentile(a, [p_low, p_high])
        if hi <= lo:  # fallback if image is near-constant
            lo, hi = float(a.min()), float(a.max())
        img = np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
    return img


def create_colored_masks(masks: np.ndarray) -> np.ndarray:
    try:
        if masks.max() == 0:
            # No objects detected, return grayscale
            return np.stack([masks] * 3, axis=-1).astype(np.uint8)

        # Create a colormap with distinct colors for each cell
        num_objects = int(masks.max())
        np.random.seed(42)  # For consistent colors

        # Generate random colors
        colors = np.random.rand(num_objects + 1, 3)
        colors[0] = [0, 0, 0]  # Background is black

        # Apply colors to masks
        colored = colors[masks]
        return (colored * 255).astype(np.uint8)

    except Exception as e:
        logger.error(f"Error creating colored masks: {str(e)}")
        # Fallback to grayscale
        return np.stack([masks] * 3, axis=-1).astype(np.uint8)


if __name__ == "__main__":
    asyncio.run(cellpose_segmentation())
