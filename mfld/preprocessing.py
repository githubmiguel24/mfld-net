"""Plain-language description of the exact input/output conventions (written as preprocessing.json)."""
from __future__ import annotations

import json
import os
from typing import Optional

from .config import IMAGENET_MEAN, IMAGENET_STD, ModelConfig


def describe_preprocessing(cfg: ModelConfig, ann_meta: Optional[dict] = None, keypoint_names=None) -> dict:
    ann_meta = ann_meta or {}
    crop = ann_meta.get("input") == "fish_box_crop"
    s, hm = cfg.img_size, cfg.heatmap_size
    return {
        "model_input": (
            f"A CROP of the fish box from the original photo, enlarged by a margin of {ann_meta.get('margin')} of the box "
            "width/height on every side (and always covering all 13 keypoints), clipped to the photo. One sample per fish "
            "annotation, so a photo with several fish gives several samples. The crop is made by prepare_betta.py from "
            "annotation 'fish_box' (fallback 'bbox'); the crop window is stored per record as 'crop' = [x0, y0, x1, y1] "
            "in original-photo pixels." if crop else "The WHOLE image (no cropping)."),
        "crop_padding": "none - no padding is added; the margin is only taken from existing image content" if crop else "none",
        "resize": f"SQUASH: the (crop) image is resized to {s}x{s} with cv2.INTER_LINEAR, ignoring aspect ratio; "
                  "no letterboxing/padding.",
        "colour": "BGR file read by OpenCV, converted to RGB",
        "normalisation": f"pixels scaled to 0-1, then (x - mean) / std with ImageNet mean {list(IMAGENET_MEAN)} and "
                         f"std {list(IMAGENET_STD)} per RGB channel",
        "heatmaps": f"{hm}x{hm}; the network outputs a spatial SOFTMAX per keypoint (each map sums to 1)",
        "decoding": (
            "Coordinates come from SOFT-ARGMAX (expectation of x and y under each softmax heatmap), not argmax. Pixel i of "
            f"an axis of S={hm} has its centre at (i + 0.5) / S, so the result is in normalised [0, 1] coordinates of the "
            f"{s}x{s} network image. No further offset or refinement."),
        "mapping_back_to_original_pixels": (
            "x_crop = x_norm * crop_width, y_crop = y_norm * crop_height (the squash is undone by scaling with the crop "
            "size; the same scale-only convention as the training labels, i.e. no half-pixel shift), then "
            "x_orig = x_crop + x0, y_orig = y_crop + y0 with (x0, y0) = record['crop'][:2]. "
            "For whole-image models x0 = y0 = 0 and the crop size is the image size." if crop else
            "x_orig = x_norm * image_width, y_orig = y_norm * image_height."),
        "confidence": "peak value of each keypoint's softmax heatmap (a probability; small, because the map sums to 1)",
        "keypoint_order": list(keypoint_names) if keypoint_names else "as in the annotation file",
        "training_augmentation_only": "horizontal/vertical flip, shift+scale, rotation, blur, RGB shift (never on val/test)",
    }


def write_preprocessing(path: str, cfg: ModelConfig, ann_meta: Optional[dict] = None, keypoint_names=None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(describe_preprocessing(cfg, ann_meta, keypoint_names), f, indent=2, ensure_ascii=False)
