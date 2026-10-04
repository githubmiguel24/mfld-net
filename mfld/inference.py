"""Loading checkpoints and predicting landmarks / measurements on new images."""
from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch

from .config import ModelConfig
from .data import normalise_image
from .model import MFLDNet
from .morphometry import KEYPOINT_NAMES, Spec, measure


def load_model(ckpt_path: str, device: torch.device | str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = MFLDNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_images(model: MFLDNet, images_rgb: Sequence[np.ndarray], device: torch.device | str = "cpu",
                   batch_size: int = 32, spec: Spec | None = None, mm_per_pixel: float = 1.0) -> List[Dict]:
    """images_rgb: list of HxWx3 uint8 RGB arrays (any size).

    Returns, per image: ``coords_norm`` (K,2) in [0,1], ``keypoints_px`` (K,2) in the original
    image's pixels, ``confidence`` (K,) peak heatmap value per keypoint, and ``measurements``
    (total/standard length, body depth, head length) in mm (or px if mm_per_pixel == 1)."""
    model.eval()
    s = model.cfg.img_size
    results = []
    for start in range(0, len(images_rgb), batch_size):
        chunk = images_rgb[start:start + batch_size]
        batch = torch.stack([normalise_image(cv2.resize(im, (s, s), interpolation=cv2.INTER_LINEAR)) for im in chunk])
        out = model(batch.to(device))
        coords = out.coords.cpu().numpy()
        conf = out.heatmaps.flatten(2).amax(-1).cpu().numpy()
        for im, c, p in zip(chunk, coords, conf):
            h, w = im.shape[:2]
            px = c * np.array([w, h], np.float32)
            results.append({"coords_norm": c, "keypoints_px": px, "confidence": p,
                            "measurements": measure(px, spec, mm_per_pixel)})
    return results


def draw_keypoints(image_rgb: np.ndarray, keypoints_px: np.ndarray, names: Sequence[str] | None = KEYPOINT_NAMES,
                   label: bool = True) -> np.ndarray:
    """Overlay of the predicted landmarks (returns a copy)."""
    out = image_rgb.copy()
    h, w = out.shape[:2]
    r = max(2, int(round(min(h, w) / 120)))
    for i, (x, y) in enumerate(keypoints_px):
        colour = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[(i * 11) % 180, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])
        cv2.circle(out, (int(round(x)), int(round(y))), r, colour, -1, cv2.LINE_AA)
        if label:
            cv2.putText(out, str(i), (int(x) + r + 1, int(y) - r), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, min(h, w) / 900), (255, 255, 255), 1, cv2.LINE_AA)
    return out
