"""Synthetic 'fish on a conveyor belt' images with exact 16-point annotations.

The real barramundi dataset of the paper is not public, so this generator lets you
exercise the whole pipeline (train / evaluate / predict) end-to-end. The keypoint
layout matches ``mfld.morphometry.KEYPOINT_NAMES``. numpy + OpenCV only.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np

from .morphometry import measure

MM_PER_PIXEL = 0.9          # pretend calibration of the camera


def _half_depth(t, hmax):
    t = np.clip(t, 0.0, 1.0)
    return hmax * (0.16 + 0.84 * np.sin(np.pi * t ** 0.75) ** 0.8)


def _local_fish(lb: float, hmax: float, lt: float, th: float):
    """Outline polygon and 16 keypoints in the fish's local frame (snout at origin, x to the tail)."""
    t = np.linspace(0, 1, 60)
    top = np.stack([t * lb, -_half_depth(t, hmax)], 1)
    bottom = np.stack([t * lb, _half_depth(t, hmax)], 1)[::-1]
    p = _half_depth(1.0, hmax)
    tail = np.array([[lb, -p], [lb + lt, -th], [lb + 0.82 * lt, 0.0], [lb + lt, th], [lb, p]])
    body = np.concatenate([top, tail, bottom], 0)
    hd = lambda u: _half_depth(u, hmax)
    kp = np.array([
        [0.0, 0.0],                          # 0 snout tip
        [0.10 * lb, -hd(0.10)],              # 1 forehead
        [0.22 * lb, -hd(0.22)],              # 2 operculum top
        [0.24 * lb, 0.0],                    # 3 operculum posterior
        [0.20 * lb, 0.95 * hd(0.20)],        # 4 operculum bottom
        [0.08 * lb, hd(0.08)],               # 5 chin
        [0.30 * lb, -hd(0.30)],              # 6 dorsal fin origin
        [0.72 * lb, -hd(0.72)],              # 7 dorsal fin end
        [0.95 * lb, -hd(0.95)],              # 8 dorsal peduncle
        [lb, 0.0],                           # 9 vertebral column end
        [lb + lt, -th],                      # 10 tail tip top
        [lb + lt, th],                       # 11 tail tip bottom
        [0.95 * lb, hd(0.95)],               # 12 ventral peduncle
        [0.78 * lb, hd(0.78)],               # 13 anal fin end
        [0.62 * lb, hd(0.62)],               # 14 anal fin origin
        [0.30 * lb, hd(0.30)],               # 15 pelvic fin origin
    ])
    return body, kp


def generate_sample(rng: np.random.Generator, size: Tuple[int, int] = (512, 288)):
    """Returns (image RGB uint8, keypoints (16,2) in pixels, scale = sqrt(fish area))."""
    w, h = size
    total = rng.uniform(0.62, 0.82) * w
    lb = total / 1.2
    lt = 0.2 * lb
    hmax = lb * rng.uniform(0.11, 0.15)
    th = hmax * rng.uniform(0.7, 0.95)
    body, kp = _local_fish(lb, hmax, lt, th)

    ang = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    centre_local = np.array([(lb + lt) / 2, 0.0])
    target = np.array([w / 2 + rng.uniform(-0.04, 0.04) * w, h / 2 + rng.uniform(-0.08, 0.08) * h])
    xf = lambda pts: (pts - centre_local) @ rot.T + target
    body_px, kp_px = xf(body), xf(kp)

    # background: conveyor belt with a lighting gradient
    base = rng.uniform(170, 225)
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1.0 + rng.uniform(-0.25, 0.25) * (xx / w - 0.5) + rng.uniform(-0.15, 0.15) * (yy / h - 0.5)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (base * grad)[..., None] * np.array([1.0, 1.0, 0.97], np.float32)
    img[: int(0.1 * h)] *= 0.55
    img[-int(0.1 * h):] *= 0.55

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(body_px).astype(np.int32)], 1)
    tone = rng.uniform(70, 125)
    tint = np.array([1.0, 0.95, 0.85]) * tone
    shade = (0.75 + 0.5 * (yy - target[1]) / (hmax * 2 + 1e-6)).clip(0.6, 1.2)      # darker back
    fish = (tint[None, None, :] * shade[..., None]).astype(np.float32)
    img = np.where(mask[..., None] > 0, fish, img)
    img *= rng.uniform(0.75, 1.15)
    img += rng.normal(0, 3.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    scale = float(np.sqrt(mask.sum()))
    return img, kp_px.astype(np.float32), scale


def generate_dataset(out_dir: str, n: int = 200, size: Tuple[int, int] = (512, 288), seed: int = 0) -> str:
    """Writes ``out_dir/images/*.png`` and ``out_dir/annotations.json``; returns the json path."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    records = []
    for i in range(n):
        img, kp, scale = generate_sample(rng, size)
        rel = f"images/fish_{i:05d}.png"
        cv2.imwrite(os.path.join(out_dir, rel), img[..., ::-1])
        records.append({
            "file": rel,
            "keypoints": kp.round(2).tolist(),
            "scale": round(scale, 2),
            "mm_per_pixel": MM_PER_PIXEL,
            "manual_mm": {k: round(v, 2) for k, v in measure(kp, mm_per_pixel=MM_PER_PIXEL).items()},
        })
    path = os.path.join(out_dir, "annotations.json")
    with open(path, "w") as f:
        json.dump({"num_keypoints": 16, "images": records}, f)
    return path
