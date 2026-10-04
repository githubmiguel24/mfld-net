"""Landmarks -> the four body measurements of the paper (Fig. 9, numpy only):
total length, standard length, body depth and head length.

IMPORTANT: the paper annotates 16 keypoints (Fig. 5) but does not publish which
index is which. ``KEYPOINT_NAMES`` / ``DEFAULT_SPEC`` below define a concrete
16-point layout that this project (and its synthetic generator) uses. If your
own annotation order differs, pass a JSON spec (see ``load_spec``) - nothing
else in the code depends on the layout.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Union

import numpy as np

KEYPOINT_NAMES: List[str] = [
    "snout_tip",              # 0
    "forehead",               # 1
    "operculum_top",          # 2
    "operculum_posterior",    # 3  posterior margin of the operculum (head length end)
    "operculum_bottom",       # 4
    "chin",                   # 5
    "dorsal_fin_origin",      # 6
    "dorsal_fin_end",         # 7
    "dorsal_peduncle",        # 8
    "vertebral_column_end",   # 9  end of the vertebral column (standard length end)
    "tail_tip_top",           # 10
    "tail_tip_bottom",        # 11
    "ventral_peduncle",       # 12
    "anal_fin_end",           # 13
    "anal_fin_origin",        # 14
    "pelvic_fin_origin",      # 15
]

# Every measurement is the distance between two *points*; a point is one keypoint
# index or a list of indices whose mean is used (e.g. the middle of the tail fin).
PointSpec = Union[int, Sequence[int]]
Spec = Dict[str, Sequence[PointSpec]]

DEFAULT_SPEC: Spec = {
    "total_length": (0, (10, 11)),      # snout tip -> end of the tail fin
    "standard_length": (0, 9),          # snout tip -> end of the vertebral column
    "body_depth": (6, 15),              # dorsal <-> ventral body surface
    "head_length": (0, 3),              # snout tip -> posterior margin of operculum
}
TRAITS = tuple(DEFAULT_SPEC)


def load_spec(path: str) -> Spec:
    """JSON like {"total_length": [0, [10, 11]], "head_length": [0, 3], ...}."""
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


def _point(kp: np.ndarray, p: PointSpec) -> np.ndarray:
    if isinstance(p, (int, np.integer)):
        return kp[int(p)]
    return kp[list(p)].mean(axis=0)


def measure(keypoints_px: np.ndarray, spec: Spec | None = None, mm_per_pixel: float = 1.0) -> Dict[str, float]:
    """keypoints_px: (K, 2) in *original image pixels*. Returns lengths in mm
    (or pixels when ``mm_per_pixel`` is 1)."""
    spec = spec or DEFAULT_SPEC
    kp = np.asarray(keypoints_px, float)
    out = {}
    for name, (a, b) in spec.items():
        out[name] = float(np.linalg.norm(_point(kp, a) - _point(kp, b)) * mm_per_pixel)
    return out


def measure_batch(keypoints_px: np.ndarray, spec: Spec | None = None, mm_per_pixel=1.0) -> Dict[str, np.ndarray]:
    """keypoints_px: (N, K, 2); mm_per_pixel: scalar or (N,)."""
    kps = np.asarray(keypoints_px, float)
    mm = np.broadcast_to(np.asarray(mm_per_pixel, float), (len(kps),))
    rows = [measure(k, spec, m) for k, m in zip(kps, mm)]
    return {name: np.array([r[name] for r in rows]) for name in (spec or DEFAULT_SPEC)}
