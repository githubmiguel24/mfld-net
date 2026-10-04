"""Landmarks -> the four body measurements of the paper (Fig. 9, numpy only):
total length, standard length, body depth and head length.

The default layout is the 13-point betta layout of the dataset export. The paper's
16-point barramundi layout (order not published - an assumption, also used by the
synthetic generator) is kept for ``--num-keypoints 16``. The layout is picked from the
keypoint count; for any other annotation order pass a JSON spec (see ``load_spec``).
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Union

import numpy as np

# --- betta layout (13 points, the order of the dataset export) - the project default -------------
BETTA_KEYPOINT_NAMES: List[str] = [
    "snout_tip",                # 0
    "eye_center",               # 1
    "dorsal_fin_base_anterior", # 2
    "dorsal_fin_base_posterior",# 3
    "caudal_peduncle_top",      # 4
    "dorsal_fin_tip",           # 5
    "caudal_peduncle_bottom",   # 6
    "caudal_fin_tip_upper",     # 7
    "caudal_fin_tip_lower",     # 8
    "caudal_fin_center",        # 9
    "anal_fin_base_anterior",   # 10
    "anal_fin_base_posterior",  # 11
    "anal_fin_tip",             # 12
]

# --- barramundi layout (16 points, an assumption - the paper does not publish its order) ----------
BARRAMUNDI_KEYPOINT_NAMES: List[str] = [
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

BETTA_SPEC: Spec = {
    "total_length": (0, 9),             # snout tip -> centre of the caudal fin
    "standard_length": (0, (4, 6)),     # snout tip -> middle of the caudal peduncle
    "body_depth": (2, 10),              # anterior dorsal base -> anterior anal base
    "head_length": (0, 1),              # snout tip -> eye centre
}
BARRAMUNDI_SPEC: Spec = {
    "total_length": (0, (10, 11)),      # snout tip -> end of the tail fin
    "standard_length": (0, 9),          # snout tip -> end of the vertebral column
    "body_depth": (6, 15),              # dorsal <-> ventral body surface
    "head_length": (0, 3),              # snout tip -> posterior margin of operculum
}

LAYOUTS = {13: (BETTA_KEYPOINT_NAMES, BETTA_SPEC), 16: (BARRAMUNDI_KEYPOINT_NAMES, BARRAMUNDI_SPEC)}
KEYPOINT_NAMES = BETTA_KEYPOINT_NAMES   # project default
DEFAULT_SPEC = BETTA_SPEC
TRAITS = tuple(DEFAULT_SPEC)


def keypoint_names(num_keypoints: int) -> List[str]:
    """Names for the built-in layout with this many keypoints ([] for a custom layout)."""
    return list(LAYOUTS[num_keypoints][0]) if num_keypoints in LAYOUTS else []


def default_spec(num_keypoints: int) -> Spec:
    """Built-in measurement spec for this keypoint count (betta for anything else)."""
    return LAYOUTS.get(num_keypoints, LAYOUTS[13])[1]


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
    kp = np.asarray(keypoints_px, float)
    spec = spec or default_spec(len(kp))
    out = {}
    for name, (a, b) in spec.items():
        out[name] = float(np.linalg.norm(_point(kp, a) - _point(kp, b)) * mm_per_pixel)
    return out


def measure_batch(keypoints_px: np.ndarray, spec: Spec | None = None, mm_per_pixel=1.0) -> Dict[str, np.ndarray]:
    """keypoints_px: (N, K, 2); mm_per_pixel: scalar or (N,)."""
    kps = np.asarray(keypoints_px, float)
    spec = spec or default_spec(kps.shape[1])
    mm = np.broadcast_to(np.asarray(mm_per_pixel, float), (len(kps),))
    rows = [measure(k, spec, m) for k, m in zip(kps, mm)]
    return {name: np.array([r[name] for r in rows]) for name in spec}
