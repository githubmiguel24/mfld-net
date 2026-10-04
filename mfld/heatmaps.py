"""Ground-truth heatmap rendering (numpy only).

Convention used everywhere in this project: coordinates are *normalised* to
[0, 1] over the (resized) image and pixel ``i`` of an S-pixel axis has its centre
at ``(i + 0.5) / S``. ``mfld.model.soft_argmax`` uses the same convention, so a
perfect heatmap decodes back to exactly the coordinate it was rendered from.
"""
from __future__ import annotations

import numpy as np


def render_gaussian_heatmaps(coords: np.ndarray, size: int = 56, sigma: float = 1.5,
                             normalize: bool = True) -> np.ndarray:
    """One symmetric 2-D Gaussian per keypoint -> (K, size, size) float32.

    With ``normalize=True`` every map sums to 1 (a probability distribution, as
    needed by the Jensen-Shannon divergence). Out-of-frame points are clipped to
    the image border.
    """
    coords = np.clip(np.asarray(coords, dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
    centres = coords * size - 0.5                       # in pixel-index units
    idx = np.arange(size, dtype=np.float64)
    dx = idx[None, None, :] - centres[:, 0][:, None, None]     # (K, 1, S)
    dy = idx[None, :, None] - centres[:, 1][:, None, None]     # (K, S, 1)
    hm = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))
    if normalize:
        hm /= hm.sum(axis=(1, 2), keepdims=True)
    return hm.astype(np.float32)


def decode_expectation(heatmaps: np.ndarray) -> np.ndarray:
    """Numpy soft-argmax of probability maps (K, H, W) -> (K, 2) normalised (x, y)."""
    k, h, w = heatmaps.shape
    xs = (np.arange(w) + 0.5) / w
    ys = (np.arange(h) + 0.5) / h
    x = (heatmaps.sum(axis=1) * xs).sum(axis=-1)
    y = (heatmaps.sum(axis=2) * ys).sum(axis=-1)
    return np.stack([x, y], axis=-1)
