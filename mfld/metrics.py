"""Evaluation metrics (numpy only): OKS-based AP / AR (Eq. 4, COCO style) and the
MAD / SDD morphometric agreement measures used in Table 3 of the paper."""
from __future__ import annotations

import numpy as np

OKS_THRESHOLDS = np.arange(0.50, 0.96, 0.05)      # .50 : .05 : .95


def euclidean(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-keypoint Euclidean distance (Eq. 1). (..., K, 2) -> (..., K)."""
    return np.sqrt(((np.asarray(pred, float) - np.asarray(gt, float)) ** 2).sum(-1))


def object_keypoint_similarity(pred: np.ndarray, gt: np.ndarray, scale, kappa=0.05,
                               visibility: np.ndarray | None = None) -> np.ndarray:
    """Eq. 4:  OKS = sum_i exp(-d_i^2 / (2 s^2 k_i^2)) * delta(v_i > 0) / sum_i delta(v_i > 0).

    pred, gt   : (N, K, 2) or (K, 2) in the same pixel units as ``scale``
    scale      : object scale s (sqrt of the object's segment area), scalar or (N,)
    kappa      : per-keypoint falloff constant k_i, scalar or (K,). The paper does not
                 list its values; 0.05 is used for every keypoint by default.
    visibility : (N, K) ground-truth visibility flags (default: all keypoints visible)
    """
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    single = pred.ndim == 2
    if single:
        pred, gt = pred[None], gt[None]
    n, k, _ = pred.shape
    s = np.broadcast_to(np.asarray(scale, float), (n,))
    kap = np.broadcast_to(np.asarray(kappa, float), (k,))
    d2 = ((pred - gt) ** 2).sum(-1)                                    # (N, K)
    sim = np.exp(-d2 / (2.0 * (s[:, None] ** 2) * (kap[None, :] ** 2) + 1e-12))
    vis = np.ones((n, k), bool) if visibility is None else (np.asarray(visibility) > 0)
    oks = (sim * vis).sum(1) / np.maximum(vis.sum(1), 1)
    return oks[0] if single else oks


def _ap_ar_at_threshold(oks: np.ndarray, conf: np.ndarray, thr: float):
    """COCO-style AP (101-point interpolation) and AR for one threshold.

    Every image holds exactly one fish, so each prediction is a true positive when
    its OKS >= thr; predictions are ranked by confidence.
    """
    n = len(oks)
    order = np.argsort(-conf, kind="stable")
    tp = (oks[order] >= thr).astype(float)
    tp_c, fp_c = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = tp_c / n
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)
    for i in range(len(precision) - 2, -1, -1):                      # monotone precision envelope
        precision[i] = max(precision[i], precision[i + 1])
    rec_pts = np.linspace(0, 1, 101)
    inds = np.searchsorted(recall, rec_pts, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in inds])
    return float(q.mean()), float(recall[-1])


def average_precision_recall(oks: np.ndarray, conf: np.ndarray | None = None) -> dict:
    """AP, AP50, AP75, AR, AR50, AR75 from per-image OKS values."""
    oks = np.asarray(oks, float)
    conf = np.zeros_like(oks) if conf is None else np.asarray(conf, float)
    aps, ars = [], []
    for t in OKS_THRESHOLDS:
        ap, ar = _ap_ar_at_threshold(oks, conf, float(t))
        aps.append(ap)
        ars.append(ar)
    aps, ars = np.array(aps), np.array(ars)
    i50, i75 = 0, 5                                                    # thresholds .50 and .75
    return {"AP": aps.mean(), "AP.50": aps[i50], "AP.75": aps[i75],
            "AR": ars.mean(), "AR.50": ars[i50], "AR.75": ars[i75]}


def mean_absolute_difference(x, y) -> float:
    """MAD = 1/n sum |x_i - y_i|  (accuracy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.mean(np.abs(x - y)))


def standard_deviation_of_difference(x, y) -> float:
    """SDD = sqrt( sum (x_i - y_i - mean(x) + mean(y))^2 / (n - 1) )  (precision)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return float("nan")
    return float(np.sqrt(((x - y - x.mean() + y.mean()) ** 2).sum() / (n - 1)))
