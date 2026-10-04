"""Multitask loss (Fig. 3 and Eqs. 1-3 of the paper).

* heatmap branch   : Jensen-Shannon divergence (Eq. 3, built on the KLD of Eq. 2)
                     between predicted and ground-truth heatmaps
* coordinate branch: Euclidean distance (Eq. 1) between predicted and
                     ground-truth keypoint coordinates
* optimisation loss: the average of the two losses
"""
from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """KLD(P||Q) = sum_i p_i log(p_i / q_i) over the last dimension (Eq. 2)."""
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def jensen_shannon_distance(p: torch.Tensor, q: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """JSD_M(P||Q) = sqrt((KLD(p||m) + KLD(q||m)) / 2), m = (p + q) / 2  (Eq. 3).

    ``p`` and ``q`` are (B, K, H, W) probability maps; the distribution is taken
    over the H*W pixels of every keypoint heatmap. Returns (B, K).
    """
    p = p.flatten(start_dim=2)
    q = q.flatten(start_dim=2)
    m = 0.5 * (p + q)
    jsd = 0.5 * (kl_divergence(p, m, eps) + kl_divergence(q, m, eps))
    return torch.sqrt(jsd.clamp_min(0.0) + eps)      # eps keeps the sqrt gradient finite at 0


def euclidean_distance(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """d(g, p) = sqrt(sum (g - p)^2) per keypoint (Eq. 1). (B, K, 2) -> (B, K)."""
    return torch.sqrt(((pred - target) ** 2).sum(dim=-1) + eps)


class MultiTaskLoss(nn.Module):
    """Average of the heatmap JSD loss and the coordinate Euclidean loss."""

    def forward(self, output, target_heatmaps: torch.Tensor, target_coords: torch.Tensor) -> dict:
        heat = jensen_shannon_distance(output.heatmaps, target_heatmaps).mean()
        coord = euclidean_distance(output.coords, target_coords).mean()
        return {"total": 0.5 * (heat + coord), "heatmap": heat, "coords": coord}
