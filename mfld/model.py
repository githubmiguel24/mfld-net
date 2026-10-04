"""MFLD-net (Mobile Fish Landmark Detection network), Fig. 2 of the paper.

    image (3x224x224)
      -> Patch embedding   Conv2d(3, 256, kernel=4, stride=4)            -> 256 x 56 x 56
      -> 8 x ConvBlock     (isometric: size/shape never change)
      -> HeatMap Conv      Conv2d(256, K, 1)                             -> K x 56 x 56
            |-> SoftMax (spatial)       -> predicted heatmaps (confidence maps)
            '-> Soft-Argmax             -> predicted (x, y) coordinates

ConvBlock (bottom panel of Fig. 2):

    x -> DepthwiseConv -> GELU -> BatchNorm -> (+ x)   residual / element-wise add
      -> SpatialDropout(0.2)
      -> PointwiseConv -> GELU -> BatchNorm

The boxes labelled "Linear" in Fig. 2 are the GELU activations that the text
says follow every convolution ("Each of the convolutions is followed by GELU
activation and BatchNorm").
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MFLDOutput(NamedTuple):
    heatmaps: torch.Tensor   # (B, K, H, W) spatial-softmax probabilities (sum to 1 per keypoint)
    coords: torch.Tensor     # (B, K, 2) soft-argmax (x, y), normalised to [0, 1]
    logits: torch.Tensor     # (B, K, H, W) raw output of the HeatMap Conv


class ConvBlock(nn.Module):
    """Depthwise (spatial mixing) + spatial dropout + pointwise (channel mixing)."""

    def __init__(self, dim: int, kernel_size: int = 9, dropout: float = 0.2):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd so the block keeps the resolution")
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.dropout = nn.Dropout2d(dropout)       # spatial dropout: drops whole channels
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.depthwise(x)                  # element-wise add (residual connection)
        x = self.dropout(x)
        return self.pointwise(x)


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Differentiable expectation of the (x, y) position under each heatmap.

    ``heatmaps`` are probabilities (B, K, H, W). Pixel ``i`` has its centre at
    ``(i + 0.5) / size`` so that the result lives in the same normalised [0, 1]
    frame as the ground-truth coordinates (see ``mfld.heatmaps``).
    """
    b, k, h, w = heatmaps.shape
    xs = (torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) / w
    ys = (torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) / h
    x = (heatmaps.sum(dim=2) * xs).sum(dim=-1)     # marginal over rows -> (B, K, W)
    y = (heatmaps.sum(dim=3) * ys).sum(dim=-1)     # marginal over cols -> (B, K, H)
    return torch.stack([x, y], dim=-1)


class MFLDNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, **kwargs):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig(**kwargs)
        c = self.cfg
        if c.img_size % c.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(c.in_chans, c.dim, kernel_size=c.patch_size, stride=c.patch_size)
        self.blocks = nn.Sequential(*[ConvBlock(c.dim, c.kernel_size, c.dropout) for _ in range(c.depth)])
        self.heatmap_conv = nn.Conv2d(c.dim, c.num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> MFLDOutput:
        feats = self.blocks(self.patch_embed(x))
        logits = self.heatmap_conv(feats)
        b, k, h, w = logits.shape
        heatmaps = F.softmax(logits.reshape(b, k, h * w), dim=-1).reshape(b, k, h, w)
        coords = soft_argmax(heatmaps)
        return MFLDOutput(heatmaps, coords, logits)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
