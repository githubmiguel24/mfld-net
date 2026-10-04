"""Hyper-parameters. Values marked [paper] are stated in the paper, values marked
[assumed] are not specified there and are reasonable defaults you can override."""
from __future__ import annotations

from dataclasses import dataclass, asdict

# ImageNet statistics for input normalisation [assumed - the paper does not say]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    img_size: int = 224          # [paper] 224 x 224 input
    patch_size: int = 4          # [paper] patch embedding: kernel 4, stride 4 -> 56 x 56 maps
    dim: int = 256               # [paper] 256 output channels of the patch embedding
    depth: int = 8               # [paper] 8 ConvBlocks
    kernel_size: int = 9         # [paper] "increasing the size of the kernel to 9"
    dropout: float = 0.2         # [paper] spatial dropout rate 0.2
    num_keypoints: int = 13      # betta dataset: 13 keypoints per fish ([paper], barramundi: 16)
    in_chans: int = 3            # [paper] RGB

    @property
    def heatmap_size(self) -> int:
        return self.img_size // self.patch_size   # 56

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    epochs: int = 50             # [paper] ~50 epochs
    batch_size: int = 64         # [paper]
    lr: float = 1e-3             # [paper]
    betas: tuple = (0.9, 0.999)  # [paper] Adam
    eps: float = 1e-8            # [paper] Adam
    lr_step: int = 30            # [paper] decay every 30 epochs
    lr_gamma: float = 0.1        # [paper] gamma = 0.1
    patience: int = 50           # [paper] converged when val loss stops improving after 50 epochs
    labelled_frac: float = 0.4   # [paper] 40 % of the data is used for training + validation
    train_frac: float = 0.7      # [paper] 70 % of that 40 % for training, 30 % for validation
    sigma: float = 1.5           # [assumed] std-dev (in heatmap pixels) of the target Gaussians
    seed: int = 0
    vflip: bool = True           # [paper] vertical flip p=.5; set False for subjects that are always upright
