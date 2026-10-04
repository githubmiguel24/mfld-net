"""Dataset, augmentation (Sect. 'Data augmentation' of the paper) and annotation loading.

Annotation file (JSON):

    {"images": [
        {"file": "images/a.jpg",                 # relative to --data-root
         "keypoints": [[x, y], ... 16 points],    # pixels in the ORIGINAL image
         "scale": 321.5,                          # optional: sqrt(fish segment area), original px
         "mm_per_pixel": 0.9,                     # optional: calibration for mm measurements
         "manual_mm": {"total_length": 412.0, ...}}   # optional: manual reference measurements
    ]}
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD, ModelConfig, TrainConfig
from .heatmaps import render_gaussian_heatmaps
from .splits import make_splits


def load_annotations(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    records = data["images"] if isinstance(data, dict) else data
    for r in records:
        r["keypoints"] = np.asarray(r["keypoints"], dtype=np.float32).reshape(-1, 2)
    return records


def load_annotation_meta(path: str) -> dict:
    """Top-level fields of the annotation file other than the records (keypoint_names, preprocessing, ...)."""
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "images"} if isinstance(data, dict) else {}


def build_train_transform(vflip: bool = True):
    """The six augmentations listed in the paper, applied to the training set only:

    (1) horizontal flip p=.5   (2) vertical flip p=.5
    (3) shift (limit 0.0625) + scale (limit 0.20) p=.5   (4) rotation limit 20 deg p=.5
    (5) blur (kernel 3) p=.3   (6) RGB shift (25, 25, 25) p=.3

    ``vflip=False`` drops (2) - an option for subjects that are always upright (a flipped fish also swaps
    the meaning of "upper"/"lower" keypoints while their indices stay put).

    NB: the paper prints the shift/scale limits with a degree sign; they are fractions
    (as in Albumentations' ShiftScaleRotate). Keypoints follow every geometric change.
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            *([A.VerticalFlip(p=0.5)] if vflip else []),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.0625, 0.0625), rotate=0, p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Blur(blur_limit=(3, 3), p=0.3),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def normalise_image(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def keypoint_scale(kp: np.ndarray) -> float:
    """Fallback object scale when the annotation has none: sqrt of the keypoints' bounding-box area."""
    w, h = np.ptp(kp[:, 0]), np.ptp(kp[:, 1])
    return float(np.sqrt(max(w * h, 1.0)))


class FishLandmarkDataset(Dataset):
    def __init__(self, records: List[dict], root: str, model_cfg: ModelConfig | None = None,
                 augment: bool = False, sigma: float = 1.5, vflip: bool = True):
        self.records = records
        self.root = root
        self.cfg = model_cfg or ModelConfig()
        self.sigma = sigma
        self.transform = build_train_transform(vflip) if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        rec = self.records[i]
        bgr = cv2.imread(os.path.join(self.root, rec["file"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(os.path.join(self.root, rec["file"]))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = self.cfg.img_size
        sx, sy = s / w, s / h
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        kp = rec["keypoints"] * np.array([sx, sy], np.float32)
        scale = float(rec.get("scale") or keypoint_scale(rec["keypoints"])) * float(np.sqrt(sx * sy))

        if self.transform is not None:
            out = self.transform(image=img, keypoints=kp.tolist())
            img = out["image"]
            kp = np.asarray([k[:2] for k in out["keypoints"]], np.float32).reshape(-1, 2)

        kp_norm = np.clip(kp / s, 0.0, 1.0).astype(np.float32)
        hm = render_gaussian_heatmaps(kp_norm, self.cfg.heatmap_size, self.sigma)
        return {
            "image": normalise_image(img),
            "heatmaps": torch.from_numpy(hm),
            "coords": torch.from_numpy(kp_norm),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "orig_size": torch.tensor([w, h], dtype=torch.float32),
            "index": torch.tensor(i),
        }


def build_datasets(root: str, annotations: str, model_cfg: ModelConfig, train_cfg: TrainConfig,
                   splits: Optional[Dict[str, List[int]]] = None):
    """Returns (train_ds, val_ds, test_ds, splits). Only the training set is augmented."""
    records = load_annotations(annotations)
    splits = splits or make_splits(len(records), train_cfg.labelled_frac, train_cfg.train_frac, train_cfg.seed)
    pick = lambda ids: [records[j] for j in ids]
    train = FishLandmarkDataset(pick(splits["train"]), root, model_cfg, augment=True, sigma=train_cfg.sigma,
                               vflip=train_cfg.vflip)
    val = FishLandmarkDataset(pick(splits["val"]), root, model_cfg, augment=False, sigma=train_cfg.sigma)
    test = FishLandmarkDataset(pick(splits["test"]), root, model_cfg, augment=False, sigma=train_cfg.sigma)
    return train, val, test, splits
