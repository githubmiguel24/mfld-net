"""Dataset splitting exactly as in Fig. 1 of the paper (numpy only):

    all images --random--> 40 % labelled pool  --random--> 70 % train / 30 % validation
                       '-> 60 % test set (never used for training or validation)
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np


def make_splits(n: int, labelled_frac: float = 0.4, train_frac: float = 0.7, seed: int = 0) -> Dict[str, List[int]]:
    if n < 3:
        raise ValueError("need at least 3 images to split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_lab = max(2, int(round(n * labelled_frac)))
    n_train = min(max(1, int(round(n_lab * train_frac))), n_lab - 1)
    return {
        "train": sorted(perm[:n_train].tolist()),
        "val": sorted(perm[n_train:n_lab].tolist()),
        "test": sorted(perm[n_lab:].tolist()),
    }


def save_splits(splits: Dict[str, List[int]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(splits, f)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path) as f:
        return json.load(f)
