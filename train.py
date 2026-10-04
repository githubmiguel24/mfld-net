#!/usr/bin/env python
"""Train MFLD-net with the recipe of the paper.

    python train.py --data-root data/synthetic --annotations data/synthetic/annotations.json --out runs/demo

Paper defaults: 224x224 input, 56x56 heatmaps, 16 keypoints (barramundi; this repo defaults to 13 betta keypoints), 8 ConvBlocks, kernel 9,
Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8), StepLR(30, 0.1), batch 64, 50 epochs,
40 % of the images for training+validation (70/30), the remaining 60 % held out for testing.
"""
import argparse
import json
import os
from dataclasses import asdict

from mfld.config import ModelConfig, TrainConfig
from mfld.data import build_datasets, load_annotation_meta
from mfld.engine import fit, get_device, set_seed
from mfld.model import MFLDNet, count_parameters
from mfld.preprocessing import write_preprocessing
from mfld.splits import load_splits, save_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--splits", default=None, help="splits.json with index lists (e.g. from prepare_betta.py); "
                                                   "default: random split by --labelled-frac / --seed")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--sigma", type=float, default=TrainConfig.sigma, help="target Gaussian std-dev (heatmap px)")
    ap.add_argument("--labelled-frac", type=float, default=TrainConfig.labelled_frac)
    ap.add_argument("--patience", type=int, default=TrainConfig.patience)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--kernel-size", type=int, default=ModelConfig.kernel_size)
    ap.add_argument("--depth", type=int, default=ModelConfig.depth)
    ap.add_argument("--dim", type=int, default=ModelConfig.dim)
    ap.add_argument("--num-keypoints", type=int, default=ModelConfig.num_keypoints)
    ap.add_argument("--no-vflip", action="store_true", help="drop the vertical-flip augmentation (not in the paper's recipe)")
    ap.add_argument("--img-size", type=int, default=ModelConfig.img_size,
                    help="network input size (multiple of 4); heatmaps are img-size/4. Paper: 224")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    set_seed(a.seed)
    model_cfg = ModelConfig(img_size=a.img_size, kernel_size=a.kernel_size, depth=a.depth, dim=a.dim, num_keypoints=a.num_keypoints)
    train_cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, sigma=a.sigma,
                            labelled_frac=a.labelled_frac, patience=a.patience, seed=a.seed,
                            vflip=not a.no_vflip)
    splits = load_splits(a.splits) if a.splits else None
    if splits is not None:
        # supplied splits are used as they are: 100 % of the train split trains, val selects the best checkpoint,
        # test is never touched here. labelled_frac only applies to the random split, so record it as 1.0.
        train_cfg.labelled_frac = 1.0
        sets = {k: set(splits[k]) for k in ("train", "val", "test")}
        assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]),             "train/val/test splits overlap"
    train_ds, val_ds, test_ds, splits = build_datasets(a.data_root, a.annotations, model_cfg, train_cfg, splits=splits)
    os.makedirs(a.out, exist_ok=True)
    save_splits(splits, os.path.join(a.out, "splits.json"))
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"model": model_cfg.to_dict(), "train": asdict(train_cfg)}, f, indent=1)

    meta = load_annotation_meta(a.annotations)
    write_preprocessing(os.path.join(a.out, "preprocessing.json"), model_cfg, meta.get("preprocessing"),
                        meta.get("keypoint_names"))

    device = get_device(a.device)
    model = MFLDNet(model_cfg).to(device)
    print(f"device {device} | params {count_parameters(model) / 1e6:.3f} M | "
          f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)} images")
    fit(model, train_ds, val_ds, a.out, model_cfg, train_cfg, device, num_workers=a.workers)
    print(f"done - best checkpoint: {os.path.join(a.out, 'best.pt')}")
    try:
        from plot_curves import plot_history
        plot_history(os.path.join(a.out, "history.json"), os.path.join(a.out, "loss_curves.png"))
    except Exception as e:                                 # plotting is optional
        print("could not plot loss curves:", e)


if __name__ == "__main__":
    main()
