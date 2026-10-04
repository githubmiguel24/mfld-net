#!/usr/bin/env python
"""Evaluate a trained MFLD-net on the held-out test split (60 % of the data).

Reports the losses (cf. Table 1), OKS-based AP / AR (cf. Table 2) and, per body trait, the
MAD / SDD between the model's measurements and the reference measurements (cf. Table 3).
Reference = ``manual_mm`` in the annotation if present, otherwise measurements derived from
the ground-truth keypoints.
"""
import argparse
import json
import os

import numpy as np

from mfld.config import TrainConfig
from mfld.data import FishLandmarkDataset
from mfld.engine import collect_predictions, get_device, keypoint_metrics
from mfld.metrics import mean_absolute_difference, standard_deviation_of_difference
from mfld.inference import load_model
from mfld.morphometry import DEFAULT_SPEC, load_spec, measure
from mfld.splits import load_splits, make_splits
from mfld.data import load_annotations


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default=None, help="splits.json written by train.py (default: next to the checkpoint)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--kappa", type=float, default=0.05, help="OKS per-keypoint falloff constant")
    ap.add_argument("--spec", default=None, help="JSON measurement spec (see mfld/morphometry.py)")
    ap.add_argument("--out", default=None, help="write the results json here")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    model, ckpt = load_model(a.checkpoint, device)
    records = load_annotations(a.annotations)
    sp_path = a.splits or os.path.join(os.path.dirname(a.checkpoint), "splits.json")
    splits = load_splits(sp_path) if os.path.exists(sp_path) else make_splits(len(records), seed=ckpt["train_cfg"]["seed"])
    ds = FishLandmarkDataset([records[i] for i in splits[a.split]], a.data_root, model.cfg,
                             augment=False, sigma=ckpt["train_cfg"].get("sigma", TrainConfig.sigma))
    preds = collect_predictions(model, ds, device)

    results = {"split": a.split, "n_images": len(ds), "losses": preds["losses"],
               "keypoints": keypoint_metrics(preds, model.cfg.img_size, a.kappa)}

    spec = load_spec(a.spec) if a.spec else DEFAULT_SPEC
    pred_m = {t: [] for t in spec}
    ref_m = {t: [] for t in spec}
    for rec, p, size in zip(ds.records, preds["pred"], preds["orig_size"]):
        mm = float(rec.get("mm_per_pixel", 1.0))
        pm = measure(p * size, spec, mm)                       # prediction mapped back to original pixels
        rm = rec.get("manual_mm") or measure(rec["keypoints"], spec, mm)
        for t in spec:
            pred_m[t].append(pm[t])
            ref_m[t].append(rm[t])
    unit = "mm" if any(r.get("mm_per_pixel") for r in ds.records) else "px"
    results["morphometry"] = {t: {"MAD": mean_absolute_difference(pred_m[t], ref_m[t]),
                                  "SDD": standard_deviation_of_difference(pred_m[t], ref_m[t]),
                                  "unit": unit} for t in spec}
    print(json.dumps(results, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
