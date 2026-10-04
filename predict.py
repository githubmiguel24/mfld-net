#!/usr/bin/env python
"""Predict landmarks and the four morphometric traits for one or more images.

    python predict.py --checkpoint runs/demo/best.pt --images fish1.jpg fish2.jpg --out preds/ --mm-per-pixel 0.9
"""
import argparse
import csv
import json
import os

import cv2

from mfld.engine import get_device
from mfld.inference import draw_keypoints, load_model, predict_images
from mfld.morphometry import KEYPOINT_NAMES, load_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--mm-per-pixel", type=float, default=1.0, help="camera calibration (1.0 = report pixels)")
    ap.add_argument("--spec", default=None, help="JSON measurement spec (see mfld/morphometry.py)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    model, _ = load_model(a.checkpoint, device)
    spec = load_spec(a.spec) if a.spec else None
    names = KEYPOINT_NAMES if model.cfg.num_keypoints == len(KEYPOINT_NAMES) else []   # default layout only
    imgs = []
    for p in a.images:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(p)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    results = predict_images(model, imgs, device, spec=spec, mm_per_pixel=a.mm_per_pixel)

    os.makedirs(a.out, exist_ok=True)
    unit = "mm" if a.mm_per_pixel != 1.0 else "px"
    summary = []
    for path, img, r in zip(a.images, imgs, results):
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(a.out, f"{stem}_landmarks.png"), draw_keypoints(img, r["keypoints_px"], names=None)[..., ::-1])
        with open(os.path.join(a.out, f"{stem}_keypoints.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "name", "x_px", "y_px", "confidence"])
            for i, ((x, y), c) in enumerate(zip(r["keypoints_px"], r["confidence"])):
                w.writerow([i, names[i] if i < len(names) else "", f"{x:.2f}", f"{y:.2f}", f"{c:.4f}"])
        summary.append({"image": path, "unit": unit, **{k: round(v, 2) for k, v in r["measurements"].items()}})
        print(path, {k: round(v, 1) for k, v in r["measurements"].items()}, unit)
    with open(os.path.join(a.out, "measurements.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
