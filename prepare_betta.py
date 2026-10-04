#!/usr/bin/env python
"""Convert the betta COCO-keypoint export (betta_data.zip) into the format train.py expects.

    python prepare_betta.py --data-dir <folder holding raw/ annotations/ splits/> --out data/betta

Input layout (as in betta_data.zip)::

    raw/*.jpg|png|webp            original photos
    annotations/annotations.json  COCO keypoints (13 points per fish, 1-4 fish per photo)
    splits/{train,val,test}.json  lists of COCO image ids

Every fish annotation becomes one record. The photo is cropped to the fish box (+ margin) and saved
under ``<out>/images/`` so that a photo with several fish yields one sample per fish and the
keypoints stay in the pixels of the saved crop. Annotations with an unlabelled keypoint (v == 0)
are dropped. Writes ``annotations.json`` (repo format) and ``splits.json`` (index lists).
"""
import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing raw/, annotations/ and splits/")
    ap.add_argument("--out", default="data/betta")
    ap.add_argument("--margin", type=float, default=0.1, help="crop margin as a fraction of the fish box")
    a = ap.parse_args()

    with open(os.path.join(a.data_dir, "annotations", "annotations.json")) as f:
        coco = json.load(f)
    cat = next(c for c in coco["categories"] if c.get("keypoints"))
    names = cat["keypoints"]
    images = {im["id"]: im for im in coco["images"]}
    img_split = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(a.data_dir, "splits", f"{split}.json")) as f:
            for i in json.load(f):
                img_split[int(i)] = split

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    records, splits, dropped = [], {"train": [], "val": [], "test": []}, Counter()
    cache = {}
    for ann in sorted(coco["annotations"], key=lambda x: x["id"]):
        im = images[ann["image_id"]]
        split = img_split.get(im["id"])
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        if split is None:
            dropped["not in any split"] += 1
            continue
        if (kp[:, 2] == 0).any():
            dropped["unlabelled keypoint"] += 1
            continue
        if im["id"] not in cache:
            cache = {im["id"]: cv2.imread(os.path.join(a.data_dir, "raw", im["file_name"]), cv2.IMREAD_COLOR)}
        img = cache[im["id"]]
        if img is None:
            dropped["unreadable image"] += 1
            continue
        H, W = img.shape[:2]
        pts = kp[:, :2]
        x, y, w, h = ann.get("fish_box") or ann["bbox"]
        # box enlarged by the margin, and always covering every keypoint
        x0 = min(x - a.margin * w, pts[:, 0].min())
        y0 = min(y - a.margin * h, pts[:, 1].min())
        x1 = max(x + (1 + a.margin) * w, pts[:, 0].max())
        y1 = max(y + (1 + a.margin) * h, pts[:, 1].max())
        x0, y0 = int(max(0, np.floor(x0))), int(max(0, np.floor(y0)))
        x1, y1 = int(min(W, np.ceil(x1))), int(min(H, np.ceil(y1)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            dropped["degenerate box"] += 1
            continue
        rel = f"images/{ann['id']:05d}.jpg"
        cv2.imwrite(os.path.join(a.out, rel), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_pts = pts - np.array([x0, y0], np.float32)
        records.append({
            "file": rel,
            "keypoints": np.round(crop_pts, 2).tolist(),
            "visibility": kp[:, 2].astype(int).tolist(),
            "scale": round(float(np.sqrt(w * h)), 2),
            "image_id": im["id"],
            "annotation_id": ann["id"],
        })
        splits[split].append(len(records) - 1)

    with open(os.path.join(a.out, "annotations.json"), "w") as f:
        json.dump({"keypoint_names": names, "images": records}, f)
    with open(os.path.join(a.out, "splits.json"), "w") as f:
        json.dump(splits, f)
    print(f"{len(records)} fish -> {a.out} | " + " / ".join(f"{k} {len(v)}" for k, v in splits.items())
          + f" | {len(names)} keypoints | dropped: {dict(dropped) or 'none'}")


if __name__ == "__main__":
    main()
