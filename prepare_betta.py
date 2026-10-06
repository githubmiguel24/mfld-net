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
import sys
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

    missing = sorted(i for i in img_split if i not in images)
    if missing:
        sys.exit(f"{len(missing)} image ids in the split files are not in annotations.json, e.g. {missing[:5]}")
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    records, splits, dropped = [], {"train": [], "val": [], "test": [], "val_all": [], "test_all": []}, Counter()
    cache = {}
    for ann in sorted(coco["annotations"], key=lambda x: x["id"]):
        im = images[ann["image_id"]]
        split = img_split.get(im["id"])
        kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
        if split is None:
            dropped["not in any split"] += 1
            continue
        unlabelled = bool((kp[:, 2] == 0).any())
        if unlabelled and split == "train":      # (0, 0) placeholders cannot be trained on without a masked loss
            dropped["unlabelled keypoint (train only)"] += 1
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
            "crop": [x0, y0, x1, y1],          # crop window in the ORIGINAL photo (x0, y0, x1, y1) - for mapping back
            "image_id": im["id"],
            "split": split,
            "labelled": not unlabelled,
            "annotation_id": ann["id"],
        })
        if not unlabelled:
            splits[split].append(len(records) - 1)              # used for training / checkpoint selection / metrics
        if split != "train":
            splits[split + "_all"].append(len(records) - 1)     # every fish, for the predictions dump

    with open(os.path.join(a.out, "annotations.json"), "w") as f:
        json.dump({"keypoint_names": names,
                   "preprocessing": {"input": "fish_box_crop", "margin": a.margin,
                                     "crop_covers_all_keypoints": True},
                   "images": records}, f)
    with open(os.path.join(a.out, "splits.json"), "w") as f:
        json.dump(splits, f)
    # reconcile every id of every split file with the fish that were kept
    report = {}
    for split in ("train", "val", "test"):
        ids = {i for i, s in img_split.items() if s == split}
        n_ann = sum(1 for an in coco["annotations"] if an["image_id"] in ids)
        all_key = split if split == "train" else split + "_all"
        kept = {records[j]["image_id"] for j in splits[all_key]}
        report[split] = {"image_ids_in_json": len(ids), "annotations_for_these_images": n_ann,
                         "fish_with_all_keypoints_labelled": len(splits[split]),
                         "fish_in_predictions_dump": len(splits[all_key]),
                         "fish_not_used": n_ann - len(splits[all_key]),
                         "image_ids_with_no_fish": sorted(ids - kept)}
    with open(os.path.join(a.out, "splits_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    for k, v in report.items():
        print(f"  {k:5s}: {v['image_ids_in_json']} ids -> {v['annotations_for_these_images']} fish annotated | "
              f"{v['fish_with_all_keypoints_labelled']} fully labelled (training/metrics) | "
              f"{v['fish_in_predictions_dump']} in predictions dump | {len(v['image_ids_with_no_fish'])} ids with no fish")
    print(f"{len(records)} fish -> {a.out} | " + " / ".join(f"{k} {len(v)}" for k, v in splits.items() if not k.endswith("_all"))
          + f" | {len(names)} keypoints | dropped: {dict(dropped) or 'none'}")


if __name__ == "__main__":
    main()
