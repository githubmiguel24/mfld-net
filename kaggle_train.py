#!/usr/bin/env python
"""One-shot Kaggle driver: locate the betta dataset, prepare it, train, evaluate, plot.

Kaggle notebook (GPU on, the dataset containing betta_data.zip attached via "Add Input"):

    !git clone https://github.com/githubmiguel24/mfld-net.git
    %cd mfld-net
    !pip install -q "albumentations>=1.4"
    !python kaggle_train.py --epochs 100

Kaggle extracts uploaded zips, so the data appears as /kaggle/input/<dataset>/{raw,annotations,splits}
(possibly one folder deeper, or still as betta_data.zip - both are handled). Results land in /kaggle/working.
"""
import argparse
import glob
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def find_dataset(input_dir: str, work: str) -> str:
    """Folder that contains raw/, annotations/ and splits/ (extracted, or inside any .zip)."""
    for root, dirs, _ in os.walk(input_dir):
        if {"raw", "annotations", "splits"} <= set(dirs):
            return root
    hits = glob.glob(os.path.join(input_dir, "**", "annotations.json"), recursive=True)
    for h in hits:                                   # annotations.json without the expected sibling folders
        root = os.path.dirname(os.path.dirname(h))
        if os.path.isdir(os.path.join(root, "raw")):
            return root
    zips = sorted(glob.glob(os.path.join(input_dir, "**", "*.zip"), recursive=True))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            if not any(n.endswith("annotations/annotations.json") for n in zf.namelist()):
                continue
            dst = os.path.join(work, "betta_raw")
            print(f"extracting {z} -> {dst}")
            zf.extractall(dst)
        return find_dataset(dst, work)
    listing = []
    for root, dirs, files in os.walk(input_dir):
        depth = root[len(input_dir):].count(os.sep)
        if depth <= 3:
            listing.append(f"{root}  dirs={dirs[:6]} files={files[:4]}")
    sys.exit(f"betta dataset (raw/, annotations/, splits/) not found under {input_dir}. Contents:\n"
             + "\n".join(listing[:40]) + "\nAttach the dataset with 'Add Input' and re-run.")


def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="/kaggle/input", help="where Kaggle mounts datasets (or a local folder)")
    ap.add_argument("--work-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kernel-size", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="betta_full", help="output folder name under <work-dir>/runs")
    ap.add_argument("--no-vflip", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    a = ap.parse_args()

    data = find_dataset(a.input_dir, a.work_dir)
    prep = os.path.join(a.work_dir, "betta")
    out = os.path.join(a.work_dir, "runs", a.run_name)
    spec = os.path.join("configs", "betta_spec.json")
    ann, splits = os.path.join(prep, "annotations.json"), os.path.join(prep, "splits.json")

    run("prepare_betta.py", "--data-dir", data, "--out", prep)
    run("train.py", "--data-root", prep, "--annotations", ann, "--splits", splits, "--out", out,
        "--num-keypoints", "13", "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
        "--workers", str(a.workers), "--kernel-size", str(a.kernel_size), "--seed", str(a.seed),
        "--img-size", str(a.img_size), *(["--no-vflip"] if a.no_vflip else []))
    run("evaluate.py", "--data-root", prep, "--annotations", ann, "--splits", splits, "--spec", spec,
        "--checkpoint", os.path.join(out, "best.pt"), "--out", os.path.join(out, "test_results.json"),
        "--dump-predictions", out)
    print(f"\nDone. Download from {out}: best.pt, preprocessing.json, predictions_val.npz, "
          f"predictions_test.npz, history.json, loss_curves.png, test_results.json")


if __name__ == "__main__":
    main()
