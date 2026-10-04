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
    """Folder that contains raw/, annotations/ and splits/."""
    hits = glob.glob(os.path.join(input_dir, "**", "annotations", "annotations.json"), recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    zips = glob.glob(os.path.join(input_dir, "**", "betta_data.zip"), recursive=True)
    if zips:
        dst = os.path.join(work, "betta_raw")
        print(f"extracting {zips[0]} -> {dst}")
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dst)
        return dst
    sys.exit(f"betta dataset not found under {input_dir} - attach it with 'Add Input' in the notebook sidebar")


def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=HERE, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="/kaggle/input", help="where Kaggle mounts datasets (or a local folder)")
    ap.add_argument("--work-dir", default="/kaggle/working")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kernel-size", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    data = find_dataset(a.input_dir, a.work_dir)
    prep = os.path.join(a.work_dir, "betta")
    out = os.path.join(a.work_dir, "runs", "betta")
    spec = os.path.join("configs", "betta_spec.json")
    ann, splits = os.path.join(prep, "annotations.json"), os.path.join(prep, "splits.json")

    run("prepare_betta.py", "--data-dir", data, "--out", prep)
    run("train.py", "--data-root", prep, "--annotations", ann, "--splits", splits, "--out", out,
        "--num-keypoints", "13", "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
        "--workers", str(a.workers), "--kernel-size", str(a.kernel_size), "--seed", str(a.seed))
    run("evaluate.py", "--data-root", prep, "--annotations", ann, "--splits", splits, "--spec", spec,
        "--checkpoint", os.path.join(out, "best.pt"), "--out", os.path.join(out, "test_results.json"))
    print(f"\nDone. Download from {out}: best.pt, history.json, loss_curves.png, test_results.json")


if __name__ == "__main__":
    main()
