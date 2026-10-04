#!/usr/bin/env python
"""Generate a synthetic fish-landmark dataset (the real barramundi data is not public)."""
import argparse

from mfld.synthetic import generate_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=250, help="number of images")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = generate_dataset(a.out, a.n, (a.width, a.height), a.seed)
    print(f"wrote {a.n} images and {path}")


if __name__ == "__main__":
    main()
