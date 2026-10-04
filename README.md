# MFLD-net — Mobile Fish Landmark Detection network (PyTorch)

Re-implementation of the network, loss, training recipe and evaluation protocol of

> Saleh A., Jones D., Jerry D., Rahimi Azghadi M. (2023). *MFLD-net: a lightweight deep learning network for
> fish morphometry using landmark detection.* Aquatic Ecology 57:913–931. <https://doi.org/10.1007/s10452-023-10044-8>

The network predicts landmarks on a fish photo (heatmaps **and** coordinates) and turns them into body
measurements (total length, standard length, body depth, head length).

## Status

All code is tested: `python -m unittest discover -s tests -v` (24 tests, including a 1-epoch end-to-end run) passes with
PyTorch + albumentations, and a 2-epoch run on the real betta data trains and evaluates end to end. Reproduced numbers
from the paper are **not** claimed — the paper's data are not public and several details are unspecified (see
"Interpretations and gaps").

## Install

```bash
pip install -r requirements.txt      # torch, numpy, opencv-python, albumentations, matplotlib (+ gradio for app.py)
python -m unittest discover -s tests -v
```

## Betta dataset / Kaggle

`betta_data.zip` (Roboflow COCO-keypoint export: `raw/` photos, `annotations/annotations.json` with 13 keypoints per fish,
`splits/{train,val,test}.json` image-id lists) is **not** in this repo (private data, git-ignored). Upload it to Kaggle as a
dataset, then in a GPU notebook:

```bash
!git clone https://github.com/githubmiguel24/mfld-net.git && cd mfld-net && pip install -q "albumentations>=1.4"
!python kaggle_train.py --epochs 50         # prepare -> train -> evaluate + dump; outputs in /kaggle/working/runs/betta_full
```

or import `notebooks/kaggle_train.ipynb`. `kaggle_train.py` finds the data under `/kaggle/input` (extracted or still zipped).
The steps it runs, usable locally too:

```bash
python prepare_betta.py --data-dir <folder with raw/ annotations/ splits/> --out data/betta
python train.py    --data-root data/betta --annotations data/betta/annotations.json --splits data/betta/splits.json \
                   --num-keypoints 13 --out runs/betta_full --epochs 50
python evaluate.py --data-root data/betta --annotations data/betta/annotations.json --splits data/betta/splits.json \
                   --checkpoint runs/betta_full/best.pt --spec configs/betta_spec.json --dump-predictions runs/betta_full
```

Notes:

* Each fish annotation becomes one sample, cropped to its box (+10 % margin). The 45 annotations with an unlabelled
  keypoint are dropped: 1610 fish = 1126 train / 245 val / 239 test, using the supplied image-level splits rather than
  the paper's 40/60 split.
* Keypoint order is the one in the export (`snout_tip, eye_center, dorsal_fin_base_anterior, …`).
  The layout and measurements are the defaults for 13 keypoints (`BETTA_*` in `mfld/morphometry.py`, mirrored in `configs/betta_spec.json`); the measurements are (total length = snout→caudal centre, standard length =
  snout→mid caudal peduncle, body depth = anterior dorsal base→anterior anal base, head length = snout→eye) — these
  are assumptions, adjust them.
* There is no pixel→mm calibration, so measurements are in pixels.
* The supplied splits are used as they are: `train.py --splits` trains on 100 % of the train split (it records
  `labelled_frac = 1.0`), selects the best checkpoint on val and never touches test. The paper's 40 % rule only applies
  to the random split used without `--splits`.
* `evaluate.py --dump-predictions DIR` writes `predictions_val.npz` / `predictions_test.npz` (`image_ids`,
  `annotation_ids`, `files`, `pred_mu` (N, 13, 2) in ORIGINAL photo pixels, `pred_conf` (N, 13), `keypoint_names`) and
  `preprocessing.json` (crop, resize, normalisation, decoding, mapping back). A photo with several fish appears once per
  fish, so match on `annotation_ids` when `image_ids` repeat. `train.py` also writes `preprocessing.json` next to the checkpoint.
* Optional, off by default (the paper's recipe): `--no-vflip` and `--img-size N` in `train.py`.
* `--workers` defaults to 4 on Kaggle; use `--workers 0` locally on Windows if data loading hangs.

## Quick start (synthetic data)

The paper's barramundi images are not public, so a generator produces fish-like images with exact 16-point (barramundi layout) labels. The project default is the 13-point betta layout, so pass `--num-keypoints 16` here:

```bash
python make_synthetic_data.py --out data/synthetic --n 250
python train.py    --data-root data/synthetic --annotations data/synthetic/annotations.json --num-keypoints 16 --out runs/demo --epochs 50
python evaluate.py --data-root data/synthetic --annotations data/synthetic/annotations.json --checkpoint runs/demo/best.pt
python predict.py  --checkpoint runs/demo/best.pt --images data/synthetic/images/fish_00000.png --mm-per-pixel 0.9
python benchmark.py                  # parameters / size / throughput (Table 1 of the paper)
python app.py --checkpoint runs/demo/best.pt --mm-per-pixel 0.9   # Gradio demo
```

## Your own data

`annotations.json`:

```json
{"images": [{"file": "images/a.jpg",
             "keypoints": [[x, y], "... K points, pixels in the ORIGINAL image"],
             "scale": 321.5,
             "mm_per_pixel": 0.9,
             "manual_mm": {"total_length": 412.0, "standard_length": 350.0, "body_depth": 98.0, "head_length": 101.0}}]}
```

`scale` (sqrt of the fish segment area — used by OKS), `mm_per_pixel` and `manual_mm` are optional. The keypoint
**order** is yours to define: edit `KEYPOINT_NAMES` / `DEFAULT_SPEC` in `mfld/morphometry.py` or pass `--spec my_spec.json`
(`{"total_length": [0, [10, 11]], ...}`; a list of indices means "mean of these points"). Use `--num-keypoints K` in
`train.py`; K = 13 (betta) and 16 (barramundi) have built-in layouts/specs, any other K needs `--spec`.

## What follows the paper

| Paper | Code |
|---|---|
| 224×224 input, 56×56 heatmaps, 16 keypoints (here 13 for betta, `--num-keypoints 16` for the paper's layout) | `mfld/config.py` |
| Patch embedding: conv 3→256, kernel 4, stride 4 | `MFLDNet.patch_embed` (`mfld/model.py`) |
| 8 isometric ConvBlocks: depthwise conv → GELU → BatchNorm → (+ residual) → spatial dropout 0.2 → pointwise conv → GELU → BatchNorm | `ConvBlock` |
| Kernel size 9, no pooling layers | `ModelConfig.kernel_size = 9` |
| HeatMap Conv → SoftMax → heatmaps; → Soft-Argmax → (x, y) | `MFLDNet.forward`, `soft_argmax` |
| Loss = average of Jensen–Shannon (heatmaps, Eq. 3 / Eq. 2) and Euclidean distance (coordinates, Eq. 1) | `mfld/losses.py` |
| Augmentations: h/v flip .5, shift+scale .5, rotate ±20° .5, blur .3, RGB-shift 25 .3 (train only) | `build_train_transform` in `mfld/data.py` |
| Adam lr 1e-3, β=(0.9, 0.999), ε=1e-8; StepLR γ=0.1 every 30 epochs; batch 64; ~50 epochs | `mfld/engine.py::fit`, `TrainConfig` |
| 40 % of images for train+val (70/30), 60 % held out as test; early stop after 50 epochs without improvement | `mfld/splits.py`, `fit` |
| OKS (Eq. 4), AP / AP.50 / AP.75, AR / AR.50 / AR.75 | `mfld/metrics.py` |
| MAD and SDD of DL vs manual measurements (Table 3) | `mfld/metrics.py`, `evaluate.py` |
| Total length, standard length, body depth, head length | `mfld/morphometry.py` |

## Interpretations and gaps in the paper (decide for yourself)

1. **Boxes labelled "Linear" in Fig. 2** are implemented as **GELU** — the text says every convolution is followed by GELU and BatchNorm.
2. **Parameter count.** With kernel 9 (as stated in the Discussion) this implementation has **0.719 M** parameters / 2.74 MB
   (fp32, MiB). The paper reports 0.65 M / 2.7 MB. Kernel 7 gives 0.654 M, kernel 9 matches the 2.7 MB size — the paper's numbers
   are not mutually consistent for any single kernel size. Default is 9 (explicit in the text); use `--kernel-size 7` for 0.65 M.
3. **Not specified in the paper, chosen here:** target Gaussian σ = 1.5 heatmap px (`--sigma`); OKS falloff k_i = 0.05 for all
   keypoints (`--kappa`); ImageNet normalisation; heatmaps are spatial softmax probabilities (sum to 1); no norm/activation after
   the patch embedding; coordinates are normalised to [0, 1] so the coordinate loss is comparable to Table 1 (~0.04).
4. **Keypoint layout** (16 points) is not published; the barramundi layout in `mfld/morphometry.py` is a reasonable assumption. The betta layout (13 points) comes from the dataset.
5. **Augmentation units:** the paper prints shift/scale limits with a degree sign; they are fractions (±6.25 % shift, ±20 % scale).
   Blur "limit 1" is implemented as a 3×3 blur. Flips keep keypoint *indices* (as in the paper), so left/right and top/bottom
   semantics are not swapped — fine for roughly symmetric labels, consider removing flips if your indices are orientation-specific.
6. The paper's Eq. 1 sums over keypoints; here the per-keypoint distance is averaged over keypoints and batch.

## Layout

```
configs/         betta_spec.json
notebooks/       kaggle_train.ipynb
mfld/            model.py losses.py data.py engine.py inference.py metrics.py morphometry.py heatmaps.py splits.py synthetic.py config.py
train.py evaluate.py predict.py benchmark.py make_synthetic_data.py plot_curves.py app.py prepare_betta.py kaggle_train.py
tests/           test_numpy_parts.py (runs anywhere)   test_torch_parts.py (needs torch + albumentations)
```

The original article is open access (CC BY 4.0); this code is an independent implementation and is not affiliated with the authors.
Code license: MIT (see `LICENSE`).
