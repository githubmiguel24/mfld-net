"""Training / validation / testing loops following the training recipe of the paper."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainConfig
from .losses import MultiTaskLoss
from .metrics import average_precision_recall, object_keypoint_similarity


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> Dict[str, float]:
    """One pass over ``loader``. Trains when an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    sums, count = defaultdict(float), 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            t_heat = batch["heatmaps"].to(device, non_blocking=True)
            t_coord = batch["coords"].to(device, non_blocking=True)
            out = model(images)
            losses = criterion(out, t_heat, t_coord)
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
            n = images.size(0)
            for k, v in losses.items():
                sums[k] += v.item() * n
            count += n
    return {k: v / max(count, 1) for k, v in sums.items()}


def fit(model, train_ds, val_ds, out_dir: str, model_cfg: ModelConfig, cfg: TrainConfig,
        device: torch.device, num_workers: int = 2, log=print) -> List[dict]:
    """Adam (lr 1e-3, betas (0.9, 0.999), eps 1e-8), StepLR(30, 0.1), batch 64, ~50 epochs;
    the best checkpoint on validation loss is kept and training stops when the validation
    loss has not improved for ``cfg.patience`` epochs."""
    os.makedirs(out_dir, exist_ok=True)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin, drop_last=len(train_ds) > cfg.batch_size)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    history, best, best_epoch = [], float("inf"), -1
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2),
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        ckpt = {"model": model.state_dict(), "model_cfg": model_cfg.to_dict(), "train_cfg": asdict(cfg),
                "epoch": epoch, "val": va}
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        if va["total"] < best:
            best, best_epoch = va["total"], epoch
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
        log(f"epoch {epoch:3d}/{cfg.epochs}  train {tr['total']:.4f}  val {va['total']:.4f} "
            f"(heat {va['heatmap']:.4f}, coord {va['coords']:.4f})  best@{best_epoch}  {row['seconds']}s")
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if epoch - best_epoch >= cfg.patience:
            log(f"validation loss has not improved for {cfg.patience} epochs - stopping")
            break
    return history


@torch.no_grad()
def collect_predictions(model, dataset, device, batch_size: int = 64, num_workers: int = 0) -> dict:
    """Runs the model over a dataset (no augmentation) and gathers everything evaluation needs."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    crit = MultiTaskLoss()
    pred, gt, conf, conf_kp, scale, size, sums, n = [], [], [], [], [], [], defaultdict(float), 0
    for batch in loader:
        out = model(batch["image"].to(device))
        losses = crit(out, batch["heatmaps"].to(device), batch["coords"].to(device))
        b = batch["image"].size(0)
        for k, v in losses.items():
            sums[k] += v.item() * b
        n += b
        pred.append(out.coords.cpu().numpy())
        gt.append(batch["coords"].numpy())
        peak = out.heatmaps.flatten(2).amax(-1)                                # (B, K) peak probability per keypoint
        conf_kp.append(peak.cpu().numpy())
        conf.append(peak.mean(-1).cpu().numpy())                               # mean peak value over keypoints
        scale.append(batch["scale"].numpy())
        size.append(batch["orig_size"].numpy())
    return {"pred": np.concatenate(pred), "gt": np.concatenate(gt), "conf": np.concatenate(conf), "conf_kp": np.concatenate(conf_kp),
            "scale": np.concatenate(scale), "orig_size": np.concatenate(size),
            "losses": {k: v / max(n, 1) for k, v in sums.items()}}


def keypoint_metrics(preds: dict, img_size: int, kappa: float = 0.05) -> dict:
    """Eq. 4 OKS -> AP / AR (.50:.05:.95, .50, .75) in the 224 x 224 network frame."""
    oks = object_keypoint_similarity(preds["pred"] * img_size, preds["gt"] * img_size, preds["scale"], kappa)
    res = average_precision_recall(oks, preds["conf"])
    res["mean_OKS"] = float(np.mean(oks))
    return {k: float(v) for k, v in res.items()}
