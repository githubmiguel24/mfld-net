#!/usr/bin/env python
"""Model cost figures as reported in Table 1 of the paper: number of parameters, model size on
disk and inference throughput (images / second)."""
import argparse
import io
import time

import torch

from mfld.config import ModelConfig
from mfld.engine import get_device
from mfld.model import MFLDNet, count_parameters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None, help="optional: benchmark a trained model")
    ap.add_argument("--kernel-size", type=int, default=ModelConfig.kernel_size)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = get_device(a.device)
    if a.checkpoint:
        from mfld.inference import load_model
        model, _ = load_model(a.checkpoint, device)
    else:
        model = MFLDNet(ModelConfig(kernel_size=a.kernel_size)).to(device).eval()
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    n = count_parameters(model)
    x = torch.randn(a.batch_size, 3, model.cfg.img_size, model.cfg.img_size, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"device     : {device}")
    print(f"parameters : {n / 1e6:.3f} M   (paper: 0.65 M)")
    print(f"size       : {buf.tell() / 2 ** 20:.2f} MB (paper: 2.7 MB)")
    print(f"throughput : {a.batch_size * a.iters / dt:.0f} img/s (batch {a.batch_size}; paper: 480 img/s on an RTX 2080 Ti)")


if __name__ == "__main__":
    main()
