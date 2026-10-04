#!/usr/bin/env python
"""Small web demo (Gradio): upload a fish photo -> landmarks + body measurements.

    pip install gradio
    python app.py --checkpoint runs/demo/best.pt --mm-per-pixel 0.9
"""
import argparse

import numpy as np

from mfld.engine import get_device
from mfld.inference import draw_keypoints, load_model, predict_images


def build_app(checkpoint: str, mm_per_pixel: float, device: str = "auto"):
    import gradio as gr

    dev = get_device(device)
    model, _ = load_model(checkpoint, dev)
    unit = "mm" if mm_per_pixel != 1.0 else "px"

    def run(image: np.ndarray, mm: float):
        if image is None:
            return None, []
        res = predict_images(model, [image], dev, mm_per_pixel=float(mm))[0]
        overlay = draw_keypoints(image, res["keypoints_px"])
        rows = [[k.replace("_", " "), round(v, 1)] for k, v in res["measurements"].items()]
        return overlay, rows

    with gr.Blocks(title="MFLD-net fish morphometry") as demo:
        gr.Markdown("## MFLD-net - fish landmark detection and morphometry")
        with gr.Row():
            inp = gr.Image(type="numpy", label="Fish image (RGB, any size)")
            out = gr.Image(label="Detected landmarks")
        mm = gr.Number(value=mm_per_pixel, label="mm per pixel (1 = report pixels)")
        table = gr.Dataframe(headers=["trait", f"length ({unit})"], label="Measurements")
        gr.Button("Measure").click(run, [inp, mm], [out, table])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mm-per-pixel", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_app(a.checkpoint, a.mm_per_pixel, a.device).launch(server_port=a.port)
