"""Training / validation loss curves (cf. Fig. 7 of the paper)."""
import json
import sys


def plot_history(history_json: str, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_json) as f:
        h = json.load(f)
    ep = [r["epoch"] for r in h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, split in zip(axes, ("train", "val")):
        ax.plot(ep, [r[f"{split}_coords"] for r in h], label=f"{split} loss (coordinates)")
        ax.plot(ep, [r[f"{split}_heatmap"] for r in h], label=f"{split} loss (heatmap)")
        ax.plot(ep, [r[f"{split}_total"] for r in h], label=f"total {split} loss")
        ax.set_title(f"MFLD-net {'Training' if split == 'train' else 'Validation'}")
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_history(sys.argv[1], sys.argv[2])
