#!/usr/bin/env python3
"""Training curves for the scoring BC head, banded over three training seeds.

Reads the per-epoch histories of the three seeds trained on the same dataset with
the same recipe. Seed 42 is the clean rerun of the deployed checkpoint (it reproduces
its best validation loss exactly, so it describes the shipped run) and carries the
checkpoint marker; the band is the mean +/- sd over all three. Two panels: the loss
pair, and the median horizontal error of the decoded grasp on the validation split.

The band is narrow throughout, which is the point: the scoring head trains to the
same place from any seed, where the regression head does not.

Training clouds carry fresh ZED noise every epoch and validation runs on clean
clouds, so validation tends to sit slightly below training: by 0.12 over the first
twenty epochs and 0.04 over the last twenty, averaged over the three seeds. The gap
is small enough that the curves cross in mid-training (validation is 0.07 above
between epochs 40 and 60), so it is a tendency, not a guarantee.

    python3 scripts/plots/plot_bc_training_curves.py \
        [--history logs/bc_pointcloud/scoring_margin05_2048_c_rerun/history.jsonl] \
        [--out docs/thesis/figures/bc_training_curves.pdf]
"""

import argparse
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "legend.fontsize": 7,
    "mathtext.fontset": "cm",
})


def band(hists, key):
    """Mean and sd across seeds at each epoch, over the epochs every seed reached."""
    n = min(len(h) for h in hists)
    Y = np.array([[r[key] for r in h[:n]] for h in hists], dtype=float)
    return np.arange(1, n + 1), Y.mean(0), Y.std(0), Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--histories", nargs="+", default=[
        "logs/bc_pointcloud/scoring_margin05_2048_c_rerun/history.jsonl",   # seed 42
        "logs/bc_pointcloud/scoring_margin05_2048_c_s43/history.jsonl",
        "logs/bc_pointcloud/scoring_margin05_2048_c_s44/history.jsonl",
    ])
    ap.add_argument("--out", default="docs/thesis/figures/bc_training_curves.pdf")
    a = ap.parse_args()

    hists = [[json.loads(l) for l in open(f)] for f in a.histories]
    N = len(hists)
    # the deployed seed is the first history; its best epoch is the shipped checkpoint
    best = min(hists[0], key=lambda r: r["val"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for key, col, lab in [("train", "#dd8452", "training (fresh ZED noise per epoch)"),
                          ("val", "#4c72b0", "validation (clean clouds)")]:
        ep, mu, sd, _ = band(hists, key)
        ax1.plot(ep, mu, color=col, lw=1.4, label=f"{lab}")
        ax1.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.20, lw=0)
    ax1.plot([best["epoch"]], [best["val"]], marker="D", ms=4.5,
             color="#4c72b0", mec="black", mew=0.6, ls="none",
             label=f"checkpoint (epoch {best['epoch']})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(frameon=False, title=f"mean $\\pm$ sd, N={N} seeds", title_fontsize=6.5)
    ax1.grid(lw=0.4, alpha=0.4)

    ep, mu, sd, Y = band(hists, "xy_err")
    ax2.plot(ep, mu, color="#55a868", lw=1.4)
    ax2.fill_between(ep, mu - sd, mu + sd, color="#55a868", alpha=0.20, lw=0)
    ax2.plot([best["epoch"]], [best["xy_err"]], marker="D", ms=4.5,
             color="#55a868", mec="black", mew=0.6, ls="none")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("median horizontal error [m]")
    ax2.grid(lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    finals = [min(h, key=lambda r: r["val"]) for h in hists]
    print(f"wrote {a.out}  (N={N}; best val "
          + ", ".join(f"{b['val']:.4f}" for b in finals)
          + "; xy-err " + ", ".join(f"{b['xy_err']:.3f}" for b in finals) + " m)")


if __name__ == "__main__":
    main()
