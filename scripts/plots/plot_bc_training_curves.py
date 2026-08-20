#!/usr/bin/env python3
"""Training curves for the deployed scoring BC checkpoint.

Reads the per-epoch history of the clean rerun (same recipe, same seed as the
deployed checkpoint; the rerun reproduces its best validation loss exactly, so
these curves describe the shipped run). Two panels: the loss pair, and the
median horizontal error of the decoded grasp on the validation split. The
best-validation epoch (the checkpointed one) is marked on both.

Validation sits below training loss by construction: training clouds carry
fresh ZED noise every epoch while validation runs on clean clouds.

    python3 scripts/plots/plot_bc_training_curves.py \
        [--history logs/bc_pointcloud/scoring_margin05_2048_c_rerun/history.jsonl] \
        [--out docs/thesis/figures/bc_training_curves.pdf]
"""

import argparse
import json

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history",
                    default="logs/bc_pointcloud/scoring_margin05_2048_c_rerun/history.jsonl")
    ap.add_argument("--out", default="docs/thesis/figures/bc_training_curves.pdf")
    a = ap.parse_args()

    h = [json.loads(l) for l in open(a.history)]
    ep = [r["epoch"] for r in h]
    best = min(h, key=lambda r: r["val"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax1.plot(ep, [r["train"] for r in h], color="#dd8452", lw=1.4,
             label="training (fresh ZED noise per epoch)")
    ax1.plot(ep, [r["val"] for r in h], color="#4c72b0", lw=1.4,
             label="validation (clean clouds)")
    ax1.plot([best["epoch"]], [best["val"]], marker="D", ms=4.5,
             color="#4c72b0", mec="black", mew=0.6, ls="none",
             label=f"checkpoint (epoch {best['epoch']})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(frameon=False)
    ax1.grid(lw=0.4, alpha=0.4)

    ax2.plot(ep, [r["xy_err"] for r in h], color="#55a868", lw=1.4)
    ax2.plot([best["epoch"]], [best["xy_err"]], marker="D", ms=4.5,
             color="#55a868", mec="black", mew=0.6, ls="none")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("median horizontal error [m]")
    ax2.grid(lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}  (best val {best['val']:.4f} @ epoch {best['epoch']}, "
          f"xy-err {best['xy_err']:.3f} m, within-30cm {best['within30']:.0f}%)")


if __name__ == "__main__":
    main()
