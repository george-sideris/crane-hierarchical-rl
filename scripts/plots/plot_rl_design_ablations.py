#!/usr/bin/env python3
"""Encoder-freezing and critic-asymmetry ablations, under the argmax protocol.

Two design choices of the fine-tuning setup are tested by rerunning it with one factor
changed: the PointNet encoder is left trainable instead of frozen, and the critic reads the
same point cloud as the actor instead of the privileged state. One run per configuration.

These arms are plotted from their ARGMAX evaluations rather than their training rollouts.
The rollout curves for the two ablation runs were never synced home before the pods were
released, and the argmax rows are the citable measure in any case (Section on the evaluation
protocol): every point is one checkpoint replayed deterministically on 40 environments times
40 episodes at a fixed seed, so all three arms face identical piles.

    python3 scripts/plots/plot_rl_design_ablations.py [--out ...]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "legend.fontsize": 7,
    "mathtext.fontset": "cm",
})

F = "logs/fleet_tb"
LIVE = "logs/fleet_tb_live"
SMOOTH = 5
XMAX = 49920

BAND = [f"{F}/S42/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_03-25-25",
        f"{F}/S43/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-25-59",
        f"{F}/S44/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-09-19"]
# Each ablation is its single uninterrupted run. The 2026-08-20/21 fragments are excluded:
# they began from a random policy because the resume flag never reached the config.
UNFROZ = f"{LIVE}/unfroz/logs/rsl_rl/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-20_10-36-48"
SYMCRIT = f"{F}/G_sigma03/crane_pointcloud_gaze_scoring_ppo_v2_cc0_symcritic/2026-08-13_01-21-35"


def load(d, tag):
    if not os.path.isdir(d):
        return np.array([]), np.array([])
    ea = EventAccumulator(d); ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    v = ea.Scalars(tag)
    s = np.array([x.step for x in v], float)
    y = np.array([x.value for x in v], float)
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/thesis/figures/rl_design_ablations.pdf")
    a = ap.parse_args()

    PANELS = [("Episode/pile_clearing_pct_final", "clearing at episode end [\%]"),
              ("Episode/stability", "stability $\\varsigma$")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    for (tag, ylab), ax in zip(PANELS, axes):
        runs = [load(d, tag) for d in BAND]
        runs = [r for r in runs if r[0].size]
        hi = min(min(r[0][-1] for r in runs), XMAX)
        grid = np.linspace(max(r[0][0] for r in runs), hi, 300)
        Y = np.stack([np.interp(grid, s, y) for s, y in runs])
        ax.plot(grid, Y.mean(0), color="#4c72b0", lw=1.5,
                label="frozen encoder, asymmetric critic (N=3)")
        ax.fill_between(grid, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                        color="#4c72b0", alpha=0.18, lw=0)
        for d, lab, col, ls in [(UNFROZ, "encoder left trainable", "#dd8452", "--"),
                                (SYMCRIT, "critic on the point cloud", "#8172b2", "-.")]:
            s, y = load(d, tag)
            if s.size:
                m = s <= XMAX
                ax.plot(s[m], y[m], color=col, ls=ls, lw=1.3, label=f"{lab} (N=1)")
        ax.set_ylabel(ylab)
        ax.set_xlabel("environment steps (grasp cycles)")
        ax.grid(lw=0.4, alpha=0.4)
        ax.set_xlim(0, XMAX)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)
    for n, d in [("unfroz", UNFROZ), ("symcritic", SYMCRIT)]:
        s, _ = load(d, "Train/mean_reward")
        print(f"  {n}: {int(s[-1]) if s.size else 0} env steps of valid training")


if __name__ == "__main__":
    main()
