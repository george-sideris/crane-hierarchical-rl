#!/usr/bin/env python3
"""Training-rollout clearing curves for the RL ablation arms (v2.1 runs only).

Metric is Episode/pile_clearing_pct_final, the end-of-episode clearing of the
STOCHASTIC training policy: unlike return it is comparable across reward
definitions (the MN/AN arms normalize the reward, so their returns live on
different scales) and across action heads (the Gaussian control). Argmax rows
remain the citable evidence; these curves show the training dynamics.

Panels:
  (a) Gaussian over coordinates (GS) against the categorical from-scratch
      seed band (SCC0/43/44) - the same-environment form of the from-scratch
      exploration comparison.
  (b) Reward structure: multiplicative+normalized (MN), additive+normalized
      (AN), no quality terms (NQ), against the cost-free categorical control
      (SCC0, same seed).

    python3 scripts/plots/plot_rl_ablation_curves.py [--out ...]
"""

import argparse

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
TAG = "Episode/pile_clearing_pct_final"
SMOOTH = 5

SCC_BAND = [
    f"{F}/SCC0/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_03-18-17",
    f"{F}/SCC43/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-15",
    f"{F}/SCC44/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-14",
]
GS = f"{F}/GS/crane_pointcloud_gaze_cossin_scratch_v0/2026-08-13_02-25-59"
REWARD_ARMS = [
    ("multiplicative, normalized", f"{F}/MN/crane_pointcloud_gaze_scoring_scratch_mn_v0/2026-08-12_03-18-20", "#dd8452"),
    ("additive, normalized",       f"{F}/AN/crane_pointcloud_gaze_scoring_scratch_an_v0/2026-08-12_03-18-15", "#8172b2"),
    ("no quality terms",           f"{F}/NQ/crane_pointcloud_gaze_scoring_scratch_nq_v0/2026-08-12_05-44-38", "#937860"),
]


def load(d):
    import glob as g
    cands = g.glob(d + "*") if not g.os.path.isdir(d) else [d]
    ea = EventAccumulator(cands[0]); ea.Reload()
    v = ea.Scalars(TAG)
    s = np.array([x.step for x in v], float)
    y = np.array([x.value for x in v], float)
    # expanding-window rolling mean: the first points average what exists so far,
    # so the curve starts at the first logged iteration instead of SMOOTH-1 in
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/thesis/figures/rl_ablation_curves.pdf")
    a = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)

    runs = [load(d) for d in SCC_BAND]
    hi = min(r[0][-1] for r in runs)
    grid = np.linspace(max(r[0][0] for r in runs), hi, 150)
    Y = np.stack([np.interp(grid, s, y) for s, y in runs])
    ax1.plot(grid, Y.mean(0), color="#c44e52", lw=1.5, label="categorical over points (N=3)")
    ax1.fill_between(grid, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                     color="#c44e52", alpha=0.18, lw=0)
    s, y = load(GS)
    ax1.plot(s, y, color="#555555", lw=1.5, ls="--", label="Gaussian over coordinates (N=1)")
    ax1.set_ylabel("training-rollout clearing [%]")
    ax1.set_xlabel("environment steps (grasp cycles)")
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(lw=0.4, alpha=0.4)
    ax1.set_title("(a) from-scratch exploration", fontsize=9)

    s0, y0 = load(SCC_BAND[0])
    ax2.plot(s0, y0, color="#c44e52", lw=1.5, label="multiplicative, unnormalized (control)")
    for name, d, c in REWARD_ARMS:
        s, y = load(d)
        ax2.plot(s, y, color=c, lw=1.3, label=name)
    ax2.set_xlabel("environment steps (grasp cycles)")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(lw=0.4, alpha=0.4)
    ax2.set_title("(b) reward structure, from scratch", fontsize=9)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
