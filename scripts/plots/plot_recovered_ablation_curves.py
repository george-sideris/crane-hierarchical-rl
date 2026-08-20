#!/usr/bin/env python3
"""Training curves of the architecture ablations, from the rescued runs.

The original symmetric-critic and unfrozen-encoder arms lost their TensorBoard
files when their pods were terminated (only checkpoints were rescued at the
time). These runs are their reruns, at the same seed and configuration, stopped
early when the campaign budget ran out; the locked recipe is drawn from the
campaign archive for reference.

Curves are training-rollout quantities under the stochastic policy: they show
the optimizer's trajectory, not deployed quality, which the argmax evaluations
measure separately.

    python3 scripts/plots/plot_recovered_ablation_curves.py [--out ...]
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

R = "logs/fleet_rescue/logs/rsl_rl"
F = "logs/fleet_tb"
RUNS = [
    ("frozen encoder, asymmetric critic",
     f"{F}/S42/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_03-25-25", "#4c72b0"),
    ("symmetric critic",
     f"{R}/crane_pointcloud_gaze_scoring_ppo_v2_cc0_symcritic/2026-08-20_10-35-12", "#dd8452"),
    ("encoder not frozen",
     f"{R}/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-20_10-36-48", "#c44e52"),
]
PANELS = [("Train/mean_reward", "training return"),
          ("Episode/pile_clearing_pct_final", "clearing at episode end [%]"),
          ("Episode/stability", "stability $\\varsigma$"),
          ("Episode/throughput", "throughput $n$ [logs/grasp]")]
SMOOTH = 5


def load(d, tag):
    ea = EventAccumulator(d); ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    v = ea.Scalars(tag)
    s = np.array([x.step for x in v], float)
    y = np.array([x.value for x in v], float)
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/rl_ablation_recovered_curves.pdf")
    a = ap.parse_args()
    # the ablation reruns were cut short, so compare everything on their common support
    ends = []
    for _, d, _ in RUNS[1:]:
        s, _ = load(d, "Train/mean_reward")
        if s is not None:
            ends.append(s[-1])
    hi = min(ends) if ends else None

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    for (tag, ylab), ax in zip(PANELS, axes.flat):
        for name, d, c in RUNS:
            s, y = load(d, tag)
            if s is None:
                continue
            m = s <= hi if hi else np.ones_like(s, bool)
            ax.plot(s[m], y[m], lw=1.4, color=c, label=name)
        ax.set_ylabel(ylab)
        ax.grid(lw=0.4, alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (grasp cycles)")
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out, f"(common support to {hi:.0f} steps)" if hi else "")


if __name__ == "__main__":
    main()
