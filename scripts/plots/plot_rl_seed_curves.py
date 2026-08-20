#!/usr/bin/env python3
"""Seed-banded RL training curves (and, once the battery lands, argmax overlays).

Bands are mean +/- std over the three locked-recipe seeds per family, v2.1 runs
only (arms relaunched after the 2026-08-12 per-env reset fixes; earlier dirs in
the same fleet_tb folders are pre-fix arms and are excluded by the DIR pins
below). x-axis is environment steps (grasp cycles). Smoothing: rolling mean
over 5 iterations, applied before banding and declared in the caption.

The deployed BC->RL checkpoint (g100) trained in the pre-fix environment, so
its return curve is not comparable and is deliberately NOT overlaid; its argmax
row appears in the results table instead.

    python3 scripts/plots/plot_rl_seed_curves.py \
        [--argmax_tsv logs/sim_eval/battery_rows.tsv] \
        [--out docs/thesis/figures/rl_training_curves.pdf]
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

FLEET = "logs/fleet_tb"
# v2.1 runs only (post reset-fix relaunch, 2026-08-12)
FAMILIES = {
    "BC$\\to$RL fine-tune": {
        "color": "#4c72b0",
        "runs": [
            f"{FLEET}/S42/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_03-25-25",
            f"{FLEET}/S43/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-25-59",
            f"{FLEET}/S44/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-09-19",
        ],
    },
    "RL from scratch": {
        "color": "#c44e52",
        "runs": [
            f"{FLEET}/SCC0/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_03-18-17",
            f"{FLEET}/SCC43/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-15",
            f"{FLEET}/SCC44/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-14",
        ],
    },
}
TAG = "Train/mean_reward"
SMOOTH = 5   # iterations


def load_run(d):
    ea = EventAccumulator(d); ea.Reload()
    tag = TAG if TAG in ea.Tags()["scalars"] else None
    if tag is None:
        cands = [t for t in ea.Tags()["scalars"] if t.endswith("mean_reward")]
        assert cands, f"no mean_reward tag in {d}: {ea.Tags()['scalars'][:6]}"
        tag = cands[0]
    v = ea.Scalars(tag)
    s = np.array([x.step for x in v], float)
    y = np.array([x.value for x in v], float)
    # expanding-window rolling mean: the first points average what exists so far,
    # so the curve starts at the first logged iteration instead of SMOOTH-1 in
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--argmax_tsv", default="",
                    help="optional TSV: family<TAB>seed<TAB>env_steps<TAB>full_pct "
                         "(wave-protocol argmax rows; plotted as a second panel)")
    ap.add_argument("--out", default="docs/thesis/figures/rl_training_curves.pdf")
    a = ap.parse_args()

    two = bool(a.argmax_tsv) and os.path.exists(a.argmax_tsv)
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=(7.0 if two else 4.6, 2.7), squeeze=False)
    ax = axes[0][0]

    for name, fam in FAMILIES.items():
        runs = [load_run(d) for d in fam["runs"]]
        hi = min(r[0][-1] for r in runs)
        grid = np.linspace(min(r[0][0] for r in runs), hi, 200)
        Y = np.stack([np.interp(grid, s, y) for s, y in runs])
        ax.plot(grid, Y.mean(0), color=fam["color"], lw=1.5,
                label=f"{name} (N={len(runs)})")
        ax.fill_between(grid, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                        color=fam["color"], alpha=0.18, lw=0)
    ax.set_xlabel("environment steps (grasp cycles)")
    ax.set_ylabel("training return")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(lw=0.4, alpha=0.4)

    if two:
        ax2 = axes[0][1]
        import csv
        rows = list(csv.reader(open(a.argmax_tsv), delimiter="\t"))
        for name, fam in FAMILIES.items():
            pts = [(float(r[2]), float(r[3]), r[1]) for r in rows if r[0] == name]
            seeds = sorted(set(p[2] for p in pts))
            for i, sd in enumerate(seeds):
                sp = sorted((x, y) for x, y, s in pts if s == sd)
                ax2.plot([p[0] for p in sp], [p[1] for p in sp],
                         color=fam["color"], lw=1.1, alpha=0.5 + 0.2 * i,
                         marker="o", ms=2.6,
                         label=f"{name} seed {sd}" if i == 0 else None)
        ax2.set_xlabel("environment steps (grasp cycles)")
        ax2.set_ylabel("argmax full-clear rate [%]")
        ax2.legend(frameon=False, loc="lower right")
        ax2.grid(lw=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
