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
        "selected": 100,
        "runs": [
            f"{FLEET}/S42/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_03-25-25",
            f"{FLEET}/S43/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-25-59",
            f"{FLEET}/S44/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-09-19",
        ],
    },
    "RL from scratch": {
        "color": "#c44e52",
        "selected": 30,
        "runs": [
            f"{FLEET}/SCC0/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_03-18-17",
            f"{FLEET}/SCC43/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-15",
            f"{FLEET}/SCC44/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-14",
        ],
    },
}
TAG = "Train/mean_reward"
SMOOTH = 5   # iterations
STEPS_PER_ITER = 320   # 40 envs x 8 grasp cycles


def load_run(d, tag=TAG):
    ea = EventAccumulator(d); ea.Reload()
    tag = tag if tag in ea.Tags()["scalars"] else None
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
                         "(wave-protocol argmax rows; plotted as a separate figure)")
    ap.add_argument("--out", default="docs/thesis/figures/rl_training_curves.pdf")
    a = ap.parse_args()

    PANELS = [
        ("Train/mean_reward",     "training return"),
        ("Episode/throughput",    "throughput $n$ [logs/grasp]"),
        ("Episode/stability",     "stability $\\varsigma$"),
        ("Episode/alignment",     "alignment $\\alpha$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    for (tag, ylab), ax in zip(PANELS, axes.flat):
        for name, fam in FAMILIES.items():
            runs = [load_run(d, tag) for d in fam["runs"]]
            # Seeds ran to different lengths for scheduling reasons, so do not clip the
            # whole family to its shortest member: average over the seeds that exist at
            # each step and let the band thin out, marking where the seed count drops.
            lo = max(r[0][0] for r in runs)
            hi = max(r[0][-1] for r in runs)
            grid = np.linspace(lo, hi, 400)
            Y = np.full((len(runs), grid.size), np.nan)
            for i, (sx, sy) in enumerate(runs):
                m = grid <= sx[-1]
                Y[i, m] = np.interp(grid[m], sx, sy)
            n = np.sum(~np.isnan(Y), axis=0)
            mean = np.nanmean(Y, axis=0)
            sd = np.nanstd(Y, axis=0)
            full = n == len(runs)
            ax.plot(grid[full], mean[full], color=fam["color"], lw=1.5,
                    label=f"{name} (N={len(runs)})")
            ax.fill_between(grid[full], (mean - sd)[full], (mean + sd)[full],
                            color=fam["color"], alpha=0.18, lw=0)
            if (~full).any():          # continue with the seeds that ran longer
                part = ~full & (n > 0)
                ax.plot(grid[part], mean[part], color=fam["color"], lw=1.1, ls="--")
                two = part & (n >= 2)
                if two.any():
                    ax.fill_between(grid[two], (mean - sd)[two], (mean + sd)[two],
                                    color=fam["color"], alpha=0.10, lw=0)
                ax.axvline(grid[full][-1] if full.any() else lo, color=fam["color"],
                           lw=0.6, ls=":", alpha=0.55)
            if fam.get("selected"):    # the checkpoint carried forward
                sel = fam["selected"] * STEPS_PER_ITER
                j = int(np.argmin(np.abs(grid - sel)))
                ax.plot([grid[j]], [mean[j]], marker="D", ms=4.5, color=fam["color"],
                        mec="black", mew=0.6, ls="none", zorder=5)
        ax.set_ylabel(ylab)
        ax.grid(lw=0.4, alpha=0.4)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (grasp cycles)")
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
