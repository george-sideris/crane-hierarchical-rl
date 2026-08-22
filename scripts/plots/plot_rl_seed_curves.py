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

# Each seed is ONE continuous run. The 2026-08-20/21 "extension" fragments are deliberately
# NOT stitched on: `agent.resume=True` was overwritten to False by update_rsl_rl_cfg (--resume
# is store_true/default False, and the assignment is guarded by `is not None`), so every
# extension started from a random policy instead of continuing. Concatenating them would
# splice independent runs into one curve. The bug is fixed in scripts/rsl_rl/train.py, but
# these runs predate the fix, so each arm is reported at its true continuous length.
FAMILIES = {
    "BC$\\to$RL fine-tune": {
        "color": "#4c72b0",
        "selected": 100,
        "runs": [
            [f"{FLEET}/S42/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_03-25-25"],
            [f"{FLEET}/S43/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-25-59"],
            [f"{FLEET}/S44/crane_pointcloud_gaze_scoring_ppo_v2_cc0/2026-08-12_05-09-19"],
        ],
    },
    "RL from scratch": {
        "color": "#c44e52",
        "selected": 30,
        "runs": [
            [f"{FLEET}/SCC0/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_03-18-17"],
            # SCC43 is NOT in this band: it starts at 9.65 logs/grasp and 97% clearing while
            # every verified from-scratch run starts near 4.5 and 57-62%, and its initial value
            # loss is 775 against 63-67. Its actor was warm-started, so it is not a from-scratch
            # seed. F_scratch is used instead: same recipe, and it starts where the others do.
            [f"{FLEET}/F_scratch/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-39-21"],
            [f"{FLEET}/SCC44/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-14"],
        ],
    },
}
TAG = "Train/mean_reward"
SMOOTH = 5   # iterations
# Iterations discarded after each resume. The logged means are running averages over
# recently finished episodes, so the first iterations of any run - including a resumed
# one - average over episodes that have only just started and read far below the policy's
# actual level (throughput at a restart reads ~4 logs/grasp against a steady ~10). The
# discard is declared in the caption; resume boundaries are marked on the axes.
WARMUP = 5
STEPS_PER_ITER = 320   # 40 envs x 8 grasp cycles


def _raw(d, tag):
    """Scalars of one run directory, without smoothing."""
    ea = EventAccumulator(d); ea.Reload()
    tags = ea.Tags()["scalars"]
    if tag not in tags:
        cands = [t for t in tags if t.endswith(tag.split("/")[-1])]
        if not cands:
            return np.array([]), np.array([])
        tag = cands[0]
    v = ea.Scalars(tag)
    return np.array([x.step for x in v], float), np.array([x.value for x in v], float)


def load_run(chain, tag=TAG):
    """One seed as a cumulative curve over its resume chain.

    Each fragment restarts its step counter, so fragment k is shifted by the total
    environment steps of fragments 0..k-1. Smoothing is applied once, to the joined
    curve, so a resume boundary is not treated as the start of a new series.
    Returns (steps, smoothed values, resume boundary steps).
    """
    if isinstance(chain, str):
        chain = [chain]
    S, Y, bounds, off = [], [], [], 0.0
    for j, d in enumerate(chain):
        if not os.path.isdir(d):
            continue
        s, y = _raw(d, tag)
        if s.size == 0:
            continue
        if j > 0:
            bounds.append(off)
            s, y = s[WARMUP:], y[WARMUP:]     # drop the post-restart transient
            if s.size == 0:
                continue
        S.append(s + off); Y.append(y)
        off += float(s[-1])
    if not S:
        return np.array([0.0]), np.array([np.nan]), []
    s = np.concatenate(S); y = np.concatenate(Y)
    o = np.argsort(s); s, y = s[o], y[o]
    # expanding-window rolling mean: the first points average what exists so far,
    # so the curve starts at the first logged iteration instead of SMOOTH-1 in
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--argmax_tsv", default="",
                    help="optional TSV: family<TAB>seed<TAB>env_steps<TAB>full_pct "
                         "(wave-protocol argmax rows; plotted as a separate figure)")
    ap.add_argument("--out", default="docs/thesis/figures/rl_training_curves.pdf")
    ap.add_argument("--xmax", type=float, default=49920,
                    help="common x limit in environment steps (156 iterations x 320); "
                         "curves are cut here so every family is compared over the same "
                         "budget and no segment is a single surviving seed")
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
            loaded = [load_run(d, tag) for d in fam["runs"]]
            runs = [(a, b) for a, b, _ in loaded]
            resumes = sorted({round(x) for _, _, bs in loaded for x in bs})
            # Seeds ran to different lengths for scheduling reasons, so do not clip the
            # whole family to its shortest member: average over the seeds that exist at
            # each step and let the band thin out, marking where the seed count drops.
            lo = max(r[0][0] for r in runs)
            hi = max(r[0][-1] for r in runs)
            if a.xmax:
                hi = min(hi, a.xmax)
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
            if (~full).any() and full.any():
                # Past the point where a seed runs out, do not keep averaging: the mean
                # would step every time the membership changes, and that step reads as a
                # feature of training rather than of the seed count. Draw the seeds that
                # actually continue, each as its own thin dashed line, starting from the
                # end of the band so the two connect.
                edge = grid[full][-1]
                for sx, sy in runs:
                    if sx[-1] <= edge:
                        continue
                    m = sx >= edge
                    ax.plot(sx[m], sy[m], color=fam["color"], lw=0.9, ls="--", alpha=0.85)
                ax.axvline(edge, color=fam["color"], lw=0.6, ls=":", alpha=0.55)
            if fam.get("selected"):    # the checkpoint carried forward
                sel = fam["selected"] * STEPS_PER_ITER
                j = int(np.argmin(np.abs(grid - sel)))
                ax.plot([grid[j]], [mean[j]], marker="D", ms=4.5, color=fam["color"],
                        mec="black", mew=0.6, ls="none", zorder=5)
            for b in resumes:      # where training was resumed after a container restart
                ax.axvline(b, color="0.55", lw=0.5, ls="-", alpha=0.35, zorder=0)
        ax.set_ylabel(ylab)
        ax.grid(lw=0.4, alpha=0.4)
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    for ax in axes.flat:
        if a.xmax:
            ax.set_xlim(0, a.xmax)
    for ax in axes[1]:
        ax.set_xlabel("environment steps (grasp cycles)")
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
