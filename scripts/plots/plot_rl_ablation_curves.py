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
TAG = "Episode/pile_clearing_pct_final"
SMOOTH = 5
XMAX = 49920   # common x limit across every RL curve figure

# One continuous run per seed. The 2026-08-20/21 extension fragments are excluded: they
# began from a random policy because the resume flag never reached the config (see
# plot_rl_seed_curves.py), so they are separate runs, not continuations.
SCC_BAND = [
    [f"{F}/SCC0/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_03-18-17"],
    [f"{F}/SCC43/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-15"],
    [f"{F}/SCC44/crane_pointcloud_gaze_scoring_scratch_cc0_v0/2026-08-12_23-15-14"],
]
WARMUP = 5   # retained for the loader signature; no chain here has a resume boundary
GS = f"{F}/GS/crane_pointcloud_gaze_cossin_scratch_v0/2026-08-13_02-25-59"
REWARD_ARMS = [
    ("multiplicative, normalized", f"{F}/MN/crane_pointcloud_gaze_scoring_scratch_mn_v0/2026-08-12_03-18-20", "#dd8452"),
    ("additive, normalized",       f"{F}/AN/crane_pointcloud_gaze_scoring_scratch_an_v0/2026-08-12_03-18-15", "#8172b2"),
    ("no quality terms",           f"{F}/NQ/crane_pointcloud_gaze_scoring_scratch_nq_v0/2026-08-12_05-44-38", "#937860"),
]


def load(chain, tag=TAG):
    """One arm as a cumulative curve over its resume chain (a bare path is a chain of one).

    Fragments restart their step counter, so each is offset by the total steps of those
    before it. The first WARMUP iterations after a resume are dropped: the logged means
    are running averages over recently finished episodes, so straight after a restart
    they average over episodes that have only just begun and read far too low.
    """
    import glob as g
    if isinstance(chain, str):
        chain = [chain]
    S, Y, off = [], [], 0.0
    for j, d in enumerate(chain):
        cands = g.glob(d + "*") if not os.path.isdir(d) else [d]
        if not cands:
            continue
        ea = EventAccumulator(cands[0]); ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            continue
        v = ea.Scalars(tag)
        sx = np.array([x.step for x in v], float)
        sy = np.array([x.value for x in v], float)
        if j > 0:
            sx, sy = sx[WARMUP:], sy[WARMUP:]
            if sx.size == 0:
                continue
        S.append(sx + off); Y.append(sy)
        off += float(sx[-1])
    s = np.concatenate(S); y = np.concatenate(Y)
    o = np.argsort(s); s, y = s[o], y[o]
    # expanding-window rolling mean: the first points average what exists so far,
    # so the curve starts at the first logged iteration instead of SMOOTH-1 in
    ys = np.array([y[max(0, i - SMOOTH + 1):i + 1].mean() for i in range(len(y))])
    return s, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_a", default="docs/thesis/figures/rl_scratch_exploration.pdf")
    ap.add_argument("--out_b", default="docs/thesis/figures/rl_reward_structure.pdf")
    a = ap.parse_args()

    # (A) from-scratch exploration: Gaussian control against the categorical band
    fig, ax1 = plt.subplots(figsize=(4.6, 2.7))
    runs = [load(d) for d in SCC_BAND]
    hi = min(r[0][-1] for r in runs)
    grid = np.linspace(max(r[0][0] for r in runs), hi, 150)
    Y = np.stack([np.interp(grid, s, y) for s, y in runs])
    ax1.plot(grid, Y.mean(0), color="#c44e52", lw=1.5, label="categorical over points (N=3)")
    ax1.fill_between(grid, Y.mean(0) - Y.std(0), Y.mean(0) + Y.std(0),
                     color="#c44e52", alpha=0.18, lw=0)
    s, y = load(GS)
    common_a = min(hi, s[-1])
    m = s <= common_a
    ax1.plot(s[m], y[m], color="#555555", lw=1.5, ls="--", label="Gaussian over coordinates (N=1)")
    ax1.set_xlim(0, XMAX)
    ax1.set_ylabel("clearing at episode end [%]")
    ax1.set_xlabel("environment steps (grasp cycles)")
    ax1.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(a.out_a, dpi=200, bbox_inches="tight")
    print("wrote", a.out_a)

    # (B) reward structure: clearing plus each reward factor, one line per arm
    PANELS = [
        ("Episode/pile_clearing_pct_final", "clearing at episode end [%]"),
        ("Episode/throughput",              "throughput $n$ [logs/grasp]"),
        ("Episode/stability",               "stability $\\varsigma$"),
        ("Episode/alignment",               "alignment $\\alpha$"),
    ]
    ARMS = [("multiplicative, unnormalized (ours)", SCC_BAND[0], "#c44e52")] + REWARD_ARMS
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True)
    # arms stopped at different steps for fleet-logistics reasons; compare on common support
    common_b = min(load(d)[0][-1] for _, d, _ in ARMS)
    for (tag, ylab), ax in zip(PANELS, axes.flat):
        for name, d, c in ARMS:
            s, y = load(d, tag)
            m = s <= common_b
            ax.plot(s[m], y[m], lw=1.3, color=c, label=name)
        ax.set_ylabel(ylab)
        ax.grid(lw=0.4, alpha=0.4)
    for ax in axes.flat:
        ax.set_xlim(0, XMAX)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v else "0"))
    for ax in axes[1]:
        ax.set_xlabel("environment steps (grasp cycles)")
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(a.out_b, dpi=200, bbox_inches="tight")
    print("wrote", a.out_b)


if __name__ == "__main__":
    main()
