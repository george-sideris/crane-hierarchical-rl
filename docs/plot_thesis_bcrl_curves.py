#!/usr/bin/env python3
"""Thesis BC->RL (P3v2 scoring) training curves: efficiency panels.

Same visual language as the IROS figure (docs/plot_reward_curves.py): serif, 3.5in
column, bold rolling mean over faint raw, diamond checkpoint markers. The story is
different though. In the paper, stability climbed during training. Here the BC init
already saturates the per-grasp metrics (87% grasp success, 0.93 stability, ~100%
final clearing), so the top panels are expected to stay FLAT at ceiling - that
flatness is the result (fine-tuning does not erode grasping) - while the bottom
panels carry the improvement: cycles-to-clear coming down from the BC init's 18.8
and full-clear rate closing the last ~10%.

Those two bottom metrics do not exist as TB scalars in this run (the env logged
only step-averaged values until commit 05902e3, which lands in the NEXT run), so
they are reconstructed from the training log's CYCLE lines: an episode completes
when an env's cycle counter resets, and its final `remaining=` value is the
authoritative end-of-episode count.

Data lives in logs/cloud_p3v2_40env/ (event file + train_progress.txt). Refresh
from the running pod and re-plot:

  SSH='ssh -i ~/.ssh/id_ed25519 -p 22120 root@194.68.245.4'
  RUN=/data/crane_testbed/logs/rsl_rl/crane_pointcloud_gaze_scoring_ppo_v2/2026-08-09_22-11-23
  $SSH "cd $RUN && tar czf - events.out.tfevents.*" | tar xzf - -C logs/cloud_p3v2_40env/
  $SSH 'grep -aE "Learning iteration [0-9]+/|\[env[0-9]+\] CYCLE .*remaining=" \
      /data/crane_testbed/logs/p3v2_train.log' > logs/cloud_p3v2_40env/train_progress.txt
  python3 docs/plot_thesis_bcrl_curves.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "logs", "cloud_p3v2_40env")

COLOR = "#d62728"          # BC->RL red, same as the IROS figure
BC_REF_COLOR = "#555555"   # BC-init reference lines
NUM_LOGS = 200
SMOOTH_WINDOW = 20         # TB scalars (per-iteration), as in the paper
EP_SMOOTH = 30             # completed episodes (sparser, noisier)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})


def smooth(values, window):
    if window <= 1 or len(values) < 2:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_scalar(tag):
    """Scalar series with x converted from rsl_rl's total-timestep steps to iterations."""
    ea = EventAccumulator(DATA_DIR)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events])
    if len(steps) > 1:
        stride = float(np.median(np.diff(steps)))  # = num_envs * num_steps_per_env
        steps = steps / stride
    elif len(steps) == 1 and steps[0] > 0:
        steps = np.array([1.0])
    return steps, values


def parse_episodes(path):
    """Completed episodes from the training log.

    Yields (iteration_at_completion, final_clearing_pct, episode_cycles). An episode
    boundary is an env's cycle counter going backwards; the previous line's
    `remaining=` is that episode's final state. Iteration = banners seen so far.
    """
    banner = re.compile(r"Learning iteration (\d+)/")
    cyc = re.compile(r"\[env(\d+)\] CYCLE (\d+)/\d+.*remaining=(\d+)")
    it = 0
    prev = {}  # env -> (cycle, remaining)
    eps = []
    with open(path) as f:
        for line in f:
            m = banner.search(line)
            if m:
                it = int(m.group(1)) + 1  # banner N prints after iteration N collected
                continue
            m = cyc.search(line)
            if not m:
                continue
            env, c, rem = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if env in prev and c < prev[env][0]:
                fin_rem = prev[env][1]
                clearing = 100.0 * (1.0 - fin_rem / NUM_LOGS)
                eps.append((it, clearing, prev[env][0]))
            prev[env] = (c, rem)
    return eps


def ep_series(eps, value_fn):
    """(iteration positions, rolling-mean values) over completed episodes in order."""
    xs = np.array([e[0] for e in eps], dtype=float)
    vs = np.array([value_fn(e) for e in eps], dtype=float)
    return xs, vs, smooth(vs, EP_SMOOTH)


def style_ax(ax, ylabel, x_max, xlabel=None):
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_xlim(0, x_max)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bc_ref(ax, y, label=None):
    ax.axhline(y, color=BC_REF_COLOR, linestyle="--", linewidth=0.6, alpha=0.8)
    if label:
        ax.annotate(label, xy=(0.98, y), xycoords=("axes fraction", "data"),
                    fontsize=6, color=BC_REF_COLOR, ha="right", va="bottom")


def main():
    steps_ret, ret = load_scalar("Episode/episode_return")
    steps_gs, gs = load_scalar("Episode/grasp_success_rate")
    steps_st, st = load_scalar("Episode/stability")
    eps = parse_episodes(os.path.join(DATA_DIR, "train_progress.txt"))
    print(f"TB points: {len(ret)}   completed episodes: {len(eps)}")

    x_max = max(50, int(max(steps_ret.max() if len(steps_ret) else 0,
                            max((e[0] for e in eps), default=0)) * 1.15))

    # BC-init references: parser values over the first iterations (pure/near-pure BC)
    bc_eps = [e for e in eps if e[0] <= 2] or eps
    bc_cycles = float(np.mean([e[2] for e in bc_eps]))
    bc_fullclear = 100.0 * sum(1 for e in bc_eps if e[1] >= 99.999) / len(bc_eps)

    fig = plt.figure(figsize=(3.5, 4.2))
    gs_grid = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.55, wspace=0.85)

    ax_main = fig.add_subplot(gs_grid[0, :])
    ax_main.plot(steps_ret, ret, color=COLOR, alpha=0.15, linewidth=0.3)
    ax_main.plot(steps_ret, smooth(ret, SMOOTH_WINDOW), color=COLOR, linewidth=0.8)
    style_ax(ax_main, "Mean Episode Return", x_max)

    # Per-grasp metrics: expected FLAT at ceiling (BC quality preserved)
    ax = fig.add_subplot(gs_grid[1, 0])
    ax.plot(steps_gs, smooth(gs, SMOOTH_WINDOW), color=COLOR, linewidth=0.8)
    if len(gs):
        bc_ref(ax, gs[0])
    style_ax(ax, "Grasp Success (%)", x_max)

    ax = fig.add_subplot(gs_grid[1, 1])
    ax.plot(steps_st, smooth(st, SMOOTH_WINDOW), color=COLOR, linewidth=0.8)
    if len(st):
        bc_ref(ax, st[0])
    style_ax(ax, "Stability", x_max)

    # Efficiency metrics: the improvement axis (reconstructed, end-of-episode)
    ax = fig.add_subplot(gs_grid[2, 0])
    if eps:
        xs, vs, sm = ep_series(eps, lambda e: 100.0 if e[1] >= 99.999 else 0.0)
        ax.plot(xs, sm, color=COLOR, linewidth=0.8)
        bc_ref(ax, bc_fullclear, "BC init")
        ax.set_ylim(0, 105)
    style_ax(ax, "Full Clears (%)", x_max)

    ax = fig.add_subplot(gs_grid[2, 1])
    if eps:
        xs, vs, sm = ep_series(eps, lambda e: e[2])
        ax.plot(xs, vs, color=COLOR, alpha=0.15, linewidth=0.3)
        ax.plot(xs, sm, color=COLOR, linewidth=0.8)
        bc_ref(ax, bc_cycles, "BC init")
    style_ax(ax, "Cycles to Clear", x_max)

    fig.align_ylabels(fig.axes)
    fig.text(0.5, -0.01, "PPO Iteration", ha="center", fontsize=9)

    out = os.path.join(OUT_DIR, "thesis_bcrl_curves.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    print(f"BC-init refs: cycles-to-clear={bc_cycles:.1f}, full-clear={bc_fullclear:.0f}%")


if __name__ == "__main__":
    main()
