#!/usr/bin/env python3
"""Training curves for the gaze BC->RL runs (same layout as the IROS reward_curves figure).

All runs: task Isaac-Crane-PointCloud-Gaze-CosSin-Raw-MR-v0, seed 42, 4 envs, sigma_init
0.05, frozen encoder. Differences (from run-dir git snapshots + weight comparison):
  Jul 15 (red)  : mound BC init (bc_20260713_182049), critic max_logs_obs=64, 380 iters;
                  model_380 deployed, model_250 stability-peak SFT candidate (both marked).
  Jul 14 (blue) : SAME mound BC init and seed, critic max_logs_obs=32 -- too few to cover
                  a mound, critic value loss diverges -> deep collapse before recovery.
  Jul 09 (gray) : earlier pre-mound BC init (stability collapses, never recovers).
This runner logs Episode/* against total env steps (16 per PPO iteration at 4 envs x 4
steps_per_env), so steps are divided by 16 to plot against PPO iterations like the paper.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = os.path.join(REPO_ROOT, "crane_testbed/logs/rsl_rl/crane_pointcloud_gaze_cossin_raw_mr_v0")

STEPS_PER_ITER = 16

RUNS = {
    r"BC $\rightarrow$ RL (critic obs 64 logs, deployed)": {
        "logdir": os.path.join(LOG_ROOT, "2026-07-15_05-08-06"),
        "ckpt_steps": [250, 380],
        "color": "#d62728",  # red
        "linestyle": "-",
    },
    r"BC $\rightarrow$ RL (critic obs 32 logs)": {
        "logdir": os.path.join(LOG_ROOT, "2026-07-14_06-29-05"),
        "ckpt_steps": [],
        "color": "#1f77b4",  # blue
        "linestyle": "-",
    },
    r"BC $\rightarrow$ RL (weaker BC init)": {
        "logdir": os.path.join(LOG_ROOT, "2026-07-09_06-04-14"),
        "ckpt_steps": [],
        "color": "#7f7f7f",  # gray
        "linestyle": "-",
    },
}

MAIN_TAG = "Episode/episode_return"
SUBPLOT_TAGS = [
    ("Episode/throughput", "Throughput"),
    ("Episode/alignment", "Alignment"),
    ("Episode/stability", "Stability"),
    ("Episode/grasp_success_rate", "Grasp Success"),
]

SMOOTH_WINDOW = 20
X_MAX = 1000

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

_EAS = {}


def load_scalar(logdir, tag):
    if logdir not in _EAS:
        ea = EventAccumulator(logdir)
        ea.Reload()
        _EAS[logdir] = ea
    events = _EAS[logdir].Scalars(tag)
    steps = np.array([e.step for e in events]) / STEPS_PER_ITER
    values = np.array([e.value for e in events])
    mask = steps <= X_MAX
    return steps[mask], values[mask]


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_curve(ax, label, steps, raw, smoothed, cfg, show_raw=True, show_ckpt=True):
    if show_raw:
        ax.plot(steps, raw, color=cfg["color"], alpha=0.15, linewidth=0.3)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=0.8,
            linestyle=cfg["linestyle"], label=label)
    if show_ckpt:
        for ckpt in cfg["ckpt_steps"]:
            idx = np.argmin(np.abs(steps - ckpt))
            ax.plot(steps[idx], smoothed[idx], marker="D", color=cfg["color"],
                    markersize=3.5, markeredgecolor="black", markeredgewidth=0.3, zorder=5)


def style_ax(ax, ylabel, xlabel=None, legend=False):
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_xlim(0, X_MAX)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="lower right", framealpha=0.9, edgecolor="none",
                  fontsize=6, handlelength=1.2, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.25)


def main():
    fig = plt.figure(figsize=(3.5, 4.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.55, wspace=0.85)

    ax_main = fig.add_subplot(gs[0, :])
    for label, cfg in RUNS.items():
        steps, raw = load_scalar(cfg["logdir"], MAIN_TAG)
        plot_curve(ax_main, label, steps, raw, smooth(raw, SMOOTH_WINDOW), cfg)
    style_ax(ax_main, "Mean Episode Return", legend=True)

    left_axes = [ax_main]
    right_axes = []
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for (row, col), (tag, title) in zip(positions, SUBPLOT_TAGS):
        ax = fig.add_subplot(gs[row, col])
        for label, cfg in RUNS.items():
            s, v = load_scalar(cfg["logdir"], tag)
            plot_curve(ax, label, s, v, smooth(v, SMOOTH_WINDOW), cfg,
                       show_raw=False, show_ckpt=False)
        style_ax(ax, title)
        (left_axes if col == 0 else right_axes).append(ax)

    fig.align_ylabels(left_axes)
    fig.align_ylabels(right_axes)
    fig.text(0.5, -0.01, "PPO Iteration", ha="center", fontsize=9)

    out = os.path.join(OUT_DIR, "reward_curves_gaze_bcrl.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out} (+ .png)")
    plt.close(fig)


if __name__ == "__main__":
    main()
