#!/usr/bin/env python3
"""Plot RL training reward curves for IROS 2026 paper.

Reads TensorBoard event files directly (no CSV export needed).
Generates two versions:
  - reward_curves_full.pdf   : main panel + 4 metric subplots (figure*)
  - reward_curves_minimal.pdf: single episode_return plot (figure)
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths (relative to IsaacLab repo root) ────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    "RL (Pose)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Pose/2026-02-23_20-19-49"),
        "ckpt_step": 940,
        "color": "#1f77b4",  # blue
        "linestyle": "-",
    },
    "RL (Seg PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21"),
        "ckpt_step": 880,
        "color": "#ff7f0e",  # orange
        "linestyle": "-",
    },
    "RL (Raw PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20"),
        "ckpt_step": 650,
        "color": "#2ca02c",  # green
        "linestyle": "-",
    },
    r"BC $\rightarrow$ RL": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30"),
        "ckpt_step": 350,
        "color": "#d62728",  # red
        "linestyle": "-",
    },
}

# Scalar tags to plot
MAIN_TAG = "Episode/episode_return"
SUBPLOT_TAGS = [
    ("Episode/throughput", "Throughput"),
    ("Episode/alignment", "Alignment"),
    ("Episode/stability", "Stability"),
    ("Episode/grasp_success_rate", "Grasp Success"),
]

SMOOTH_WINDOW = 20
X_MAX = 1000


# ── Style ──────────────────────────────────────────────────────────────
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


# ── Data loading ───────────────────────────────────────────────────────
def load_scalar(logdir, tag):
    """Load a scalar time series from a TensorBoard logdir."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window):
    """Rolling average with same-length output (causal, no future leak)."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_all(tag):
    """Load and return {label: (steps, raw, smoothed)} for all runs."""
    data = {}
    for label, cfg in RUNS.items():
        steps, values = load_scalar(cfg["logdir"], tag)
        mask = steps <= X_MAX
        steps, values = steps[mask], values[mask]
        smoothed = smooth(values, SMOOTH_WINDOW)
        data[label] = (steps, values, smoothed)
    return data


# ── Plotting helpers ───────────────────────────────────────────────────
def plot_curve(ax, label, steps, raw, smoothed, cfg, show_raw=True, show_ckpt=True):
    """Plot a single curve with optional raw background and checkpoint marker."""
    if show_raw:
        ax.plot(steps, raw, color=cfg["color"], alpha=0.15, linewidth=0.3)
    ax.plot(steps, smoothed, color=cfg["color"], linewidth=0.8,
            linestyle=cfg["linestyle"], label=label)
    if show_ckpt:
        ckpt = cfg["ckpt_step"]
        idx = np.argmin(np.abs(steps - ckpt))
        marker = "D"
        ms = 3.5
        ax.plot(steps[idx], smoothed[idx], marker=marker, color=cfg["color"],
                markersize=ms, markeredgecolor="black", markeredgewidth=0.3,
                zorder=5)


def style_ax(ax, ylabel, xlabel=None, legend=False, legend_loc="best"):
    """Apply common axis styling."""
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_xlim(0, X_MAX)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc=legend_loc, framealpha=0.9, edgecolor="none",
                  fontsize=6, handlelength=1.2, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.25)


# ── Version A: Full (main + 2×2 subplots) ─────────────────────────────
def plot_full():
    fig = plt.figure(figsize=(3.5, 4.2))  # single-column IEEE width
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.55, wspace=0.85)

    # Main panel spans top row (both columns)
    ax_main = fig.add_subplot(gs[0, :])
    main_data = load_all(MAIN_TAG)
    for label, cfg in RUNS.items():
        steps, raw, smoothed = main_data[label]
        plot_curve(ax_main, label, steps, raw, smoothed, cfg)
    style_ax(ax_main, "Mean Episode Return", legend=True, legend_loc="lower right")

    # 2×2 subplots below
    left_axes = [ax_main]
    right_axes = []
    positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    for (row, col), (tag, title) in zip(positions, SUBPLOT_TAGS):
        ax = fig.add_subplot(gs[row, col])
        sub_data = load_all(tag)
        for label, cfg in RUNS.items():
            steps, raw, smoothed = sub_data[label]
            plot_curve(ax, label, steps, raw, smoothed, cfg,
                       show_raw=False, show_ckpt=False)
        style_ax(ax, title)
        if col == 0:
            left_axes.append(ax)
        else:
            right_axes.append(ax)

    # Align y-axis labels within each column
    fig.align_ylabels(left_axes)
    fig.align_ylabels(right_axes)

    # Shared x-label at the bottom
    fig.text(0.5, -0.01, "PPO Iteration", ha="center", fontsize=9)

    out = os.path.join(OUT_DIR, "reward_curves_full.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    # Also save as reward_curves.pdf (name referenced in paper)
    shutil.copy2(out, os.path.join(OUT_DIR, "reward_curves.pdf"))
    shutil.copy2(out.replace(".pdf", ".png"), os.path.join(OUT_DIR, "reward_curves.png"))
    print(f"Saved {out} (+ reward_curves.pdf)")
    plt.close(fig)


# ── Version B: Minimal (single plot) ──────────────────────────────────
def plot_minimal():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))  # single-column IEEE width
    main_data = load_all(MAIN_TAG)
    for label, cfg in RUNS.items():
        steps, raw, smoothed = main_data[label]
        plot_curve(ax, label, steps, raw, smoothed, cfg)
    style_ax(ax, "Mean Episode Return", xlabel="PPO Iteration", legend=True)

    out = os.path.join(OUT_DIR, "reward_curves_minimal.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading TensorBoard data...")
    plot_full()
    plot_minimal()
    print("Done.")
