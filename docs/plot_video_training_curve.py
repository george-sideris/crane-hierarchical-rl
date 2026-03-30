#!/usr/bin/env python3
"""Plot training reward curves for IROS 2026 supplementary video.

Single-panel episode return plot with larger fonts/lines than the paper figure.
Output: crane_testbed/media/training_curves.png (300 DPI)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_PATH = os.path.join(REPO_ROOT, "crane_testbed", "media", "training_curves.png")

RUNS = {
    "RL (Pose)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Pose/2026-02-23_20-19-49"),
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "RL (Seg PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21"),
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "RL (Raw PCD)": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20"),
        "color": "#2ca02c",
        "linestyle": "-",
    },
    r"BC $\rightarrow$ RL": {
        "logdir": os.path.join(REPO_ROOT, "crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30"),
        "color": "#d62728",
        "linestyle": "-",
    },
}

MAIN_TAG = "Episode/episode_return"
SMOOTH_WINDOW = 20
X_MAX = 800

# ── Video-friendly style (larger than paper) ──────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    print("Loading TensorBoard data...")
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, cfg in RUNS.items():
        steps, values = load_scalar(cfg["logdir"], MAIN_TAG)
        mask = steps <= X_MAX
        steps, values = steps[mask], values[mask]
        smoothed = smooth(values, SMOOTH_WINDOW)

        # Raw trace (faint)
        ax.plot(steps, values, color=cfg["color"], alpha=0.12, linewidth=0.5)
        # Smoothed curve
        ax.plot(steps, smoothed, color=cfg["color"], linewidth=2.5,
                linestyle=cfg["linestyle"], label=label)

    ax.set_xlabel("PPO Iteration")
    ax.set_ylabel("Mean Episode Return")
    ax.set_xlim(0, X_MAX)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none")

    fig.savefig(OUT_PATH)
    print(f"Saved {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
