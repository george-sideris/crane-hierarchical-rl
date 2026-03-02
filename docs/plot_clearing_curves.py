#!/usr/bin/env python3
"""Plot clearing progression curves for IROS 2026 paper (Fig. clearing_curves).

Reads eval_metrics JSON files containing per-episode clearing_curves data.
Plots fraction of pile cleared vs. grasp attempt number, mean ± shaded std.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Each entry: label -> path to eval_metrics JSON with clearing_curves
RUNS = {
    "Heuristic": {
        "json": os.path.join(REPO_ROOT, "logs/heuristic_baseline/eval_metrics_20260301_160802.json"),
        "color": "#1f77b4",   # blue
    },
    "RL": {
        "json": os.path.join(REPO_ROOT, "crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20/eval_metrics_20260301_150413.json"),
        "color": "#ff7f0e",   # orange
    },
    "BC": {
        "json": os.path.join(REPO_ROOT, "crane_testbed/results/BC_Raw_PCD/eval_metrics_20260301_224854.json"),
        "color": "#2ca02c",   # green
    },
    r"BC$\to$RL": {
        "json": os.path.join(REPO_ROOT, "crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/eval_metrics_20260301_211210.json"),
        "color": "#d62728",   # red
    },
}

MAX_GRASPS = 30  # episode horizon

# ── Style ─────────────────────────────────────────────────────────────
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


def load_clearing_curves(json_path):
    """Load clearing curves from eval metrics JSON.

    Returns a 2D array of shape (num_episodes, MAX_GRASPS) with clearing %
    at each grasp attempt. Shorter episodes are forward-filled with their
    final value (pile was fully cleared).
    """
    with open(json_path) as f:
        data = json.load(f)

    curves = data["per_episode"]["clearing_curves"]
    # Pad all curves to MAX_GRASPS length (forward-fill with last value)
    padded = np.zeros((len(curves), MAX_GRASPS))
    for i, curve in enumerate(curves):
        n = min(len(curve), MAX_GRASPS)
        padded[i, :n] = curve[:n]
        if n < MAX_GRASPS:
            padded[i, n:] = curve[-1] if curve else 0.0
    return padded


def plot():
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(1, MAX_GRASPS + 1)

    for label, cfg in RUNS.items():
        if not os.path.exists(cfg["json"]):
            print(f"  Skipping {label}: {cfg['json']} not found")
            continue

        curves = load_clearing_curves(cfg["json"])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        ax.plot(x, mean, color=cfg["color"], linewidth=0.8, label=label)
        ax.fill_between(x, mean - std, mean + std, color=cfg["color"], alpha=0.15)

    ax.set_xlabel("Grasp Attempt")
    ax.set_ylabel("Pile Cleared (%)")
    ax.set_xlim(1, MAX_GRASPS)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="none",
              fontsize=6, handlelength=1.2, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)

    out = os.path.join(OUT_DIR, "clearing_curves.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading clearing curves...")
    plot()
    print("Done.")
