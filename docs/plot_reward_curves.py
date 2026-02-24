#!/usr/bin/env python3
"""Plot RL training reward curves for IROS 2026 paper.

No smoothing — raw Train/mean_reward data plotted directly.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ── Load CSV data (Train/mean_reward) ─────────────────────────────────
pose_csv = pd.read_csv("rsl_rl_crane_full_cossin_mr_v0_2026-02-22_00-46-38.csv")
pcd_csv = pd.read_csv("rsl_rl_crane_pointcloud_cossin_mr_v0_2026-02-22_11-02-21(1).csv")
bcrl_csv = pd.read_csv("crane_pointcloud_cossin_v0_2026-02-23_02-03-45.csv")

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.4))  # single-column IEEE width

curves = [
    ("RL (pose)",                  pose_csv, "C0"),
    ("RL (PCD)",                   pcd_csv,  "C1"),
    ("BC $\\rightarrow$ RL (PCD)", bcrl_csv, "C2"),
]

for label, df, color in curves:
    x = df["Step"].values
    y = df["Value"].values
    ax.plot(x, y, color=color, linewidth=1.0, label=label)

ax.set_xlabel("Iteration")
ax.set_ylabel("Mean episode reward")
ax.legend(loc="lower right", framealpha=0.9)
ax.set_xlim(left=0)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("reward_curves.pdf")
fig.savefig("reward_curves.png")
print("Saved reward_curves.pdf and reward_curves.png")
