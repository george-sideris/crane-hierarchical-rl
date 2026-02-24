#!/usr/bin/env python3
"""Plot alignment and stability scores as a function of angle deviation."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

theta = np.linspace(0, 90, 500)  # degrees
theta_rad = np.deg2rad(theta)

# Alignment: a = |cos(theta)|^8
# theta = 0 means perfectly aligned, 90 means perpendicular
alignment = np.abs(np.cos(theta_rad)) ** 8

# Stability: s = max(0, cos(theta))^4
# theta = 0 means perfectly level, 90 means horizontal
stability = np.maximum(0, np.cos(theta_rad)) ** 4

# Also show unshaped versions for comparison
alignment_raw = np.abs(np.cos(theta_rad))
stability_raw = np.maximum(0, np.cos(theta_rad))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)
fig.subplots_adjust(wspace=0.35)

# Alignment
ax1.plot(theta, alignment_raw, "--", color="C0", linewidth=0.8, alpha=0.5, label=r"$|\cos\theta|$")
ax1.plot(theta, alignment, color="C0", linewidth=1.5, label=r"$|\cos\theta|^8$")
ax1.set_xlabel(r"Misalignment angle $\theta_a$ (deg)")
ax1.set_ylabel("Score")
ax1.set_title("Alignment $a$")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 1.05)
ax1.grid(True, alpha=0.3, linewidth=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Stability
ax2.plot(theta, stability_raw, "--", color="C1", linewidth=0.8, alpha=0.5, label=r"$\max(0,\cos\theta)$")
ax2.plot(theta, stability, color="C1", linewidth=1.5, label=r"$\max(0,\cos\theta)^4$")
ax2.set_xlabel(r"Tilt angle $\theta_s$ (deg)")
ax2.set_title("Stability $s$")
ax2.legend(loc="upper right", framealpha=0.9)
ax2.set_xlim(0, 90)
ax2.grid(True, alpha=0.3, linewidth=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("outcome_scores.pdf")
fig.savefig("outcome_scores.png")
print("Saved outcome_scores.pdf and outcome_scores.png")
