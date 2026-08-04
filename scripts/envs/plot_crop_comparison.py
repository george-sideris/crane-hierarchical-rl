#!/usr/bin/env python3
"""Side-by-side of what the policy TRAINS on (sim) vs what it SEES at deployment (real).

Row 1  sim, current tight crop (= the action box, fitted to the rack interior)
Row 2  sim, widened observation crop (--crop_margin), from the uncropped clouds we already store
Row 3  real, as recorded by the node

The point of the figure: the tight crop removes the bed plane, the side rails and the pole
corners BY CONSTRUCTION, so the network never trains on them - yet they are 80%+ of a real
endgame cloud, and they are what the crane mis-targets (22% of real cycles, 2026-08-03 trials).

    python3 scripts/envs/plot_crop_comparison.py --margin 0.5 --out /tmp/crop_comparison.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
DS = "logs/bc_pointcloud/bc_policy_aug1_v2"
REAL = "logs/crane_policy_debug/run_20260803_174017"     # jagged trial: runs down to bare bed


def crop(p, margin):
    g = margin
    m = ((p[:, 0] >= B_MIN[0] - g) & (p[:, 0] <= B_MAX[0] + g) &
         (p[:, 1] >= B_MIN[1] - g) & (p[:, 1] <= B_MAX[1] + g) &
         (p[:, 2] >= B_MIN[2] - g) & (p[:, 2] <= B_MAX[2]))
    return p[m]


def valid(p):
    return p[np.abs(p).sum(-1) > 1e-6]


def panel(ax, p, title, margin=0.0):
    if len(p):
        ax.scatter(p[:, 1], p[:, 2], s=1.2, c=p[:, 2], cmap="viridis",
                   vmin=B_MIN[2] - 0.2, vmax=-0.4, linewidths=0)
    ax.axhline(B_MIN[2], color="tab:red", lw=0.8, ls="--", alpha=0.7)
    for yb in (B_MIN[1], B_MAX[1]):
        ax.axvline(yb, color="tab:red", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(B_MIN[1] - 0.8, B_MAX[1] + 0.8)
    ax.set_ylim(B_MIN[2] - 0.8, 0.2)
    struct = 100 * (p[:, 2] < B_MIN[2] + 0.06).mean() if len(p) else 0.0
    ax.set_title(f"{title}\n{len(p)} pts | {struct:.0f}% at bed level", fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default="/tmp/crop_comparison.png")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    full = np.load(os.path.join(a.root, DS, "pointclouds_full.npy"), mmap_mode="r")
    tight = np.load(os.path.join(a.root, DS, "pointclouds.npy"), mmap_mode="r")

    # pick sim scenes by how much of the pile is left (valid points in the tight crop)
    n = np.array([len(valid(np.asarray(tight[i]))) for i in range(0, len(tight), 23)])
    idx = np.arange(0, len(tight), 23)
    sim_pick = [int(idx[np.argsort(np.abs(n - t))[0]]) for t in (1024, 600, 90)]
    real_pick = ["002", "020", "036"]
    stage = ["full pile", "mid clearing", "endgame (bed nearly bare)"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for c in range(3):
        panel(axes[0][c], valid(np.asarray(tight[sim_pick[c]])),
              f"SIM tight crop (current) - {stage[c]}")
        panel(axes[1][c], crop(valid(np.asarray(full[sim_pick[c]])), a.margin),
              f"SIM crop_margin {a.margin} (proposed) - {stage[c]}")
        rp = np.load(os.path.join(a.root, REAL, f"policy_debug_{real_pick[c]}.npz"))
        p = valid(rp["points"].astype(np.float32)).copy()
        p[:, 1] -= float(rp["rack_y_shift"])          # into training coords
        panel(axes[2][c], p, f"REAL as deployed - cycle {real_pick[c]}")

    for r, lab in enumerate(["SIM (train)", "SIM + margin", "REAL (deploy)"]):
        axes[r][0].set_ylabel(f"{lab}\nz [m]", fontsize=8)
    for c in range(3):
        axes[2][c].set_xlabel("y [m]  (side view, dashed red = action box)", fontsize=7)
    fig.suptitle("What the policy trains on vs what it sees: the tight crop deletes the bed "
                 "plane and rails that the real crane mis-targets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}  (sim idx {sim_pick}, real cycles {real_pick})")


if __name__ == "__main__":
    main()
