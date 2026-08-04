#!/usr/bin/env python3
"""Sim vs real point clouds under the CURRENT tight crop and the PROPOSED widened crop.

Real clouds come from the rosbag via bag_cloud_extract.py (the recorded policy_debug npz only
keeps the tight crop, so the extra band is not recoverable from it). Extraction was validated
against the node's own recording: median 0.2 cm / p95 3.0 cm agreement.

    python3 scripts/envs/plot_crop_sim_vs_real.py --scratch <dir> --margin 0.5 --out fig.png
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
RACK_Y_SHIFT = -0.55            # real crop follows the physical rack; shift back to train coords
DS = "logs/bc_pointcloud/bc_policy_aug1_v2"


def crop(p, g, shift=0.0):
    lo, hi = B_MIN + [0, shift, 0], B_MAX + [0, shift, 0]
    m = ((p[:, 0] >= lo[0] - g) & (p[:, 0] <= hi[0] + g) &
         (p[:, 1] >= lo[1] - g) & (p[:, 1] <= hi[1] + g) &
         (p[:, 2] >= lo[2] - g) & (p[:, 2] <= hi[2]))
    q = p[m].copy()
    q[:, 1] -= shift
    return q


def valid(p):
    return p[np.abs(p).sum(-1) > 1e-6]


def panel(ax, p, title):
    if len(p):
        ax.scatter(p[:, 1], p[:, 2], s=1.0, c=p[:, 2], cmap="viridis",
                   vmin=-1.45, vmax=-0.4, linewidths=0)
    ax.axhline(B_MIN[2], color="tab:red", lw=0.8, ls="--", alpha=0.6)
    for yb in (B_MIN[1], B_MAX[1]):
        ax.axvline(yb, color="tab:red", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlim(B_MIN[1] - 0.9, B_MAX[1] + 0.9)
    ax.set_ylim(-1.75, 0.25)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True, help="dir holding real_full.npz / real_end.npz")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    full = np.load(os.path.join(a.root, DS, "pointclouds_full.npy"), mmap_mode="r")
    tight = np.load(os.path.join(a.root, DS, "pointclouds.npy"), mmap_mode="r")
    n = np.array([len(valid(np.asarray(tight[i]))) for i in range(0, len(tight), 23)])
    idx = np.arange(0, len(tight), 23)
    sim_pick = [int(idx[np.argsort(np.abs(n - t))[0]]) for t in (1024, 90)]
    real = [np.load(os.path.join(a.scratch, f))["points"] for f in ("real_full.npz", "real_end.npz")]
    stage = ["FULL PILE", "ENDGAME (bed nearly bare)"]

    fig, ax = plt.subplots(4, 2, figsize=(13, 11))
    for c in range(2):
        s_t = valid(np.asarray(tight[sim_pick[c]]))
        s_m = crop(valid(np.asarray(full[sim_pick[c]])), a.margin)
        r_t = crop(real[c], 0.0, RACK_Y_SHIFT)
        r_m = crop(real[c], a.margin, RACK_Y_SHIFT)
        for r, (p, lab) in enumerate([(s_t, "SIM  tight crop  (what BC trains on today)"),
                                      (s_m, f"SIM  crop_margin {a.margin}  (proposed)"),
                                      (r_t, "REAL tight crop  (what deployment feeds it)"),
                                      (r_m, f"REAL crop_margin {a.margin}  (proposed)")]):
            panel(ax[r][c], p, f"{lab}\n{stage[c]} - {len(p)} pts")
    for r, lab in enumerate(["SIM tight", "SIM +margin", "REAL tight", "REAL +margin"]):
        ax[r][0].set_ylabel(f"{lab}\nz [m]", fontsize=8)
    for c in range(2):
        ax[3][c].set_xlabel("y [m]   side view; dashed red = action box", fontsize=7)
    fig.suptitle("The tight crop deletes the bed plane, rails and poles in SIM but cannot in REAL "
                 "-> the net never trains on what it mis-targets", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
