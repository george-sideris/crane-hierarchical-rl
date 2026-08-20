#!/usr/bin/env python3
"""Argmax evaluation trajectories: deployable quality against training progress.

Every tenth checkpoint of a run is evaluated deterministically (argmax, exactly as
deployed) under the wave protocol, 40 environments x 1 episode. Training return is
not the quantity of interest and is not plotted here; these curves are what the
checkpoint-selection doctrine of the chapter actually reads.

Rows come from the fleet battery (logs/fleet_tb_live/*/battery/wave_<arm>_<iter>/),
synced by eval_scripts/fleet_tb_sync.sh. Wave rows RANK checkpoints; the citable
table numbers are the deep 100-episode rows.

    python3 scripts/plots/plot_argmax_trajectories.py [--out ...] [--arms s44,symcritic,unfroz]
"""

import argparse
import glob
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "legend.fontsize": 7,
    "mathtext.fontset": "cm",
})

STEPS_PER_ITER = 320          # 40 envs x 8 grasp cycles per rollout
LABELS = {
    "s44":       ("frozen encoder, asymmetric critic (locked recipe)", "#4c72b0"),
    "symcritic": ("symmetric critic", "#dd8452"),
    "unfroz":    ("encoder not frozen", "#c44e52"),
    "scc44":     ("from scratch", "#937860"),
    "mn":        ("multiplicative, normalized", "#8172b2"),
    "an":        ("additive, normalized", "#55a868"),
    "nq":        ("no quality terms", "#8c8c8c"),
}


def load_rows():
    arms = defaultdict(dict)
    for f in glob.glob("logs/fleet_tb_live/*/battery/*/eval_metrics_*.json"):
        tag = f.split("/")[-2]
        m = re.match(r"wave_(.+)_(\d+)$", tag)
        if not m:
            continue
        arm, it = m.group(1), int(m.group(2))
        s = json.load(open(f))["summary"]
        prev = arms[arm].get(it)
        if prev is None or f > prev[0]:      # a re-run row supersedes the older file
            arms[arm][it] = (f, s["full_clear_rate"], s["cycles"]["mean"])
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="s44,symcritic,unfroz")
    ap.add_argument("--out", default="docs/thesis/figures/rl_argmax_trajectories.pdf")
    a = ap.parse_args()

    arms = load_rows()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for arm in a.arms.split(","):
        pts = sorted(arms.get(arm, {}).items())
        if not pts:
            print(f"  (no rows for {arm})")
            continue
        label, color = LABELS.get(arm, (arm, None))
        x = [it * STEPS_PER_ITER for it, _ in pts]
        ax1.plot(x, [v[1] for _, v in pts], marker="o", ms=3, lw=1.3, color=color, label=label)
        ax2.plot(x, [v[2] for _, v in pts], marker="o", ms=3, lw=1.3, color=color)
        best = max(pts, key=lambda kv: (kv[1][1], -kv[1][2]))
        print(f"{arm:10s} n={len(pts):2d}  best iter {best[0]:3d}: "
              f"full {best[1][1]:.1f}%, cycles {best[1][2]:.2f}")
    ax1.set_ylabel("argmax full-clear rate [%]")
    ax2.set_ylabel("argmax cycles to clear")
    for ax in (ax1, ax2):
        ax.set_xlabel("environment steps (grasp cycles)")
        ax.grid(lw=0.4, alpha=0.4)
    ax1.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
