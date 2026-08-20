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

import numpy as np
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
# family -> (label, colour, [arm tags, one per seed])
FAMILIES = {
    "locked":    ("frozen encoder, asymmetric critic", "#4c72b0", ["s42", "s43", "s44"]),
    "symcritic": ("symmetric critic",                  "#dd8452", ["symcritic", "sc43", "sc44"]),
    "unfroz":    ("encoder not frozen",                "#c44e52", ["unfroz", "uf43", "uf44"]),
    "scratch":   ("from scratch",                      "#937860", ["scc0", "scc43", "scc44"]),
    "mn":        ("multiplicative, normalized",        "#8172b2", ["mn", "mn43", "mn44"]),
    "an":        ("additive, normalized",              "#55a868", ["an", "an43", "an44"]),
    "nq":        ("no quality terms",                  "#8c8c8c", ["nq", "nq43", "nq44"]),
    "gs":        ("Gaussian over coordinates",         "#555555", ["gs", "gs43", "gs44"]),
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
    ap.add_argument("--families", default="locked,symcritic,unfroz")
    ap.add_argument("--out", default="docs/thesis/figures/rl_argmax_trajectories.pdf")
    a = ap.parse_args()

    arms = load_rows()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for fam in a.families.split(","):
        label, color, tags = FAMILIES[fam]
        seeds = [sorted(arms[t].items()) for t in tags if arms.get(t)]
        if not seeds:
            print(f"  (no rows for {fam})")
            continue
        # band over the seeds that exist, on their common iteration grid
        common = sorted(set.intersection(*[{it for it, _ in s_} for s_ in seeds])) if len(seeds) > 1 else [it for it, _ in seeds[0]]
        x = [it * STEPS_PER_ITER for it in common]
        for j, (ax, idx) in enumerate(((ax1, 1), (ax2, 2))):
            Y = np.array([[dict(s_)[it][idx] for it in common] for s_ in seeds], float)
            m, sd = Y.mean(0), Y.std(0)
            ax.plot(x, m, marker="o", ms=3, lw=1.3, color=color,
                    label=f"{label} (N={len(seeds)})" if j == 0 else None)
            if len(seeds) > 1:
                ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18, lw=0)
        flat = [kv for s_ in seeds for kv in s_]
        best = max(flat, key=lambda kv: (kv[1][1], -kv[1][2]))
        print(f"{fam:10s} seeds={len(seeds)} pts={len(flat):2d}  best iter {best[0]:3d}: "
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
