#!/usr/bin/env python3
"""Cumulative logs moved against cycle for the hardware trials.

Rebuilds docs/thesis figures/real_trials/clearing_curves_real.png from the
manually scored spreadsheet (baseline and scoring trials) and the shared CSV
export (BC+RL Trial 03, double mound). Empty cycles are marked with an x.

    python3 scripts/envs/render_real_clearing_curves.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_grasp_depth import read_ods_rows, parse_block, parse_csv_trial

TRIALS = [
    # (label, style)
    ("Baseline double", ("Heuristic", "double"), "#1f77b4", "--"),
    ("Baseline flat",   ("Heuristic", "flat"),   "#17becf", "--"),
    ("Baseline single", ("Heuristic", "single"), "#4c72b0", "--"),
    ("Scoring single",  ("BC", "single"),        "#dd8452", "-"),
    ("Scoring double",  ("BC", "double"),        "#c44e52", "-"),
    ("Scoring flat",    ("BC", "flat"),          "#8172b2", "-"),
    ("BC->RL flat",     ("BCRL", "flat"),        "#937860", "-"),
    ("BC->RL double",   None,                    "#2ca02c", "-"),
]
BCRL_CSV = "~/Downloads/log_loader_ML_experimental_data(03_BC_RL).csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/thesis/figures/real_trials/clearing_curves_real.png")
    ap.add_argument("--ods", default="~/Documents/real_trials.ods")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_ods_rows(args.ods)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for label, key, color, ls in TRIALS:
        t = parse_csv_trial(BCRL_CSV) if key is None else parse_block(rows, *key)
        logs = np.array([l for l, _ in t])
        cyc = np.arange(1, len(logs) + 1)
        cum = np.cumsum(logs)
        ax.plot(cyc, cum, ls, color=color, lw=1.9,
                label=f"{label} ({cum[-1]} logs / {len(logs)} cy)")
        empty = logs == 0
        ax.plot(cyc[empty], cum[empty], "x", color=color, ms=7, mew=2)
    ax.set_xlabel("cycle")
    ax.set_ylabel("cumulative logs moved")
    ax.set_title("Clearing curves, real trials (x = empty cycle)")
    ax.grid(lw=0.4, alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.expanduser(args.out), dpi=170)
    print("->", args.out)


if __name__ == "__main__":
    main()
