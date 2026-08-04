#!/usr/bin/env python3
"""Summarise the observation-degradation sweep into one table per metric, with slopes.

The number that matters is not any single cell but the SLOPE of each policy across degradation
levels. The RL chapter's premise is that BC->RL degrades more slowly than BC: BC is trained to
MATCH a privileged expert, so it inherits the expert's confidence without the expert's information,
while RL optimises outcome under the observations actually available and can hedge.

    python3 scripts/envs/summarize_noise_sweep.py --dir logs/sim_eval/noise_sweep
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

LEVELS = ["clean", "1x", "3x", "8x"]
COEFF = {"clean": 0.0, "1x": 0.0014, "3x": 0.0042, "8x": 0.0112}
POLICIES = ["heuristic", "heuristicG", "bc", "bcrl", "bcrl2", "scoring"]

# label -> (path in summary, higher-is-better)
METRICS = [
    ("grasp success %", "grasp_success_pct", True),
    ("logs / success", "throughput", True),
    ("pile cleared %", "pile_cleared_pct", True),
    ("cycles", "cycles", False),
    ("cycles to 95%", "cycles_to_95pct", False),
    ("stability", "stability", True),
    ("alignment", "alignment", True),
]


def load(d, tag):
    p = os.path.join(d, tag)
    js = sorted(glob.glob(os.path.join(p, "eval_metrics_*.json")), key=os.path.getmtime)
    if not js:
        return None
    try:
        with open(js[-1]) as f:
            return json.load(f)
    except Exception:
        return None


def get(m, key):
    if m is None:
        return None, None
    s = m.get("summary", {}).get(key)
    if isinstance(s, dict):
        return s.get("mean"), s.get("std")
    return (s, None) if isinstance(s, (int, float)) else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/sim_eval/noise_sweep")
    ap.add_argument("--std", action="store_true", help="show std alongside each mean")
    a = ap.parse_args()

    data = {(p, lv): load(a.dir, f"{p}_{lv}") for p in POLICIES for lv in LEVELS}
    # metrics_version 1 = clearing/c95 from the cumulative logs_grasped tally (inflated, can
    # exceed 100%); 2 = from the measured post-despawn rack count. Mixing them in one column
    # compares different quantities, so say so loudly rather than printing a tidy table.
    vers = {k: (v or {}).get("eval_config", {}).get("metrics_version", 1)
            for k, v in data.items() if v}
    if len(set(vers.values())) > 1:
        v1 = sorted(f"{p}_{l}" for (p, l), n in vers.items() if n == 1)
        print("!! MIXED metrics_version - clearing %, cycles to 95%, logs/success and full-clear")
        print("!! are NOT comparable across these rows. v1 (cumulative-grasp, inflated):")
        print("!!   " + ", ".join(v1))
        print("!! grasp success %, stability, alignment and cycles ARE comparable.\n")
    mism = {k: (v or {}).get("eval_config", {}).get("accounting_mismatch_cycles", -1)
            for k, v in data.items() if v}
    bad = {k: m for k, m in mism.items() if m > 0}
    if bad:
        print("!! rows with derived-vs-measured accounting mismatches: "
              + ", ".join(f"{p}_{l}({m})" for (p, l), m in sorted(bad.items())) + "\n")
    have = {p for (p, _), v in data.items() if v}
    print(f"[sweep] {sum(1 for v in data.values() if v)}/{len(data)} rows in {a.dir}")
    print(f"[sweep] policies present: {', '.join(sorted(have)) or 'none'}\n")

    for label, key, higher_better in METRICS:
        rows = []
        for p in POLICIES:
            vals = [get(data[(p, lv)], key)[0] for lv in LEVELS]
            if all(v is None for v in vals):
                continue
            stds = [get(data[(p, lv)], key)[1] for lv in LEVELS]
            xs = [COEFF[lv] for lv, v in zip(LEVELS, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            slope = np.polyfit(xs, ys, 1)[0] * 0.0014 if len(ys) >= 3 else None
            rows.append((p, vals, stds, slope))
        if not rows:
            continue
        print(f"--- {label}  ({'higher' if higher_better else 'lower'} is better)")
        print(f"{'policy':10s} " + " ".join(f"{lv:>12s}" for lv in LEVELS) + "    per-1x")
        for p, vals, stds, slope in rows:
            cells = []
            for v, s in zip(vals, stds):
                if v is None:
                    cells.append(f"{'-':>12s}")
                elif a.std and s is not None:
                    cells.append(f"{v:7.1f}+-{s:3.1f}")
                else:
                    cells.append(f"{v:12.2f}")
            sl = f"{slope:+9.2f}" if slope is not None else f"{'n/a':>9s}"
            print(f"{p:10s} " + " ".join(cells) + f"  {sl}")
        print()

    print("per-1x = linear-fit change per one multiple of the characterised ZED coefficient")
    print("         (0.0014). Compare SLOPES across policies, not absolute cells: 10 episodes")
    print("         carries roughly +-8-10 points on grasp success.")
    print("The privileged expert is absent BY DESIGN - it reads ground-truth log poses and never")
    print("touches the cloud, so it is flat under this sweep and forms the ceiling.")


if __name__ == "__main__":
    main()
