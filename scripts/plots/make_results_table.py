#!/usr/bin/env python3
"""Emit the LaTeX body of the simulated-comparison table from deep eval JSONs.

Rows are read from the citable deep set (logs/eval_archive/deep/*_100ep_fixedenv)
and, where seed replicates exist, from the fleet battery's deep rows
(logs/fleet_tb_live/*/battery/deep_<tag>). A row with several seeds reports the
mean over seeds and, in the Full column, the seed spread; per-episode standard
deviations stay as they are for single-seed rows so the table never mixes the two
meanings silently.

    python3 scripts/plots/make_results_table.py            # print the tabular body
    python3 scripts/plots/make_results_table.py --check    # list which rows are missing
"""

import argparse
import glob
import json
import os

# label -> list of eval directories (one per seed)
ROWS = [
    ("Privileged expert",   ["logs/eval_archive/deep/expert_100ep_fixedenv"]),
    ("Geometric heuristic", ["logs/eval_archive/deep/heuristic_100ep_fixedenv"]),
    (None, None),                                    # \hline
    ("RL (scratch)",        ["logs/eval_archive/deep/scc0_30_100ep_fixedenv",
                             "logs/fleet_tb_live/*/battery/deep_scc43",
                             "logs/fleet_tb_live/*/battery/deep_scc44"]),
    ("BC, regression head", ["logs/eval_archive/deep/regression_100ep_fixedenv",
                             "logs/fleet_tb_live/*/battery/deep_bc_reg_s43",
                             "logs/fleet_tb_live/*/battery/deep_bc_reg_s44"]),
    ("BC, scoring head",    ["logs/eval_archive/deep/bc_100ep_fixedenv",
                             "logs/fleet_tb_live/*/battery/deep_bc_scoring_s43",
                             "logs/fleet_tb_live/*/battery/deep_bc_scoring_s44"]),
    ("BC$\\to$RL",          ["logs/eval_archive/deep/g100_100ep_fixedenv",
                             "logs/fleet_tb_live/*/battery/deep_s43",
                             "logs/fleet_tb_live/*/battery/deep_s44"]),
]


def load(pattern):
    for d in sorted(glob.glob(pattern)):
        js = sorted(glob.glob(os.path.join(d, "eval_metrics_*.json")))
        if js:
            return json.load(open(js[-1]))["summary"]
    return None


def fmt(rows, key, sub, prec):
    """mean +/- std over episodes for one seed; mean over seeds when replicated."""
    vals = [r[key][sub] if sub else r[key] for r in rows]
    if len(rows) == 1:
        return f"${vals[0]:.{prec}f} \\pm {rows[0][key]['std']:.{prec}f}$" if sub == "mean" else f"{vals[0]:.{prec}f}"
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return f"${m:.{prec}f} \\pm {sd:.{prec}f}$" if sub == "mean" else f"{m:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for label, pats in ROWS:
        if label is None:
            if not a.check:
                print("\\hline")
            continue
        found = [s for s in (load(p) for p in pats) if s]
        if a.check:
            missing = [p for p in pats if load(p) is None]
            print(f"{label:22s} have {len(found)}/{len(pats)} seeds" +
                  (f"  missing: {', '.join(os.path.basename(m.rstrip('/')) for m in missing)}" if missing else ""))
            continue
        if not found:
            print(f"% {label}: no eval rows found")
            continue
        n = fmt(found, "throughput", "mean", 2)
        al = fmt(found, "alignment", "mean", 3)
        st = fmt(found, "stability", "mean", 3)
        su = fmt(found, "grasp_success_pct", "mean", 1)
        fu = fmt(found, "full_clear_rate", None, 0)
        c95 = fmt(found, "cycles_to_95pct", "mean", 1)
        cy = fmt(found, "cycles", "mean", 1)
        tag = "" if len(found) == 1 else f"  % N={len(found)} seeds"
        print(f"{label} & {n} & {al} & {st} & {su} & {fu} & {c95} & {cy} \\\\{tag}")


if __name__ == "__main__":
    main()
