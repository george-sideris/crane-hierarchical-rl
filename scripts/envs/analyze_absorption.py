"""Absorption analysis over recorded sim-eval decision files.

Defines a LOCK mechanism-agnostically, matching what the field trials showed:
a run of consecutive empty cycles in one episode whose successive targets sit
within LOCK_D of each other in the horizontal plane (the same target reissued
against an unchanged scene). No structure map is needed, so the definition
applies identically to every policy and noise level, and transfers to the
real trials.

Per-episode endpoints:
  absorbed   - episode contains a lock run of length >= LOCK_LEN that begins
               with more than MIN_INV_FRAC of the inventory still in the rack
  recovery   - fraction of empty cycles followed by a productive cycle
  max_lock   - longest lock run

Episodes are independent (seeded, independently randomized piles), so
between-arm comparisons are exact tests at the episode level.

Usage: python3 scripts/envs/analyze_absorption.py <label>=<decisions.npz> ...
"""
import glob
import sys

import numpy as np
from scipy import stats

LOCK_D = 0.25        # m, successive-target displacement that counts as "same target"
LOCK_LEN = 3         # consecutive same-target empty cycles = a lock
MIN_INV_FRAC = 0.05  # lock must begin with >5% of inventory remaining
NUM_LOGS = 200


def episodes_of(path):
    z = np.load(path, allow_pickle=True)
    env, ep, cyc = z["env"], z["episode"], z["cycle"]
    tgt, grasped = z["target"], z["logs_grasped"]
    # rows recorded before the measured rack count existed carry only the
    # DERIVED remaining count; it can drift negative, so clamp - it is used
    # only for the >5%-inventory gate on lock runs
    in_rack = z["logs_in_rack"] if "logs_in_rack" in z.files \
        else np.clip(z["logs_remaining"], 0, None)
    out = {}
    for i in range(len(env)):
        out.setdefault((int(env[i]), int(ep[i])), []).append(
            (int(cyc[i]), np.asarray(tgt[i][:2], float), int(grasped[i]), int(in_rack[i])))
    for k in out:
        out[k].sort()
    return list(out.values())


def analyze(cycles):
    """Return (absorbed, recovery_num, recovery_den, max_lock) for one episode."""
    empt = [(c, t, r) for c, t, g, r in cycles if g == 0]
    succ = {c for c, t, g, r in cycles if g > 0}
    n = len(cycles)
    rec_n = rec_d = 0
    for c, t, g, r in cycles:
        if g == 0 and c + 1 <= cycles[-1][0]:
            rec_d += 1
            rec_n += 1 if (c + 1) in succ else 0
    # lock runs: consecutive cycles, all empty, successive targets within LOCK_D
    max_lock, run, absorbed = 0, [], False
    prev_c = prev_t = None
    for c, t, g, r in cycles:
        if g == 0 and prev_c == c - 1 and prev_t is not None \
                and np.linalg.norm(t - prev_t) < LOCK_D and run:
            run.append((c, r))
        elif g == 0:
            run = [(c, r)]
        else:
            run = []
        if run:
            max_lock = max(max_lock, len(run))
            if len(run) >= LOCK_LEN and run[0][1] > MIN_INV_FRAC * NUM_LOGS:
                absorbed = True
        prev_c, prev_t = c, (t if g == 0 else None)
    return absorbed, rec_n, rec_d, max_lock


def summarize(label, path):
    eps = episodes_of(path)
    res = [analyze(e) for e in eps]
    absorbed = sum(r[0] for r in res)
    rec_n = sum(r[1] for r in res)
    rec_d = sum(r[2] for r in res)
    max_locks = [r[3] for r in res]
    return {"label": label, "n_ep": len(eps), "absorbed": absorbed,
            "rec_n": rec_n, "rec_d": rec_d,
            "recovery": rec_n / max(rec_d, 1),
            "mean_max_lock": float(np.mean(max_locks)),
            "p95_max_lock": float(np.percentile(max_locks, 95))}


def main():
    rows = []
    for a in sys.argv[1:]:
        label, pat = a.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            print(f"  !! no file for {label}: {pat}")
            continue
        # merge multiple seeds into one arm
        merged = {"label": label, "n_ep": 0, "absorbed": 0, "rec_n": 0, "rec_d": 0,
                  "_locks": []}
        for f in files:
            s = summarize(label, f)
            merged["n_ep"] += s["n_ep"]
            merged["absorbed"] += s["absorbed"]
            merged["rec_n"] += s["rec_n"]
            merged["rec_d"] += s["rec_d"]
            merged["_locks"].append((s["mean_max_lock"], s["n_ep"]))
        merged["recovery"] = merged["rec_n"] / max(merged["rec_d"], 1)
        merged["mean_max_lock"] = sum(m * n for m, n in merged["_locks"]) / merged["n_ep"]
        rows.append(merged)

    print(f"{'arm':18s} {'eps':>4s} {'absorbed':>9s} {'abs%':>6s} "
          f"{'recovery':>12s} {'mean maxlock':>12s}")
    for r in rows:
        print(f"{r['label']:18s} {r['n_ep']:4d} {r['absorbed']:6d}/{r['n_ep']:<3d}"
              f" {100*r['absorbed']/r['n_ep']:5.1f} "
              f"{r['rec_n']:4d}/{r['rec_d']:<4d}={100*r['recovery']:4.1f}%"
              f" {r['mean_max_lock']:10.2f}")

    if len(rows) >= 2:
        print("\npairwise absorbed-fraction Fisher exact:")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                p = stats.fisher_exact([[a["absorbed"], a["n_ep"] - a["absorbed"]],
                                        [b["absorbed"], b["n_ep"] - b["absorbed"]]])[1]
                print(f"  {a['label']} vs {b['label']}: p = {p:.4g}")


if __name__ == "__main__":
    main()
