#!/usr/bin/env python3
"""Assemble sim-eval rows into one table, with TRUE (measured) grasp success recomputed.

Why the recomputation exists (measured 2026-08-07 on the 100-episode rows):
`logs_grasped` counts logs within 1.5 m of the grapple that also passed the rise gate, at the
moment the grasp is evaluated. `_despawn_grasped_logs` removes by proximity a moment later in
the FSM. The two sets disagree on ~17% of cycles, so a cycle can be recorded as a successful
grasp while the rack count does not move. Over a 200-log episode the expert "grasps" 240 logs
and the heuristic 249, while only ~194/196 actually leave the rack (+24-26% over-count), and
the reported success rate is inflated by 16-18 points.

What is and is not affected:
  affected   : grasp_success_pct   (proximity-derived; INFLATED ~16-18 pts)
               throughput          (measured logs / INFLATED success count -> biased LOW ~25%)
               total logs grasped  (proximity-derived; +24-26%)
  unaffected : pile_cleared_pct, cycles_to_95pct, cycles, full_clear_rate
               (computed from the MEASURED post-despawn rack count, metrics_version 2)

The two corrections move in OPPOSITE directions and share one cause: cycles that record a
grasp but remove nothing. Fewer successes than reported, each taking a bigger real bite
(expert and heuristic both land at 11.32 logs per true success).

TRUE success per cycle = (prev_logs_in_rack - logs_in_rack) - knocked_off > 0
i.e. the rack actually shrank, and not merely because logs were knocked out of bounds.

    python3 scripts/envs/summarize_eval_rows.py --dir logs/sim_eval/citable
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load_metrics(row_dir):
    f = sorted(glob.glob(os.path.join(row_dir, "eval_metrics_*.json")))
    if not f:
        return None
    with open(f[-1]) as fh:
        return json.load(fh)


def pool_chunks(base_dir, tag):
    """Pool chunked rows (<tag>_s42, _s43, ...) into one weighted row.

    Chunking exists because eval writes only at the end of a run, so a crash loses the whole
    row; 4x25 episodes with different seeds costs the same GPU time, survives crashes, and is
    statistically equivalent (same generator, same seeds across every policy -> still paired).
    Means are weighted by each chunk's episode count.
    """
    dirs = sorted(glob.glob(os.path.join(base_dir, f"{tag}_s*")))
    dirs = [d for d in dirs if os.path.isdir(d) and load_metrics(d)]
    if not dirs:
        return None, []
    tot_eps, acc = 0, {}
    keys = ["grasp_success_pct", "throughput", "pile_cleared_pct", "cycles",
            "cycles_to_95pct", "full_clear_rate", "stability", "alignment"]
    for d in dirs:
        m = load_metrics(d)
        n = m["eval_config"]["num_episodes"]
        tot_eps += n
        for k in keys:
            v = m["summary"].get(k)
            v = v["mean"] if isinstance(v, dict) else v
            if v is not None:
                acc[k] = acc.get(k, 0.0) + v * n
    pooled = {"summary": {k: {"mean": v / tot_eps} for k, v in acc.items()},
              "eval_config": {"num_episodes": tot_eps, "chunks": len(dirs)}}
    return pooled, dirs


def true_success_multi(row_dirs):
    """true_success() aggregated over several chunk dirs."""
    tot = rep_s = true_s = rep_g = true_g = 0
    for rd in row_dirs:
        r = _true_counts(rd)
        if r:
            tot += r[0]; rep_s += r[1]; true_s += r[2]; rep_g += r[3]; true_g += r[4]
    if tot == 0:
        return None
    return (100.0 * true_s / tot, 100.0 * rep_s / tot,
            100.0 * (rep_g - true_g) / max(true_g, 1), tot, true_g / max(true_s, 1))


def _true_counts(row_dir):
    r = true_success(row_dir, raw=True)
    return r


def true_success(row_dir, raw=False):
    """(true_succ_pct, reported_succ_pct, over_pct, n_cycles, true_throughput) or None."""
    f = sorted(glob.glob(os.path.join(row_dir, "decisions_*.npz")))
    if not f:
        return None
    d = np.load(f[-1], allow_pickle=True)
    if "logs_in_rack" not in d.files:
        return None                      # row predates the measured counter
    env, ep, cyc = d["env"], d["episode"], d["cycle"]
    lg = d["logs_grasped"].astype(int)
    ink = d["logs_in_rack"].astype(int)
    ko = d["knocked_off"].astype(int)
    tot = rep_s = true_s = rep_g = true_g = 0
    for e in np.unique(env):
        for p in np.unique(ep[env == e]):
            m = (env == e) & (ep == p)
            o = np.argsort(cyc[m])
            g, k, K = lg[m][o], ink[m][o], ko[m][o]
            if len(g) < 2:
                continue
            # cycle i+1 removed (k[i] - k[i+1]) from the rack; knocked-off is not a grasp
            removed = (k[:-1] - k[1:]) - K[1:]
            rep = g[1:]
            tot += len(rep)
            rep_s += int((rep > 0).sum())
            true_s += int((removed > 0).sum())
            rep_g += int(rep.sum())
            true_g += int(removed[removed > 0].sum())
    if tot == 0:
        return None
    if raw:
        return tot, rep_s, true_s, rep_g, true_g
    over = 100.0 * (rep_g - true_g) / max(true_g, 1)
    true_thr = true_g / max(true_s, 1)      # logs actually removed per TRUE success
    return 100.0 * true_s / tot, 100.0 * rep_s / tot, over, tot, true_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/sim_eval/citable")
    ap.add_argument("--rows", nargs="*", default=None, help="row names, default = all subdirs")
    ap.add_argument("--pool", nargs="*", default=None,
                    help="pool chunked rows: give base tags (e.g. --pool bc_reg p2c) and the "
                         "tool aggregates <tag>_s42, <tag>_s43, ... into one weighted row")
    a = ap.parse_args()

    if a.pool:
        hdr = (f"{'row (pooled)':22} {'eps':>4} {'succ%':>7} {'TRUE%':>7} {'l/succ':>7} "
               f"{'TRUEl/s':>8} {'clear%':>7} {'cycles':>7} {'c95':>6} {'full%':>6} {'over':>6}")
        print(hdr); print("-" * len(hdr))
        for tag in a.pool:
            m, dirs = pool_chunks(a.dir, tag)
            if not m:
                print(f"{tag:22} {'(no chunks yet)':>40}"); continue
            s_, e = m["summary"], m["eval_config"]
            g = lambda k: s_[k]["mean"] if isinstance(s_.get(k), dict) else s_.get(k)  # noqa: E731
            ts = true_success_multi(dirs)
            tcol = f"{ts[0]:7.2f}" if ts else f"{'n/a':>7}"
            thcol = f"{ts[4]:8.2f}" if ts else f"{'n/a':>8}"
            ocol = f"{ts[2]:+5.0f}%" if ts else f"{'n/a':>6}"
            label = f"{tag} [{e.get('chunks', 0)}ch]"
            print(f"{label:22} "
                  f"{e['num_episodes']:4d} {g('grasp_success_pct'):7.2f} {tcol} "
                  f"{g('throughput'):7.2f} {thcol} {g('pile_cleared_pct'):7.2f} "
                  f"{g('cycles'):7.2f} {g('cycles_to_95pct'):6.2f} {g('full_clear_rate'):6.1f} {ocol}")
        return

    rows = a.rows or sorted(d for d in os.listdir(a.dir)
                            if os.path.isdir(os.path.join(a.dir, d)))
    hdr = (f"{'row':22} {'eps':>4} {'succ%':>7} {'TRUE%':>7} {'l/succ':>7} {'TRUEl/s':>8} "
           f"{'clear%':>7} {'cycles':>7} {'c95':>6} {'full%':>6} {'over':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p = os.path.join(a.dir, r)
        m = load_metrics(p)
        if not m:
            print(f"{r:22} {'(running / no metrics yet)':>40}")
            continue
        s, e = m["summary"], m["eval_config"]
        g = lambda k: s[k]["mean"] if isinstance(s.get(k), dict) else s.get(k)  # noqa: E731
        ts = true_success(p)
        tcol = f"{ts[0]:7.2f}" if ts else f"{'n/a':>7}"
        thcol = f"{ts[4]:8.2f}" if ts else f"{'n/a':>8}"
        ocol = f"{ts[2]:+5.0f}%" if ts else f"{'n/a':>6}"
        print(f"{r:22} {e['num_episodes']:4d} {g('grasp_success_pct'):7.2f} {tcol} "
              f"{g('throughput'):7.2f} {thcol} {g('pile_cleared_pct'):7.2f} {g('cycles'):7.2f} "
              f"{g('cycles_to_95pct'):6.2f} {g('full_clear_rate'):6.1f} {ocol}")
    print("\nsucc%   = reported (logs_grasped > 0)          - INFLATED ~16-18 pts")
    print("TRUE%   = rack actually shrank, knocked-off subtracted   <- CITE THIS")
    print("l/succ  = reported (measured logs / INFLATED successes) - biased LOW ~25%")
    print("TRUEl/s = logs actually removed per TRUE success        <- CITE THIS")
    print("over    = reported logs grasped vs logs that actually left the rack")
    print("clear%, cycles, c95, full% use the MEASURED rack count and are unaffected.")


if __name__ == "__main__":
    main()
