#!/usr/bin/env python3
"""Analyse reward-audit jsonl: does reward track ACTUAL rack decrease?

This is the analyser the audit writer's docstring has referenced since the v2 rebuild but
which never existed (found missing 2026-08-11) - audits were being written and never read.

Each record (one per grasp cycle): reward, logs_grasped (counted by the rise-gated proximity
check), rew_logs (what was actually paid: 0 when the lift gate failed), lift_ok, and
rack_before_despawn (bookkeeping count BEFORE this cycle's despawn). Per env, the rack count
of the NEXT record tells how many logs actually left during this cycle; a jump UP marks an
episode boundary (fresh pile).

Hacking signatures this surfaces:
  - paid-vs-removed mismatch: rew_logs paid but the rack did not shrink by that much
    (the exact channel of the take-1 failure, pre reward_requires_lift)
  - unpaid removals: rack shrank more than paid logs = knocked-off / lost logs
  - per-episode Sum(reward) vs logs actually removed (should be <= removed + bonuses)

Usage:
  python3 scripts/envs/check_reward_alignment.py logs/reward_audit/audit_*.jsonl
"""

import json
import sys
from collections import defaultdict


def main(paths):
    per_env = defaultdict(list)
    n_bad = 0
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    r = json.loads(line)
                    per_env[(path, r["env"])].append(r)
                except Exception:
                    n_bad += 1

    episodes = []  # (cycles, paid_logs, removed_logs, reward_sum, mismatched_cycles)
    overpaid = underpaid = 0
    for key, recs in per_env.items():
        ep = {"cycles": 0, "paid": 0, "removed": 0, "reward": 0.0, "mismatch": 0}
        for cur, nxt in zip(recs, recs[1:] + [None]):
            ep["cycles"] += 1
            ep["paid"] += cur["rew_logs"]
            ep["reward"] += cur["reward"]
            if nxt is not None:
                delta = cur["rack_before_despawn"] - nxt["rack_before_despawn"]
                if delta < 0:  # rack refilled -> episode boundary
                    episodes.append(dict(ep))
                    ep = {"cycles": 0, "paid": 0, "removed": 0, "reward": 0.0, "mismatch": 0}
                    continue
                ep["removed"] += delta
                if cur["rew_logs"] > delta:
                    ep["mismatch"] += 1
                    overpaid += 1
                elif delta > cur["rew_logs"]:
                    underpaid += 1  # knocked-off / lost logs (unpaid removals)
        if ep["cycles"] > 1:
            episodes.append(ep)

    if not episodes:
        print("no complete episodes found in audit")
        return

    n = len(episodes)
    tot = lambda k: sum(e[k] for e in episodes)
    print(f"files: {len(paths)}  envs: {len(per_env)}  episodes: {n}  bad lines: {n_bad}")
    print(f"cycles/episode:     {tot('cycles')/n:8.1f}")
    print(f"paid logs/episode:  {tot('paid')/n:8.1f}")
    print(f"removed/episode:    {tot('removed')/n:8.1f}   (rack decrease, excludes final cycle)")
    print(f"reward/episode:     {tot('reward')/n:8.1f}")
    print(f"reward per paid log:{tot('reward')/max(tot('paid'),1):8.2f}")
    print(f"OVERPAID cycles (paid > actually removed): {overpaid}  <- hacking channel if > 0")
    print(f"unpaid removals (removed > paid, knocked/lost): {underpaid}")
    worst = sorted(episodes, key=lambda e: e["mismatch"], reverse=True)[:3]
    for e in worst:
        if e["mismatch"]:
            print(f"  worst ep: cycles={e['cycles']} paid={e['paid']} removed={e['removed']} "
                  f"reward={e['reward']:.1f} mismatched_cycles={e['mismatch']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
