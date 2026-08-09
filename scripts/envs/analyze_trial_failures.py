#!/usr/bin/env python3
"""On the REAL cycles where the deployed policy mis-targeted, what would each policy pick?

This is the counterfactual for the 2026-08-03 baseline trials. 78 grasp cycles were recorded
across three runs, each with the exact cloud the policy saw and the target it chose. Roughly a
fifth of them put the grapple somewhere with no logs under it - typically the near rail (x at
the +x box edge, z pinned to the bed floor) or the far end of the rack. Those are the cycles
that dominated the real failure count, and the question for the thesis is whether a learned
policy would have failed on the SAME clouds, or picked a real log.

Metric is `support`: cloud points within SUPPORT_R of the chosen target, measured on the cloud
the policy actually saw. Support ~0 means the grapple closes on nothing. Reporting support
rather than "did it grasp" is deliberate - we cannot re-run the crane on those clouds, but we
can measure whether each policy's target has logs under it.

CAVEAT recorded in the output: the saved clouds are the TIGHT crop at 1024 points, so the
margin-trained policies see them zero-padded to 2048 with no margin context. That is a domain
mismatch working AGAINST the margin policies, so any advantage they show here is a lower bound.

    python3 scripts/envs/analyze_trial_failures.py --draws 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy                      # noqa: E402
from noise_sensitivity import (RegressionBC, dec5, pad_to,       # noqa: E402
                               zed_noise, heuristic_target, n_modes,
                               B_MIN, B_MAX)

SUPPORT_R = 0.50
RUNS = "logs/crane_policy_debug/run_20260803_*"


def support_of(target, cloud, r=SUPPORT_R):
    if target is None or not np.all(np.isfinite(target)):
        return 0, float("inf")
    d = np.linalg.norm(cloud - np.asarray(target, np.float32)[None, :3], axis=1)
    return int((d <= r).sum()), float(d.min())


def load_cycles():
    out = []
    for d in sorted(glob.glob(RUNS)):
        jf = os.path.join(d, "decisions.jsonl")
        if not os.path.exists(jf):
            continue
        for r in [json.loads(l) for l in open(jf)]:
            if r["kind"] != "gaze":
                continue
            f = os.path.join(d, r["npz"])
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            p = z["points"].astype(np.float32)
            p = p[np.abs(p).sum(-1) > 1e-6]
            if len(p) < 50:
                continue
            out.append({"run": os.path.basename(d)[-6:], "cycle": int(r["cycle"]),
                        "pts": p, "deployed": np.asarray(r["target"][:3], np.float32),
                        "shift": float(z["rack_y_shift"]) if "rack_y_shift" in z.files else -0.55})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=16, help="noise draws per cloud (0 = clean only)")
    ap.add_argument("--coeff", type=float, default=0.0014)
    ap.add_argument("--cam", default="logs/bc_pointcloud/bc_margin05_2048_500/cam_pos_base.npy")
    ap.add_argument("--scoring", default="logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt")
    ap.add_argument("--reg", default="logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt")
    ap.add_argument("--fail_sup", type=int, default=15, help="support below this = mis-target")
    ap.add_argument("--out", default="logs/trial_failure_counterfactual.json")
    a = ap.parse_args()

    cam = np.load(a.cam).astype(np.float32).reshape(3)
    sc = ScoringGraspPolicy(num_points=2048)
    sc.load_state_dict(torch.load(a.scoring, map_location="cpu", weights_only=False)["model_state_dict"])
    rg = RegressionBC()
    rg.load_state_dict(torch.load(a.reg, map_location="cpu", weights_only=False)["model_state_dict"])
    sc.eval(); rg.eval()

    cycles = load_cycles()
    print(f"{len(cycles)} recorded grasp cycles from the 2026-08-03 baseline trials\n")

    recs = []
    for c in cycles:
        p = c["pts"]
        shifted = p.copy(); shifted[:, 1] -= c["shift"]      # into training coords
        inp = torch.from_numpy(pad_to(shifted, 2048))[None]
        with torch.no_grad():
            t_sc = np.asarray(sc.act(inp)[0].numpy()[:3], np.float32)
            t_rg = dec5(rg(inp)[0].numpy())
        # back to the recorded frame so support is measured against the saved cloud
        t_sc[1] += c["shift"]; t_rg[1] += c["shift"]
        t_he = heuristic_target(p)

        row = {"run": c["run"], "cycle": c["cycle"], "n": int(len(p))}
        for name, t in (("deployed", c["deployed"]), ("heuristic", t_he),
                        ("bc_reg", t_rg), ("scoring", t_sc)):
            s, nn = support_of(t, p)
            row[name] = {"sup": s, "nn": nn, "t": [float(x) for x in np.asarray(t)[:3]]}
        recs.append(row)

    fails = [r for r in recs if r["deployed"]["sup"] < a.fail_sup]
    print(f"MIS-TARGETED cycles (deployed support < {a.fail_sup}): "
          f"{len(fails)} of {len(recs)} ({100*len(fails)/len(recs):.0f}%)\n")

    hdr = f"{'run/cycle':13} {'deployed':>9} {'heuristic':>10} {'bc_reg':>8} {'scoring':>8}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(fails, key=lambda r: r["deployed"]["sup"]):
        print(f"{r['run']}/c{r['cycle']:02d}    {r['deployed']['sup']:9d} "
              f"{r['heuristic']['sup']:10d} {r['bc_reg']['sup']:8d} {r['scoring']['sup']:8d}")

    print(f"\n{'set':22} {'policy':11} {'mean support':>13} {'recovered%':>11}")
    print("-" * 60)
    out = {}
    for label, subset in (("mis-targeted only", fails), ("all cycles", recs)):
        for k in ("deployed", "heuristic", "bc_reg", "scoring"):
            sup = np.array([r[k]["sup"] for r in subset], float)
            rec = 100.0 * float(np.mean(sup >= a.fail_sup))
            out[f"{label}/{k}"] = {"mean_support": float(sup.mean()), "ok_pct": rec,
                                   "n": len(subset)}
            print(f"{label:22} {k:11} {sup.mean():13.1f} {rec:10.0f}%")
        print()

    if a.draws:
        print(f"Noise stability on the {len(fails)} mis-targeted clouds ({a.draws} draws):")
        print(f"{'policy':11} {'spread(m)':>10} {'bistable%':>10} {'mean support':>13}")
        print("-" * 47)
        agg = {k: {"sp": [], "bi": [], "su": []} for k in ("heuristic", "bc_reg", "scoring")}
        for r in fails:
            c = next(x for x in cycles if x["run"] == r["run"] and x["cycle"] == r["cycle"])
            p = c["pts"]; rng = np.random.default_rng(0)
            T = {k: [] for k in agg}
            for _ in range(a.draws):
                q = zed_noise(p, cam, a.coeff, rng)
                qs = q.copy(); qs[:, 1] -= c["shift"]
                inp = torch.from_numpy(pad_to(qs, 2048))[None]
                with torch.no_grad():
                    t = np.asarray(sc.act(inp)[0].numpy()[:3], np.float32); t[1] += c["shift"]
                    T["scoring"].append(t)
                    t = dec5(rg(inp)[0].numpy()); t[1] += c["shift"]
                    T["bc_reg"].append(t)
                T["heuristic"].append(heuristic_target(q))
            for k, v in T.items():
                V = np.asarray([x for x in v if np.all(np.isfinite(x))], np.float32)
                if len(V) < 2:
                    continue
                agg[k]["sp"].append(float(np.linalg.norm(V - V.mean(0), axis=1).mean()))
                agg[k]["bi"].append(1.0 if n_modes(V) > 1 else 0.0)
                agg[k]["su"].append(float(np.mean([support_of(x, p)[0] for x in V])))
        for k, v in agg.items():
            if v["sp"]:
                print(f"{k:11} {np.mean(v['sp']):10.3f} {100*np.mean(v['bi']):10.1f} "
                      f"{np.mean(v['su']):13.1f}")
                out[f"noise/{k}"] = {"spread": float(np.mean(v["sp"])),
                                     "bistable_pct": float(100 * np.mean(v["bi"])),
                                     "support": float(np.mean(v["su"]))}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"summary": out, "per_cycle": recs}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("CAVEAT: recorded clouds are the TIGHT 1024 crop, so the margin-trained policies run\n"
          "zero-padded with no margin context - a handicap, so their margin here is a LOWER BOUND.")


if __name__ == "__main__":
    main()
