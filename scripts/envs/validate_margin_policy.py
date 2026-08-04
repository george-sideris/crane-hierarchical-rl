#!/usr/bin/env python3
"""Pre-deployment gate: run a scoring policy over REAL crane clouds and stress it with noise.

No policy goes on the crane without passing this. Three checks, all on real data:

  1. RECORDED clouds (policy_debug npz, tight-crop, node pipeline): does the policy commit to
     observed material (support) and to a mound rather than the gap (the deployed regression
     BC's 2026-08-03 failure)?
  2. NOISE STRESS: K perturbation draws per cloud -> how far does the chosen target move, and
     does support survive? This is the failure the sim sweep exposed (raw argmax: 93% -> 50%
     grasp success at 1x ZED noise); neighbourhood aggregation is the fix under test.
  3. RAW bag clouds (bag_cloud_extract output): emulate the DEPLOYED pipeline for a margin
     policy - margin crop, shift to training coords, FPS to the checkpoint's num_points -
     and check it does not target tall structure (the rack end-board that absorbed the real
     heuristic's endgame).

    python3 scripts/envs/validate_margin_policy.py --scoring <ckpt> [--margin 0.5]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
RECORDED = "logs/bc_pointcloud/bc_aug1v2_dig25/policy_debug/run_20260803_142901"
RAW_DIR = "logs/real_raw_clouds"          # bag_cloud_extract outputs (extracted ahead of time)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ScoringGraspPolicy(num_points=ck.get("num_points", 1024))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, int(ck.get("num_points", 1024))


def support(pts, xy, r=0.5):
    return int((np.linalg.norm(pts[:, :2] - xy[None, :], axis=1) < r).sum())


def fps(p, k, seed=0):
    if len(p) <= k:
        out = np.zeros((k, 3), np.float32); out[:len(p)] = p; return out
    rng = np.random.default_rng(seed)
    if len(p) > 20000:                       # pre-thin: FPS on 90k raw points is needlessly slow
        p = p[rng.choice(len(p), 20000, replace=False)]
    idx = np.empty(k, np.int64); idx[0] = rng.integers(len(p))
    d = np.linalg.norm(p - p[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(p - p[idx[i]], axis=1))
    return p[idx].astype(np.float32)


def act(m, pts, num_points, agg):
    x = np.zeros((num_points, 3), np.float32)
    n = min(len(pts), num_points)
    x[:n] = pts[:n]
    with torch.no_grad():
        return m.act(torch.from_numpy(x)[None], support_frac=agg)[0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", required=True)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="observation-crop margin the policy was TRAINED with (0 = tight)")
    ap.add_argument("--agg", type=float, default=0.25, help="support_frac for the gate (0 disables)")
    ap.add_argument("--noise_draws", type=int, default=10)
    ap.add_argument("--noise_sigma", type=float, default=0.03)
    a = ap.parse_args()

    m, npts = load_model(a.scoring)
    print(f"[gate] {a.scoring} (num_points={npts}, margin={a.margin}, agg={a.agg})\n")

    # ---- 1+2: recorded clouds (tight pipeline; for a margin policy these UNDER-represent
    # structure, so treat them as the commitment/noise check, not the structure check)
    files = sorted(glob.glob(os.path.join(RECORDED, "policy_debug_*.npz")))
    rng = np.random.default_rng(7)
    rows = []
    for f in files:
        d = np.load(f)
        pts = d["points"].astype(np.float32).copy()
        pts[:, 1] -= float(d["rack_y_shift"])
        base = {}
        for tag, r in (("raw", 0.0), ("gate", a.agg)):
            t = act(m, pts, npts, r)
            base[tag] = (t, support(pts, t[:2]))
        scat = {"raw": [], "gate": []}
        supm = {"raw": [], "gate": []}
        for k in range(a.noise_draws):
            noisy = pts + rng.normal(0, a.noise_sigma, pts.shape).astype(np.float32)
            for tag, r in (("raw", 0.0), ("gate", a.agg)):
                t = act(m, noisy, npts, r)
                scat[tag].append(np.linalg.norm(t[:2] - base[tag][0][:2]))
                supm[tag].append(support(pts, t[:2]))
        rows.append((os.path.basename(f)[13:16], base, scat, supm))

    print("=== recorded real clouds: commitment + noise stress "
          f"({a.noise_draws} draws, sigma {a.noise_sigma*100:.0f} cm) ===")
    print(f"{'cl':>4} | {'raw target y':>12} {'sup':>4} {'scatter':>8} {'minsup':>6} | "
          f"{'gated target y':>12} {'sup':>4} {'scatter':>8} {'minsup':>6}")
    stats = {"raw": [], "gate": []}
    for cl, base, scat, supm in rows:
        line = f"{cl:>4} |"
        for tag in ("raw", "gate"):
            t, s = base[tag]
            line += (f" {t[1]:12.2f} {s:4d} {np.median(scat[tag]):8.3f} "
                     f"{int(np.min(supm[tag])):6d} |")
            stats[tag].append((s, np.median(scat[tag]), np.min(supm[tag])))
        print(line)
    for tag in ("raw", "gate"):
        st = np.array(stats[tag])
        print(f"[{tag:3s}] support median {np.median(st[:,0]):4.0f} | zero-support "
              f"{100*(st[:,0]==0).mean():3.0f}% | scatter median {np.median(st[:,1])*100:4.1f} cm "
              f"| worst-draw support<20: {100*(st[:,2]<20).mean():3.0f}% of clouds")

    # ---- 3: raw bag clouds through the deployed margin pipeline
    raws = sorted(glob.glob(os.path.join(RAW_DIR, "*.npz")))
    if raws:
        print(f"\n=== raw bag clouds via margin pipeline ({len(raws)} clouds) ===")
        g = a.margin
        for f in raws:
            p = np.load(f)["points"].astype(np.float32)
            sh = -0.55
            lo, hi = B_MIN + [0, sh, 0], B_MAX + [0, sh, 0]
            mask = ((p[:, 0] >= lo[0]-g) & (p[:, 0] <= hi[0]+g) &
                    (p[:, 1] >= lo[1]-g) & (p[:, 1] <= hi[1]+g) &
                    (p[:, 2] >= lo[2]-g) & (p[:, 2] <= hi[2]))
            q = p[mask].copy(); q[:, 1] -= sh
            cloud = fps(q, npts)
            t = act(m, cloud, npts, a.agg)
            sup = support(q, t[:2])
            # tall structure = anything 0.4 m above the local pile p90; targeting within 0.4 m
            # of a tall column is the heuristic's rack-grab failure
            zref = np.percentile(q[:, 2], 90)
            tall = q[q[:, 2] > zref + 0.4]
            near_tall = (len(tall) > 0 and
                         np.linalg.norm(tall[:, :2] - t[:2], axis=1).min() < 0.4)
            print(f"  {os.path.basename(f):34s} -> y={t[1]:6.2f} z={t[2]:6.2f} sup={sup:4d} "
                  f"{'!! NEAR TALL STRUCTURE' if near_tall else 'clear of structure'}")
    else:
        print(f"\n[gate] no raw clouds in {RAW_DIR} - structure check skipped")


if __name__ == "__main__":
    main()
