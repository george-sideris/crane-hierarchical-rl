#!/usr/bin/env python3
"""THE stump test: did pole-variation training teach the policy that the broken-pole base is
not a log? Runs A (and optionally B) scoring checkpoints over the raw trial clouds through the
margin pipeline and reports, per cloud:

  * the chosen target + its support
  * whether the target lands in the STUMP ZONE (auto-located on the single-mound clouds as the
    elevated cluster in the known band y_train [4.6, 5.25], z > -1.25)
  * the stump cluster's best SCORE PERCENTILE among all valid points - the mechanistic readout:
    rank ~100% = the stump wins the argmax; the goal is to push it well below the pile.

    python3 scripts/envs/stump_test.py \
        --a logs/bc_pointcloud/scoring_margin05_2048/scoring_policy.pt \
        --b logs/bc_pointcloud/scoring_margin05_2048_stub/scoring_policy.pt
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
from render_policy_comparison import fps, crop, pad_to, SH, RAW  # noqa: E402

STUMP_Y = (4.6, 5.25)          # training coords (shift -0.55); confirmed on single_c09-c14


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = ScoringGraspPolicy(num_points=ck.get("num_points", 2048))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, int(ck.get("num_points", 2048))


def stump_cluster(cloud):
    """Elevated points in the known band -> (centroid_xy, mask) or None."""
    v = cloud[np.abs(cloud).sum(-1) > 1e-6]
    m = (v[:, 1] > STUMP_Y[0]) & (v[:, 1] < STUMP_Y[1]) & (v[:, 2] > -1.25) & (v[:, 2] < -0.55)
    if m.sum() < 8:
        return None
    c = v[m][:, :2].mean(0)
    return c, v


def analyse(model, npts, cloud):
    x = torch.from_numpy(pad_to(cloud, npts))[None]
    with torch.no_grad():
        t = model.act(x)[0].numpy()
        score, _, _, pts = model.forward(x)
    s = score[0].numpy()
    p = pts[0].numpy()
    valid = np.abs(p).sum(-1) > 1e-6
    sup = int((np.linalg.norm(cloud[:, :2] - t[:2][None, :], axis=1) < 0.5).sum())
    out = {"t": t, "sup": sup, "stump_pct": None, "in_stump": None}
    sc = stump_cluster(cloud)
    if sc is not None:
        c, _ = sc
        near = valid & (np.linalg.norm(p[:, :2] - c[None, :], axis=1) < 0.35)
        if near.sum() >= 5:
            best_stump = s[near].max()
            out["stump_pct"] = float((s[valid] < best_stump).mean() * 100)
            out["in_stump"] = bool(np.hypot(t[0] - c[0], t[1] - c[1]) < 0.45)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline scoring checkpoint (e.g. P2)")
    ap.add_argument("--b", default=None, help="candidate checkpoint (e.g. P2b stub-aug)")
    args = ap.parse_args()

    A, na = load(args.a)
    B, nb = load(args.b) if args.b else (None, 0)
    names = [("A", A, na)] + ([("B", B, nb)] if B else [])
    print(f"A = {args.a}\nB = {args.b}\n")
    hdr = f"{'cloud':16s}"
    for nm, _, _ in names:
        hdr += f" | {nm}: {'y':>6} {'z':>6} {'sup':>4} {'stump%':>7} {'IN?':>4}"
    print(hdr)
    hits = {nm: [0, 0] for nm, _, _ in names}
    pcts = {nm: [] for nm, _, _ in names}
    for f in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        p = np.load(f)["points"].astype(np.float32).copy()
        p[:, 1] -= SH
        cloud = fps(crop(p, 0.5), 2048)
        cloud = cloud[np.abs(cloud).sum(-1) > 1e-6]
        line = f"{os.path.basename(f)[:-4]:16s}"
        for nm, m, npts in names:
            r = analyse(m, npts, cloud)
            pct = f"{r['stump_pct']:7.1f}" if r["stump_pct"] is not None else f"{'-':>7}"
            ins = ("YES" if r["in_stump"] else "no") if r["in_stump"] is not None else "-"
            line += (f" | {nm}: {r['t'][1]:6.2f} {r['t'][2]:6.2f} {r['sup']:4d} "
                     f"{pct} {ins:>4}")
            if r["in_stump"] is not None:
                hits[nm][1] += 1
                hits[nm][0] += int(r["in_stump"])
            if r["stump_pct"] is not None:
                pcts[nm].append(r["stump_pct"])
        print(line)
    print()
    for nm, _, _ in names:
        if hits[nm][1]:
            print(f"[{nm}] targets IN the stump zone on {hits[nm][0]}/{hits[nm][1]} "
                  f"stump-bearing clouds | stump score percentile: "
                  f"median {np.median(pcts[nm]):.1f} (100 = stump wins the argmax)")


if __name__ == "__main__":
    main()
