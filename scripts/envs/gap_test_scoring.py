#!/usr/bin/env python3
"""Offline gap test: does the scoring head commit to a mound where the regression head averages?

Replays recorded real clouds (policy_debug_*.npz from a crane run) through both heads and reports
where each one aims. On the 2026-08-03 double-mound run the regression BC repeatedly targeted the
empty gap between the two mounds; the argmax controllers never do (0.0% in sim).

    python3 scripts/envs/gap_test_scoring.py \
        --run logs/bc_pointcloud/bc_aug1v2_dig25/policy_debug/run_20260803_142901 \
        --scoring logs/bc_pointcloud/scoring_v1/scoring_policy.pt \
        --bc logs/bc_pointcloud/bc_aug1v2_dig25/bc_policy.pt
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


def local_support(pts: np.ndarray, xy: np.ndarray, r: float = 0.5) -> int:
    """How many observed points sit within r of the commanded xy - 0 means aiming at nothing."""
    return int((np.linalg.norm(pts[:, :2] - xy[None, :], axis=1) < r).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="dir of policy_debug_*.npz")
    p.add_argument("--scoring", required=True)
    p.add_argument("--bc", default=None, help="optional regression BC checkpoint for comparison")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--support_radius", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.scoring, map_location=dev, weights_only=False)
    model = ScoringGraspPolicy(num_points=ck.get("num_points", a.num_points)).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    bc = None
    if a.bc and os.path.exists(a.bc):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from bc_pointcloud_policy import BCPointCloudPolicy  # noqa: E402
        bck = torch.load(a.bc, map_location=dev, weights_only=False)
        bc = BCPointCloudPolicy(num_points=bck.get("num_points", a.num_points),
                                action_dim=bck.get("action_dim", 5)).to(dev)
        bc.load_state_dict(bck["model_state_dict"])
        bc.eval()

    files = sorted(glob.glob(os.path.join(a.run, "policy_debug_*.npz")))
    print(f"[gap] {len(files)} recorded clouds from {a.run}\n")
    hdr = (f"{'cloud':>6} {'pts':>5} | {'recorded target':>22} {'sup':>4} | "
           f"{'scoring head':>22} {'sup':>4}")
    print(hdr + ("" if bc is None else f" | {'BC regression':>22} {'sup':>4}"))
    print("-" * (len(hdr) + (0 if bc is None else 32)))

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        # npz stores the PHYSICAL cropped cloud and the PHYSICAL executed target; the node's
        # shim subtracts rack_y_shift before the net and adds it back after. Replicate that or
        # the policy is evaluated 0.55 m off its training frame (and it is not shift-robust).
        shift = float(d["rack_y_shift"]) if "rack_y_shift" in d else 0.0
        pts = d["points"].astype(np.float32).copy()
        pts[:, 1] -= shift
        rec = d["target"].astype(np.float32).copy()
        rec[1] -= shift

        x = np.zeros((a.num_points, 3), dtype=np.float32)
        n = min(len(pts), a.num_points)
        sel = np.random.default_rng(0).choice(len(pts), n, replace=False) if len(pts) > a.num_points \
            else np.arange(len(pts))
        x[:n] = pts[sel]
        xb = torch.from_numpy(x).unsqueeze(0).to(dev)

        with torch.no_grad():
            sc = model.act(xb)[0].cpu().numpy()
        s_sup = local_support(pts, sc[:2], a.support_radius)
        r_sup = local_support(pts, rec[:2], a.support_radius)

        line = (f"{os.path.basename(f)[13:16]:>6} {len(pts):5d} | "
                f"({rec[0]:6.2f},{rec[1]:6.2f},{rec[2]:6.2f}) {r_sup:4d} | "
                f"({sc[0]:6.2f},{sc[1]:6.2f},{sc[2]:6.2f}) {s_sup:4d}")
        row = dict(f=os.path.basename(f), rec=rec, sc=sc, r_sup=r_sup, s_sup=s_sup, pts=pts)

        if bc is not None:
            with torch.no_grad():
                act = bc(xb)[0].cpu().numpy()
            bmin, bmax = d["bounds_min"], d["bounds_max"]
            xyz = bmin + (np.tanh(act[:3]) + 1.0) / 2.0 * (bmax - bmin)
            b_sup = local_support(pts, xyz[:2], a.support_radius)
            line += f" | ({xyz[0]:6.2f},{xyz[1]:6.2f},{xyz[2]:6.2f}) {b_sup:4d}"
            row.update(bc=xyz, b_sup=b_sup)
        print(line)
        rows.append(row)

    print("\nSUPPORT = observed points within "
          f"{a.support_radius} m of the commanded xy. 0 support = aiming at empty space.")
    for key, name in [("r_sup", "recorded (deployed BC)"), ("s_sup", "scoring head"),
                      ("b_sup", "BC regression (replay)")]:
        v = [r[key] for r in rows if key in r]
        if not v:
            continue
        v = np.array(v)
        print(f"  {name:24s} median support {np.median(v):6.0f} | "
              f"zero-support {100*(v == 0).mean():4.0f}% | "
              f"under-20 {100*(v < 20).mean():4.0f}%")


if __name__ == "__main__":
    main()
