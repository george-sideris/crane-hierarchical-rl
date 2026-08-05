#!/usr/bin/env python3
"""Concatenate two BC collection dirs into one (paired arrays only, order preserved).

Collections are time-seeded (cfg.seed=0 -> pattern_seed from time.time per reset), so runs are
disjoint by construction and plain concatenation is valid. Used to top up the 450-episode
margin/2048 collection to 500.

    python3 scripts/envs/merge_bc_datasets.py --a <dir450> --b <dir50> --out <dir500>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

PAIRED = ["pointclouds.npy", "actions.npy", "pointclouds_raw.npy", "pointclouds_full.npy"]
CONCAT1D = ["episode_rewards.npy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n_a = n_b = None
    for f in PAIRED:
        pa, pb = os.path.join(args.a, f), os.path.join(args.b, f)
        if not os.path.exists(pa):
            continue
        if not os.path.exists(pb):
            raise SystemExit(f"[merge] {f} present in --a but missing in --b; refusing a "
                             f"half-paired merge")
        A, B = np.load(pa, mmap_mode="r"), np.load(pb, mmap_mode="r")
        if A.shape[1:] != B.shape[1:] or A.dtype != B.dtype:
            raise SystemExit(f"[merge] {f}: shape/dtype mismatch {A.shape}{A.dtype} vs "
                             f"{B.shape}{B.dtype} - the runs were not collected identically")
        n_a, n_b = len(A), len(B)
        np.save(os.path.join(args.out, f), np.concatenate([A, B], axis=0))
        print(f"[merge] {f}: {len(A)} + {len(B)} -> {len(A) + len(B)}")
    for f in CONCAT1D:
        pa, pb = os.path.join(args.a, f), os.path.join(args.b, f)
        if os.path.exists(pa) and os.path.exists(pb):
            np.save(os.path.join(args.out, f),
                    np.concatenate([np.load(pa), np.load(pb)]))
    # camera pose is a per-dataset constant (same rig) - carry it over
    cam = os.path.join(args.a, "cam_pos_base.npy")
    if os.path.exists(cam):
        shutil.copy(cam, os.path.join(args.out, "cam_pos_base.npy"))
    # merged metadata = --a's, with provenance
    mp = os.path.join(args.a, "metadata.json")
    if os.path.exists(mp):
        md = json.load(open(mp))
        md["num_samples"] = (n_a or 0) + (n_b or 0)
        md["merged_from"] = {"a": args.a, "b": args.b, "n_a": n_a, "n_b": n_b}
        json.dump(md, open(os.path.join(args.out, "metadata.json"), "w"), indent=2)
    print(f"[merge] -> {args.out}")


if __name__ == "__main__":
    main()
