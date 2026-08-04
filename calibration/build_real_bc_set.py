#!/usr/bin/env python3
"""Build a BC fine-tune set from REAL deployed-run policy inputs.

Collects policy_debug_*.npz clouds from crane_policy_node run dirs (each is the exact
1024-pt base-frame input the policy saw on the crane), strips the rack poles (older runs
predate the deploy filter), and auto-labels each cloud with the same geometric rule the
sim expert follows: grasp the highest region, at log-center depth, yaw along the log axis.

Labels are arctanh(normalized) 5D (x, y, z, cos 2yaw, sin 2yaw), identical encoding to the
sim dataset, so train_bc_pointcloud.py --train_only can consume the output directly.

Usage:
    python3 build_real_bc_set.py --runs <run_dir_or_parent> [...] --out <dataset_dir>
        [--exclude <substring>] [--log_radius 0.056]
"""
import argparse
import glob
import json
import os

import numpy as np

BMIN = np.array([-5.364, -1.684, -1.30])
BMAX = np.array([-3.364, 5.316, 0.10])


def strip_poles(p):
    keep = ~(((p[:, 0] > -3.6) & (p[:, 1] < -1.35)) | ((p[:, 0] > -3.8) & (p[:, 1] > 5.1)))
    return p[keep]


def auto_label(p, log_radius):
    """Expert rule on a real cloud: target the highest local region.

    Returns (x, y, z, yaw) or None if the cloud has no credible pile region.
    Robust top: the highest point whose 0.25 m xy-neighborhood holds >= 8 points within
    15 cm of it in z (rejects isolated speckle). Label z = ridge top - log_radius (center).
    Yaw = principal direction of the top region (logs are long -> dominant axis).
    """
    if len(p) < 100:
        return None
    order = np.argsort(p[:, 2])[::-1]
    for j in order[:60]:
        c = p[j]
        dxy = np.linalg.norm(p[:, :2] - c[:2], axis=1)
        nb = p[(dxy < 0.25) & (np.abs(p[:, 2] - c[2]) < 0.15)]
        if len(nb) >= 8:
            top = nb
            break
    else:
        return None
    x, y = top[:, 0].mean(), top[:, 1].mean()
    z = top[:, 2].max() - log_radius
    # log axis: PCA over a wider slab around the ridge (a single log is long in one direction)
    dxy = np.linalg.norm(p[:, :2] - np.array([x, y]), axis=1)
    slab = p[(dxy < 0.6) & (p[:, 2] > top[:, 2].max() - 0.18)]
    if len(slab) >= 12:
        xy = slab[:, :2] - slab[:, :2].mean(0)
        w, v = np.linalg.eigh(xy.T @ xy)
        ax = v[:, -1]
        yaw = float(np.arctan2(ax[1], ax[0]))
    else:
        yaw = np.pi / 2  # logs lie along y on this rack
    # wrap to the +-pi/2 log-axis convention (cos/sin of 2*yaw is symmetric anyway)
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    return float(x), float(y), float(z), yaw


def encode(x, y, z, yaw):
    n = 2.0 * (np.array([x, y, z]) - BMIN) / (BMAX - BMIN) - 1.0
    n = np.clip(n, -0.999, 0.999)
    tgt = np.concatenate([n, [np.cos(2 * yaw), np.sin(2 * yaw)]])
    tgt = np.clip(tgt, -0.999, 0.999)
    return np.arctanh(tgt).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs (or parents of run_*/) holding policy_debug_*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None, help="skip npz paths containing this substring")
    ap.add_argument("--log_radius", type=float, default=0.056,
                    help="label depth below the ridge top (sim expert: log center = radius)")
    ap.add_argument("--num_points", type=int, default=1024)
    args = ap.parse_args()

    files = []
    for r in args.runs:
        files += glob.glob(os.path.join(r, "policy_debug_*.npz"))
        files += glob.glob(os.path.join(r, "run_*", "policy_debug_*.npz"))
    files = sorted(set(files))
    if args.exclude:
        files = [f for f in files if args.exclude not in f]
    print(f"{len(files)} candidate clouds")

    clouds, actions, meta = [], [], []
    for f in files:
        d = np.load(f)
        p = d["points"].reshape(-1, 3).astype(np.float32)
        p = p[np.abs(p).sum(axis=1) > 0]
        p = strip_poles(p)
        lab = auto_label(p, args.log_radius)
        if lab is None:
            print(f"  skip (no pile): {f}")
            continue
        x, y, z, yaw = lab
        # pad/trim back to num_points (pole strip removed a few)
        if len(p) < args.num_points:
            pad = np.zeros((args.num_points - len(p), 3), np.float32)
            p = np.vstack([p, pad])
        else:
            p = p[:args.num_points]
        clouds.append(p)
        actions.append(encode(x, y, z, yaw))
        meta.append({"src": f, "label": [x, y, z, yaw]})

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "pointclouds.npy"), np.stack(clouds))
    np.save(os.path.join(args.out, "actions.npy"), np.stack(actions))
    np.save(os.path.join(args.out, "episode_rewards.npy"), np.zeros(len(clouds)))
    md = {"num_points": args.num_points, "obs_dim": args.num_points * 3, "action_dim": 5,
          "num_samples": len(clouds), "gaze": True, "raw_pcd": True, "crop_to_bounds": True,
          "real_data": True, "log_radius": args.log_radius,
          "sources": [m["src"] for m in meta]}
    with open(os.path.join(args.out, "metadata.json"), "w") as fp:
        json.dump(md, fp, indent=2)
    with open(os.path.join(args.out, "labels_debug.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    labs = np.array([m["label"] for m in meta])
    print(f"\nwrote {len(clouds)} samples -> {args.out}")
    print(f"label z: {labs[:,2].min():+.2f}..{labs[:,2].max():+.2f}  "
          f"y: {labs[:,1].min():+.2f}..{labs[:,1].max():+.2f}")


if __name__ == "__main__":
    main()
