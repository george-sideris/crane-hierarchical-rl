#!/usr/bin/env python3
"""
Merge BC point-cloud collection directories into one dataset.

train_bc_pointcloud.py collects each run into its own directory and OVERWRITES
on re-collect, so new demos never accumulate automatically. This helper
concatenates the pointclouds/actions from several collection dirs into a single
dir that --train_only can read, after checking their obs settings match.

Each input dir is expected to contain (as written by train_bc_pointcloud.py):
    pointclouds.npy     (N, num_points, 3)   required
    actions.npy         (N, 5)               required
    metadata.json                            required
    episode_rewards.npy                      optional
    cam_pos_base.npy    (3,)                  optional (ZED-noise aug apex)

Usage:
    python crane_testbed/scripts/envs/merge_bc_data.py \\
        logs/bc_gaze/run1 logs/bc_gaze/run2 -o logs/bc_gaze/merged

Then train on the merged dir:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_pointcloud.py \\
        --train_only logs/bc_gaze/merged --output_dir logs/bc_gaze/merged --epochs 200
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

# Preprocessing flags that MUST match across runs (from metadata; not encoded in
# the arrays). num_points and action_dim are validated from the ACTUAL array
# shapes instead of metadata, because metadata can be stale (e.g. a dir whose
# actions were converted 4D->5D cossin without rewriting metadata.json).
_FLAG_KEYS = ("gaze", "raw_pcd", "crop_to_bounds", "depth_range")


def _load_dir(d):
    """Load one collection dir. Returns a dict; raises on missing required files."""
    pc_path = os.path.join(d, "pointclouds.npy")
    act_path = os.path.join(d, "actions.npy")
    meta_path = os.path.join(d, "metadata.json")
    for p in (pc_path, act_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{d}: missing required file {os.path.basename(p)}")

    pointclouds = np.load(pc_path)
    actions = np.load(act_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    if pointclouds.shape[0] != actions.shape[0]:
        raise ValueError(
            f"{d}: pointclouds ({pointclouds.shape[0]}) and actions "
            f"({actions.shape[0]}) have different sample counts")

    rewards_path = os.path.join(d, "episode_rewards.npy")
    rewards = np.load(rewards_path) if os.path.exists(rewards_path) else None

    cam_path = os.path.join(d, "cam_pos_base.npy")
    cam_pos = np.load(cam_path) if os.path.exists(cam_path) else None

    return {
        "dir": d,
        "pointclouds": pointclouds,
        "actions": actions,
        "metadata": metadata,
        "rewards": rewards,
        "cam_pos": cam_pos,
    }


def _flags(metadata):
    return {k: metadata.get(k) for k in _FLAG_KEYS}


def main():
    parser = argparse.ArgumentParser(
        description="Merge BC point-cloud collection dirs into one --train_only dataset")
    parser.add_argument("inputs", nargs="+", help="Collection dirs to merge (2 or more)")
    parser.add_argument("-o", "--output", required=True, help="Output dir for the merged dataset")
    parser.add_argument("--allow-metadata-mismatch", action="store_true",
                        help="Merge even if obs settings (num_points, gaze, bounds, ...) differ. "
                             "Unsafe: mixed preprocessing corrupts training. Off by default.")
    parser.add_argument("--cam-tol", type=float, default=0.05,
                        help="Warn if cam_pos_base differs by more than this (m) across runs.")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        print("[merge-bc] Need at least 2 input dirs to merge.", file=sys.stderr)
        return 1

    out = os.path.abspath(args.output)
    if out in {os.path.abspath(d) for d in args.inputs}:
        print("[merge-bc] Output dir must not be one of the input dirs (avoids overwriting "
              "source data).", file=sys.stderr)
        return 1

    # Load all runs.
    runs = []
    for d in args.inputs:
        try:
            runs.append(_load_dir(d))
        except (FileNotFoundError, ValueError) as e:
            print(f"[merge-bc] ERROR {e}", file=sys.stderr)
            return 1

    # Warn when metadata disagrees with the actual arrays (stale metadata).
    for r in runs:
        meta_pts = r["metadata"].get("num_points")
        meta_ad = r["metadata"].get("action_dim")
        pts = r["pointclouds"].shape[1]
        ad = r["actions"].shape[1]
        if meta_pts is not None and meta_pts != pts:
            print(f"[merge-bc] WARNING {r['dir']}: metadata num_points={meta_pts} but "
                  f"pointclouds have {pts}; trusting the array.")
        if meta_ad is not None and meta_ad != ad:
            print(f"[merge-bc] WARNING {r['dir']}: metadata action_dim={meta_ad} but "
                  f"actions have {ad}; trusting the array.")

    # Validate preprocessing flags match across runs (from metadata).
    ref = _flags(runs[0]["metadata"])
    mismatches = []
    for r in runs[1:]:
        cur = _flags(r["metadata"])
        if cur != ref:
            diff = {k: (ref[k], cur[k]) for k in _FLAG_KEYS if cur.get(k) != ref.get(k)}
            mismatches.append((r["dir"], diff))
    if mismatches:
        print(f"[merge-bc] Preprocessing-flag mismatch vs {runs[0]['dir']}:", file=sys.stderr)
        for d, diff in mismatches:
            for k, (a, b) in diff.items():
                print(f"    {d}: {k} = {b!r}  (expected {a!r})", file=sys.stderr)
        if not args.allow_metadata_mismatch:
            print("[merge-bc] Aborting. Re-run with --allow-metadata-mismatch only if you are "
                  "sure the preprocessing is compatible.", file=sys.stderr)
            return 1
        print("[merge-bc] Proceeding despite mismatch (--allow-metadata-mismatch).",
              file=sys.stderr)

    # Validate array shapes are stackable (ground truth for num_points + action_dim).
    pc_tail = runs[0]["pointclouds"].shape[1:]
    act_tail = runs[0]["actions"].shape[1:]
    for r in runs[1:]:
        if r["pointclouds"].shape[1:] != pc_tail:
            print(f"[merge-bc] ERROR {r['dir']}: pointclouds shape {r['pointclouds'].shape} "
                  f"incompatible with {pc_tail} (num_points differs)", file=sys.stderr)
            return 1
        if r["actions"].shape[1:] != act_tail:
            print(f"[merge-bc] ERROR {r['dir']}: actions shape {r['actions'].shape} "
                  f"incompatible with {act_tail} (action_dim differs)", file=sys.stderr)
            return 1

    # Concatenate.
    pointclouds = np.concatenate([r["pointclouds"] for r in runs], axis=0)
    actions = np.concatenate([r["actions"] for r in runs], axis=0)

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "pointclouds.npy"), pointclouds)
    np.save(os.path.join(out, "actions.npy"), actions)

    # episode_rewards: only merge if every run has it (otherwise the record is partial).
    if all(r["rewards"] is not None for r in runs):
        rewards = np.concatenate([r["rewards"] for r in runs], axis=0)
        np.save(os.path.join(out, "episode_rewards.npy"), rewards)
    else:
        missing = [r["dir"] for r in runs if r["rewards"] is None]
        print(f"[merge-bc] Skipping episode_rewards.npy (absent in: {', '.join(missing)}).")

    # cam_pos_base: single (3,) gaze apex; copy the first run's and warn if they diverge.
    cam_runs = [r for r in runs if r["cam_pos"] is not None]
    if cam_runs:
        cam0 = np.asarray(cam_runs[0]["cam_pos"], dtype=np.float64)
        for r in cam_runs[1:]:
            d = float(np.max(np.abs(np.asarray(r["cam_pos"], dtype=np.float64) - cam0)))
            if d > args.cam_tol:
                print(f"[merge-bc] WARNING cam_pos_base in {r['dir']} differs by {d:.3f} m from "
                      f"{cam_runs[0]['dir']} (> {args.cam_tol} m). ZED-noise aug uses ONE apex for "
                      f"all samples, so mixed viewpoints may be inaccurate.")
        shutil.copy(os.path.join(cam_runs[0]["dir"], "cam_pos_base.npy"),
                    os.path.join(out, "cam_pos_base.npy"))
        print(f"[merge-bc] cam_pos_base.npy copied from {cam_runs[0]['dir']} "
              f"(needed for --zed_noise training).")
        if len(cam_runs) < len(runs):
            print(f"[merge-bc] Note: {len(runs) - len(cam_runs)} run(s) had no cam_pos_base.npy; "
                  f"used the above for all merged samples.")
    else:
        print("[merge-bc] WARNING no cam_pos_base.npy in any run; --zed_noise training on the "
              "merged set will DISABLE augmentation (no apex).")

    # Merged metadata: start from the first run's flags, but set num_points /
    # action_dim / obs_dim / num_samples from the ACTUAL merged arrays so the
    # output is correct even if a source dir had stale metadata.
    metadata = dict(runs[0]["metadata"])
    metadata["num_points"] = int(pointclouds.shape[1])
    metadata["action_dim"] = int(actions.shape[1])
    metadata["obs_dim"] = int(pointclouds.shape[1] * pointclouds.shape[2])
    metadata["num_samples"] = int(pointclouds.shape[0])
    metadata["merged_from"] = [os.path.abspath(r["dir"]) for r in runs]
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary.
    print("[merge-bc] Merged:")
    for r in runs:
        print(f"    {r['dir']}: {r['pointclouds'].shape[0]} samples")
    print(f"[merge-bc] -> {out}: {pointclouds.shape[0]} samples "
          f"(pointclouds {pointclouds.shape}, actions {actions.shape})")
    print(f"[merge-bc] Train with: --train_only {args.output} --output_dir {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
