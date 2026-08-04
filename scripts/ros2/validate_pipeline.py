#!/usr/bin/env python3
"""Validate standalone ROS2 pipeline against sim ground truth.

Loads a snapshot saved by save_sim_snapshot.py and runs the standalone
pointcloud_pipeline.py on the same depth image. Compares:
  1. Point cloud in base frame (standalone vs sim)
  2. Policy output (if checkpoint provided)

Usage:
    # First, save snapshot from sim:
    ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/save_sim_snapshot.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 --num_envs 1 --headless --enable_cameras

    # Then validate standalone pipeline:
    python3 validate_pipeline.py --snapshot_dir /tmp/sim_snapshot
    python3 validate_pipeline.py --snapshot_dir /tmp/sim_snapshot --checkpoint /path/to/model.pt
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

# Add ros2 scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pointcloud_pipeline import (
    depth_to_pointcloud, camera_to_world, world_to_base_frame,
    farthest_point_sampling, process_depth,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_dir", default="/tmp/sim_snapshot")
    parser.add_argument("--checkpoint", default=None, help="RSL-RL .pt checkpoint for policy comparison")
    parser.add_argument("--policy_type", default="bcrl", choices=["bc", "bcrl", "rl", "sac", "heuristic"])
    args = parser.parse_args()

    d = args.snapshot_dir
    print(f"Loading snapshot from {d}/\n")

    # Load ground truth
    depth = np.load(os.path.join(d, "depth.npy"))
    intrinsics = np.load(os.path.join(d, "intrinsics.npy"))
    cam_pos = np.load(os.path.join(d, "cam_pos.npy"))
    cam_rot = np.load(os.path.join(d, "cam_rot.npy"))
    base_pos = np.load(os.path.join(d, "base_pos.npy"))
    base_rot = np.load(os.path.join(d, "base_rot.npy"))
    pc_world_gt = np.load(os.path.join(d, "pc_world_gt.npy"))
    pc_base_gt = np.load(os.path.join(d, "pc_base_gt.npy"))
    obs_gt = np.load(os.path.join(d, "obs_gt.npy"))
    bounds_min = np.load(os.path.join(d, "bounds_min.npy"))
    bounds_max = np.load(os.path.join(d, "bounds_max.npy"))

    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)

    num_points = meta["num_points"]
    depth_range = tuple(meta["depth_range"])

    print(f"Depth: {depth.shape}, range=[{depth[np.isfinite(depth)].min():.2f}, {depth[np.isfinite(depth)].max():.2f}]m")
    print(f"Sim PC world: {pc_world_gt.shape[0]} points")
    print(f"Sim PC base:  {pc_base_gt.shape[0]} points")
    print(f"Sim obs:      {obs_gt.shape}")
    print()

    # ── Test 1: Depth → world-frame point cloud ──────────────────────
    print("=" * 60)
    print("TEST 1: Depth → world-frame point cloud")
    print("=" * 60)

    pc_cam = depth_to_pointcloud(depth, intrinsics, depth_range)
    pc_world_ours = camera_to_world(pc_cam, cam_pos, cam_rot)

    print(f"  Standalone: {pc_world_ours.shape[0]} points")
    print(f"  Sim GT:     {pc_world_gt.shape[0]} points")

    # Compare bounding boxes
    if pc_world_ours.shape[0] > 0 and pc_world_gt.shape[0] > 0:
        ours_min = pc_world_ours.min(axis=0)
        ours_max = pc_world_ours.max(axis=0)
        gt_min = pc_world_gt.min(axis=0)
        gt_max = pc_world_gt.max(axis=0)
        print(f"  Standalone bbox: [{ours_min[0]:.2f},{ours_min[1]:.2f},{ours_min[2]:.2f}] → [{ours_max[0]:.2f},{ours_max[1]:.2f},{ours_max[2]:.2f}]")
        print(f"  Sim GT bbox:     [{gt_min[0]:.2f},{gt_min[1]:.2f},{gt_min[2]:.2f}] → [{gt_max[0]:.2f},{gt_max[1]:.2f},{gt_max[2]:.2f}]")
        bbox_diff = max(np.abs(ours_min - gt_min).max(), np.abs(ours_max - gt_max).max())
        print(f"  Bbox max diff: {bbox_diff:.4f}m")
        if bbox_diff < 0.1:
            print("  ✓ PASS: Bounding boxes match within 10cm")
        else:
            print("  ✗ FAIL: Bounding box mismatch > 10cm")
    print()

    # ── Test 2: World → base frame transform ─────────────────────────
    print("=" * 60)
    print("TEST 2: World → base-frame transform")
    print("=" * 60)

    pc_base_ours = world_to_base_frame(pc_world_ours, base_pos, base_rot)

    if pc_base_ours.shape[0] > 0 and pc_base_gt.shape[0] > 0:
        ours_min = pc_base_ours.min(axis=0)
        ours_max = pc_base_ours.max(axis=0)
        gt_min = pc_base_gt.min(axis=0)
        gt_max = pc_base_gt.max(axis=0)
        print(f"  Standalone bbox: [{ours_min[0]:.2f},{ours_min[1]:.2f},{ours_min[2]:.2f}] → [{ours_max[0]:.2f},{ours_max[1]:.2f},{ours_max[2]:.2f}]")
        print(f"  Sim GT bbox:     [{gt_min[0]:.2f},{gt_min[1]:.2f},{gt_min[2]:.2f}] → [{gt_max[0]:.2f},{gt_max[1]:.2f},{gt_max[2]:.2f}]")
        bbox_diff = max(np.abs(ours_min - gt_min).max(), np.abs(ours_max - gt_max).max())
        print(f"  Bbox max diff: {bbox_diff:.4f}m")
        if bbox_diff < 0.1:
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
    print()

    # ── Test 3: Full pipeline (depth → obs tensor) ───────────────────
    print("=" * 60)
    print("TEST 3: Full pipeline (depth → flattened obs)")
    print("=" * 60)

    obs_ours = process_depth(depth, intrinsics, cam_pos, cam_rot,
                             base_pos, base_rot, num_points, depth_range)
    obs_ours_np = obs_ours.numpy()

    print(f"  Standalone obs: shape={obs_ours_np.shape}, norm={np.linalg.norm(obs_ours_np):.2f}")
    print(f"  Sim GT obs:     shape={obs_gt.shape}, norm={np.linalg.norm(obs_gt):.2f}")

    # Compare reshaped point clouds (FPS is random, so exact match unlikely)
    pc_ours = obs_ours_np.reshape(-1, 3)
    pc_gt = obs_gt.reshape(-1, 3)

    # Compare centroid and spread
    valid_ours = np.any(pc_ours != 0, axis=1)
    valid_gt = np.any(pc_gt != 0, axis=1)
    print(f"  Valid points: standalone={valid_ours.sum()}, sim={valid_gt.sum()}")

    if valid_ours.any() and valid_gt.any():
        centroid_ours = pc_ours[valid_ours].mean(axis=0)
        centroid_gt = pc_gt[valid_gt].mean(axis=0)
        centroid_diff = np.linalg.norm(centroid_ours - centroid_gt)
        print(f"  Centroid standalone: ({centroid_ours[0]:.3f}, {centroid_ours[1]:.3f}, {centroid_ours[2]:.3f})")
        print(f"  Centroid sim GT:     ({centroid_gt[0]:.3f}, {centroid_gt[1]:.3f}, {centroid_gt[2]:.3f})")
        print(f"  Centroid diff: {centroid_diff:.4f}m")

        spread_ours = pc_ours[valid_ours].std(axis=0)
        spread_gt = pc_gt[valid_gt].std(axis=0)
        print(f"  Spread standalone: ({spread_ours[0]:.3f}, {spread_ours[1]:.3f}, {spread_ours[2]:.3f})")
        print(f"  Spread sim GT:     ({spread_gt[0]:.3f}, {spread_gt[1]:.3f}, {spread_gt[2]:.3f})")

        if centroid_diff < 0.2:
            print("  ✓ PASS: Centroids match within 20cm (FPS randomness expected)")
        else:
            print("  ✗ FAIL: Centroid mismatch > 20cm — check transforms")
    print()

    # ── Test 4: Policy comparison (optional) ─────────────────────────
    if args.checkpoint:
        print("=" * 60)
        print("TEST 4: Policy output comparison")
        print("=" * 60)

        from policy_loader import load_policy, decode_action

        # Run policy on standalone obs
        policy = load_policy(args.policy_type, args.checkpoint,
                             bounds_min, bounds_max, cossin=True, device="cpu")
        x_ours, y_ours, z_ours, yaw_ours = policy.get_target(obs_ours)
        print(f"  Standalone target: ({x_ours:.3f}, {y_ours:.3f}, {z_ours:.3f}), yaw={math.degrees(yaw_ours):.1f}deg")

        # Run policy on sim GT obs
        obs_gt_t = torch.from_numpy(obs_gt).float()
        x_gt, y_gt, z_gt, yaw_gt = policy.get_target(obs_gt_t)
        print(f"  Sim GT target:     ({x_gt:.3f}, {y_gt:.3f}, {z_gt:.3f}), yaw={math.degrees(yaw_gt):.1f}deg")

        pos_diff = math.sqrt((x_ours - x_gt)**2 + (y_ours - y_gt)**2 + (z_ours - z_gt)**2)
        yaw_diff = abs(yaw_ours - yaw_gt)
        print(f"  Position diff: {pos_diff:.4f}m")
        print(f"  Yaw diff: {math.degrees(yaw_diff):.2f}deg")

        if pos_diff < 0.3:
            print("  ✓ PASS: Policy targets within 30cm (FPS randomness expected)")
        else:
            print("  ✗ FAIL: Policy targets diverge > 30cm")

        # Also test with identical obs (should be exact match)
        x2, y2, z2, yaw2 = policy.get_target(obs_gt_t)
        print(f"\n  Same-input check: ({x2:.3f}, {y2:.3f}, {z2:.3f}) vs ({x_gt:.3f}, {y_gt:.3f}, {z_gt:.3f})")
        assert x2 == x_gt and y2 == y_gt, "  ✗ Policy is not deterministic!"
        print("  ✓ Policy is deterministic on same input")
    else:
        print("(Skipping policy test — no --checkpoint provided)")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
