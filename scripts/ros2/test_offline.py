#!/usr/bin/env python3
"""Offline test: run policy + FSM on a saved or synthetic depth image.

No ROS2 or Isaac Lab required. Tests the full pipeline:
  depth -> point cloud -> policy -> FSM -> EE commands

Usage:
    python3 test_offline.py --policy_type heuristic
    python3 test_offline.py --policy_type bcrl --checkpoint /path/to/model.pt
    python3 test_offline.py --depth_image /path/to/depth.npy  # saved from sim
"""

import argparse
import math
import numpy as np
import torch

from pointcloud_pipeline import process_depth
from fsm import CraneFSM, FSMConfig
from policy_loader import load_policy


def make_synthetic_depth(width=640, height=360):
    """Create a fake depth image with a pile of cylinders (logs).

    Returns:
        depth: (H, W) float32 depth in meters.
        intrinsics: (3, 3) camera intrinsic matrix.
    """
    # ZED X-like intrinsics
    fx = fy = 350.0
    cx, cy = width / 2, height / 2
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Start with far background
    depth = np.full((height, width), 8.0, dtype=np.float32)

    # Place some "logs" as horizontal cylinders at various depths
    rng = np.random.RandomState(42)
    for _ in range(20):
        # Random log center in image
        u_center = rng.randint(width // 4, 3 * width // 4)
        v_center = rng.randint(height // 4, 3 * height // 4)
        log_depth = rng.uniform(3.0, 6.0)
        radius_px = int(15 * (5.0 / log_depth))  # apparent radius
        length_px = int(60 * (5.0 / log_depth))

        # Draw horizontal cylinder (simplified as rectangle with depth variation)
        for du in range(-length_px, length_px):
            for dv in range(-radius_px, radius_px):
                u, v = u_center + du, v_center + dv
                if 0 <= u < width and 0 <= v < height:
                    # Cylinder depth profile
                    r_frac = abs(dv) / max(radius_px, 1)
                    if r_frac < 1.0:
                        surface_depth = log_depth - 0.1 * math.sqrt(1 - r_frac**2)
                        depth[v, u] = min(depth[v, u], surface_depth)

    return depth, intrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", default="heuristic",
                        choices=["heuristic", "bc", "bcrl", "rl", "sac"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--depth_image", default=None, help="Path to saved depth .npy file")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_ticks", type=int, default=500, help="FSM ticks to simulate")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Workspace bounds (must match training config)
    bounds_min = np.array([-2.0, -3.0, -0.5], dtype=np.float32)
    bounds_max = np.array([2.0, 3.0, 1.5], dtype=np.float32)

    # Camera extrinsics (world frame) — adjust to match your setup
    cam_pos = np.array([5.0, -1.0, 3.0], dtype=np.float32)
    cam_rot = np.eye(3, dtype=np.float32)  # simplified: identity rotation

    # Crane base (world origin for simplicity)
    base_pos = np.zeros(3, dtype=np.float32)
    base_rot = np.eye(3, dtype=np.float32)

    # Load or generate depth
    if args.depth_image:
        depth = np.load(args.depth_image)
        # Use ZED X-like intrinsics
        h, w = depth.shape
        fx = fy = 350.0
        intrinsics = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float32)
        print(f"Loaded depth from {args.depth_image}: shape={depth.shape}")
    else:
        depth, intrinsics = make_synthetic_depth()
        print(f"Using synthetic depth: shape={depth.shape}")

    # Point cloud pipeline
    obs = process_depth(depth, intrinsics, cam_pos, cam_rot,
                        base_pos, base_rot, args.num_points)
    pc = obs.numpy().reshape(-1, 3)
    valid = np.any(pc != 0, axis=1)
    print(f"Point cloud: {valid.sum()}/{args.num_points} valid points")
    if valid.any():
        print(f"  XYZ range: x=[{pc[valid,0].min():.2f}, {pc[valid,0].max():.2f}], "
              f"y=[{pc[valid,1].min():.2f}, {pc[valid,1].max():.2f}], "
              f"z=[{pc[valid,2].min():.2f}, {pc[valid,2].max():.2f}]")

    # Load policy
    policy = load_policy(args.policy_type, args.checkpoint, bounds_min, bounds_max,
                         cossin=True, device=args.device)

    # Get target from policy
    x, y, z, yaw = policy.get_target(obs)
    print(f"\nPolicy target: ({x:.3f}, {y:.3f}, {z:.3f}), yaw={math.degrees(yaw):.1f}deg")

    # Run FSM
    fsm = CraneFSM()
    fsm.set_target(x, y, z, yaw)

    # Simulate FSM ticks with a fake EE that slowly moves toward target
    ee_pos = [0.0, 0.0, 2.0]  # start position
    ee_yaw = 0.0
    gripper = 0.05  # open

    print(f"\nRunning FSM for {args.num_ticks} ticks...")
    print(f"{'Tick':>5}  {'Phase':<18}  {'EE cmd X':>8} {'Y':>8} {'Z':>8}  {'Yaw':>6}  {'Grip':>5}")
    print("-" * 75)

    last_phase = None
    for tick in range(args.num_ticks):
        cmd = fsm.tick(ee_pos, ee_yaw, gripper)

        # Print on phase change or every 50 ticks
        if cmd.phase != last_phase or tick % 50 == 0:
            print(f"{tick:5d}  {cmd.phase.name:<18}  "
                  f"{cmd.position[0]:8.3f} {cmd.position[1]:8.3f} {cmd.position[2]:8.3f}  "
                  f"{math.degrees(cmd.yaw):6.1f}  {cmd.gripper:5.2f}")
            last_phase = cmd.phase

        # Simulate EE moving toward command (simple P-controller)
        alpha = 0.05  # convergence rate
        ee_pos[0] += alpha * (cmd.position[0] - ee_pos[0])
        ee_pos[1] += alpha * (cmd.position[1] - ee_pos[1])
        ee_pos[2] += alpha * (cmd.position[2] - ee_pos[2])
        ee_yaw += alpha * (cmd.yaw - ee_yaw)
        gripper += 0.3 * (cmd.gripper - gripper)

        if cmd.cycle_complete:
            print(f"\n  >>> Cycle {fsm.cycle_count} complete! <<<\n")
            break

    print(f"\nFinal state: phase={cmd.phase.name}, cycles={fsm.cycle_count}")


if __name__ == "__main__":
    main()
