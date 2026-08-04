"""Save a ground-truth snapshot from a running sim for pipeline validation.

Run inside Isaac Lab (needs sim + camera). Saves:
  - depth image (H, W) float32 .npy
  - camera intrinsics (3, 3)
  - camera pose (pos + quat)
  - crane base pose (pos + quat)
  - sim point cloud in base frame (ground truth)
  - sim observation tensor (ground truth)
  - policy action + decoded target (ground truth)

Usage:
    ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/save_sim_snapshot.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
        --num_envs 1 --headless --enable_cameras \
        --checkpoint /workspace/crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Save sim snapshot for ROS2 pipeline validation.")
parser.add_argument("--task", type=str, default="Isaac-Crane-PointCloud-CosSin-Raw-MR-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL .pt checkpoint for policy comparison")
parser.add_argument("--output_dir", type=str, default="/tmp/sim_snapshot")
parser.add_argument("--num_warmup_steps", type=int, default=5, help="Steps to run before saving (lets pile settle)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import json

import isaaclab_tasks  # noqa: F401
import crane_testbed.tasks  # noqa: F401


def quat_to_rot_matrix(quat_wxyz):
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)


def main():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Create PointCloud env directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
    from crane_pointcloud_direct_env import CranePointCloudDirectEnv
    env = CranePointCloudDirectEnv(env_cfg)

    base_env = env._base_env
    env_idx = 0

    # Run a few steps to let the pile settle and camera update
    print(f"Running {args_cli.num_warmup_steps} warmup steps...")
    obs, _ = env.reset()
    for _ in range(args_cli.num_warmup_steps):
        # Random action to trigger FSM cycle
        action = torch.zeros(args_cli.num_envs, env.num_actions, device=env.device)
        obs, _, _, _, _ = env.step(action)

    # Update camera one more time
    base_env._camera.update(dt=base_env.cfg.sim.dt)

    # ── Extract all data ──────────────────────────────────────────────

    # 1. Depth image
    depth = base_env._camera.data.output["depth"][env_idx].squeeze(-1).cpu().numpy()

    # 2. Camera intrinsics
    intrinsics = base_env._camera.data.intrinsic_matrices[env_idx].cpu().numpy()

    # 3. Camera pose (world frame)
    cam_pos = base_env._camera.data.pos_w[env_idx].cpu().numpy()
    cam_quat_ros = base_env._camera.data.quat_w_ros[env_idx].cpu().numpy()  # wxyz, ROS convention

    # 4. Crane base pose (world frame)
    base_pos = base_env.crane.data.root_pos_w[env_idx].cpu().numpy()
    base_quat = base_env.crane.data.root_quat_w[env_idx].cpu().numpy()  # wxyz

    # 5. Ground truth: sim point cloud in world frame
    use_raw = getattr(env_cfg, 'use_raw_pointcloud', True)
    if use_raw:
        pc_world_gt = base_env.get_pointcloud_world(
            env_idx, max_points=5000,
            depth_range=(getattr(env_cfg, 'depth_range_min', 1.0),
                        getattr(env_cfg, 'depth_range_max', 10.0))
        ).cpu().numpy()
    else:
        pc_world_gt = base_env.get_log_pointcloud_world(
            env_idx, max_points=5000,
            depth_range=(getattr(env_cfg, 'depth_range_min', 1.0),
                        getattr(env_cfg, 'depth_range_max', 10.0))
        ).cpu().numpy()

    # 6. Ground truth: sim point cloud in base frame
    pc_world_t = torch.from_numpy(pc_world_gt).to(env.device)
    pc_base_gt = env._world_to_base_frame(pc_world_t, env_idx).cpu().numpy()

    # 7. Ground truth: sim observation (FPS + flattened)
    obs_gt = obs[env_idx].cpu().numpy()  # (3072,)

    # 8. Action bounds
    bounds_min = base_env._action_bounds_min[env_idx].cpu().numpy()
    bounds_max = base_env._action_bounds_max[env_idx].cpu().numpy()

    # ── Save everything ───────────────────────────────────────────────
    os.makedirs(args_cli.output_dir, exist_ok=True)
    out = args_cli.output_dir

    np.save(os.path.join(out, "depth.npy"), depth)
    np.save(os.path.join(out, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(out, "cam_pos.npy"), cam_pos)
    np.save(os.path.join(out, "cam_quat_ros.npy"), cam_quat_ros)
    np.save(os.path.join(out, "base_pos.npy"), base_pos)
    np.save(os.path.join(out, "base_quat.npy"), base_quat)
    np.save(os.path.join(out, "pc_world_gt.npy"), pc_world_gt)
    np.save(os.path.join(out, "pc_base_gt.npy"), pc_base_gt)
    np.save(os.path.join(out, "obs_gt.npy"), obs_gt)
    np.save(os.path.join(out, "bounds_min.npy"), bounds_min)
    np.save(os.path.join(out, "bounds_max.npy"), bounds_max)

    # Rotation matrices for the standalone pipeline
    cam_rot = quat_to_rot_matrix(cam_quat_ros)
    base_rot = quat_to_rot_matrix(base_quat)
    np.save(os.path.join(out, "cam_rot.npy"), cam_rot)
    np.save(os.path.join(out, "base_rot.npy"), base_rot)

    metadata = {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "depth_shape": list(depth.shape),
        "num_pc_world_points": int(pc_world_gt.shape[0]),
        "num_pc_base_points": int(pc_base_gt.shape[0]),
        "obs_dim": int(obs_gt.shape[0]),
        "use_raw_pointcloud": use_raw,
        "depth_range": [getattr(env_cfg, 'depth_range_min', 1.0),
                       getattr(env_cfg, 'depth_range_max', 10.0)],
        "num_points": getattr(env_cfg, 'num_points', 1024),
        "cam_pos": cam_pos.tolist(),
        "base_pos": base_pos.tolist(),
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSnapshot saved to {out}/")
    print(f"  depth:       {depth.shape}, range=[{depth[np.isfinite(depth)].min():.2f}, {depth[np.isfinite(depth)].max():.2f}]m")
    print(f"  intrinsics:  fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}")
    print(f"  cam_pos:     {cam_pos}")
    print(f"  base_pos:    {base_pos}")
    print(f"  pc_world:    {pc_world_gt.shape[0]} points")
    print(f"  pc_base:     {pc_base_gt.shape[0]} points")
    print(f"  obs:         {obs_gt.shape}")
    print(f"  bounds:      min={bounds_min[:3]}, max={bounds_max[:3]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
