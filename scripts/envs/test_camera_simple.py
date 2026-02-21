#!/usr/bin/env python3
"""
Simple camera test - just captures camera data without running full episodes.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/test_camera_simple.py --enable_cameras --save_images
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# IsaacLab imports
from isaaclab.app import AppLauncher

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Simple camera test")
parser.add_argument("--save_images", action="store_true", help="Save RGB and depth images")
parser.add_argument("--output_dir", type=str, default="camera_test_output", help="Output directory")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import environment after AppLauncher
import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


def save_images(rgb, depth, output_dir, step):
    """Save RGB and depth images using matplotlib."""
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Save RGB
    if rgb is not None:
        rgb_np = rgb[0].cpu().numpy()  # First env, (H, W, 4) RGBA
        if rgb_np.max() > 1.0:
            rgb_np = rgb_np / 255.0
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_np[:, :, :3])  # RGB only
        plt.title(f"RGB - Step {step}")
        plt.savefig(os.path.join(output_dir, f"rgb_step_{step:04d}.png"))
        plt.close()
        print(f"  Saved RGB: shape={rgb_np.shape}, range=[{rgb_np.min():.2f}, {rgb_np.max():.2f}]")

    # Save Depth
    if depth is not None:
        depth_np = depth[0].cpu().numpy().squeeze()  # First env, (H, W)
        # Replace inf with max valid value for visualization
        valid_mask = ~np.isinf(depth_np)
        if valid_mask.any():
            max_valid = depth_np[valid_mask].max()
            depth_viz = np.where(valid_mask, depth_np, max_valid)
        else:
            depth_viz = depth_np

        plt.figure(figsize=(10, 6))
        plt.imshow(depth_viz, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.title(f"Depth - Step {step}")
        plt.savefig(os.path.join(output_dir, f"depth_step_{step:04d}.png"))
        plt.close()
        if valid_mask.any():
            print(f"  Saved Depth: shape={depth_np.shape}, range=[{depth_np[valid_mask].min():.2f}, {depth_np[valid_mask].max():.2f}]m")


def save_pointcloud(points, output_dir, filename):
    """Save point cloud as PLY file."""
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    points_np = points.cpu().numpy()

    ply_path = os.path.join(output_dir, filename)

    # Write PLY file
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_np)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points_np:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    print(f"  Saved point cloud: {len(points_np)} points to {ply_path}")


def main():
    # Create environment with camera enabled
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = 1
    cfg.enable_camera = True
    cfg.use_hierarchical_rl = False  # Don't run FSM controller

    print("\n" + "="*60)
    print("Camera Test Configuration:")
    print(f"  Resolution: {cfg.camera_cfg.width}x{cfg.camera_cfg.height}")
    print(f"  Data types: {cfg.camera_cfg.data_types}")
    print(f"  Position: {cfg.camera_cfg.offset.pos}")
    print(f"  Rotation: {cfg.camera_cfg.offset.rot}")
    print(f"  Convention: {cfg.camera_cfg.offset.convention}")
    print("="*60 + "\n")

    # Create env with render_mode for camera support
    env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    print(f"[Camera Test] Environment created with {env.num_envs} envs")

    # Reset to initialize everything
    print("[Camera Test] Initializing environment...")
    env.reset()

    # Just update the scene a few times without calling step()
    # This renders the scene without triggering FSM
    print("[Camera Test] Rendering scene...")
    for i in range(5):
        env.scene.update(dt=cfg.sim.dt)

    print("\n[Camera Test] Capturing camera data...")

    # Get RGBD data
    cam_data = env.get_camera_data()

    if cam_data:
        print("\nCamera Data:")
        if "rgb" in cam_data:
            rgb = cam_data["rgb"]
            print(f"  RGB: shape={rgb.shape}, dtype={rgb.dtype}")

        if "depth" in cam_data:
            depth = cam_data["depth"]
            valid = ~torch.isinf(depth)
            if valid.any():
                print(f"  Depth: shape={depth.shape}, range=[{depth[valid].min():.2f}, {depth[valid].max():.2f}]m")
            else:
                print(f"  Depth: shape={depth.shape}, all invalid (inf)")

        if "intrinsics" in cam_data:
            K = cam_data["intrinsics"]
            print(f"  Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

        # Get point clouds
        print("\nPoint Clouds:")
        pc = env.get_pointcloud(env_idx=0, max_points=10000)
        print(f"  Camera frame: {pc.shape[0]} points")

        pc_world = env.get_pointcloud_world(env_idx=0, max_points=10000)
        print(f"  World frame: {pc_world.shape[0]} points")
        if pc_world.shape[0] > 0:
            print(f"    X range: [{pc_world[:,0].min():.2f}, {pc_world[:,0].max():.2f}]")
            print(f"    Y range: [{pc_world[:,1].min():.2f}, {pc_world[:,1].max():.2f}]")
            print(f"    Z range: [{pc_world[:,2].min():.2f}, {pc_world[:,2].max():.2f}]")

        pc_logs = env.get_log_pointcloud_world(env_idx=0, max_points=10000)
        print(f"  Logs only (world frame): {pc_logs.shape[0]} points")
        if pc_logs.shape[0] > 0:
            print(f"    X range: [{pc_logs[:,0].min():.2f}, {pc_logs[:,0].max():.2f}]")
            print(f"    Y range: [{pc_logs[:,1].min():.2f}, {pc_logs[:,1].max():.2f}]")
            print(f"    Z range: [{pc_logs[:,2].min():.2f}, {pc_logs[:,2].max():.2f}]")

        # Save if requested
        if args_cli.save_images:
            print(f"\nSaving outputs to: {args_cli.output_dir}/")
            save_images(
                cam_data.get("rgb"),
                cam_data.get("depth"),
                args_cli.output_dir,
                0
            )
            if pc_world.shape[0] > 0:
                save_pointcloud(pc_world, args_cli.output_dir, "pointcloud_full.ply")
            if pc_logs.shape[0] > 0:
                save_pointcloud(pc_logs, args_cli.output_dir, "pointcloud_logs_only.ply")
            print("Done!")
    else:
        print("ERROR: Camera not enabled or no data available")

    print("\n[Camera Test] Complete!")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
