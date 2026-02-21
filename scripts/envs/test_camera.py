#!/usr/bin/env python3
"""
Test the ZED X camera setup - verify RGBD and point cloud output.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/test_camera.py --num_envs 1

    # Save images
    ./isaaclab.sh -p crane_testbed/scripts/envs/test_camera.py --num_envs 1 --save_images
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
parser = argparse.ArgumentParser(description="Test camera setup")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--save_images", action="store_true", help="Save RGB and depth images")
parser.add_argument("--output_dir", type=str, default="camera_test_output", help="Output directory")
parser.add_argument("--num_steps", type=int, default=5, help="Number of grasp cycles to run")
# Add AppLauncher args (includes --headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch app with all args
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
        print(f"  Saved Depth: shape={depth_np.shape}, range=[{depth_np[valid_mask].min():.2f}, {depth_np[valid_mask].max():.2f}]m")


def save_pointcloud(points, output_dir, step, name="full"):
    """Save point cloud as PLY file and 2D top-down visualization."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    points_np = points.cpu().numpy()

    # Save PLY file
    ply_path = os.path.join(output_dir, f"pointcloud_{name}_step_{step:04d}.ply")
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

    # Save 2D top-down visualization (X vs Y, colored by Z height)
    png_path = os.path.join(output_dir, f"pointcloud_{name}_step_{step:04d}.png")

    # Subsample for faster rendering
    max_vis_points = 5000
    if len(points_np) > max_vis_points:
        idx = np.random.choice(len(points_np), max_vis_points, replace=False)
        vis_points = points_np[idx]
    else:
        vis_points = points_np

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot X vs Y (top-down view), color by Z (height)
    scatter = ax.scatter(vis_points[:, 0], vis_points[:, 1],
                        c=vis_points[:, 2], cmap='viridis', s=3, alpha=0.7)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"{name.upper()} Point Cloud (Top-Down) - Step {step}\n({len(points_np)} points)")
    ax.set_aspect('equal')

    # Add colorbar for height
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Height Z (m)')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {ply_path} + {png_path}")


def save_depth_observation(env, output_dir, step):
    """Save the actual depth observation that the policy sees."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    os.makedirs(output_dir, exist_ok=True)

    # Get raw camera data
    env._camera.update(dt=env.cfg.sim.dt)
    depth_raw = env._camera.data.output["depth"][0].squeeze(-1).cpu().numpy()  # (H, W)

    # Get semantic segmentation
    sem_seg = env._camera.data.output["semantic_segmentation"][0]
    if sem_seg.dim() == 3:
        sem_ids = sem_seg[..., 0].cpu().numpy()
    else:
        sem_ids = sem_seg.cpu().numpy()

    # Create log mask
    log_mask = (sem_ids > 0)

    # Masked depth (logs only)
    depth_min, depth_max = 1.5, 8.0
    depth_masked = np.where(log_mask, depth_raw, depth_max)
    depth_masked = np.clip(depth_masked, depth_min, depth_max)

    # Normalize to [0, 1]
    depth_norm = (depth_masked - depth_min) / (depth_max - depth_min)

    # Policy sees 256x256 (same as camera res, no downsampling)
    depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
    depth_256 = F.interpolate(depth_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    depth_256 = depth_256.squeeze().numpy()

    # Create visualization with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Raw depth (full resolution)
    ax1 = axes[0, 0]
    valid = ~np.isinf(depth_raw)
    depth_viz = np.where(valid, depth_raw, np.nan)
    im1 = ax1.imshow(depth_viz, cmap='viridis')
    ax1.set_title(f'Raw Depth ({depth_raw.shape[1]}x{depth_raw.shape[0]})')
    plt.colorbar(im1, ax=ax1, label='Depth (m)')

    # 2. Semantic mask (log pixels)
    ax2 = axes[0, 1]
    ax2.imshow(log_mask, cmap='gray')
    ax2.set_title(f'Log Mask (semantic > 0)\n{log_mask.sum()} log pixels')

    # 3. Masked depth (logs only, normalized)
    ax3 = axes[1, 0]
    im3 = ax3.imshow(depth_norm, cmap='viridis', vmin=0, vmax=1)
    ax3.set_title(f'Masked & Normalized Depth\n(logs only, range [{depth_min}, {depth_max}]m)')
    plt.colorbar(im3, ax=ax3, label='Normalized [0,1]')

    # 4. Final 256x256 observation (what policy sees!)
    ax4 = axes[1, 1]
    im4 = ax4.imshow(depth_256, cmap='viridis', vmin=0, vmax=1)
    ax4.set_title(f'POLICY OBSERVATION (256x256)\nThis is what the CNN sees!')
    plt.colorbar(im4, ax=ax4, label='Normalized [0,1]')

    plt.suptitle(f'Depth Observation Pipeline - Step {step}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"depth_obs_step_{step:04d}.png"), dpi=150)
    plt.close()

    print(f"  Saved depth observation visualization: depth_obs_step_{step:04d}.png")
    print(f"    Raw depth: {depth_raw.shape}, Log pixels: {log_mask.sum()}")
    print(f"    Policy obs: 256x256, range [{depth_256.min():.3f}, {depth_256.max():.3f}]")


def main():
    # Create environment with camera enabled
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.enable_camera = True  # Enable camera!
    cfg.use_hierarchical_rl = False  # Standard mode for testing

    # Use same camera settings as training!
    cfg.camera_cfg.width = 256
    cfg.camera_cfg.height = 256
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]

    print("\n" + "="*60)
    print("Camera Test Configuration:")
    print(f"  Resolution: {cfg.camera_cfg.width}x{cfg.camera_cfg.height}")
    print(f"  Data types: {cfg.camera_cfg.data_types}")
    print(f"  Position: {cfg.camera_cfg.offset.pos}")
    print(f"  Policy observation: 84x84 masked depth")
    print("="*60 + "\n")

    # Create env with render_mode for camera support
    env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    print(f"[Camera Test] Environment created with {env.num_envs} envs")

    # Reset to initialize everything
    env.reset()

    print("\n[Camera Test] Starting camera capture test...", flush=True)

    # Capture camera data before each grasp cycle
    for step in range(args_cli.num_steps):
        print(f"\n{'='*60}", flush=True)
        print(f"=== CYCLE {step}: Capturing camera BEFORE grasp ===", flush=True)
        print(f"{'='*60}", flush=True)

        # Get RGBD data BEFORE the grasp
        cam_data = env.get_camera_data()

        if cam_data:
            if "rgb" in cam_data:
                rgb = cam_data["rgb"]
                print(f"RGB: shape={rgb.shape}, dtype={rgb.dtype}")

            if "depth" in cam_data:
                depth = cam_data["depth"]
                valid = ~torch.isinf(depth)
                if valid.any():
                    print(f"Depth: shape={depth.shape}, range=[{depth[valid].min():.2f}, {depth[valid].max():.2f}]m")
                else:
                    print(f"Depth: shape={depth.shape}, all invalid (inf)")

            if "intrinsics" in cam_data:
                K = cam_data["intrinsics"]
                print(f"Intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

            # Debug: check semantic segmentation output
            if "semantic_segmentation" in cam_data:
                sem_seg = cam_data.get("semantic_segmentation")
                if sem_seg is not None:
                    print(f"Semantic seg: shape={sem_seg.shape}, dtype={sem_seg.dtype}")
                    unique_ids = torch.unique(sem_seg)
                    print(f"  Unique semantic IDs: {unique_ids.tolist()[:20]}{'...' if len(unique_ids) > 20 else ''}")

            # Get point clouds (limit depth range to pile area ~2-8m from camera)
            depth_range = (1.5, 8.0)  # meters
            pc_world = env.get_pointcloud_world(env_idx=0, max_points=10000, depth_range=depth_range)
            print(f"Point cloud (world frame, depth {depth_range}): {pc_world.shape[0]} points")

            # Get LOGS ONLY point cloud (filtered by semantic segmentation)
            pc_logs = env.get_log_pointcloud_world(env_idx=0, max_points=10000, depth_range=depth_range)
            print(f"Point cloud (LOGS ONLY): {pc_logs.shape[0]} points")

            if pc_logs.shape[0] > 0:
                print(f"  Logs X: [{pc_logs[:,0].min():.2f}, {pc_logs[:,0].max():.2f}]")
                print(f"  Logs Y: [{pc_logs[:,1].min():.2f}, {pc_logs[:,1].max():.2f}]")
                print(f"  Logs Z: [{pc_logs[:,2].min():.2f}, {pc_logs[:,2].max():.2f}]")

            # Save if requested
            if args_cli.save_images:
                # Save the depth observation pipeline visualization
                save_depth_observation(env, args_cli.output_dir, step)

                # Save raw images
                save_images(
                    cam_data.get("rgb"),
                    cam_data.get("depth"),
                    args_cli.output_dir,
                    step
                )
                # Save full world point cloud (PLY + visualization PNG)
                if pc_world.shape[0] > 0:
                    save_pointcloud(pc_world, args_cli.output_dir, step, name="full")
                # Save logs-only point cloud (PLY + visualization PNG)
                if pc_logs.shape[0] > 0:
                    save_pointcloud(pc_logs, args_cli.output_dir, step, name="logs")
                else:
                    print(f"  WARNING: No log points to save (pc_logs is empty)")
        else:
            print("Camera not enabled or no data available")

        # Now execute the grasp cycle
        print(f"\n--- Executing grasp cycle {step} ---", flush=True)
        env.step(torch.zeros((env.num_envs, 4), device=env.device))

        # Render scene after grasp/despawn so next camera capture shows updated pile
        # This ensures despawned logs are removed from the visual before next capture
        env.sim.render()
        if hasattr(env, '_camera') and env._camera is not None:
            env._camera.update(dt=env.cfg.sim.dt)

    print("\n[Camera Test] Complete!")
    if args_cli.save_images:
        print(f"Images saved to: {args_cli.output_dir}/")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
