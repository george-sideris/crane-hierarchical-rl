#!/usr/bin/env python3
"""
Play a trained BC PointCloud policy in the crane environment.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint logs/bc_pointcloud/bc_XXXX/bc_pointcloud_policy.pt \
        --num_envs 1 --num_episodes 10
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import numpy as np

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Play BC pointcloud policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics")
parser.add_argument("--visualize", action="store_true", help="Save per-step visualization PNGs")
parser.add_argument("--viz_dir", type=str, default=None, help="Directory for viz PNGs (default: checkpoint dir / viz)")
parser.add_argument("--paper_viz", action="store_true", help="Save paper-quality pipeline and progression figures")
parser.add_argument("--raw_pcd", action="store_true", help="Use raw (unmasked) point cloud instead of segmented log-only points")
parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic evaluation")
parser.add_argument("--obs_noise", type=float, default=0.0, help="Gaussian noise σ added to PCD coordinates (meters)")
parser.add_argument("--action_noise", type=float, default=0.0, help="Gaussian noise σ added to policy action output")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

# Import the environment
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class PointNetEncoder(nn.Module):
    """PointNet encoder - must match training architecture exactly."""

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        x = x.view(batch_size * num_points, -1)
        x = self.mlp1(x)
        x = x.view(batch_size, num_points, -1)
        x = x.max(dim=1)[0]
        x = self.fc(x)
        return x


class BCPointNetPolicy(nn.Module):
    """PointNet policy matching the BC training architecture."""

    def __init__(self, num_points: int = 1024, action_dim: int = 4, latent_dim: int = 256):
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim

        self.encoder = PointNetEncoder(input_dim=3, output_dim=latent_dim)

        self.actor_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        if points.dim() == 2:
            points = points.view(batch_size, self.num_points, 3)
        latent = self.encoder(points)
        return self.actor_mlp(latent)


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """FPS for point cloud downsampling."""
    device = points.device
    N = points.shape[0]

    if N <= num_samples:
        if N == 0:
            return torch.zeros((num_samples, 3), device=device)
        padding = torch.zeros((num_samples - N, 3), device=device)
        return torch.cat([points, padding], dim=0)

    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    current_idx = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_indices[i] = current_idx
        current_point = points[current_idx:current_idx+1]
        dist_to_current = torch.norm(points - current_point, dim=1)
        distances = torch.minimum(distances, dist_to_current)
        current_idx = torch.argmax(distances).item()

    return points[sampled_indices]


def get_log_pointcloud_base_frame(env, env_idx: int, num_points: int,
                                   depth_range: tuple = (1.0, 10.0),
                                   raw_pcd: bool = False) -> torch.Tensor:
    """Get point cloud in crane base frame (masked or raw)."""
    if raw_pcd:
        pc_world = env.get_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
    else:
        pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)

    if pc_world.shape[0] == 0:
        return torch.zeros((num_points, 3), device=env.device)

    # Transform to base frame
    base_pos_w = env.crane.data.root_pos_w[env_idx]
    base_quat_w = env.crane.data.root_quat_w[env_idx]

    pc_translated = pc_world - base_pos_w

    w, x, y, z = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])

    pc_base = pc_translated @ R
    pc_sampled = farthest_point_sampling(pc_base, num_points)

    return pc_sampled


def load_bc_policy(checkpoint_path: str, device: str) -> tuple:
    """Load BC policy from checkpoint."""
    print(f"[Play] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_points = checkpoint.get('num_points', 1024)
    action_dim = checkpoint.get('action_dim', 4)
    metadata = checkpoint.get('metadata', {})

    print(f"[Play] Policy: {num_points} points -> {action_dim} actions")

    policy = BCPointNetPolicy(num_points=num_points, action_dim=action_dim).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    return policy, num_points, metadata


def decode_action(raw_action, min_bounds, max_bounds):
    """Decode raw network output to (x, y, z, yaw) in base frame.

    Mirrors the env's _apply_action decode logic exactly:
      xyz: min + (tanh(a) + 1) / 2 * (max - min)
      yaw: 5D -> atan2(tanh(a4), tanh(a3)) / 2
           4D -> tanh(a3) * pi/2
    """
    a = raw_action.cpu().numpy()
    x = float(min_bounds[0] + (np.tanh(a[0]) + 1) / 2 * (max_bounds[0] - min_bounds[0]))
    y = float(min_bounds[1] + (np.tanh(a[1]) + 1) / 2 * (max_bounds[1] - min_bounds[1]))
    z = float(min_bounds[2] + (np.tanh(a[2]) + 1) / 2 * (max_bounds[2] - min_bounds[2]))
    if len(a) == 5:
        yaw = float(np.arctan2(np.tanh(a[4]), np.tanh(a[3])) / 2.0)
    else:
        yaw = float(np.tanh(a[3]) * (np.pi / 2))
    return x, y, z, yaw


def save_step_viz(points_np, x, y, z, yaw, step_idx, viz_dir,
                  logs_grasped=None, alignment=None,
                  bounds_min=None, bounds_max=None):
    """Save a 2-panel visualization PNG for one grasp step.

    Left:  top-down view (X vs Y) — horizontal placement + yaw arrow
    Right: side view (Y vs Z) — height targeting
    Both overlay: point cloud scatter + red star at predicted target.
    Optionally draws dashed action-bounds rectangle on each panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out zero-padding points for cleaner viz
    mask = np.any(points_np != 0.0, axis=1)
    pts = points_np[mask] if mask.any() else points_np

    # Compute point cloud bounding box for reference
    pc_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
    pc_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)

    # -- Left panel: top-down (X vs Y) colored by height --
    if len(pts) > 0:
        sc = ax_top.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis',
                            s=1, alpha=0.5, label='Point cloud')
        plt.colorbar(sc, ax=ax_top, label='Z (height)')
    ax_top.plot(x, y, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                label=f'Target ({x:.2f}, {y:.2f})')
    # Yaw direction arrow
    arrow_len = 0.3
    ax_top.annotate('', xy=(x + arrow_len * np.cos(yaw), y + arrow_len * np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    # Draw point cloud bounding box (X-Y)
    if len(pts) > 0:
        ax_top.add_patch(Rectangle(
            (pc_min[0], pc_min[1]), pc_max[0] - pc_min[0], pc_max[1] - pc_min[1],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
    # Draw action bounds rectangle (X-Y)
    if bounds_min is not None and bounds_max is not None:
        bw = bounds_max[0] - bounds_min[0]
        bh = bounds_max[1] - bounds_min[1]
        ax_top.add_patch(Rectangle(
            (bounds_min[0], bounds_min[1]), bw, bh,
            linewidth=1.5, edgecolor='orange', facecolor='none',
            linestyle='--', label='Action bounds'))
    ax_top.set_xlabel('X (base frame)')
    ax_top.set_ylabel('Y (base frame)')
    ax_top.set_title('Top-down (X vs Y)')
    ax_top.set_aspect('equal')
    ax_top.legend(loc='upper right', fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # -- Right panel: side view (Y vs Z) --
    if len(pts) > 0:
        ax_side.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap='viridis',
                        s=1, alpha=0.5, label='Point cloud')
    ax_side.plot(y, z, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                 label=f'Target (z={z:.2f})')
    # Draw point cloud bounding box (Y-Z)
    if len(pts) > 0:
        ax_side.add_patch(Rectangle(
            (pc_min[1], pc_min[2]), pc_max[1] - pc_min[1], pc_max[2] - pc_min[2],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
    # Draw action bounds rectangle (Y-Z)
    if bounds_min is not None and bounds_max is not None:
        bw = bounds_max[1] - bounds_min[1]
        bh = bounds_max[2] - bounds_min[2]
        ax_side.add_patch(Rectangle(
            (bounds_min[1], bounds_min[2]), bw, bh,
            linewidth=1.5, edgecolor='orange', facecolor='none',
            linestyle='--', label='Action bounds'))
    ax_side.set_xlabel('Y (base frame)')
    ax_side.set_ylabel('Z (base frame)')
    ax_side.set_title('Side view (Y vs Z)')
    ax_side.set_aspect('equal')
    ax_side.legend(loc='upper right', fontsize=8)
    ax_side.grid(True, alpha=0.3)

    # Suptitle with step info
    title = f'Step {step_idx} | Target: ({x:.3f}, {y:.3f}, {z:.3f}) yaw={np.degrees(yaw):.1f}°'
    if logs_grasped is not None:
        result = 'SUCCESS' if logs_grasped > 0 else 'MISS'
        title += f' | {result} ({logs_grasped} logs)'
    if alignment is not None and logs_grasped and logs_grasped > 0:
        title += f' | align={alignment:.3f}'
    fig.suptitle(title, fontsize=11)

    fig.savefig(os.path.join(viz_dir, f"step_{step_idx:04d}.png"),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


def draw_grapple_footprint(ax, x, y, yaw, width=1.5, length=0.5,
                           success=True, alpha=0.3):
    """Draw a rotated rectangle representing the grapple footprint.

    Args:
        ax: Matplotlib axes to draw on.
        x, y: Target position (center of grapple).
        yaw: Grapple rotation angle in radians.
        width: Grapple opening width (meters).
        length: Grapple depth (meters).
        success: If True, green fill; if False, red fill.
        alpha: Fill transparency.
    """
    color = '#2ecc71' if success else '#e74c3c'
    edge_color = '#27ae60' if success else '#c0392b'

    # Rectangle corners centered at origin.
    # The grapple opening (width = long side) spans ALONG the yaw direction,
    # and the grapple depth (length = short side) is perpendicular.
    # Before rotation: long side along X (yaw dir), short side along Y.
    hw = width / 2   # long half  (along yaw = opening)
    hl = length / 2  # short half (perpendicular to yaw)
    corners = np.array([
        [-hw, -hl],
        [ hw, -hl],
        [ hw,  hl],
        [-hw,  hl],
        [-hw, -hl],  # close
    ])

    # Rotate by yaw so the arrow direction stays along the short side
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ R.T

    # Translate to (x, y)
    rotated[:, 0] += x
    rotated[:, 1] += y

    # Draw filled polygon
    from matplotlib.patches import Polygon
    poly = Polygon(rotated[:-1], closed=True,
                   facecolor=color, edgecolor=edge_color,
                   alpha=alpha, linewidth=1.5, zorder=5)
    ax.add_patch(poly)

    # Draw yaw direction arrow from center (perpendicular to opening)
    arrow_len = length * 0.8
    dx = arrow_len * np.cos(yaw)
    dy = arrow_len * np.sin(yaw)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=edge_color, lw=2),
                zorder=6)


def get_pipeline_data(env, env_idx, num_points, depth_range=(1.0, 10.0)):
    """Collect all vision pipeline stage data for one environment.

    Returns:
        dict with keys:
            - rgb: (H, W, 3) uint8 numpy array (or None if not available)
            - depth: (H, W) float32 numpy array
            - semantic_ids: (H, W) int numpy array
            - log_mask: (H, W) bool numpy array
            - masked_depth: (H, W) float32 numpy (non-log pixels = NaN)
            - world_points: (N, 3) numpy array — log points in world frame
            - base_points: (M, 3) numpy array — log points in base frame (pre-FPS)
            - fps_points: (num_points, 3) numpy array — FPS-sampled in base frame
    """
    data = {}

    # --- (a) RGB ---
    if "rgb" in env.cfg.camera_cfg.data_types:
        rgb_tensor = env._camera.data.output["rgb"][env_idx]  # (H, W, 4) RGBA
        data["rgb"] = rgb_tensor[:, :, :3].cpu().numpy().astype(np.uint8)
    else:
        data["rgb"] = None

    # --- (b) Raw depth ---
    depth_tensor = env._camera.data.output["depth"][env_idx].squeeze(-1)  # (H, W)
    depth_np = depth_tensor.cpu().numpy().copy()
    data["depth"] = depth_np

    # --- (c) Semantic mask + masked depth ---
    sem_seg = env._camera.data.output["semantic_segmentation"][env_idx]
    if sem_seg.dim() == 3:
        semantic_ids = sem_seg[..., 0]
    else:
        semantic_ids = sem_seg
    semantic_ids_np = semantic_ids.cpu().numpy().astype(np.int32)
    data["semantic_ids"] = semantic_ids_np

    min_d, max_d = depth_range
    log_mask = (semantic_ids_np > 0) & ~np.isinf(depth_np) & (depth_np >= min_d) & (depth_np <= max_d)
    data["log_mask"] = log_mask

    masked_depth = depth_np.copy()
    masked_depth[~log_mask] = np.nan
    data["masked_depth"] = masked_depth

    # --- (d) 3D points in world frame (all log points, before FPS) ---
    world_pts = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
    data["world_points"] = world_pts.cpu().numpy()

    # --- Transform to base frame ---
    if world_pts.shape[0] > 0:
        base_pos_w = env.crane.data.root_pos_w[env_idx]
        base_quat_w = env.crane.data.root_quat_w[env_idx]
        pc_translated = world_pts - base_pos_w
        w, bx, by, bz = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]
        R = torch.stack([
            torch.stack([1 - 2*by*by - 2*bz*bz, 2*bx*by - 2*w*bz, 2*bx*bz + 2*w*by]),
            torch.stack([2*bx*by + 2*w*bz, 1 - 2*bx*bx - 2*bz*bz, 2*by*bz - 2*w*bx]),
            torch.stack([2*bx*bz - 2*w*by, 2*by*bz + 2*w*bx, 1 - 2*bx*bx - 2*by*by]),
        ])
        base_pts = (pc_translated @ R).cpu().numpy()
    else:
        base_pts = np.zeros((0, 3))
    data["base_points"] = base_pts

    # --- (e) FPS-sampled points ---
    fps_pts = get_log_pointcloud_base_frame(env, env_idx, num_points, depth_range=depth_range, raw_pcd=args_cli.raw_pcd)
    data["fps_points"] = fps_pts.cpu().numpy()

    return data


def save_pipeline_viz(pipeline_data, x, y, z, yaw, step_idx, viz_dir,
                      logs_grasped=None, alignment=None, stability=None,
                      depth_range=(1.0, 10.0),
                      bounds_min=None, bounds_max=None):
    """Save a 5-panel pipeline figure for one grasp step.

    Panels: (a) RGB, (b) raw depth, (c) segmented depth, (d) 3D PCD, (e) FPS PCD + prediction.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(22, 4.2))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.28)

    success = logs_grasped is not None and logs_grasped > 0
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    # --- (a) RGB ---
    ax_rgb = fig.add_subplot(gs[0, 0])
    if pipeline_data["rgb"] is not None:
        ax_rgb.imshow(pipeline_data["rgb"])
    else:
        ax_rgb.text(0.5, 0.5, 'RGB\nnot available', ha='center', va='center',
                    transform=ax_rgb.transAxes, fontsize=10, color='gray')
    ax_rgb.set_title(f'{panel_labels[0]} Camera RGB', fontsize=9, fontweight='bold')
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])

    # --- (b) Raw depth ---
    ax_depth = fig.add_subplot(gs[0, 1])
    depth_img = pipeline_data["depth"].copy()
    depth_clipped = np.clip(depth_img, depth_range[0], depth_range[1])
    depth_clipped[np.isinf(depth_img)] = np.nan
    im_d = ax_depth.imshow(depth_clipped, cmap='viridis', vmin=depth_range[0], vmax=depth_range[1])
    cb = plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label('depth (m)', fontsize=6)
    ax_depth.set_title(f'{panel_labels[1]} Raw depth', fontsize=9, fontweight='bold')
    ax_depth.set_xticks([])
    ax_depth.set_yticks([])

    # --- (c) Segmented depth (log-only) ---
    ax_seg = fig.add_subplot(gs[0, 2])
    masked = pipeline_data["masked_depth"].copy()
    im_s = ax_seg.imshow(masked, cmap='viridis', vmin=depth_range[0], vmax=depth_range[1])
    # Set background (NaN) to light gray
    ax_seg.set_facecolor('#e0e0e0')
    cb = plt.colorbar(im_s, ax=ax_seg, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label('depth (m)', fontsize=6)
    ax_seg.set_title(f'{panel_labels[2]} Log depth (masked)', fontsize=9, fontweight='bold')
    ax_seg.set_xticks([])
    ax_seg.set_yticks([])

    # --- (d) 3D point cloud in base frame (top-down, all log points) ---
    # Axes swapped (Y horizontal, X vertical) and vertical inverted to match camera view.
    ax_pcd = fig.add_subplot(gs[0, 3])
    base_pts = pipeline_data["base_points"]
    if len(base_pts) > 0:
        mask = np.any(base_pts != 0.0, axis=1)
        pts = base_pts[mask] if mask.any() else base_pts
        if len(pts) > 0:
            sc = ax_pcd.scatter(pts[:, 1], pts[:, 0], c=pts[:, 2], cmap='viridis',
                                s=0.5, alpha=0.6, rasterized=True)
            cb = plt.colorbar(sc, ax=ax_pcd, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
            cb.set_label('Z (m)', fontsize=6)
    if bounds_min is not None and bounds_max is not None:
        ax_pcd.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        ax_pcd.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)  # inverted
    ax_pcd.set_xlabel('Y (m)', fontsize=7)
    ax_pcd.set_ylabel('X (m)', fontsize=7)
    ax_pcd.set_title(f'{panel_labels[3]} 3D points (base frame)', fontsize=9, fontweight='bold')
    ax_pcd.set_aspect('equal')
    ax_pcd.tick_params(labelsize=6)

    # --- (e) FPS PCD + grapple prediction ---
    # Axes swapped (Y horizontal, X vertical) to match landscape camera panels.
    ax_fps = fig.add_subplot(gs[0, 4])
    fps_pts = pipeline_data["fps_points"]
    mask = np.any(fps_pts != 0.0, axis=1)
    pts = fps_pts[mask] if mask.any() else fps_pts
    if len(pts) > 0:
        sc = ax_fps.scatter(pts[:, 1], pts[:, 0], c=pts[:, 2], cmap='viridis',
                            s=1.0, alpha=0.6, rasterized=True)
        cb = plt.colorbar(sc, ax=ax_fps, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        cb.set_label('Z (m)', fontsize=6)

    # Draw grapple footprint (swap x↔y, adjust yaw for display axes)
    draw_grapple_footprint(ax_fps, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                           success=success, alpha=0.25)

    # Draw action bounds if available (vertical axis inverted)
    if bounds_min is not None and bounds_max is not None:
        from matplotlib.patches import Rectangle
        bw = bounds_max[1] - bounds_min[1]
        bh = bounds_max[0] - bounds_min[0]
        ax_fps.add_patch(Rectangle(
            (bounds_min[1], bounds_max[0]), bw, -bh,
            linewidth=0.8, edgecolor='#e67e22', facecolor='none',
            linestyle='--', alpha=0.6, zorder=3))

    # Annotate result (one item per line to keep legend compact)
    result_lines = [f"{'HIT' if success else 'MISS'}"]
    if logs_grasped is not None and logs_grasped > 0:
        result_lines[0] += f" ({logs_grasped} logs)"
        if alignment is not None:
            result_lines.append(f"align={alignment:.2f}")
        if stability is not None:
            result_lines.append(f"stab={stability:.2f}")
    knocked = pipeline_data.get("knocked_off")
    if knocked and knocked > 0:
        result_lines.append(f"knocked={knocked}")
    result_str = "\n".join(result_lines)
    result_color = '#27ae60' if success else '#c0392b'
    ax_fps.text(0.02, 0.98, result_str, transform=ax_fps.transAxes,
                fontsize=7, verticalalignment='top', color=result_color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    if bounds_min is not None and bounds_max is not None:
        ax_fps.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        ax_fps.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)  # inverted
    ax_fps.set_xlabel('Y (m)', fontsize=7)
    ax_fps.set_ylabel('X (m)', fontsize=7)
    ax_fps.set_title(f'{panel_labels[4]} FPS ({fps_pts.shape[0]} pts) + prediction',
                     fontsize=9, fontweight='bold')
    ax_fps.set_aspect('equal')
    ax_fps.tick_params(labelsize=6)

    # Suptitle (1-indexed)
    title = f'Grasp {step_idx + 1}'
    if logs_grasped is not None:
        title += f'  |  target=({x:.2f}, {y:.2f}, {z:.2f})  yaw={np.degrees(yaw):.0f}\u00b0'
        title += f'  |  {"SUCCESS" if success else "MISS"}'
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)

    out_path = os.path.join(viz_dir, f"pipeline_step_{step_idx:04d}.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved pipeline: {out_path}")


def save_stacked_pipeline_viz(grasp_data_list, episode_idx, viz_dir,
                              depth_range=(1.0, 10.0),
                              bounds_min=None, bounds_max=None,
                              suffix=""):
    """Save a vertically stacked pipeline figure with one row per grasp.

    Each row has 5 panels: (a) RGB, (b) raw depth, (c) segmented depth,
    (d) 3D PCD, (e) FPS PCD + prediction.

    Args:
        grasp_data_list: list of pipeline data dicts (one per selected grasp).
        episode_idx: episode number for filename.
        viz_dir: output directory.
        depth_range: clipping range for depth images.
        bounds_min, bounds_max: action bounds for overlaying on FPS panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_grasps = len(grasp_data_list)
    if n_grasps == 0:
        return

    row_height = 3.2
    fig = plt.figure(figsize=(22, row_height * n_grasps + 0.3))
    gs = gridspec.GridSpec(n_grasps, 5, figure=fig, wspace=0.25, hspace=0.25)

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for row, d in enumerate(grasp_data_list):
        x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
        step_idx = d["step_idx"]
        logs_grasped = d.get("logs_grasped")
        alignment = d.get("alignment")
        success = logs_grasped is not None and logs_grasped > 0

        # Row label on the left side
        result_tag = "HIT" if success else "MISS"
        n_grabbed = logs_grasped if logs_grasped else 0
        remaining = d.get("logs_remaining")
        knocked = d.get("knocked_off")
        row_label = f"Grasp {step_idx + 1}"
        if remaining is not None:
            row_label += f" | {remaining} left"
        row_label += f" | {result_tag} ({n_grabbed})"
        if knocked and knocked > 0:
            row_label += f" | {knocked} KO"

        # --- (a) RGB ---
        ax_rgb = fig.add_subplot(gs[row, 0])
        if d["rgb"] is not None:
            ax_rgb.imshow(d["rgb"])
        else:
            ax_rgb.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax_rgb.transAxes, fontsize=10, color='gray')
        if row == 0:
            ax_rgb.set_title(f'{panel_labels[0]} RGB', fontsize=9, fontweight='bold')
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        ax_rgb.set_ylabel(row_label, fontsize=8, fontweight='bold', rotation=90, labelpad=8)

        # --- (b) Raw depth ---
        ax_depth = fig.add_subplot(gs[row, 1])
        depth_img = d["depth"].copy()
        depth_clipped = np.clip(depth_img, depth_range[0], depth_range[1])
        depth_clipped[np.isinf(depth_img)] = np.nan
        im_d = ax_depth.imshow(depth_clipped, cmap='viridis',
                               vmin=depth_range[0], vmax=depth_range[1])
        cb_d = plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04, label='depth (m)')
        cb_d.ax.tick_params(labelsize=5)
        cb_d.set_label('depth (m)', fontsize=6)
        if row == 0:
            ax_depth.set_title(f'{panel_labels[1]} Raw depth', fontsize=9, fontweight='bold')
        ax_depth.set_xticks([])
        ax_depth.set_yticks([])

        # --- (c) Segmented depth ---
        ax_seg = fig.add_subplot(gs[row, 2])
        masked = d["masked_depth"].copy()
        im_s = ax_seg.imshow(masked, cmap='viridis',
                             vmin=depth_range[0], vmax=depth_range[1])
        ax_seg.set_facecolor('#e0e0e0')
        cb_s = plt.colorbar(im_s, ax=ax_seg, fraction=0.046, pad=0.04, label='depth (m)')
        cb_s.ax.tick_params(labelsize=5)
        cb_s.set_label('depth (m)', fontsize=6)
        if row == 0:
            ax_seg.set_title(f'{panel_labels[2]} Log depth', fontsize=9, fontweight='bold')
        ax_seg.set_xticks([])
        ax_seg.set_yticks([])

        # --- (d) 3D PCD (base frame, top-down) ---
        # Axes swapped (Y horizontal, X vertical), vertical inverted to match camera.
        ax_pcd = fig.add_subplot(gs[row, 3])
        base_pts = d["base_points"]
        sc_pcd = None
        if len(base_pts) > 0:
            mask = np.any(base_pts != 0.0, axis=1)
            pts = base_pts[mask] if mask.any() else base_pts
            if len(pts) > 0:
                sc_pcd = ax_pcd.scatter(pts[:, 1], pts[:, 0], c=pts[:, 2], cmap='viridis',
                                        s=0.5, alpha=0.6, rasterized=True)
        if bounds_min is not None and bounds_max is not None:
            ax_pcd.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax_pcd.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)  # inverted
        if row == 0:
            ax_pcd.set_title(f'{panel_labels[3]} 3D points', fontsize=9, fontweight='bold')
        if sc_pcd is not None:
            cb_pcd = plt.colorbar(sc_pcd, ax=ax_pcd, fraction=0.046, pad=0.04)
            cb_pcd.ax.tick_params(labelsize=5)
            cb_pcd.set_label('Z (m)', fontsize=6)
        ax_pcd.set_aspect('equal')
        ax_pcd.tick_params(labelsize=5)

        # --- (e) FPS PCD + grapple prediction ---
        # Axes swapped (Y horizontal, X vertical), vertical inverted to match camera.
        ax_fps = fig.add_subplot(gs[row, 4])
        fps_pts = d["fps_points"]
        mask = np.any(fps_pts != 0.0, axis=1)
        pts = fps_pts[mask] if mask.any() else fps_pts
        sc_fps = None
        if len(pts) > 0:
            sc_fps = ax_fps.scatter(pts[:, 1], pts[:, 0], c=pts[:, 2], cmap='viridis',
                                    s=1.0, alpha=0.6, rasterized=True)

        draw_grapple_footprint(ax_fps, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                               success=success, alpha=0.25)

        if bounds_min is not None and bounds_max is not None:
            from matplotlib.patches import Rectangle
            bw = bounds_max[1] - bounds_min[1]
            bh = bounds_max[0] - bounds_min[0]
            ax_fps.add_patch(Rectangle(
                (bounds_min[1], bounds_max[0]), bw, -bh,
                linewidth=0.8, edgecolor='#e67e22', facecolor='none',
                linestyle='--', alpha=0.6, zorder=3))

        # Result annotation (one item per line)
        stab = d.get("stability")
        knocked = d.get("knocked_off")
        result_lines = [f"{'HIT' if success else 'MISS'}"]
        if logs_grasped and logs_grasped > 0:
            result_lines[0] += f" ({logs_grasped})"
            if alignment is not None:
                result_lines.append(f"align={alignment:.2f}")
            if stab is not None:
                result_lines.append(f"stab={stab:.2f}")
        if knocked and knocked > 0:
            result_lines.append(f"knocked={knocked}")
        result_str = "\n".join(result_lines)
        result_color = '#27ae60' if success else '#c0392b'
        ax_fps.text(0.02, 0.98, result_str, transform=ax_fps.transAxes,
                    fontsize=6, verticalalignment='top', color=result_color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        if bounds_min is not None and bounds_max is not None:
            ax_fps.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax_fps.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)  # inverted
        if row == 0:
            ax_fps.set_title(f'{panel_labels[4]} FPS + prediction', fontsize=9, fontweight='bold')
        if sc_fps is not None:
            cb_fps = plt.colorbar(sc_fps, ax=ax_fps, fraction=0.046, pad=0.04)
            cb_fps.ax.tick_params(labelsize=5)
            cb_fps.set_label('Z (m)', fontsize=6)
        ax_fps.set_aspect('equal')
        ax_fps.tick_params(labelsize=5)

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_pipeline{suffix}.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved stacked pipeline: {out_path}")


def save_episode_progression(episode_data, episode_idx, viz_dir, bounds_min=None, bounds_max=None):
    """Save a 3x5 grid showing grasp predictions across an episode.

    Args:
        episode_data: list of dicts, one per grasp step. Each dict has:
            fps_points, x, y, z, yaw, step_idx, logs_grasped, alignment, logs_remaining
        episode_idx: episode number for filename.
        viz_dir: output directory.
        bounds_min, bounds_max: (3,) arrays for consistent axis limits.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_total = len(episode_data)
    if n_total == 0:
        return

    # Select grasps for beginning, middle, end (up to 5 each)
    n_per_row = 5

    # Beginning: first 5 grasps
    early_indices = list(range(min(n_per_row, n_total)))

    # Middle: 5 grasps around midpoint
    mid_center = n_total // 2
    mid_start = max(0, mid_center - n_per_row // 2)
    mid_indices = list(range(mid_start, min(mid_start + n_per_row, n_total)))

    # End: last 5 grasps
    end_start = max(0, n_total - n_per_row)
    end_indices = list(range(end_start, n_total))

    rows = [
        ("Early", early_indices),
        ("Mid", mid_indices),
        ("Late", end_indices),
    ]

    # Compute consistent axis limits from bounds or from all data.
    # Axes are swapped (Y horizontal, X vertical) for landscape ratio,
    # so x_lim (plot horizontal) uses Y bounds, y_lim (plot vertical) uses X bounds.
    # Vertical axis inverted (high→low) so front of scene matches camera bottom.
    if bounds_min is not None and bounds_max is not None:
        x_lim = (bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        y_lim = (bounds_max[0] + 0.5, bounds_min[0] - 0.5)  # inverted
    else:
        all_pts = np.concatenate([d["fps_points"] for d in episode_data], axis=0)
        mask = np.any(all_pts != 0.0, axis=1)
        valid = all_pts[mask] if mask.any() else all_pts
        if len(valid) > 0:
            margin = 0.5
            x_lim = (valid[:, 1].min() - margin, valid[:, 1].max() + margin)
            y_lim = (valid[:, 0].max() + margin, valid[:, 0].min() - margin)  # inverted
        else:
            x_lim = (-5, 5)
            y_lim = (5, -5)

    n_rows = len(rows)
    max_cols = max(len(idx) for _, idx in rows)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(4 * max_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if max_cols == 1:
        axes = axes[:, np.newaxis]

    for row_i, (row_label, indices) in enumerate(rows):
        for col_j in range(max_cols):
            ax = axes[row_i, col_j]
            if col_j >= len(indices):
                ax.axis('off')
                continue

            d = episode_data[indices[col_j]]
            fps_pts = d["fps_points"]
            mask = np.any(fps_pts != 0.0, axis=1)
            pts = fps_pts[mask] if mask.any() else fps_pts

            success = d["logs_grasped"] is not None and d["logs_grasped"] > 0

            # Axes swapped (Y horizontal, X vertical) for landscape ratio.
            if len(pts) > 0:
                ax.scatter(pts[:, 1], pts[:, 0], c=pts[:, 2], cmap='viridis',
                           s=1.0, alpha=0.6, rasterized=True)

            # Draw grapple footprint (swap x↔y, adjust yaw)
            draw_grapple_footprint(ax, d["y"], d["x"], np.pi / 2 - d["yaw"],
                                   width=1.5, length=0.5,
                                   success=success, alpha=0.3)

            # Draw action bounds (vertical axis inverted)
            if bounds_min is not None and bounds_max is not None:
                from matplotlib.patches import Rectangle
                bw = bounds_max[1] - bounds_min[1]
                bh = bounds_max[0] - bounds_min[0]
                ax.add_patch(Rectangle(
                    (bounds_min[1], bounds_max[0]), bw, -bh,
                    linewidth=0.5, edgecolor='#e67e22', facecolor='none',
                    linestyle='--', alpha=0.4, zorder=3))

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.tick_params(labelsize=5)

            # Column header (1-indexed)
            result_tag = "HIT" if success else "MISS"
            n_grasped = d["logs_grasped"] if d["logs_grasped"] else 0
            col_title = f"grasp {d['step_idx'] + 1}"
            if d.get("logs_remaining") is not None:
                col_title += f"\n{d['logs_remaining']} left"
            col_title += f"\n{result_tag} ({n_grasped})"
            knocked = d.get("knocked_off")
            if knocked and knocked > 0:
                col_title += f" / {knocked} KO"
            ax.set_title(col_title, fontsize=7, pad=3)

            if col_j == 0:
                first_g = episode_data[indices[0]]["step_idx"] + 1
                last_g = episode_data[indices[-1]]["step_idx"] + 1
                grasp_range = f"{first_g}\u2013{last_g}" if len(indices) > 1 else str(first_g)
                ax.set_ylabel(f'{row_label}\n(grasps {grasp_range})',
                              fontsize=8, fontweight='bold')
            else:
                ax.set_yticklabels([])

            if row_i < n_rows - 1:
                ax.set_xticklabels([])

    fig.suptitle(f'Episode {episode_idx} \u2014 Grasp Progression ({n_total} total grasps)',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_progression.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved progression: {out_path}")


def main():
    # Load policy
    policy, num_points, metadata = load_bc_policy(args_cli.checkpoint, args_cli.device)

    # Detect action dim from checkpoint
    checkpoint = torch.load(args_cli.checkpoint, map_location=args_cli.device)
    action_dim = checkpoint.get('action_dim', 4)
    print(f"[Play] Num points: {num_points}, Action dim: {action_dim}")

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.action_space = action_dim  # Match policy output (4D or 5D)
    cfg.enable_camera = True
    if args_cli.paper_viz:
        cfg.camera_cfg.data_types = ["rgb", "depth", "semantic_segmentation"]
    else:
        cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.enable_domain_randomization = args_cli.domain_randomization
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[Play] Environment created with {env.num_envs} envs")
    print(f"[Play] Domain randomization: {args_cli.domain_randomization}")
    if args_cli.seed is not None:
        print(f"[Play] Seed: {args_cli.seed}")
    if args_cli.obs_noise > 0:
        print(f"[Play] Observation noise σ: {args_cli.obs_noise} m")
    if args_cli.action_noise > 0:
        print(f"[Play] Action noise σ: {args_cli.action_noise}")

    # Set up visualization directory
    if args_cli.visualize:
        viz_dir = args_cli.viz_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[Play] Saving visualizations to: {viz_dir}")

    # Set up paper viz directory
    if args_cli.paper_viz:
        paper_viz_dir = args_cli.viz_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "paper_viz")
        os.makedirs(paper_viz_dir, exist_ok=True)
        print(f"[Play] Saving paper visualizations to: {paper_viz_dir}")
        # Per-episode data collection: list of per-step dicts
        paper_episode_data = []

    # Reset and initialize camera
    env.reset()
    env.sim.render()
    env._camera.update(dt=env.cfg.sim.dt)

    # Ensure action bounds are computed before visualization loop
    # (bounds are lazily computed inside _apply_action, so they're zero before the first step)
    env._compute_action_space_bounds()

    # Diagnostic: verify bounds were actually computed
    if args_cli.visualize:
        root_pos = env.crane.data.root_pos_w[0].cpu().numpy()
        root_quat = env.crane.data.root_quat_w[0].cpu().numpy()
        b_min = env._action_bounds_min[0].cpu().numpy()
        b_max = env._action_bounds_max[0].cpu().numpy()
        print(f"[Viz-Init] Crane root pos_w: {root_pos}")
        print(f"[Viz-Init] Crane root quat_w: {root_quat}")
        print(f"[Viz-Init] Computed bounds min: {b_min}")
        print(f"[Viz-Init] Computed bounds max: {b_max}")
        print(f"[Viz-Init] Bounds range: X={b_max[0]-b_min[0]:.2f}, "
              f"Y={b_max[1]-b_min[1]:.2f}, Z={b_max[2]-b_min[2]:.2f}")
        if np.allclose(b_min, 0) and np.allclose(b_max, 0):
            print("[Viz-Init] WARNING: bounds are still zero!")

    # Tracking
    episodes_done = 0
    total_reward = 0.0
    total_logs_grasped = 0
    total_grasps = 0
    successful_grasps = 0
    failed_grasps = 0
    total_alignment = 0.0
    total_stability = 0.0
    clearing_percentages = []
    episode_rewards_list = []
    logs_per_episode = []
    piles_fully_cleared = 0
    total_knocked_off = 0
    knocked_off_per_episode = []
    logs_cleared_per_episode = []

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_successful_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_failed_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_alignment_sum = torch.zeros(env.num_envs, device=env.device)
    ep_stability_sum = torch.zeros(env.num_envs, device=env.device)
    ep_cycle_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_clearing_curves = [[] for _ in range(env.num_envs)]  # per-env list of clearing % at each cycle

    # Per-episode result lists (for mean ± std reporting)
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []
    per_ep_cycles = []
    per_ep_clearing_curves = []

    # Get starting log counts
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[Play] Logs per pile: {sum(log_counts)/len(log_counts):.0f} avg")
    else:
        default_logs = 200
        episode_starting_logs[:] = default_logs
        print(f"[Play] Logs per pile: {default_logs}")

    print(f"\n[Play] Running {args_cli.num_episodes} episodes...")

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Get point cloud observations
            pc_batch = []
            for i in range(env.num_envs):
                pc = get_log_pointcloud_base_frame(env, i, num_points, raw_pcd=args_cli.raw_pcd)
                pc_batch.append(pc)
            obs = torch.stack(pc_batch)  # (num_envs, num_points, 3)

            # Inject observation noise (Gaussian on PCD coordinates)
            if args_cli.obs_noise > 0:
                obs = obs + torch.randn_like(obs) * args_cli.obs_noise

            # Get policy action
            actions = policy(obs)

            # Inject action noise (Gaussian on raw policy output, before workspace scaling)
            if args_cli.action_noise > 0:
                actions = actions + torch.randn_like(actions) * args_cli.action_noise

            # Debug: print action stats occasionally
            if total_grasps < 5 or total_grasps % 100 == 0:
                print(f"[Debug] Grasp {total_grasps}: points={obs.shape}, "
                      f"action=[{actions.min():.2f},{actions.max():.2f}]")

            # Pre-step: collect pipeline data for paper viz (before env.step modifies state)
            if args_cli.paper_viz:
                paper_pre_step = []
                for i in range(env.num_envs):
                    min_b = env._action_bounds_min[i].cpu().numpy()
                    max_b = env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    pdata = get_pipeline_data(env, i, num_points)
                    pdata["x"] = x
                    pdata["y"] = y
                    pdata["z"] = z
                    pdata["yaw"] = yaw
                    pdata["bounds_min"] = min_b.copy()
                    pdata["bounds_max"] = max_b.copy()
                    pdata["step_idx"] = total_grasps + i
                    paper_pre_step.append(pdata)

            # Pre-step: decode actions for visualization (before env.step modifies state)
            if args_cli.visualize:
                viz_decoded = []
                viz_bounds = []
                for i in range(env.num_envs):
                    min_b = env._action_bounds_min[i].cpu().numpy()
                    max_b = env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    pts = obs[i].cpu().numpy()
                    if obs[i].dim() == 1:
                        pts = pts.reshape(-1, 3)
                    viz_decoded.append((pts, x, y, z, yaw))
                    viz_bounds.append((min_b.copy(), max_b.copy()))

                    # One-time diagnostic
                    if total_grasps == 0 and i == 0:
                        mask = np.any(pts != 0.0, axis=1)
                        diag_pts = pts[mask] if mask.any() else pts
                        pc_min = diag_pts.min(axis=0)
                        pc_max = diag_pts.max(axis=0)
                        print(f"[Viz] Action bounds min: {min_b}")
                        print(f"[Viz] Action bounds max: {max_b}")
                        print(f"[Viz] Point cloud  min: {pc_min}")
                        print(f"[Viz] Point cloud  max: {pc_max}")
                        print(f"[Viz] Decoded target: ({x:.3f}, {y:.3f}, {z:.3f})")
                        overlap_x = min_b[0] <= pc_max[0] and max_b[0] >= pc_min[0]
                        overlap_y = min_b[1] <= pc_max[1] and max_b[1] >= pc_min[1]
                        overlap_z = min_b[2] <= pc_max[2] and max_b[2] >= pc_min[2]
                        star_in_pc = (pc_min[0] <= x <= pc_max[0] and
                                      pc_min[1] <= y <= pc_max[1] and
                                      pc_min[2] <= z <= pc_max[2])
                        print(f"[Viz] Bounds/PC overlap: X={overlap_x} Y={overlap_y} Z={overlap_z}")
                        print(f"[Viz] Star inside point cloud bbox: {star_in_pc}")

            # Step environment
            _, rew, terminated, truncated, _ = env.step(actions)
            env.sim.render()
            env._camera.update(dt=env.cfg.sim.dt)

            episode_rewards += rew

            # Track metrics
            for i in range(env.num_envs):
                logs_grasped = int(env._prev_logs_grasped[i].item())
                alignment = env._prev_grasp_alignment[i].item() if hasattr(env, '_prev_grasp_alignment') else 0.0
                stability = env._prev_grasp_stability[i].item() if hasattr(env, '_prev_grasp_stability') else 1.0

                # Save visualization with grasp result
                if args_cli.visualize:
                    pts, vx, vy, vz, vyaw = viz_decoded[i]
                    vmin_b, vmax_b = viz_bounds[i]
                    save_step_viz(pts, vx, vy, vz, vyaw,
                                  total_grasps, viz_dir,
                                  logs_grasped=logs_grasped,
                                  alignment=alignment,
                                  bounds_min=vmin_b, bounds_max=vmax_b)

                # Collect paper viz data (post-step: now we know the grasp result)
                if args_cli.paper_viz:
                    pdata = paper_pre_step[i]
                    pdata["logs_grasped"] = logs_grasped
                    pdata["alignment"] = alignment
                    pdata["stability"] = stability
                    # Compute remaining from pre-step snapshot (env may have
                    # already reset if episode terminated inside step()).
                    knocked_off = int(env._prev_cycle_knocked_off[i].item()) if hasattr(env, '_prev_cycle_knocked_off') else 0
                    if hasattr(env, '_prev_logs_remaining'):
                        pre_remaining = int(env._prev_logs_remaining[i].item())
                        logs_remaining = pre_remaining - logs_grasped - knocked_off
                    else:
                        logs_remaining = None
                    pdata["logs_remaining"] = logs_remaining
                    pdata["knocked_off"] = knocked_off
                    paper_episode_data.append(pdata)

                total_grasps += 1
                total_logs_grasped += logs_grasped
                episode_logs_cleared[i] += logs_grasped
                ep_cycle_count[i] += 1
                starting = max(1, int(episode_starting_logs[i].item()))
                clear_pct_now = int(episode_logs_cleared[i].item()) / starting * 100
                ep_clearing_curves[i].append(round(clear_pct_now, 1))
                if logs_grasped > 0:
                    successful_grasps += 1
                    total_alignment += alignment
                    total_stability += stability
                    ep_successful_grasps[i] += 1
                    ep_alignment_sum[i] += alignment * logs_grasped
                    ep_stability_sum[i] += stability * logs_grasped
                else:
                    failed_grasps += 1
                    ep_failed_grasps[i] += 1

            # Check episode completion
            done = terminated | truncated
            for i in range(env.num_envs):
                if done[i]:
                    ep_reward = episode_rewards[i].item()
                    total_reward += ep_reward
                    episodes_done += 1

                    starting_logs = int(episode_starting_logs[i].item())
                    logs_cleared = int(episode_logs_cleared[i].item())
                    clear_pct = (logs_cleared / max(1, starting_logs)) * 100
                    clearing_percentages.append(clear_pct)
                    episode_rewards_list.append(ep_reward)
                    logs_per_episode.append(starting_logs)

                    if logs_cleared >= starting_logs:
                        piles_fully_cleared += 1

                    # Capture knocked-off count before env resets it
                    # Read episode knocked-off from snapshot (survives _reset_idx)
                    if hasattr(env, '_final_episode_knocked_off'):
                        ep_knocked_off = int(env._final_episode_knocked_off[i].item())
                    elif hasattr(env, '_logs_knocked_off'):
                        ep_knocked_off = int(env._logs_knocked_off[i].item())
                    else:
                        ep_knocked_off = 0
                    total_knocked_off += ep_knocked_off
                    knocked_off_per_episode.append(ep_knocked_off)
                    logs_cleared_per_episode.append(logs_cleared)

                    per_ep_cycles.append(int(ep_cycle_count[i].item()))
                    per_ep_clearing_curves.append(ep_clearing_curves[i][:])  # copy the curve

                    # Per-episode grasp metrics
                    n_success = int(ep_successful_grasps[i].item())
                    n_fail = int(ep_failed_grasps[i].item())
                    n_total = n_success + n_fail
                    per_ep_success_rates.append(n_success / max(1, n_total) * 100)
                    per_ep_throughputs.append(logs_cleared / max(1, n_success))
                    per_ep_alignments.append(float(ep_alignment_sum[i].item()) / max(1, logs_cleared))
                    per_ep_stabilities.append(float(ep_stability_sum[i].item()) / max(1, logs_cleared))

                    print(f"[Play] Episode {episodes_done}: reward={ep_reward:.2f}, "
                          f"cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs}), knocked_off={ep_knocked_off}")

                    # Generate paper viz figures for this episode
                    if args_cli.paper_viz and len(paper_episode_data) > 0:
                        # Trim stuck tail: consecutive misses at the end with
                        # the same remaining count are redundant (policy stuck).
                        # Keep none of the stuck repeats — the last successful
                        # or first-miss grasp is more informative.
                        trimmed = list(paper_episode_data)
                        if len(trimmed) >= 2:
                            last_remaining = trimmed[-1].get("logs_remaining")
                            if last_remaining is not None:
                                # Walk backward to find where the stuck run starts
                                stuck_start = len(trimmed)
                                for si in range(len(trimmed) - 1, -1, -1):
                                    d = trimmed[si]
                                    if (d.get("logs_remaining") == last_remaining
                                            and (d.get("logs_grasped") or 0) == 0):
                                        stuck_start = si
                                    else:
                                        break
                                # Remove ALL stuck repeats
                                if stuck_start < len(trimmed):
                                    trimmed = trimmed[:stuck_start]
                                    n_removed = len(paper_episode_data) - len(trimmed)
                                    if n_removed > 0:
                                        print(f"[PaperViz] Trimmed {n_removed} "
                                              f"stuck repeated grasps (remaining={last_remaining})")
                        # Safety: keep at least 1 entry
                        if len(trimmed) == 0:
                            trimmed = [paper_episode_data[0]]

                        # Get bounds for consistent axes
                        ep_bounds_min = trimmed[0].get("bounds_min")
                        ep_bounds_max = trimmed[0].get("bounds_max")

                        # Save stacked pipeline figure for representative grasps
                        # (one from beginning, middle, end)
                        n_ep = len(trimmed)
                        representative = [0]
                        if n_ep > 2:
                            representative.append(n_ep // 2)
                        if n_ep > 1:
                            representative.append(n_ep - 1)
                        rep_data = [trimmed[idx] for idx in representative]
                        save_stacked_pipeline_viz(
                            rep_data, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min,
                            bounds_max=ep_bounds_max,
                        )

                        # Save a second stacked pipeline with 3 consecutive
                        # grasps from beginning, middle, and end
                        if n_ep >= 6:
                            consec = []
                            # Beginning: grasps 0,1,2
                            consec.extend(trimmed[0:3])
                            # Middle: 3 around midpoint
                            mid = n_ep // 2
                            mid_start = max(0, mid - 1)
                            consec.extend(trimmed[mid_start:mid_start + 3])
                            # End: last 3
                            consec.extend(trimmed[max(0, n_ep - 3):n_ep])
                            save_stacked_pipeline_viz(
                                consec, episodes_done, paper_viz_dir,
                                bounds_min=ep_bounds_min,
                                bounds_max=ep_bounds_max,
                                suffix="_consecutive",
                            )

                        # Save episode progression grid
                        save_episode_progression(
                            trimmed, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min,
                            bounds_max=ep_bounds_max,
                        )

                        # --- Paper-compact variants (2-row, \textwidth) ---
                        from eval_viz import (
                            save_paper_pipeline_viz, save_paper_progression_viz,
                            save_paper_pipeline_quadrant_single,
                            save_paper_pipeline_quadrant_rgb_fps,
                            save_paper_pipeline_quadrant_depth_fps,
                            pick_early_late_grasps,
                        )
                        # Pipeline: random successful early + late grasp
                        paper_rep = pick_early_late_grasps(trimmed)
                        save_paper_pipeline_viz(
                            paper_rep, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min,
                            bounds_max=ep_bounds_max,
                        )
                        # Quadrant variants (2x2, \columnwidth)
                        save_paper_pipeline_quadrant_single(
                            paper_rep, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min, bounds_max=ep_bounds_max,
                        )
                        save_paper_pipeline_quadrant_rgb_fps(
                            paper_rep, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min, bounds_max=ep_bounds_max,
                        )
                        save_paper_pipeline_quadrant_depth_fps(
                            paper_rep, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min, bounds_max=ep_bounds_max,
                        )
                        # Progression: 2 rows x 5 cols (early + late)
                        save_paper_progression_viz(
                            trimmed, episodes_done, paper_viz_dir,
                            bounds_min=ep_bounds_min,
                            bounds_max=ep_bounds_max,
                        )

                        # Reset for next episode
                        paper_episode_data = []

                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    ep_successful_grasps[i] = 0
                    ep_failed_grasps[i] = 0
                    ep_alignment_sum[i] = 0.0
                    ep_stability_sum[i] = 0.0
                    ep_cycle_count[i] = 0
                    ep_clearing_curves[i] = []
                    if has_variable_logs:
                        episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # Helper: mean and sample std dev
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals, mean_val):
        return (sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0

    # All metrics are per-episode averages (each episode weighted equally)
    avg_reward = _mean(episode_rewards_list)
    avg_clear_pct = _mean(clearing_percentages)
    avg_success_rate = _mean(per_ep_success_rates)
    avg_throughput = _mean(per_ep_throughputs)
    avg_alignment = _mean(per_ep_alignments)
    avg_stability = _mean(per_ep_stabilities)
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    knocked_off_pcts = [knocked_off_per_episode[j] / max(1, logs_per_episode[j]) * 100
                        for j in range(len(knocked_off_per_episode))]
    avg_knocked_off_pct = _mean(knocked_off_pcts)

    avg_cycles = _mean(per_ep_cycles)
    std_cycles = _std(per_ep_cycles, avg_cycles)

    # Derive cycles to 95% from clearing curves
    per_ep_cycles_to_95 = []
    for curve in per_ep_clearing_curves:
        c95 = next((i + 1 for i, pct in enumerate(curve) if pct >= 95.0), len(curve))
        per_ep_cycles_to_95.append(c95)
    avg_cycles_to_95 = _mean(per_ep_cycles_to_95)
    std_cycles_to_95 = _std(per_ep_cycles_to_95, avg_cycles_to_95)

    std_reward = _std(episode_rewards_list, avg_reward)
    std_clear_pct = _std(clearing_percentages, avg_clear_pct)
    std_success_rate = _std(per_ep_success_rates, avg_success_rate)
    std_throughput = _std(per_ep_throughputs, avg_throughput)
    std_alignment = _std(per_ep_alignments, avg_alignment)
    std_stability = _std(per_ep_stabilities, avg_stability)
    std_knocked_off_pct = _std(knocked_off_pcts, avg_knocked_off_pct)

    # Print summary (all values are mean ± std across episodes)
    print("=" * 60)
    print(f"\n[Play] ====== RESULTS ({episodes_done} episodes) ======")
    print(f"[Play] Episode Reward:      {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"[Play] Pile Cleared:        {avg_clear_pct:.1f} ± {std_clear_pct:.1f}%")
    print(f"[Play] Full Clear Rate:     {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
    print(f"[Play] Grasp Success Rate:  {avg_success_rate:.1f} ± {std_success_rate:.1f}%")
    print(f"[Play] Throughput:          {avg_throughput:.2f} ± {std_throughput:.2f} logs/grasp")
    print(f"[Play] Alignment:           {avg_alignment:.3f} ± {std_alignment:.3f}")
    print(f"[Play] Stability:           {avg_stability:.3f} ± {std_stability:.3f}")
    print(f"[Play] Knocked Off:         {avg_knocked_off_pct:.1f} ± {std_knocked_off_pct:.1f}%")
    print(f"[Play] Avg Cycles:          {avg_cycles:.1f} ± {std_cycles:.1f}")
    print(f"[Play] Cycles to 95%:       {avg_cycles_to_95:.1f} ± {std_cycles_to_95:.1f}")
    print(f"[Play] Total Logs Grasped:  {total_logs_grasped}")
    print(f"[Play] =======================")

    # Save metrics
    if args_cli.save_metrics:
        output_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

        metrics = {
            "eval_config": {
                "method": "bc_pointcloud",
                "checkpoint": args_cli.checkpoint,
                "num_envs": args_cli.num_envs,
                "num_episodes": episodes_done,
                "domain_randomization": args_cli.domain_randomization,
                "seed": args_cli.seed,
                "obs_noise": args_cli.obs_noise,
                "action_noise": args_cli.action_noise,
                "num_points": num_points,
                "timestamp": timestamp,
            },
            "episodes": {
                "total": episodes_done,
                "logs_per_pile_avg": float(_mean(logs_per_episode)),
                "logs_per_pile_min": int(min(logs_per_episode)) if logs_per_episode else 0,
                "logs_per_pile_max": int(max(logs_per_episode)) if logs_per_episode else 0,
            },
            "summary": {
                "reward":            {"mean": avg_reward, "std": std_reward},
                "pile_cleared_pct":  {"mean": avg_clear_pct, "std": std_clear_pct},
                "full_clear_rate":   full_clear_rate,
                "grasp_success_pct": {"mean": avg_success_rate, "std": std_success_rate},
                "throughput":        {"mean": avg_throughput, "std": std_throughput},
                "alignment":         {"mean": avg_alignment, "std": std_alignment},
                "stability":         {"mean": avg_stability, "std": std_stability},
                "knocked_off_pct":   {"mean": avg_knocked_off_pct, "std": std_knocked_off_pct},
                "cycles":            {"mean": avg_cycles, "std": std_cycles},
                "cycles_to_95pct":   {"mean": avg_cycles_to_95, "std": std_cycles_to_95},
                "total_logs_grasped": total_logs_grasped,
                "total_grasps": total_grasps,
                "total_successful_grasps": successful_grasps,
                "total_failed_grasps": failed_grasps,
            },
            "per_episode": {
                "rewards": episode_rewards_list,
                "clearing_pcts": clearing_percentages,
                "success_rates": per_ep_success_rates,
                "throughputs": per_ep_throughputs,
                "alignments": per_ep_alignments,
                "stabilities": per_ep_stabilities,
                "knocked_off_counts": knocked_off_per_episode,
                "knocked_off_pcts": knocked_off_pcts,
                "logs_cleared": logs_cleared_per_episode,
                "logs_per_pile": logs_per_episode,
                "cycles": per_ep_cycles,
                "cycles_to_95pct": per_ep_cycles_to_95,
                "clearing_curves": per_ep_clearing_curves,
            },
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Play] Metrics saved to: {metrics_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
