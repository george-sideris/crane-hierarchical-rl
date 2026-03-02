"""Evaluation visualization utilities for raw PCD pipeline.

Shared utilities (FPS, base-frame transform, action decode, grapple drawing,
step viz) are standalone copies from play_bc_pointcloud.py so that play.py
(RL eval) can import them without pulling in the BC-specific code.

New functions produce 4-panel raw-pipeline figures (no segmentation step):
  (a) Camera RGB  (b) Raw depth  (c) 3D PCD (all points)  (d) FPS PCD + prediction
"""

import os
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
    from matplotlib.patches import Polygon

    color = '#2ecc71' if success else '#e74c3c'
    edge_color = '#27ae60' if success else '#c0392b'

    hw = width / 2
    hl = length / 2
    corners = np.array([
        [-hw, -hl],
        [ hw, -hl],
        [ hw,  hl],
        [-hw,  hl],
        [-hw, -hl],
    ])

    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = corners @ R.T

    rotated[:, 0] += x
    rotated[:, 1] += y

    poly = Polygon(rotated[:-1], closed=True,
                   facecolor=color, edgecolor=edge_color,
                   alpha=alpha, linewidth=1.5, zorder=5)
    ax.add_patch(poly)

    arrow_len = length * 0.8
    dx = arrow_len * np.cos(yaw)
    dy = arrow_len * np.sin(yaw)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color=edge_color, lw=2),
                zorder=6)


def save_step_viz(points_np, x, y, z, yaw, step_idx, viz_dir,
                  logs_grasped=None, alignment=None,
                  bounds_min=None, bounds_max=None):
    """Save a 2-panel visualization PNG for one grasp step.

    Left:  top-down view (X vs Y) -- horizontal placement + yaw arrow
    Right: side view (Y vs Z) -- height targeting
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))

    mask = np.any(points_np != 0.0, axis=1)
    pts = points_np[mask] if mask.any() else points_np

    pc_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
    pc_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)

    # -- Left panel: top-down (X vs Y) colored by height --
    if len(pts) > 0:
        sc = ax_top.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis',
                            s=1, alpha=0.5, label='Point cloud')
        plt.colorbar(sc, ax=ax_top, label='Z (height)')
    ax_top.plot(x, y, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                label=f'Target ({x:.2f}, {y:.2f})')
    arrow_len = 0.3
    ax_top.annotate('', xy=(x + arrow_len * np.cos(yaw), y + arrow_len * np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    if len(pts) > 0:
        ax_top.add_patch(Rectangle(
            (pc_min[0], pc_min[1]), pc_max[0] - pc_min[0], pc_max[1] - pc_min[1],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
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
    if len(pts) > 0:
        ax_side.add_patch(Rectangle(
            (pc_min[1], pc_min[2]), pc_max[1] - pc_min[1], pc_max[2] - pc_min[2],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
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

    title = f'Step {step_idx} | Target: ({x:.3f}, {y:.3f}, {z:.3f}) yaw={np.degrees(yaw):.1f}\u00b0'
    if logs_grasped is not None:
        result = 'SUCCESS' if logs_grasped > 0 else 'MISS'
        title += f' | {result} ({logs_grasped} logs)'
    if alignment is not None and logs_grasped and logs_grasped > 0:
        title += f' | align={alignment:.3f}'
    fig.suptitle(title, fontsize=11)

    fig.savefig(os.path.join(viz_dir, f"step_{step_idx:04d}.png"),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Raw PCD pipeline data collector
# ---------------------------------------------------------------------------

def get_raw_pipeline_data(env, env_idx: int, num_points: int,
                          depth_range: tuple = (1.0, 10.0),
                          raw_pcd: bool = False) -> dict:
    """Collect 4-panel raw pipeline data for one environment.

    Returns dict with keys:
        rgb: (H, W, 3) uint8 numpy or None
        depth: (H, W) float32 numpy
        base_points: (N, 3) numpy -- all valid 3D points in base frame (before FPS)
        fps_points: (num_points, 3) numpy -- FPS-sampled subset
    """
    data = {}

    # --- (a) RGB ---
    if hasattr(env, '_camera') and "rgb" in env.cfg.camera_cfg.data_types:
        rgb_tensor = env._camera.data.output["rgb"][env_idx]  # (H, W, 4) RGBA
        data["rgb"] = rgb_tensor[:, :, :3].cpu().numpy().astype(np.uint8)
    else:
        data["rgb"] = None

    # --- (b) Raw depth ---
    depth_tensor = env._camera.data.output["depth"][env_idx].squeeze(-1)  # (H, W)
    data["depth"] = depth_tensor.cpu().numpy().copy()

    # --- Base-frame rotation matrix (used for both raw and log points) ---
    base_pos_w = env.crane.data.root_pos_w[env_idx]
    base_quat_w = env.crane.data.root_quat_w[env_idx]
    w, bx, by, bz = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]
    R = torch.stack([
        torch.stack([1 - 2*by*by - 2*bz*bz, 2*bx*by - 2*w*bz, 2*bx*bz + 2*w*by]),
        torch.stack([2*bx*by + 2*w*bz, 1 - 2*bx*bx - 2*bz*bz, 2*by*bz - 2*w*bx]),
        torch.stack([2*bx*bz - 2*w*by, 2*by*bz + 2*w*bx, 1 - 2*bx*bx - 2*by*by]),
    ])

    # --- (c) All valid 3D points in base frame (before FPS) ---
    if raw_pcd:
        pc_world = env.get_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
    else:
        pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)

    if pc_world.shape[0] > 0:
        pc_translated = pc_world - base_pos_w
        base_pts = (pc_translated @ R).cpu().numpy()
    else:
        base_pts = np.zeros((0, 3))
    data["base_points"] = base_pts

    # --- Log-only points for visualization labeling (not fed to policy) ---
    if raw_pcd:
        log_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
        if log_world.shape[0] > 0:
            log_translated = log_world - base_pos_w
            log_base = (log_translated @ R).cpu().numpy()
        else:
            log_base = np.zeros((0, 3))
        data["log_base_points"] = log_base

    # --- (d) FPS-sampled points ---
    fps_pts = get_log_pointcloud_base_frame(env, env_idx, num_points,
                                            depth_range=depth_range, raw_pcd=raw_pcd)
    data["fps_points"] = fps_pts.cpu().numpy()

    return data


# ---------------------------------------------------------------------------
# 4-panel raw pipeline figure
# ---------------------------------------------------------------------------

def save_raw_pipeline_viz(pipeline_data, x, y, z, yaw, step_idx, viz_dir,
                          logs_grasped=None, alignment=None, stability=None,
                          depth_range=(1.0, 10.0),
                          bounds_min=None, bounds_max=None):
    """Save a 4-panel raw pipeline figure for one grasp step.

    Panels: (a) RGB, (b) raw depth, (c) 3D PCD (all points), (d) FPS PCD + prediction.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 4.2))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    success = logs_grasped is not None and logs_grasped > 0
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

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
    im_d = ax_depth.imshow(depth_clipped, cmap='viridis',
                           vmin=depth_range[0], vmax=depth_range[1])
    cb = plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=5)
    cb.set_label('depth (m)', fontsize=6)
    ax_depth.set_title(f'{panel_labels[1]} Raw depth', fontsize=9, fontweight='bold')
    ax_depth.set_xticks([])
    ax_depth.set_yticks([])

    # --- (c) 3D point cloud in base frame (top-down) — dual-layer ---
    ax_pcd = fig.add_subplot(gs[0, 2])
    base_pts = pipeline_data["base_points"]
    log_pts = pipeline_data.get("log_base_points", np.zeros((0, 3)))

    # Background: all raw points as faint gray
    if len(base_pts) > 0:
        mask = np.any(base_pts != 0.0, axis=1)
        bg = base_pts[mask] if mask.any() else base_pts
        if len(bg) > 0:
            ax_pcd.scatter(bg[:, 1], bg[:, 0], c='#aaaaaa', s=0.3, alpha=0.3, rasterized=True)

    # Foreground: log-only points in viridis
    if len(log_pts) > 0:
        mask = np.any(log_pts != 0.0, axis=1)
        lp = log_pts[mask] if mask.any() else log_pts
        if len(lp) > 0:
            sc = ax_pcd.scatter(lp[:, 1], lp[:, 0], c=lp[:, 2], cmap='viridis',
                                s=1.0, alpha=0.7, rasterized=True)
            cb = plt.colorbar(sc, ax=ax_pcd, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
            cb.set_label('Z (m)', fontsize=6)
    if bounds_min is not None and bounds_max is not None:
        from matplotlib.patches import Rectangle
        bw = bounds_max[1] - bounds_min[1]
        bh = bounds_max[0] - bounds_min[0]
        ax_pcd.add_patch(Rectangle(
            (bounds_min[1], bounds_max[0]), bw, -bh,
            linewidth=0.8, edgecolor='#e67e22', facecolor='none',
            linestyle='--', alpha=0.6, zorder=3))
        ax_pcd.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        ax_pcd.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    ax_pcd.set_xlabel('Y (m)', fontsize=7)
    ax_pcd.set_ylabel('X (m)', fontsize=7)
    ax_pcd.set_title(f'{panel_labels[2]} 3D points (base frame)', fontsize=9, fontweight='bold')
    ax_pcd.set_aspect('equal')
    ax_pcd.tick_params(labelsize=6)

    # --- (d) FPS PCD + grapple prediction — z-threshold split ---
    ax_fps = fig.add_subplot(gs[0, 3])
    fps_pts = pipeline_data["fps_points"]
    mask = np.any(fps_pts != 0.0, axis=1)
    pts = fps_pts[mask] if mask.any() else fps_pts

    # Compute floor threshold from log points
    log_pts_d = pipeline_data.get("log_base_points", np.zeros((0, 3)))
    if len(log_pts_d) > 0:
        valid_log = log_pts_d[np.any(log_pts_d != 0.0, axis=1)]
        z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
    else:
        z_floor_thresh = -999

    if len(pts) > 0:
        floor_mask = pts[:, 2] < z_floor_thresh
        log_mask = ~floor_mask

        # Floor points: faint gray
        if floor_mask.any():
            ax_fps.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                           c='#aaaaaa', s=0.5, alpha=0.25, rasterized=True)
        # Log-height points: viridis
        if log_mask.any():
            sc = ax_fps.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                c=pts[log_mask, 2], cmap='viridis',
                                s=2.0, alpha=0.7, rasterized=True)
            cb = plt.colorbar(sc, ax=ax_fps, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=5)
            cb.set_label('Z (m)', fontsize=6)

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

    # Result annotation
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
        ax_fps.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    ax_fps.set_xlabel('Y (m)', fontsize=7)
    ax_fps.set_ylabel('X (m)', fontsize=7)
    ax_fps.set_title(f'{panel_labels[3]} FPS ({fps_pts.shape[0]} pts) + prediction',
                     fontsize=9, fontweight='bold')
    ax_fps.set_aspect('equal')
    ax_fps.tick_params(labelsize=6)

    # Suptitle
    title = f'Grasp {step_idx + 1}'
    if logs_grasped is not None:
        title += f'  |  target=({x:.2f}, {y:.2f}, {z:.2f})  yaw={np.degrees(yaw):.0f}\u00b0'
        title += f'  |  {"SUCCESS" if success else "MISS"}'
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)

    out_path = os.path.join(viz_dir, f"pipeline_step_{step_idx:04d}.png")
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved pipeline: {out_path}")


# ---------------------------------------------------------------------------
# Stacked raw pipeline figure (N rows x 4 columns)
# ---------------------------------------------------------------------------

def save_raw_stacked_pipeline_viz(grasp_data_list, episode_idx, viz_dir,
                                  depth_range=(1.0, 10.0),
                                  bounds_min=None, bounds_max=None,
                                  suffix=""):
    """Save a vertically stacked pipeline figure with one row per grasp.

    Each row has 4 panels: (a) RGB, (b) raw depth, (c) 3D PCD, (d) FPS PCD + prediction.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_grasps = len(grasp_data_list)
    if n_grasps == 0:
        return

    row_height = 3.2
    fig = plt.figure(figsize=(18, row_height * n_grasps + 0.3))
    gs = gridspec.GridSpec(n_grasps, 4, figure=fig, wspace=0.32, hspace=0.25)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for row, d in enumerate(grasp_data_list):
        x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
        step_idx = d["step_idx"]
        logs_grasped = d.get("logs_grasped")
        alignment = d.get("alignment")
        success = logs_grasped is not None and logs_grasped > 0

        # Row label
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

        # --- (c) 3D PCD (base frame, top-down) — dual-layer ---
        ax_pcd = fig.add_subplot(gs[row, 2])
        base_pts = d["base_points"]
        log_pts = d.get("log_base_points", np.zeros((0, 3)))
        sc_pcd = None

        # Background: all raw points as faint gray
        if len(base_pts) > 0:
            mask = np.any(base_pts != 0.0, axis=1)
            bg = base_pts[mask] if mask.any() else base_pts
            if len(bg) > 0:
                ax_pcd.scatter(bg[:, 1], bg[:, 0], c='#aaaaaa', s=0.3, alpha=0.3, rasterized=True)

        # Foreground: log-only points in viridis
        if len(log_pts) > 0:
            mask = np.any(log_pts != 0.0, axis=1)
            lp = log_pts[mask] if mask.any() else log_pts
            if len(lp) > 0:
                sc_pcd = ax_pcd.scatter(lp[:, 1], lp[:, 0], c=lp[:, 2], cmap='viridis',
                                        s=1.0, alpha=0.7, rasterized=True)

        if bounds_min is not None and bounds_max is not None:
            ax_pcd.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax_pcd.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        if row == 0:
            ax_pcd.set_title(f'{panel_labels[2]} 3D points', fontsize=9, fontweight='bold')
        if sc_pcd is not None:
            cb_pcd = plt.colorbar(sc_pcd, ax=ax_pcd, fraction=0.046, pad=0.04)
            cb_pcd.ax.tick_params(labelsize=5)
            cb_pcd.set_label('Z (m)', fontsize=6)
        ax_pcd.set_aspect('equal')
        ax_pcd.tick_params(labelsize=5)

        # --- (d) FPS PCD + grapple prediction — z-threshold split ---
        ax_fps = fig.add_subplot(gs[row, 3])
        fps_pts = d["fps_points"]
        mask = np.any(fps_pts != 0.0, axis=1)
        pts = fps_pts[mask] if mask.any() else fps_pts

        # Compute floor threshold from log points
        if len(log_pts) > 0:
            valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
            z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
        else:
            z_floor_thresh = -999

        sc_fps = None
        if len(pts) > 0:
            floor_mask = pts[:, 2] < z_floor_thresh
            log_mask = ~floor_mask

            # Floor points: faint gray
            if floor_mask.any():
                ax_fps.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                               c='#aaaaaa', s=0.5, alpha=0.25, rasterized=True)
            # Log-height points: viridis
            if log_mask.any():
                sc_fps = ax_fps.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                        c=pts[log_mask, 2], cmap='viridis',
                                        s=2.0, alpha=0.7, rasterized=True)

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

        # Result annotation
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
            ax_fps.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        if row == 0:
            ax_fps.set_title(f'{panel_labels[3]} FPS + prediction', fontsize=9, fontweight='bold')
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


# ---------------------------------------------------------------------------
# Episode progression grid (3x5: Early / Mid / Late)
# ---------------------------------------------------------------------------

def save_raw_episode_progression(episode_data, episode_idx, viz_dir,
                                 bounds_min=None, bounds_max=None):
    """Save a 3x5 grid showing grasp predictions across an episode.

    Uses FPS points only, so identical for raw vs segmented pipelines.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_total = len(episode_data)
    if n_total == 0:
        return

    n_per_row = 5

    early_indices = list(range(min(n_per_row, n_total)))

    mid_center = n_total // 2
    mid_start = max(0, mid_center - n_per_row // 2)
    mid_indices = list(range(mid_start, min(mid_start + n_per_row, n_total)))

    end_start = max(0, n_total - n_per_row)
    end_indices = list(range(end_start, n_total))

    rows = [
        ("Early", early_indices),
        ("Mid", mid_indices),
        ("Late", end_indices),
    ]

    if bounds_min is not None and bounds_max is not None:
        x_lim = (bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        y_lim = (bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    else:
        all_pts = np.concatenate([d["fps_points"] for d in episode_data], axis=0)
        mask = np.any(all_pts != 0.0, axis=1)
        valid = all_pts[mask] if mask.any() else all_pts
        if len(valid) > 0:
            margin = 0.5
            x_lim = (valid[:, 1].min() - margin, valid[:, 1].max() + margin)
            y_lim = (valid[:, 0].max() + margin, valid[:, 0].min() - margin)
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

            success = d.get("logs_grasped") is not None and d.get("logs_grasped", 0) > 0

            # Compute floor threshold from log points
            log_pts = d.get("log_base_points", np.zeros((0, 3)))
            if len(log_pts) > 0:
                valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
                z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
            else:
                z_floor_thresh = -999

            if len(pts) > 0:
                floor_mask = pts[:, 2] < z_floor_thresh
                log_mask = ~floor_mask

                # Floor points: faint gray
                if floor_mask.any():
                    ax.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                               c='#aaaaaa', s=0.5, alpha=0.25, rasterized=True)
                # Log-height points: viridis
                if log_mask.any():
                    ax.scatter(pts[log_mask, 1], pts[log_mask, 0],
                               c=pts[log_mask, 2], cmap='viridis',
                               s=1.5, alpha=0.7, rasterized=True)

            draw_grapple_footprint(ax, d["y"], d["x"], np.pi / 2 - d["yaw"],
                                   width=1.5, length=0.5,
                                   success=success, alpha=0.3)

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

            result_tag = "HIT" if success else "MISS"
            n_grasped = d.get("logs_grasped", 0) or 0
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
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved progression: {out_path}")


# ---------------------------------------------------------------------------
# Paper-compact pipeline figure (2 rows x 4 cols, \textwidth-wide)
# ---------------------------------------------------------------------------

def save_paper_pipeline_viz(grasp_data_list, episode_idx, viz_dir,
                            depth_range=(1.0, 10.0),
                            bounds_min=None, bounds_max=None,
                            suffix="_paper"):
    """Save a compact pipeline figure for the paper (2 rows x 4 cols).

    Columns: (a) RGB, (b) raw depth, (c) 3D PCD, (d) FPS PCD + prediction.
    Rows: top = dense pile, bottom = sparse endgame.
    Sized to fit IEEE \textwidth (7.16 in) at 150 dpi.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_grasps = len(grasp_data_list)
    if n_grasps == 0:
        return

    row_height = 1.6
    fig = plt.figure(figsize=(7.16, row_height * n_grasps + 0.15))
    gs = gridspec.GridSpec(n_grasps, 4, figure=fig, wspace=0.45, hspace=0.18)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    for row, d in enumerate(grasp_data_list):
        x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
        logs_grasped = d.get("logs_grasped")
        alignment = d.get("alignment")
        success = logs_grasped is not None and logs_grasped > 0

        # --- (a) RGB ---
        ax_rgb = fig.add_subplot(gs[row, 0])
        if d["rgb"] is not None:
            ax_rgb.imshow(d["rgb"])
        else:
            ax_rgb.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax_rgb.transAxes, fontsize=7, color='gray')
        ax_rgb.text(0.02, 0.98, panel_labels[0], transform=ax_rgb.transAxes,
                    fontsize=6, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])

        # --- (b) Raw depth ---
        ax_depth = fig.add_subplot(gs[row, 1])
        depth_img = d["depth"].copy()
        depth_clipped = np.clip(depth_img, depth_range[0], depth_range[1])
        depth_clipped[np.isinf(depth_img)] = np.nan
        im_d = ax_depth.imshow(depth_clipped, cmap='viridis',
                               vmin=depth_range[0], vmax=depth_range[1])
        cb_d = plt.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)
        cb_d.ax.tick_params(labelsize=4)
        ax_depth.text(0.02, 0.98, panel_labels[1], transform=ax_depth.transAxes,
                      fontsize=6, fontweight='bold', va='top',
                      bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        ax_depth.set_xticks([])
        ax_depth.set_yticks([])

        # --- (c) 3D PCD (base frame, top-down) — dual-layer ---
        ax_pcd = fig.add_subplot(gs[row, 2])
        base_pts = d["base_points"]
        log_pts = d.get("log_base_points", np.zeros((0, 3)))
        sc_pcd = None

        if len(base_pts) > 0:
            mask = np.any(base_pts != 0.0, axis=1)
            bg = base_pts[mask] if mask.any() else base_pts
            if len(bg) > 0:
                ax_pcd.scatter(bg[:, 1], bg[:, 0], c='#aaaaaa', s=0.2, alpha=0.3, rasterized=True)

        if len(log_pts) > 0:
            mask = np.any(log_pts != 0.0, axis=1)
            lp = log_pts[mask] if mask.any() else log_pts
            if len(lp) > 0:
                sc_pcd = ax_pcd.scatter(lp[:, 1], lp[:, 0], c=lp[:, 2], cmap='viridis',
                                        s=0.6, alpha=0.7, rasterized=True)

        if bounds_min is not None and bounds_max is not None:
            ax_pcd.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax_pcd.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        ax_pcd.text(0.02, 0.98, panel_labels[2], transform=ax_pcd.transAxes,
                    fontsize=6, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        if sc_pcd is not None:
            cb_pcd = plt.colorbar(sc_pcd, ax=ax_pcd, fraction=0.038, pad=0.04)
            cb_pcd.ax.tick_params(labelsize=4)
        ax_pcd.set_aspect('equal')
        ax_pcd.set_xticklabels([])
        ax_pcd.set_yticklabels([])

        # --- (d) FPS PCD + grapple prediction — z-threshold split ---
        ax_fps = fig.add_subplot(gs[row, 3])
        fps_pts = d["fps_points"]
        mask = np.any(fps_pts != 0.0, axis=1)
        pts = fps_pts[mask] if mask.any() else fps_pts

        if len(log_pts) > 0:
            valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
            z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
        else:
            z_floor_thresh = -999

        sc_fps = None
        if len(pts) > 0:
            floor_mask = pts[:, 2] < z_floor_thresh
            log_mask = ~floor_mask

            if floor_mask.any():
                ax_fps.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                               c='#aaaaaa', s=0.3, alpha=0.25, rasterized=True)
            if log_mask.any():
                sc_fps = ax_fps.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                        c=pts[log_mask, 2], cmap='viridis',
                                        s=1.2, alpha=0.7, rasterized=True)

        draw_grapple_footprint(ax_fps, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                               success=success, alpha=0.25)

        if bounds_min is not None and bounds_max is not None:
            from matplotlib.patches import Rectangle
            bw = bounds_max[1] - bounds_min[1]
            bh = bounds_max[0] - bounds_min[0]
            ax_fps.add_patch(Rectangle(
                (bounds_min[1], bounds_max[0]), bw, -bh,
                linewidth=0.5, edgecolor='#e67e22', facecolor='none',
                linestyle='--', alpha=0.6, zorder=3))

        # Result annotation
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
                    fontsize=5, verticalalignment='top', color=result_color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

        if bounds_min is not None and bounds_max is not None:
            ax_fps.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax_fps.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        ax_fps.text(0.72, 0.98, panel_labels[3], transform=ax_fps.transAxes,
                    fontsize=6, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        if sc_fps is not None:
            cb_fps = plt.colorbar(sc_fps, ax=ax_fps, fraction=0.038, pad=0.04)
            cb_fps.ax.tick_params(labelsize=4)
        ax_fps.set_aspect('equal')
        ax_fps.set_xticklabels([])
        ax_fps.set_yticklabels([])

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_pipeline{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved paper pipeline: {out_path}")


def pick_early_late_grasps(grasp_data_list, rng=None):
    """Pick one random successful early grasp and one random successful late grasp.

    Splits the episode into first-half (early/dense) and second-half (late/sparse).
    Prefers successful grasps (logs_grasped > 0); falls back to any grasp in each half.
    Returns a list of 1 or 2 grasp dicts suitable for paper_rep.
    """
    import random
    if rng is None:
        rng = random.Random()

    n = len(grasp_data_list)
    if n == 0:
        return []

    mid = max(n // 2, 1)
    early = grasp_data_list[:mid]
    late = grasp_data_list[mid:] if n > 1 else []

    def _pick(candidates):
        hits = [g for g in candidates if (g.get("logs_grasped") or 0) > 0]
        pool = hits if hits else candidates
        return rng.choice(pool)

    result = [_pick(early)]
    if late:
        result.append(_pick(late))
    return result


# ---------------------------------------------------------------------------
# Quadrant variants (2x2, \columnwidth = 3.5 in)
# ---------------------------------------------------------------------------

def _quadrant_result_annotation(ax, d):
    """Add HIT/MISS annotation to an axis."""
    logs_grasped = d.get("logs_grasped")
    alignment = d.get("alignment")
    success = logs_grasped is not None and logs_grasped > 0
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
    ax.text(0.02, 0.98, result_str, transform=ax.transAxes,
            fontsize=5, verticalalignment='top', color=result_color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))



def save_paper_pipeline_quadrant_single(grasp_data_list, episode_idx, viz_dir,
                                         depth_range=(1.0, 10.0),
                                         bounds_min=None, bounds_max=None,
                                         suffix="_quad_single"):
    """Variant 1: Single grasp, all 4 pipeline stages in a 2x2 quadrant.

    (a) RGB      | (b) Depth
    (c) 3D PCD   | (d) FPS + prediction
    Sized to fit IEEE \\columnwidth (3.5 in).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if len(grasp_data_list) == 0:
        return

    # Pick the first (dense pile) grasp as representative
    d = grasp_data_list[0]
    x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
    logs_grasped = d.get("logs_grasped")
    success = logs_grasped is not None and logs_grasped > 0

    fig = plt.figure(figsize=(3.5, 3.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.30)

    # --- (a) RGB ---
    ax = fig.add_subplot(gs[0, 0])
    if d["rgb"] is not None:
        ax.imshow(d["rgb"])
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                transform=ax.transAxes, fontsize=6, color='gray')
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])

    # --- (b) Depth ---
    ax = fig.add_subplot(gs[0, 1])
    depth_img = d["depth"].copy()
    depth_clipped = np.clip(depth_img, depth_range[0], depth_range[1])
    depth_clipped[np.isinf(depth_img)] = np.nan
    im_d = ax.imshow(depth_clipped, cmap='viridis',
                     vmin=depth_range[0], vmax=depth_range[1])
    cb = plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=3.5)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
    ax.set_xticks([]); ax.set_yticks([])

    # --- (c) 3D PCD ---
    ax = fig.add_subplot(gs[1, 0])
    base_pts = d["base_points"]
    log_pts = d.get("log_base_points", np.zeros((0, 3)))
    sc_pcd = None
    if len(base_pts) > 0:
        mask = np.any(base_pts != 0.0, axis=1)
        bg = base_pts[mask] if mask.any() else base_pts
        if len(bg) > 0:
            ax.scatter(bg[:, 1], bg[:, 0], c='#aaaaaa', s=0.15, alpha=0.3, rasterized=True)
    if len(log_pts) > 0:
        mask = np.any(log_pts != 0.0, axis=1)
        lp = log_pts[mask] if mask.any() else log_pts
        if len(lp) > 0:
            sc_pcd = ax.scatter(lp[:, 1], lp[:, 0], c=lp[:, 2], cmap='viridis',
                                s=0.4, alpha=0.7, rasterized=True)
    if bounds_min is not None and bounds_max is not None:
        ax.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        ax.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    if sc_pcd is not None:
        cb = plt.colorbar(sc_pcd, ax=ax, fraction=0.038, pad=0.04)
        cb.ax.tick_params(labelsize=3.5)
    ax.set_aspect('equal')
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
    ax.set_xticklabels([]); ax.set_yticklabels([])

    # --- (d) FPS + prediction ---
    ax = fig.add_subplot(gs[1, 1])
    fps_pts = d["fps_points"]
    mask = np.any(fps_pts != 0.0, axis=1)
    pts = fps_pts[mask] if mask.any() else fps_pts
    if len(log_pts) > 0:
        valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
        z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
    else:
        z_floor_thresh = -999
    sc_fps = None
    if len(pts) > 0:
        floor_mask = pts[:, 2] < z_floor_thresh
        log_mask = ~floor_mask
        if floor_mask.any():
            ax.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                       c='#aaaaaa', s=0.2, alpha=0.25, rasterized=True)
        if log_mask.any():
            sc_fps = ax.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                c=pts[log_mask, 2], cmap='viridis',
                                s=0.8, alpha=0.7, rasterized=True)
    draw_grapple_footprint(ax, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                           success=success, alpha=0.25)
    if bounds_min is not None and bounds_max is not None:
        ax.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        ax.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    if sc_fps is not None:
        cb = plt.colorbar(sc_fps, ax=ax, fraction=0.038, pad=0.04)
        cb.ax.tick_params(labelsize=3.5)
    ax.set_aspect('equal')
    ax.text(0.72, 0.98, '(d)', transform=ax.transAxes, fontsize=6,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    _quadrant_result_annotation(ax, d)

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_pipeline{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved quadrant (single): {out_path}")


def save_paper_pipeline_quadrant_rgb_fps(grasp_data_list, episode_idx, viz_dir,
                                          depth_range=(1.0, 10.0),
                                          bounds_min=None, bounds_max=None,
                                          suffix="_quad_rgb_fps"):
    """Variant 2: Two grasps (early + late), RGB + FPS+prediction per row.

    Row 0 (dense):  (a) RGB  | (b) FPS + prediction
    Row 1 (sparse): (c) RGB  | (d) FPS + prediction
    Sized to fit IEEE \\columnwidth (3.5 in).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if len(grasp_data_list) == 0:
        return

    # Use first and last grasp
    rows = [grasp_data_list[0]]
    if len(grasp_data_list) > 1:
        rows.append(grasp_data_list[-1])
    n_rows = len(rows)

    fig = plt.figure(figsize=(3.5, 1.6 * n_rows + 0.15))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, wspace=0.25, hspace=0.30)
    labels = [('(a)', '(b)'), ('(c)', '(d)')]

    for row, d in enumerate(rows):
        x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
        logs_grasped = d.get("logs_grasped")
        success = logs_grasped is not None and logs_grasped > 0
        log_pts = d.get("log_base_points", np.zeros((0, 3)))
        lbl = labels[row]

        # --- RGB ---
        ax = fig.add_subplot(gs[row, 0])
        if d["rgb"] is not None:
            ax.imshow(d["rgb"])
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=6, color='gray')
        ax.text(0.02, 0.98, lbl[0], transform=ax.transAxes, fontsize=6,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])

        # --- FPS + prediction ---
        ax = fig.add_subplot(gs[row, 1])
        fps_pts = d["fps_points"]
        mask = np.any(fps_pts != 0.0, axis=1)
        pts = fps_pts[mask] if mask.any() else fps_pts
        if len(log_pts) > 0:
            valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
            z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
        else:
            z_floor_thresh = -999
        sc_fps = None
        if len(pts) > 0:
            floor_mask = pts[:, 2] < z_floor_thresh
            log_mask = ~floor_mask
            if floor_mask.any():
                ax.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                           c='#aaaaaa', s=0.2, alpha=0.25, rasterized=True)
            if log_mask.any():
                sc_fps = ax.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                    c=pts[log_mask, 2], cmap='viridis',
                                    s=0.8, alpha=0.7, rasterized=True)
        draw_grapple_footprint(ax, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                               success=success, alpha=0.25)
        if bounds_min is not None and bounds_max is not None:
            ax.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        ax.text(0.72, 0.98, lbl[1], transform=ax.transAxes, fontsize=6,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        if sc_fps is not None:
            cb = plt.colorbar(sc_fps, ax=ax, fraction=0.038, pad=0.04)
            cb.ax.tick_params(labelsize=3.5)
        ax.set_aspect('equal')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        _quadrant_result_annotation(ax, d)

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_pipeline{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved quadrant (RGB+FPS): {out_path}")


def save_paper_pipeline_quadrant_depth_fps(grasp_data_list, episode_idx, viz_dir,
                                            depth_range=(1.0, 10.0),
                                            bounds_min=None, bounds_max=None,
                                            suffix="_quad_depth_fps"):
    """Variant 3: Two grasps (early + late), depth + FPS+prediction per row.

    Row 0 (dense):  (a) Depth  | (b) FPS + prediction
    Row 1 (sparse): (c) Depth  | (d) FPS + prediction
    Sized to fit IEEE \\columnwidth (3.5 in).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if len(grasp_data_list) == 0:
        return

    rows = [grasp_data_list[0]]
    if len(grasp_data_list) > 1:
        rows.append(grasp_data_list[-1])
    n_rows = len(rows)

    fig = plt.figure(figsize=(3.5, 1.6 * n_rows + 0.15))
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, wspace=0.30, hspace=0.30)
    labels = [('(a)', '(b)'), ('(c)', '(d)')]

    for row, d in enumerate(rows):
        x, y, z, yaw = d["x"], d["y"], d["z"], d["yaw"]
        logs_grasped = d.get("logs_grasped")
        success = logs_grasped is not None and logs_grasped > 0
        log_pts = d.get("log_base_points", np.zeros((0, 3)))
        lbl = labels[row]

        # --- Depth ---
        ax = fig.add_subplot(gs[row, 0])
        depth_img = d["depth"].copy()
        depth_clipped = np.clip(depth_img, depth_range[0], depth_range[1])
        depth_clipped[np.isinf(depth_img)] = np.nan
        im_d = ax.imshow(depth_clipped, cmap='viridis',
                         vmin=depth_range[0], vmax=depth_range[1])
        cb = plt.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=3.5)
        ax.text(0.02, 0.98, lbl[0], transform=ax.transAxes, fontsize=6,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])

        # --- FPS + prediction ---
        ax = fig.add_subplot(gs[row, 1])
        fps_pts = d["fps_points"]
        mask = np.any(fps_pts != 0.0, axis=1)
        pts = fps_pts[mask] if mask.any() else fps_pts
        if len(log_pts) > 0:
            valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
            z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
        else:
            z_floor_thresh = -999
        sc_fps = None
        if len(pts) > 0:
            floor_mask = pts[:, 2] < z_floor_thresh
            log_mask = ~floor_mask
            if floor_mask.any():
                ax.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                           c='#aaaaaa', s=0.2, alpha=0.25, rasterized=True)
            if log_mask.any():
                sc_fps = ax.scatter(pts[log_mask, 1], pts[log_mask, 0],
                                    c=pts[log_mask, 2], cmap='viridis',
                                    s=0.8, alpha=0.7, rasterized=True)
        draw_grapple_footprint(ax, y, x, np.pi / 2 - yaw, width=1.5, length=0.5,
                               success=success, alpha=0.25)
        if bounds_min is not None and bounds_max is not None:
            ax.set_xlim(bounds_min[1] - 0.5, bounds_max[1] + 0.5)
            ax.set_ylim(bounds_max[0] + 0.5, bounds_min[0] - 0.5)
        ax.text(0.72, 0.98, lbl[1], transform=ax.transAxes, fontsize=6,
                fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.8))
        if sc_fps is not None:
            cb = plt.colorbar(sc_fps, ax=ax, fraction=0.038, pad=0.04)
            cb.ax.tick_params(labelsize=3.5)
        ax.set_aspect('equal')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        _quadrant_result_annotation(ax, d)

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_pipeline{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved quadrant (Depth+FPS): {out_path}")


# ---------------------------------------------------------------------------
# Paper-compact progression figure (2 rows x 5 cols, \textwidth-wide)
# ---------------------------------------------------------------------------

def save_paper_progression_viz(episode_data, episode_idx, viz_dir,
                               bounds_min=None, bounds_max=None,
                               n_per_row=5, suffix="_paper"):
    """Save a compact 2x5 progression figure for the paper.

    Rows: early grasps (dense pile) and late grasps (sparse endgame).
    Sized to fit IEEE \textwidth (7.16 in) at 150 dpi.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_total = len(episode_data)
    if n_total == 0:
        return

    early_indices = list(range(min(n_per_row, n_total)))

    end_start = max(0, n_total - n_per_row)
    end_indices = list(range(end_start, n_total))

    rows = [
        ("Early", early_indices),
        ("Late", end_indices),
    ]

    if bounds_min is not None and bounds_max is not None:
        x_lim = (bounds_min[1] - 0.5, bounds_max[1] + 0.5)
        y_lim = (bounds_max[0] + 0.5, bounds_min[0] - 0.5)
    else:
        all_pts = np.concatenate([d["fps_points"] for d in episode_data], axis=0)
        mask = np.any(all_pts != 0.0, axis=1)
        valid = all_pts[mask] if mask.any() else all_pts
        if len(valid) > 0:
            margin = 0.5
            x_lim = (valid[:, 1].min() - margin, valid[:, 1].max() + margin)
            y_lim = (valid[:, 0].max() + margin, valid[:, 0].min() - margin)
        else:
            x_lim = (-5, 5)
            y_lim = (5, -5)

    n_rows = len(rows)
    max_cols = max(len(idx) for _, idx in rows)
    col_width = 7.16 / max_cols
    fig, axes = plt.subplots(n_rows, max_cols,
                             figsize=(7.16, col_width * 0.85 * n_rows))
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

            success = d.get("logs_grasped") is not None and d.get("logs_grasped", 0) > 0

            log_pts = d.get("log_base_points", np.zeros((0, 3)))
            if len(log_pts) > 0:
                valid_log = log_pts[np.any(log_pts != 0.0, axis=1)]
                z_floor_thresh = valid_log[:, 2].min() - 0.05 if len(valid_log) > 0 else -999
            else:
                z_floor_thresh = -999

            if len(pts) > 0:
                floor_mask = pts[:, 2] < z_floor_thresh
                log_mask = ~floor_mask

                if floor_mask.any():
                    ax.scatter(pts[floor_mask, 1], pts[floor_mask, 0],
                               c='#aaaaaa', s=0.3, alpha=0.25, rasterized=True)
                if log_mask.any():
                    ax.scatter(pts[log_mask, 1], pts[log_mask, 0],
                               c=pts[log_mask, 2], cmap='viridis',
                               s=1.0, alpha=0.7, rasterized=True)

            draw_grapple_footprint(ax, d["y"], d["x"], np.pi / 2 - d["yaw"],
                                   width=1.5, length=0.5,
                                   success=success, alpha=0.3)

            if bounds_min is not None and bounds_max is not None:
                from matplotlib.patches import Rectangle
                bw = bounds_max[1] - bounds_min[1]
                bh = bounds_max[0] - bounds_min[0]
                ax.add_patch(Rectangle(
                    (bounds_min[1], bounds_max[0]), bw, -bh,
                    linewidth=0.4, edgecolor='#e67e22', facecolor='none',
                    linestyle='--', alpha=0.4, zorder=3))

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.tick_params(labelsize=4)

            result_tag = "HIT" if success else "MISS"
            n_grasped = d.get("logs_grasped", 0) or 0
            col_title = f"grasp {d['step_idx'] + 1}"
            if d.get("logs_remaining") is not None:
                col_title += f" | {d['logs_remaining']} left"
            col_title += f" | {result_tag} ({n_grasped})"
            ax.set_title(col_title, fontsize=5.5, pad=2)

            if col_j == 0:
                first_g = episode_data[indices[0]]["step_idx"] + 1
                last_g = episode_data[indices[-1]]["step_idx"] + 1
                grasp_range = f"{first_g}\u2013{last_g}" if len(indices) > 1 else str(first_g)
                ax.set_ylabel(f'{row_label}\n(grasps {grasp_range})',
                              fontsize=6, fontweight='bold')
            else:
                ax.set_yticklabels([])

            if row_i < n_rows - 1:
                ax.set_xticklabels([])

    fig.tight_layout()

    out_path = os.path.join(viz_dir, f"episode_{episode_idx:03d}_progression{suffix}.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[PaperViz] Saved paper progression: {out_path}")
