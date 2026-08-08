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
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to BC policy checkpoint (required for --policy_type bc)")
parser.add_argument("--policy_type", type=str, default="bc", choices=["bc", "heuristic", "scoring", "expert"],
                    help="'bc' = neural policy from --checkpoint. 'heuristic' = the DEPLOYED "
                         "real-crane HeuristicPolicy (policy_loader.py: supported-highest-point, "
                         "surface - dig, gated-PCA yaw) ported into sim eval. This is the bridge "
                         "row between the sim and real tables: the same baseline controller "
                         "measured in both domains.")
parser.add_argument("--heuristic_dig", type=float, default=0.25,
                    help="Dig below the perceived surface for --policy_type heuristic "
                         "(0.25 = current real-crane setting)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--cam_width", type=int, default=1280, help="camera render width (match training; VRAM)")
parser.add_argument("--cam_height", type=int, default=720, help="camera render height (match training; VRAM)")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON")
parser.add_argument("--save_decisions", action="store_true",
                    help="Save one record per grasp cycle (policy input cloud, raw action, decoded "
                         "target, outcome) to decisions_<timestamp>.npz next to the metrics JSON. "
                         "With a fixed --seed, records are paired across policies (same piles) for "
                         "state-vs-decision analysis and offline counterfactual replay.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics")
parser.add_argument("--visualize", action="store_true", help="Save per-step visualization PNGs")
parser.add_argument("--viz_dir", type=str, default=None, help="Directory for viz PNGs (default: checkpoint dir / viz)")
parser.add_argument("--paper_viz", action="store_true", help="Save paper-quality pipeline and progression figures")
parser.add_argument("--raw_pcd", action="store_true", help="Use raw (unmasked) point cloud instead of segmented log-only points")
parser.add_argument("--num_points", type=int, default=0,
                    help="override the observation point count (0 = policy default: 1024 for "
                         "heuristic/expert, else the checkpoint's). Used for the DENSITY-CONTROL "
                         "row: the heuristic is parameter-free in point density, so comparing it "
                         "at 1024 against a 2048-point scoring policy would be an unearned "
                         "handicap; this runs it at matched density.")
parser.add_argument("--expert_dig", type=float, default=0.25,
                    help="--policy_type expert: dig convention for the PRIVILEGED expert row "
                         "(targets top-log CENTRE = surface-0.056, relabelled to surface-dig, "
                         "bed-floor clamped) - matches the collection-path expert_dig.")
parser.add_argument("--support_gate", type=float, default=0.0,
                    help="drop cloud points with 0.5m xy support < frac*max BEFORE the policy "
                         "(policy-agnostic outlier gate; parity layer for the noise sweep)")
parser.add_argument("--crop_margin", type=float, default=0.0,
                    help="Widen the OBSERVATION crop by this many metres in x/y and BELOW in z (action box "
                         "unchanged), so the bed plane / rails / pole corners are visible. Must match the "
                         "value the policy was TRAINED with.")
parser.add_argument("--crop_to_bounds", action="store_true",
                    help="Crop the point cloud to the action-bounds box before FPS (removes grapple/trailer/"
                         "background). MUST be matched at deployment for sim2real.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic evaluation")
parser.add_argument("--obs_noise", type=float, default=0.0, help="Gaussian noise σ added to PCD coordinates (meters)")
parser.add_argument("--zed_noise", action="store_true",
                    help="ZED-realistic sensor degradation: axial noise sigma = coeff*depth^2 applied to the "
                         "DEPTH IMAGE before unprojection (so it lies along the camera ray, like the real "
                         "stereo error) plus random pixel dropout. Unlike --obs_noise (isotropic on the "
                         "cloud) this is the physical process, which is what makes it the right x-axis for "
                         "an observation-degradation sweep.")
parser.add_argument("--zed_axial_coeff", type=float, default=0.0014,
                    help="axial sigma = coeff * depth^2 (m). 0.0014 = the characterised real ZED X; "
                         "multiples of it degrade observation quality along the physically correct axis.")
parser.add_argument("--zed_dropout", type=float, default=0.06,
                    help="fraction of depth pixels randomly dropped (holes)")
parser.add_argument("--action_noise", type=float, default=0.0, help="Gaussian noise σ added to policy action output")
parser.add_argument("--record_video", action="store_true", help="Record video frames during episode (requires --num_envs 1)")
parser.add_argument("--video_out", type=str, default="bc_policy_view.mp4", help="Output path for policy-view video")
parser.add_argument("--overview_out", type=str, default="bc_overview.mp4", help="Output path for overview video")
parser.add_argument("--sideview_out", type=str, default="bc_sideview.mp4", help="Output path for sideview video")
parser.add_argument("--video_fps", type=int, default=30, help="Output video frame rate")
parser.add_argument("--gaze", action="store_true",
                    help="Load the gaze env (basemast cam + PH_GAZE phase) instead of the full env. "
                         "(store_true to avoid argparse abbreviation clashing with the env's --env_spacing)")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

# Import the environment (same class names; gaze variant mounts the cam on the basemast)
if args_cli.gaze:
    from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull
else:
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
        # raw / sim2real: straight optical -> base (no world pivot), mirrors the real tf chain.
        # High cap so the rack survives the crop (basemast scene is wide, rack ~8%); FPS-to-num_points
        # after the crop is the real downsampler. Must match train_bc_pointcloud.py.
        pc_base = env.get_pointcloud_base(env_idx, max_points=50000, depth_range=depth_range)
    else:
        # segmented log-only (sim-only): world cloud, then world -> base.
        pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
        if pc_world.shape[0] == 0:
            return torch.zeros((num_points, 3), device=env.device)
        base_pos_w = env.crane.data.root_pos_w[env_idx]
        base_quat_w = env.crane.data.root_quat_w[env_idx]
        w, x, y, z = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])
        pc_base = (pc_world - base_pos_w) @ R

    if pc_base.shape[0] == 0:
        return torch.zeros((num_points, 3), device=env.device)

    # Optionally crop to the action-bounds box (removes grapple/trailer/background; sim2real consistency).
    # The SAME crop must be applied at deployment (pointcloud_pipeline.py) for a matching input.
    # --crop_margin widens the OBSERVATION crop only (bed plane / rails / pole corners enter the
    # cloud); the action box is untouched. Must match the value used at collection AND deployment.
    if getattr(args_cli, "crop_to_bounds", False) and getattr(env, "_action_bounds_min", None) is not None:
        bmin = env._action_bounds_min[env_idx]
        bmax = env._action_bounds_max[env_idx]
        g = float(getattr(args_cli, "crop_margin", 0.0))
        m = ((pc_base[:, 0] >= bmin[0] - g) & (pc_base[:, 0] <= bmax[0] + g) &
             (pc_base[:, 1] >= bmin[1] - g) & (pc_base[:, 1] <= bmax[1] + g) &
             (pc_base[:, 2] >= bmin[2] - g) & (pc_base[:, 2] <= bmax[2]))
        pc_base = pc_base[m]
        if pc_base.shape[0] == 0:
            return torch.zeros((num_points, 3), device=env.device)

    pc_sampled = farthest_point_sampling(pc_base, num_points)

    # --support_gate: policy-agnostic cloud preprocessing - zero out points whose 0.5 m xy
    # support is below frac * (cloud max). Parity layer for the sweep: the DEPLOYED heuristic
    # already has ROR + a supported-candidate walk, the sim path has neither, so without this
    # the sim heuristic's high-noise collapse conflates outlier-vulnerability with policy
    # quality. Verified on the 26 real trial clouds: a support gate moves ZERO deployed-
    # heuristic targets (structure is well-supported), so this is processing, not learning.
    g = float(getattr(args_cli, "support_gate", 0.0))
    if g > 0:
        vmask = pc_sampled.abs().sum(1) > 1e-6
        if int(vmask.sum()) > 2:
            q = pc_sampled[vmask]
            cnt = (torch.cdist(q[:, :2], q[:, :2]) < 0.5).sum(1).float()
            drop = cnt < g * cnt.max()
            if bool(drop.any()):
                idx = torch.nonzero(vmask, as_tuple=False).squeeze(1)
                pc_sampled[idx[drop]] = 0.0

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


def _write_video(frames: list, out_path: str, fps: int, label: str = "video"):
    """Write a list of (H, W, 3) uint8 numpy frames to an MP4 using imageio."""
    if not frames:
        print(f"[Video] No {label} frames captured — skipping.")
        return
    try:
        import imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"[Video] Saved {len(frames)} {label} frames → {out_path}")
    except Exception as e:
        print(f"[Video] Failed to write {label} video: {e}")


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


def save_gaze_rgbd(env, env_idx, pc, out_dir, pick_idx):
    """ONE-SHOT: save the FULL raw SIM point cloud (base frame, no crop, no FPS) so it can be
    compared interactively against the real ZED cloud (see calibration/compare_pcd.py)."""
    if getattr(save_gaze_rgbd, "_done", False):
        return
    # raw SIM cloud (full, no FPS / no downsample), world -> base frame
    pc_world = env.get_pointcloud_world(env_idx, max_points=400000, depth_range=(0.3, 20.0))
    base_pos = env.crane.data.root_pos_w[env_idx]
    bq = env.crane.data.root_quat_w[env_idx]
    pcw = pc_world - base_pos
    w, x, y, z = bq[0], bq[1], bq[2], bq[3]
    Rm = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])
    sim = (pcw @ Rm).cpu().numpy().astype(np.float32)
    saved = None
    for outp in ("/workspace/crane_testbed/calibration/out/sim_full_pcd.npy",
                 os.path.join(out_dir, "sim_full_pcd.npy")):
        try:
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            np.save(outp, sim); saved = outp; break
        except Exception:
            pass
    # SIM camera pose in the BASE frame — compare directly with the real ZED:
    #   real ZED at gaze: pos (-0.08,-0.15,1.17)  look (-0.79,0.55,-0.28)
    try:
        cpw, cqw = env._gaze_camera_pose_w(env_idx)  # true pose from live mast (sensor pose is stale)
        cam_b = ((cpw - base_pos) @ Rm).cpu().numpy()
        w2, x2, y2, z2 = [float(v) for v in cqw]
        # optical forward = +Z column of the camera rotation, world -> base
        fwd_w = torch.tensor([2*(x2*z2 + w2*y2), 2*(y2*z2 - w2*x2), 1 - 2*(x2*x2 + y2*y2)],
                             device=base_pos.device, dtype=base_pos.dtype)
        fwd_b = (fwd_w @ Rm).cpu().numpy()
        slew_now = float(env.crane.data.joint_pos[env_idx, env._ctrl_joint_idx[0]].item())
        print(f"[gaze] SIM camera in BASE: pos ({cam_b[0]:.2f},{cam_b[1]:.2f},{cam_b[2]:.2f})  "
              f"look ({fwd_b[0]:.2f},{fwd_b[1]:.2f},{fwd_b[2]:.2f})  [optical +Z, quat_w_ros]")
        print(f"       REAL ZED in BASE : pos (-0.08,-0.15,1.17)  look (-0.79,0.55,-0.28)")
        print(f"       slew joint at capture = {slew_now:.4f} rad (gaze target = 0.9913)")
    except Exception as e:
        print(f"[gaze] camera-pose print failed: {e}")
    save_gaze_rgbd._done = True
    print(f"[gaze] SAVED full sim PCD ({sim.shape[0]} pts, base frame) -> {saved}\n"
          f"       Now run (in the calibration env):  python3 calibration/compare_pcd.py")


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
    if args_cli.policy_type == "expert":
        # PRIVILEGED EXPERT through the SAME eval harness as every learned policy, so its row
        # carries metrics_version 2 and the identical protocol. Reads ground-truth log poses
        # (env._target_top_log_center_b) and re-encodes with the collection path's exact
        # arctanh/cos-sin scheme - the previous expert rows came from the COLLECTION script,
        # which never had the corrected clearing/c95 metrics, so they were not comparable.
        _LOG_RADIUS = 0.056
        num_points = args_cli.num_points if args_cli.num_points > 0 else 1024
        metadata, action_dim = {}, 5

        def policy(obs):
            out = torch.zeros((obs.shape[0], 5), device=obs.device)
            for b in range(obs.shape[0]):
                log_pos_b, sel_id, log_quat_w = env._target_top_log_center_b(b)
                if sel_id == -1:
                    continue
                env._target_log_pos_b[b] = log_pos_b
                env._current_target_log_id[b] = sel_id
                env._target_log_quat_w[b] = log_quat_w
                x, y, z = (float(log_pos_b[0]), float(log_pos_b[1]), float(log_pos_b[2]))
                if args_cli.expert_dig > 0:
                    z = z + _LOG_RADIUS - args_cli.expert_dig
                    z = max(z, float(env._action_bounds_min[b][2])
                            + float(getattr(env.cfg, "platform_bed_margin", 0.0)))
                yaw = float(env._get_target_grapple_yaw_b(b))
                if not env._action_bounds_valid[b]:
                    env._compute_action_space_bounds()
                mn, mx = env._action_bounds_min[b], env._action_bounds_max[b]

                def _enc(v, lo, hi):
                    n = np.clip(2.0 * (v - float(lo)) / (float(hi) - float(lo)) - 1.0,
                                -0.999, 0.999)
                    return float(np.arctanh(n))

                out[b, 0] = _enc(x, mn[0], mx[0])
                out[b, 1] = _enc(y, mn[1], mx[1])
                out[b, 2] = _enc(z, mn[2], mx[2])
                out[b, 3] = float(np.cos(2.0 * yaw))
                out[b, 4] = float(np.sin(2.0 * yaw))
            return out

        if args_cli.checkpoint is None:
            args_cli.checkpoint = args_cli.output_dir or "logs/expert_sim_eval"
            os.makedirs(args_cli.checkpoint, exist_ok=True)
        print(f"[Play] PRIVILEGED EXPERT (ground-truth poses), dig={args_cli.expert_dig}")
    elif args_cli.policy_type == "scoring":
        # Per-point argmax head (scoring_head.py). Like the heuristic it returns METRES, so it goes
        # through the same arctanh/cos-sin re-encoding and the env decode reproduces it exactly -
        # NOT through load_bc_policy, whose checkpoints are action-space regressors.
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "fpi_crane_ros2", "fpi_crane_rl", "fpi_crane_rl"))
        from policy_loader import ScoringHeadPolicy
        assert args_cli.checkpoint, "--checkpoint is required for --policy_type scoring"
        _bmin = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
        _bmax = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
        _sc = ScoringHeadPolicy(args_cli.checkpoint, _bmin, _bmax, cossin=True,
                                device=args_cli.device)
        num_points, metadata, action_dim = _sc.num_points, {}, 5

        def _enc(v, lo, hi):
            n = np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -0.999, 0.999)
            return float(np.arctanh(n))

        def policy(obs):
            out = torch.zeros((obs.shape[0], 5), device=obs.device)
            for b in range(obs.shape[0]):
                x, y, z, yaw = _sc.get_target(obs[b])
                out[b, 0] = _enc(x, _bmin[0], _bmax[0])
                out[b, 1] = _enc(y, _bmin[1], _bmax[1])
                out[b, 2] = _enc(z, _bmin[2], _bmax[2])
                out[b, 3] = float(np.cos(2.0 * yaw))
                out[b, 4] = float(np.sin(2.0 * yaw))
            return out

        print(f"[Play] SCORING head, num_points={num_points}")
    elif args_cli.policy_type == "heuristic":
        # The deployed real-crane baseline, verbatim (fpi policy_loader.HeuristicPolicy), fed the
        # SAME cropped training-coords cloud the nets get, its metre-space target re-encoded with
        # the expert's arctanh/cos-sin scheme so the env decode reproduces it exactly.
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "fpi_crane_ros2", "fpi_crane_rl", "fpi_crane_rl"))
        from policy_loader import HeuristicPolicy
        _bmin = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
        _bmax = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
        _heur = HeuristicPolicy(_bmin, _bmax, cossin=True, dig=args_cli.heuristic_dig)
        num_points = args_cli.num_points if args_cli.num_points > 0 else 1024
        metadata, action_dim = {}, 5

        def _enc(v, lo, hi):
            n = np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -0.999, 0.999)
            return float(np.arctanh(n))

        def policy(obs):
            out = torch.zeros((obs.shape[0], 5), device=obs.device)
            for b in range(obs.shape[0]):
                x, y, z, yaw = _heur.get_target(obs[b])
                out[b, 0] = _enc(x, _bmin[0], _bmax[0])
                out[b, 1] = _enc(y, _bmin[1], _bmax[1])
                out[b, 2] = _enc(z, _bmin[2], _bmax[2])
                out[b, 3] = float(np.cos(2.0 * yaw))
                out[b, 4] = float(np.sin(2.0 * yaw))
            return out

        if args_cli.checkpoint is None:
            args_cli.checkpoint = args_cli.output_dir or "logs/heuristic_sim_eval"
            os.makedirs(args_cli.checkpoint, exist_ok=True)
        print(f"[Play] HEURISTIC baseline (deployed real-crane rule), dig={args_cli.heuristic_dig}")
    else:
        assert args_cli.checkpoint, "--checkpoint is required for --policy_type bc"
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
    cfg.camera_cfg.width = args_cli.cam_width      # match training render res + VRAM safety
    cfg.camera_cfg.height = args_cli.cam_height
    if args_cli.paper_viz:
        cfg.camera_cfg.data_types = ["rgb", "depth", "semantic_segmentation"]
    else:
        cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.enable_domain_randomization = args_cli.domain_randomization
    # Sensor degradation is applied at RENDER time by the env (_depth_for_cloud), i.e. along the
    # camera ray before unprojection. Set explicitly here because this script owns its own parser.
    cfg.zed_noise = bool(args_cli.zed_noise)
    cfg.zed_axial_coeff = float(args_cli.zed_axial_coeff)
    cfg.zed_dropout = float(args_cli.zed_dropout)
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed
    if args_cli.record_video:
        assert args_cli.num_envs == 1, "--record_video requires --num_envs 1"
        cfg.record_video = True
        cfg.camera_cfg.data_types = ["rgb", "depth", "semantic_segmentation"]
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _media_dir = "/workspace/crane_testbed/media"
        os.makedirs(_media_dir, exist_ok=True)
        if args_cli.video_out == "bc_policy_view.mp4":
            args_cli.video_out = f"{_media_dir}/bc_policy_view_{_ts}.mp4"
        if args_cli.overview_out == "bc_overview.mp4":
            args_cli.overview_out = f"{_media_dir}/bc_overview_{_ts}.mp4"
        if args_cli.sideview_out == "bc_sideview.mp4":
            args_cli.sideview_out = f"{_media_dir}/bc_sideview_{_ts}.mp4"
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

    # Open streaming video writers
    if args_cli.record_video:
        import imageio
        env._video_writer = imageio.get_writer(
            args_cli.video_out, fps=args_cli.video_fps, codec="libx264",
            quality=8, macro_block_size=1)
        env._overview_writer = imageio.get_writer(
            args_cli.overview_out, fps=args_cli.video_fps, codec="libx264",
            quality=8, macro_block_size=1)
        env._sideview_writer = imageio.get_writer(
            args_cli.sideview_out, fps=args_cli.video_fps, codec="libx264",
            quality=8, macro_block_size=1)
        print(f"[Video] Streaming writers opened -> {args_cli.video_out} / {args_cli.overview_out} / {args_cli.sideview_out}")

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
    episode_true_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    # GAZE input validation: save basemast RGB + depth (the policy inputs) at every gaze step.
    gaze_viz_on = bool(getattr(args_cli, "gaze", False)) and hasattr(env, "PH_GAZE")
    if gaze_viz_on:
        gaze_viz_dir = os.path.join(args_cli.viz_dir or os.path.dirname(args_cli.checkpoint), "gaze_input")
        os.makedirs(gaze_viz_dir, exist_ok=True)
        gaze_step_idx = 0
        print(f"[Gaze] Saving basemast RGB+depth at every gaze step to: {gaze_viz_dir}")
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_successful_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_failed_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_alignment_sum = torch.zeros(env.num_envs, device=env.device)
    ep_stability_sum = torch.zeros(env.num_envs, device=env.device)
    ep_cycle_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_clearing_curves = [[] for _ in range(env.num_envs)]  # per-env list of clearing % at each cycle
    ep_clearing_curves_cum = [[] for _ in range(env.num_envs)]  # legacy cumulative-grasp curve
    # Per-grasp decision records. ep_local_idx pairs episodes across policies under a fixed seed
    # (env i's k-th episode is the same pile for every policy).
    dec_records = []
    ep_local_idx = [0] * env.num_envs

    # Per-episode result lists (for mean ± std reporting)
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []
    per_ep_cycles = []
    per_ep_clearing_curves = []
    per_ep_clearing_curves_cum = []
    clearing_percentages_cum = []

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

            # GAZE: once per pick cycle, when the crane has settled at the gaze pose, save the
            # policy input (basemast RGB + this PCD) so the training input can be validated.
            if gaze_viz_on and int(env._phase[0].item()) != env.PH_GAZE:
                # After the decision-point restructuring, step() returns with the crane at the gaze
                # pose and phase==HOVER_UP, so obs/camera here is the gaze view. We skip the very
                # first iteration (still PH_GAZE at reset = home pose, before any step()).
                # Warm up the color pass (RTX denoises over a few frames) for a non-blank RGB.
                for _ in range(4):
                    env.sim.render()
                env._camera.update(dt=env.cfg.sim.dt)
                for i in range(env.num_envs):
                    save_gaze_rgbd(env, i, obs[i], gaze_viz_dir, gaze_step_idx)
                if gaze_step_idx % 10 == 0:
                    print(f"[Gaze] saved gaze input (RGB+depth+PCD) #{gaze_step_idx}")
                gaze_step_idx += 1

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

            # Pre-step: snapshot policy input + decoded target (before env.step modifies state)
            if args_cli.save_decisions:
                dec_pre = []
                _bed_margin = float(getattr(env.cfg, "platform_bed_margin", 0.0))
                for i in range(env.num_envs):
                    min_b = env._action_bounds_min[i].cpu().numpy()
                    max_b = env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    # record the EXECUTED z (env applies the platform bed floor at its own
                    # decode) + the raw pre-clamp z, mirroring the real node's decisions.jsonl
                    z_raw = z
                    if _bed_margin > 0.0:
                        z = max(z, float(min_b[2]) + _bed_margin)
                    dec_pre.append((obs[i].cpu().numpy().reshape(-1, 3).copy(),
                                    actions[i].cpu().numpy().copy(),
                                    np.array([x, y, z, yaw], dtype=np.float32),
                                    min_b.copy(), max_b.copy(), np.float32(z_raw)))

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

                if args_cli.save_decisions:
                    pts_i, raw_i, tgt_i, bmin_i, bmax_i, zraw_i = dec_pre[i]
                    dec_knocked = int(env._prev_cycle_knocked_off[i].item()) if hasattr(env, '_prev_cycle_knocked_off') else 0
                    if hasattr(env, '_prev_logs_remaining'):
                        dec_remaining = int(env._prev_logs_remaining[i].item()) - logs_grasped - dec_knocked
                    else:
                        dec_remaining = -1
                    dec_records.append({
                        "env": i, "episode": ep_local_idx[i], "cycle": int(ep_cycle_count[i].item()),
                        "points": pts_i, "raw_action": raw_i, "target": tgt_i,
                        "bounds_min": bmin_i, "bounds_max": bmax_i, "z_raw": zraw_i,
                        "logs_grasped": logs_grasped, "alignment": alignment,
                        "stability": stability, "reward": float(rew[i].item()),
                        "logs_remaining": dec_remaining, "knocked_off": dec_knocked,
                        # MEASURED post-despawn rack count. logs_remaining above is DERIVED
                        # (prev - grasped - knocked) and has been seen to go negative; prefer
                        # this one for any clearing / c95 analysis. Added additively so rows
                        # recorded before this change stay comparable on every other field.
                        "logs_in_rack": (int(env._post_cycle_logs_in_rack[i].item())
                                         if hasattr(env, "_post_cycle_logs_in_rack") else -1),
                    })

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
                # Clearing is (starting - what is still in the rack), MEASURED after despawn.
                # It used to be the cumulative sum of logs_grasped, which is a grasp tally rather
                # than a clearing measure - it could exceed 100% (216/200 observed) and c95 read
                # off that same curve, so it could cross 95% before the pile actually had. The
                # cumulative version is kept alongside under *_cumgrasp so rows recorded before
                # this change stay matchable; see metrics_version in the output.
                if hasattr(env, "_post_cycle_logs_in_rack"):
                    in_rack = int(env._post_cycle_logs_in_rack[i].item())
                    episode_true_cleared[i] = max(0, starting - in_rack)
                    clear_pct_now = int(episode_true_cleared[i].item()) / starting * 100
                else:
                    episode_true_cleared[i] = episode_logs_cleared[i]
                    clear_pct_now = int(episode_logs_cleared[i].item()) / starting * 100
                ep_clearing_curves[i].append(round(clear_pct_now, 1))
                ep_clearing_curves_cum[i].append(
                    round(int(episode_logs_cleared[i].item()) / starting * 100, 1))
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
                    ep_local_idx[i] += 1
                    ep_reward = episode_rewards[i].item()
                    total_reward += ep_reward
                    episodes_done += 1

                    starting_logs = int(episode_starting_logs[i].item())
                    logs_cleared_cum = int(episode_logs_cleared[i].item())   # legacy grasp tally
                    logs_cleared = int(episode_true_cleared[i].item())       # measured clearing
                    clear_pct = (logs_cleared / max(1, starting_logs)) * 100
                    clearing_percentages.append(clear_pct)
                    clearing_percentages_cum.append((logs_cleared_cum / max(1, starting_logs)) * 100)
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
                    per_ep_clearing_curves_cum.append(ep_clearing_curves_cum[i][:])

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
                    ep_clearing_curves_cum[i] = []
                    episode_true_cleared[i] = 0
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
    per_ep_cycles_to_95, per_ep_cycles_to_95_cum = [], []
    for curve in per_ep_clearing_curves:
        c95 = next((i + 1 for i, pct in enumerate(curve) if pct >= 95.0), len(curve))
        per_ep_cycles_to_95.append(c95)
    for curve in per_ep_clearing_curves_cum:
        per_ep_cycles_to_95_cum.append(
            next((i + 1 for i, pct in enumerate(curve) if pct >= 95.0), len(curve)))
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

    # Save per-grasp decision records
    if args_cli.save_decisions and dec_records:
        dec_out_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
        os.makedirs(dec_out_dir, exist_ok=True)
        dec_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dec_file = os.path.join(dec_out_dir, f"decisions_{dec_ts}.npz")
        np.savez_compressed(
            dec_file,
            points=np.stack([r["points"] for r in dec_records]).astype(np.float32),
            env=np.array([r["env"] for r in dec_records], dtype=np.int32),
            episode=np.array([r["episode"] for r in dec_records], dtype=np.int32),
            cycle=np.array([r["cycle"] for r in dec_records], dtype=np.int32),
            raw_action=np.stack([r["raw_action"] for r in dec_records]).astype(np.float32),
            target=np.stack([r["target"] for r in dec_records]),
            bounds_min=np.stack([r["bounds_min"] for r in dec_records]).astype(np.float32),
            bounds_max=np.stack([r["bounds_max"] for r in dec_records]).astype(np.float32),
            z_raw=np.array([r["z_raw"] for r in dec_records], dtype=np.float32),
            logs_grasped=np.array([r["logs_grasped"] for r in dec_records], dtype=np.int32),
            logs_remaining=np.array([r["logs_remaining"] for r in dec_records], dtype=np.int32),
            logs_in_rack=np.array([r["logs_in_rack"] for r in dec_records], dtype=np.int32),
            knocked_off=np.array([r["knocked_off"] for r in dec_records], dtype=np.int32),
            alignment=np.array([r["alignment"] for r in dec_records], dtype=np.float32),
            stability=np.array([r["stability"] for r in dec_records], dtype=np.float32),
            reward=np.array([r["reward"] for r in dec_records], dtype=np.float32),
            checkpoint=np.array(args_cli.checkpoint),
            seed=np.array(args_cli.seed if args_cli.seed is not None else -1),
            # Frame index at the end of each cycle (empty unless --record_video). Lets
            # overlay_metrics_video.py map a frame back to its cycle; cycle length varies with
            # how long the FSM takes, so it cannot be recovered from fps.
            video_cycle_bounds=np.array(getattr(env, "_video_cycle_bounds", []), dtype=np.int64),
            video_fps=np.array(args_cli.video_fps, dtype=np.int32),
        )
        print(f"[Play] Decisions saved to: {dec_file} ({len(dec_records)} grasp records)")

    # Save metrics
    if args_cli.save_metrics:
        output_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
        os.makedirs(output_dir, exist_ok=True)
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
                # cycles where the DERIVED remaining disagreed with the MEASURED rack count;
                # non-zero means pile_cleared_pct / cycles_to_95pct for this row are suspect
                "accounting_mismatch_cycles": (int(env._accounting_mismatch.sum().item())
                                               if hasattr(env, "_accounting_mismatch") else -1),
                "zed_noise": bool(args_cli.zed_noise),
                "zed_axial_coeff": float(args_cli.zed_axial_coeff) if args_cli.zed_noise else 0.0,
                "zed_dropout": float(args_cli.zed_dropout) if args_cli.zed_noise else 0.0,
                "num_points": num_points,
                "timestamp": timestamp,
                # 1 = clearing/c95 from cumulative logs_grasped (inflated, can exceed
                # 100%); 2 = from the measured post-despawn rack count.
                "metrics_version": 2,
            },
            "episodes": {
                "total": episodes_done,
                "logs_per_pile_avg": float(_mean(logs_per_episode)),
                "logs_per_pile_min": int(min(logs_per_episode)) if logs_per_episode else 0,
                "logs_per_pile_max": int(max(logs_per_episode)) if logs_per_episode else 0,
            },
            "summary": {
                "reward":            {"mean": avg_reward, "std": std_reward},
                # MEASURED: (starting - logs_in_rack)/starting. *_cumgrasp is the legacy
                # cumulative-logs_grasped definition every row before metrics_version 2 used;
                # it can exceed 100% and must NOT be mixed with the measured one in a table.
                "pile_cleared_pct":  {"mean": avg_clear_pct, "std": std_clear_pct},
                "pile_cleared_pct_cumgrasp": {"mean": _mean(clearing_percentages_cum),
                                              "std": _std(clearing_percentages_cum,
                                                          _mean(clearing_percentages_cum))},
                "cycles_to_95pct_cumgrasp": {"mean": _mean(per_ep_cycles_to_95_cum),
                                             "std": _std(per_ep_cycles_to_95_cum,
                                                         _mean(per_ep_cycles_to_95_cum))},
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

        # Raw per-grasp tilt progressions (windowed stability), if the env recorded them,
        # so the stability metric can be recomputed offline under a different definition.
        _stab_recs = getattr(env, "_stab_records", None)
        if _stab_recs and any(_stab_recs):
            metrics["stability_records"] = {f"env{e}": r for e, r in enumerate(_stab_recs) if r}

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Play] Metrics saved to: {metrics_file}")

    if args_cli.record_video:
        if env._video_writer is not None:
            env._video_writer.close()
            print(f"[Video] Saved {env._video_frame_count} policy-view frames -> {args_cli.video_out}")
        if env._overview_writer is not None:
            env._overview_writer.close()
            print(f"[Video] Saved {env._overview_frame_count} overview frames -> {args_cli.overview_out}")
        if env._sideview_writer is not None:
            env._sideview_writer.close()
            print(f"[Video] Saved {env._sideview_frame_count} sideview frames -> {args_cli.sideview_out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
