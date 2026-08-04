#!/usr/bin/env python3
"""
Behavioral Cloning with Point Cloud observations.

Collects (log_pointcloud, expert_action) pairs and trains a PointNet policy.
Uses masked point cloud (logs only) in base frame.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_pointcloud.py \
        --num_envs 8 --num_episodes 200 --epochs 100 --headless
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from datetime import datetime

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Train BC policy with point cloud observations")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to collect")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--headless", action="store_true", help="Run headless")
# Point cloud settings
parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample")
parser.add_argument("--depth_range_min", type=float, default=1.0, help="Min depth for point cloud")
parser.add_argument("--depth_range_max", type=float, default=10.0, help="Max depth for point cloud")
# Expert options
parser.add_argument("--pos_noise", type=float, default=0.0, help="Position noise (cm)")
parser.add_argument("--yaw_noise", type=float, default=0.0, help="Yaw noise (degrees)")
# Environment options
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--gaze", action="store_true",
                    help="Use the gaze env (basemast cam + PH_GAZE decision point) instead of the full env")
parser.add_argument("--raw_pcd", action="store_true",
                    help="Raw (unmasked) point cloud instead of segmented log-only. Use for gaze/sim2real "
                         "(the real ZED can't segment). Headline BC was raw.")
parser.add_argument("--crop_to_bounds", action="store_true",
                    help="Crop the PCD to the action-bounds box before FPS (removes grapple/trailer/background). "
                         "MUST be matched at deployment.")
parser.add_argument("--save_raw_cap", type=int, default=0,
                    help="ALSO store the PRE-FPS cloud per sample (random-subsampled to this cap, "
                         "fp16) as pointclouds_raw.npy, so FPS count / crop / margin become "
                         "POST-HOC decisions instead of collection-time commitments. ~100 KB/sample "
                         "at 16384. The in-box raw is bounded by get_pointcloud_base's 50k cap.")
parser.add_argument("--raw_margin", type=float, default=1.0,
                    help="raw storage region: action box +- this margin (z_max +0.3). Covers any "
                         "future --crop_margin <= this value.")
parser.add_argument("--crop_margin", type=float, default=0.0,
                    help="Widen the OBSERVATION crop by this many metres in x/y (both sides) and BELOW in z, "
                         "keeping the action box itself unchanged. The tight action-box crop is fitted to the "
                         "rack interior, so it removes the bed plane, the side rails and the pole corners "
                         "BY CONSTRUCTION -- the exact off-pile structure the real crane mis-targets (22% of "
                         "real cycles, 2026-08-03 trials). With a margin those points enter the cloud while the "
                         "expert labels stay on logs, so every structure point becomes an implicit NEGATIVE. "
                         "z_max is left alone (no sky/mast). MUST be matched at deployment (crop_margin param).")
parser.add_argument("--zed_noise", action="store_true",
                    help="Apply ZED axial range-noise as a per-EPOCH training augmentation. Clouds are collected "
                         "CLEAN; fresh noise is drawn each batch so the policy sees many noise draws of the same "
                         "geometry (better than baking one noise realization at collection).")
parser.add_argument("--zed_axial_coeff", type=float, default=0.0014,
                    help="axial noise sigma = coeff * range^2 (m), fitted to the real ZED floor")
parser.add_argument("--z_shift_aug", type=float, default=0.0,
                    help="Per-batch aug: shift each cloud AND its action label UP by a random dz ~ U(0, this) "
                         "meters (capped so the cloud top stays inside the action-box z max). Teaches "
                         "z-equivariance so the policy handles piles TALLER than the sim training piles "
                         "(the real pile tops ~0.6-0.9 m above sim). 0 = off. Try 0.8.")
parser.add_argument("--mound_warp_aug", type=float, default=0.0,
                    help="Per-batch aug: warp the cloud into a MOUND peaked near the label y (z += "
                         "A*gauss(y - y_label), A ~ U(0, this) m, random width), shifting the label z "
                         "consistently. Sim piles settle nearly FLAT, so 'where the top is' is not "
                         "learnable from them (label_y vs top-y corr 0.43); the real pile is a mound. "
                         "This manufactures 'target sits at the peak, wherever the peak is'. 0 = off. Try 0.5.")
parser.add_argument("--dual_save", action="store_true",
                    help="Also save an UNCROPPED full-scene cloud (pointclouds_full.npy) alongside the cropped "
                         "pointclouds.npy, so ONE collection yields both a cropped and a non-cropped dataset. "
                         "The full cloud is FPS'd from the whole scene (no action-box crop). Needs --raw_pcd.")
parser.add_argument("--full_points", type=int, default=2048,
                    help="FPS target for the uncropped full-scene cloud saved by --dual_save. Larger than "
                         "--num_points since the scene is ~12x the rack box.")
parser.add_argument("--use_full_pcd", action="store_true",
                    help="Train-only: load pointclouds_full.npy (uncropped) instead of pointclouds.npy, to train "
                         "the non-cropped policy from a --dual_save collection.")
# Collection render resolution. Lower = less VRAM (the cloud is FPS'd to num_points anyway).
# 8 GB GPUs need this + few envs. (Deploy uses the real ZED at its native res; FPS normalizes.)
parser.add_argument("--cam_width", type=int, default=1280, help="collection camera width (VRAM)")
parser.add_argument("--cam_height", type=int, default=720, help="collection camera height (VRAM)")
# Training options
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
# Collection options
parser.add_argument("--save_interval", type=int, default=20, help="Save checkpoint every N episodes")
parser.add_argument("--save_decisions", action="store_true",
                    help="EVAL mode: record EVERY grasp cycle (failures included) to decisions_*.npz "
                         "in the same schema play_bc_pointcloud writes, so an expert --collect_only "
                         "run can join the per-grasp analyses. Training collection stores successful "
                         "grasps only; this flag adds the complete record alongside it.")
parser.add_argument("--collect_only", action="store_true", help="Only collect, skip training")
parser.add_argument("--train_only", type=str, default=None, help="Skip collection, train from this data dir")
parser.add_argument("--mix_real", type=str, default=None,
                    help="Train-only: also mix in a REAL-cloud dataset dir (from build_real_bc_set.py). "
                         "Teaches the real input distribution (fixes the sim2real z offset) without "
                         "forgetting sim behavior.")
parser.add_argument("--mix_real_repeat", type=int, default=40,
                    help="Oversampling factor for the (small) real set when mixing")
parser.add_argument("--label_dig", type=float, default=0.0,
                    help="COLLECT-time: relabel expert grasp z as (local surface - this) instead of "
                         "the log CENTER convention (= surface - 0.056). Prefer leaving this 0 and "
                         "using --train_label_dig, which applies the same shift at TRAIN time and "
                         "keeps the dataset convention-agnostic (no irreversible floor clamping).")
parser.add_argument("--log_radius", type=float, default=0.056,
                    help="Sim log RADIUS used to convert expert log-CENTRE labels to a surface-"
                         "relative dig (--label_dig / --train_label_dig). Must match the collection: "
                         "0.056 * log_scale_mean (e.g. 0.070 for --log_scale_mean 1.25).")
parser.add_argument("--train_label_dig", type=float, default=0.0,
                    help="TRAIN-time dig convention: shift the loaded SIM label z to (surface - this) "
                         "from the log-center convention, clamped to the bed floor. Sweep this to "
                         "pick a dig without recollecting. Real mixed-in frames are left untouched "
                         "(their labels already carry the executed real convention).")
parser.add_argument("--z_squash_aug", type=float, default=0.0,
                    help="If >0, per-batch pile-height squash toward the cloud floor with scale "
                         "s ~ U(this, 1), label z squashed identically. Teaches DOWNWARD surface "
                         "tracking (measured slope 0.28 on real low piles; z_shift only goes up). "
                         "Suggested 0.5.")
parser.add_argument("--sft_z_only", action="store_true",
                    help="With --init_from: train ONLY the z output (last-layer z row + bias); "
                         "all other params frozen and x/y/yaw gradients masked, so the init's "
                         "target-choice behavior (e.g. RL-learned) is provably preserved. "
                         "Ablation: isolates how much of the sim2real fix is the z correction.")
parser.add_argument("--bimodal_aug", type=float, default=0.0,
                    help="Probability per sample of splicing a second, clearly-lower mound from "
                         "another TRAIN cloud into the far side of the box (label unchanged). "
                         "Teaches commitment on multimodal scenes instead of mode-averaging into "
                         "the gap (observed on the real crane 2026-07-29).")
parser.add_argument("--init_from", type=str, default=None,
                    help="Train-only: initialize from a checkpoint instead of scratch. Accepts a BC "
                         "checkpoint (bc_pointcloud_policy.pt) or an RSL-RL PointNetActorCritic "
                         "checkpoint (model_N.pt): encoder is kept, actor.* is remapped to "
                         "actor_mlp.*, std/critic are dropped. Use a low --lr (e.g. 2e-5) so the "
                         "fine-tune adapts without erasing the initialization.")
args_cli, _ = parser.parse_known_args()

# Skip Isaac imports if train_only mode
if args_cli.train_only is None:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    # Import environment after app launch
    if args_cli.gaze:
        from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull
    else:
        from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class PointNetEncoder(nn.Module):
    """PointNet encoder for point cloud feature extraction."""

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()

        # Shared MLP (applied per-point)
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

        # After max-pooling, project to output dimension
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_points, 3) point cloud

        Returns:
            (batch_size, output_dim) global features
        """
        batch_size, num_points, _ = x.shape

        # Reshape for batch norm: (batch * points, features)
        x = x.view(batch_size * num_points, -1)
        x = self.mlp1(x)

        # Reshape back: (batch, points, features)
        x = x.view(batch_size, num_points, -1)

        # Max pooling across points (global feature)
        x = x.max(dim=1)[0]  # (batch, 256)

        # Final projection
        x = self.fc(x)  # (batch, output_dim)

        return x


class BCPointNetPolicy(nn.Module):
    """PointNet policy for BC, outputs 5D action (x, y, z, cos(2*yaw), sin(2*yaw))."""

    def __init__(self, num_points: int = 1024, action_dim: int = 5,
                 latent_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim

        # PointNet encoder
        self.encoder = PointNetEncoder(input_dim=3, output_dim=latent_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Actor MLP - matches RL structure [128, 64]
        self.actor_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

        print(f"[BC-PointNet] Created: {num_points} points -> {latent_dim} latent -> {action_dim} actions")
        if dropout > 0:
            print(f"[BC-PointNet] Dropout: {dropout}")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (batch_size, num_points, 3) or (batch_size, num_points * 3)

        Returns:
            (batch_size, action_dim) actions
        """
        batch_size = points.shape[0]

        # Handle flattened input
        if points.dim() == 2:
            points = points.view(batch_size, self.num_points, 3)

        latent = self.encoder(points)
        latent = self.dropout(latent)
        return self.actor_mlp(latent)


class PointCloudExpert:
    """Expert that computes 4D actions using env's methods."""

    def __init__(self, env, pos_noise_cm: float = 0.0, yaw_noise_deg: float = 0.0,
                 label_dig: float = 0.0):
        self.env = env
        self.pos_noise = pos_noise_cm / 100.0
        self.yaw_noise = np.radians(yaw_noise_deg)
        self.label_dig = float(label_dig)

    def get_action(self, env_i: int) -> torch.Tensor:
        """Get 4D expert action [x, y, z, yaw] for given environment."""
        env = self.env

        # Use env's existing method to get target position (highest log)
        log_pos_b, selected_id, log_quat_w = env._target_top_log_center_b(env_i)

        if selected_id == -1:
            return torch.zeros(4, device=env.device)

        # Store for yaw calculation
        env._target_log_pos_b[env_i] = log_pos_b
        env._current_target_log_id[env_i] = selected_id
        env._target_log_quat_w[env_i] = log_quat_w

        # Target position
        x_target = log_pos_b[0].item()
        y_target = log_pos_b[1].item()
        z_target = log_pos_b[2].item()

        # LABEL DIG CONVENTION. The expert targets the top log's CENTER, which by construction is
        # local_surface - LOG_RADIUS (0.056). Real close mechanics need a deeper bite: executed
        # real grasps sit 0.2-0.4 m below the surface, and the 2026-07-30 hardware probe with the
        # sim convention (dig 0.056) raked and missed. label_dig relabels to surface - label_dig
        # (constant shift, expert EXECUTION unchanged - collection dynamics stay as validated),
        # clamped to the platform bed floor so every label is executable.
        if self.label_dig > 0.0:
            z_target = z_target + args_cli.log_radius - self.label_dig
            _floor = float(env._action_bounds_min[env_i][2]) + float(
                getattr(env.cfg, "platform_bed_margin", 0.0))
            z_target = max(z_target, _floor)

        # Add noise
        if self.pos_noise > 0:
            x_target += np.random.normal(0, self.pos_noise)
            y_target += np.random.normal(0, self.pos_noise)
            z_target += np.random.normal(0, self.pos_noise)

        # Get target basegrapple yaw in basemast frame (state-independent)
        # Policy predicts this, then env converts to joint position at runtime
        yaw_target = env._get_target_grapple_yaw_b(env_i)
        if self.yaw_noise > 0:
            yaw_target += np.random.normal(0, self.yaw_noise)

        # Get action bounds
        if not env._action_bounds_valid[env_i]:
            env._compute_action_space_bounds()

        min_b = env._action_bounds_min[env_i]
        max_b = env._action_bounds_max[env_i]

        # Normalize to [-1, 1] then arctanh (same as other BC scripts)
        def normalize_and_arctanh(val, vmin, vmax):
            range_v = vmax - vmin
            if range_v <= 0:
                return 0.0
            norm = 2.0 * (val - vmin) / range_v - 1.0
            norm = max(-0.999, min(0.999, norm))
            return np.arctanh(norm)

        x_action = normalize_and_arctanh(x_target, min_b[0].item(), max_b[0].item())
        y_action = normalize_and_arctanh(y_target, min_b[1].item(), max_b[1].item())
        z_action = normalize_and_arctanh(z_target, min_b[2].item(), max_b[2].item())
        # 5D action: yaw as (cos(2*yaw), sin(2*yaw)) -> continuous, bounded [-1,1], and
        # automatically symmetry-aware (the grapple is symmetric, so yaw and yaw+pi are the
        # SAME grasp; doubling the angle folds them together). Far more learnable than a single
        # arctanh(yaw), which is circular + contradictory. Matches the cossin BC variant.
        yaw_cos = float(np.cos(2.0 * yaw_target))
        yaw_sin = float(np.sin(2.0 * yaw_target))
        return torch.tensor([x_action, y_action, z_action, yaw_cos, yaw_sin],
                           device=env.device, dtype=torch.float32)


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling for point cloud downsampling.

    Args:
        points: (N, 3) point cloud
        num_samples: number of points to sample

    Returns:
        (num_samples, 3) sampled points
    """
    device = points.device
    N = points.shape[0]

    if N <= num_samples:
        # Pad with zeros if not enough points
        if N == 0:
            return torch.zeros((num_samples, 3), device=device)
        padding = torch.zeros((num_samples - N, 3), device=device)
        return torch.cat([points, padding], dim=0)

    # FPS algorithm
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    # Start with random point
    current_idx = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_indices[i] = current_idx
        current_point = points[current_idx:current_idx+1]  # (1, 3)

        # Update distances
        dist_to_current = torch.norm(points - current_point, dim=1)
        distances = torch.minimum(distances, dist_to_current)

        # Select farthest point
        current_idx = torch.argmax(distances).item()

    return points[sampled_indices]


# side-channel for --save_raw_cap: env_idx -> last pre-FPS raw cloud (fp16 numpy). Written by
# get_log_pointcloud_base_frame, read by the collect loop in the same iteration.
_LAST_RAW = {}


def get_log_pointcloud_base_frame(env, env_idx: int, num_points: int,
                                   depth_range: tuple = (1.0, 10.0),
                                   raw_pcd: bool = False,
                                   full_points: int = 0):
    """
    Get point cloud in crane base frame (raw full-scene or segmented log-only).
    Kept identical to play_bc_pointcloud.py so train/eval/deploy share one preprocessing.

    Args:
        env: Environment
        env_idx: Environment index
        num_points: Number of points to return (via FPS) for the (optionally cropped) deploy cloud
        depth_range: (min, max) depth range
        raw_pcd: if True, full-scene cloud (matches the real ZED); else log-only via segmentation
        full_points: if > 0, ALSO return an UNCROPPED full-scene cloud FPS'd to this many points
                     (for --dual_save). Both clouds come from the same base-frame scene.

    Returns:
        (num_points, 3) cropped cloud, OR
        ((num_points, 3), (full_points, 3)) when full_points > 0.
    """
    def _ret(cropped, full):
        return (cropped, full) if full_points else cropped

    # Point cloud in the crane BASE frame.
    if raw_pcd:
        # raw / sim2real: straight optical -> base (no world pivot), mirrors the real tf chain.
        # NOTE: cap kept high so the rack survives the crop. The basemast cam sees a wide scene and
        # the rack is ~8% of it; a low cap here (was 5000) leaves <1024 in-box pts -> FPS pads zeros.
        # The real downsampling is the FPS-to-num_points AFTER the crop.
        pc_base = env.get_pointcloud_base(env_idx, max_points=50000, depth_range=depth_range)
    else:
        # segmented log-only (sim-only): world cloud, then world -> base.
        pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)
        if pc_world.shape[0] == 0:
            return _ret(torch.zeros((num_points, 3), device=env.device),
                        torch.zeros((full_points, 3), device=env.device))
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
        return _ret(torch.zeros((num_points, 3), device=env.device),
                    torch.zeros((full_points, 3), device=env.device))

    # UNCROPPED full-scene cloud (for --dual_save). Pre-subsample huge clouds before FPS so the
    # O(full_points * N) FPS loop stays cheap; random pre-sampling is uniform so it doesn't bias FPS.
    full_sampled = None
    if full_points:
        pc_full = pc_base
        cap = full_points * 6
        if pc_full.shape[0] > cap:
            idx = torch.randperm(pc_full.shape[0], device=pc_full.device)[:cap]
            pc_full = pc_full[idx]
        full_sampled = farthest_point_sampling(pc_full, full_points)

    # PRE-FPS raw storage (--save_raw_cap): everything in the generous raw box, random-subsampled
    # (cheap, unbiased) instead of FPS'd, fp16 on CPU. This is what makes num_points/crop/margin
    # re-derivable offline later; FPS at collection time is a one-way door.
    if getattr(args_cli, "save_raw_cap", 0) and getattr(env, "_action_bounds_min", None) is not None:
        _b0 = env._action_bounds_min[env_idx]
        _b1 = env._action_bounds_max[env_idx]
        _g = float(getattr(args_cli, "raw_margin", 1.0))
        _m = ((pc_base[:, 0] >= _b0[0] - _g) & (pc_base[:, 0] <= _b1[0] + _g) &
              (pc_base[:, 1] >= _b0[1] - _g) & (pc_base[:, 1] <= _b1[1] + _g) &
              (pc_base[:, 2] >= _b0[2] - _g) & (pc_base[:, 2] <= _b1[2] + 0.3))
        _r = pc_base[_m]
        _cap = int(args_cli.save_raw_cap)
        if _r.shape[0] > _cap:
            _r = _r[torch.randperm(_r.shape[0], device=_r.device)[:_cap]]
        _out = torch.zeros((_cap, 3), dtype=torch.float16)
        _out[:_r.shape[0]] = _r.detach().cpu().half()
        _LAST_RAW[env_idx] = _out.numpy()

    # Optionally crop to the action-bounds box before FPS (same crop as eval + deploy).
    # NOTE --crop_margin widens the OBSERVATION crop only; the action box (and therefore the
    # arctanh target encoding) is untouched, so datasets and policies stay interchangeable.
    if getattr(args_cli, "crop_to_bounds", False) and getattr(env, "_action_bounds_min", None) is not None:
        bmin = env._action_bounds_min[env_idx]
        bmax = env._action_bounds_max[env_idx]
        g = float(getattr(args_cli, "crop_margin", 0.0))
        m = ((pc_base[:, 0] >= bmin[0] - g) & (pc_base[:, 0] <= bmax[0] + g) &
             (pc_base[:, 1] >= bmin[1] - g) & (pc_base[:, 1] <= bmax[1] + g) &
             (pc_base[:, 2] >= bmin[2] - g) & (pc_base[:, 2] <= bmax[2]))
        pc_base = pc_base[m]
        if pc_base.shape[0] == 0:
            return _ret(torch.zeros((num_points, 3), device=env.device), full_sampled)

    # FPS to fixed number of points
    pc_sampled = farthest_point_sampling(pc_base, num_points)

    return _ret(pc_sampled, full_sampled)


def collect_demonstrations(env, num_episodes: int, output_dir: str,
                           num_points: int, depth_range: tuple,
                           pos_noise_cm: float, yaw_noise_deg: float,
                           save_interval: int = 20, full_points: int = 0):
    """Collect (pointcloud, action) pairs from expert.

    When full_points > 0 (--dual_save), also collect an UNCROPPED full-scene cloud per sample
    (pointclouds_full[i] pairs with pointclouds[i] and actions[i])."""
    device = env.device

    pointclouds = []
    pointclouds_raw = []  # --save_raw_cap: pre-FPS fp16 clouds
    dec_records = []  # --save_decisions: full per-cycle record (failures included)
    pointclouds_full = []   # uncropped, only populated when full_points > 0
    actions = []
    episode_rewards = []
    current_reward = torch.zeros(env.num_envs, device=device)

    # Metrics
    successful_grasps = 0
    failed_grasps = 0
    total_logs_grasped = 0
    total_alignment = 0.0
    piles_fully_cleared = 0
    clearing_percentages = []
    logs_per_episode = []
    episodes_done = 0

    episode_logs_cleared = torch.zeros(env.num_envs, device=device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=device, dtype=torch.int32)

    expert = PointCloudExpert(env, pos_noise_cm=pos_noise_cm, yaw_noise_deg=yaw_noise_deg,
                              label_dig=args_cli.label_dig)

    print(f"[BC-PointCloud] Collecting {num_episodes} episodes")
    print(f"[BC-PointCloud] Points per sample: {num_points}")
    print(f"[BC-PointCloud] Only keeping SUCCESSFUL grasps")

    env.reset()
    env.sim.render()
    env._camera.update(dt=env.cfg.sim.dt)

    # Get starting log counts
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[BC-PointCloud] Logs per pile: {np.mean(log_counts):.0f} avg")
    else:
        episode_starting_logs[:] = 200

    while episodes_done < num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Get point cloud observations for all envs BEFORE step
            pc_batch = []
            pcf_batch = []
            for i in range(env.num_envs):
                out = get_log_pointcloud_base_frame(env, i, num_points, depth_range,
                                                    raw_pcd=args_cli.raw_pcd,
                                                    full_points=full_points)
                if full_points:
                    pc, pcf = out
                    pcf_batch.append(pcf.cpu().numpy())
                else:
                    pc = out
                pc_batch.append(pc.cpu().numpy())
            pc_batch = np.stack(pc_batch)  # (num_envs, num_points, 3)
            pcf_batch = np.stack(pcf_batch) if full_points else None
            raw_cap = int(getattr(args_cli, "save_raw_cap", 0))
            raw_batch = (np.stack([_LAST_RAW.get(i, np.zeros((raw_cap, 3), np.float16))
                                   for i in range(env.num_envs)]) if raw_cap else None)

            # Get expert actions
            expert_actions = torch.stack([expert.get_action(i) for i in range(env.num_envs)])
            action_batch = expert_actions.cpu().numpy()

            # Step environment
            _, rew, terminated, truncated, _ = env.step(expert_actions)
            env.sim.render()
            env._camera.update(dt=env.cfg.sim.dt)

            # Check outcomes for each env
            for env_i in range(env.num_envs):
                logs_this_grasp = int(env._prev_logs_grasped[env_i].item())
                alignment_this_grasp = env._prev_grasp_alignment[env_i].item()

                episode_logs_cleared[env_i] += logs_this_grasp

                if args_cli.save_decisions:
                    # decode the expert action exactly as the env does (expert path is unclamped,
                    # so executed target == raw target; record z_raw = target z for schema parity)
                    bmin = env._action_bounds_min[env_i].cpu().numpy()
                    bmax = env._action_bounds_max[env_i].cpu().numpy()
                    a = action_batch[env_i]
                    tgt = bmin[:3] + (np.tanh(a[:3]) + 1.0) / 2.0 * (bmax[:3] - bmin[:3])
                    yaw = 0.5 * np.arctan2(a[4], a[3]) if len(a) >= 5 else float(a[3])
                    dec_records.append(dict(
                        points=pc_batch[env_i], env=env_i, episode=episodes_done,
                        cycle=int(env._cycle_count[env_i].item()),
                        raw_action=a, target=np.array([tgt[0], tgt[1], tgt[2], yaw], np.float32),
                        bounds_min=bmin, bounds_max=bmax, z_raw=float(tgt[2]),
                        logs_grasped=logs_this_grasp,
                        logs_remaining=int(env._prev_logs_remaining[env_i].item()) - logs_this_grasp,
                        knocked_off=int(env._prev_cycle_knocked_off[env_i].item()),
                        alignment=float(alignment_this_grasp),
                        stability=float(env._prev_grasp_stability[env_i].item()),
                        reward=float(rew[env_i].item())))

                if logs_this_grasp > 0:  # Successful grasp
                    pointclouds.append(pc_batch[env_i])
                    if raw_batch is not None:
                        pointclouds_raw.append(raw_batch[env_i])
                    if full_points:
                        pointclouds_full.append(pcf_batch[env_i])
                    actions.append(action_batch[env_i])
                    successful_grasps += 1
                    total_logs_grasped += logs_this_grasp
                    total_alignment += alignment_this_grasp
                else:
                    failed_grasps += 1

            current_reward += rew
            done = terminated | truncated

            for env_i in range(env.num_envs):
                if done[env_i]:
                    episode_rewards.append(current_reward[env_i].item())

                    total_logs_this_env = int(episode_starting_logs[env_i].item())
                    logs_cleared = int(episode_logs_cleared[env_i].item())
                    # Prefer the env's authoritative clearing (1 - remaining/starting), captured at
                    # termination before auto-reset. The grasp-count estimate below is a fallback for
                    # envs without the buffer; it can exceed 100% because one cycle's proximity check
                    # may count several logs.
                    clear_frac = None
                    if hasattr(env, "_last_episode_clearing"):
                        v = float(env._last_episode_clearing[env_i].item())
                        clear_frac = v if v >= 0.0 else None
                    if clear_frac is not None:
                        clear_pct = clear_frac * 100.0
                    else:
                        clear_pct = (logs_cleared / max(1, total_logs_this_env)) * 100
                    clearing_percentages.append(clear_pct)
                    logs_per_episode.append(total_logs_this_env)

                    if clear_pct >= 100.0:
                        piles_fully_cleared += 1

                    # Reset tracking
                    current_reward[env_i] = 0.0
                    episode_logs_cleared[env_i] = 0
                    if has_variable_logs:
                        episode_starting_logs[env_i] = int(env._per_env_log_counts[env_i].item())
                    episodes_done += 1

                    # Print progress
                    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
                    avg_clear_pct = np.mean(clearing_percentages) if clearing_percentages else 0.0

                    print(f"[BC-PointCloud] Ep {episodes_done}/{num_episodes} | "
                          f"Samples: {len(pointclouds)} | "
                          f"Grasp: {grasp_rate:.0f}% | "
                          f"Clear: {avg_clear_pct:.1f}%")

                    # Save checkpoint
                    if episodes_done % save_interval == 0:
                        save_checkpoint(output_dir, pointclouds, actions, {
                            "episode_rewards": episode_rewards,
                            "clearing_percentages": clearing_percentages,
                        }, num_points,
                            pointclouds_full=pointclouds_full if full_points else None,
                            full_points=full_points,
                            pointclouds_raw=pointclouds_raw if pointclouds_raw else None)

                    if episodes_done >= num_episodes:
                        break

    if args_cli.save_decisions and dec_records:
        from datetime import datetime as _dt
        dec_file = os.path.join(output_dir, f"decisions_{_dt.now().strftime('%Y%m%d_%H%M%S')}.npz")
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
            knocked_off=np.array([r["knocked_off"] for r in dec_records], dtype=np.int32),
            alignment=np.array([r["alignment"] for r in dec_records], dtype=np.float32),
            stability=np.array([r["stability"] for r in dec_records], dtype=np.float32),
            reward=np.array([r["reward"] for r in dec_records], dtype=np.float32),
            checkpoint=np.array("expert:" + getattr(args_cli, "expert_target", "highest")
                                + f"_dig{getattr(args_cli, 'expert_dig', 0.0)}"),
            seed=np.array(getattr(args_cli, "seed", -1)),
        )
        print(f"[BC-PointCloud] Decisions saved to: {dec_file} ({len(dec_records)} grasp records)")

    # Summary
    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
    avg_clear_pct = np.mean(clearing_percentages) if clearing_percentages else 0.0
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_align = total_alignment / max(1, successful_grasps)

    print(f"\n[BC-PointCloud] ====== COLLECTION SUMMARY ======")
    print(f"  Episodes: {episodes_done}")
    print(f"  Samples collected: {len(pointclouds)}")
    print(f"  Grasp success rate: {grasp_rate:.1f}%")
    print(f"  Avg pile cleared: {avg_clear_pct:.1f}%")
    print(f"  Full clears: {full_clear_rate:.1f}%")
    print(f"  Avg throughput: {avg_throughput:.2f} logs/grasp")
    print(f"  Avg alignment: {avg_align:.3f}")
    print(f"=" * 50)

    metrics = {
        "collection": {
            "episodes": episodes_done,
            "samples": len(pointclouds),
            "num_points": num_points,
        },
        "grasp": {
            "success_rate": grasp_rate,
            "successful": successful_grasps,
            "failed": failed_grasps,
        },
        "pile_clearing": {
            "avg_cleared_pct": avg_clear_pct,
            "full_clear_rate": full_clear_rate,
            "full_clears": piles_fully_cleared,
        },
        "performance": {
            "avg_throughput": avg_throughput,
            "avg_alignment": avg_align,
        },
        "episode_rewards": episode_rewards,
        "clearing_percentages": clearing_percentages,
    }

    full_arr = np.stack(pointclouds_full) if full_points else None
    raw_arr = np.stack(pointclouds_raw).astype(np.float16) if pointclouds_raw else None
    return np.stack(pointclouds), np.stack(actions), metrics, full_arr, raw_arr


def save_checkpoint(output_dir, pointclouds, actions, metrics, num_points,
                    pointclouds_full=None, full_points=0, pointclouds_raw=None):
    """Save collection checkpoint. When pointclouds_full is given, also writes the uncropped
    pointclouds_full.npy (paired 1:1 with pointclouds.npy). pointclouds_raw (--save_raw_cap) is
    the PRE-FPS fp16 cloud per sample - the re-derivation source for any future num_points/crop/
    margin choice - written as pointclouds_raw.npy, also paired 1:1."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "pointclouds.npy"), np.stack(pointclouds))
    if pointclouds_raw:
        np.save(os.path.join(output_dir, "pointclouds_raw.npy"),
                np.stack(pointclouds_raw).astype(np.float16))
    np.save(os.path.join(output_dir, "actions.npy"), np.stack(actions))
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(metrics.get("episode_rewards", [])))
    if pointclouds_full:
        np.save(os.path.join(output_dir, "pointclouds_full.npy"), np.stack(pointclouds_full))

    metadata = {
        "num_points": num_points,
        "obs_dim": num_points * 3,
        "action_dim": 5,
        "num_samples": len(pointclouds),
        # obs preprocessing — eval/deploy MUST use the same to transfer
        "gaze": bool(getattr(args_cli, "gaze", False)),
        "raw_pcd": bool(getattr(args_cli, "raw_pcd", False)),
        "crop_to_bounds": bool(getattr(args_cli, "crop_to_bounds", False)),
        "depth_range": [args_cli.depth_range_min, args_cli.depth_range_max],
    }
    if pointclouds_raw:
        metadata["raw_pcd_store"] = {
            "file": "pointclouds_raw.npy", "dtype": "float16",
            "cap": int(getattr(args_cli, "save_raw_cap", 0)),
            "raw_margin": float(getattr(args_cli, "raw_margin", 1.0)),
            "note": "pre-FPS, random-subsampled, action box +- raw_margin (z_max +0.3); "
                    "derive any (crop, margin<=raw_margin, num_points) dataset from this offline",
        }
    if pointclouds_full:
        # The uncropped companion dataset (train with --train_only <dir> --use_full_pcd).
        metadata["full_pcd"] = {
            "file": "pointclouds_full.npy",
            "num_points": full_points,
            "crop_to_bounds": False,
        }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    extra = f" (+{len(pointclouds_full)} full)" if pointclouds_full else ""
    print(f"[BC-PointCloud] Checkpoint saved: {len(pointclouds)} samples{extra}")


def gaze_cam_pos_base(env, env_idx: int = 0) -> np.ndarray:
    """The basemast-camera optical origin in the crane base frame at the (held) gaze pose.
    Used as the apex for range-dependent ZED noise. Range-dependence is gentle, so small
    pose error here is harmless."""
    t_opt_w, q_opt_w = env._gaze_camera_pose_w(env_idx)
    bq = env.crane.data.root_quat_w[env_idx]
    bp = env.crane.data.root_pos_w[env_idx]
    w, x, y, z = bq[0], bq[1], bq[2], bq[3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])
    return (R.transpose(0, 1) @ (t_opt_w - bp)).detach().cpu().numpy()


def apply_zed_noise(pc, cam_pos, coeff):
    """Per-batch ZED axial noise: each point perturbed along its camera ray by N(0, coeff*range^2).
    pc: (B, N, 3) base-frame; cam_pos: (3,) base-frame tensor. Zero-padding points (all-zero) are
    left untouched. Returns a noised copy."""
    ray = pc - cam_pos                                   # (B,N,3) cam->point
    rng = ray.norm(dim=-1, keepdim=True)                 # (B,N,1)
    dirn = ray / (rng + 1e-6)
    sigma = coeff * rng ** 2
    noised = pc + dirn * (torch.randn_like(rng) * sigma)
    keep = (pc.abs().sum(-1, keepdim=True) > 1e-6)       # don't move zero-padded slots
    return torch.where(keep, noised, pc)


# Action-box bounds (gaze action box; matches deploy crop + label encoding).
Z_BOX_MIN, Z_BOX_MAX = -1.30, 0.10
LOG_RADIUS = 0.056   # sim log radius: expert label z = local surface - LOG_RADIUS
BED_MARGIN = 0.10    # platform bed floor margin (matches crane_rl_env_gaze platform_bed_margin)
Y_BOX_MIN, Y_BOX_MAX = -1.684, 5.316


def apply_mound_warp(pc, act, max_amp: float):
    """Per-batch shape aug: warp each (near-flat) cloud into a mound peaked near its label y.

    z_i += A * exp(-0.5*((y_i - y0)/w)^2) with A ~ U(0, max_amp), w ~ U(0.6, 1.8) m, and
    y0 = label_y + N(0, 0.25). The label z is shifted by the SAME warp evaluated at the label's
    own y, so the pair stays consistent: the expert's log (already the marginally-highest in the
    flat pile) becomes the peak of a pronounced mound -> the net learns to put the target at the
    peak WHEREVER the peak is, which flat sim piles cannot teach (label_y vs top-y corr 0.43).
    The warp is scaled down per-sample so the cloud top stays <= Z_BOX_MAX (deploy crops there).
    Zero-padded slots untouched. pc: (B, N, 3); act: (B, A) arctanh-encoded, y at 1, z at 2."""
    B = pc.shape[0]
    dev = pc.device
    keep = (pc.abs().sum(-1) > 1e-6)                                  # (B, N)
    ty = Y_BOX_MIN + (torch.tanh(act[:, 1]) + 1.0) / 2.0 * (Y_BOX_MAX - Y_BOX_MIN)
    A = torch.rand(B, device=dev) * max_amp                           # (B,)
    # Two profile shapes, half/half per sample:
    # - BUMP: gaussian mound peaked near the label (localized high spot).
    # - STEP: sigmoid plateau, label on the RAISED side (the real post-grasp pile is a low near
    #   band + broad elevated plateau; a bump-only aug doesn't fire on plateaus). The label's
    #   whole side shifts ~uniformly, so "label = top" is preserved exactly.
    is_step = torch.rand(B, device=dev) < 0.5
    w = 0.6 + torch.rand(B, device=dev) * 1.9                         # (B,) bump width in m
    y0 = ty + torch.randn(B, device=dev) * 0.25                       # bump peak near the label
    bump = A[:, None] * torch.exp(-0.5 * ((pc[:, :, 1] - y0[:, None]) / w[:, None]) ** 2)
    side = torch.where(torch.rand(B, device=dev) < 0.5,
                       torch.ones(B, device=dev), -torch.ones(B, device=dev))
    edge = ty - side * (0.3 + torch.rand(B, device=dev) * 1.7)        # label on the raised side
    w_s = 0.15 + torch.rand(B, device=dev) * 0.35                     # step edge sharpness in m
    step = A[:, None] * torch.sigmoid(side[:, None] * (pc[:, :, 1] - edge[:, None]) / w_s[:, None])
    warp = torch.where(is_step[:, None], step, bump)
    warp = torch.where(keep, warp, torch.zeros_like(warp))
    # Scale so the warped top stays inside the box: s = (ZMAX - z*) / warp* at the would-be top.
    warped_z = pc[:, :, 2] + warp
    zmaxed = torch.where(keep, warped_z, torch.full_like(warped_z, -1e9))
    star = zmaxed.argmax(dim=1)                                       # (B,)
    z_star = pc[torch.arange(B), star, 2]
    w_star = warp[torch.arange(B), star]
    s = torch.where(w_star > 1e-6,
                    ((Z_BOX_MAX - z_star) / (w_star + 1e-9)).clamp(0.0, 1.0),
                    torch.ones_like(w_star))
    warp = warp * s[:, None]
    pc = pc.clone()
    pc[:, :, 2] = torch.where(keep, (pc[:, :, 2] + warp).clamp(max=Z_BOX_MAX), pc[:, :, 2])
    # Label z += warp evaluated at the label's own (y), same scale.
    lab_bump = A * torch.exp(-0.5 * ((ty - y0) / w) ** 2)
    lab_step = A * torch.sigmoid(side * (ty - edge) / w_s)
    lab_warp = s * torch.where(is_step, lab_step, lab_bump)
    act = act.clone()
    z_norm = torch.tanh(act[:, 2])
    z_m = (Z_BOX_MIN + (z_norm + 1.0) / 2.0 * (Z_BOX_MAX - Z_BOX_MIN) + lab_warp).clamp(max=Z_BOX_MAX)
    z_norm2 = (2.0 * (z_m - Z_BOX_MIN) / (Z_BOX_MAX - Z_BOX_MIN) - 1.0).clamp(-0.999, 0.999)
    act[:, 2] = torch.atanh(z_norm2)
    return pc, act


def apply_z_shift(pc, act, max_shift: float):
    """Per-batch z-equivariance aug: shift each cloud AND its z-label up by the same random dz.

    A valid grasp shifted up is still a valid grasp of the shifted cloud, so the (cloud, label)
    pair stays consistent while the policy learns that target-z tracks pile-z (instead of
    memorizing the sim pile's absolute height, which tops ~0.6-0.9 m below the real pile).
    dz ~ U(0, max_shift), capped per-sample so the cloud top stays <= Z_BOX_MAX (deploy crops
    there, so higher points would never be seen). Labels are arctanh(normalized) -> decode,
    shift in meters, re-encode. Zero-padded slots are left untouched.

    pc: (B, N, 3), act: (B, A) with z at index 2. Returns (pc_shifted, act_shifted)."""
    B = pc.shape[0]
    keep = (pc.abs().sum(-1) > 1e-6)                      # (B, N) non-pad points
    zvals = torch.where(keep, pc[:, :, 2], torch.full_like(pc[:, :, 2], -1e9))
    top = zvals.max(dim=1).values                         # (B,) per-cloud top z
    dz = torch.rand(B, device=pc.device) * max_shift
    dz = torch.minimum(dz, (Z_BOX_MAX - top).clamp(min=0.0))
    pc = pc.clone()
    pc[:, :, 2] = torch.where(keep, pc[:, :, 2] + dz[:, None], pc[:, :, 2])
    act = act.clone()
    z_norm = torch.tanh(act[:, 2])                        # label stored as arctanh(normalized)
    z_m = Z_BOX_MIN + (z_norm + 1.0) / 2.0 * (Z_BOX_MAX - Z_BOX_MIN) + dz
    z_norm2 = (2.0 * (z_m - Z_BOX_MIN) / (Z_BOX_MAX - Z_BOX_MIN) - 1.0).clamp(-0.999, 0.999)
    act[:, 2] = torch.atanh(z_norm2)
    return pc, act


def load_init_state(path: str) -> dict:
    """BCPointNetPolicy state dict from a BC or RSL-RL checkpoint.

    RSL-RL PointNetActorCritic stores the head as actor.N.*; BC uses actor_mlp.N.*
    (same shapes). Encoder keys are identical in both. std/critic* have no BC
    counterpart and are dropped.
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("actor.") for k in state):
        remapped = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                remapped[k] = v
            elif k.startswith("actor."):
                remapped["actor_mlp." + k[len("actor."):]] = v
        state = remapped
    return state


def apply_z_squash(pc, act, min_scale: float):
    """Per-batch pile-height squash: compress each cloud toward its own floor by a random
    scale s ~ U(min_scale, 1), label z compressed identically.

    Teaches DOWNWARD surface tracking. Measured 2026-07-29 (416 July real frames): the
    pre-SFT policy's commanded z follows the local surface with slope 0.28 instead of 1.0 -
    it anchors near the training pile-height band, hovering +0.34 m over July's low piles
    while being spot-on at sim-typical heights. apply_z_shift only ever shifts piles UP, and
    a LOW pile is not a translated pile (the floor stays put, the pile is shallower), so the
    physically consistent aug is compression toward the floor.

    pc: (B, N, 3), act: (B, A) with z at index 2 (arctanh-normalized)."""
    B = pc.shape[0]
    keep = (pc.abs().sum(-1) > 1e-6)
    zvals = torch.where(keep, pc[:, :, 2], torch.full_like(pc[:, :, 2], 1e9))
    floor = zvals.min(dim=1).values                       # (B,) per-cloud floor
    s = min_scale + torch.rand(B, device=pc.device) * (1.0 - min_scale)
    pc = pc.clone()
    pc[:, :, 2] = torch.where(keep, floor[:, None] + s[:, None] * (pc[:, :, 2] - floor[:, None]),
                              pc[:, :, 2])
    act = act.clone()
    z_norm = torch.tanh(act[:, 2])
    z_m = Z_BOX_MIN + (z_norm + 1.0) / 2.0 * (Z_BOX_MAX - Z_BOX_MIN)
    z_m = floor + s * (z_m - floor)
    z_norm2 = (2.0 * (z_m - Z_BOX_MIN) / (Z_BOX_MAX - Z_BOX_MIN) - 1.0).clamp(-0.999, 0.999)
    act[:, 2] = torch.atanh(z_norm2)
    return pc, act


def apply_bimodal_mix(pc, act, pool_pc, prob: float):
    """Per-batch multimodal-commitment aug: carve the cloud into a partially-cleared scene
    with the labeled mound plus (usually) a second, clearly LOWER distractor mound across an
    empty gap. Labels are unchanged.

    Real two-mound scenes (crane, 2026-07-29) made the MSE head average the two modes and
    target the empty gap between them; every training scene is a single connected full-length
    pile, so the net never had to choose. Sim clouds span the whole rack, so bimodality has to
    be MANUFACTURED by carving: keep a mound-width window around the label (the pile remnant
    that still holds the label), then splice in a window from another train cloud on the far
    side, top dropped 0.15-0.40 m below the primary's, so the expert convention (target the
    taller pile) stays label-consistent and the net learns to COMMIT to the dominant mound.
    ~30%% of selected samples get carve-only (a single remnant mound, no distractor), which
    doubles as depleted-state coverage. Near-tie scenes are not generated (still ambiguous).

    pc: (B, N, 3), act: (B, A); pool_pc: (M, N, 3) TRAIN-split clouds for distractors.
    Returns (pc_mixed, act)."""
    B, N, _ = pc.shape
    sel = torch.rand(B, device=pc.device) < prob
    if not sel.any():
        return pc, act
    pc = pc.clone()
    for i in torch.nonzero(sel).flatten().tolist():
        cloud = pc[i][pc[i].abs().sum(-1) > 1e-6]
        if len(cloud) < 64:
            continue
        y_lab = float(Y_BOX_MIN + (torch.tanh(act[i, 1]) + 1.0) / 2.0 * (Y_BOX_MAX - Y_BOX_MIN))
        w_p = 0.6 + torch.rand(1).item() * 1.0                      # primary remnant half-width
        prim = cloud[(cloud[:, 1] - y_lab).abs() <= w_p]
        if len(prim) < 100:
            continue
        pieces = [prim]
        if torch.rand(1).item() > 0.3:                              # 70%: add distractor mound
            part = pool_pc[torch.randint(len(pool_pc), (1,)).item()]
            part = part[part.abs().sum(-1) > 1e-6]
            w_d = 0.4 + torch.rand(1).item() * 0.8                  # distractor half-width
            gap = 0.5 + torch.rand(1).item() * 1.0                  # empty gap between mounds
            side = 1.0 if y_lab < 0.5 * (Y_BOX_MIN + Y_BOX_MAX) else -1.0
            y_d = y_lab + side * (w_p + gap + w_d)
            if Y_BOX_MIN + 0.2 < y_d - w_d and y_d + w_d < Y_BOX_MAX - 0.2:
                c_j = part[:, 1].median()
                part = part.clone()
                part[:, 1] += y_d - c_j
                part = part[(part[:, 1] - y_d).abs() <= w_d]
                if len(part) >= 30:
                    # Keep a clear dominance margin (>= 0.15 m, well above sensor noise): an MSE
                    # head must only be supervised where the answer is unambiguous given the
                    # input. Near-tie pairs under input noise are ambiguous supervision and
                    # would re-teach averaging; genuine ties at deploy time are left to
                    # per-capture noise to break (either mound is a valid grasp).
                    drop = 0.15 + torch.rand(1).item() * 0.25
                    part[:, 2] += (prim[:, 2].max() - drop) - part[:, 2].max()
                    part = part[part[:, 2] >= Z_BOX_MIN]
                    if len(part) >= 20:
                        pieces.append(part)
        merged = torch.cat(pieces, dim=0)
        keep_idx = torch.randperm(len(merged), device=pc.device)[:N]
        out = torch.zeros(N, 3, device=pc.device, dtype=pc.dtype)
        out[:len(keep_idx)] = merged[keep_idx]
        pc[i] = out
    return pc, act


def train_policy(pointclouds: np.ndarray, actions: np.ndarray,
                 num_points: int, batch_size: int, lr: float,
                 epochs: int, device: str, dropout: float = 0.2,
                 zed_noise: bool = False, cam_pos=None, zed_axial_coeff: float = 0.0014,
                 z_shift_aug: float = 0.0, mound_warp_aug: float = 0.0,
                 bimodal_aug: float = 0.0, z_squash_aug: float = 0.0,
                 init_state: dict = None, sft_z_only: bool = False):
    """Train PointNet policy via supervised learning."""

    print(f"\n[BC-PointCloud] Training PointNet policy")
    print(f"  Samples: {len(pointclouds)}")
    print(f"  Points: {num_points}")
    print(f"  Action dim: {actions.shape[1]}")
    print("=" * 60)

    # Create dataset - flatten point clouds for DataLoader
    pc_flat = pointclouds.reshape(len(pointclouds), -1)  # (N, num_points * 3)
    pc_t = torch.from_numpy(pc_flat).float().to(device)
    act_t = torch.from_numpy(actions).float().to(device)
    dataset = TensorDataset(pc_t, act_t)

    # Split
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Create model
    policy = BCPointNetPolicy(num_points=num_points, action_dim=actions.shape[1], dropout=dropout).to(device)
    if init_state is not None:
        policy.load_state_dict(init_state)
        print(f"  Initialized from checkpoint ({len(init_state)} tensors)")

    if sft_z_only:
        assert init_state is not None, "--sft_z_only requires --init_from"
        # Freeze everything except the last layer; mask its gradients so only the z output
        # row (index 2) and its bias train. x/y/yaw and all shared features are untouched,
        # so the init's target-choice behavior is preserved exactly.
        last = policy.actor_mlp[-1]
        for p in policy.parameters():
            p.requires_grad = False
        last.weight.requires_grad = True
        last.bias.requires_grad = True
        z_row = torch.zeros_like(last.weight)
        z_row[2] = 1.0
        z_bias = torch.zeros_like(last.bias)
        z_bias[2] = 1.0
        last.weight.register_hook(lambda g: g * z_row)
        last.bias.register_hook(lambda g: g * z_bias)
        policy.encoder.eval()   # freeze BatchNorm running stats too
        _orig_train = policy.train
        def _train_frozen_bn(mode=True):
            _orig_train(mode)
            policy.encoder.eval()
            return policy
        policy.train = _train_frozen_bn
        n_train = 64 + 1
        print(f"  SFT Z-ONLY: {n_train} trainable values (last-layer z row + bias); "
              f"x/y/yaw and all features frozen")

    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

    cam_pos_t = None
    if zed_noise and cam_pos is not None:
        cam_pos_t = torch.as_tensor(np.asarray(cam_pos, dtype=np.float32), device=device)
        print(f"  ZED noise AUG: per-epoch axial noise, coeff={zed_axial_coeff}, cam_pos(base)={np.round(np.asarray(cam_pos),3).tolist()}")
    elif zed_noise:
        print("  [WARN] --zed_noise set but no cam_pos available; augmentation DISABLED")
    if z_shift_aug > 0:
        print(f"  Z-SHIFT AUG: cloud+label shifted up by dz ~ U(0, {z_shift_aug}) m per batch "
              f"(capped at box z_max {Z_BOX_MAX}) -> z-equivariance for taller-than-sim piles")
    bimodal_pool = None
    if bimodal_aug > 0:
        pool_idx = torch.as_tensor(train_ds.indices, device=device)
        bimodal_pool = pc_t[pool_idx].view(-1, num_points, 3)
        print(f"  BIMODAL AUG: p={bimodal_aug}, lower distractor mound spliced from "
              f"{len(pool_idx)} train clouds -> commitment on multimodal scenes")
    if mound_warp_aug > 0:
        print(f"  MOUND-WARP AUG: cloud warped into a mound peaked near label y, A ~ U(0, {mound_warp_aug}) m "
              f"-> teaches 'target the peak wherever it is' (flat sim piles can't)")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    early_stop_patience = 30

    for epoch in range(epochs):
        # Train
        policy.train()
        train_loss = 0.0
        for b_pc, b_act in train_loader:
            optimizer.zero_grad()
            if cam_pos_t is not None:   # fresh ZED noise this batch (clouds collected clean)
                bp3 = apply_zed_noise(b_pc.reshape(b_pc.shape[0], num_points, 3), cam_pos_t, zed_axial_coeff)
            else:
                bp3 = b_pc.reshape(b_pc.shape[0], num_points, 3)
            if mound_warp_aug > 0:      # shape aug: mound peaked near the label (warps label z too)
                bp3, b_act = apply_mound_warp(bp3, b_act, mound_warp_aug)
            if z_shift_aug > 0:         # z-equivariance aug (shifts label z consistently)
                bp3, b_act = apply_z_shift(bp3, b_act, z_shift_aug)
            if z_squash_aug > 0:        # downward tracking aug (squash pile toward floor)
                bp3, b_act = apply_z_squash(bp3, b_act, z_squash_aug)
            if bimodal_aug > 0:         # multimodal-commitment aug (label unchanged)
                bp3, b_act = apply_bimodal_mix(bp3, b_act, bimodal_pool, bimodal_aug)
            bp_in = bp3.reshape(b_pc.shape[0], -1)
            pred = policy(bp_in)
            loss = criterion(pred, b_act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_pc, b_act in val_loader:
                pred = policy(b_pc)
                loss = criterion(pred, b_act)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}")

        if patience_counter >= early_stop_patience:
            print(f"\n[BC-PointCloud] Early stopping at epoch {epoch}")
            break

    policy.load_state_dict(best_state)
    print(f"\n[BC-PointCloud] Best val loss: {best_val_loss:.6f}")

    return policy, best_val_loss


def save_policy(policy: BCPointNetPolicy, output_dir: str, num_points: int, metadata: dict):
    """Save policy in BC and RSL-RL compatible formats."""
    os.makedirs(output_dir, exist_ok=True)

    # BC checkpoint
    torch.save({
        'model_state_dict': policy.state_dict(),
        'num_points': num_points,
        'obs_dim': num_points * 3,
        'action_dim': 5,
        'metadata': metadata,
    }, os.path.join(output_dir, "bc_pointcloud_policy.pt"))

    # RSL-RL compatible checkpoint
    # Map BC weights to PointNetActorCritic structure
    rsl_state = {}
    bc_state = policy.state_dict()

    # Copy encoder weights
    for name, param in bc_state.items():
        if name.startswith("encoder."):
            rsl_state[name] = param.cpu().clone()

    # Map actor_mlp -> actor
    for name, param in bc_state.items():
        if name.startswith("actor_mlp."):
            new_name = name.replace("actor_mlp.", "actor.")
            rsl_state[new_name] = param.cpu().clone()

    # Initialize critic with random weights
    for name, param in bc_state.items():
        if name.startswith("actor_mlp."):
            new_name = name.replace("actor_mlp.", "critic.")
            if "4.weight" in name:  # Last layer
                rsl_state[new_name] = torch.randn(1, param.shape[1])
            elif "4.bias" in name:
                rsl_state[new_name] = torch.zeros(1)
            else:
                rsl_state[new_name] = torch.randn_like(param.cpu())

    # Add std for action noise
    rsl_state["std"] = torch.ones(4) * 0.3

    torch.save({
        'model_state_dict': rsl_state,
        'optimizer_state_dict': {},
        'iter': 0,
        'infos': {
            'bc_pretrained': True,
            'num_points': num_points,
            'obs_dim': num_points * 3,
        },
    }, os.path.join(output_dir, "bc_pointcloud_policy_rsl_rl.pt"))

    print(f"\n[BC-PointCloud] Saved to {output_dir}/")
    print(f"  - bc_pointcloud_policy.pt (standalone)")
    print(f"  - bc_pointcloud_policy_rsl_rl.pt (for RL fine-tuning)")


def main():
    print(f"[BC-PointCloud] Args: num_episodes={args_cli.num_episodes}, num_envs={args_cli.num_envs}")
    # Handle train_only mode
    if args_cli.train_only:
        print(f"[BC-PointCloud] Train-only mode from {args_cli.train_only}")
        with open(os.path.join(args_cli.train_only, "metadata.json")) as f:
            metadata = json.load(f)

        if args_cli.use_full_pcd:
            # Non-cropped companion dataset from a --dual_save collection.
            pcd_file = os.path.join(args_cli.train_only, "pointclouds_full.npy")
            if not os.path.exists(pcd_file):
                raise FileNotFoundError(
                    f"--use_full_pcd but no pointclouds_full.npy in {args_cli.train_only} "
                    f"(collect with --dual_save first)")
            pointclouds = np.load(pcd_file)
            full_meta = metadata.get("full_pcd", {})
            num_points = full_meta.get("num_points", pointclouds.shape[1])
            metadata = {**metadata, "num_points": num_points, "obs_dim": num_points * 3,
                        "crop_to_bounds": False}
            print(f"[BC-PointCloud] Training NON-CROPPED policy: {pointclouds.shape} pts/sample")
        else:
            pointclouds = np.load(os.path.join(args_cli.train_only, "pointclouds.npy"))
            num_points = metadata["num_points"]
        actions = np.load(os.path.join(args_cli.train_only, "actions.npy"))
        if args_cli.train_label_dig > 0.0:
            # sim labels are arctanh-encoded log CENTERS (= local surface - LOG_RADIUS); shift them
            # to (surface - train_label_dig) in metres, then re-encode and clamp to the bed floor.
            _shift = args_cli.log_radius - args_cli.train_label_dig
            _floor = Z_BOX_MIN + BED_MARGIN
            _z = Z_BOX_MIN + (np.tanh(actions[:, 2]) + 1.0) / 2.0 * (Z_BOX_MAX - Z_BOX_MIN)
            _z = np.maximum(_z + _shift, _floor)
            _n = np.clip(2.0 * (_z - Z_BOX_MIN) / (Z_BOX_MAX - Z_BOX_MIN) - 1.0, -0.999, 0.999)
            actions = actions.copy()
            actions[:, 2] = np.arctanh(_n)
            print(f"[BC-PointCloud] TRAIN label dig: z -> surface-{args_cli.train_label_dig:.2f} "
                  f"(shift {_shift:+.3f} m, floor {_floor:.2f}); "
                  f"{100.0 * (_z <= _floor + 1e-6).mean():.0f}% of labels sit at the floor")
        if args_cli.mix_real:
            # real-crane clouds with auto expert labels: oversample and concat. The real samples
            # carry the true input distribution (real log geometry/surface stats) that the sim
            # data misses; mixing keeps sim behavior while correcting the real-input z offset.
            rp = np.load(os.path.join(args_cli.mix_real, "pointclouds.npy"))
            ra = np.load(os.path.join(args_cli.mix_real, "actions.npy"))
            assert rp.shape[1:] == pointclouds.shape[1:], \
                f"real set {rp.shape} incompatible with sim set {pointclouds.shape}"
            rep = max(1, int(args_cli.mix_real_repeat))
            pointclouds = np.concatenate([pointclouds, np.tile(rp, (rep, 1, 1))])
            actions = np.concatenate([actions, np.tile(ra, (rep, 1))])
            print(f"[BC-PointCloud] Mixed in REAL set: {len(rp)} samples x{rep} "
                  f"({len(rp)*rep}/{len(pointclouds)} = {100*len(rp)*rep/len(pointclouds):.0f}% of training)")
        # camera apex for ZED-noise aug (saved at collection); None if absent
        _cam_path = os.path.join(args_cli.train_only, "cam_pos_base.npy")
        cam_pos = np.load(_cam_path) if (args_cli.zed_noise and os.path.exists(_cam_path)) else None

        # Create output dir
        if args_cli.output_dir:
            output_dir = args_cli.output_dir
        else:
            base_output_dir = os.path.abspath(os.path.join("logs", "bc_pointcloud"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_output_dir, f"bc_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        init_state = None
        if args_cli.init_from:
            init_state = load_init_state(args_cli.init_from)
            metadata = {**metadata, "init_from": args_cli.init_from}

        policy, best_loss = train_policy(
            pointclouds, actions, num_points,
            batch_size=args_cli.batch_size,
            lr=args_cli.lr,
            epochs=args_cli.epochs,
            device=args_cli.device,
            dropout=args_cli.dropout,
            zed_noise=args_cli.zed_noise, cam_pos=cam_pos, zed_axial_coeff=args_cli.zed_axial_coeff,
            z_shift_aug=args_cli.z_shift_aug, mound_warp_aug=args_cli.mound_warp_aug,
            bimodal_aug=args_cli.bimodal_aug, z_squash_aug=args_cli.z_squash_aug,
            init_state=init_state, sft_z_only=args_cli.sft_z_only
        )

        save_policy(policy, output_dir, num_points, metadata)
        return

    # Full pipeline: collect + train
    if args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        base_output_dir = os.path.abspath(os.path.join("logs", "bc_pointcloud"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"bc_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[BC-PointCloud] Output: {output_dir}")

    # Save config
    config = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_envs": args_cli.num_envs,
        "num_episodes": args_cli.num_episodes,
        "num_points": args_cli.num_points,
        "depth_range": [args_cli.depth_range_min, args_cli.depth_range_max],
        "domain_randomization": args_cli.domain_randomization,
        "batch_size": args_cli.batch_size,
        "lr": args_cli.lr,
        "epochs": args_cli.epochs,
        "dropout": args_cli.dropout,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.enable_camera = True
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.camera_cfg.width = args_cli.cam_width     # lower render res -> less VRAM (cloud is FPS'd anyway)
    cfg.camera_cfg.height = args_cli.cam_height
    cfg.zed_noise = False   # BC collects CLEAN clouds; ZED noise is applied as a per-epoch train aug (--zed_noise)
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    print(f"[BC-PointCloud] Domain randomization: {args_cli.domain_randomization}")
    env = CraneDirectEnvFull(cfg)
    print(f"[BC-PointCloud] Environment created with camera")

    # Collect
    depth_range = (args_cli.depth_range_min, args_cli.depth_range_max)
    full_points = args_cli.full_points if args_cli.dual_save else 0
    if full_points and not args_cli.raw_pcd:
        print("[BC-PointCloud] WARNING: --dual_save is only meaningful with --raw_pcd; ignoring.")
        full_points = 0
    pointclouds, actions, metrics, pointclouds_full, pointclouds_raw = collect_demonstrations(
        env, args_cli.num_episodes, output_dir,
        num_points=args_cli.num_points,
        depth_range=depth_range,
        pos_noise_cm=args_cli.pos_noise,
        yaw_noise_deg=args_cli.yaw_noise,
        save_interval=args_cli.save_interval,
        full_points=full_points
    )

    # Camera apex for the per-epoch ZED noise augmentation (saved for --train_only too)
    # ALWAYS record the gaze camera apex: clouds are collected clean and the ZED-noise aug is
    # applied per-batch at TRAIN time, which needs this file. Gating it on --zed_noise (a
    # collection-time flag nobody passes) meant --zed_noise silently disabled itself later.
    cam_pos = gaze_cam_pos_base(env, 0)
    if cam_pos is not None:
        np.save(os.path.join(output_dir, "cam_pos_base.npy"), cam_pos)
        print(f"[BC-PointCloud] saved cam_pos_base {np.round(cam_pos,3).tolist()} for ZED-noise aug")

    # Final save
    save_checkpoint(output_dir, list(pointclouds), list(actions), metrics, args_cli.num_points,
                    pointclouds_full=list(pointclouds_full) if full_points else None,
                    full_points=full_points,
                    pointclouds_raw=list(pointclouds_raw) if pointclouds_raw is not None else None)
    if full_points:
        print(f"[BC-PointCloud] Uncropped companion: pointclouds_full.npy "
              f"({len(pointclouds_full)} x {full_points} pts). Train it with:")
        print(f"  python train_bc_pointcloud.py --train_only {output_dir} --use_full_pcd --raw_pcd --epochs 100")

    # Save full metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Close env to free memory
    env.close()

    if args_cli.collect_only:
        print(f"\n[BC-PointCloud] Collection complete! Data saved to {output_dir}")
        print(f"To train later:")
        print(f"  python train_bc_pointcloud.py --train_only {output_dir} --epochs 100")
        simulation_app.close()
        return

    # Train
    print(f"\n[BC-PointCloud] Starting training...")
    policy, best_loss = train_policy(
        pointclouds, actions, args_cli.num_points,
        batch_size=args_cli.batch_size,
        lr=args_cli.lr,
        epochs=args_cli.epochs,
        device=args_cli.device,
        dropout=args_cli.dropout,
        zed_noise=args_cli.zed_noise, cam_pos=cam_pos, zed_axial_coeff=args_cli.zed_axial_coeff,
        z_shift_aug=args_cli.z_shift_aug, mound_warp_aug=args_cli.mound_warp_aug
    )

    metadata = {
        "num_points": args_cli.num_points,
        "obs_dim": args_cli.num_points * 3,
        "action_dim": 5,
    }
    save_policy(policy, output_dir, args_cli.num_points, metadata)

    print(f"\n[BC-PointCloud] ====== COMPLETE ======")
    print(f"[BC-PointCloud] Output: {output_dir}")
    print(f"[BC-PointCloud]")
    print(f"[BC-PointCloud] To fine-tune with RL:")
    print(f"  ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \\")
    print(f"      --task Isaac-Crane-PointCloud-v0 --num_envs 16 --headless \\")
    print(f"      --checkpoint {output_dir}/bc_pointcloud_policy_rsl_rl.pt")

    simulation_app.close()


if __name__ == "__main__":
    main()
