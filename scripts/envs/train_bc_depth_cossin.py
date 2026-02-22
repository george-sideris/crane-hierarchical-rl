#!/usr/bin/env python3
"""
Behavioral Cloning with Depth Images - Cos/Sin Yaw Encoding

Collects (depth_image, expert_action) pairs and trains a CNN policy.
Uses cos/sin encoding for yaw to handle π-periodicity (log symmetry).

Action space: [x, y, z, cos(2*yaw), sin(2*yaw)] - 5D instead of 4D
This encoding makes yaw and yaw+π map to the same values automatically.

Based on Morrison et al. / Kumra et al. grasp detection papers:
"This component-wise encoding makes it easier for a neural network to
learn the π-periodicity of the grapple orientation."

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_depth_cossin.py \
        --num_envs 4 --num_episodes 100 --epochs 100 --headless

    # With diversity (recommended)
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_depth_cossin.py \
        --num_envs 8 --num_episodes 500 --epochs 100 --headless \
        --top_k 3 --pos_noise 3.0 --yaw_noise 5.0
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

# Add crane_testbed agents path for ResNet encoder (avoid triggering Isaac imports)
# Script is at: crane_testbed/scripts/envs/train_bc_depth.py
# Module is at: crane_testbed/source/crane_testbed/crane_testbed/agents/resnet_depth_encoder.py
crane_testbed_root = crane_scripts_path.parent.parent  # Go up from scripts/envs to crane_testbed root
crane_agents_path = crane_testbed_root / "source" / "crane_testbed" / "crane_testbed" / "agents"
sys.path.insert(0, str(crane_agents_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Train BC policy with depth images")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--headless", action="store_true", help="Run headless")
# Depth settings
parser.add_argument("--depth_height", type=int, default=128, help="Depth image height")
parser.add_argument("--depth_width", type=int, default=128, help="Depth image width")
parser.add_argument("--use_semantic_mask", action="store_true", default=True, help="Use semantic mask")
parser.add_argument("--no_semantic_mask", action="store_true", help="Disable semantic mask")
# Expert options
parser.add_argument("--top_k", type=int, default=1, help="Pick from top K logs (not implemented)")
parser.add_argument("--pos_noise", type=float, default=0.0, help="Position noise (cm)")
parser.add_argument("--yaw_noise", type=float, default=0.0, help="Yaw noise (degrees)")
# Environment options
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
# Training options
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--augmentation", action="store_true", default=True, help="Enable mild augmentation (noise/contrast only, no spatial)")
parser.add_argument("--no_augmentation", action="store_true", help="Disable augmentation")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (0 to disable)")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
parser.add_argument("--encoder_type", type=str, default="cnn", choices=["cnn", "resnet"], help="Encoder architecture (cnn or resnet)")
parser.add_argument("--apply_mask_during_training", action="store_true", default=False, help="Mask background pixels during training (not recommended - causes distribution mismatch)")
parser.add_argument("--no_mask_during_training", action="store_true", help="Disable training-time masking")
# Collection options
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N episodes")
parser.add_argument("--collect_only", action="store_true", help="Only collect, skip training")
parser.add_argument("--train_only", type=str, default=None, help="Skip collection, train from this data dir")
parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint directory")
args_cli, _ = parser.parse_known_args()

if args_cli.no_semantic_mask:
    args_cli.use_semantic_mask = False

if args_cli.no_augmentation:
    args_cli.augmentation = False

if args_cli.no_mask_during_training:
    args_cli.apply_mask_during_training = False

# Skip Isaac imports if train_only mode
if args_cli.train_only is None:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    # Import environment after app launch
    from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class BCCNNPolicy(nn.Module):
    """CNN policy for BC, matching the RL CNNActorCritic architecture."""

    def __init__(self, obs_dim: int, action_dim: int = 5, dropout: float = 0.0):
        super().__init__()

        # Detect image size from obs_dim
        import math
        sqrt = int(math.sqrt(obs_dim))
        if sqrt * sqrt != obs_dim:
            raise ValueError(f"obs_dim={obs_dim} is not a perfect square")
        self.img_size = sqrt
        self.dropout_rate = dropout

        print(f"[BC-CNN] Image size: {self.img_size}x{self.img_size}")

        # Build encoder based on image size - MUST match RL CNNActorCritic exactly!
        # Using ELU to match RL default activation
        if self.img_size == 256:
            # 256x256 encoder: 256 -> 128 -> 64 -> 32 -> 32 -> 16 -> 8 (6 conv layers)
            # Added BatchNorm after each Conv2d for training stability
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 256 -> 128
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 32 (stride=1!)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
                nn.BatchNorm2d(128),
                nn.ELU(),
            )
            self.flatten_dim = 128 * 8 * 8  # 8192
        elif self.img_size == 128:
            # 128x128 encoder: 128 -> 64 -> 32 -> 16 -> 8 (6 conv layers)
            # Added BatchNorm after each Conv2d for training stability
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 128 -> 64
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64 -> 64 (stride=1!)
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 32 (stride=1!)
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
                nn.BatchNorm2d(128),
                nn.ELU(),
            )
            self.flatten_dim = 128 * 8 * 8  # 8192
        else:
            # 48x48 encoder: 48 -> 24 -> 12 -> 10 -> 8 (4 conv layers)
            # Added BatchNorm after each Conv2d for training stability
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 48 -> 24
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 24 -> 12
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 12 -> 10
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), # 10 -> 8
                nn.BatchNorm2d(64),
                nn.ELU(),
            )
            self.flatten_dim = 64 * 8 * 8  # 4096

        # Latent projection - matches RL encoder_fc
        self.latent_dim = 256
        self.latent_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, self.latent_dim),
            nn.ELU(),
        )

        # Dropout for regularization (only applied in latent space)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Actor MLP - matches RL actor with hidden_dims=[128, 64]
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 128),  # 0: 256 -> 128
            nn.ELU(),                          # 1
            nn.Linear(128, 64),                # 2: 128 -> 64
            nn.ELU(),                          # 3
            nn.Linear(64, action_dim),         # 4: 64 -> 4
        )

        print(f"[BC-CNN] Encoder: {self.flatten_dim} -> latent: {self.latent_dim} -> action: {action_dim}")
        if dropout > 0:
            print(f"[BC-CNN] Dropout: {dropout}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_size, self.img_size)
        features = self.encoder(x)
        features = features.view(batch_size, -1)
        latent = self.latent_proj(features)
        latent = self.dropout(latent)  # Apply dropout in latent space
        return self.actor_mlp(latent)


class BCResNetPolicy(nn.Module):
    """ResNet + CBAM policy for BC with better feature extraction."""

    def __init__(self, obs_dim: int, action_dim: int = 5, dropout: float = 0.0, latent_dim: int = 256):
        super().__init__()

        # Detect image size from obs_dim
        import math
        sqrt = int(math.sqrt(obs_dim))
        if sqrt * sqrt != obs_dim:
            raise ValueError(f"obs_dim={obs_dim} is not a perfect square")
        self.img_size = sqrt
        self.dropout_rate = dropout

        print(f"[BC-ResNet] Image size: {self.img_size}x{self.img_size}")

        # Import ResNet encoder (direct import to avoid Isaac Lab dependencies)
        from resnet_depth_encoder import ResNetDepthEncoder

        # ResNet encoder with CBAM attention
        self.encoder = ResNetDepthEncoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Actor MLP - matches BCCNNPolicy structure
        self.actor_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

        print(f"[BC-ResNet] Latent: {self.latent_dim} -> action: {action_dim}")
        if dropout > 0:
            print(f"[BC-ResNet] Dropout: {dropout}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_size, self.img_size)
        latent = self.encoder(x)
        latent = self.dropout(latent)  # Apply dropout in latent space
        return self.actor_mlp(latent)


class DepthExpert:
    """Expert that computes 4D actions using env's methods."""

    def __init__(self, env, top_k: int = 1, pos_noise_cm: float = 0.0, yaw_noise_deg: float = 0.0):
        self.env = env
        self.top_k = max(1, top_k)
        self.pos_noise = pos_noise_cm / 100.0
        self.yaw_noise = np.radians(yaw_noise_deg)

    def get_action(self, env_i: int) -> torch.Tensor:
        """Get 4D expert action [x, y, z, yaw] for given environment."""
        env = self.env

        # Use env's existing method to get target position (highest log)
        # Returns: log_pos_b, selected_log_id, log_quat_w
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

        # Normalize to [-1, 1] then arctanh
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

        # Cos/Sin encoding for yaw with π-periodicity (handles log symmetry)
        # Following Morrison et al. / Kumra et al. grasp detection papers
        # yaw and yaw+π map to same [cos(2*yaw), sin(2*yaw)] encoding
        yaw_cos = np.cos(2.0 * yaw_target)
        yaw_sin = np.sin(2.0 * yaw_target)

        return torch.tensor([x_action, y_action, z_action, yaw_cos, yaw_sin],
                           device=env.device, dtype=torch.float32)


def get_depth_observation(env, depth_height: int, depth_width: int,
                          use_semantic_mask: bool, depth_min: float = 1.5,
                          depth_max: float = 8.0) -> torch.Tensor:
    """Get processed depth observation (same as CraneDepthDirectEnv)."""
    env._camera.update(dt=env.cfg.sim.dt)

    depth = env._camera.data.output["depth"].squeeze(-1)

    if use_semantic_mask and "semantic_segmentation" in env.cfg.camera_cfg.data_types:
        sem_seg = env._camera.data.output["semantic_segmentation"]
        if sem_seg.dim() == 4:
            sem_ids = sem_seg[..., 0]
        else:
            sem_ids = sem_seg
        mask = (sem_ids > 0)
        depth = torch.where(mask, depth, torch.tensor(depth_max, device=depth.device))

    depth = torch.clamp(depth, depth_min, depth_max)
    depth = (depth - depth_min) / (depth_max - depth_min)

    depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=(depth_height, depth_width),
                          mode='bilinear', align_corners=False)
    depth = depth.squeeze(1)

    return depth.view(env.num_envs, -1)


def collect_demonstrations(env, num_episodes: int, output_dir: str,
                           depth_height: int, depth_width: int,
                           use_semantic_mask: bool,
                           top_k: int, pos_noise_cm: float, yaw_noise_deg: float,
                           save_interval: int = 10):
    """Collect (depth, action) pairs from expert. Only keeps successful grasps.

    FIXED: Simplified to match pose-based BC flow exactly:
    1. Get observation BEFORE step
    2. Get expert action BEFORE step
    3. Execute step
    4. Keep samples only if grasp succeeded
    """
    device = env.device
    obs_dim = depth_height * depth_width

    depth_observations = []
    actions = []
    episode_rewards = []
    current_reward = torch.zeros(env.num_envs, device=device)

    # Metrics
    successful_grasps = 0
    failed_grasps = 0
    total_logs_grasped = 0
    total_alignment = 0.0
    piles_fully_cleared = 0
    grasps_to_clear = []
    clearing_percentages = []
    logs_per_episode = []
    episode_grasps = torch.zeros(env.num_envs, device=device, dtype=torch.int32)
    episode_logs_cleared = torch.zeros(env.num_envs, device=device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=device, dtype=torch.int32)

    expert = DepthExpert(env, top_k=top_k, pos_noise_cm=pos_noise_cm, yaw_noise_deg=yaw_noise_deg)

    print(f"\n[BC-Depth] Collecting {num_episodes} episodes")
    print(f"[BC-Depth] Depth: {depth_height}x{depth_width} = {obs_dim} dims")
    print(f"[BC-Depth] Semantic mask: {use_semantic_mask}")
    print(f"[BC-Depth] Only keeping SUCCESSFUL grasps (logs > 0)")
    if pos_noise_cm > 0 or yaw_noise_deg > 0:
        print(f"[BC-Depth] Noise: pos={pos_noise_cm}cm, yaw={yaw_noise_deg}deg")
    print("=" * 60)

    env.reset()
    env.sim.render()

    # Update camera to see initial state
    env._camera.update(dt=env.cfg.sim.dt)

    # Get starting log counts
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

    episodes_done = 0

    while episodes_done < num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # BEFORE step: Get depth observation for ALL envs (like pose-based BC)
            depth_obs = get_depth_observation(env, depth_height, depth_width,
                                              use_semantic_mask)
            obs_batch = depth_obs.cpu().numpy()

            # BEFORE step: Get expert actions for ALL envs (single call, no redundancy)
            expert_actions = torch.stack([expert.get_action(i) for i in range(env.num_envs)])
            action_batch = expert_actions.cpu().numpy()

            # Step runs entire grasp cycle (hierarchical mode)
            _, rew, terminated, truncated, _ = env.step(expert_actions)
            env.sim.render()

            # Update camera for next iteration
            env._camera.update(dt=env.cfg.sim.dt)

            # Track rewards
            current_reward += rew

            # AFTER step: Check outcomes and keep successful grasps
            for env_i in range(env.num_envs):
                episode_grasps[env_i] += 1
                logs_this_grasp = int(env._prev_logs_grasped[env_i].item())
                alignment = env._prev_grasp_alignment[env_i].item() if hasattr(env, '_prev_grasp_alignment') else 0.0
                episode_logs_cleared[env_i] += logs_this_grasp

                if logs_this_grasp > 0:
                    # Successful grasp - save the obs/action pair
                    depth_observations.append(obs_batch[env_i])
                    actions.append(action_batch[env_i])
                    successful_grasps += 1
                    total_logs_grasped += logs_this_grasp
                    total_alignment += alignment
                else:
                    # Failed grasp - discard
                    failed_grasps += 1

            # Check completions
            done = terminated | truncated
            for env_i in range(env.num_envs):
                if done[env_i]:
                    episode_rewards.append(current_reward[env_i].item())

                    # Clearing percentage
                    starting = int(episode_starting_logs[env_i].item())
                    if starting <= 0:
                        starting = 200  # Fallback
                    cleared = int(episode_logs_cleared[env_i].item())
                    clear_pct = (cleared / max(1, starting)) * 100
                    clearing_percentages.append(clear_pct)
                    logs_per_episode.append(starting)

                    # Check if pile was fully cleared
                    if cleared >= starting:
                        piles_fully_cleared += 1
                        grasps_to_clear.append(int(episode_grasps[env_i].item()))

                    # Reset tracking
                    current_reward[env_i] = 0.0
                    episode_grasps[env_i] = 0
                    episode_logs_cleared[env_i] = 0
                    if has_variable_logs:
                        episode_starting_logs[env_i] = int(env._per_env_log_counts[env_i].item())

                    episodes_done += 1

                    # Progress
                    if episodes_done % 10 == 0 or episodes_done == num_episodes:
                        grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
                        avg_clear = np.mean(clearing_percentages) if clearing_percentages else 0
                        print(f"[BC-Depth] Ep {episodes_done}/{num_episodes} | "
                              f"Samples: {len(depth_observations)} | "
                              f"Grasp: {grasp_rate:.0f}% | "
                              f"Clear: {avg_clear:.1f}%")

                    # Save checkpoint
                    if episodes_done % save_interval == 0:
                        checkpoint_metrics = {
                            "episode_rewards": episode_rewards,
                            "clearing_percentages": clearing_percentages,
                        }
                        save_checkpoint(output_dir, depth_observations, actions,
                                       checkpoint_metrics, depth_height, depth_width,
                                       use_semantic_mask)

                    if episodes_done >= num_episodes:
                        break

    # Final summary
    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
    avg_clear = np.mean(clearing_percentages) if clearing_percentages else 0
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_align = total_alignment / max(1, successful_grasps)

    print("\n" + "=" * 60)
    print(f"[BC-Depth] ====== COLLECTION SUMMARY ======")
    print(f"  Episodes: {episodes_done}")
    print(f"  Samples collected: {len(depth_observations)}")
    if logs_per_episode:
        print(f"  Logs per pile: {np.mean(logs_per_episode):.1f} avg ({min(logs_per_episode)}-{max(logs_per_episode)} range)")
    print(f"  Grasp success rate: {grasp_rate:.1f}% ({successful_grasps}/{successful_grasps + failed_grasps})")
    print(f"  Avg pile cleared: {avg_clear:.1f}%")
    print(f"  Full clears: {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done} episodes)")
    print(f"  Avg throughput: {avg_throughput:.2f} logs/grasp")
    print(f"  Avg alignment: {avg_align:.3f}")
    print("=" * 60)

    metrics = {
        "collection": {
            "episodes": episodes_done,
            "samples": len(depth_observations),
            "logs_per_pile_avg": float(np.mean(logs_per_episode)) if logs_per_episode else 0,
            "logs_per_pile_min": int(min(logs_per_episode)) if logs_per_episode else 0,
            "logs_per_pile_max": int(max(logs_per_episode)) if logs_per_episode else 0,
            "depth_height": depth_height,
            "depth_width": depth_width,
            "use_semantic_mask": use_semantic_mask,
        },
        "grasp": {
            "success_rate": grasp_rate,
            "successful": successful_grasps,
            "failed": failed_grasps,
            "total": successful_grasps + failed_grasps,
        },
        "pile_clearing": {
            "avg_cleared_pct": avg_clear,
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

    return np.stack(depth_observations), np.stack(actions), metrics


def save_checkpoint(output_dir, depth_obs, actions, metrics, depth_height, depth_width, use_semantic_mask):
    """Save collection checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "depth_observations.npy"), np.stack(depth_obs))
    np.save(os.path.join(output_dir, "actions.npy"), np.stack(actions))
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(metrics.get("episode_rewards", [])))

    metadata = {
        "depth_height": depth_height,
        "depth_width": depth_width,
        "use_semantic_mask": use_semantic_mask,
        "obs_dim": depth_height * depth_width,
        "action_dim": 5,  # [x, y, z, cos(2*yaw), sin(2*yaw)]
        "num_samples": len(depth_obs),
        "num_episodes": len(metrics.get("episode_rewards", [])),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save full metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[BC-Depth] Checkpoint saved: {len(depth_obs)} samples")


def apply_mask_during_training(obs: torch.Tensor, img_size: int, threshold: float = 0.99) -> torch.Tensor:
    """Apply semantic masking during training to prevent background overfitting.

    Background pixels (value ~1.0) are replaced with the mean log depth to force
    the network to focus on log features instead of memorizing background patterns.

    Args:
        obs: Flattened depth observations [B, H*W]
        img_size: Image height/width
        threshold: Values >= threshold are considered background

    Returns:
        Masked observations [B, H*W]
    """
    batch_size = obs.shape[0]
    imgs = obs.view(batch_size, 1, img_size, img_size)

    # Identify background pixels (depth = 1.0)
    mask = (imgs < threshold)  # True for log pixels, False for background

    # For each image in batch, replace background with mean log depth
    masked_imgs = imgs.clone()
    for i in range(batch_size):
        if mask[i].any():  # If there are any log pixels
            log_mean = imgs[i][mask[i]].mean()
            masked_imgs[i] = torch.where(mask[i], imgs[i], log_mean)
        # else: all background, leave as-is

    return masked_imgs.view(batch_size, -1)


def augment_depth_batch(obs: torch.Tensor, img_size: int) -> torch.Tensor:
    """Apply MILD data augmentation to depth images during training.

    NOTE: For BC, we must NOT use spatial augmentations (flip, rotate, crop)
    because the expert action was computed for the original image orientation.
    Flipping the image without adjusting the action creates inconsistent pairs.

    Safe augmentations (don't change spatial structure):
    - Small Gaussian noise (simulates sensor noise)
    - Mild contrast adjustment (simulates lighting variation)
    """
    batch_size = obs.shape[0]
    # Reshape to images for augmentation
    imgs = obs.view(batch_size, 1, img_size, img_size)

    # NO horizontal flip - this would break obs-action correspondence!
    # The expert action was computed for the original image.

    # Add Gaussian noise (std=0.02 - stronger for better regularization)
    noise = torch.randn_like(imgs) * 0.02
    imgs = imgs + noise

    # Contrast adjustment (0.9 to 1.1 multiplier - stronger variation)
    contrast = 0.9 + 0.2 * torch.rand(batch_size, 1, 1, 1, device=obs.device)
    imgs = imgs * contrast

    # Clamp to valid range
    imgs = torch.clamp(imgs, 0, 1)

    return imgs.view(batch_size, -1)


def train_policy(depth_obs: np.ndarray, actions: np.ndarray,
                 batch_size: int, lr: float, epochs: int, device: str,
                 use_augmentation: bool = True, dropout: float = 0.1,
                 weight_decay: float = 1e-3, encoder_type: str = "cnn",
                 apply_mask: bool = True):
    """Train depth BC policy via supervised learning.

    Args:
        depth_obs: Depth observations [N, H*W]
        actions: Expert actions [N, 4]
        batch_size: Training batch size
        lr: Learning rate
        epochs: Number of training epochs
        device: Device to train on
        use_augmentation: Whether to apply data augmentation
        dropout: Dropout rate for regularization
        weight_decay: Weight decay for optimizer
        encoder_type: "cnn" or "resnet"
        apply_mask: Whether to apply training-time masking of background
    """
    obs_dim = depth_obs.shape[1]
    action_dim = actions.shape[1]
    img_size = int(np.sqrt(obs_dim))

    print(f"\n[BC-Depth] Training {encoder_type.upper()} policy")
    print(f"  Samples: {len(depth_obs)}")
    print(f"  Obs dim: {obs_dim} ({img_size}x{img_size})")
    print(f"  Action dim: {action_dim}")
    print(f"  Encoder: {encoder_type}")
    print(f"  Augmentation: {use_augmentation}")
    print(f"  Training-time masking: {apply_mask}")
    print(f"  Dropout: {dropout}")
    print(f"  Weight decay: {weight_decay}")
    print("=" * 60)

    # Create dataset
    obs_t = torch.from_numpy(depth_obs).float().to(device)
    act_t = torch.from_numpy(actions).float().to(device)
    dataset = TensorDataset(obs_t, act_t)

    # Split - use 15% for validation to get better generalization signal
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Create model with selected encoder
    if encoder_type == "resnet":
        policy = BCResNetPolicy(obs_dim, action_dim, dropout=dropout).to(device)
    else:
        policy = BCCNNPolicy(obs_dim, action_dim, dropout=dropout).to(device)

    # Use AdamW with weight decay for regularization
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    # Use ReduceLROnPlateau for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    early_stop_patience = 10  # Stop earlier with stronger regularization

    for epoch in range(epochs):
        # Train with masking and augmentation
        policy.train()
        train_loss = 0.0
        for b_obs, b_act in train_loader:
            # Apply training-time masking first (if enabled)
            if apply_mask:
                b_obs = apply_mask_during_training(b_obs, img_size)

            # Then apply augmentation
            if use_augmentation:
                b_obs = augment_depth_batch(b_obs, img_size)

            optimizer.zero_grad()
            pred = policy(b_obs)
            loss = criterion(pred, b_act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate (with masking if enabled, no augmentation)
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_obs, b_act in val_loader:
                # Apply masking to validation too (consistent with training)
                if apply_mask:
                    b_obs = apply_mask_during_training(b_obs, img_size)

                pred = policy(b_obs)
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

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n[BC-Depth] Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
            break

    policy.load_state_dict(best_state)
    print(f"\n[BC-Depth] Best val loss: {best_val_loss:.6f}")

    return policy, best_val_loss


def save_policy(policy, output_dir: str, obs_dim: int, metadata: dict, encoder_type: str = "cnn"):
    """Save policy in BC and RSL-RL compatible formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Add encoder type to metadata
    metadata_with_encoder = metadata.copy()
    metadata_with_encoder['encoder_type'] = encoder_type

    # BC checkpoint
    torch.save({
        'model_state_dict': policy.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': 5,  # [x, y, z, cos(2*yaw), sin(2*yaw)]
        'img_size': policy.img_size,
        'metadata': metadata_with_encoder,
        'encoder_type': encoder_type,  # Also save at top level for easy access
    }, os.path.join(output_dir, "bc_depth_policy.pt"))

    # RSL-RL compatible checkpoint
    # Map BC weights to CNNActorCritic structure
    # BC: encoder.*, latent_proj.*, actor_mlp.*
    # RL: encoder.*, encoder_fc.*, actor.*, critic.*, std

    rsl_state = {}
    bc_state = policy.state_dict()

    # Copy encoder weights directly (same naming)
    for name, param in bc_state.items():
        if name.startswith("encoder."):
            rsl_state[name] = param.cpu().clone()

    # Map latent_proj -> encoder_fc
    for name, param in bc_state.items():
        if name.startswith("latent_proj."):
            new_name = name.replace("latent_proj.", "encoder_fc.")
            rsl_state[new_name] = param.cpu().clone()

    # Map actor_mlp -> actor
    for name, param in bc_state.items():
        if name.startswith("actor_mlp."):
            new_name = name.replace("actor_mlp.", "actor.")
            rsl_state[new_name] = param.cpu().clone()

    # Initialize critic with random weights (same structure as actor)
    for name, param in bc_state.items():
        if name.startswith("actor_mlp."):
            new_name = name.replace("actor_mlp.", "critic.")
            # Last layer outputs 1 for critic, 4 for actor
            if "4.weight" in name:  # Last layer weight
                rsl_state[new_name] = torch.randn(1, param.shape[1])
            elif "4.bias" in name:  # Last layer bias
                rsl_state[new_name] = torch.zeros(1)
            else:
                rsl_state[new_name] = torch.randn_like(param.cpu())

    # Add std for action noise (same as pose-based BC)
    rsl_state["std"] = torch.ones(4) * 0.3

    # Empty optimizer state (same as pose-based BC)
    # RSL-RL will create fresh optimizer

    torch.save({
        'model_state_dict': rsl_state,
        'optimizer_state_dict': {},  # Empty, same as pose-based BC
        'iter': 0,
        'infos': {
            'bc_pretrained': True,
            'obs_dim': obs_dim,
            'img_size': policy.img_size,
        },
    }, os.path.join(output_dir, "bc_depth_policy_rsl_rl.pt"))

    print(f"\n[BC-Depth] Saved to {output_dir}/")
    print(f"  - bc_depth_policy.pt (standalone)")
    print(f"  - bc_depth_policy_rsl_rl.pt (for RL fine-tuning)")


def main():
    # Handle resume mode
    if args_cli.resume:
        output_dir = args_cli.resume
        if not os.path.exists(output_dir):
            print(f"[BC-Depth] ERROR: Resume directory does not exist: {output_dir}")
            return
        print(f"[BC-Depth] Resuming from: {output_dir}")
    elif args_cli.output_dir:
        output_dir = args_cli.output_dir
    else:
        base_output_dir = os.path.abspath(os.path.join("logs", "bc_depth"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_output_dir, f"bc_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Create "latest" symlink
        latest_link = os.path.join(base_output_dir, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        try:
            os.symlink(f"bc_{timestamp}", latest_link)
        except OSError:
            pass
        print(f"[BC-Depth] Output: {output_dir}")
        print(f"[BC-Depth] (also linked as {latest_link})")

    # Train-only mode (no Isaac needed)
    if args_cli.train_only:
        print(f"[BC-Depth] Train-only mode from {args_cli.train_only}")
        depth_obs = np.load(os.path.join(args_cli.train_only, "depth_observations.npy"))
        actions = np.load(os.path.join(args_cli.train_only, "actions.npy"))
        with open(os.path.join(args_cli.train_only, "metadata.json")) as f:
            metadata = json.load(f)

        policy, best_loss = train_policy(
            depth_obs, actions,
            batch_size=args_cli.batch_size,
            lr=args_cli.lr,
            epochs=args_cli.epochs,
            device=args_cli.device,
            use_augmentation=args_cli.augmentation,
            dropout=args_cli.dropout,
            weight_decay=args_cli.weight_decay,
            encoder_type=args_cli.encoder_type,
            apply_mask=args_cli.apply_mask_during_training
        )

        save_policy(policy, output_dir, metadata["obs_dim"], metadata, encoder_type=args_cli.encoder_type)
        print(f"\n[BC-Depth] Done! Fine-tune with:")
        print(f"  ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \\")
        print(f"      --task Isaac-Crane-Depth-v0 --num_envs 16 --headless \\")
        print(f"      --checkpoint {output_dir}/bc_depth_policy_rsl_rl.pt")
        return

    # Full pipeline: collect + train
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "num_envs": args_cli.num_envs,
        "num_episodes": args_cli.num_episodes,
        "depth_height": args_cli.depth_height,
        "depth_width": args_cli.depth_width,
        "use_semantic_mask": args_cli.use_semantic_mask,
        "domain_randomization": args_cli.domain_randomization,
        "pos_noise_cm": args_cli.pos_noise,
        "yaw_noise_deg": args_cli.yaw_noise,
        "batch_size": args_cli.batch_size,
        "lr": args_cli.lr,
        "epochs": args_cli.epochs,
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
    cfg.camera_cfg.width = max(args_cli.depth_width, 256)
    cfg.camera_cfg.height = max(args_cli.depth_height, 256)
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    print(f"[BC-Depth] Domain randomization: {args_cli.domain_randomization}")
    env = CraneDirectEnvFull(cfg)
    print(f"[BC-Depth] Environment created with camera")

    # Collect
    depth_obs, actions, metrics = collect_demonstrations(
        env, args_cli.num_episodes, output_dir,
        depth_height=args_cli.depth_height,
        depth_width=args_cli.depth_width,
        use_semantic_mask=args_cli.use_semantic_mask,
        top_k=args_cli.top_k,
        pos_noise_cm=args_cli.pos_noise,
        yaw_noise_deg=args_cli.yaw_noise,
        save_interval=args_cli.save_interval
    )

    # Final save
    save_checkpoint(output_dir, list(depth_obs), list(actions),
                   metrics, args_cli.depth_height, args_cli.depth_width,
                   args_cli.use_semantic_mask)

    # Close env to free GPU memory (but NOT simulation_app yet - that kills Python!)
    env.close()

    # Clear CUDA cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if args_cli.collect_only:
        print(f"\n[BC-Depth] Collection complete! Data saved to {output_dir}")
        print(f"To train later:")
        print(f"  python train_bc_depth.py --train_only {output_dir} --epochs 100")
        simulation_app.close()
        return

    # Train
    print(f"\n[BC-Depth] Starting training...")
    policy, best_loss = train_policy(
        depth_obs, actions,
        batch_size=args_cli.batch_size,
        lr=args_cli.lr,
        epochs=args_cli.epochs,
        device=args_cli.device,
        use_augmentation=args_cli.augmentation,
        dropout=args_cli.dropout,
        weight_decay=args_cli.weight_decay,
        encoder_type=args_cli.encoder_type,
        apply_mask=args_cli.apply_mask_during_training
    )

    metadata = {
        "depth_height": args_cli.depth_height,
        "depth_width": args_cli.depth_width,
        "use_semantic_mask": args_cli.use_semantic_mask,
        "obs_dim": args_cli.depth_height * args_cli.depth_width,
        "action_dim": 5,  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    }
    save_policy(policy, output_dir, metadata["obs_dim"], metadata, encoder_type=args_cli.encoder_type)

    print(f"\n[BC-Depth] ====== COMPLETE ======")
    print(f"[BC-Depth] Output: {output_dir}")
    print(f"[BC-Depth]")
    print(f"[BC-Depth] To fine-tune with RL:")
    print(f"  ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \\")
    print(f"      --task Isaac-Crane-Depth-v0 --num_envs 16 --headless \\")
    print(f"      --checkpoint {output_dir}/bc_depth_policy_rsl_rl.pt")

    # Close simulation at the very end
    simulation_app.close()


if __name__ == "__main__":
    main()
