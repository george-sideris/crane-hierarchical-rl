#!/usr/bin/env python3
"""
Behavioral Cloning with Point Cloud observations + Cos/Sin yaw encoding.

Collects (log_pointcloud, expert_action) pairs and trains a PointNet policy.
Uses masked point cloud (logs only) in base frame.
5D action: [x, y, z, cos(2*yaw), sin(2*yaw)] - handles log symmetry automatically.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_pointcloud_cossin.py \
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
# Training options
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
parser.add_argument("--norm_type", type=str, default="batchnorm", choices=["batchnorm", "layernorm"],
                    help="Normalization type for PointNet encoder (default: batchnorm for compat)")
# Collection options
parser.add_argument("--save_interval", type=int, default=20, help="Save checkpoint every N episodes")
parser.add_argument("--collect_only", action="store_true", help="Only collect, skip training")
parser.add_argument("--train_only", type=str, default=None, help="Skip collection, train from this data dir")
args_cli, _ = parser.parse_known_args()

# Skip Isaac imports if train_only mode
if args_cli.train_only is None:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
    simulation_app = app_launcher.app

    # Import environment after app launch
    from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class PointNetEncoder(nn.Module):
    """PointNet encoder for point cloud feature extraction.

    Args:
        input_dim: Per-point feature dimension (default 3 for XYZ).
        output_dim: Output latent dimension.
        norm_type: "layernorm" (recommended for RL) or "batchnorm" (legacy).
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 256, norm_type: str = "batchnorm"):
        super().__init__()

        norm_cls = nn.LayerNorm if norm_type == "layernorm" else nn.BatchNorm1d

        # Shared MLP (applied per-point)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            norm_cls(64),
            nn.ELU(),
            nn.Linear(64, 128),
            norm_cls(128),
            nn.ELU(),
            nn.Linear(128, 256),
            norm_cls(256),
            nn.ELU(),
        )

        # After max-pooling, project to output dimension
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            norm_cls(output_dim),
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

        # Reshape to (batch * points, features) for per-point MLP
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
    """PointNet policy for BC, outputs 5D action [x, y, z, cos(2*yaw), sin(2*yaw)]."""

    def __init__(self, num_points: int = 1024, action_dim: int = 5,
                 latent_dim: int = 256, dropout: float = 0.2, norm_type: str = "batchnorm"):
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim

        # PointNet encoder
        self.encoder = PointNetEncoder(input_dim=3, output_dim=latent_dim, norm_type=norm_type)

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
    """Expert that computes 5D actions [x, y, z, cos(2*yaw), sin(2*yaw)] using env's methods."""

    def __init__(self, env, pos_noise_cm: float = 0.0, yaw_noise_deg: float = 0.0):
        self.env = env
        self.pos_noise = pos_noise_cm / 100.0
        self.yaw_noise = np.radians(yaw_noise_deg)

    def get_action(self, env_i: int) -> torch.Tensor:
        """Get 5D expert action [x, y, z, cos(2*yaw), sin(2*yaw)] for given environment."""
        env = self.env

        # Use env's existing method to get target position (highest log)
        log_pos_b, selected_id, log_quat_w = env._target_top_log_center_b(env_i)

        if selected_id == -1:
            return torch.zeros(5, device=env.device)

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
        yaw_target = env._get_target_grapple_yaw_b(env_i)
        if self.yaw_noise > 0:
            yaw_target += np.random.normal(0, self.yaw_noise)

        # Get action bounds
        if not env._action_bounds_valid[env_i]:
            env._compute_action_space_bounds()

        min_b = env._action_bounds_min[env_i]
        max_b = env._action_bounds_max[env_i]

        # Normalize to [-1, 1] then arctanh for XYZ
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
        yaw_cos = np.cos(2.0 * yaw_target)
        yaw_sin = np.sin(2.0 * yaw_target)

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


def get_log_pointcloud_base_frame(env, env_idx: int, num_points: int,
                                   depth_range: tuple = (1.0, 10.0)) -> torch.Tensor:
    """
    Get masked log point cloud in crane base frame.

    Args:
        env: Environment
        env_idx: Environment index
        num_points: Number of points to return (via FPS)
        depth_range: (min, max) depth range

    Returns:
        (num_points, 3) point cloud in base frame
    """
    # Get log point cloud in world frame
    pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)

    if pc_world.shape[0] == 0:
        return torch.zeros((num_points, 3), device=env.device)

    # Transform to base frame
    # Get crane base pose in world frame
    base_pos_w = env.crane.data.root_pos_w[env_idx]  # (3,)
    base_quat_w = env.crane.data.root_quat_w[env_idx]  # (4,) wxyz

    # Translate to base origin
    pc_translated = pc_world - base_pos_w

    # Rotate by inverse of base orientation
    # quat_wxyz to rotation matrix, then inverse (transpose for rotation)
    w, x, y, z = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]

    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])  # (3, 3)

    # Apply inverse rotation (R.T @ points.T).T = points @ R
    pc_base = pc_translated @ R

    # FPS to fixed number of points
    pc_sampled = farthest_point_sampling(pc_base, num_points)

    return pc_sampled


def collect_demonstrations(env, num_episodes: int, output_dir: str,
                           num_points: int, depth_range: tuple,
                           pos_noise_cm: float, yaw_noise_deg: float,
                           save_interval: int = 20):
    """Collect (pointcloud, action) pairs from expert."""
    device = env.device

    pointclouds = []
    actions = []
    episode_rewards = []
    current_reward = torch.zeros(env.num_envs, device=device)

    # Metrics
    successful_grasps = 0
    failed_grasps = 0
    total_logs_grasped = 0
    total_alignment = 0.0
    total_stability = 0.0
    piles_fully_cleared = 0
    clearing_percentages = []
    logs_per_episode = []
    knocked_off_per_episode = []
    cycles_per_episode = []
    episodes_done = 0

    episode_logs_cleared = torch.zeros(env.num_envs, device=device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=device, dtype=torch.int32)

    expert = PointCloudExpert(env, pos_noise_cm=pos_noise_cm, yaw_noise_deg=yaw_noise_deg)

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
            for i in range(env.num_envs):
                pc = get_log_pointcloud_base_frame(env, i, num_points, depth_range)
                pc_batch.append(pc.cpu().numpy())
            pc_batch = np.stack(pc_batch)  # (num_envs, num_points, 3)

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
                stability_this_grasp = env._prev_grasp_stability[env_i].item() if hasattr(env, '_prev_grasp_stability') else 0.0

                episode_logs_cleared[env_i] += logs_this_grasp

                if logs_this_grasp > 0:  # Successful grasp
                    pointclouds.append(pc_batch[env_i])
                    actions.append(action_batch[env_i])
                    successful_grasps += 1
                    total_logs_grasped += logs_this_grasp
                    total_alignment += alignment_this_grasp
                    total_stability += stability_this_grasp
                else:
                    failed_grasps += 1

            current_reward += rew
            done = terminated | truncated

            for env_i in range(env.num_envs):
                if done[env_i]:
                    episode_rewards.append(current_reward[env_i].item())

                    total_logs_this_env = int(episode_starting_logs[env_i].item())
                    logs_cleared = int(episode_logs_cleared[env_i].item())
                    clear_pct = (logs_cleared / max(1, total_logs_this_env)) * 100
                    clearing_percentages.append(clear_pct)
                    logs_per_episode.append(total_logs_this_env)

                    # Track knocked-off logs and cycles for this episode
                    if hasattr(env, '_final_episode_knocked_off'):
                        knocked_off = int(env._final_episode_knocked_off[env_i].item())
                    elif hasattr(env, '_logs_knocked_off'):
                        knocked_off = int(env._logs_knocked_off[env_i].item())
                    else:
                        knocked_off = 0
                    cycles = int(env._cycle_count[env_i].item()) if hasattr(env, '_cycle_count') else 0
                    knocked_off_per_episode.append(knocked_off)
                    cycles_per_episode.append(cycles)

                    if logs_cleared >= total_logs_this_env:
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
                        }, num_points)

                    if episodes_done >= num_episodes:
                        break

    # Summary
    grasp_rate = successful_grasps / max(1, successful_grasps + failed_grasps) * 100
    avg_clear_pct = np.mean(clearing_percentages) if clearing_percentages else 0.0
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_align = total_alignment / max(1, successful_grasps)
    avg_stability = total_stability / max(1, successful_grasps)
    avg_knocked_off = np.mean(knocked_off_per_episode) if knocked_off_per_episode else 0.0
    avg_cycles = np.mean(cycles_per_episode) if cycles_per_episode else 0.0

    print(f"\n[BC-PointCloud] ====== COLLECTION SUMMARY ======")
    print(f"  Episodes: {episodes_done}")
    print(f"  Samples collected: {len(pointclouds)}")
    print(f"  Grasp success rate: {grasp_rate:.1f}%")
    print(f"  Avg pile cleared: {avg_clear_pct:.1f}%")
    print(f"  Full clears: {full_clear_rate:.1f}%")
    print(f"  Avg throughput: {avg_throughput:.2f} logs/grasp")
    print(f"  Avg alignment: {avg_align:.3f}")
    print(f"  Avg stability: {avg_stability:.3f}")
    print(f"  Avg knocked off: {avg_knocked_off:.1f} logs/episode")
    print(f"  Avg cycles/episode: {avg_cycles:.1f}")
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
            "avg_stability": avg_stability,
            "avg_knocked_off": avg_knocked_off,
            "avg_cycles_per_episode": avg_cycles,
        },
        "episode_rewards": episode_rewards,
        "clearing_percentages": clearing_percentages,
        "knocked_off_per_episode": knocked_off_per_episode,
        "cycles_per_episode": cycles_per_episode,
    }

    return np.stack(pointclouds), np.stack(actions), metrics


def save_checkpoint(output_dir, pointclouds, actions, metrics, num_points):
    """Save collection checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "pointclouds.npy"), np.stack(pointclouds))
    np.save(os.path.join(output_dir, "actions.npy"), np.stack(actions))
    np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(metrics.get("episode_rewards", [])))

    metadata = {
        "num_points": num_points,
        "obs_dim": num_points * 3,
        "action_dim": 5,
        "num_samples": len(pointclouds),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[BC-PointCloud] Checkpoint saved: {len(pointclouds)} samples")


def train_policy(pointclouds: np.ndarray, actions: np.ndarray,
                 num_points: int, batch_size: int, lr: float,
                 epochs: int, device: str, dropout: float = 0.2,
                 norm_type: str = "batchnorm"):
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
    policy = BCPointNetPolicy(num_points=num_points, action_dim=5, dropout=dropout, norm_type=norm_type).to(device)

    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()

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
            pred = policy(b_pc)
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

    # Add std for action noise (0.1 preserves BC mean during RL fine-tuning)
    rsl_state["std"] = torch.ones(5) * 0.1

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
        pointclouds = np.load(os.path.join(args_cli.train_only, "pointclouds.npy"))
        actions = np.load(os.path.join(args_cli.train_only, "actions.npy"))
        with open(os.path.join(args_cli.train_only, "metadata.json")) as f:
            metadata = json.load(f)

        num_points = metadata["num_points"]

        # Create output dir
        if args_cli.output_dir:
            output_dir = args_cli.output_dir
        else:
            base_output_dir = os.path.abspath(os.path.join("logs", "bc_pointcloud"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(base_output_dir, f"bc_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        policy, best_loss = train_policy(
            pointclouds, actions, num_points,
            batch_size=args_cli.batch_size,
            lr=args_cli.lr,
            epochs=args_cli.epochs,
            device=args_cli.device,
            dropout=args_cli.dropout,
            norm_type=args_cli.norm_type
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
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    print(f"[BC-PointCloud] Domain randomization: {args_cli.domain_randomization}")
    env = CraneDirectEnvFull(cfg)
    print(f"[BC-PointCloud] Environment created with camera")

    # Collect
    depth_range = (args_cli.depth_range_min, args_cli.depth_range_max)
    pointclouds, actions, metrics = collect_demonstrations(
        env, args_cli.num_episodes, output_dir,
        num_points=args_cli.num_points,
        depth_range=depth_range,
        pos_noise_cm=args_cli.pos_noise,
        yaw_noise_deg=args_cli.yaw_noise,
        save_interval=args_cli.save_interval
    )

    # Final save
    save_checkpoint(output_dir, list(pointclouds), list(actions), metrics, args_cli.num_points)

    # Save full metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Close env and simulator to free GPU memory before training
    env.close()
    simulation_app.close()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if args_cli.collect_only:
        print(f"\n[BC-PointCloud] Collection complete! Data saved to {output_dir}")
        print(f"To train later:")
        print(f"  python train_bc_pointcloud.py --train_only {output_dir} --epochs 100")
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
        norm_type=args_cli.norm_type
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
    print(f"[BC-PointCloud] To fine-tune with RL (5D CosSin):")
    print(f"  ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \\")
    print(f"      --task Isaac-Crane-Full-CosSin-MR-v0 --num_envs 4 --headless \\")
    print(f"      --load_checkpoint {output_dir}/bc_pointcloud_policy_rsl_rl.pt")


if __name__ == "__main__":
    main()
