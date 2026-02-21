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
                                   depth_range: tuple = (1.0, 10.0)) -> torch.Tensor:
    """Get masked log point cloud in crane base frame."""
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


def main():
    # Load policy
    policy, num_points, metadata = load_bc_policy(args_cli.checkpoint, args_cli.device)

    print(f"[Play] Num points: {num_points}")

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

    env = CraneDirectEnvFull(cfg)
    print(f"[Play] Environment created with {env.num_envs} envs")
    print(f"[Play] Domain randomization: {args_cli.domain_randomization}")

    # Reset and initialize camera
    env.reset()
    env.sim.render()
    env._camera.update(dt=env.cfg.sim.dt)

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

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

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
                pc = get_log_pointcloud_base_frame(env, i, num_points)
                pc_batch.append(pc)
            obs = torch.stack(pc_batch)  # (num_envs, num_points, 3)

            # Get policy action
            actions = policy(obs)

            # Debug: print action stats occasionally
            if total_grasps < 5 or total_grasps % 100 == 0:
                print(f"[Debug] Grasp {total_grasps}: points={obs.shape}, "
                      f"action=[{actions.min():.2f},{actions.max():.2f}]")

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
                total_grasps += 1
                total_logs_grasped += logs_grasped
                episode_logs_cleared[i] += logs_grasped
                if logs_grasped > 0:
                    successful_grasps += 1
                    total_alignment += alignment
                    total_stability += stability
                else:
                    failed_grasps += 1

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

                    print(f"[Play] Episode {episodes_done}: reward={ep_reward:.2f}, "
                          f"cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs})")

                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    if has_variable_logs:
                        episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # Summary
    avg_clear_pct = sum(clearing_percentages) / max(1, len(clearing_percentages))
    grasp_success_rate = successful_grasps / max(1, total_grasps) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_alignment = total_alignment / max(1, successful_grasps)
    avg_stability = total_stability / max(1, successful_grasps)
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_reward = total_reward / max(1, episodes_done)

    print("=" * 60)
    print(f"\n[Play] ====== RESULTS ======")
    print(f"[Play] Episodes: {episodes_done}")
    print(f"[Play] Avg Episode Reward: {avg_reward:.2f}")
    print(f"[Play] Avg Pile Cleared: {avg_clear_pct:.1f}%")
    print(f"[Play] Full Clear Rate: {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
    print(f"[Play] Grasp Success Rate: {grasp_success_rate:.1f}%")
    print(f"[Play] Avg Throughput: {avg_throughput:.2f} logs/grasp")
    print(f"[Play] Avg Alignment: {avg_alignment:.3f}")
    print(f"[Play] Avg Stability: {avg_stability:.3f}")
    print(f"[Play] Avg Alignment: {avg_alignment:.3f}")
    print(f"[Play] Total Logs Grasped: {total_logs_grasped}")
    print(f"[Play] =======================")

    # Save metrics
    if args_cli.save_metrics:
        output_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

        metrics = {
            "eval_config": {
                "checkpoint": args_cli.checkpoint,
                "num_envs": args_cli.num_envs,
                "num_episodes": episodes_done,
                "domain_randomization": args_cli.domain_randomization,
                "num_points": num_points,
                "timestamp": timestamp,
            },
            "grasp": {
                "success_rate": grasp_success_rate,
                "successful": successful_grasps,
                "failed": failed_grasps,
                "total": total_grasps,
            },
            "pile_clearing": {
                "avg_cleared_pct": avg_clear_pct,
                "full_clear_rate": full_clear_rate,
                "full_clears": piles_fully_cleared,
            },
            "performance": {
                "avg_episode_reward": avg_reward,
                "avg_throughput": avg_throughput,
                "avg_alignment": avg_alignment,
                "avg_stability": avg_stability,
                "total_logs_grasped": total_logs_grasped,
            },
            "episode_rewards": episode_rewards_list,
            "clearing_percentages": clearing_percentages,
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Play] Metrics saved to: {metrics_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
