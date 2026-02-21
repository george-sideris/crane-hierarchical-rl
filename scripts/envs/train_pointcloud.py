#!/usr/bin/env python3
"""
Train crane grasping with point cloud observations.

This script trains a PointNet-based policy to grasp logs using
point cloud observations of the log pile.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_pointcloud.py --num_envs 16 --enable_cameras
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add paths
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before Isaac imports
parser = argparse.ArgumentParser(description="Train crane grasping with point cloud observations")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
parser.add_argument("--num_points", type=int, default=512, help="Number of points in observation")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/pointcloud_train", help="Log directory")
parser.add_argument("--save_interval", type=int, default=100, help="Save model every N iterations")
parser.add_argument("--eval_interval", type=int, default=50, help="Evaluate every N iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

# PPO hyperparameters
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
parser.add_argument("--num_steps", type=int, default=24, help="Steps per environment per update")
parser.add_argument("--num_epochs", type=int, default=4, help="PPO epochs per update")
parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac and custom modules
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull
from crane_pointcloud_env import CranePointCloudEnv, PointCloudObsConfig
from pointcloud_actor_critic import PointNetActorCritic


class PPOBuffer:
    """Rollout buffer for PPO."""

    def __init__(self, num_envs: int, num_steps: int, obs_dim: int, action_dim: int, device: str):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        # Storage
        self.observations = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs, action_dim), device=device)

        self.step = 0

    def insert(self, obs, action, reward, done, value, log_prob):
        """Insert one step of data."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.squeeze(-1)
        self.log_probs[self.step] = log_prob
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """Compute returns and advantages using GAE."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value.squeeze(-1)
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae

        returns = advantages + self.values
        return returns, advantages

    def get_batches(self, returns, advantages, batch_size):
        """Generate random mini-batches."""
        total_size = self.num_steps * self.num_envs
        indices = torch.randperm(total_size, device=self.device)

        # Flatten data
        obs_flat = self.observations.view(total_size, -1)
        actions_flat = self.actions.view(total_size, -1)
        returns_flat = returns.view(total_size)
        advantages_flat = advantages.view(total_size)
        log_probs_flat = self.log_probs.view(total_size, -1)

        for start in range(0, total_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                obs_flat[batch_indices],
                actions_flat[batch_indices],
                returns_flat[batch_indices],
                advantages_flat[batch_indices],
                log_probs_flat[batch_indices],
            )


def train():
    """Main training function."""
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"\n{'='*60}")
    print(f"Point Cloud RL Training")
    print(f"{'='*60}")
    print(f"Environments: {args.num_envs}")
    print(f"Points per observation: {args.num_points}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Create base environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_camera = True

    print("[Train] Creating base environment...")
    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")

    # Wrap with point cloud observations
    obs_config = PointCloudObsConfig(
        num_points=args.num_points,
        depth_range=(1.5, 8.0),
        normalize_points=True,
    )
    env = CranePointCloudEnv(base_env, obs_config)

    print(f"[Train] Environment created:")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action dim: {env.action_space.shape[0]}")

    # Calculate dimensions
    action_dim = env.action_space.shape[0]
    proprio_dim = env.proprio_dim

    # Create actor-critic network
    print("[Train] Creating PointNet actor-critic...")
    policy = PointNetActorCritic(
        num_points=args.num_points,
        proprio_dim=proprio_dim,
        num_actions=action_dim,
        pointnet_latent=64,
        proprio_latent=32,
        actor_hidden=[128, 64],
        critic_hidden=[128, 64],
        init_noise_std=1.0,
    ).to(env.device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Resume from checkpoint if specified
    start_iteration = 0
    if args.resume:
        print(f"[Train] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        policy.load_state_dict(checkpoint['policy'])
        start_iteration = checkpoint.get('iteration', 0)
        print(f"  Resuming from iteration {start_iteration}")

    # Optimizer
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # Rollout buffer
    buffer = PPOBuffer(
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        obs_dim=env.obs_dim,
        action_dim=action_dim,
        device=env.device,
    )

    # Training loop
    print("\n[Train] Starting training...")
    obs, _ = env.reset()
    total_steps = 0
    episode_rewards = []
    episode_lengths = []

    for iteration in range(start_iteration, args.max_iterations):
        # Collect rollout
        policy.eval()
        for step in range(args.num_steps):
            with torch.no_grad():
                action = policy.act(obs)
                value = policy.get_value(obs)
                _, log_probs, _ = policy.evaluate(obs, action)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            # Store
            buffer.insert(obs, action, reward, done.float(), value, log_probs)

            obs = next_obs
            total_steps += args.num_envs

            # Track episode stats
            if 'episode_return' in info:
                episode_rewards.extend(info['episode_return'])
            if 'episode_length' in info:
                episode_lengths.extend(info['episode_length'])

        # Compute returns
        with torch.no_grad():
            last_value = policy.get_value(obs)
        returns, advantages = buffer.compute_returns(last_value, args.gamma, args.gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        policy.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(args.num_epochs):
            for batch in buffer.get_batches(returns, advantages, args.batch_size):
                batch_obs, batch_actions, batch_returns, batch_advantages, batch_old_log_probs = batch

                # Evaluate actions
                values, log_probs, entropy = policy.evaluate(batch_obs, batch_actions)

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs.sum(-1) - batch_old_log_probs.sum(-1))
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values.squeeze(-1) - batch_returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Logging
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates

        writer.add_scalar("Loss/policy", avg_policy_loss, iteration)
        writer.add_scalar("Loss/value", avg_value_loss, iteration)
        writer.add_scalar("Loss/entropy", avg_entropy, iteration)
        writer.add_scalar("Train/total_steps", total_steps, iteration)

        if episode_rewards:
            mean_reward = np.mean(episode_rewards[-100:])
            writer.add_scalar("Train/mean_reward", mean_reward, iteration)

        # Print progress
        if iteration % 10 == 0:
            reward_str = f"{np.mean(episode_rewards[-100:]):.2f}" if episode_rewards else "N/A"
            print(f"[Iter {iteration:4d}] Steps: {total_steps:8d} | "
                  f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
                  f"Mean Reward: {reward_str}")

        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint_path = log_dir / f"checkpoint_{iteration:05d}.pt"
            torch.save({
                'iteration': iteration,
                'policy': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Final save
    final_path = log_dir / "final_model.pt"
    torch.save({
        'iteration': args.max_iterations,
        'policy': policy.state_dict(),
    }, final_path)
    print(f"\n[Train] Training complete! Final model saved to {final_path}")

    # Cleanup
    writer.close()
    env.close()


if __name__ == "__main__":
    train()
    simulation_app.close()
