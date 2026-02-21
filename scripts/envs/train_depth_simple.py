#!/usr/bin/env python3
"""
Train crane grasping with depth observations using simple PPO.

No RSL-RL dependency - standalone implementation.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_depth_simple.py --num_envs 16 --enable_cameras
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.distributions import Normal

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Parse args before Isaac imports
parser = argparse.ArgumentParser(description="Train crane grasping with depth observations")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Maximum training iterations")
parser.add_argument("--save_interval", type=int, default=10, help="Save checkpoint every N iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/depth_train", help="Log directory")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

# PPO hyperparameters
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
parser.add_argument("--num_steps", type=int, default=4, help="Grasp cycles per update (each step = 1 full grasp cycle)")
parser.add_argument("--num_epochs", type=int, default=5, help="PPO epochs per update")
parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Now import environment
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull
from crane_depth_env import CraneDepthEnv, DepthObsConfig


class CNNEncoder(nn.Module):
    """CNN encoder for depth images."""
    def __init__(self, img_height=128, img_width=128, latent_dim=256):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width

        # CNN encoder for 128x128 input (higher res for log orientation detection)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
        )
        # Compute flatten dim: 128 * 8 * 8 = 8192
        self.flatten_dim = 128 * 8 * 8
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.img_height, self.img_width)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        x = torch.relu(self.fc(x))
        return x


class CNNActorCritic(nn.Module):
    """Actor-Critic with CNN encoder."""
    def __init__(self, img_height=128, img_width=128, act_dim=4, latent_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(img_height, img_width, latent_dim)

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        latent = self.encoder(obs)
        mean = self.actor(latent)
        value = self.critic(latent)
        return mean, value

    def act(self, obs, deterministic=False):
        mean, _ = self.forward(obs)
        if deterministic:
            return mean
        std = self.log_std.exp()
        dist = Normal(mean, std)
        return dist.sample()

    def evaluate(self, obs, actions):
        latent = self.encoder(obs)
        mean = self.actor(latent)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent)
        return value, log_probs, entropy


class RolloutBuffer:
    """Simple rollout buffer."""
    def __init__(self, num_envs, num_steps, obs_dim, act_dim, device):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs, act_dim), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)

        self.step = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.values[self.step] = value.squeeze(-1)
        self.log_probs[self.step] = log_prob
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, last_value, gamma, gae_lambda):
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

    def get_batches(self, batch_size):
        total = self.num_steps * self.num_envs
        indices = torch.randperm(total, device=self.device)

        obs_flat = self.obs.view(total, -1)
        actions_flat = self.actions.view(total, -1)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            idx = indices[start:end]
            yield idx, obs_flat[idx], actions_flat[idx]

    def clear(self):
        self.step = 0


def train():
    print(f"\n{'='*60}", flush=True)
    print(f"Depth RL Training (Simple PPO)", flush=True)
    print(f"{'='*60}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Environments: {args.num_envs}", flush=True)
    print(f"Log directory: {log_dir}", flush=True)

    # Create environment
    print("\n[Train] Creating environment...", flush=True)
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_camera = True
    cfg.use_hierarchical_rl = True  # Use policy actions for target selection!

    # Camera settings - 128x128 for log orientation detection (no downsampling needed)
    cfg.camera_cfg.width = 128
    cfg.camera_cfg.height = 128
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]  # Skip RGB

    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    env = CraneDepthEnv(base_env)

    obs_dim = env.obs_dim
    act_dim = env.action_space.shape[0]
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}", flush=True)

    # Create policy
    print("[Train] Creating CNN policy...", flush=True)
    policy = CNNActorCritic(env.cfg.height, env.cfg.width, act_dim).to(env.device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {total_params:,}", flush=True)

    # Resume if specified
    start_iter = 0
    if args.resume:
        print(f"[Train] Loading checkpoint: {args.resume}", flush=True)
        ckpt = torch.load(args.resume)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iteration", 0)

    # Rollout buffer
    buffer = RolloutBuffer(args.num_envs, args.num_steps, obs_dim, act_dim, env.device)

    # Training
    print("\n[Train] Starting training...", flush=True)
    obs, _ = env.reset()
    print(f"[Train] Initial obs shape: {obs.shape}", flush=True)

    total_steps = 0
    ep_rewards = []
    import time

    for iteration in range(start_iter, args.max_iterations):
        iter_start = time.time()
        collect_start = time.time()

        # Collect rollout
        policy.eval()
        rollout_rewards = []
        for step in range(args.num_steps):
            with torch.no_grad():
                action = policy.act(obs)
                _, value = policy.forward(obs)
                std = policy.log_std.exp()
                dist = Normal(policy.forward(obs)[0], std)
                log_prob = dist.log_prob(action).sum(dim=-1)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            buffer.add(obs, action, reward, done, value, log_prob)
            obs = next_obs
            total_steps += args.num_envs
            ep_rewards.append(reward.mean().item())
            rollout_rewards.append(reward.mean().item())

        collect_time = time.time() - collect_start
        print(f"[Iter {iteration}] Collection done: {args.num_steps} steps in {collect_time:.1f}s", flush=True)
        learn_start = time.time()

        # Compute returns
        with torch.no_grad():
            _, last_value = policy.forward(obs)
        returns, advantages = buffer.compute_returns(last_value, args.gamma, args.gae_lambda)

        # Normalize advantages
        adv_flat = advantages.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.view(args.num_steps, args.num_envs)
        returns_flat = returns.view(-1)
        log_probs_flat = buffer.log_probs.view(-1)

        # PPO update
        policy.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(args.num_epochs):
            for idx, batch_obs, batch_actions in buffer.get_batches(args.batch_size):
                batch_returns = returns_flat[idx]
                batch_advantages = adv_flat[idx]
                batch_old_log_probs = log_probs_flat[idx]

                values, log_probs, entropy = policy.evaluate(batch_obs, batch_actions)

                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values.squeeze(-1) - batch_returns) ** 2).mean()

                # Entropy
                entropy_loss = -entropy.mean()

                # Total
                loss = policy_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        learn_time = time.time() - learn_start
        iter_time = time.time() - iter_start

        buffer.clear()

        # Logging - every iteration for visibility
        mean_reward = np.mean(rollout_rewards) if rollout_rewards else 0
        mean_reward_100 = np.mean(ep_rewards[-100:]) if ep_rewards else 0
        avg_loss = total_loss / max(num_updates, 1)
        avg_policy_loss = total_policy_loss / max(num_updates, 1)
        avg_value_loss = total_value_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)
        std = policy.log_std.exp().mean().item()
        fps = (args.num_steps * args.num_envs) / iter_time

        # TensorBoard logging
        writer.add_scalar("Train/mean_reward", mean_reward_100, iteration)
        writer.add_scalar("Train/rollout_reward", mean_reward, iteration)
        writer.add_scalar("Loss/total", avg_loss, iteration)
        writer.add_scalar("Loss/policy", avg_policy_loss, iteration)
        writer.add_scalar("Loss/value", avg_value_loss, iteration)
        writer.add_scalar("Policy/entropy", avg_entropy, iteration)
        writer.add_scalar("Policy/std", std, iteration)
        writer.add_scalar("Perf/fps", fps, iteration)
        writer.add_scalar("Perf/collect_time", collect_time, iteration)
        writer.add_scalar("Perf/learn_time", learn_time, iteration)

        # Print every iteration (RSL-RL style) - make it stand out
        print(f"\n{'='*80}", flush=True)
        print(f">>> LEARNING ITERATION {iteration:4d} <<<", flush=True)
        print(f"    Reward: {mean_reward:6.2f} (avg100: {mean_reward_100:6.2f})", flush=True)
        print(f"    Policy Loss: {avg_policy_loss:7.4f} | Value Loss: {avg_value_loss:7.4f}", flush=True)
        print(f"    Std: {std:.3f} | Entropy: {avg_entropy:.3f}", flush=True)
        print(f"    FPS: {fps:5.0f} | Collect: {collect_time:.1f}s | Learn: {learn_time:.1f}s", flush=True)
        print(f"    Total steps: {total_steps}", flush=True)
        print(f"{'='*80}\n", flush=True)

        # Flush TensorBoard to ensure data is visible
        writer.flush()

        # Save checkpoint
        if iteration % args.save_interval == 0 and iteration > 0:
            ckpt_path = log_dir / f"model_{iteration:05d}.pt"
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}", flush=True)

    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": args.max_iterations,
    }, final_path)
    print(f"\n[Train] Done! Final model: {final_path}", flush=True)

    writer.close()
    env.close()


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
