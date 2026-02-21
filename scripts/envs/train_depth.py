#!/usr/bin/env python3
"""
Train crane grasping with depth observations using RSL-RL.

Uses CNN actor-critic with proper RSL-RL integration.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_depth.py --num_envs 16 --enable_cameras
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Parse args
parser = argparse.ArgumentParser(description="Train crane with depth observations")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=1500, help="Max iterations")
parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
parser.add_argument("--log_dir", type=str, default="logs/depth_train", help="Log directory")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull
from crane_depth_env import CraneDepthEnv, DepthObsConfig

# Import CNN actor-critic
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "source" / "crane_testbed" / "crane_testbed" / "agents"))
from cnn_actor_critic import CNNActorCritic


def train():
    """Main training function."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}", flush=True)
    print(f"Depth RL Training with RSL-RL PPO", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Environments: {args.num_envs}", flush=True)
    print(f"Max iterations: {args.max_iterations}", flush=True)

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Create environment
    print("[Train] Creating environment...", flush=True)
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_camera = True
    cfg.use_hierarchical_rl = True  # Policy controls target selection

    # Camera config - 128x128 for log orientation detection
    cfg.camera_cfg.width = 128
    cfg.camera_cfg.height = 128
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]

    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    env = CraneDepthEnv(base_env)

    num_obs = env.obs_dim
    num_actions = env.action_space.shape[0]
    print(f"  Observation dim: {num_obs}", flush=True)
    print(f"  Action dim: {num_actions}", flush=True)

    # Create CNN actor-critic
    print("[Train] Creating CNN Actor-Critic...", flush=True)
    actor_critic = CNNActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        img_height=128,  # Updated for 128x128 depth images
        img_width=128,
        encoder_features=256,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        init_noise_std=1.0,
    ).to(env.device)

    total_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"  Parameters: {total_params:,}", flush=True)

    # PPO settings
    num_steps_per_env = 4  # Same as 32-pose training
    num_learning_epochs = 5  # Same as 32-pose training
    num_mini_batches = 4
    clip_param = 0.2
    gamma = 0.99
    lam = 0.95
    value_loss_coef = 1.0
    entropy_coef = 0.01
    learning_rate = 3e-4
    max_grad_norm = 1.0

    # Create PPO
    print("[Train] Creating PPO algorithm...", flush=True)
    alg = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
        clip_param=clip_param,
        gamma=gamma,
        lam=lam,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        device=env.device,
    )
    print("[Train] PPO created.", flush=True)

    # Initialize storage
    print("[Train] Initializing storage...", flush=True)
    alg.init_storage(
        num_envs=env.num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=(num_obs,),
        critic_obs_shape=(num_obs,),
        action_shape=(num_actions,),
    )
    print("[Train] Storage initialized.", flush=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=str(log_dir))

    # Resume if specified
    start_iter = 0
    if args.resume:
        print(f"[Train] Loading checkpoint: {args.resume}", flush=True)
        loaded = torch.load(args.resume)
        actor_critic.load_state_dict(loaded["model_state_dict"])
        alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        start_iter = loaded.get("iteration", 0)

    # Training loop
    print("\n[Train] Starting training...", flush=True)
    obs, _ = env.reset()
    print(f"[Train] Initial obs shape: {obs.shape}", flush=True)

    total_timesteps = 0
    ep_rewards = []

    for iteration in range(start_iter, args.max_iterations):
        # Collect rollout
        actor_critic.eval()
        for step in range(num_steps_per_env):
            with torch.no_grad():
                actions = actor_critic.act(obs)
                values = actor_critic.evaluate(obs)
                actions_log_prob = actor_critic.get_actions_log_prob(actions)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            dones = terminated | truncated

            # Store transition
            alg.storage.add_transitions(
                observations=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                values=values.squeeze(-1),
                actions_log_prob=actions_log_prob,
            )

            obs = next_obs
            total_timesteps += env.num_envs
            ep_rewards.append(rewards.mean().item())

        # Compute returns
        with torch.no_grad():
            last_values = actor_critic.evaluate(obs).squeeze(-1)
        alg.storage.compute_returns(last_values)

        # PPO update
        actor_critic.train()
        mean_value_loss, mean_surrogate_loss = alg.update()
        alg.storage.clear()

        # Logging
        mean_reward = np.mean(ep_rewards[-100:]) if ep_rewards else 0
        writer.add_scalar("Loss/value", mean_value_loss, iteration)
        writer.add_scalar("Loss/surrogate", mean_surrogate_loss, iteration)
        writer.add_scalar("Train/mean_reward", mean_reward, iteration)
        writer.add_scalar("Train/timesteps", total_timesteps, iteration)

        if iteration % 10 == 0:
            print(f"[Iter {iteration:4d}] Steps: {total_timesteps:8d} | "
                  f"Reward: {mean_reward:.3f} | "
                  f"Value Loss: {mean_value_loss:.4f} | "
                  f"Policy Loss: {mean_surrogate_loss:.4f}", flush=True)

        # Save checkpoint
        if iteration % 100 == 0 and iteration > 0:
            ckpt_path = log_dir / f"model_{iteration:05d}.pt"
            torch.save({
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": alg.optimizer.state_dict(),
                "iteration": iteration,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}", flush=True)

    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save({
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": alg.optimizer.state_dict(),
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
