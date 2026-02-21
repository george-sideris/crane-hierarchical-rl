#!/usr/bin/env python3
"""
Train crane grasping with depth observations using RSL-RL.

Uses CNN encoder for depth image observations with PPO.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/train_depth_rsl.py --num_envs 16 --enable_cameras
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Parse args before Isaac imports
parser = argparse.ArgumentParser(description="Train crane grasping with depth observations")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/depth_train", help="Log directory")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
import numpy as np
from torch.distributions import Normal

from rsl_rl.algorithms import PPO

from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull
from crane_depth_env import CraneDepthEnv, DepthObsConfig


class CNNActorCritic(nn.Module):
    """Actor-Critic with CNN encoder for depth observations.

    Compatible with RSL-RL's ActorCritic interface.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        img_height: int = 48,
        img_width: int = 48,
        latent_dim: int = 256,
        actor_hidden: list = [128, 64],
        critic_hidden: list = [128, 64],
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        # CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 12 -> 10
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 10 -> 8
            nn.ReLU(),
        )
        self.flatten_dim = 64 * 8 * 8
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, latent_dim),
            nn.ELU(),
        )

        # Actor
        actor_layers = []
        in_dim = latent_dim
        for h_dim in actor_hidden:
            actor_layers.extend([nn.Linear(in_dim, h_dim), nn.ELU()])
            in_dim = h_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = []
        in_dim = latent_dim
        for h_dim in critic_hidden:
            critic_layers.extend([nn.Linear(in_dim, h_dim), nn.ELU()])
            in_dim = h_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Disable args for RSL-RL compatibility
        Normal.set_default_validate_args(False)

    def _encode(self, obs):
        """Encode observation through CNN."""
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_height, self.img_width)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.fc_encoder(x)
        return x

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        latent = self._encode(observations)
        mean = self.actor(latent)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        latent = self._encode(observations)
        return self.actor(latent)

    def evaluate(self, critic_observations, **kwargs):
        latent = self._encode(critic_observations)
        return self.critic(latent)


class DepthEnvWrapper:
    """Wrapper to make CraneDepthEnv compatible with RSL-RL runner."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env.obs_dim
        self.num_actions = env.action_space.shape[0]
        self.max_episode_length = 1000  # Adjust as needed

        # RSL-RL expects these
        self.num_privileged_obs = None
        self.obs_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)
        self.extras = {}

    def reset(self):
        obs, info = self.env.reset()
        self.obs_buf = obs
        self.episode_length_buf.zero_()
        return obs, None  # obs, privileged_obs

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)

        self.obs_buf = obs
        self.rew_buf = reward
        self.reset_buf = (terminated | truncated).float()
        self.episode_length_buf += 1

        # Track episode stats
        dones = terminated | truncated
        if dones.any():
            self.extras["episode"] = {}
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            self.extras["episode"]["lengths"] = self.episode_length_buf[done_indices].cpu().numpy()

        return obs, None, reward, (terminated | truncated), info

    def get_observations(self):
        return self.obs_buf, None


def train():
    """Main training function."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Depth RL Training with RSL-RL")
    print(f"{'='*60}")
    print(f"Environments: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*60}\n")

    # Create base environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_camera = True

    print("[Train] Creating environments...")
    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    depth_env = CraneDepthEnv(base_env)
    env = DepthEnvWrapper(depth_env)

    print(f"  Observation dim: {env.num_obs}")
    print(f"  Action dim: {env.num_actions}")

    # Create actor-critic
    print("[Train] Creating CNN Actor-Critic...")
    actor_critic = CNNActorCritic(
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        img_height=depth_env.cfg.height,
        img_width=depth_env.cfg.width,
    ).to(env.device)

    total_params = sum(p.numel() for p in actor_critic.parameters())
    print(f"  Parameters: {total_params:,}", flush=True)

    # PPO config
    num_steps_per_env = 24

    # PPO algorithm
    print("[Train] Creating PPO...", flush=True)
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=4,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=3e-4,
        max_grad_norm=1.0,
        device=env.device,
    )
    print("[Train] Initializing storage...", flush=True)
    ppo.init_storage(
        num_envs=env.num_envs,
        num_transitions_per_env=num_steps_per_env,
        actor_obs_shape=(env.num_obs,),
        critic_obs_shape=(env.num_obs,),
        action_shape=(env.num_actions,),
    )
    print("[Train] PPO ready.", flush=True)

    # Load checkpoint if resuming
    if args.resume:
        print(f"[Train] Loading checkpoint: {args.resume}", flush=True)
        loaded = torch.load(args.resume)
        actor_critic.load_state_dict(loaded["model_state_dict"])
        ppo.optimizer.load_state_dict(loaded["optimizer_state_dict"])

    # Training loop
    print("\n[Train] Starting training...", flush=True)
    print("[Train] Resetting environment...", flush=True)
    obs, _ = env.reset()
    print(f"[Train] Got initial obs: {obs.shape}", flush=True)

    tot_timesteps = 0
    ep_rewards = []

    for iteration in range(args.max_iterations):
        if iteration == 0:
            print("[Train] Entering training loop...", flush=True)
        # Collect rollouts
        actor_critic.eval()
        for step in range(num_steps_per_env):
            with torch.no_grad():
                actions = actor_critic.act(obs)
                values = actor_critic.evaluate(obs)
                actions_log_prob = actor_critic.get_actions_log_prob(actions)

            obs_next, _, rewards, dones, infos = env.step(actions)

            # Store transition
            ppo.storage.add_transitions(
                observations=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                values=values.squeeze(-1),
                actions_log_prob=actions_log_prob,
            )

            obs = obs_next
            tot_timesteps += env.num_envs
            ep_rewards.append(rewards.mean().item())

            # Handle resets
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                obs_reset, _ = env.reset()
                obs = obs_reset

        # Compute returns
        with torch.no_grad():
            last_values = actor_critic.evaluate(obs).squeeze(-1)
        ppo.storage.compute_returns(last_values)

        # Update policy
        actor_critic.train()
        mean_value_loss, mean_surrogate_loss = ppo.update()
        ppo.storage.clear()

        # Logging
        if iteration % 10 == 0:
            mean_reward = np.mean(ep_rewards[-100:]) if ep_rewards else 0
            print(f"[Iter {iteration:4d}] Steps: {tot_timesteps:8d} | "
                  f"Reward: {mean_reward:.3f} | "
                  f"Value Loss: {mean_value_loss:.4f} | "
                  f"Policy Loss: {mean_surrogate_loss:.4f}", flush=True)

        # Save checkpoint
        if iteration % 100 == 0 and iteration > 0:
            ckpt_path = log_dir / f"model_{iteration:05d}.pt"
            torch.save({
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": ppo.optimizer.state_dict(),
                "iteration": iteration,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = log_dir / "model_final.pt"
    torch.save({
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": ppo.optimizer.state_dict(),
        "iteration": args.max_iterations,
    }, final_path)
    print(f"\n[Train] Done! Final model: {final_path}")

    depth_env.close()


if __name__ == "__main__":
    try:
        train()
    finally:
        simulation_app.close()
