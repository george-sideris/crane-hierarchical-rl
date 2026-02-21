#!/usr/bin/env python3
"""
Crane environment with depth image observations and CNN encoder.

Wrapper around the base crane environment that uses masked depth
images as observations for visual RL with a CNN-based policy.

Architecture:
    Depth Image (48x48) -> CNN Encoder -> Latent (256) -> Actor/Critic MLP

CNN Encoder:
    Conv2d(1, 32, 4, stride=2)   # 48 -> 24
    Conv2d(32, 64, 4, stride=2)  # 24 -> 12
    Conv2d(64, 64, 3, stride=1)  # 12 -> 10
    Conv2d(64, 64, 3, stride=1)  # 10 -> 8
    Flatten -> Linear(4096, 256)

Usage:
    # Test the environment
    ./isaaclab.sh -p crane_testbed/scripts/envs/crane_depth_env.py --num_envs 4 --enable_cameras

    # Train with CNN policy
    ./isaaclab.sh -p crane_testbed/scripts/envs/crane_depth_env.py --num_envs 16 --enable_cameras --train
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class DepthObsConfig:
    """Configuration for depth observations."""
    height: int = 128  # Higher res for log orientation/edge detection
    width: int = 128   # Needed for yaw control
    depth_min: float = 1.5  # Min depth (meters)
    depth_max: float = 8.0  # Max depth (meters)
    use_semantic_mask: bool = True  # Mask to logs only


class CraneDepthEnv:
    """Wrapper that provides depth image observations.

    Observation: Flattened masked depth image (height * width,)
    Action: Same as base env (target_x for grasp)
    """

    def __init__(self, base_env, cfg: DepthObsConfig = None):
        self.env = base_env
        self.cfg = cfg or DepthObsConfig()

        # Ensure camera is enabled
        if not hasattr(self.env, '_camera') or self.env._camera is None:
            raise RuntimeError("Camera must be enabled (enable_camera=True)")

        self.obs_dim = self.cfg.height * self.cfg.width
        print(f"[DepthEnv] Observation: {self.cfg.height}x{self.cfg.width} = {self.obs_dim} dims")

    def _get_depth_obs(self) -> torch.Tensor:
        """Get masked depth observation for all envs."""
        # Update camera (render already happened, just update sensor data)
        self.env._camera.update(dt=0.0)  # dt=0 skips physics, just captures

        # Get depth (num_envs, H, W, 1)
        depth = self.env._camera.data.output["depth"].squeeze(-1)  # (num_envs, H, W)

        # Get semantic mask if enabled
        if self.cfg.use_semantic_mask and "semantic_segmentation" in self.env.cfg.camera_cfg.data_types:
            sem_seg = self.env._camera.data.output["semantic_segmentation"]
            if sem_seg.dim() == 4:
                sem_ids = sem_seg[..., 0]  # (num_envs, H, W)
            else:
                sem_ids = sem_seg
            # Mask: keep only log pixels (semantic_id > 0)
            mask = (sem_ids > 0)
            depth = torch.where(mask, depth, torch.tensor(self.cfg.depth_max, device=depth.device))

        # Clip depth range
        depth = torch.clamp(depth, self.cfg.depth_min, self.cfg.depth_max)

        # Normalize to [0, 1]
        depth = (depth - self.cfg.depth_min) / (self.cfg.depth_max - self.cfg.depth_min)

        # Downsample to observation size
        # (num_envs, H, W) -> (num_envs, 1, H, W) for interpolate
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(self.cfg.height, self.cfg.width), mode='bilinear', align_corners=False)
        depth = depth.squeeze(1)  # (num_envs, obs_h, obs_w)

        # Flatten
        obs = depth.view(self.env.num_envs, -1)  # (num_envs, obs_dim)

        return obs

    def reset(self):
        """Reset and return depth observation."""
        base_obs, info = self.env.reset()
        self.env.sim.render()
        obs = self._get_depth_obs()
        return obs, info

    def step(self, action):
        """Step and return depth observation."""
        base_obs, reward, terminated, truncated, info = self.env.step(action)
        self.env.sim.render()
        obs = self._get_depth_obs()
        return obs, reward, terminated, truncated, info

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def observation_space(self):
        import gymnasium as gym
        return gym.spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()


# ============ Main: Test or Train ============
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--max_iterations", type=int, default=500)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import after app launch
    from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull

    # Create base env with camera
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args.num_envs
    cfg.enable_camera = True

    print(f"\n[DepthEnv] Creating environment with {args.num_envs} envs...")
    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")
    env = CraneDepthEnv(base_env)

    if args.train:
        # ============ Simple PPO Training with CNN ============
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Normal

        print(f"\n[Train] Starting training for {args.max_iterations} iterations...")

        # CNN encoder for depth images
        class CNNEncoder(nn.Module):
            """CNN encoder for 48x48 depth images."""
            def __init__(self, img_height=48, img_width=48, latent_dim=256):
                super().__init__()
                self.img_height = img_height
                self.img_width = img_width

                # Conv layers: 48x48 -> 23x23 -> 10x10 -> 8x8
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 48 -> 24
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 24 -> 12
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 12 -> 10
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 10 -> 8
                    nn.ReLU(),
                )
                # Calculate flattened size: 64 * 8 * 8 = 4096
                self.flatten_dim = 64 * 8 * 8
                self.fc = nn.Linear(self.flatten_dim, latent_dim)

            def forward(self, x):
                # x: (batch, obs_dim) flattened depth image
                batch_size = x.shape[0]
                # Reshape to image: (batch, 1, H, W)
                x = x.view(batch_size, 1, self.img_height, self.img_width)
                # Conv encoding
                x = self.conv(x)
                # Flatten and project
                x = x.view(batch_size, -1)
                x = self.fc(x)
                return x

        class CNNActorCritic(nn.Module):
            """Actor-Critic with CNN encoder for depth observations."""
            def __init__(self, img_height, img_width, act_dim, latent_dim=256):
                super().__init__()
                self.encoder = CNNEncoder(img_height, img_width, latent_dim)

                # Actor head
                self.actor = nn.Sequential(
                    nn.Linear(latent_dim, 128), nn.ELU(),
                    nn.Linear(128, 64), nn.ELU(),
                    nn.Linear(64, act_dim),
                )
                self.log_std = nn.Parameter(torch.zeros(act_dim))

                # Critic head
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

            def act(self, obs):
                mean, _ = self.forward(obs)
                std = self.log_std.exp()
                return Normal(mean, std).sample()

            def get_value(self, obs):
                latent = self.encoder(obs)
                return self.critic(latent)

        act_dim = env.action_space.shape[0]
        policy = CNNActorCritic(env.cfg.height, env.cfg.width, act_dim).to(env.device)
        optimizer = optim.Adam(policy.parameters(), lr=3e-4)

        # Print architecture info
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"[Train] CNN Actor-Critic: {total_params:,} parameters")

        obs, _ = env.reset()
        episode_rewards = []

        for it in range(args.max_iterations):
            # Collect rollout
            rewards = []
            for _ in range(24):
                action = policy.act(obs)
                obs, reward, term, trunc, info = env.step(action)
                rewards.append(reward.mean().item())
                if term.any() or trunc.any():
                    obs, _ = env.reset()

            episode_rewards.append(np.mean(rewards))

            if it % 20 == 0:
                print(f"[Iter {it:4d}] Mean reward: {np.mean(episode_rewards[-20:]):.3f}")

        print("\n[Train] Done!")

    else:
        # ============ Test Mode ============
        print(f"\n[Test] Running {args.num_steps} steps...")
        obs, _ = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        for step in range(args.num_steps):
            action = torch.zeros((env.num_envs, 4), device=env.device)
            obs, reward, term, trunc, info = env.step(action)
            print(f"  Step {step}: obs=[{obs.min():.2f}, {obs.max():.2f}], reward={reward.mean():.2f}")

        print("\n[Test] Done!")

    env.close()
    simulation_app.close()
