"""
Crane grasping environment with point cloud observations.

This environment wraps the base crane environment and provides
point cloud observations of the log pile for visual RL.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Import base environment (after isaaclab app starts)
# These imports happen in main() after AppLauncher


@dataclass
class PointCloudObsConfig:
    """Configuration for point cloud observations."""

    # Point cloud settings
    num_points: int = 512  # Fixed number of points in observation
    depth_range: Tuple[float, float] = (1.5, 8.0)  # Depth range to capture

    # Normalization (center and scale point cloud)
    normalize_points: bool = True
    center_x: float = 2.0  # Approximate center of pile in world X
    center_y: float = -3.0  # Approximate center of pile in world Y
    center_z: float = 0.5  # Approximate center of pile in world Z
    scale: float = 5.0  # Scale factor (divide by this)

    # Proprioception (non-visual state)
    include_proprio: bool = True
    proprio_keys: Tuple[str, ...] = (
        "grapple_pos",  # Current grapple position (3)
        "grapple_open",  # Grapple open/closed (1)
        "target_x",  # Current target X (1)
        "remaining_logs",  # Remaining logs normalized (1)
    )


class CranePointCloudEnv:
    """Wrapper that adds point cloud observations to the base crane environment.

    This wrapper:
    1. Captures log-only point cloud from camera with semantic segmentation
    2. Processes to fixed-size observation (pad/sample to num_points)
    3. Optionally adds proprioceptive state
    4. Returns combined observation for RL policy
    """

    def __init__(
        self,
        base_env,
        obs_config: PointCloudObsConfig = None,
    ):
        """Initialize point cloud environment wrapper.

        Args:
            base_env: Base CraneDirectEnvFull instance
            obs_config: Point cloud observation configuration
        """
        self.env = base_env
        self.cfg = obs_config or PointCloudObsConfig()

        # Ensure camera is enabled
        if not hasattr(self.env, '_camera') or self.env._camera is None:
            raise RuntimeError("Base environment must have camera enabled (enable_camera=True)")

        # Calculate observation dimensions
        self.point_cloud_dim = self.cfg.num_points * 3
        self.proprio_dim = self._calculate_proprio_dim()
        self.obs_dim = self.point_cloud_dim + self.proprio_dim

        print(f"[PointCloudEnv] Observation dim: {self.obs_dim}")
        print(f"  Point cloud: {self.cfg.num_points} points x 3 = {self.point_cloud_dim}")
        print(f"  Proprioception: {self.proprio_dim}")

    def _calculate_proprio_dim(self) -> int:
        """Calculate proprioception dimension based on config."""
        if not self.cfg.include_proprio:
            return 0

        dim = 0
        for key in self.cfg.proprio_keys:
            if key == "grapple_pos":
                dim += 3
            elif key in ["grapple_open", "target_x", "remaining_logs"]:
                dim += 1
        return dim

    def _get_point_cloud_obs(self, env_idx: int = 0) -> torch.Tensor:
        """Get fixed-size point cloud observation for one environment.

        Args:
            env_idx: Environment index

        Returns:
            Tensor of shape (num_points, 3) - normalized XYZ points
        """
        # Get log-only point cloud from base env
        pc = self.env.get_log_pointcloud_world(
            env_idx=env_idx,
            max_points=None,  # Get all points first
            depth_range=self.cfg.depth_range,
        )

        num_points = self.cfg.num_points
        device = self.env.device

        if pc.shape[0] == 0:
            # No points - return zeros
            return torch.zeros((num_points, 3), device=device)

        # Normalize points (center and scale)
        if self.cfg.normalize_points:
            pc = pc.clone()
            pc[:, 0] = (pc[:, 0] - self.cfg.center_x) / self.cfg.scale
            pc[:, 1] = (pc[:, 1] - self.cfg.center_y) / self.cfg.scale
            pc[:, 2] = (pc[:, 2] - self.cfg.center_z) / self.cfg.scale

        # Pad or sample to fixed size
        if pc.shape[0] < num_points:
            # Pad with zeros
            padding = torch.zeros((num_points - pc.shape[0], 3), device=device)
            pc = torch.cat([pc, padding], dim=0)
        elif pc.shape[0] > num_points:
            # Random sample
            indices = torch.randperm(pc.shape[0], device=device)[:num_points]
            pc = pc[indices]

        return pc

    def _get_proprio_obs(self, env_idx: int = 0) -> torch.Tensor:
        """Get proprioceptive observation for one environment.

        Args:
            env_idx: Environment index

        Returns:
            Tensor of shape (proprio_dim,)
        """
        if not self.cfg.include_proprio:
            return torch.tensor([], device=self.env.device)

        obs_parts = []

        for key in self.cfg.proprio_keys:
            if key == "grapple_pos":
                # Get grapple position from crane articulation
                bg_pose = self.env.crane.data.body_pos_w[env_idx, self.env._basegrapple_body_id]
                obs_parts.append(bg_pose[:3])

            elif key == "grapple_open":
                # Grapple open state (0 = closed, 1 = open)
                is_open = 1.0 if self.env._grapple_state[env_idx] == "open" else 0.0
                obs_parts.append(torch.tensor([is_open], device=self.env.device))

            elif key == "target_x":
                # Current target X position (normalized)
                target_x = self.env._current_target_x[env_idx] if hasattr(self.env, '_current_target_x') else 0.0
                obs_parts.append(torch.tensor([target_x / 5.0], device=self.env.device))  # Normalize

            elif key == "remaining_logs":
                # Remaining logs (normalized by initial count)
                initial = self.env._per_env_log_counts[env_idx]
                deposited = len(self.env._deposited_logs[env_idx])
                remaining = (initial - deposited) / initial
                obs_parts.append(torch.tensor([remaining], device=self.env.device))

        return torch.cat(obs_parts, dim=0)

    def get_observation(self) -> torch.Tensor:
        """Get full observation for all environments.

        Returns:
            Tensor of shape (num_envs, obs_dim) containing flattened
            point cloud + proprioception for each environment
        """
        num_envs = self.env.num_envs
        device = self.env.device

        # Update camera to get fresh data
        if self.env._camera is not None:
            self.env._camera.update(dt=self.env.cfg.sim.dt)

        obs_list = []
        for env_idx in range(num_envs):
            # Get point cloud observation
            pc = self._get_point_cloud_obs(env_idx)  # (num_points, 3)
            pc_flat = pc.view(-1)  # (num_points * 3,)

            # Get proprioception
            proprio = self._get_proprio_obs(env_idx)  # (proprio_dim,)

            # Combine
            obs = torch.cat([pc_flat, proprio], dim=0)
            obs_list.append(obs)

        return torch.stack(obs_list, dim=0)  # (num_envs, obs_dim)

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset environment and return initial observation.

        Returns:
            obs: Initial observation (num_envs, obs_dim)
            info: Additional info dict
        """
        # Reset base environment
        base_obs, info = self.env.reset()

        # Render to update camera
        self.env.sim.render()
        if self.env._camera is not None:
            self.env._camera.update(dt=self.env.cfg.sim.dt)

        # Get point cloud observation
        obs = self.get_observation()

        return obs, info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step environment with action.

        Args:
            action: Action tensor (num_envs, action_dim)

        Returns:
            obs: Next observation (num_envs, obs_dim)
            reward: Reward (num_envs,)
            terminated: Termination flags (num_envs,)
            truncated: Truncation flags (num_envs,)
            info: Additional info dict
        """
        # Step base environment
        base_obs, reward, terminated, truncated, info = self.env.step(action)

        # Render to update camera after step (captures despawned logs)
        self.env.sim.render()
        if self.env._camera is not None:
            self.env._camera.update(dt=self.env.cfg.sim.dt)

        # Get point cloud observation
        obs = self.get_observation()

        return obs, reward, terminated, truncated, info

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def observation_space(self):
        """Return observation space for RL algorithms."""
        import gymnasium as gym
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        """Return action space from base environment."""
        return self.env.action_space

    def close(self):
        """Close environment."""
        self.env.close()


def create_pointcloud_env(num_envs: int = 1, **kwargs):
    """Factory function to create point cloud environment.

    Args:
        num_envs: Number of parallel environments
        **kwargs: Additional arguments passed to PointCloudObsConfig

    Returns:
        CranePointCloudEnv instance
    """
    # Import here after isaaclab app starts
    from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull

    # Create base environment config
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = num_envs
    cfg.enable_camera = True  # Must enable camera

    # Create base environment
    base_env = CraneDirectEnvFull(cfg, render_mode="rgb_array")

    # Create point cloud observation config
    obs_config = PointCloudObsConfig(**kwargs)

    # Wrap with point cloud observations
    env = CranePointCloudEnv(base_env, obs_config)

    return env


# Test script
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import sys

    # Add path
    crane_scripts_path = Path(__file__).resolve().parent
    sys.path.insert(0, str(crane_scripts_path))

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=5)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import environment
    from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull

    # Create environment
    print("\n[Test] Creating point cloud environment...")
    env = create_pointcloud_env(num_envs=args.num_envs)

    print(f"\n[Test] Environment created:")
    print(f"  Observation dim: {env.obs_dim}")
    print(f"  Action space: {env.action_space}")

    # Reset
    print("\n[Test] Resetting environment...")
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Step
    print(f"\n[Test] Running {args.num_steps} steps...")
    for step in range(args.num_steps):
        action = torch.zeros((env.num_envs, 4), device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step}: obs_shape={obs.shape}, reward={reward.mean():.3f}")

    print("\n[Test] Done!")
    env.close()
    simulation_app.close()
