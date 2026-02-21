#!/usr/bin/env python3
"""
Crane environment with depth image observations as a proper DirectRLEnv.

This extends CraneDirectEnvFull to return depth observations instead of
log pose observations, for proper RSL-RL integration.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
from isaaclab.utils import configclass

# Will import CraneDirectEnvFull after app launch
# from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


@configclass
class CraneDepthEnvCfg:
    """Additional config for depth observations."""
    # Depth image settings
    depth_height: int = 128
    depth_width: int = 128
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = True


class CraneDepthDirectEnv(gym.Env):
    """Crane environment that returns depth image observations.

    This is a proper DirectRLEnv-compatible class that can be used with RSL-RL.
    It extends CraneDirectEnvFull's functionality but overrides observation handling.

    Observation: Flattened masked depth image (128*128 = 16384 dims)
    Action: Same as base env (4D: x, y, z, yaw)
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__()

        # Import here after Isaac app is launched
        from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull

        # Depth observation config - get from cfg FIRST
        self.depth_height = getattr(cfg, 'depth_height', 128)
        self.depth_width = getattr(cfg, 'depth_width', 128)
        self.depth_min = getattr(cfg, 'depth_min', 1.5)
        self.depth_max = getattr(cfg, 'depth_max', 8.0)
        self.use_semantic_mask = getattr(cfg, 'use_semantic_mask', True)
        self._obs_dim = self.depth_height * self.depth_width

        print(f"[DepthDirectEnv] Depth config: {self.depth_height}x{self.depth_width} = {self._obs_dim}", flush=True)

        # Merge depth config into base config
        base_cfg = CraneDirectEnvCfgFull()

        # Copy all attributes from cfg to base_cfg
        for attr in dir(cfg):
            if not attr.startswith('_') and hasattr(base_cfg, attr):
                try:
                    setattr(base_cfg, attr, getattr(cfg, attr))
                except AttributeError:
                    pass

        # Explicitly set observation_space to depth dimensions
        base_cfg.observation_space = self._obs_dim
        print(f"[DepthDirectEnv] Set base_cfg.observation_space = {base_cfg.observation_space}", flush=True)

        # Ensure camera is enabled
        base_cfg.enable_camera = True

        # Create base environment
        self._base_env = CraneDirectEnvFull(base_cfg, render_mode=render_mode, **kwargs)

        # Verify and patch single_observation_space
        import numpy as np
        print(f"[DepthDirectEnv] Base env single_observation_space before patch: {self._base_env.single_observation_space}", flush=True)
        depth_obs_space = gym.spaces.Box(low=0, high=1, shape=(self._obs_dim,), dtype=np.float32)
        self._base_env.single_observation_space["policy"] = depth_obs_space
        print(f"[DepthDirectEnv] Base env single_observation_space after patch: {self._base_env.single_observation_space}", flush=True)

        # Monkey-patch base env's _get_observations to return depth
        # This is needed because RslRlVecEnvWrapper calls unwrapped._get_observations()
        depth_env = self  # Capture reference for closure
        original_get_obs = self._base_env._get_observations

        def patched_get_observations():
            depth_obs = depth_env._get_depth_observation()
            return {"policy": depth_obs}

        self._base_env._get_observations = patched_get_observations
        print(f"[DepthDirectEnv] Patched base env _get_observations", flush=True)

    def _get_depth_observation(self) -> torch.Tensor:
        """Get masked depth observations for all environments."""
        # Update camera
        self._base_env._camera.update(dt=self._base_env.cfg.sim.dt)

        # Get depth (num_envs, H, W)
        depth = self._base_env._camera.data.output["depth"].squeeze(-1)

        # Apply semantic mask if enabled
        if self.use_semantic_mask and "semantic_segmentation" in self._base_env.cfg.camera_cfg.data_types:
            sem_seg = self._base_env._camera.data.output["semantic_segmentation"]
            if sem_seg.dim() == 4:
                sem_ids = sem_seg[..., 0]
            else:
                sem_ids = sem_seg
            # Keep only log pixels
            mask = (sem_ids > 0)
            depth = torch.where(mask, depth, torch.tensor(self.depth_max, device=depth.device))

        # Clip and normalize
        depth = torch.clamp(depth, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min)

        # Downsample
        depth = depth.unsqueeze(1)  # (num_envs, 1, H, W)
        depth = F.interpolate(depth, size=(self.depth_height, self.depth_width),
                              mode='bilinear', align_corners=False)
        depth = depth.squeeze(1)  # (num_envs, depth_h, depth_w)

        # Flatten
        obs = depth.view(self.num_envs, -1)

        # Verification: log every 1000 calls to confirm we're using depth
        if not hasattr(self, '_obs_call_count'):
            self._obs_call_count = 0
        self._obs_call_count += 1
        if self._obs_call_count <= 3 or self._obs_call_count % 1000 == 0:
            import sys
            expected_dim = self.depth_height * self.depth_width
            sys.stderr.write(f"[VERIFY] _get_depth_observation call #{self._obs_call_count}: "
                           f"shape={obs.shape}, expected=({self.num_envs}, {expected_dim}), "
                           f"range=[{obs.min():.3f}, {obs.max():.3f}]\n")
            assert obs.shape[1] == expected_dim, f"WRONG OBS DIM! Got {obs.shape[1]}, expected {expected_dim}"

        return obs

    def reset(self, **kwargs):
        """Reset and return depth observations in dict format for RslRlVecEnvWrapper."""
        import sys
        obs_dict, info = self._base_env.reset(**kwargs)
        self._base_env.sim.render()

        # Replace observations with depth - THIS IS THE KEY PART
        # We discard obs_dict from base env and return depth instead
        depth_obs = self._get_depth_observation()

        # Verify we're returning depth, not pose observations
        expected_dim = self.depth_height * self.depth_width
        assert depth_obs.shape[1] == expected_dim, \
            f"[CRITICAL] reset() returning wrong obs! Got {depth_obs.shape[1]}, expected {expected_dim} (depth)"
        sys.stderr.write(f"[DepthDirectEnv] reset() returning DEPTH obs: {depth_obs.shape} (NOT pose obs)\n")

        # Return in dict format for RSL-RL wrapper compatibility
        return {"policy": depth_obs}, info

    def step(self, action):
        """Step and return depth observations in dict format for RslRlVecEnvWrapper."""
        obs_dict, reward, terminated, truncated, info = self._base_env.step(action)
        self._base_env.sim.render()

        # Replace observations with depth
        depth_obs = self._get_depth_observation()

        # Return in dict format for RSL-RL wrapper compatibility
        return {"policy": depth_obs}, reward, terminated, truncated, info

    def _get_observations(self):
        """Override to return depth observations when wrapper calls this directly."""
        depth_obs = self._get_depth_observation()
        return {"policy": depth_obs}

    # ============ Proxy all other attributes to base env ============

    @property
    def num_envs(self):
        return self._base_env.num_envs

    @property
    def device(self):
        return self._base_env.device

    @property
    def cfg(self):
        return self._base_env.cfg

    @property
    def sim(self):
        return self._base_env.sim

    @property
    def num_observations(self):
        return self._obs_dim

    @property
    def num_actions(self):
        return self._base_env.num_actions

    @property
    def observation_space(self):
        import gymnasium as gym
        import numpy as np
        return gym.spaces.Box(low=0, high=1, shape=(self._obs_dim,), dtype=np.float32)

    @property
    def action_space(self):
        return self._base_env.action_space

    @property
    def max_episode_length(self):
        return self._base_env.max_episode_length

    @property
    def episode_length_buf(self):
        return self._base_env.episode_length_buf

    @property
    def unwrapped(self):
        """Return base env for DirectRLEnv isinstance check (obs space is patched)."""
        return self._base_env

    def close(self):
        self._base_env.close()

    def __getattr__(self, name):
        """Proxy any missing attributes to base env."""
        return getattr(self._base_env, name)


# Test
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=5)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from crane_rl_env_full import CraneDirectEnvCfgFull

    # Create config with depth settings
    @configclass
    class TestDepthCfg(CraneDirectEnvCfgFull):
        depth_height: int = 48
        depth_width: int = 48
        depth_min: float = 1.5
        depth_max: float = 8.0
        use_semantic_mask: bool = True

    cfg = TestDepthCfg()
    cfg.scene.num_envs = args.num_envs

    print("\n[Test] Creating depth environment...")
    env = CraneDepthDirectEnv(cfg)

    print(f"  Observation dim: {env.num_observations}")
    print(f"  Action dim: {env.num_actions}")

    print("\n[Test] Resetting...")
    obs, info = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    print(f"\n[Test] Running {args.num_steps} steps...")
    for i in range(args.num_steps):
        action = torch.zeros((env.num_envs, 4), device=env.device)
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i}: obs=[{obs.min():.2f}, {obs.max():.2f}], reward={reward.mean():.2f}")

    print("\n[Test] Done!")
    env.close()
    simulation_app.close()
