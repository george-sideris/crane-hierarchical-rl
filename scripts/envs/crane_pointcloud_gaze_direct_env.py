#!/usr/bin/env python3
"""
GAZE variant of the point-cloud RL env (for BC->RL fine-tuning on the gaze policy).

Mirrors crane_pointcloud_direct_env.py, but wraps the GAZE env (crane_rl_env_gaze) so the
observation matches the BC gaze pipeline EXACTLY: the July-2 BasemastCam, the gaze slew, and
`get_pointcloud_base` (optical->base, no world pivot) -> crop to the action box -> FPS to
num_points. This is the same preprocessing as train_bc_pointcloud.get_log_pointcloud_base_frame
with --raw_pcd --crop_to_bounds. Fine-tuning through the OLD full-env pointcloud wrapper
(crane_pointcloud_direct_env.py) would use the wrong camera + world-pivot cloud and undo the
sim2real alignment, so BC->RL for the gaze policy MUST use this env.
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any

import gymnasium as gym
from isaaclab.utils import configclass

# Will import CraneDirectEnvFull after app launch
# from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


@configclass
class CranePointCloudEnvCfg:
    """Additional config for point cloud observations."""
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    save_debug_pointclouds: bool = False
    asymmetric_critic: bool = False
    """If True, also return 128D state vector as 'critic' obs for asymmetric actor-critic."""


class CranePointCloudGazeDirectEnv(gym.Env):
    """Gaze-env crane environment that returns point cloud observations for BC->RL.

    Same as CranePointCloudDirectEnv but wraps the GAZE base env and captures the cloud with
    the gaze pipeline (optical->base + crop-to-bounds + FPS), matching the BC gaze policy.

    Observation: Flattened base-frame point cloud (num_points * 3 dims)
    Action: Same as base env (5D with cos/sin yaw)
    """

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__()

        # Import here after Isaac app is launched. GAZE env (July-2 BasemastCam + gaze slew).
        from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull

        # Point cloud observation config
        self.num_points = getattr(cfg, 'num_points', 1024)
        self.depth_range_min = getattr(cfg, 'depth_range_min', 1.0)
        self.depth_range_max = getattr(cfg, 'depth_range_max', 10.0)
        self.save_debug_pointclouds = getattr(cfg, 'save_debug_pointclouds', False)
        self.asymmetric_critic = getattr(cfg, 'asymmetric_critic', False)
        # P3 (BC->RL scoring): widened observation crop (action box +- margin, z-down only) and
        # point-index actions ([idx, dz, cos2yaw, sin2yaw] resolved against the CACHED obs cloud, re-encoded
        # to the base env's 5D arctanh action so the base decode/bed-clamp path is unchanged).
        self.obs_crop_margin = float(getattr(cfg, 'obs_crop_margin', 0.0))
        self.point_index_actions = bool(getattr(cfg, 'point_index_actions', False))
        self._cached_clouds = None      # (num_envs, num_points, 3) - the obs the policy last saw
        self.use_raw_pointcloud = getattr(cfg, 'use_raw_pointcloud', False)
        self._obs_dim = self.num_points * 3
        # top-N logs x (x, y, z, yaw), height-sorted, from _build_target_selection_obs
        self._critic_obs_dim = getattr(cfg, 'max_logs_obs', 32) * 4

        print(f"[PointCloudDirectEnv] Config: {self.num_points} points, "
              f"depth=[{self.depth_range_min}, {self.depth_range_max}], "
              f"obs_dim={self._obs_dim}, asymmetric_critic={self.asymmetric_critic}, "
              f"raw_pcd={self.use_raw_pointcloud}", flush=True)

        # Merge point cloud config into base config
        base_cfg = CraneDirectEnvCfgFull()

        # Copy all attributes from cfg to base_cfg
        for attr in dir(cfg):
            if not attr.startswith('_') and hasattr(base_cfg, attr):
                try:
                    setattr(base_cfg, attr, getattr(cfg, attr))
                except AttributeError:
                    pass

        # Explicitly set observation_space to point cloud dimensions
        base_cfg.observation_space = self._obs_dim
        print(f"[PointCloudDirectEnv] Set base_cfg.observation_space = {base_cfg.observation_space}", flush=True)

        # Ensure camera is enabled
        base_cfg.enable_camera = True

        # Create base environment
        self._base_env = CraneDirectEnvFull(base_cfg, render_mode=render_mode, **kwargs)

        # Verify and patch single_observation_space
        import numpy as np
        print(f"[PointCloudDirectEnv] Base env single_observation_space before patch: "
              f"{self._base_env.single_observation_space}", flush=True)
        pc_obs_space = gym.spaces.Box(low=-float('inf'), high=float('inf'),
                                       shape=(self._obs_dim,), dtype=np.float32)
        self._base_env.single_observation_space["policy"] = pc_obs_space
        if self.point_index_actions:
            self._base_env.single_action_space = gym.spaces.Box(
                low=-float('inf'), high=float('inf'), shape=(4,), dtype=np.float32)
            self._base_env.action_space = gym.vector.utils.batch_space(
                self._base_env.single_action_space, self.num_envs)
            print("[PointCloudDirectEnv] point-index actions: exposed 3D action space", flush=True)
        if self.asymmetric_critic:
            critic_obs_space = gym.spaces.Box(low=-1.0, high=1.0,
                                               shape=(self._critic_obs_dim,), dtype=np.float32)
            self._base_env.single_observation_space["critic"] = critic_obs_space
            # RslRlVecEnvWrapper requires num_states attr to detect privileged obs
            self._base_env.num_states = self._critic_obs_dim
        print(f"[PointCloudDirectEnv] Base env single_observation_space after patch: "
              f"{self._base_env.single_observation_space}", flush=True)

        # Monkey-patch base env's _get_observations to return point cloud
        # This is needed because RslRlVecEnvWrapper calls unwrapped._get_observations()
        pc_env = self  # Capture reference for closure
        self._skip_internal_obs = False  # Guard flag to avoid double PC computation in step()

        def patched_get_observations():
            if pc_env._skip_internal_obs:
                # Return dummy — wrapper's step() will compute real PC obs after render
                obs = {"policy": torch.zeros(pc_env.num_envs, pc_env._obs_dim, device=pc_env.device)}
                if pc_env.asymmetric_critic:
                    obs["critic"] = torch.zeros(pc_env.num_envs, pc_env._critic_obs_dim, device=pc_env.device)
                return obs
            pc_obs = pc_env._get_pointcloud_observation()
            obs = {"policy": pc_obs}
            if pc_env.asymmetric_critic:
                obs["critic"] = pc_env._base_env._build_target_selection_obs()
            return obs

        self._base_env._get_observations = patched_get_observations
        print(f"[PointCloudDirectEnv] Patched base env _get_observations (with skip guard)", flush=True)

    def _gaze_cloud(self, env_idx: int) -> torch.Tensor:
        """Gaze-pipeline cloud for one env: optical->base -> crop to action box -> FPS.

        Mirrors train_bc_pointcloud.get_log_pointcloud_base_frame(raw_pcd=True, crop_to_bounds=True)
        so RL fine-tuning sees the SAME observation the BC gaze policy was trained on.
        """
        # optical -> base directly (no world pivot), full scene (matches BC max_points=50000)
        pc_base = self._base_env.get_pointcloud_base(
            env_idx, max_points=50000,
            depth_range=(self.depth_range_min, self.depth_range_max))
        if pc_base.shape[0] == 0:
            return torch.zeros((self.num_points, 3), device=self.device)

        # crop to the action-bounds box (same crop as BC + deploy)
        if getattr(self._base_env, "_action_bounds_min", None) is not None:
            if not bool(self._base_env._action_bounds_valid[env_idx]):
                self._base_env._compute_action_space_bounds()
            bmin = self._base_env._action_bounds_min[env_idx]
            bmax = self._base_env._action_bounds_max[env_idx]
            g = self.obs_crop_margin
            m = ((pc_base[:, 0] >= bmin[0] - g) & (pc_base[:, 0] <= bmax[0] + g) &
                 (pc_base[:, 1] >= bmin[1] - g) & (pc_base[:, 1] <= bmax[1] + g) &
                 (pc_base[:, 2] >= bmin[2] - g) & (pc_base[:, 2] <= bmax[2]))
            pc_base = pc_base[m]
            if pc_base.shape[0] == 0:
                return torch.zeros((self.num_points, 3), device=self.device)

        return self._farthest_point_sampling(pc_base, self.num_points)

    def _get_pointcloud_observation(self) -> torch.Tensor:
        """Get base-frame point cloud observations for all environments (gaze pipeline).

        Per env: get_pointcloud_base (optical->base) -> crop to action box -> FPS to num_points
        -> flatten to (num_points * 3,). Matches the BC gaze obs exactly.

        Returns:
            (num_envs, num_points * 3) tensor
        """
        # Update camera
        self._base_env._camera.update(dt=self._base_env.cfg.sim.dt)

        all_obs = []
        for env_idx in range(self.num_envs):
            pc_sampled = self._gaze_cloud(env_idx)   # cropped+FPS base-frame cloud (matches BC)
            all_obs.append(pc_sampled.view(-1))

        obs = torch.stack(all_obs, dim=0)  # (num_envs, num_points * 3)
        self._cached_clouds = obs.view(self.num_envs, self.num_points, 3).clone()

        # Verification logging
        if not hasattr(self, '_obs_call_count'):
            self._obs_call_count = 0
        self._obs_call_count += 1

        # One-time diagnostic: report the gaze cloud's extent (should sit in the action box).
        if self._obs_call_count == 1:
            import sys
            for i in range(min(2, self.num_envs)):
                rl_obs = obs[i].view(self.num_points, 3)
                nz = rl_obs[~torch.all(rl_obs == 0, dim=1)]
                if nz.shape[0]:
                    sys.stderr.write(
                        f"[DIAGNOSTIC] env{i}: gaze cloud non-pad pts={nz.shape[0]}, "
                        f"x[{nz[:,0].min():.2f},{nz[:,0].max():.2f}] "
                        f"y[{nz[:,1].min():.2f},{nz[:,1].max():.2f}] "
                        f"z[{nz[:,2].min():.2f},{nz[:,2].max():.2f}]\n"
                    )

        # Save debug point clouds + plots on first call (opt-in)
        if self._obs_call_count == 1 and self.save_debug_pointclouds:
            import numpy as np, os
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Save alongside policy logs if log_dir is set, else fallback
            base_dir = getattr(self, 'log_dir', None) or '.'
            debug_dir = os.path.join(base_dir, "debug_pointclouds")
            os.makedirs(debug_dir, exist_ok=True)

            for i in range(self.num_envs):
                pc = obs[i].view(self.num_points, 3).cpu().numpy()
                np.save(os.path.join(debug_dir, f"env{i}_pointcloud.npy"), pc)

                # Render 3D scatter plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap='viridis')
                ax.set_xlabel('X (base frame)')
                ax.set_ylabel('Y (base frame)')
                ax.set_zlabel('Z (base frame)')
                ax.set_title(f'Env {i}: {self.num_points} points (base frame)')

                # Top-down view
                fig2 = plt.figure(figsize=(10, 8))
                ax2 = fig2.add_subplot(111)
                sc = ax2.scatter(pc[:, 0], pc[:, 1], s=1, c=pc[:, 2], cmap='viridis')
                ax2.set_xlabel('X (base frame)')
                ax2.set_ylabel('Y (base frame)')
                ax2.set_title(f'Env {i}: Top-Down View ({self.num_points} points)')
                ax2.set_aspect('equal')
                plt.colorbar(sc, label='Z height')

                fig.savefig(os.path.join(debug_dir, f"env{i}_3d.png"), dpi=150, bbox_inches='tight')
                fig2.savefig(os.path.join(debug_dir, f"env{i}_topdown.png"), dpi=150, bbox_inches='tight')
                plt.close(fig)
                plt.close(fig2)

            print(f"[PointCloudDirectEnv] Saved debug point clouds + plots to {debug_dir}/")

        if self._obs_call_count <= 3 or self._obs_call_count % 1000 == 0:
            import sys
            sys.stderr.write(f"[VERIFY] _get_pointcloud_observation call #{self._obs_call_count}: "
                           f"shape={obs.shape}, expected=({self.num_envs}, {self._obs_dim}), "
                           f"range=[{obs.min():.3f}, {obs.max():.3f}]\n")
            assert obs.shape[1] == self._obs_dim, \
                f"WRONG OBS DIM! Got {obs.shape[1]}, expected {self._obs_dim}"

        return obs

    def _world_to_base_frame(self, pc_world: torch.Tensor, env_idx: int) -> torch.Tensor:
        """Transform world-frame point cloud to crane base frame.

        Matches the transform in train_bc_pointcloud.py:get_log_pointcloud_base_frame.

        Args:
            pc_world: (N, 3) points in world frame
            env_idx: Environment index

        Returns:
            (N, 3) points in base frame
        """
        # Get crane base pose in world frame
        base_pos_w = self._base_env.crane.data.root_pos_w[env_idx]    # (3,)
        base_quat_w = self._base_env.crane.data.root_quat_w[env_idx]  # (4,) wxyz

        # Translate to base origin
        pc_translated = pc_world - base_pos_w

        # Rotate by inverse of base orientation
        w, x, y, z = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]

        # Rotation matrix from quaternion (wxyz convention)
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])  # (3, 3)

        # Apply inverse rotation: (R.T @ points.T).T = points @ R
        pc_base = pc_translated @ R

        return pc_base

    def _farthest_point_sampling(self, points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Farthest Point Sampling for point cloud downsampling.

        Matches the FPS in train_bc_pointcloud.py.

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

    def reset(self, **kwargs):
        """Reset and return point cloud observations in dict format for RslRlVecEnvWrapper."""
        import sys
        obs_dict, info = self._base_env.reset(**kwargs)
        self._base_env.sim.render()

        # Replace observations with point cloud
        pc_obs = self._get_pointcloud_observation()

        # Verify we're returning point cloud, not pose observations
        assert pc_obs.shape[1] == self._obs_dim, \
            f"[CRITICAL] reset() returning wrong obs! Got {pc_obs.shape[1]}, expected {self._obs_dim} (pointcloud)"
        sys.stderr.write(f"[PointCloudDirectEnv] reset() returning POINTCLOUD obs: {pc_obs.shape}\n")

        obs = {"policy": pc_obs}
        if self.asymmetric_critic:
            obs["critic"] = self._base_env._build_target_selection_obs()
        return obs, info

    def _convert_point_index_action(self, action):
        """[idx, dz, cos2yaw, sin2yaw] -> base env 5D arctanh action, via the cached obs clouds.

        The chosen point IS one the policy was shown (idx into the cached cloud), so the
        decoded metre target is guaranteed on observed material; dz offsets its z; the result
        is re-encoded exactly like the heuristic/scoring bridges so the base env's decode
        (incl. the bed-floor clamp) reproduces it.
        """
        if self._cached_clouds is None:
            raise RuntimeError("point_index_actions: no cached obs cloud (step before reset?)")
        a5 = torch.zeros((action.shape[0], 5), device=action.device)
        for i in range(action.shape[0]):
            idx = int(action[i, 0].item())
            idx = max(0, min(self.num_points - 1, idx))
            p = self._cached_clouds[i, idx]
            bmin = self._base_env._action_bounds_min[i]
            bmax = self._base_env._action_bounds_max[i]
            x = float(p[0]); y = float(p[1]); z = float(p[2] + action[i, 1])
            yaw = 0.5 * float(torch.atan2(action[i, 3], action[i, 2]))
            def _enc(v, lo, hi):
                n = torch.clamp(2.0 * (v - lo) / (hi - lo + 1e-9) - 1.0, -0.999, 0.999)
                return torch.atanh(n)
            a5[i, 0] = _enc(torch.tensor(x), bmin[0], bmax[0])
            a5[i, 1] = _enc(torch.tensor(y), bmin[1], bmax[1])
            a5[i, 2] = _enc(torch.tensor(z), bmin[2], bmax[2])
            a5[i, 3] = torch.cos(torch.tensor(2.0 * yaw))
            a5[i, 4] = torch.sin(torch.tensor(2.0 * yaw))
        return a5

    def step(self, action):
        """Step and return point cloud observations in dict format for RslRlVecEnvWrapper."""
        if self.point_index_actions:
            action = self._convert_point_index_action(action)
        # Skip the expensive PC computation inside base_env.step() → _get_observations()
        # since it reads a stale camera buffer (render hasn't happened yet).
        # We compute the real observation below after rendering.
        self._skip_internal_obs = True
        obs_dict, reward, terminated, truncated, info = self._base_env.step(action)
        self._skip_internal_obs = False

        self._base_env.sim.render()

        # Compute point cloud from the freshly rendered camera buffer
        pc_obs = self._get_pointcloud_observation()

        obs = {"policy": pc_obs}
        if self.asymmetric_critic:
            obs["critic"] = self._base_env._build_target_selection_obs()
        return obs, reward, terminated, truncated, info

    def _get_observations(self):
        """Override to return point cloud observations when wrapper calls this directly."""
        pc_obs = self._get_pointcloud_observation()
        obs = {"policy": pc_obs}
        if self.asymmetric_critic:
            obs["critic"] = self._base_env._build_target_selection_obs()
        return obs

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
        return gym.spaces.Box(low=-float('inf'), high=float('inf'),
                              shape=(self._obs_dim,), dtype=np.float32)

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

    from crane_rl_env_gaze import CraneDirectEnvCfgFull

    # Create config with point cloud settings
    @configclass
    class TestPointCloudCfg(CraneDirectEnvCfgFull):
        num_points: int = 1024
        depth_range_min: float = 1.0
        depth_range_max: float = 10.0
        use_raw_pointcloud: bool = True

    cfg = TestPointCloudCfg()
    cfg.scene.num_envs = args.num_envs

    print("\n[Test] Creating gaze point cloud environment...")
    env = CranePointCloudGazeDirectEnv(cfg)

    print(f"  Observation dim: {env.num_observations}")
    print(f"  Action dim: {env.num_actions}")

    print("\n[Test] Resetting...")
    obs, info = env.reset()
    print(f"  Obs shape: {obs['policy'].shape}")
    print(f"  Obs range: [{obs['policy'].min():.3f}, {obs['policy'].max():.3f}]")

    print(f"\n[Test] Running {args.num_steps} steps...")
    for i in range(args.num_steps):
        action = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i}: obs=[{obs['policy'].min():.2f}, {obs['policy'].max():.2f}], reward={reward.mean():.2f}")

    print("\n[Test] Done!")
    env.close()
    simulation_app.close()
