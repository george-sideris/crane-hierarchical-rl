#!/usr/bin/env python3
"""
Behavioral Cloning Data Collection Script

Runs the crane environment with the expert heuristic and collects
(observation, action) pairs for training a BC policy.

The expert heuristic targets the highest available log in the rack.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Collect BC demonstrations from expert heuristic")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to collect")
parser.add_argument("--output_dir", type=str, default="bc_data", help="Output directory for data")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Now import the environment
import crane_rl_env_simplified
from crane_rl_env_simplified import CraneDirectEnvSimplified, CraneDirectEnvCfg
from isaaclab.utils.math import subtract_frame_transforms


class BCDataCollector:
    """Collects (observation, expert_action) pairs from the crane environment."""

    def __init__(self, env: CraneDirectEnvSimplified):
        self.env = env
        self.device = env.device

        # Storage for collected data
        self.observations = []
        self.actions = []
        self.episode_rewards = []

        # Track episodes
        self.current_episode_reward = torch.zeros(env.num_envs, device=self.device)
        self.episodes_completed = 0

    def get_expert_action(self, env_i: int) -> torch.Tensor:
        """
        Get the expert's action for a given environment.

        The expert targets the highest available log and returns
        normalized (y, z) action in [-1, 1] range.
        """
        # Get highest log position using the heuristic's method
        log_pos_b, selected_log_id = self.env._target_top_log_center_b(env_i)

        if selected_log_id == -1:
            # No logs available - return zero action
            return torch.zeros(2, device=self.device)

        # Get action bounds
        if not self.env._action_bounds_valid[env_i]:
            self.env._compute_action_space_bounds()

        min_b = self.env._action_bounds_min[env_i]
        max_b = self.env._action_bounds_max[env_i]

        # Extract Y and Z from log position
        y_target = log_pos_b[1].item()
        z_target = log_pos_b[2].item()

        # Normalize to [-1, 1] range
        # Inverse of: val = min + (norm + 1) / 2 * (max - min)
        # => norm = 2 * (val - min) / (max - min) - 1
        y_range = max_b[1].item() - min_b[1].item()
        z_range = max_b[2].item() - min_b[2].item()

        if y_range > 0:
            y_norm = 2.0 * (y_target - min_b[1].item()) / y_range - 1.0
        else:
            y_norm = 0.0

        if z_range > 0:
            z_norm = 2.0 * (z_target - min_b[2].item()) / z_range - 1.0
        else:
            z_norm = 0.0

        # Clamp to valid range
        y_norm = max(-1.0, min(1.0, y_norm))
        z_norm = max(-1.0, min(1.0, z_norm))

        # Apply inverse tanh (arctanh) since the env applies tanh to actions
        # tanh(action) = norm => action = arctanh(norm)
        # Clamp slightly to avoid inf at boundaries
        y_norm = max(-0.999, min(0.999, y_norm))
        z_norm = max(-0.999, min(0.999, z_norm))

        y_action = np.arctanh(y_norm)
        z_action = np.arctanh(z_norm)

        return torch.tensor([y_action, z_action], device=self.device, dtype=torch.float32)

    def collect_step(self):
        """
        Collect one step of demonstrations from all environments.

        Should be called when environments are at decision points (HOVER_UP phase).
        """
        # Check which environments are at decision points
        at_decision = (self.env._phase == self.env.PH_HOVER_UP) & (~self.env._target_frozen)

        for env_i in range(self.env.num_envs):
            if at_decision[env_i]:
                # Get observation for this environment
                obs = self.env._build_target_selection_obs()[env_i]

                # Get expert action
                expert_action = self.get_expert_action(env_i)

                # Store
                self.observations.append(obs.cpu().numpy())
                self.actions.append(expert_action.cpu().numpy())

    def collect_episodes(self, num_episodes: int, max_steps_per_episode: int = 2000):
        """Collect data from multiple episodes."""

        print(f"[BC] Starting data collection for {num_episodes} episodes...")

        # Reset environment
        self.env.reset()

        step_count = 0
        episodes_done = 0

        while episodes_done < num_episodes and simulation_app.is_running():
            with torch.inference_mode():
                # Collect expert demonstrations at decision points
                self.collect_step()

                # Get expert actions for all envs
                expert_actions = torch.zeros(self.env.num_envs, 2, device=self.device)
                for env_i in range(self.env.num_envs):
                    expert_actions[env_i] = self.get_expert_action(env_i)

                # Step environment with expert actions
                obs, rew, terminated, truncated, info = self.env.step(expert_actions)

                # Track rewards
                self.current_episode_reward += rew

                # Check for episode completions
                done = terminated | truncated
                for env_i in range(self.env.num_envs):
                    if done[env_i]:
                        self.episode_rewards.append(self.current_episode_reward[env_i].item())
                        self.current_episode_reward[env_i] = 0.0
                        episodes_done += 1

                        if episodes_done % 10 == 0:
                            avg_reward = np.mean(self.episode_rewards[-10:])
                            print(f"[BC] Episodes: {episodes_done}/{num_episodes}, "
                                  f"Avg reward (last 10): {avg_reward:.2f}, "
                                  f"Samples: {len(self.observations)}")

                        if episodes_done >= num_episodes:
                            break

                step_count += 1

                # Safety limit
                if step_count > num_episodes * max_steps_per_episode:
                    print(f"[BC] Warning: Reached step limit, collected {episodes_done} episodes")
                    break

        print(f"[BC] Collection complete: {len(self.observations)} samples from {episodes_done} episodes")
        return self.observations, self.actions

    def save_data(self, output_dir: str):
        """Save collected data to files."""
        os.makedirs(output_dir, exist_ok=True)

        obs_array = np.stack(self.observations, axis=0)
        act_array = np.stack(self.actions, axis=0)

        np.save(os.path.join(output_dir, "observations.npy"), obs_array)
        np.save(os.path.join(output_dir, "actions.npy"), act_array)
        np.save(os.path.join(output_dir, "episode_rewards.npy"), np.array(self.episode_rewards))

        print(f"[BC] Saved data to {output_dir}/")
        print(f"     - observations.npy: {obs_array.shape}")
        print(f"     - actions.npy: {act_array.shape}")
        print(f"     - episode_rewards.npy: {len(self.episode_rewards)} episodes")
        print(f"     - Avg episode reward: {np.mean(self.episode_rewards):.2f}")


def main():
    # Create environment config
    cfg = CraneDirectEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    # Use hierarchical mode to get proper observations
    cfg.use_hierarchical_rl = True

    # Standard physics settings
    cfg.sim.physx.solver_type = 1  # TGS
    cfg.sim.physx.enable_stabilization = True
    cfg.sim.physx.min_position_iteration_count = 192
    cfg.sim.physx.max_position_iteration_count = 255

    # Create environment
    env = CraneDirectEnvSimplified(cfg)
    print("[BC] Environment created")

    # Create collector
    collector = BCDataCollector(env)

    # Collect demonstrations
    collector.collect_episodes(num_episodes=args_cli.num_episodes)

    # Save data
    collector.save_data(args_cli.output_dir)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
