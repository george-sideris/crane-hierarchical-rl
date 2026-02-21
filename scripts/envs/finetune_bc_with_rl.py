#!/usr/bin/env python3
"""
Fine-tune BC Policy with RL (RSL-RL PPO)

This script:
1. Loads a BC-pretrained actor
2. Creates a full ActorCritic model with BC actor weights + random critic
3. Runs PPO fine-tuning

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/finetune_bc_with_rl.py \
        --bc_checkpoint logs/bc_policy/bc_20260118_072928/bc_policy.pt \
        --num_envs 64 --max_iterations 500

    # With domain randomization
    ./isaaclab.sh -p crane_testbed/scripts/envs/finetune_bc_with_rl.py \
        --bc_checkpoint logs/bc_policy/bc_20260118_072928/bc_policy.pt \
        --num_envs 64 --max_iterations 500 --domain_randomization
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Fine-tune BC policy with RL")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="Path to BC policy checkpoint")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=500, help="RL training iterations")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for logs")
# PPO hyperparameters
parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--init_noise_std", type=float, default=0.3, help="Initial action noise std")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Steps per env before update")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

import torch
import numpy as np
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO

# Import our environment
import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class BCPolicy(nn.Module):
    """MLP policy matching BC training architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def load_bc_policy(checkpoint_path: str, device: str):
    """Load BC policy from checkpoint."""
    print(f"[RL] Loading BC checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    hidden_dims = checkpoint['hidden_dims']

    print(f"[RL] BC architecture: {obs_dim} -> {hidden_dims} -> {action_dim}")

    policy = BCPolicy(obs_dim, action_dim, hidden_dims)
    policy.load_state_dict(checkpoint['model_state_dict'])

    return policy, obs_dim, action_dim, hidden_dims


def create_actor_critic_from_bc(bc_policy: BCPolicy, obs_dim: int, action_dim: int,
                                 hidden_dims: list, init_noise_std: float, device: str):
    """
    Create RSL-RL ActorCritic model with BC actor weights.

    The critic is randomly initialized.
    """
    print(f"[RL] Creating ActorCritic with BC actor weights...")

    # Create ActorCritic with matching architecture
    actor_critic = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,  # Same obs for critic
        num_actions=action_dim,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,  # Same architecture for critic
        activation='elu',
        init_noise_std=init_noise_std,
    ).to(device)

    # Map BC weights to ActorCritic actor
    # BC: network.0.weight -> ActorCritic: actor.0.weight
    bc_state = bc_policy.state_dict()
    ac_state = actor_critic.state_dict()

    print("[RL] Mapping BC weights to ActorCritic actor:")
    for bc_key, bc_param in bc_state.items():
        # Convert: network.X.weight -> actor.X.weight
        ac_key = bc_key.replace('network.', 'actor.')
        if ac_key in ac_state:
            print(f"  {bc_key} -> {ac_key}: {bc_param.shape}")
            ac_state[ac_key] = bc_param.clone()
        else:
            print(f"  WARNING: {ac_key} not found in ActorCritic!")

    # Load the modified state dict
    actor_critic.load_state_dict(ac_state)

    print(f"[RL] ActorCritic created with init_noise_std={init_noise_std}")
    return actor_critic


class SimpleVecEnvWrapper:
    """Simple wrapper to make our env compatible with RSL-RL."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_obs = env._build_target_selection_obs().shape[1]
        self.num_privileged_obs = self.num_obs  # No privileged obs
        self.num_actions = 4  # [x, y, z, yaw]
        self.max_episode_length = 30  # cycles

        # For RSL-RL compatibility
        self.obs_buf = None
        self.rew_buf = None
        self.reset_buf = None
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device)

    def get_observations(self):
        obs = self.env._build_target_selection_obs()
        return obs, {"observations": obs}

    def reset(self):
        self.env.reset()
        self.episode_length_buf[:] = 0
        return self.get_observations()[0], None

    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        dones = terminated | truncated
        self.episode_length_buf += 1

        # RSL-RL expects (obs, privileged_obs, rewards, dones, infos)
        obs_tensor = self.env._build_target_selection_obs()
        return obs_tensor, obs_tensor, rew, dones, info


def main():
    # Load BC policy
    bc_policy, obs_dim, action_dim, hidden_dims = load_bc_policy(
        args_cli.bc_checkpoint, args_cli.device
    )

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[RL] Environment created with {env.num_envs} envs")
    print(f"[RL] Domain randomization: {args_cli.domain_randomization}")

    # Wrap environment
    env_wrapper = SimpleVecEnvWrapper(env)

    # Create ActorCritic from BC weights
    actor_critic = create_actor_critic_from_bc(
        bc_policy, obs_dim, action_dim, hidden_dims,
        args_cli.init_noise_std, args_cli.device
    )

    # Create PPO algorithm
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=args_cli.learning_rate,
        gamma=0.99,
        lam=0.95,
        entropy_coef=args_cli.entropy_coef,
        value_loss_coef=1.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        schedule='adaptive',
        desired_kl=0.01,
        device=args_cli.device,
    )

    # Setup logging directory
    if args_cli.output_dir:
        log_dir = args_cli.output_dir
    else:
        bc_dir = os.path.dirname(args_cli.bc_checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(bc_dir, f"rl_finetune_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)
    print(f"[RL] Logging to: {log_dir}")

    # Training loop
    print(f"\n[RL] Starting PPO fine-tuning for {args_cli.max_iterations} iterations...")
    print(f"[RL] Steps per env: {args_cli.num_steps_per_env}")
    print("=" * 60)

    env_wrapper.reset()
    obs, _ = env_wrapper.get_observations()

    total_steps = 0
    best_reward = float('-inf')

    for iteration in range(args_cli.max_iterations):
        # Collect rollouts
        episode_rewards = []
        episode_lengths = []

        for step in range(args_cli.num_steps_per_env):
            with torch.inference_mode():
                actions = actor_critic.act(obs)

            obs, _, rewards, dones, infos = env_wrapper.step(actions)

            # Store transition for PPO
            ppo.process_env_step(rewards, dones, infos)

            # Track completed episodes
            for i in range(env_wrapper.num_envs):
                if dones[i]:
                    episode_rewards.append(env_wrapper.episode_length_buf[i].item())
                    env_wrapper.episode_length_buf[i] = 0

            total_steps += env_wrapper.num_envs

        # Update policy
        mean_value_loss, mean_surrogate_loss = ppo.update()

        # Logging
        if len(episode_rewards) > 0:
            mean_reward = np.mean(episode_rewards)
        else:
            mean_reward = 0.0

        if iteration % 10 == 0 or iteration == args_cli.max_iterations - 1:
            print(f"[RL] Iter {iteration:4d}/{args_cli.max_iterations} | "
                  f"Steps: {total_steps:8d} | "
                  f"Mean Ep Len: {mean_reward:.1f} | "
                  f"Value Loss: {mean_value_loss:.4f} | "
                  f"Policy Loss: {mean_surrogate_loss:.4f}")

        # Save checkpoint
        if iteration % 50 == 0 or iteration == args_cli.max_iterations - 1:
            checkpoint_path = os.path.join(log_dir, f"model_{iteration}.pt")
            torch.save({
                'model_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'iter': iteration,
                'infos': {
                    'bc_checkpoint': args_cli.bc_checkpoint,
                    'domain_randomization': args_cli.domain_randomization,
                }
            }, checkpoint_path)
            print(f"[RL] Saved checkpoint: {checkpoint_path}")

    print("\n" + "=" * 60)
    print(f"[RL] Training complete!")
    print(f"[RL] Final checkpoint: {log_dir}/model_{args_cli.max_iterations - 1}.pt")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
