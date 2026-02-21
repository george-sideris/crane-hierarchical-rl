#!/usr/bin/env python3
"""
Convert BC Policy checkpoint to RSL-RL ActorCritic format.

This script converts a BC-trained policy to the format expected by RSL-RL's
ActorCritic model, enabling proper fine-tuning with PPO.

Usage:
    python convert_bc_to_rsl_rl.py \
        --bc_checkpoint logs/bc_policy/latest/bc_policy.pt \
        --output logs/bc_policy/latest/bc_policy_rsl_rl_fixed.pt
"""

import argparse
import os
import torch
import torch.nn as nn


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


def convert_bc_to_rsl_rl(bc_checkpoint_path: str, output_path: str, init_std: float = 0.3):
    """
    Convert BC checkpoint to RSL-RL ActorCritic format.

    Args:
        bc_checkpoint_path: Path to bc_policy.pt
        output_path: Path to save converted checkpoint
        init_std: Initial action noise standard deviation
    """
    print(f"Loading BC checkpoint: {bc_checkpoint_path}")
    bc_ckpt = torch.load(bc_checkpoint_path, map_location='cpu')

    # Extract BC policy info
    obs_dim = bc_ckpt['obs_dim']
    action_dim = bc_ckpt['action_dim']
    hidden_dims = bc_ckpt['hidden_dims']
    bc_state_dict = bc_ckpt['model_state_dict']

    print(f"BC Architecture: {obs_dim} -> {hidden_dims} -> {action_dim}")
    print(f"BC state dict keys: {list(bc_state_dict.keys())}")

    # Create RSL-RL compatible state dict
    rsl_rl_state_dict = {}

    # Map BC weights to actor weights
    # BC uses: network.0.weight, network.0.bias, network.2.weight, etc.
    # RSL-RL expects: actor.0.weight, actor.0.bias, actor.2.weight, etc.
    print("\nMapping BC weights to ActorCritic actor:")
    for bc_key, param in bc_state_dict.items():
        # Convert: network.X.weight -> actor.X.weight
        if bc_key.startswith('network.'):
            actor_key = bc_key.replace('network.', 'actor.')
        else:
            actor_key = f"actor.{bc_key}"
        rsl_rl_state_dict[actor_key] = param.clone()
        print(f"  {bc_key} -> {actor_key}: {param.shape}")

    # Initialize critic with same architecture (random weights)
    print("\nInitializing critic (random weights):")
    for bc_key, param in bc_state_dict.items():
        if bc_key.startswith('network.'):
            critic_key = bc_key.replace('network.', 'critic.')
        else:
            critic_key = f"critic.{bc_key}"

        # Critic has 1 output instead of action_dim for the final layer
        if 'weight' in critic_key and param.shape[0] == action_dim:
            # This is the output layer - critic outputs 1 value
            critic_param = torch.randn(1, param.shape[1]) * 0.01
        elif 'bias' in critic_key and param.shape[0] == action_dim:
            critic_param = torch.zeros(1)
        else:
            critic_param = torch.randn_like(param) * 0.01

        rsl_rl_state_dict[critic_key] = critic_param
        print(f"  {critic_key}: {critic_param.shape}")

    # Initialize action noise std
    rsl_rl_state_dict["std"] = torch.ones(action_dim) * init_std
    print(f"\nInitialized std: {rsl_rl_state_dict['std'].shape} (value={init_std})")

    # Save in RSL-RL format
    # Note: optimizer_state_dict needs proper structure or RSL-RL will fail
    # We create a minimal valid structure that RSL-RL can load
    rsl_rl_ckpt = {
        'model_state_dict': rsl_rl_state_dict,
        'optimizer_state_dict': {
            'state': {},
            'param_groups': [{
                'lr': 3e-4,
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'weight_decay': 0,
                'amsgrad': False,
                'maximize': False,
                'foreach': None,
                'capturable': False,
                'differentiable': False,
                'fused': None,
                'params': []  # Will be populated by RSL-RL
            }]
        },
        'iter': 0,
        'infos': {
            'bc_pretrained': True,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'hidden_dims': hidden_dims,
            'converted_from': bc_checkpoint_path,
        },
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torch.save(rsl_rl_ckpt, output_path)

    print(f"\nSaved RSL-RL checkpoint to: {output_path}")
    print(f"Total keys in state dict: {len(rsl_rl_state_dict)}")

    # Verify the checkpoint
    print("\n=== Verification ===")
    loaded = torch.load(output_path, map_location='cpu')
    print(f"Checkpoint keys: {list(loaded.keys())}")
    print(f"State dict keys: {list(loaded['model_state_dict'].keys())}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert BC checkpoint to RSL-RL format")
    parser.add_argument("--bc_checkpoint", type=str, required=True, help="Path to bc_policy.pt")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: same dir, bc_policy_rsl_rl_fixed.pt)")
    parser.add_argument("--init_std", type=float, default=0.3, help="Initial action noise std")
    args = parser.parse_args()

    if args.output is None:
        bc_dir = os.path.dirname(args.bc_checkpoint)
        args.output = os.path.join(bc_dir, "bc_policy_rsl_rl_fixed.pt")

    convert_bc_to_rsl_rl(args.bc_checkpoint, args.output, args.init_std)


if __name__ == "__main__":
    main()
