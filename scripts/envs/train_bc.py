#!/usr/bin/env python3
"""
Behavioral Cloning Training Script

Trains a policy network via supervised learning to imitate the expert heuristic.
The trained policy can then be fine-tuned with RL.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path


class BCDataset(Dataset):
    """Dataset for behavioral cloning."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """
    Simple MLP policy for behavioral cloning.

    Architecture matches RSL-RL's actor network for easy weight transfer.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [128, 64]):
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


def train_bc(
    data_dir: str,
    output_dir: str,
    hidden_dims: list = [128, 64],
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    val_split: float = 0.1,
    device: str = "cuda:0",
):
    """Train a BC policy on collected demonstrations."""

    print(f"[BC Train] Loading data from {data_dir}/")

    # Load data
    obs = np.load(os.path.join(data_dir, "observations.npy"))
    actions = np.load(os.path.join(data_dir, "actions.npy"))

    print(f"[BC Train] Loaded {len(obs)} samples")
    print(f"           Observation shape: {obs.shape}")
    print(f"           Action shape: {actions.shape}")

    # Create dataset
    dataset = BCDataset(obs, actions)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"[BC Train] Train samples: {train_size}, Val samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    policy = BCPolicy(obs_dim, action_dim, hidden_dims).to(device)

    print(f"[BC Train] Policy architecture: {obs_dim} -> {hidden_dims} -> {action_dim}")

    # Optimizer and loss
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # Train
        policy.train()
        train_loss = 0.0
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            pred_actions = policy(batch_obs)
            loss = criterion(pred_actions, batch_actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)

                pred_actions = policy(batch_obs)
                loss = criterion(pred_actions, batch_actions)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'hidden_dims': hidden_dims,
            }, os.path.join(output_dir, "bc_policy_best.pt"))

        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"[BC Train] Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Best Val: {best_val_loss:.6f}")

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'hidden_dims': hidden_dims,
    }, os.path.join(output_dir, "bc_policy_final.pt"))

    print(f"\n[BC Train] Training complete!")
    print(f"           Best validation loss: {best_val_loss:.6f}")
    print(f"           Models saved to {output_dir}/")

    return policy


def convert_to_rsl_rl_format(bc_checkpoint: str, output_path: str, device: str = "cuda:0"):
    """
    Convert BC policy checkpoint to RSL-RL compatible format.

    This allows loading the BC-trained weights into RSL-RL for fine-tuning.
    """
    print(f"[BC Convert] Loading BC checkpoint from {bc_checkpoint}")

    checkpoint = torch.load(bc_checkpoint, map_location=device)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    hidden_dims = checkpoint['hidden_dims']

    # Load BC policy
    bc_policy = BCPolicy(obs_dim, action_dim, hidden_dims)
    bc_policy.load_state_dict(checkpoint['model_state_dict'])

    # Create RSL-RL compatible state dict
    # RSL-RL uses: actor.0.weight, actor.0.bias, actor.2.weight, etc.
    rsl_rl_state_dict = {}

    # Map BC network layers to RSL-RL format
    layer_idx = 0
    for name, param in bc_policy.network.named_parameters():
        # BC uses: network.0.weight, network.0.bias, network.2.weight, etc.
        # RSL-RL uses: actor.0.weight, actor.0.bias, etc.
        new_name = name.replace("network.", "actor.")
        rsl_rl_state_dict[new_name] = param.data.clone()
        print(f"  {name} -> {new_name}: {param.shape}")

    # Save in RSL-RL format
    # RSL-RL expects: {'model_state_dict': {...}, 'optimizer_state_dict': {...}}
    rsl_rl_checkpoint = {
        'model_state_dict': {
            'actor': rsl_rl_state_dict,
            # Critic will be randomly initialized during RL fine-tuning
        },
        'optimizer_state_dict': {},
        'iter': 0,
        'infos': {'bc_pretrained': True},
    }

    torch.save(rsl_rl_checkpoint, output_path)
    print(f"[BC Convert] Saved RSL-RL compatible checkpoint to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BC policy from demonstrations")
    parser.add_argument("--data_dir", type=str, default="bc_data", help="Directory with collected data")
    parser.add_argument("--output_dir", type=str, default="bc_policy", help="Output directory for trained policy")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 64], help="Hidden layer dimensions")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--convert_rsl_rl", action="store_true", help="Also convert to RSL-RL format")
    args = parser.parse_args()

    # Train BC policy
    policy = train_bc(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
    )

    # Convert to RSL-RL format if requested
    if args.convert_rsl_rl:
        bc_checkpoint = os.path.join(args.output_dir, "bc_policy_best.pt")
        rsl_rl_path = os.path.join(args.output_dir, "bc_policy_rsl_rl.pt")
        convert_to_rsl_rl_format(bc_checkpoint, rsl_rl_path, args.device)
