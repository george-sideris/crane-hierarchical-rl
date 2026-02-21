"""
PointNet Actor-Critic for point cloud observations.

Compatible with RSL-RL's ActorCritic interface.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class PointNetEncoder(nn.Module):
    """PointNet encoder for point cloud feature extraction.

    Architecture: SharedMLP per point -> MaxPool -> FC projection
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()

        # Shared MLP (applied per-point)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )

        # After max-pooling, project to output dimension
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ELU(),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_points, 3) point cloud

        Returns:
            (batch_size, output_dim) global features
        """
        batch_size, num_points, _ = x.shape

        # Reshape for batch norm: (batch * points, features)
        x = x.view(batch_size * num_points, -1)
        x = self.mlp1(x)

        # Reshape back: (batch, points, features)
        x = x.view(batch_size, num_points, -1)

        # Max pooling across points (global feature)
        x = x.max(dim=1)[0]  # (batch, 256)

        # Final projection
        x = self.fc(x)  # (batch, output_dim)

        return x


class PointNetActorCritic(nn.Module):
    """Actor-Critic with PointNet encoder for point cloud observations.

    Architecture:
        Points (N, 3) -> PointNet Encoder -> Latent (256) -> Actor/Critic MLP
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_points: int = None,
        encoder_features: int = 256,
        actor_hidden_dims: list = [128, 64],
        critic_hidden_dims: list = [128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Auto-detect num_points from observation dimension
        if num_points is None:
            if num_actor_obs % 3 == 0:
                num_points = num_actor_obs // 3
            else:
                num_points = 1024  # Default fallback
            print(f"[PointNetActorCritic] Auto-detected {num_points} points from {num_actor_obs} obs")

        self.num_points = num_points

        # Activation
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # PointNet encoder (shared between actor and critic)
        self.encoder = PointNetEncoder(input_dim=3, output_dim=encoder_features)

        # Actor MLP
        actor_layers = []
        in_dim = encoder_features
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic MLP
        critic_layers = []
        in_dim = encoder_features
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # Disable validation for faster sampling
        Normal.set_default_validate_args(False)

        # RSL-RL compatibility
        self.is_recurrent = False

        print(f"[PointNetActorCritic] Created: {num_points} points -> {encoder_features} latent -> {num_actions} actions")

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation through PointNet."""
        batch_size = obs.shape[0]

        # Handle flattened input
        if obs.dim() == 2:
            obs = obs.view(batch_size, self.num_points, 3)

        return self.encoder(obs)

    def reset(self, dones=None):
        """Reset internal state (no-op for this architecture)."""
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

    def update_distribution(self, observations: torch.Tensor):
        """Update action distribution given observations."""
        latent = self._encode(observations)
        mean = self.actor(latent)
        self.distribution = Normal(mean, self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample action from distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference."""
        latent = self._encode(observations)
        return self.actor(latent)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Get value estimate."""
        latent = self._encode(critic_observations)
        return self.critic(latent)


# Factory function for RSL-RL
def create_pointnet_actor_critic(num_actor_obs, num_critic_obs, num_actions, **kwargs):
    """Factory function to create PointNet actor-critic."""
    return PointNetActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        **kwargs,
    )
