"""
PointNet Actor-Critic for point cloud observations.

Compatible with RSL-RL's ActorCritic interface.
Supports:
  - BatchNorm (default for compat) or LayerNorm via norm_type parameter
  - Asymmetric actor-critic: PointNet actor with state-based MLP critic
  - Separate encoder learning rate via get_param_groups()
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class PointNetEncoder(nn.Module):
    """PointNet encoder for point cloud feature extraction.

    Architecture: SharedMLP per point -> MaxPool -> FC projection

    Args:
        input_dim: Per-point feature dimension (default 3 for XYZ).
        output_dim: Output latent dimension.
        norm_type: "batchnorm" (default for compat) or "layernorm" (recommended for RL).
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 256, norm_type: str = "batchnorm"):
        super().__init__()

        self.norm_type = norm_type
        norm_cls = nn.LayerNorm if norm_type == "layernorm" else nn.BatchNorm1d

        # Shared MLP (applied per-point)
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            norm_cls(64),
            nn.ELU(),
            nn.Linear(64, 128),
            norm_cls(128),
            nn.ELU(),
            nn.Linear(128, 256),
            norm_cls(256),
            nn.ELU(),
        )

        # After max-pooling, project to output dimension
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            norm_cls(output_dim),
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

        # Reshape to (batch * points, features) for per-point MLP
        # Required for BatchNorm1d; also works with LayerNorm
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
        Actor: Points (N, 3) -> PointNet Encoder -> Latent -> MLP -> Actions
        Critic (symmetric):  Points (N, 3) -> shared PointNet -> Latent -> MLP -> Value
        Critic (asymmetric): State (D,) -> MLP Encoder -> Latent -> MLP -> Value

    Asymmetric mode activates when num_critic_obs != num_actor_obs.
    """

    @staticmethod
    def _to_tensor(obs):
        """Extract raw tensor from dict/tensordict observations.

        Newer RSL-RL versions pass observations as {'policy': tensor} dicts
        or TensorDicts instead of raw tensors.
        """
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, dict):
            return obs.get('policy', next(iter(obs.values())))
        # TensorDict or similar — try dict-like access
        if hasattr(obs, 'get'):
            try:
                return obs.get('policy')
            except Exception:
                pass
        if hasattr(obs, '__getitem__'):
            try:
                return obs['policy']
            except Exception:
                pass
        return obs

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions: int,
        num_points: int = 1024,
        encoder_features: int = 256,
        actor_hidden_dims: list = [128, 64],
        critic_hidden_dims: list = [128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        norm_type: str = "batchnorm",
        encoder_lr_scale: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        # Handle TensorDict obs specs from newer RSL-RL (extract integer sizes)
        if not isinstance(num_actor_obs, (int, float)):
            if hasattr(num_actor_obs, 'shape'):
                num_actor_obs = num_actor_obs.shape[-1]
            elif isinstance(num_actor_obs, dict) or hasattr(num_actor_obs, 'get'):
                try:
                    pol = num_actor_obs.get('policy', num_actor_obs)
                    num_actor_obs = pol.shape[-1] if hasattr(pol, 'shape') else int(pol)
                except Exception:
                    num_actor_obs = int(num_actor_obs)
        if not isinstance(num_critic_obs, (int, float)):
            if hasattr(num_critic_obs, 'shape'):
                num_critic_obs = num_critic_obs.shape[-1]
            elif isinstance(num_critic_obs, dict) or hasattr(num_critic_obs, 'get'):
                try:
                    crt = num_critic_obs.get('critic', num_critic_obs.get('policy', num_critic_obs))
                    num_critic_obs = crt.shape[-1] if hasattr(crt, 'shape') else int(crt)
                except Exception:
                    num_critic_obs = int(num_critic_obs)
        num_actor_obs = int(num_actor_obs)
        num_critic_obs = int(num_critic_obs)

        self.num_points = num_points
        self.encoder_lr_scale = encoder_lr_scale

        # Activation
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # PointNet encoder for actor (and critic if symmetric)
        self.encoder = PointNetEncoder(input_dim=3, output_dim=encoder_features, norm_type=norm_type)

        # Actor MLP
        actor_layers = []
        in_dim = encoder_features
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(in_dim, hidden_dim), act_fn()])
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Asymmetric critic: state-based MLP when critic obs differs from actor obs
        self._asymmetric = (num_critic_obs != num_actor_obs) and (num_critic_obs > 0)
        if self._asymmetric:
            # Critic encoder: state vector -> latent
            self.critic_encoder = nn.Sequential(
                nn.Linear(num_critic_obs, 256),
                act_fn(),
                nn.Linear(256, encoder_features),
                act_fn(),
            )
            critic_in_dim = encoder_features
        else:
            critic_in_dim = encoder_features

        # Critic MLP
        critic_layers = []
        in_dim = critic_in_dim
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

        mode = "asymmetric (state-based critic)" if self._asymmetric else "symmetric"
        print(f"[PointNetActorCritic] Created: {num_points} points -> {encoder_features} latent -> {num_actions} actions")
        print(f"[PointNetActorCritic] norm={norm_type}, mode={mode}, encoder_lr_scale={encoder_lr_scale}")

    def _encode(self, obs) -> torch.Tensor:
        """Encode observation through PointNet."""
        obs = self._to_tensor(obs)
        batch_size = obs.shape[0]

        # Handle flattened input
        if obs.dim() == 2:
            obs = obs.view(batch_size, self.num_points, 3)

        return self.encoder(obs)

    def reset(self, dones=None):
        """Reset internal state (no-op for this architecture)."""
        pass

    def update_normalization(self, obs=None, critic_obs=None):
        """Update observation normalization (no-op — point clouds are already in base frame)."""
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
        """Update action distribution given observations."""
        latent = self._encode(observations)
        mean = self.actor(latent)
        self.distribution = Normal(mean, self.std)

    def act(self, observations, **kwargs) -> torch.Tensor:
        """Sample action from distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations) -> torch.Tensor:
        """Deterministic action for inference."""
        latent = self._encode(observations)
        return self.actor(latent)

    def evaluate(self, critic_observations, **kwargs) -> torch.Tensor:
        """Get value estimate.

        In asymmetric mode, critic_observations is the state vector (not point cloud).
        In symmetric mode, critic_observations is the same point cloud as the actor.
        """
        if self._asymmetric:
            obs = self._to_tensor(critic_observations)
            latent = self.critic_encoder(obs)
        else:
            latent = self._encode(critic_observations)
        return self.critic(latent)

    def get_param_groups(self, base_lr: float) -> list:
        """Return parameter groups with lower encoder learning rate.

        Usage: pass to optimizer instead of model.parameters()
            optimizer = Adam(model.get_param_groups(lr=3e-4))
        """
        encoder_lr = base_lr * self.encoder_lr_scale
        groups = [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": list(self.actor.parameters()), "lr": base_lr},
            {"params": list(self.critic.parameters()), "lr": base_lr},
            {"params": [self.std], "lr": base_lr},
        ]
        if self._asymmetric:
            groups.append({"params": list(self.critic_encoder.parameters()), "lr": base_lr})
        return groups


# Factory function for RSL-RL
def create_pointnet_actor_critic(num_actor_obs, num_critic_obs, num_actions, **kwargs):
    """Factory function to create PointNet actor-critic."""
    return PointNetActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        **kwargs,
    )
