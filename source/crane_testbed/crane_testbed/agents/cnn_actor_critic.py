"""
CNN Actor-Critic for depth image observations.

Compatible with RSL-RL's ActorCritic interface.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class CNNActorCritic(nn.Module):
    """Actor-Critic with CNN encoder for depth observations.

    Architecture (48x48):
        Depth (1, 48, 48) -> CNN Encoder -> Latent (256) -> Actor/Critic MLP

    Architecture (128x128):
        Depth (1, 128, 128) -> CNN Encoder -> Latent (256) -> Actor/Critic MLP
    """

    @staticmethod
    def _to_tensor(obs):
        """Extract raw tensor from dict/tensordict observations."""
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, dict):
            return obs.get('policy', next(iter(obs.values())))
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
        img_height: int = 48,
        img_width: int = 48,
        encoder_features: int = 256,
        actor_hidden_dims: list = [128, 64],
        critic_hidden_dims: list = [128, 64],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width

        # Activation
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ELU

        # CNN encoder - supports 48x48, 128x128, and 256x256
        # Added BatchNorm after each Conv2d for training stability
        if img_height == 48 and img_width == 48:
            # 48x48 encoder: 48 -> 24 -> 12 -> 10 -> 8
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 48 -> 24
                nn.BatchNorm2d(32),
                act_fn(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 24 -> 12
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 12 -> 10
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # 10 -> 8
                nn.BatchNorm2d(64),
                act_fn(),
            )
            flatten_dim = 64 * 8 * 8
        elif img_height == 128 and img_width == 128:
            # 128x128 encoder: 128 -> 64 -> 32 -> 16 -> 8
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 128 -> 64
                nn.BatchNorm2d(32),
                act_fn(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 64 -> 64
                nn.BatchNorm2d(32),
                act_fn(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 32
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
                nn.BatchNorm2d(128),
                act_fn(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
                nn.BatchNorm2d(128),
                act_fn(),
            )
            flatten_dim = 128 * 8 * 8
        else:
            # 256x256 encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 256 -> 128
                nn.BatchNorm2d(32),
                act_fn(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 128 -> 64
                nn.BatchNorm2d(32),
                act_fn(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64 -> 32
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 32
                nn.BatchNorm2d(64),
                act_fn(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 32 -> 16
                nn.BatchNorm2d(128),
                act_fn(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 16 -> 8
                nn.BatchNorm2d(128),
                act_fn(),
            )
            flatten_dim = 128 * 8 * 8
        self.encoder_fc = nn.Sequential(
            nn.Linear(flatten_dim, encoder_features),
            act_fn(),
        )

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

        print(f"[CNNActorCritic] Created: {img_height}x{img_width} -> {encoder_features} -> {num_actions} actions")

    def _encode(self, obs) -> torch.Tensor:
        """Encode observation through CNN."""
        obs = self._to_tensor(obs)
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_height, self.img_width)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.encoder_fc(x)
        return x

    def reset(self, dones=None):
        """Reset internal state (no-op for this architecture)."""
        pass

    def update_normalization(self, obs=None, critic_obs=None):
        """Update observation normalization (no-op)."""
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
        """Get value estimate."""
        latent = self._encode(critic_observations)
        return self.critic(latent)


# Factory function for RSL-RL
def create_cnn_actor_critic(num_actor_obs, num_critic_obs, num_actions, **kwargs):
    """Factory function to create CNN actor-critic."""
    return CNNActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        **kwargs,
    )
