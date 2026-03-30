"""SB3 features extractor wrapping the PointNet encoder.

Usage in train_sac.py:
    from crane_testbed.agents.pointnet_extractor_sb3 import PointNetFeaturesExtractor

    policy_kwargs = {
        "features_extractor_class": PointNetFeaturesExtractor,
        "features_extractor_kwargs": {"num_points": 1024, "output_dim": 256, "norm_type": "layernorm"},
    }
    agent = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from crane_testbed.agents.pointnet_actor_critic import PointNetEncoder


class PointNetFeaturesExtractor(BaseFeaturesExtractor):
    """Reshapes flat (batch, num_points*3) obs to (batch, num_points, 3) and
    passes through PointNetEncoder -> (batch, output_dim)."""

    def __init__(self, observation_space: gym.Space, num_points: int = 1024,
                 output_dim: int = 256, norm_type: str = "layernorm"):
        super().__init__(observation_space, features_dim=output_dim)
        self.num_points = num_points
        self.encoder = PointNetEncoder(input_dim=3, output_dim=output_dim, norm_type=norm_type)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, num_points * 3) flat from SB3
        x = observations.view(-1, self.num_points, 3)
        return self.encoder(x)
