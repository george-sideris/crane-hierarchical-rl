"""
Custom Actor-Critic network with PointNet encoder for point cloud observations.

Compatible with RSL-RL training framework.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class PointNetActorCritic(nn.Module):
    """Actor-Critic network that processes point cloud observations with PointNet.

    Architecture:
        Point Cloud (N, 3) -> PointNet -> latent (64)
        Proprioception (P,) -> MLP -> latent (32)
        Combined latent (96) -> Actor MLP -> action mean
                             -> Critic MLP -> value
    """

    def __init__(
        self,
        num_points: int,
        proprio_dim: int,
        num_actions: int,
        pointnet_latent: int = 64,
        proprio_latent: int = 32,
        actor_hidden: list = [128, 64],
        critic_hidden: list = [128, 64],
        init_noise_std: float = 1.0,
    ):
        """Initialize PointNet Actor-Critic.

        Args:
            num_points: Number of points in point cloud
            proprio_dim: Proprioception dimension
            num_actions: Action dimension
            pointnet_latent: PointNet output dimension
            proprio_latent: Proprioception encoder output dimension
            actor_hidden: Actor MLP hidden layers
            critic_hidden: Critic MLP hidden layers
            init_noise_std: Initial action noise standard deviation
        """
        super().__init__()

        self.num_points = num_points
        self.proprio_dim = proprio_dim
        self.num_actions = num_actions

        # PointNet encoder (per-point MLP + max pool)
        self.point_mlp1 = nn.Linear(3, 64)
        self.point_mlp2 = nn.Linear(64, 128)
        self.point_mlp3 = nn.Linear(128, pointnet_latent)

        # Proprioception encoder
        if proprio_dim > 0:
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, 32),
                nn.ELU(),
                nn.Linear(32, proprio_latent),
                nn.ELU(),
            )
        else:
            self.proprio_encoder = None
            proprio_latent = 0

        combined_dim = pointnet_latent + proprio_latent

        # Actor network
        actor_layers = []
        in_dim = combined_dim
        for hidden_dim in actor_hidden:
            actor_layers.append(nn.Linear(in_dim, hidden_dim))
            actor_layers.append(nn.ELU())
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic network
        critic_layers = []
        in_dim = combined_dim
        for hidden_dim in critic_hidden:
            critic_layers.append(nn.Linear(in_dim, hidden_dim))
            critic_layers.append(nn.ELU())
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Action noise (learnable log std)
        self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))

        # Store dimensions for easy access
        self.pointnet_latent = pointnet_latent
        self.proprio_latent = proprio_latent

    def _encode_pointcloud(self, pc: torch.Tensor) -> torch.Tensor:
        """Encode point cloud with PointNet.

        Args:
            pc: Point cloud (batch, N, 3)

        Returns:
            Latent (batch, pointnet_latent)
        """
        # Per-point MLP
        x = torch.relu(self.point_mlp1(pc))  # (batch, N, 64)
        x = torch.relu(self.point_mlp2(x))   # (batch, N, 128)
        x = torch.relu(self.point_mlp3(x))   # (batch, N, pointnet_latent)

        # Max pooling over points
        x = torch.max(x, dim=1)[0]  # (batch, pointnet_latent)

        return x

    def _encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode full observation (point cloud + proprioception).

        Args:
            obs: Flattened observation (batch, num_points * 3 + proprio_dim)

        Returns:
            Combined latent (batch, combined_dim)
        """
        batch_size = obs.shape[0]

        # Split observation
        pc_dim = self.num_points * 3
        pc_flat = obs[:, :pc_dim]
        proprio = obs[:, pc_dim:] if self.proprio_dim > 0 else None

        # Reshape point cloud
        pc = pc_flat.view(batch_size, self.num_points, 3)

        # Encode point cloud
        pc_latent = self._encode_pointcloud(pc)

        # Encode proprioception
        if proprio is not None and self.proprio_encoder is not None:
            proprio_latent = self.proprio_encoder(proprio)
            combined = torch.cat([pc_latent, proprio_latent], dim=-1)
        else:
            combined = pc_latent

        return combined

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns action mean.

        Args:
            obs: Observation (batch, obs_dim)

        Returns:
            Action mean (batch, num_actions)
        """
        latent = self._encode_observation(obs)
        action_mean = self.actor(latent)
        return action_mean

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy.

        Args:
            obs: Observation (batch, obs_dim)
            deterministic: If True, return mean action

        Returns:
            Action (batch, num_actions)
        """
        action_mean = self.forward(obs)

        if deterministic:
            return action_mean

        # Sample from Gaussian
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()

        return action

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action for inference."""
        return self.act(obs, deterministic=True)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """Evaluate actions for PPO update.

        Args:
            obs: Observations (batch, obs_dim)
            actions: Actions taken (batch, num_actions)

        Returns:
            values: Value estimates (batch, 1)
            log_probs: Log probabilities of actions (batch, num_actions)
            entropy: Action entropy (batch,)
        """
        latent = self._encode_observation(obs)

        # Actor
        action_mean = self.actor(latent)
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().sum(dim=-1)

        # Critic
        values = self.critic(latent)

        return values, log_probs, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate.

        Args:
            obs: Observation (batch, obs_dim)

        Returns:
            Value (batch, 1)
        """
        latent = self._encode_observation(obs)
        value = self.critic(latent)
        return value


class PointNetActorCriticFactory:
    """Factory class for creating PointNet Actor-Critic compatible with RSL-RL."""

    def __init__(
        self,
        num_points: int = 512,
        proprio_dim: int = 6,
        pointnet_latent: int = 64,
        proprio_latent: int = 32,
        actor_hidden: list = None,
        critic_hidden: list = None,
        init_noise_std: float = 1.0,
    ):
        self.num_points = num_points
        self.proprio_dim = proprio_dim
        self.pointnet_latent = pointnet_latent
        self.proprio_latent = proprio_latent
        self.actor_hidden = actor_hidden or [128, 64]
        self.critic_hidden = critic_hidden or [128, 64]
        self.init_noise_std = init_noise_std

    def __call__(self, num_actions: int, device: str = "cuda") -> PointNetActorCritic:
        """Create actor-critic network.

        Args:
            num_actions: Action dimension
            device: Device to create network on

        Returns:
            PointNetActorCritic instance
        """
        model = PointNetActorCritic(
            num_points=self.num_points,
            proprio_dim=self.proprio_dim,
            num_actions=num_actions,
            pointnet_latent=self.pointnet_latent,
            proprio_latent=self.proprio_latent,
            actor_hidden=self.actor_hidden,
            critic_hidden=self.critic_hidden,
            init_noise_std=self.init_noise_std,
        )
        return model.to(device)


# Test
if __name__ == "__main__":
    batch_size = 4
    num_points = 512
    proprio_dim = 6
    num_actions = 4

    obs_dim = num_points * 3 + proprio_dim

    # Create network
    network = PointNetActorCritic(
        num_points=num_points,
        proprio_dim=proprio_dim,
        num_actions=num_actions,
    )

    # Test forward
    obs = torch.randn(batch_size, obs_dim)
    action = network.act(obs)
    print(f"Observation: {obs.shape}")
    print(f"Action: {action.shape}")

    # Test evaluate
    values, log_probs, entropy = network.evaluate(obs, action)
    print(f"Values: {values.shape}")
    print(f"Log probs: {log_probs.shape}")
    print(f"Entropy: {entropy.shape}")

    # Test factory
    factory = PointNetActorCriticFactory(num_points=num_points, proprio_dim=proprio_dim)
    network2 = factory(num_actions=num_actions, device="cpu")
    print(f"Factory created network: {type(network2)}")

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Total parameters: {total_params:,}")
