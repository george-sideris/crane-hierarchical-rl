"""
PointNet encoder for point cloud observations.

Based on: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
https://arxiv.org/abs/1612.00593
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """PointNet encoder that maps point clouds to fixed-size latent vectors.

    Input: (batch, N, 3) point cloud
    Output: (batch, latent_dim) latent vector
    """

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 128,
        hidden_dims: list = [64, 128, 256],
        use_batch_norm: bool = True,
    ):
        """Initialize PointNet encoder.

        Args:
            input_dim: Input point dimension (3 for XYZ, 6 for XYZ+RGB, etc.)
            latent_dim: Output latent dimension
            hidden_dims: Hidden layer dimensions for per-point MLP
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Per-point MLP (shared across all points)
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        self.point_mlp = nn.ModuleList(layers)
        self.final_hidden_dim = hidden_dims[-1]

        # Final projection to latent space after max pooling
        self.fc_out = nn.Sequential(
            nn.Linear(self.final_hidden_dim, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Point cloud tensor of shape (batch, N, input_dim)

        Returns:
            Latent vector of shape (batch, latent_dim)
        """
        batch_size, num_points, _ = x.shape

        # Reshape for batch norm: (batch * N, dim)
        x = x.view(batch_size * num_points, -1)

        # Per-point MLP
        for layer in self.point_mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x)

        # Reshape back: (batch, N, hidden_dim)
        x = x.view(batch_size, num_points, -1)

        # Max pooling over points (permutation invariant)
        x = torch.max(x, dim=1)[0]  # (batch, hidden_dim)

        # Final projection
        x = self.fc_out(x)  # (batch, latent_dim)

        return x


class PointNetEncoderLight(nn.Module):
    """Lightweight PointNet encoder for faster training.

    Simpler architecture without batch norm for easier use in RL.
    """

    def __init__(
        self,
        input_dim: int = 3,
        latent_dim: int = 64,
        hidden_dims: list = [64, 128],
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Simple per-point MLP
        self.mlp1 = nn.Linear(input_dim, hidden_dims[0])
        self.mlp2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        # After max pooling
        self.fc = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Point cloud (batch, N, input_dim)

        Returns:
            Latent vector (batch, latent_dim)
        """
        # Per-point features
        x = F.relu(self.mlp1(x))  # (batch, N, 64)
        x = F.relu(self.mlp2(x))  # (batch, N, 128)

        # Max pooling (permutation invariant)
        x = torch.max(x, dim=1)[0]  # (batch, 128)

        # Final projection
        x = F.relu(self.fc(x))  # (batch, latent_dim)

        return x


class PointCloudObservationEncoder(nn.Module):
    """Combined encoder for point cloud + proprioceptive observations.

    Used as the observation encoder in the RL policy.
    """

    def __init__(
        self,
        num_points: int = 512,
        point_dim: int = 3,
        proprio_dim: int = 10,
        pointnet_latent: int = 64,
        combined_latent: int = 128,
    ):
        """Initialize combined encoder.

        Args:
            num_points: Number of points in point cloud observation
            point_dim: Dimension per point (3 for XYZ)
            proprio_dim: Dimension of proprioceptive observation
            pointnet_latent: PointNet output dimension
            combined_latent: Final combined latent dimension
        """
        super().__init__()

        self.num_points = num_points
        self.point_dim = point_dim
        self.proprio_dim = proprio_dim

        # Point cloud encoder
        self.pointnet = PointNetEncoderLight(
            input_dim=point_dim,
            latent_dim=pointnet_latent,
        )

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Combined projection
        self.combined = nn.Sequential(
            nn.Linear(pointnet_latent + 32, combined_latent),
            nn.ReLU(inplace=True),
        )

        self.output_dim = combined_latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Combined observation tensor of shape (batch, num_points * point_dim + proprio_dim)
                 First num_points * point_dim elements are flattened point cloud
                 Last proprio_dim elements are proprioceptive state

        Returns:
            Encoded observation (batch, combined_latent)
        """
        batch_size = obs.shape[0]

        # Split observation
        pc_flat_dim = self.num_points * self.point_dim
        pc_flat = obs[:, :pc_flat_dim]
        proprio = obs[:, pc_flat_dim:]

        # Reshape point cloud
        pc = pc_flat.view(batch_size, self.num_points, self.point_dim)

        # Encode
        pc_latent = self.pointnet(pc)  # (batch, pointnet_latent)
        proprio_latent = self.proprio_encoder(proprio)  # (batch, 32)

        # Combine
        combined = torch.cat([pc_latent, proprio_latent], dim=-1)
        output = self.combined(combined)

        return output


# Test
if __name__ == "__main__":
    # Test PointNet encoder
    batch_size = 4
    num_points = 512

    encoder = PointNetEncoderLight(input_dim=3, latent_dim=64)
    x = torch.randn(batch_size, num_points, 3)
    z = encoder(x)
    print(f"PointNetLight: {x.shape} -> {z.shape}")

    # Test combined encoder
    proprio_dim = 10
    combined_encoder = PointCloudObservationEncoder(
        num_points=num_points,
        point_dim=3,
        proprio_dim=proprio_dim,
        pointnet_latent=64,
        combined_latent=128,
    )

    obs = torch.randn(batch_size, num_points * 3 + proprio_dim)
    z = combined_encoder(obs)
    print(f"Combined encoder: {obs.shape} -> {z.shape}")
    print(f"Output dim: {combined_encoder.output_dim}")
