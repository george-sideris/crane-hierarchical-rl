"""ResNet-style encoder with CBAM attention for depth images.

This module provides a more advanced encoder architecture for depth-based BC,
featuring residual connections and attention mechanisms to improve feature learning
and reduce overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel attention and spatial attention to help the network
    focus on relevant features (logs) and ignore irrelevant ones (background).

    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # Spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        """Apply channel and spatial attention.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        # Channel attention
        avg_out = self.fc(self.avg_pool(x).flatten(1))
        max_out = self.fc(self.max_pool(x).flatten(1))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att

        # Spatial attention
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        spatial_att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att

        return x


class ResBlock(nn.Module):
    """Residual block with skip connection and CBAM attention.

    The skip connection enables better gradient flow, while CBAM helps the
    network focus on relevant spatial regions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAMBlock(out_channels)

        # Skip connection (projection if dimensions change)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, C_in, H, W]

        Returns:
            Output tensor [B, C_out, H', W']
        """
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)

        out = out + identity
        out = F.relu(out, inplace=True)

        return out


class ResNetDepthEncoder(nn.Module):
    """ResNet-style encoder for depth images with CBAM attention.

    This encoder provides better feature extraction for depth-based BC by:
    1. Using residual connections for better gradient flow
    2. Applying CBAM attention to focus on logs (not background)
    3. Using global average pooling to handle varying log positions

    Architecture:
        Input: [B, 1, 128, 128] depth image
        Conv1: [B, 32, 64, 64]
        Pool:  [B, 32, 32, 32]
        Layer1 (ResBlock): [B, 64, 16, 16]
        Layer2 (ResBlock): [B, 128, 8, 8]
        Layer3 (ResBlock): [B, 256, 4, 4]
        Global Pool: [B, 256]
        FC: [B, latent_dim]
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.latent_dim = latent_dim

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResBlocks with CBAM
        self.layer1 = ResBlock(32, 64, stride=2)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)

        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        """Encode depth image to latent representation.

        Args:
            x: Depth image [B, 1, H, W] (typically [B, 1, 128, 128])

        Returns:
            Latent features [B, latent_dim]
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)

        # ResBlocks with CBAM attention
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Global pooling and projection
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def get_latent_dim(self):
        """Return the dimension of the latent representation."""
        return self.latent_dim


if __name__ == "__main__":
    # Test the encoder
    encoder = ResNetDepthEncoder(latent_dim=256)

    # Test with 128x128 depth image
    x = torch.randn(4, 1, 128, 128)
    out = encoder(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Latent dim: {encoder.get_latent_dim()}")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {total_params:,}")
