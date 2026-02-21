#!/usr/bin/env python3
"""Visualize depth observations from BC training.

Usage:
    # Visualize latest training run
    python scripts/visualize_depth_observations.py --log_dir logs/bc_depth/latest

    # Specific run with custom settings
    python scripts/visualize_depth_observations.py \
        --log_dir logs/bc_depth/bc_20240115_143022 \
        --img_size 128 \
        --num_samples 32 \
        --output_dir depth_analysis
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_observations(log_dir: str):
    """Load depth observations from training logs."""
    log_path = Path(log_dir)

    # Handle 'latest' symlink
    if log_path.name == "latest" and log_path.is_symlink():
        log_path = log_path.resolve()
        print(f"Resolved 'latest' to: {log_path}")

    obs_path = log_path / "depth_observations.npy"
    if not obs_path.exists():
        # Try alternative names
        for alt_name in ["observations.npy", "obs.npy", "depth_obs.npy"]:
            alt_path = log_path / alt_name
            if alt_path.exists():
                obs_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"No depth observations found at {log_path}. "
                f"Looked for: depth_observations.npy, observations.npy, obs.npy"
            )

    print(f"Loading: {obs_path}")
    return np.load(obs_path)


def load_metadata(log_dir: str):
    """Load metadata if available."""
    log_path = Path(log_dir)
    if log_path.name == "latest" and log_path.is_symlink():
        log_path = log_path.resolve()

    metadata_path = log_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def reshape_to_images(obs: np.ndarray, img_size: int = 128):
    """Reshape flattened observations to images."""
    n_samples = obs.shape[0]
    obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]

    # Infer image size if not specified
    inferred_size = int(np.sqrt(obs_dim))
    if inferred_size * inferred_size != obs_dim:
        raise ValueError(
            f"Cannot reshape observations of dim {obs_dim} to square images. "
            f"Try specifying --img_size explicitly."
        )

    if img_size != inferred_size:
        print(f"Warning: Specified img_size={img_size} but data suggests {inferred_size}. Using {inferred_size}.")
        img_size = inferred_size

    return obs.reshape(n_samples, img_size, img_size), img_size


def plot_observation_grid(images: np.ndarray,
                          indices: list,
                          output_path: str,
                          title: str = "Depth Observations"):
    """Plot grid of observations at specified indices."""
    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        if idx < len(images):
            im = ax.imshow(images[idx], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Sample {idx}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.axis('off')

    # Hide empty axes
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_histogram(images: np.ndarray, output_path: str):
    """Plot histogram of depth values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of all values
    ax = axes[0]
    ax.hist(images.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Depth Value (normalized)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of All Depth Values')
    ax.axvline(x=0, color='red', linestyle='--', label='Min (0)')
    ax.axvline(x=1, color='green', linestyle='--', label='Max/Background (1)')
    ax.legend()

    # Per-image mean distribution
    ax = axes[1]
    per_image_means = images.mean(axis=(1, 2))
    ax.hist(per_image_means, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Per-Image Mean Depth')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-Image Mean Values')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def compute_statistics(images: np.ndarray):
    """Compute observation statistics."""
    stats = {
        "shape": str(images.shape),
        "min": float(images.min()),
        "max": float(images.max()),
        "mean": float(images.mean()),
        "std": float(images.std()),
        "zeros_pct": float((images == 0).mean() * 100),
        "ones_pct": float((images == 1).mean() * 100),
        "near_ones_pct": float((images > 0.99).mean() * 100),
        "valid_depth_pct": float(((images > 0) & (images < 1)).mean() * 100),
    }

    # Per-image statistics
    per_image_means = images.mean(axis=(1, 2))
    per_image_stds = images.std(axis=(1, 2))
    stats["per_image_mean_avg"] = float(per_image_means.mean())
    stats["per_image_mean_std"] = float(per_image_means.std())
    stats["per_image_std_avg"] = float(per_image_stds.mean())

    return stats


def diagnose_observations(stats: dict):
    """Diagnose potential issues with the observations."""
    issues = []
    recommendations = []

    # Check if all values are near 1 (background)
    if stats["ones_pct"] > 80:
        issues.append("HIGH BACKGROUND: >80% of pixels are exactly 1.0 (background)")
        recommendations.append("Check semantic masking - logs may not be visible")

    if stats["near_ones_pct"] > 90:
        issues.append("MOSTLY BACKGROUND: >90% of pixels are near 1.0")
        recommendations.append("Camera may be too far from logs or depth range is wrong")

    # Check if all zeros
    if stats["zeros_pct"] > 50:
        issues.append("MANY ZEROS: >50% of pixels are 0.0")
        recommendations.append("Logs may be too close (below depth_min)")

    # Check variance
    if stats["std"] < 0.05:
        issues.append("LOW VARIANCE: std < 0.05")
        recommendations.append("Observations lack diversity - may not be informative")

    if stats["per_image_std_avg"] < 0.02:
        issues.append("LOW PER-IMAGE VARIANCE: avg per-image std < 0.02")
        recommendations.append("Individual images are too uniform")

    # Check valid depth range
    if stats["valid_depth_pct"] < 10:
        issues.append("FEW VALID DEPTHS: <10% of pixels in (0, 1)")
        recommendations.append("Most pixels are clipped - adjust depth_min/depth_max")

    return issues, recommendations


def main():
    parser = argparse.ArgumentParser(description="Visualize depth observations from BC training")
    parser.add_argument("--log_dir", type=str,
                       default="logs/bc_depth/latest",
                       help="Path to training log directory")
    parser.add_argument("--img_size", type=int, default=128,
                       help="Image size (128 or 256)")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to visualize in grid")
    parser.add_argument("--output_dir", type=str,
                       default="/workspace/crane_testbed/depth_viz",
                       help="Output directory for visualizations")
    args = parser.parse_args()

    # Load and reshape
    print(f"Loading from {args.log_dir}...")
    obs = load_observations(args.log_dir)
    print(f"Loaded {len(obs)} observations, shape: {obs.shape}")

    # Load metadata if available
    metadata = load_metadata(args.log_dir)
    if metadata:
        print(f"Metadata: {json.dumps(metadata, indent=2)}")

    images, actual_size = reshape_to_images(obs, args.img_size)
    print(f"Reshaped to: {images.shape} (img_size={actual_size})")

    # Statistics
    stats = compute_statistics(images)
    print(f"\n{'='*60}")
    print("OBSERVATION STATISTICS")
    print('='*60)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Diagnose issues
    issues, recommendations = diagnose_observations(stats)
    if issues:
        print(f"\n{'='*60}")
        print("POTENTIAL ISSUES DETECTED")
        print('='*60)
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print(f"\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  → {rec}")
    else:
        print(f"\n✅ Observations look reasonable!")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Plot samples evenly distributed
    n_total = len(images)
    n_samples = min(args.num_samples, n_total)
    indices = [int(i * n_total / n_samples) for i in range(n_samples)]

    plot_observation_grid(
        images, indices,
        str(output_dir / "depth_grid.png"),
        f"Depth Observations (n={n_total}, size={actual_size}x{actual_size})"
    )

    # Plot early vs late
    n_early = min(8, n_total)
    n_late = min(8, n_total)
    early = list(range(n_early))
    late = list(range(max(0, n_total - n_late), n_total))

    plot_observation_grid(images, early,
                         str(output_dir / "depth_early.png"),
                         "Early Episode Observations (First 8)")
    plot_observation_grid(images, late,
                         str(output_dir / "depth_late.png"),
                         "Late Episode Observations (Last 8)")

    # Plot histogram
    plot_histogram(images, str(output_dir / "depth_histogram.png"))

    # Save stats to JSON
    stats_path = output_dir / "depth_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to: {stats_path}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}/")
    print('='*60)
    print("Files:")
    print("  - depth_grid.png: Grid of samples across training")
    print("  - depth_early.png: First 8 observations")
    print("  - depth_late.png: Last 8 observations")
    print("  - depth_histogram.png: Value distribution")
    print("  - depth_stats.json: Detailed statistics")


if __name__ == "__main__":
    main()
