#!/usr/bin/env python3
"""Visualize point cloud observations from BC training.

Usage:
    # Visualize latest training run
    python scripts/visualize_pointcloud_observations.py --log_dir logs/bc_pointcloud/latest

    # Specific run with custom settings
    python scripts/visualize_pointcloud_observations.py \
        --log_dir logs/bc_pointcloud/bc_20260217_192651 \
        --num_samples 32 \
        --output_dir /workspace/crane_testbed/pcd_viz
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_observations(log_dir: str):
    """Load point cloud observations from training logs."""
    log_path = Path(log_dir)

    # Handle 'latest' symlink
    if log_path.name == "latest" and log_path.is_symlink():
        log_path = log_path.resolve()
        print(f"Resolved 'latest' to: {log_path}")

    obs_path = log_path / "pointclouds.npy"
    if not obs_path.exists():
        # Try alternative names
        for alt_name in ["pointcloud_observations.npy", "observations.npy", "pcd.npy"]:
            alt_path = log_path / alt_name
            if alt_path.exists():
                obs_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"No point cloud observations found at {log_path}. "
                f"Looked for: pointclouds.npy, pointcloud_observations.npy, observations.npy"
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


def reshape_to_pointclouds(obs: np.ndarray, num_points: int = 1024):
    """Reshape flattened observations to point clouds."""
    # Check if already shaped as (N, num_points, 3)
    if len(obs.shape) == 3 and obs.shape[2] == 3:
        print(f"Data already shaped as point clouds: {obs.shape}")
        actual_num_points = obs.shape[1]
        if num_points != actual_num_points:
            print(f"Note: Specified num_points={num_points} but data has {actual_num_points} points. Using {actual_num_points}.")
        return obs, actual_num_points

    # Otherwise, reshape from flattened format
    n_samples = obs.shape[0]
    obs_dim = obs.shape[1] if len(obs.shape) > 1 else obs.shape[0]

    # Infer num_points if not specified
    if obs_dim % 3 != 0:
        raise ValueError(
            f"Observation dimension {obs_dim} is not divisible by 3 (x,y,z). "
            f"Cannot reshape to point cloud."
        )

    inferred_points = obs_dim // 3
    if num_points != inferred_points:
        print(f"Warning: Specified num_points={num_points} but data suggests {inferred_points}. Using {inferred_points}.")
        num_points = inferred_points

    return obs.reshape(n_samples, num_points, 3), num_points


def plot_pointcloud_grid(pointclouds: np.ndarray,
                          indices: list,
                          output_path: str,
                          title: str = "Point Cloud Observations",
                          max_points_per_plot: int = 2000):
    """Plot grid of point cloud observations."""
    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5*cols, 5*rows))

    for plot_idx, obs_idx in enumerate(indices):
        if obs_idx >= len(pointclouds):
            continue

        ax = fig.add_subplot(rows, cols, plot_idx + 1, projection='3d')
        points = pointclouds[obs_idx]

        # Remove invalid points (zeros or outside reasonable range)
        valid_mask = ~np.all(points == 0, axis=1)
        points = points[valid_mask]

        if len(points) == 0:
            ax.set_title(f"Sample {obs_idx}\n(empty)")
            continue

        # Subsample for faster rendering
        if len(points) > max_points_per_plot:
            idx = np.random.choice(len(points), max_points_per_plot, replace=False)
            points = points[idx]

        # Color by height (Z)
        colors = points[:, 2]

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Sample {obs_idx}\n({len(points)} points)")

        # Equal aspect ratio
        if len(points) > 0:
            max_range = np.max([
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min()
            ]) / 2.0

            mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
            mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
            mid_z = (points[:, 2].max() + points[:, 2].min()) / 2

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_down_view(pointclouds: np.ndarray,
                        indices: list,
                        output_path: str,
                        title: str = "Top-Down View"):
    """Plot top-down 2D views of point clouds."""
    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for plot_idx, obs_idx in enumerate(indices):
        ax = axes[plot_idx]
        if obs_idx >= len(pointclouds):
            ax.axis('off')
            continue

        points = pointclouds[obs_idx]

        # Remove invalid points
        valid_mask = ~np.all(points == 0, axis=1)
        points = points[valid_mask]

        if len(points) == 0:
            ax.set_title(f"Sample {obs_idx} (empty)")
            ax.axis('off')
            continue

        # Plot X-Y projection colored by Z
        scatter = ax.scatter(points[:, 0], points[:, 1],
                            c=points[:, 2], cmap='viridis',
                            s=1, alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"Sample {obs_idx} ({len(points)} pts)")
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Z (height)')

    # Hide empty axes
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def compute_statistics(pointclouds: np.ndarray):
    """Compute point cloud statistics."""
    # Remove zero-padding
    valid_masks = ~np.all(pointclouds == 0, axis=2)  # [N, num_points]
    valid_points_per_cloud = valid_masks.sum(axis=1)  # [N]

    # Flatten all valid points
    all_points = []
    for i in range(len(pointclouds)):
        valid_pts = pointclouds[i][valid_masks[i]]
        if len(valid_pts) > 0:
            all_points.append(valid_pts)

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
    else:
        all_points = np.array([])

    stats = {
        "shape": str(pointclouds.shape),
        "total_clouds": len(pointclouds),
        "avg_points_per_cloud": float(valid_points_per_cloud.mean()),
        "min_points_per_cloud": int(valid_points_per_cloud.min()),
        "max_points_per_cloud": int(valid_points_per_cloud.max()),
    }

    if len(all_points) > 0:
        stats.update({
            "total_valid_points": len(all_points),
            "x_min": float(all_points[:, 0].min()),
            "x_max": float(all_points[:, 0].max()),
            "x_mean": float(all_points[:, 0].mean()),
            "y_min": float(all_points[:, 1].min()),
            "y_max": float(all_points[:, 1].max()),
            "y_mean": float(all_points[:, 1].mean()),
            "z_min": float(all_points[:, 2].min()),
            "z_max": float(all_points[:, 2].max()),
            "z_mean": float(all_points[:, 2].mean()),
            "zero_points_pct": float((pointclouds == 0).all(axis=2).mean() * 100),
        })

    return stats


def diagnose_observations(stats: dict):
    """Diagnose potential issues with point cloud observations."""
    issues = []
    recommendations = []

    # Check if mostly zeros
    if "zero_points_pct" in stats and stats["zero_points_pct"] > 50:
        issues.append(f"HIGH ZERO PADDING: {stats['zero_points_pct']:.1f}% of points are zeros")
        recommendations.append("Point clouds may be too sparse or poorly sampled")

    # Check if few points per cloud
    if "avg_points_per_cloud" in stats and stats["avg_points_per_cloud"] < 100:
        issues.append(f"FEW POINTS: Average only {stats['avg_points_per_cloud']:.0f} points/cloud")
        recommendations.append("Increase point cloud density or check downsampling")

    # Check spatial extent
    if "x_min" in stats:
        x_range = stats["x_max"] - stats["x_min"]
        y_range = stats["y_max"] - stats["y_min"]
        z_range = stats["z_max"] - stats["z_min"]

        if x_range < 0.1 or y_range < 0.1 or z_range < 0.1:
            issues.append(f"SMALL SPATIAL EXTENT: X={x_range:.3f}, Y={y_range:.3f}, Z={z_range:.3f}")
            recommendations.append("Point clouds may not capture full scene geometry")

    return issues, recommendations


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud observations from BC training")
    parser.add_argument("--log_dir", type=str,
                       default="logs/bc_pointcloud/latest",
                       help="Path to training log directory")
    parser.add_argument("--num_points", type=int, default=1024,
                       help="Number of points per cloud (default: 1024)")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to visualize in grid")
    parser.add_argument("--output_dir", type=str,
                       default="/workspace/crane_testbed/pcd_viz",
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
        if "num_points" in metadata:
            args.num_points = metadata["num_points"]

    pointclouds, actual_num_points = reshape_to_pointclouds(obs, args.num_points)
    print(f"Reshaped to: {pointclouds.shape} (num_points={actual_num_points})")

    # Statistics
    stats = compute_statistics(pointclouds)
    print(f"\n{'='*60}")
    print("POINT CLOUD STATISTICS")
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
    n_total = len(pointclouds)
    n_samples = min(args.num_samples, n_total)
    indices = [int(i * n_total / n_samples) for i in range(n_samples)]

    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print('='*60)

    # 3D grid
    plot_pointcloud_grid(
        pointclouds, indices,
        str(output_dir / "pcd_grid_3d.png"),
        f"Point Cloud Observations (n={n_total}, pts={actual_num_points})"
    )

    # Top-down view
    plot_top_down_view(
        pointclouds, indices,
        str(output_dir / "pcd_grid_topdown.png"),
        "Top-Down View (X-Y Projection)"
    )

    # Early vs late
    n_early = min(8, n_total)
    n_late = min(8, n_total)
    early = list(range(n_early))
    late = list(range(max(0, n_total - n_late), n_total))

    plot_pointcloud_grid(pointclouds, early,
                         str(output_dir / "pcd_early.png"),
                         "Early Episode Observations")
    plot_pointcloud_grid(pointclouds, late,
                         str(output_dir / "pcd_late.png"),
                         "Late Episode Observations")

    # Save stats to JSON
    stats_path = output_dir / "pcd_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to: {stats_path}")

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}/")
    print('='*60)
    print("Files:")
    print("  - pcd_grid_3d.png: 3D grid of samples")
    print("  - pcd_grid_topdown.png: Top-down 2D projections")
    print("  - pcd_early.png: First 8 observations")
    print("  - pcd_late.png: Last 8 observations")
    print("  - pcd_stats.json: Detailed statistics")


if __name__ == "__main__":
    main()
