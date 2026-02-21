#!/usr/bin/env python3
"""
Visualize point cloud PLY files.

Usage:
    python visualize_pointcloud.py pointcloud_step_0000.ply
    python visualize_pointcloud.py camera_output/  # visualize all PLY files in folder
    python visualize_pointcloud.py *.ply --side_by_side  # compare multiple files
"""

import argparse
import numpy as np
from pathlib import Path

def load_ply(filepath):
    """Load PLY file and return (N, 3) numpy array."""
    points = []
    with open(filepath, 'r') as f:
        # Skip header
        in_header = True
        vertex_count = 0
        for line in f:
            line = line.strip()
            if in_header:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line == "end_header":
                    in_header = False
            else:
                parts = line.split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points)


def visualize_matplotlib(points_dict, title="Point Cloud"):
    """Visualize point clouds using matplotlib (works without display server)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n_clouds = len(points_dict)
    fig = plt.figure(figsize=(6 * n_clouds, 6))

    for i, (name, points) in enumerate(points_dict.items()):
        ax = fig.add_subplot(1, n_clouds, i + 1, projection='3d')

        if len(points) == 0:
            ax.set_title(f"{name}\n(empty)")
            continue

        # Subsample for faster rendering
        max_points = 5000
        if len(points) > max_points:
            idx = np.random.choice(len(points), max_points, replace=False)
            points = points[idx]

        # Color by height (Z)
        colors = points[:, 2]

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, cmap='viridis', s=1, alpha=0.6)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{name}\n({len(points)} points)")

        # Equal aspect ratio
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

    plt.tight_layout()
    return fig


def visualize_open3d(points_dict):
    """Visualize point clouds using Open3D (interactive)."""
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return None

    geometries = []
    colors = [
        [1, 0, 0],    # red
        [0, 1, 0],    # green
        [0, 0, 1],    # blue
        [1, 1, 0],    # yellow
        [1, 0, 1],    # magenta
        [0, 1, 1],    # cyan
    ]

    for i, (name, points) in enumerate(points_dict.items()):
        if len(points) == 0:
            print(f"  {name}: empty")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Color by height or use fixed color for multiple clouds
        if len(points_dict) == 1:
            # Single cloud: color by height
            z = points[:, 2]
            z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
            colors_array = plt.cm.viridis(z_norm)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors_array)
        else:
            # Multiple clouds: fixed color per cloud
            pcd.paint_uniform_color(colors[i % len(colors)])

        geometries.append(pcd)
        print(f"  {name}: {len(points)} points")

    if geometries:
        o3d.visualization.draw_geometries(geometries, window_name="Point Clouds")

    return geometries


def main():
    parser = argparse.ArgumentParser(description="Visualize PLY point cloud files")
    parser.add_argument("paths", nargs="+", help="PLY file(s) or directory containing PLY files")
    parser.add_argument("--backend", choices=["matplotlib", "open3d", "auto"], default="auto",
                        help="Visualization backend (default: auto)")
    parser.add_argument("--save", type=str, help="Save figure to file (matplotlib only)")
    parser.add_argument("--logs_only", action="store_true", help="Only show *_logs_* files")
    args = parser.parse_args()

    # Collect all PLY files
    ply_files = []
    for path in args.paths:
        p = Path(path)
        if p.is_dir():
            ply_files.extend(sorted(p.glob("*.ply")))
        elif p.suffix.lower() == ".ply":
            ply_files.append(p)

    if not ply_files:
        print("No PLY files found!")
        return

    # Filter if requested
    if args.logs_only:
        ply_files = [f for f in ply_files if "logs" in f.stem.lower()]

    print(f"Loading {len(ply_files)} point cloud(s)...")

    # Load all point clouds
    points_dict = {}
    for ply_file in ply_files:
        points = load_ply(ply_file)
        points_dict[ply_file.stem] = points
        print(f"  {ply_file.name}: {len(points)} points")
        if len(points) > 0:
            print(f"    X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
            print(f"    Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
            print(f"    Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    # Choose backend
    backend = args.backend
    if backend == "auto":
        try:
            import open3d
            backend = "open3d"
        except ImportError:
            backend = "matplotlib"

    print(f"\nUsing {backend} backend...")

    if backend == "open3d":
        visualize_open3d(points_dict)
    else:
        import matplotlib
        if args.save:
            matplotlib.use('Agg')  # Non-interactive backend for saving
        import matplotlib.pyplot as plt

        fig = visualize_matplotlib(points_dict)

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Saved to {args.save}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
