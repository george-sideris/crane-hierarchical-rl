#!/usr/bin/env python3
"""
Play a trained BC PointCloud policy in the crane environment.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint logs/bc_pointcloud/bc_XXXX/bc_pointcloud_policy.pt \
        --num_envs 1 --num_episodes 10
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
import numpy as np

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Play BC pointcloud policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics")
parser.add_argument("--visualize", action="store_true", help="Save per-step visualization PNGs")
parser.add_argument("--viz_dir", type=str, default=None, help="Directory for viz PNGs (default: checkpoint dir / viz)")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

# Import the environment
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class PointNetEncoder(nn.Module):
    """PointNet encoder - must match training architecture exactly."""

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()

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

        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        x = x.view(batch_size * num_points, -1)
        x = self.mlp1(x)
        x = x.view(batch_size, num_points, -1)
        x = x.max(dim=1)[0]
        x = self.fc(x)
        return x


class BCPointNetPolicy(nn.Module):
    """PointNet policy matching the BC training architecture."""

    def __init__(self, num_points: int = 1024, action_dim: int = 4, latent_dim: int = 256):
        super().__init__()

        self.num_points = num_points
        self.latent_dim = latent_dim

        self.encoder = PointNetEncoder(input_dim=3, output_dim=latent_dim)

        self.actor_mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        if points.dim() == 2:
            points = points.view(batch_size, self.num_points, 3)
        latent = self.encoder(points)
        return self.actor_mlp(latent)


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """FPS for point cloud downsampling."""
    device = points.device
    N = points.shape[0]

    if N <= num_samples:
        if N == 0:
            return torch.zeros((num_samples, 3), device=device)
        padding = torch.zeros((num_samples - N, 3), device=device)
        return torch.cat([points, padding], dim=0)

    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    current_idx = torch.randint(0, N, (1,), device=device).item()

    for i in range(num_samples):
        sampled_indices[i] = current_idx
        current_point = points[current_idx:current_idx+1]
        dist_to_current = torch.norm(points - current_point, dim=1)
        distances = torch.minimum(distances, dist_to_current)
        current_idx = torch.argmax(distances).item()

    return points[sampled_indices]


def get_log_pointcloud_base_frame(env, env_idx: int, num_points: int,
                                   depth_range: tuple = (1.0, 10.0)) -> torch.Tensor:
    """Get masked log point cloud in crane base frame."""
    pc_world = env.get_log_pointcloud_world(env_idx, max_points=5000, depth_range=depth_range)

    if pc_world.shape[0] == 0:
        return torch.zeros((num_points, 3), device=env.device)

    # Transform to base frame
    base_pos_w = env.crane.data.root_pos_w[env_idx]
    base_quat_w = env.crane.data.root_quat_w[env_idx]

    pc_translated = pc_world - base_pos_w

    w, x, y, z = base_quat_w[0], base_quat_w[1], base_quat_w[2], base_quat_w[3]
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
    ])

    pc_base = pc_translated @ R
    pc_sampled = farthest_point_sampling(pc_base, num_points)

    return pc_sampled


def load_bc_policy(checkpoint_path: str, device: str) -> tuple:
    """Load BC policy from checkpoint."""
    print(f"[Play] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_points = checkpoint.get('num_points', 1024)
    action_dim = checkpoint.get('action_dim', 4)
    metadata = checkpoint.get('metadata', {})

    print(f"[Play] Policy: {num_points} points -> {action_dim} actions")

    policy = BCPointNetPolicy(num_points=num_points, action_dim=action_dim).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    return policy, num_points, metadata


def decode_action(raw_action, min_bounds, max_bounds):
    """Decode raw network output to (x, y, z, yaw) in base frame.

    Mirrors the env's _apply_action decode logic exactly:
      xyz: min + (tanh(a) + 1) / 2 * (max - min)
      yaw: 5D -> atan2(tanh(a4), tanh(a3)) / 2
           4D -> tanh(a3) * pi/2
    """
    a = raw_action.cpu().numpy()
    x = float(min_bounds[0] + (np.tanh(a[0]) + 1) / 2 * (max_bounds[0] - min_bounds[0]))
    y = float(min_bounds[1] + (np.tanh(a[1]) + 1) / 2 * (max_bounds[1] - min_bounds[1]))
    z = float(min_bounds[2] + (np.tanh(a[2]) + 1) / 2 * (max_bounds[2] - min_bounds[2]))
    if len(a) == 5:
        yaw = float(np.arctan2(np.tanh(a[4]), np.tanh(a[3])) / 2.0)
    else:
        yaw = float(np.tanh(a[3]) * (np.pi / 2))
    return x, y, z, yaw


def save_step_viz(points_np, x, y, z, yaw, step_idx, viz_dir,
                  logs_grasped=None, alignment=None,
                  bounds_min=None, bounds_max=None):
    """Save a 2-panel visualization PNG for one grasp step.

    Left:  top-down view (X vs Y) — horizontal placement + yaw arrow
    Right: side view (Y vs Z) — height targeting
    Both overlay: point cloud scatter + red star at predicted target.
    Optionally draws dashed action-bounds rectangle on each panel.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(14, 6))

    # Filter out zero-padding points for cleaner viz
    mask = np.any(points_np != 0.0, axis=1)
    pts = points_np[mask] if mask.any() else points_np

    # Compute point cloud bounding box for reference
    pc_min = pts.min(axis=0) if len(pts) > 0 else np.zeros(3)
    pc_max = pts.max(axis=0) if len(pts) > 0 else np.zeros(3)

    # -- Left panel: top-down (X vs Y) colored by height --
    if len(pts) > 0:
        sc = ax_top.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='viridis',
                            s=1, alpha=0.5, label='Point cloud')
        plt.colorbar(sc, ax=ax_top, label='Z (height)')
    ax_top.plot(x, y, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                label=f'Target ({x:.2f}, {y:.2f})')
    # Yaw direction arrow
    arrow_len = 0.3
    ax_top.annotate('', xy=(x + arrow_len * np.cos(yaw), y + arrow_len * np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    # Draw point cloud bounding box (X-Y)
    if len(pts) > 0:
        ax_top.add_patch(Rectangle(
            (pc_min[0], pc_min[1]), pc_max[0] - pc_min[0], pc_max[1] - pc_min[1],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
    # Draw action bounds rectangle (X-Y)
    if bounds_min is not None and bounds_max is not None:
        bw = bounds_max[0] - bounds_min[0]
        bh = bounds_max[1] - bounds_min[1]
        ax_top.add_patch(Rectangle(
            (bounds_min[0], bounds_min[1]), bw, bh,
            linewidth=1.5, edgecolor='orange', facecolor='none',
            linestyle='--', label='Action bounds'))
    ax_top.set_xlabel('X (base frame)')
    ax_top.set_ylabel('Y (base frame)')
    ax_top.set_title('Top-down (X vs Y)')
    ax_top.set_aspect('equal')
    ax_top.legend(loc='upper right', fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # -- Right panel: side view (Y vs Z) --
    if len(pts) > 0:
        ax_side.scatter(pts[:, 1], pts[:, 2], c=pts[:, 0], cmap='viridis',
                        s=1, alpha=0.5, label='Point cloud')
    ax_side.plot(y, z, 'r*', markersize=15, markeredgecolor='k', markeredgewidth=0.5,
                 label=f'Target (z={z:.2f})')
    # Draw point cloud bounding box (Y-Z)
    if len(pts) > 0:
        ax_side.add_patch(Rectangle(
            (pc_min[1], pc_min[2]), pc_max[1] - pc_min[1], pc_max[2] - pc_min[2],
            linewidth=1, edgecolor='lime', facecolor='none',
            linestyle='-', label='PC bbox'))
    # Draw action bounds rectangle (Y-Z)
    if bounds_min is not None and bounds_max is not None:
        bw = bounds_max[1] - bounds_min[1]
        bh = bounds_max[2] - bounds_min[2]
        ax_side.add_patch(Rectangle(
            (bounds_min[1], bounds_min[2]), bw, bh,
            linewidth=1.5, edgecolor='orange', facecolor='none',
            linestyle='--', label='Action bounds'))
    ax_side.set_xlabel('Y (base frame)')
    ax_side.set_ylabel('Z (base frame)')
    ax_side.set_title('Side view (Y vs Z)')
    ax_side.set_aspect('equal')
    ax_side.legend(loc='upper right', fontsize=8)
    ax_side.grid(True, alpha=0.3)

    # Suptitle with step info
    title = f'Step {step_idx} | Target: ({x:.3f}, {y:.3f}, {z:.3f}) yaw={np.degrees(yaw):.1f}°'
    if logs_grasped is not None:
        result = 'SUCCESS' if logs_grasped > 0 else 'MISS'
        title += f' | {result} ({logs_grasped} logs)'
    if alignment is not None and logs_grasped and logs_grasped > 0:
        title += f' | align={alignment:.3f}'
    fig.suptitle(title, fontsize=11)

    fig.savefig(os.path.join(viz_dir, f"step_{step_idx:04d}.png"),
                dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    # Load policy
    policy, num_points, metadata = load_bc_policy(args_cli.checkpoint, args_cli.device)

    # Detect action dim from checkpoint
    checkpoint = torch.load(args_cli.checkpoint, map_location=args_cli.device)
    action_dim = checkpoint.get('action_dim', 4)
    print(f"[Play] Num points: {num_points}, Action dim: {action_dim}")

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.action_space = action_dim  # Match policy output (4D or 5D)
    cfg.enable_camera = True
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[Play] Environment created with {env.num_envs} envs")
    print(f"[Play] Domain randomization: {args_cli.domain_randomization}")

    # Set up visualization directory
    if args_cli.visualize:
        viz_dir = args_cli.viz_dir or os.path.join(os.path.dirname(args_cli.checkpoint), "viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[Play] Saving visualizations to: {viz_dir}")

    # Reset and initialize camera
    env.reset()
    env.sim.render()
    env._camera.update(dt=env.cfg.sim.dt)

    # Ensure action bounds are computed before visualization loop
    # (bounds are lazily computed inside _apply_action, so they're zero before the first step)
    env._compute_action_space_bounds()

    # Diagnostic: verify bounds were actually computed
    if args_cli.visualize:
        root_pos = env.crane.data.root_pos_w[0].cpu().numpy()
        root_quat = env.crane.data.root_quat_w[0].cpu().numpy()
        b_min = env._action_bounds_min[0].cpu().numpy()
        b_max = env._action_bounds_max[0].cpu().numpy()
        print(f"[Viz-Init] Crane root pos_w: {root_pos}")
        print(f"[Viz-Init] Crane root quat_w: {root_quat}")
        print(f"[Viz-Init] Computed bounds min: {b_min}")
        print(f"[Viz-Init] Computed bounds max: {b_max}")
        print(f"[Viz-Init] Bounds range: X={b_max[0]-b_min[0]:.2f}, "
              f"Y={b_max[1]-b_min[1]:.2f}, Z={b_max[2]-b_min[2]:.2f}")
        if np.allclose(b_min, 0) and np.allclose(b_max, 0):
            print("[Viz-Init] WARNING: bounds are still zero!")

    # Tracking
    episodes_done = 0
    total_reward = 0.0
    total_logs_grasped = 0
    total_grasps = 0
    successful_grasps = 0
    failed_grasps = 0
    total_alignment = 0.0
    total_stability = 0.0
    clearing_percentages = []
    episode_rewards_list = []
    logs_per_episode = []
    piles_fully_cleared = 0

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    # Get starting log counts
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[Play] Logs per pile: {sum(log_counts)/len(log_counts):.0f} avg")
    else:
        default_logs = 200
        episode_starting_logs[:] = default_logs
        print(f"[Play] Logs per pile: {default_logs}")

    print(f"\n[Play] Running {args_cli.num_episodes} episodes...")

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Get point cloud observations
            pc_batch = []
            for i in range(env.num_envs):
                pc = get_log_pointcloud_base_frame(env, i, num_points)
                pc_batch.append(pc)
            obs = torch.stack(pc_batch)  # (num_envs, num_points, 3)

            # Get policy action
            actions = policy(obs)

            # Debug: print action stats occasionally
            if total_grasps < 5 or total_grasps % 100 == 0:
                print(f"[Debug] Grasp {total_grasps}: points={obs.shape}, "
                      f"action=[{actions.min():.2f},{actions.max():.2f}]")

            # Pre-step: decode actions for visualization (before env.step modifies state)
            if args_cli.visualize:
                viz_decoded = []
                viz_bounds = []
                for i in range(env.num_envs):
                    min_b = env._action_bounds_min[i].cpu().numpy()
                    max_b = env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    pts = obs[i].cpu().numpy()
                    if obs[i].dim() == 1:
                        pts = pts.reshape(-1, 3)
                    viz_decoded.append((pts, x, y, z, yaw))
                    viz_bounds.append((min_b.copy(), max_b.copy()))

                    # One-time diagnostic
                    if total_grasps == 0 and i == 0:
                        mask = np.any(pts != 0.0, axis=1)
                        diag_pts = pts[mask] if mask.any() else pts
                        pc_min = diag_pts.min(axis=0)
                        pc_max = diag_pts.max(axis=0)
                        print(f"[Viz] Action bounds min: {min_b}")
                        print(f"[Viz] Action bounds max: {max_b}")
                        print(f"[Viz] Point cloud  min: {pc_min}")
                        print(f"[Viz] Point cloud  max: {pc_max}")
                        print(f"[Viz] Decoded target: ({x:.3f}, {y:.3f}, {z:.3f})")
                        overlap_x = min_b[0] <= pc_max[0] and max_b[0] >= pc_min[0]
                        overlap_y = min_b[1] <= pc_max[1] and max_b[1] >= pc_min[1]
                        overlap_z = min_b[2] <= pc_max[2] and max_b[2] >= pc_min[2]
                        star_in_pc = (pc_min[0] <= x <= pc_max[0] and
                                      pc_min[1] <= y <= pc_max[1] and
                                      pc_min[2] <= z <= pc_max[2])
                        print(f"[Viz] Bounds/PC overlap: X={overlap_x} Y={overlap_y} Z={overlap_z}")
                        print(f"[Viz] Star inside point cloud bbox: {star_in_pc}")

            # Step environment
            _, rew, terminated, truncated, _ = env.step(actions)
            env.sim.render()
            env._camera.update(dt=env.cfg.sim.dt)

            episode_rewards += rew

            # Track metrics
            for i in range(env.num_envs):
                logs_grasped = int(env._prev_logs_grasped[i].item())
                alignment = env._prev_grasp_alignment[i].item() if hasattr(env, '_prev_grasp_alignment') else 0.0
                stability = env._prev_grasp_stability[i].item() if hasattr(env, '_prev_grasp_stability') else 1.0

                # Save visualization with grasp result
                if args_cli.visualize:
                    pts, vx, vy, vz, vyaw = viz_decoded[i]
                    vmin_b, vmax_b = viz_bounds[i]
                    save_step_viz(pts, vx, vy, vz, vyaw,
                                  total_grasps, viz_dir,
                                  logs_grasped=logs_grasped,
                                  alignment=alignment,
                                  bounds_min=vmin_b, bounds_max=vmax_b)

                total_grasps += 1
                total_logs_grasped += logs_grasped
                episode_logs_cleared[i] += logs_grasped
                if logs_grasped > 0:
                    successful_grasps += 1
                    total_alignment += alignment
                    total_stability += stability
                else:
                    failed_grasps += 1

            # Check episode completion
            done = terminated | truncated
            for i in range(env.num_envs):
                if done[i]:
                    ep_reward = episode_rewards[i].item()
                    total_reward += ep_reward
                    episodes_done += 1

                    starting_logs = int(episode_starting_logs[i].item())
                    logs_cleared = int(episode_logs_cleared[i].item())
                    clear_pct = (logs_cleared / max(1, starting_logs)) * 100
                    clearing_percentages.append(clear_pct)
                    episode_rewards_list.append(ep_reward)
                    logs_per_episode.append(starting_logs)

                    if logs_cleared >= starting_logs:
                        piles_fully_cleared += 1

                    print(f"[Play] Episode {episodes_done}: reward={ep_reward:.2f}, "
                          f"cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs})")

                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    if has_variable_logs:
                        episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # Summary
    avg_clear_pct = sum(clearing_percentages) / max(1, len(clearing_percentages))
    grasp_success_rate = successful_grasps / max(1, total_grasps) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_alignment = total_alignment / max(1, successful_grasps)
    avg_stability = total_stability / max(1, successful_grasps)
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_reward = total_reward / max(1, episodes_done)

    print("=" * 60)
    print(f"\n[Play] ====== RESULTS ======")
    print(f"[Play] Episodes: {episodes_done}")
    print(f"[Play] Avg Episode Reward: {avg_reward:.2f}")
    print(f"[Play] Avg Pile Cleared: {avg_clear_pct:.1f}%")
    print(f"[Play] Full Clear Rate: {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
    print(f"[Play] Grasp Success Rate: {grasp_success_rate:.1f}%")
    print(f"[Play] Avg Throughput: {avg_throughput:.2f} logs/grasp")
    print(f"[Play] Avg Alignment: {avg_alignment:.3f}")
    print(f"[Play] Avg Stability: {avg_stability:.3f}")
    print(f"[Play] Avg Alignment: {avg_alignment:.3f}")
    print(f"[Play] Total Logs Grasped: {total_logs_grasped}")
    print(f"[Play] =======================")

    # Save metrics
    if args_cli.save_metrics:
        output_dir = args_cli.output_dir or os.path.dirname(args_cli.checkpoint)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

        metrics = {
            "eval_config": {
                "checkpoint": args_cli.checkpoint,
                "num_envs": args_cli.num_envs,
                "num_episodes": episodes_done,
                "domain_randomization": args_cli.domain_randomization,
                "num_points": num_points,
                "timestamp": timestamp,
            },
            "grasp": {
                "success_rate": grasp_success_rate,
                "successful": successful_grasps,
                "failed": failed_grasps,
                "total": total_grasps,
            },
            "pile_clearing": {
                "avg_cleared_pct": avg_clear_pct,
                "full_clear_rate": full_clear_rate,
                "full_clears": piles_fully_cleared,
            },
            "performance": {
                "avg_episode_reward": avg_reward,
                "avg_throughput": avg_throughput,
                "avg_alignment": avg_alignment,
                "avg_stability": avg_stability,
                "total_logs_grasped": total_logs_grasped,
            },
            "episode_rewards": episode_rewards_list,
            "clearing_percentages": clearing_percentages,
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[Play] Metrics saved to: {metrics_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
