#!/usr/bin/env python3
"""
Play a trained BC depth policy in the crane environment.

This mirrors play_bc.py but uses depth observations instead of pose observations.
Uses the EXACT same observation function as training for consistency.

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_depth.py \
        --checkpoint logs/bc_depth/bc_XXXX/bc_depth_policy.pt \
        --num_envs 1 --num_episodes 10
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
import numpy as np

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Add crane_testbed agents path for ResNet encoder
crane_testbed_root = crane_scripts_path.parent.parent
crane_agents_path = crane_testbed_root / "source" / "crane_testbed" / "crane_testbed" / "agents"
sys.path.insert(0, str(crane_agents_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Play BC depth policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC depth policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless, enable_cameras=True)
simulation_app = app_launcher.app

# Import the environment
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class BCCNNPolicy(nn.Module):
    """CNN policy matching the BC training architecture."""

    def __init__(self, obs_dim: int, action_dim: int = 4):
        super().__init__()

        import math
        sqrt = int(math.sqrt(obs_dim))
        if sqrt * sqrt != obs_dim:
            raise ValueError(f"obs_dim={obs_dim} is not a perfect square")
        self.img_size = sqrt

        # Build encoder based on image size - MUST match training exactly
        # Includes BatchNorm after each Conv2d for training stability
        if self.img_size == 256:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
            )
            self.flatten_dim = 128 * 8 * 8
        elif self.img_size == 128:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ELU(),
            )
            self.flatten_dim = 128 * 8 * 8
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ELU(),
            )
            self.flatten_dim = 64 * 8 * 8

        self.latent_dim = 256
        self.latent_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, self.latent_dim),
            nn.ELU(),
        )

        self.actor_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_size, self.img_size)
        features = self.encoder(x)
        features = features.view(batch_size, -1)
        latent = self.latent_proj(features)
        return self.actor_mlp(latent)


class BCResNetPolicy(nn.Module):
    """ResNet + CBAM policy for BC."""

    def __init__(self, obs_dim: int, action_dim: int = 4, latent_dim: int = 256):
        super().__init__()

        import math
        sqrt = int(math.sqrt(obs_dim))
        if sqrt * sqrt != obs_dim:
            raise ValueError(f"obs_dim={obs_dim} is not a perfect square")
        self.img_size = sqrt

        # Import ResNet encoder
        from resnet_depth_encoder import ResNetDepthEncoder

        self.encoder = ResNetDepthEncoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

        self.actor_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        x = obs.view(batch_size, 1, self.img_size, self.img_size)
        latent = self.encoder(x)
        return self.actor_mlp(latent)


def get_depth_observation(env, depth_height: int, depth_width: int,
                          use_semantic_mask: bool, depth_min: float = 1.5,
                          depth_max: float = 8.0) -> torch.Tensor:
    """Get processed depth observation - EXACT same as training."""
    env._camera.update(dt=env.cfg.sim.dt)

    depth = env._camera.data.output["depth"].squeeze(-1)

    if use_semantic_mask and "semantic_segmentation" in env.cfg.camera_cfg.data_types:
        sem_seg = env._camera.data.output["semantic_segmentation"]
        if sem_seg.dim() == 4:
            sem_ids = sem_seg[..., 0]
        else:
            sem_ids = sem_seg
        mask = (sem_ids > 0)
        depth = torch.where(mask, depth, torch.tensor(depth_max, device=depth.device))

    depth = torch.clamp(depth, depth_min, depth_max)
    depth = (depth - depth_min) / (depth_max - depth_min)

    depth = depth.unsqueeze(1)
    depth = F.interpolate(depth, size=(depth_height, depth_width),
                          mode='bilinear', align_corners=False)
    depth = depth.squeeze(1)

    return depth.view(env.num_envs, -1)


def load_bc_policy(checkpoint_path: str, device: str) -> tuple:
    """Load BC policy from checkpoint."""
    print(f"[Play] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint.get('action_dim', 4)
    img_size = checkpoint.get('img_size', int(np.sqrt(obs_dim)))
    metadata = checkpoint.get('metadata', {})
    encoder_type = checkpoint.get('encoder_type', 'cnn')  # Default to CNN for old checkpoints

    print(f"[Play] Policy: {img_size}x{img_size} depth -> {action_dim} actions")
    print(f"[Play] Encoder: {encoder_type}")

    # Load appropriate policy architecture
    if encoder_type == 'resnet':
        policy = BCResNetPolicy(obs_dim, action_dim).to(device)
    else:
        policy = BCCNNPolicy(obs_dim, action_dim).to(device)

    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    return policy, metadata


def main():
    # Load policy
    policy, metadata = load_bc_policy(args_cli.checkpoint, args_cli.device)

    # Get depth settings from metadata or defaults
    depth_height = metadata.get('depth_height', 256)
    depth_width = metadata.get('depth_width', 256)
    use_semantic_mask = metadata.get('use_semantic_mask', True)

    print(f"[Play] Depth settings: {depth_height}x{depth_width}, mask={use_semantic_mask}")

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.enable_camera = True
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.camera_cfg.width = max(depth_width, 256)
    cfg.camera_cfg.height = max(depth_height, 256)
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[Play] Environment created with {env.num_envs} envs")
    print(f"[Play] Domain randomization: {args_cli.domain_randomization}")

    # Reset and initialize camera
    env.reset()
    env.sim.render()
    env._camera.update(dt=env.cfg.sim.dt)

    # Verify observation dimensions
    test_obs = get_depth_observation(env, depth_height, depth_width, use_semantic_mask)
    expected_dim = depth_height * depth_width
    assert test_obs.shape[1] == expected_dim, f"Obs dim mismatch: got {test_obs.shape[1]}, expected {expected_dim}"
    print(f"[Play] Observation dim verified: {expected_dim}")

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
            # Get depth observation - EXACT same as training
            obs = get_depth_observation(env, depth_height, depth_width, use_semantic_mask)

            # Get policy action
            actions = policy(obs)

            # VERIFICATION TEST: Uncomment to use random actions instead of policy
            # If performance stays good with random actions, something is wrong!
            # actions = torch.randn_like(actions) * 2.0  # Random actions

            # Debug: print action stats EVERY grasp to verify policy is running
            print(f"[POLICY] Grasp {total_grasps}: obs=[{obs.min():.3f},{obs.max():.3f}], "
                  f"action=[{actions[0,0]:.3f}, {actions[0,1]:.3f}, {actions[0,2]:.3f}, {actions[0,3]:.3f}]")

            # Debug yaw prediction
            # Decode from normalized range [-π/2, π/2]
            policy_yaw = np.tanh(actions[0,3].item()) * (np.pi / 2)
            correct_yaw = env._get_target_grapple_yaw_b(0)

            # Both values now in [-π/2, π/2], simple comparison
            diff = abs(policy_yaw - correct_yaw)

            print(f"[YAW] policy={policy_yaw:.3f}, correct={correct_yaw:.3f}, diff={diff:.3f}")

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
                "depth_height": depth_height,
                "depth_width": depth_width,
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
