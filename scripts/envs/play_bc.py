#!/usr/bin/env python3
"""
Play a trained BC policy in the crane environment.

Usage:
    # Basic inference with visualization
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc.py \
        --checkpoint logs/bc_policy/bc_20260118_072928/bc_policy.pt \
        --num_envs 1 --num_episodes 10

    # With domain randomization and metrics saving
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc.py \
        --checkpoint logs/bc_policy/bc_20260118_072928/bc_policy.pt \
        --num_envs 4 --num_episodes 50 --domain_randomization --save_metrics

    # Headless evaluation (faster)
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc.py \
        --checkpoint logs/bc_policy/bc_20260118_072928/bc_policy.pt \
        --num_envs 8 --num_episodes 100 --domain_randomization --save_metrics --headless
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# Parse args before IsaacLab imports
parser = argparse.ArgumentParser(description="Play BC policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC policy checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization (variable pile sizes)")
parser.add_argument("--headless", action="store_true", help="Run without visualization")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON file")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for metrics (default: same as checkpoint)")
args_cli, _ = parser.parse_known_args()

# IsaacLab imports
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# Import the environment
import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


class BCPolicy(nn.Module):
    """MLP policy matching the BC training architecture."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def load_bc_policy(checkpoint_path: str, device: str) -> tuple:
    """Load BC policy from checkpoint."""
    print(f"[Play] Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    hidden_dims = checkpoint['hidden_dims']

    print(f"[Play] Policy architecture: {obs_dim} -> {hidden_dims} -> {action_dim}")

    policy = BCPolicy(obs_dim, action_dim, hidden_dims).to(device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    return policy, obs_dim, action_dim


def main():
    # Load policy
    policy, obs_dim, action_dim = load_bc_policy(args_cli.checkpoint, args_cli.device)

    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = True
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[Play] Environment created with {env.num_envs} envs")
    print(f"[Play] Domain randomization: {args_cli.domain_randomization}")

    # Verify observation/action dimensions match
    env.reset()
    test_obs = env._build_target_selection_obs()
    assert test_obs.shape[1] == obs_dim, f"Obs dim mismatch: env={test_obs.shape[1]}, policy={obs_dim}"
    print(f"[Play] Observation dim verified: {obs_dim}")

    # Run episodes
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

    # Track per-episode metrics
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    # Reset environment FIRST (this sets _per_env_log_counts with domain randomization)
    env.reset()

    # Now read the starting log counts AFTER reset
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[Play] Logs per pile: {sum(log_counts)/len(log_counts):.0f} avg (range: {min(log_counts)}-{max(log_counts)})")
    else:
        default_logs = int(env._per_env_target) if hasattr(env, '_per_env_target') and env._per_env_target > 0 else 200
        episode_starting_logs[:] = default_logs
        print(f"[Play] Logs per pile: {default_logs}")

    print(f"\n[Play] Running {args_cli.num_episodes} episodes...")

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Get observation
            obs = env._build_target_selection_obs()

            # Get policy action
            actions = policy(obs)

            # Step environment
            _, rew, terminated, truncated, _ = env.step(actions)

            episode_rewards += rew

            # Track metrics
            for i in range(env.num_envs):
                logs_grasped = int(env._prev_logs_grasped[i].item())
                alignment = env._prev_grasp_alignment[i].item()
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

            # Check for episode completion
            done = terminated | truncated
            for i in range(env.num_envs):
                if done[i]:
                    ep_reward = episode_rewards[i].item()
                    total_reward += ep_reward
                    episodes_done += 1

                    # Calculate clearing percentage
                    starting_logs = int(episode_starting_logs[i].item())
                    logs_cleared = int(episode_logs_cleared[i].item())
                    clear_pct = (logs_cleared / max(1, starting_logs)) * 100
                    clearing_percentages.append(clear_pct)
                    episode_rewards_list.append(ep_reward)
                    logs_per_episode.append(starting_logs)

                    # Track full clears
                    if logs_cleared >= starting_logs:
                        piles_fully_cleared += 1

                    print(f"[Play] Episode {episodes_done}: reward={ep_reward:.2f}, cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs} logs)")

                    # Reset per-episode tracking
                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    # Update starting logs for next episode
                    if has_variable_logs:
                        episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # Compute summary metrics
    avg_clear_pct = sum(clearing_percentages) / max(1, len(clearing_percentages))
    grasp_success_rate = successful_grasps / max(1, total_grasps) * 100
    avg_throughput = total_logs_grasped / max(1, successful_grasps)
    avg_alignment = total_alignment / max(1, successful_grasps)
    avg_stability = total_stability / max(1, successful_grasps)
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    avg_reward = total_reward / max(1, episodes_done)

    # Print summary
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

    # Save metrics if requested
    if args_cli.save_metrics:
        # Determine output directory
        if args_cli.output_dir:
            output_dir = args_cli.output_dir
        else:
            # Save next to checkpoint
            output_dir = os.path.dirname(args_cli.checkpoint)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

        metrics = {
            "eval_config": {
                "checkpoint": args_cli.checkpoint,
                "num_envs": args_cli.num_envs,
                "num_episodes": args_cli.num_episodes,
                "domain_randomization": args_cli.domain_randomization,
                "timestamp": timestamp,
            },
            "episodes": {
                "total": episodes_done,
                "logs_per_pile_avg": float(sum(logs_per_episode) / max(1, len(logs_per_episode))),
                "logs_per_pile_min": int(min(logs_per_episode)) if logs_per_episode else 0,
                "logs_per_pile_max": int(max(logs_per_episode)) if logs_per_episode else 0,
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
