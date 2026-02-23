#!/usr/bin/env python3
"""
Play the built-in heuristic baseline (target top log + optimal yaw alignment).

Usage:
    # Basic heuristic inference with visualization
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
        --num_envs 1 --num_episodes 10

    # With domain randomization and metrics saving
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
        --num_envs 4 --num_episodes 50 --domain_randomization --save_metrics

    # Headless evaluation (faster)
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
        --num_envs 8 --num_episodes 100 --domain_randomization --save_metrics --headless
"""

import os
import sys
import argparse
import json
import torch
from datetime import datetime
from pathlib import Path

# Add the envs path
crane_scripts_path = Path(__file__).resolve().parent
sys.path.insert(0, str(crane_scripts_path))

# IsaacLab imports (must come before argparse for proper arg handling)
from isaaclab.app import AppLauncher

# Parse args - use AppLauncher's standard args
parser = argparse.ArgumentParser(description="Play heuristic baseline")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--domain_randomization", action="store_true", help="Enable domain randomization (variable pile sizes)")
parser.add_argument("--save_metrics", action="store_true", help="Save metrics to JSON file")
parser.add_argument("--output_dir", type=str, default="logs/heuristic_baseline", help="Output directory for metrics")
# Add standard AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator with full args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import the environment
import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull


def main():
    # Create environment
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_hierarchical_rl = False  # Pure heuristic mode (no policy)
    cfg.enable_domain_randomization = args_cli.domain_randomization
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True

    env = CraneDirectEnvFull(cfg)
    print(f"[Heuristic] Environment created with {env.num_envs} envs")
    print(f"[Heuristic] Domain randomization: {args_cli.domain_randomization}")
    print(f"[Heuristic] Running built-in heuristic FSM (target top log + optimal yaw)")

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
    total_knocked_off = 0
    knocked_off_per_episode = []
    logs_cleared_per_episode = []

    # Track per-episode metrics
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_logs_cleared = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_successful_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_failed_grasps = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    ep_alignment_sum = torch.zeros(env.num_envs, device=env.device)
    ep_stability_sum = torch.zeros(env.num_envs, device=env.device)

    # Per-episode result lists (for mean ± std reporting)
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []

    # Reset environment FIRST (this sets _per_env_log_counts with domain randomization)
    env.reset()

    # Now read the starting log counts AFTER reset
    has_variable_logs = hasattr(env, '_per_env_log_counts') and env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(env.num_envs):
            episode_starting_logs[i] = int(env._per_env_log_counts[i].item())
        log_counts = [int(env._per_env_log_counts[i].item()) for i in range(env.num_envs)]
        print(f"[Heuristic] Logs per pile: {sum(log_counts)/len(log_counts):.0f} avg (range: {min(log_counts)}-{max(log_counts)})")
    else:
        default_logs = int(env._per_env_target) if hasattr(env, '_per_env_target') and env._per_env_target > 0 else 200
        episode_starting_logs[:] = default_logs
        print(f"[Heuristic] Logs per pile: {default_logs}")

    print(f"\n[Heuristic] Running {args_cli.num_episodes} episodes...")

    while episodes_done < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            # Zero actions = heuristic FSM takes over
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)

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
                    ep_successful_grasps[i] += 1
                    ep_alignment_sum[i] += alignment * logs_grasped
                    ep_stability_sum[i] += stability * logs_grasped
                else:
                    failed_grasps += 1
                    ep_failed_grasps[i] += 1

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

                    # Capture knocked-off count before env resets it
                    ep_knocked_off = int(env._logs_knocked_off[i].item()) if hasattr(env, '_logs_knocked_off') else 0
                    total_knocked_off += ep_knocked_off
                    knocked_off_per_episode.append(ep_knocked_off)
                    logs_cleared_per_episode.append(logs_cleared)

                    # Per-episode grasp metrics
                    n_success = int(ep_successful_grasps[i].item())
                    n_fail = int(ep_failed_grasps[i].item())
                    n_total = n_success + n_fail
                    per_ep_success_rates.append(n_success / max(1, n_total) * 100)
                    per_ep_throughputs.append(logs_cleared / max(1, n_success))
                    per_ep_alignments.append(float(ep_alignment_sum[i].item()) / max(1, logs_cleared))
                    per_ep_stabilities.append(float(ep_stability_sum[i].item()) / max(1, logs_cleared))

                    print(f"[Heuristic] Episode {episodes_done}: reward={ep_reward:.2f}, cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs} logs), knocked_off={ep_knocked_off}")

                    # Reset per-episode tracking
                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    ep_successful_grasps[i] = 0
                    ep_failed_grasps[i] = 0
                    ep_alignment_sum[i] = 0.0
                    ep_stability_sum[i] = 0.0
                    # Update starting logs for next episode
                    if has_variable_logs:
                        episode_starting_logs[i] = int(env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # Helper: mean and sample std dev
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _std(vals, mean_val):
        return (sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0

    # All metrics are per-episode averages (each episode weighted equally)
    avg_reward = _mean(episode_rewards_list)
    avg_clear_pct = _mean(clearing_percentages)
    avg_success_rate = _mean(per_ep_success_rates)
    avg_throughput = _mean(per_ep_throughputs)
    avg_alignment = _mean(per_ep_alignments)
    avg_stability = _mean(per_ep_stabilities)
    full_clear_rate = piles_fully_cleared / max(1, episodes_done) * 100
    knocked_off_pcts = [knocked_off_per_episode[j] / max(1, logs_per_episode[j]) * 100
                        for j in range(len(knocked_off_per_episode))]
    avg_knocked_off_pct = _mean(knocked_off_pcts)

    std_reward = _std(episode_rewards_list, avg_reward)
    std_clear_pct = _std(clearing_percentages, avg_clear_pct)
    std_success_rate = _std(per_ep_success_rates, avg_success_rate)
    std_throughput = _std(per_ep_throughputs, avg_throughput)
    std_alignment = _std(per_ep_alignments, avg_alignment)
    std_stability = _std(per_ep_stabilities, avg_stability)
    std_knocked_off_pct = _std(knocked_off_pcts, avg_knocked_off_pct)

    # Print summary (all values are mean ± std across episodes)
    print("=" * 60)
    print(f"\n[Heuristic] ====== RESULTS ({episodes_done} episodes) ======")
    print(f"[Heuristic] Episode Reward:      {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"[Heuristic] Pile Cleared:        {avg_clear_pct:.1f} ± {std_clear_pct:.1f}%")
    print(f"[Heuristic] Full Clear Rate:     {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
    print(f"[Heuristic] Grasp Success Rate:  {avg_success_rate:.1f} ± {std_success_rate:.1f}%")
    print(f"[Heuristic] Throughput:          {avg_throughput:.2f} ± {std_throughput:.2f} logs/grasp")
    print(f"[Heuristic] Alignment:           {avg_alignment:.3f} ± {std_alignment:.3f}")
    print(f"[Heuristic] Stability:           {avg_stability:.3f} ± {std_stability:.3f}")
    print(f"[Heuristic] Knocked Off:         {avg_knocked_off_pct:.1f} ± {std_knocked_off_pct:.1f}%")
    print(f"[Heuristic] Total Logs Grasped:  {total_logs_grasped}")
    print(f"[Heuristic] ======================")

    # Save metrics if requested
    if args_cli.save_metrics:
        output_dir = args_cli.output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

        metrics = {
            "eval_config": {
                "method": "heuristic_baseline",
                "num_envs": args_cli.num_envs,
                "num_episodes": episodes_done,
                "domain_randomization": args_cli.domain_randomization,
                "timestamp": timestamp,
            },
            "episodes": {
                "total": episodes_done,
                "logs_per_pile_avg": float(_mean(logs_per_episode)),
                "logs_per_pile_min": int(min(logs_per_episode)) if logs_per_episode else 0,
                "logs_per_pile_max": int(max(logs_per_episode)) if logs_per_episode else 0,
            },
            "summary": {
                "reward":            {"mean": avg_reward, "std": std_reward},
                "pile_cleared_pct":  {"mean": avg_clear_pct, "std": std_clear_pct},
                "full_clear_rate":   full_clear_rate,
                "grasp_success_pct": {"mean": avg_success_rate, "std": std_success_rate},
                "throughput":        {"mean": avg_throughput, "std": std_throughput},
                "alignment":         {"mean": avg_alignment, "std": std_alignment},
                "stability":         {"mean": avg_stability, "std": std_stability},
                "knocked_off_pct":   {"mean": avg_knocked_off_pct, "std": std_knocked_off_pct},
                "total_logs_grasped": total_logs_grasped,
                "total_grasps": total_grasps,
                "total_successful_grasps": successful_grasps,
                "total_failed_grasps": failed_grasps,
            },
            "per_episode": {
                "rewards": episode_rewards_list,
                "clearing_pcts": clearing_percentages,
                "success_rates": per_ep_success_rates,
                "throughputs": per_ep_throughputs,
                "alignments": per_ep_alignments,
                "stabilities": per_ep_stabilities,
                "knocked_off_counts": knocked_off_per_episode,
                "knocked_off_pcts": knocked_off_pcts,
                "logs_cleared": logs_cleared_per_episode,
                "logs_per_pile": logs_per_episode,
            },
        }

        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n[Heuristic] Metrics saved to: {metrics_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
