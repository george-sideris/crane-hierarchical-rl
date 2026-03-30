"""Script to evaluate a trained SAC agent (SB3).

Usage:
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play_sac.py \
        --task Isaac-Crane-Full-CosSin-MR-v0 \
        --checkpoint logs/sac/crane_full_cossin_mr_v0/2026-03-23_05-36-45/model_final.zip \
        --num_envs 20 --num_episodes 100 --seed 42 --headless --save_metrics
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a trained SAC agent (SB3).")
parser.add_argument("--num_envs", type=int, default=20, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Crane-Full-CosSin-MR-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAC model .zip file.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--save_metrics", action="store_true", default=False, help="Save evaluation metrics to JSON.")
parser.add_argument("--metrics_output_dir", type=str, default=None, help="Output directory for metrics.")
parser.add_argument("--domain_randomization", action="store_true", default=False,
                    help="Enable domain randomization (random log count 20-200)")
parser.add_argument("--obs_noise", type=float, default=0.0,
                    help="Gaussian noise sigma added to observations (meters)")
parser.add_argument("--action_noise", type=float, default=0.0,
                    help="Gaussian noise sigma added to actions")
parser.add_argument("--stochastic", action="store_true", default=False,
                    help="Use stochastic actions instead of deterministic")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import torch
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnv, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
import crane_testbed.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Evaluate SAC agent."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    if args_cli.domain_randomization and hasattr(env_cfg, 'enable_domain_randomization'):
        env_cfg.enable_domain_randomization = True
        print(f"[INFO] Domain randomization enabled")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Get underlying env for metrics access
    underlying_env = env.unwrapped

    # Wrap for SB3
    sb3_env = Sb3VecEnvWrapper(env)

    # Load SAC model
    checkpoint = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading SAC model from: {checkpoint}")
    model = SAC.load(checkpoint, env=sb3_env, device="auto")
    log_dir = os.path.dirname(checkpoint)

    num_envs = underlying_env.num_envs
    device = underlying_env.device

    # Noise info
    if args_cli.obs_noise > 0:
        print(f"[INFO] Observation noise sigma: {args_cli.obs_noise} m")
    if args_cli.action_noise > 0:
        print(f"[INFO] Action noise sigma: {args_cli.action_noise}")

    # ── Per-env tracking ──────────────────────────────────────────────
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_logs_cleared = torch.zeros(num_envs, device=device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_successful_grasps = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_failed_grasps = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_alignment_sum = torch.zeros(num_envs, device=device)
    ep_stability_sum = torch.zeros(num_envs, device=device)
    ep_cycle_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_clearing_curves = [[] for _ in range(num_envs)]

    # ── Aggregate tracking ────────────────────────────────────────────
    episodes_done = 0
    total_reward = 0.0
    total_logs_grasped = 0
    total_grasps = 0
    successful_grasps = 0
    failed_grasps = 0
    clearing_percentages = []
    episode_rewards_list = []
    logs_per_episode = []
    piles_fully_cleared = 0
    total_knocked_off = 0
    knocked_off_per_episode = []
    logs_cleared_per_episode = []
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []
    per_ep_cycles = []
    per_ep_clearing_curves = []

    # Reset and get initial obs
    obs = sb3_env.reset()

    # Read starting log counts
    has_variable_logs = hasattr(underlying_env, '_per_env_log_counts') and underlying_env._per_env_log_counts is not None
    if has_variable_logs:
        for i in range(num_envs):
            episode_starting_logs[i] = int(underlying_env._per_env_log_counts[i].item())
        log_counts = [int(underlying_env._per_env_log_counts[i].item()) for i in range(num_envs)]
        print(f"[Eval] Logs per pile: {sum(log_counts)/len(log_counts):.0f} avg (range: {min(log_counts)}-{max(log_counts)})")
    else:
        default_logs = int(underlying_env._per_env_target) if hasattr(underlying_env, '_per_env_target') and underlying_env._per_env_target > 0 else 200
        episode_starting_logs[:] = default_logs
        print(f"[Eval] Logs per pile: {default_logs}")

    print(f"[Eval] Running evaluation for {args_cli.num_episodes} episodes...")
    print("=" * 60)

    # ── Eval loop ─────────────────────────────────────────────────────
    while simulation_app.is_running() and episodes_done < args_cli.num_episodes:
        with torch.inference_mode():
            # Inject observation noise
            if args_cli.obs_noise > 0:
                obs = obs + np.random.randn(*obs.shape).astype(np.float32) * args_cli.obs_noise

            # Get action from SAC
            actions, _ = model.predict(obs, deterministic=not args_cli.stochastic)

            # Inject action noise
            if args_cli.action_noise > 0:
                actions = actions + np.random.randn(*actions.shape).astype(np.float32) * args_cli.action_noise

            # Step environment
            obs, rewards, dones, infos = sb3_env.step(actions)

            # Convert rewards to torch for accumulation
            rewards_t = torch.from_numpy(rewards).to(device)
            episode_rewards += rewards_t

            # Track per-grasp metrics
            if hasattr(underlying_env, '_prev_logs_grasped'):
                for i in range(num_envs):
                    logs_grasped = int(underlying_env._prev_logs_grasped[i].item())
                    alignment = underlying_env._prev_grasp_alignment[i].item() if hasattr(underlying_env, '_prev_grasp_alignment') else 0.0
                    stability = underlying_env._prev_grasp_stability[i].item() if hasattr(underlying_env, '_prev_grasp_stability') else 1.0
                    total_grasps += 1
                    total_logs_grasped += logs_grasped
                    episode_logs_cleared[i] += logs_grasped
                    ep_cycle_count[i] += 1
                    starting = max(1, int(episode_starting_logs[i].item()))
                    clear_pct_now = int(episode_logs_cleared[i].item()) / starting * 100
                    ep_clearing_curves[i].append(round(clear_pct_now, 1))
                    if logs_grasped > 0:
                        successful_grasps += 1
                        ep_successful_grasps[i] += 1
                        ep_alignment_sum[i] += alignment * logs_grasped
                        ep_stability_sum[i] += stability * logs_grasped
                    else:
                        failed_grasps += 1
                        ep_failed_grasps[i] += 1

            # Check for episode completion
            for i in range(num_envs):
                if dones[i]:
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

                    if hasattr(underlying_env, '_final_episode_knocked_off'):
                        ep_knocked_off = int(underlying_env._final_episode_knocked_off[i].item())
                    elif hasattr(underlying_env, '_logs_knocked_off'):
                        ep_knocked_off = int(underlying_env._logs_knocked_off[i].item())
                    else:
                        ep_knocked_off = 0
                    total_knocked_off += ep_knocked_off
                    knocked_off_per_episode.append(ep_knocked_off)
                    logs_cleared_per_episode.append(logs_cleared)

                    per_ep_cycles.append(int(ep_cycle_count[i].item()))
                    per_ep_clearing_curves.append(ep_clearing_curves[i][:])

                    n_success = int(ep_successful_grasps[i].item())
                    n_fail = int(ep_failed_grasps[i].item())
                    n_total = n_success + n_fail
                    per_ep_success_rates.append(n_success / max(1, n_total) * 100)
                    per_ep_throughputs.append(logs_cleared / max(1, n_success))
                    per_ep_alignments.append(float(ep_alignment_sum[i].item()) / max(1, logs_cleared))
                    per_ep_stabilities.append(float(ep_stability_sum[i].item()) / max(1, logs_cleared))

                    print(f"[Eval] Episode {episodes_done}: reward={ep_reward:.2f}, cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs} logs), knocked_off={ep_knocked_off}")

                    # Reset per-episode tracking
                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    ep_successful_grasps[i] = 0
                    ep_failed_grasps[i] = 0
                    ep_alignment_sum[i] = 0.0
                    ep_stability_sum[i] = 0.0
                    ep_cycle_count[i] = 0
                    ep_clearing_curves[i] = []
                    if has_variable_logs:
                        episode_starting_logs[i] = int(underlying_env._per_env_log_counts[i].item())

                    if episodes_done >= args_cli.num_episodes:
                        break

    # ── Report results ────────────────────────────────────────────────
    if episodes_done > 0:
        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        def _std(vals, mean_val):
            return (sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0

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
        avg_cycles = _mean(per_ep_cycles)
        std_cycles = _std(per_ep_cycles, avg_cycles)

        per_ep_cycles_to_95 = []
        for curve in per_ep_clearing_curves:
            c95 = next((i + 1 for i, pct in enumerate(curve) if pct >= 95.0), len(curve))
            per_ep_cycles_to_95.append(c95)
        avg_cycles_to_95 = _mean(per_ep_cycles_to_95)
        std_cycles_to_95 = _std(per_ep_cycles_to_95, avg_cycles_to_95)

        std_reward = _std(episode_rewards_list, avg_reward)
        std_clear_pct = _std(clearing_percentages, avg_clear_pct)
        std_success_rate = _std(per_ep_success_rates, avg_success_rate)
        std_throughput = _std(per_ep_throughputs, avg_throughput)
        std_alignment = _std(per_ep_alignments, avg_alignment)
        std_stability = _std(per_ep_stabilities, avg_stability)
        std_knocked_off_pct = _std(knocked_off_pcts, avg_knocked_off_pct)

        print("=" * 60)
        print(f"\n[Eval] ====== RESULTS ({episodes_done} episodes) ======")
        print(f"[Eval] Episode Reward:      {avg_reward:.2f} +/- {std_reward:.2f}")
        print(f"[Eval] Pile Cleared:        {avg_clear_pct:.1f} +/- {std_clear_pct:.1f}%")
        print(f"[Eval] Full Clear Rate:     {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
        print(f"[Eval] Grasp Success Rate:  {avg_success_rate:.1f} +/- {std_success_rate:.1f}%")
        print(f"[Eval] Throughput:          {avg_throughput:.2f} +/- {std_throughput:.2f} logs/grasp")
        print(f"[Eval] Alignment:           {avg_alignment:.3f} +/- {std_alignment:.3f}")
        print(f"[Eval] Stability:           {avg_stability:.3f} +/- {std_stability:.3f}")
        print(f"[Eval] Knocked Off:         {avg_knocked_off_pct:.1f} +/- {std_knocked_off_pct:.1f}%")
        print(f"[Eval] Avg Cycles:          {avg_cycles:.1f} +/- {std_cycles:.1f}")
        print(f"[Eval] Cycles to 95%:       {avg_cycles_to_95:.1f} +/- {std_cycles_to_95:.1f}")
        print(f"[Eval] Total Logs Grasped:  {total_logs_grasped}")
        print(f"[Eval] =======================")

        if args_cli.save_metrics:
            output_dir = args_cli.metrics_output_dir or log_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

            metrics = {
                "eval_config": {
                    "method": "sac",
                    "checkpoint": checkpoint,
                    "task": args_cli.task,
                    "num_envs": num_envs,
                    "num_episodes": episodes_done,
                    "seed": args_cli.seed,
                    "domain_randomization": args_cli.domain_randomization,
                    "obs_noise": args_cli.obs_noise,
                    "action_noise": args_cli.action_noise,
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
                    "cycles":            {"mean": avg_cycles, "std": std_cycles},
                    "cycles_to_95pct":   {"mean": avg_cycles_to_95, "std": std_cycles_to_95},
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
                    "cycles": per_ep_cycles,
                    "cycles_to_95pct": per_ep_cycles_to_95,
                    "clearing_curves": per_ep_clearing_curves,
                },
            }

            os.makedirs(output_dir, exist_ok=True)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n[Eval] Metrics saved to: {metrics_file}")

    sb3_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
