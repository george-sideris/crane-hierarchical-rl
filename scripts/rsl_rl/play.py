# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Evaluation arguments
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to run for evaluation (if set, exits after completion)")
parser.add_argument("--save_metrics", action="store_true", default=False, help="Save evaluation metrics to JSON file")
parser.add_argument("--metrics_output_dir", type=str, default=None, help="Output directory for metrics (default: checkpoint directory)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# automatically enable cameras for depth/pointcloud tasks
if args_cli.task and ("Depth" in args_cli.task or "PointCloud" in args_cli.task):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import os
import time
import torch
from datetime import datetime

import rsl_rl.modules
from rsl_rl.runners import OnPolicyRunner

# Register custom CNN actor-critic with RSL-RL so OnPolicyRunner can find it
from crane_testbed.agents.cnn_actor_critic import CNNActorCritic
rsl_rl.modules.CNNActorCritic = CNNActorCritic

# Register custom PointNet actor-critic with RSL-RL so OnPolicyRunner can find it
from crane_testbed.agents.pointnet_actor_critic import PointNetActorCritic
rsl_rl.modules.PointNetActorCritic = PointNetActorCritic

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import crane_testbed.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    if "Depth" in args_cli.task:
        # Special handling for depth tasks - create env directly to avoid gym wrapper issues
        import sys
        from pathlib import Path
        envs_dir = Path(__file__).parent.parent / "envs"
        if str(envs_dir) not in sys.path:
            sys.path.insert(0, str(envs_dir))
        from crane_depth_direct_env import CraneDepthDirectEnv
        print(f"[INFO] Creating CraneDepthDirectEnv for depth-based inference")
        env = CraneDepthDirectEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)
    elif "PointCloud" in args_cli.task:
        # Special handling for point cloud tasks
        import sys
        from pathlib import Path
        envs_dir = Path(__file__).parent.parent / "envs"
        if str(envs_dir) not in sys.path:
            sys.path.insert(0, str(envs_dir))
        from crane_pointcloud_direct_env import CranePointCloudDirectEnv
        print(f"[INFO] Creating CranePointCloudDirectEnv for pointcloud-based inference")
        env = CranePointCloudDirectEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # Get underlying environment for metrics tracking
    underlying_env = env.unwrapped
    if hasattr(underlying_env, 'env'):
        underlying_env = underlying_env.env  # Unwrap further if needed

    # Metrics tracking
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

    # Per-env tracking
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_logs_cleared = torch.zeros(num_envs, device=device, dtype=torch.int32)
    episode_starting_logs = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_successful_grasps = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_failed_grasps = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_alignment_sum = torch.zeros(num_envs, device=device)
    ep_stability_sum = torch.zeros(num_envs, device=device)

    # Per-episode result lists (for mean ± std reporting)
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []

    # Get initial observations (triggers reset if needed)
    obs, _ = env.get_observations()

    # Read starting log counts AFTER reset (so we get the actual randomized counts)
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

    if args_cli.num_episodes:
        print(f"[Eval] Running evaluation for {args_cli.num_episodes} episodes...")
    print("=" * 60)
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)

            # Track rewards
            episode_rewards += rewards

            # Track per-grasp metrics if available
            if hasattr(underlying_env, '_prev_logs_grasped'):
                for i in range(num_envs):
                    logs_grasped = int(underlying_env._prev_logs_grasped[i].item())
                    alignment = underlying_env._prev_grasp_alignment[i].item() if hasattr(underlying_env, '_prev_grasp_alignment') else 0.0
                    stability = underlying_env._prev_grasp_stability[i].item() if hasattr(underlying_env, '_prev_grasp_stability') else 1.0
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
            for i in range(num_envs):
                if dones[i]:
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
                    ep_knocked_off = int(underlying_env._logs_knocked_off[i].item()) if hasattr(underlying_env, '_logs_knocked_off') else 0
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

                    print(f"[Eval] Episode {episodes_done}: reward={ep_reward:.2f}, cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs} logs), knocked_off={ep_knocked_off}")

                    # Reset per-episode tracking
                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    ep_successful_grasps[i] = 0
                    ep_failed_grasps[i] = 0
                    ep_alignment_sum[i] = 0.0
                    ep_stability_sum[i] = 0.0
                    # Update starting logs for next episode
                    if has_variable_logs:
                        episode_starting_logs[i] = int(underlying_env._per_env_log_counts[i].item())

                    # Check if we've done enough episodes
                    if args_cli.num_episodes and episodes_done >= args_cli.num_episodes:
                        break

        # Check exit conditions
        if args_cli.num_episodes and episodes_done >= args_cli.num_episodes:
            break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Print and save metrics if episodes were tracked
    if episodes_done > 0:
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
        print(f"\n[Eval] ====== RESULTS ({episodes_done} episodes) ======")
        print(f"[Eval] Episode Reward:      {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"[Eval] Pile Cleared:        {avg_clear_pct:.1f} ± {std_clear_pct:.1f}%")
        print(f"[Eval] Full Clear Rate:     {full_clear_rate:.1f}% ({piles_fully_cleared}/{episodes_done})")
        print(f"[Eval] Grasp Success Rate:  {avg_success_rate:.1f} ± {std_success_rate:.1f}%")
        print(f"[Eval] Throughput:          {avg_throughput:.2f} ± {std_throughput:.2f} logs/grasp")
        print(f"[Eval] Alignment:           {avg_alignment:.3f} ± {std_alignment:.3f}")
        print(f"[Eval] Stability:           {avg_stability:.3f} ± {std_stability:.3f}")
        print(f"[Eval] Knocked Off:         {avg_knocked_off_pct:.1f} ± {std_knocked_off_pct:.1f}%")
        print(f"[Eval] Total Logs Grasped:  {total_logs_grasped}")
        print(f"[Eval] =======================")

        # Save metrics if requested
        if args_cli.save_metrics:
            if args_cli.metrics_output_dir:
                output_dir = args_cli.metrics_output_dir
            else:
                output_dir = log_dir

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(output_dir, f"eval_metrics_{timestamp}.json")

            metrics = {
                "eval_config": {
                    "method": "rl",
                    "checkpoint": resume_path,
                    "task": args_cli.task,
                    "num_envs": num_envs,
                    "num_episodes": episodes_done,
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

            os.makedirs(output_dir, exist_ok=True)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"\n[Eval] Metrics saved to: {metrics_file}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
