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
parser.add_argument("--visualize", action="store_true", default=False,
                    help="Save per-step 2-panel PCD viz PNGs (PointCloud tasks only)")
parser.add_argument("--paper_viz", action="store_true", default=False,
                    help="Save paper-quality pipeline + progression figures (PointCloud tasks only)")
parser.add_argument("--viz_dir", type=str, default=None,
                    help="Output directory for viz PNGs (default: checkpoint dir / viz or paper_viz)")
parser.add_argument("--domain_randomization", action="store_true", default=False,
                    help="Enable domain randomization (random log count 20-200)")
parser.add_argument("--obs_noise", type=float, default=0.0,
                    help="Gaussian noise σ added to PCD coordinates before PointNet (meters)")
parser.add_argument("--action_noise", type=float, default=0.0,
                    help="Gaussian noise σ added to policy action output before workspace scaling")
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

# enable cameras when viz flags are used with PointCloud tasks
if (args_cli.visualize or args_cli.paper_viz) and args_cli.task and "PointCloud" in args_cli.task:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
# Filter out empty/whitespace-only args that cause Hydra LexerNoViableAltException
hydra_args = [a for a in hydra_args if a.strip()]
if hydra_args:
    print(f"[DEBUG] Unrecognized args passed to Hydra: {hydra_args}")
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

    # domain randomization (random log count 20-200)
    if args_cli.domain_randomization and hasattr(env_cfg, 'enable_domain_randomization'):
        env_cfg.enable_domain_randomization = True
        print(f"[INFO] Domain randomization enabled")

    # noise injection info
    if args_cli.obs_noise > 0:
        print(f"[INFO] Observation noise σ: {args_cli.obs_noise} m")
    if args_cli.action_noise > 0:
        print(f"[INFO] Action noise σ: {args_cli.action_noise}")

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
        # Ensure RGB is available for paper viz, strip it otherwise for performance
        if args_cli.paper_viz and hasattr(env_cfg, 'camera_cfg'):
            if "rgb" not in env_cfg.camera_cfg.data_types:
                env_cfg.camera_cfg.data_types = list(env_cfg.camera_cfg.data_types) + ["rgb"]
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
    ppo_runner.load(resume_path, load_optimizer=False)

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

    # --- Visualization setup ---
    is_pointcloud_task = args_cli.task and "PointCloud" in args_cli.task
    paper_viz_active = args_cli.paper_viz and is_pointcloud_task
    step_viz_active = args_cli.visualize and is_pointcloud_task

    if (args_cli.paper_viz or args_cli.visualize) and not is_pointcloud_task:
        print("[WARN] --paper_viz / --visualize only work with PointCloud tasks, ignoring")

    if paper_viz_active or step_viz_active:
        from eval_viz import (
            get_raw_pipeline_data, get_log_pointcloud_base_frame,
            decode_action, save_step_viz, save_raw_pipeline_viz,
            save_raw_stacked_pipeline_viz, save_raw_episode_progression,
        )
        import numpy as np
        _use_raw_pcd = getattr(env_cfg, 'use_raw_pointcloud', False)
        _num_points_viz = getattr(env_cfg, 'num_points', 1024)
        _depth_range_viz = (getattr(env_cfg, 'depth_range_min', 1.0),
                            getattr(env_cfg, 'depth_range_max', 10.0))
        underlying_env._compute_action_space_bounds()

    if step_viz_active:
        viz_dir = args_cli.viz_dir or os.path.join(log_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"[INFO] Saving per-step viz to: {viz_dir}")

    if paper_viz_active:
        paper_viz_dir = args_cli.viz_dir or os.path.join(log_dir, "paper_viz")
        os.makedirs(paper_viz_dir, exist_ok=True)
        print(f"[INFO] Saving paper viz to: {paper_viz_dir}")

    total_viz_step = 0

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
    ep_cycle_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
    ep_clearing_curves = [[] for _ in range(num_envs)]  # per-env list of clearing % at each cycle
    if paper_viz_active:
        paper_episode_data = [[] for _ in range(num_envs)]

    # Per-episode result lists (for mean ± std reporting)
    per_ep_success_rates = []
    per_ep_throughputs = []
    per_ep_alignments = []
    per_ep_stabilities = []
    per_ep_cycles = []
    per_ep_clearing_curves = []

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
            # Inject observation noise (Gaussian on PCD coordinates)
            if args_cli.obs_noise > 0:
                obs = obs + torch.randn_like(obs) * args_cli.obs_noise

            # agent stepping
            actions = policy(obs)

            # Inject action noise (Gaussian on raw policy output, before workspace scaling)
            if args_cli.action_noise > 0:
                actions = actions + torch.randn_like(actions) * args_cli.action_noise

            # Pre-step: collect viz data before env.step modifies state
            if paper_viz_active:
                paper_pre_step = {}
                for i in range(num_envs):
                    min_b = underlying_env._action_bounds_min[i].cpu().numpy()
                    max_b = underlying_env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    pdata = get_raw_pipeline_data(underlying_env, i, _num_points_viz,
                                                  depth_range=_depth_range_viz, raw_pcd=_use_raw_pcd)
                    pdata.update({"x": x, "y": y, "z": z, "yaw": yaw,
                                  "bounds_min": min_b.copy(), "bounds_max": max_b.copy(),
                                  "step_idx": int(ep_cycle_count[i].item())})
                    paper_pre_step[i] = pdata
            elif step_viz_active:
                viz_pre_step = {}
                for i in range(num_envs):
                    min_b = underlying_env._action_bounds_min[i].cpu().numpy()
                    max_b = underlying_env._action_bounds_max[i].cpu().numpy()
                    x, y, z, yaw = decode_action(actions[i], min_b, max_b)
                    pc = get_log_pointcloud_base_frame(underlying_env, i, _num_points_viz,
                                                       depth_range=_depth_range_viz, raw_pcd=_use_raw_pcd)
                    viz_pre_step[i] = {"pts": pc.cpu().numpy(), "x": x, "y": y, "z": z, "yaw": yaw,
                                       "bounds_min": min_b.copy(), "bounds_max": max_b.copy()}

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
                    ep_cycle_count[i] += 1
                    starting = max(1, int(episode_starting_logs[i].item()))
                    clear_pct_now = int(episode_logs_cleared[i].item()) / starting * 100
                    ep_clearing_curves[i].append(round(clear_pct_now, 1))
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

                    # --- Per-step visualization ---
                    if step_viz_active and not paper_viz_active:
                        d = viz_pre_step[i]
                        save_step_viz(d["pts"], d["x"], d["y"], d["z"], d["yaw"],
                                      total_viz_step, viz_dir,
                                      logs_grasped=logs_grasped, alignment=alignment,
                                      bounds_min=d["bounds_min"], bounds_max=d["bounds_max"])

                    if paper_viz_active:
                        pdata = paper_pre_step[i]
                        pdata["logs_grasped"] = logs_grasped
                        pdata["alignment"] = alignment
                        pdata["stability"] = stability
                        knocked_off = int(underlying_env._prev_cycle_knocked_off[i].item()) if hasattr(underlying_env, '_prev_cycle_knocked_off') else 0
                        if hasattr(underlying_env, '_prev_logs_remaining'):
                            pre_remaining = int(underlying_env._prev_logs_remaining[i].item())
                            pdata["logs_remaining"] = pre_remaining - logs_grasped - knocked_off
                        else:
                            pdata["logs_remaining"] = None
                        pdata["knocked_off"] = knocked_off
                        paper_episode_data[i].append(pdata)
                        save_raw_pipeline_viz(pdata, pdata["x"], pdata["y"], pdata["z"], pdata["yaw"],
                                              total_viz_step, paper_viz_dir,
                                              logs_grasped=logs_grasped, alignment=alignment, stability=stability,
                                              depth_range=_depth_range_viz,
                                              bounds_min=pdata["bounds_min"], bounds_max=pdata["bounds_max"])

                    if paper_viz_active or step_viz_active:
                        total_viz_step += 1

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
                    per_ep_clearing_curves.append(ep_clearing_curves[i][:])  # copy the curve

                    # Per-episode grasp metrics
                    n_success = int(ep_successful_grasps[i].item())
                    n_fail = int(ep_failed_grasps[i].item())
                    n_total = n_success + n_fail
                    per_ep_success_rates.append(n_success / max(1, n_total) * 100)
                    per_ep_throughputs.append(logs_cleared / max(1, n_success))
                    per_ep_alignments.append(float(ep_alignment_sum[i].item()) / max(1, logs_cleared))
                    per_ep_stabilities.append(float(ep_stability_sum[i].item()) / max(1, logs_cleared))

                    print(f"[Eval] Episode {episodes_done}: reward={ep_reward:.2f}, cleared={clear_pct:.1f}% ({logs_cleared}/{starting_logs} logs), knocked_off={ep_knocked_off}")

                    # Generate paper viz figures for this episode
                    if paper_viz_active and len(paper_episode_data[i]) > 0:
                        ep_data = paper_episode_data[i]
                        # Trim stuck tail (consecutive misses at same remaining count)
                        trimmed = list(ep_data)
                        if len(trimmed) >= 2:
                            last_remaining = trimmed[-1].get("logs_remaining")
                            if last_remaining is not None:
                                stuck_start = len(trimmed)
                                for si in range(len(trimmed) - 1, -1, -1):
                                    d = trimmed[si]
                                    if (d.get("logs_remaining") == last_remaining
                                            and (d.get("logs_grasped") or 0) == 0):
                                        stuck_start = si
                                    else:
                                        break
                                if stuck_start < len(trimmed):
                                    n_removed = len(trimmed) - stuck_start
                                    trimmed = trimmed[:stuck_start]
                                    if n_removed > 0:
                                        print(f"[PaperViz] Trimmed {n_removed} stuck repeated grasps (remaining={last_remaining})")
                            if len(trimmed) == 0:
                                trimmed = [ep_data[0]]

                        ep_bmin = trimmed[0].get("bounds_min")
                        ep_bmax = trimmed[0].get("bounds_max")

                        # Representative grasps (begin, mid, end)
                        n = len(trimmed)
                        rep = [0] + ([n // 2] if n > 2 else []) + ([n - 1] if n > 1 else [])
                        save_raw_stacked_pipeline_viz([trimmed[j] for j in rep], episodes_done, paper_viz_dir,
                                                      depth_range=_depth_range_viz, bounds_min=ep_bmin, bounds_max=ep_bmax)

                        # 9 consecutive grasps (3 from each section) if enough data
                        if n >= 6:
                            consec = list(trimmed[0:3])
                            mid = n // 2
                            consec += trimmed[max(0, mid-1):max(0, mid-1)+3]
                            consec += trimmed[max(0, n-3):n]
                            save_raw_stacked_pipeline_viz(consec, episodes_done, paper_viz_dir,
                                                          depth_range=_depth_range_viz, bounds_min=ep_bmin, bounds_max=ep_bmax,
                                                          suffix="_consecutive")

                        # Episode progression grid
                        save_raw_episode_progression(trimmed, episodes_done, paper_viz_dir,
                                                     bounds_min=ep_bmin, bounds_max=ep_bmax)

                    # Reset per-episode tracking
                    episode_rewards[i] = 0.0
                    episode_logs_cleared[i] = 0
                    ep_successful_grasps[i] = 0
                    ep_failed_grasps[i] = 0
                    ep_alignment_sum[i] = 0.0
                    ep_stability_sum[i] = 0.0
                    ep_cycle_count[i] = 0
                    ep_clearing_curves[i] = []
                    if paper_viz_active:
                        paper_episode_data[i] = []
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

        avg_cycles = _mean(per_ep_cycles)
        std_cycles = _std(per_ep_cycles, avg_cycles)

        # Derive cycles to 95% from clearing curves
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
        print(f"[Eval] Avg Cycles:          {avg_cycles:.1f} ± {std_cycles:.1f}")
        print(f"[Eval] Cycles to 95%:       {avg_cycles_to_95:.1f} ± {std_cycles_to_95:.1f}")
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
