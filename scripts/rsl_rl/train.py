# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--bc_checkpoint", type=str, default=None, help="Path to BC checkpoint for fine-tuning (loads actor weights only, skips optimizer).")
parser.add_argument("--freeze_encoder", action="store_true", default=False, help="Freeze PointNet encoder weights (use with --bc_checkpoint).")
parser.add_argument("--save_debug_pointclouds", action="store_true", default=False, help="Save debug point cloud .npy and plots on first observation.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
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

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
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
from isaaclab.utils.dict import print_dict

# Compatibility: dump_pickle/dump_yaml may be missing or broken in some Isaac Lab versions
import os as _os, pickle as _pickle, yaml as _yaml
try:
    from isaaclab.utils.io import dump_yaml as _isaaclab_dump_yaml
except Exception:
    _isaaclab_dump_yaml = None

try:
    from isaaclab.utils.dict import class_to_dict
except Exception:
    from isaaclab.utils import class_to_dict


def dump_pickle(filename, data):
    _os.makedirs(_os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        _pickle.dump(data, f)


def dump_yaml(filename, data, sort_keys=False):
    _os.makedirs(_os.path.dirname(filename), exist_ok=True)
    if _isaaclab_dump_yaml is not None:
        try:
            return _isaaclab_dump_yaml(filename, data, sort_keys=sort_keys)
        except Exception:
            pass
    try:
        if not isinstance(data, dict):
            data = class_to_dict(data)
    except Exception:
        if hasattr(data, "__dict__"):
            data = data.__dict__
    with open(filename, "w") as f:
        _yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import crane_testbed.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments (use task name for clarity)
    # Convert task name to folder-friendly format: Isaac-Crane-Full-DR-MR-v0 -> crane_full_dr_mr_v0
    task_folder = args_cli.task.lower().replace("isaac-", "").replace("-", "_")
    log_root_path = os.path.join("logs", "rsl_rl", task_folder)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # create isaac environment
    import sys as _sys
    _sys.stderr.write(f"[DEBUG] Creating environment for task: {args_cli.task}\n")
    _sys.stderr.flush()
    if "Depth" in args_cli.task:
        _sys.stderr.write(f"[DEBUG] Detected Depth task - using CraneDepthDirectEnv\n")
        _sys.stderr.flush()
        # Special handling for depth tasks - create env directly to avoid gym wrapper issues
        import sys
        from pathlib import Path
        envs_dir = Path(__file__).parent.parent / "envs"
        _sys.stderr.write(f"[DEBUG] Adding envs_dir to path: {envs_dir}\n")
        if str(envs_dir) not in sys.path:
            sys.path.insert(0, str(envs_dir))
        try:
            from crane_depth_direct_env import CraneDepthDirectEnv
            _sys.stderr.write(f"[DEBUG] Import succeeded\n")
        except Exception as e:
            _sys.stderr.write(f"[DEBUG] Import failed: {e}\n")
            raise
        _sys.stderr.write(f"[DEBUG] Creating CraneDepthDirectEnv with cfg observation_space={getattr(env_cfg, 'observation_space', 'N/A')}\n")
        env = CraneDepthDirectEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)
        _sys.stderr.write(f"[DEBUG] CraneDepthDirectEnv created with {env.num_observations} obs\n")
    elif "PointCloud" in args_cli.task:
        _sys.stderr.write(f"[DEBUG] Detected PointCloud task - using CranePointCloudDirectEnv\n")
        _sys.stderr.flush()
        import sys
        from pathlib import Path
        envs_dir = Path(__file__).parent.parent / "envs"
        if str(envs_dir) not in sys.path:
            sys.path.insert(0, str(envs_dir))
        from crane_pointcloud_direct_env import CranePointCloudDirectEnv
        _sys.stderr.write(f"[DEBUG] Creating CranePointCloudDirectEnv with cfg observation_space={getattr(env_cfg, 'observation_space', 'N/A')}\n")
        env = CranePointCloudDirectEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)
        env.log_dir = log_dir
        env.save_debug_pointclouds = args_cli.save_debug_pointclouds
        _sys.stderr.write(f"[DEBUG] CranePointCloudDirectEnv created with {env.num_observations} obs\n")
    else:
        _sys.stderr.write(f"[DEBUG] Using gym.make() for non-Depth task\n")
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # Override optimizer if policy supports differential learning rates (e.g., PointNet encoder_lr_scale)
    # RSL-RL's PPO creates optimizer with policy.parameters() at a single LR, ignoring get_param_groups()
    if hasattr(runner.alg.actor_critic, 'get_param_groups'):
        base_lr = agent_cfg.algorithm.learning_rate
        param_groups = runner.alg.actor_critic.get_param_groups(base_lr)
        runner.alg.optimizer = torch.optim.Adam(param_groups, lr=base_lr)
        print(f"[INFO]: Optimizer overridden with encoder_lr_scale param groups (base_lr={base_lr})")

    # load the checkpoint
    if args_cli.bc_checkpoint:
        # BC fine-tuning: load only actor weights, skip optimizer
        print(f"[INFO]: Loading BC checkpoint for fine-tuning: {args_cli.bc_checkpoint}")
        bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=agent_cfg.device)
        model_state = bc_ckpt.get('model_state_dict', bc_ckpt)
        # RSL-RL stores actor_critic on the algorithm's actor_critic attribute
        # Try different possible locations
        if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor_critic'):
            actor_critic = runner.alg.actor_critic
        elif hasattr(runner, 'actor_critic'):
            actor_critic = runner.actor_critic
        else:
            # Fallback: check PPO algorithm structure
            actor_critic = runner.alg.policy if hasattr(runner.alg, 'policy') else None
            if actor_critic is None:
                raise AttributeError(f"Cannot find actor_critic. Runner attrs: {dir(runner)}, Alg attrs: {dir(runner.alg)}")
        # Load with strict=False to allow missing critic weights if needed
        # RSL-RL's load_state_dict may return bool or tuple depending on version
        result = actor_critic.load_state_dict(model_state, strict=False)
        if isinstance(result, tuple):
            missing, unexpected = result
            if missing:
                print(f"[INFO]: Missing keys (will be randomly initialized): {missing}")
            if unexpected:
                print(f"[INFO]: Unexpected keys (ignored): {unexpected}")
        print("[INFO]: BC actor weights loaded. Optimizer starting fresh.")
        if args_cli.freeze_encoder:
            if hasattr(actor_critic, 'encoder'):
                frozen_params = 0
                for param in actor_critic.encoder.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                # CRITICAL: Freeze BatchNorm running statistics too.
                # requires_grad=False only freezes learnable params (weight, bias)
                # but running_mean/running_var are buffers that update in train mode.
                actor_critic.encoder.eval()
                # Monkey-patch train() so PPO's policy.train() doesn't re-enable encoder training mode
                _orig_train = actor_critic.train
                def _patched_train(mode=True):
                    _orig_train(mode)
                    actor_critic.encoder.eval()  # Always keep encoder in eval mode
                    return actor_critic
                actor_critic.train = _patched_train
                print(f"[INFO]: Encoder frozen ({frozen_params} params, BatchNorm stats locked)")
            else:
                print("[WARN]: --freeze_encoder set but model has no 'encoder' attribute")
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
