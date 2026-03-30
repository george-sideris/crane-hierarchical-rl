# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # -- PPO hyperparameter overrides (for sweeps)
    ppo_group = parser.add_argument_group("ppo_overrides", description="PPO hyperparameter overrides.")
    ppo_group.add_argument("--learning_rate", type=float, default=None, help="PPO learning rate.")
    ppo_group.add_argument("--clip_param", type=float, default=None, help="PPO clip range.")
    ppo_group.add_argument("--entropy_coef", type=float, default=None, help="Entropy bonus coefficient.")
    ppo_group.add_argument("--num_learning_epochs", type=int, default=None, help="PPO epochs per update.")
    ppo_group.add_argument("--num_mini_batches", type=int, default=None, help="Number of minibatches per epoch.")
    ppo_group.add_argument("--num_steps_per_env", type=int, default=None, help="Rollout length per env.")
    ppo_group.add_argument("--gae_lambda", type=float, default=None, help="GAE lambda.")
    ppo_group.add_argument("--desired_kl", type=float, default=None, help="Desired KL for adaptive LR.")
    ppo_group.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    ppo_group.add_argument("--init_noise_std", type=float, default=None, help="Initial action noise std.")


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    # -- PPO hyperparameter overrides
    if hasattr(args_cli, "learning_rate") and args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if hasattr(args_cli, "clip_param") and args_cli.clip_param is not None:
        agent_cfg.algorithm.clip_param = args_cli.clip_param
    if hasattr(args_cli, "entropy_coef") and args_cli.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args_cli.entropy_coef
    if hasattr(args_cli, "num_learning_epochs") and args_cli.num_learning_epochs is not None:
        agent_cfg.algorithm.num_learning_epochs = args_cli.num_learning_epochs
    if hasattr(args_cli, "num_mini_batches") and args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches
    if hasattr(args_cli, "num_steps_per_env") and args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if hasattr(args_cli, "gae_lambda") and args_cli.gae_lambda is not None:
        agent_cfg.algorithm.lam = args_cli.gae_lambda
    if hasattr(args_cli, "desired_kl") and args_cli.desired_kl is not None:
        agent_cfg.algorithm.desired_kl = args_cli.desired_kl
    if hasattr(args_cli, "gamma") and args_cli.gamma is not None:
        agent_cfg.algorithm.gamma = args_cli.gamma
    if hasattr(args_cli, "init_noise_std") and args_cli.init_noise_std is not None:
        agent_cfg.policy.init_noise_std = args_cli.init_noise_std

    return agent_cfg
