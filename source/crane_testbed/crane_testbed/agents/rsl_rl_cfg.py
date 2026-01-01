# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class CranePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane hierarchical RL."""

    num_steps_per_env = 4  # Collect 4 grasp cycles before policy update
    max_iterations = 5000  # 5000 iterations (20000 total cycles across 4 envs)
    save_interval = 100  # Save every 100 iterations
    experiment_name = "crane_hierarchical"
    empirical_normalization = False

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # Larger network for complex decision
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Encourage exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,  # Immediate reward (cycle completes in same step)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
