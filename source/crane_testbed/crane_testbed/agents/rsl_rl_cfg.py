# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

# Import CNN actor-critic for depth observations
from crane_testbed.agents.cnn_actor_critic import CNNActorCritic
# Import PointNet actor-critic for point cloud observations
from crane_testbed.agents.pointnet_actor_critic import PointNetActorCritic


@configclass
class CranePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane hierarchical RL (original version with strategic state)."""

    num_steps_per_env = 4  # Collect 4 grasp cycles before policy update
    max_iterations = 5000  # 5000 iterations (20000 total cycles across 4 envs)
    save_interval = 10  # Save every 10 iterations (more frequent for testing)
    experiment_name = "crane_hierarchical"
    empirical_normalization = False  # No normalization (for backward compatibility)

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


@configclass
class CranePPORunnerCfg_Markov(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane hierarchical RL (Markovian version).

    Differences from CranePPORunnerCfg:
    - Removes strategic state (logs_remaining, cycle_count) from observations
    - Enables normalization for stable critic training
    - Forces purely Markovian policy based on visible log positions
    """

    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_hierarchical_markov"
    empirical_normalization = True  # Enable normalization

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_Simplified(RslRlOnPolicyRunnerCfg):
    """PPO configuration for simplified crane RL.

    Simplified environment:
    - 2D action space: [y, z] (X and yaw are fixed)
    - 64D observation: Top 32 logs × (y, z) positions
    """

    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_simplified"
    empirical_normalization = False  # Observations already normalized in env

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Network for 64 obs -> 2 action
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Standard entropy coefficient
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_Full(RslRlOnPolicyRunnerCfg):
    """PPO configuration for full crane RL.

    Full environment:
    - 4D action space: [x, y, z, yaw] (policy controls all dimensions)
    - 128D observation: Top 32 logs × (x, y, z, yaw) positions
    """

    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_full"
    empirical_normalization = False  # Observations already normalized in env

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Larger network for 128 obs -> 4 action
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Standard entropy coefficient
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_Depth(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane RL with depth image observations.

    Uses CNN actor-critic instead of MLP.

    Environment:
    - Observation: 128x128 masked depth image (16384 dims)
    - Action: 4D [x, y, z, yaw]
    """

    num_steps_per_env = 4  # Same as 32-pose training
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_depth"
    empirical_normalization = False  # Depth already normalized in env

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.CNNActorCritic",  # Use our CNN actor-critic (registered in train.py)
        init_noise_std=1.0,
        # These are passed to CNNActorCritic but some are ignored
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,  # Same as 32-pose training
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_PointCloud(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane RL with point cloud observations.

    Uses PointNet actor-critic instead of MLP.

    Environment:
    - Observation: 1024-point cloud in base frame (1024 * 3 = 3072 dims)
    - Action: 4D [x, y, z, yaw]
    """

    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_pointcloud"
    empirical_normalization = False  # Point cloud coords are already meaningful

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",  # Registered in train.py
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
