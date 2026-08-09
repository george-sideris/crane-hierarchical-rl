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
from crane_testbed.agents.scoring_actor_critic import ScoringActorCritic


@configclass
class PointNetActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic config with PointNet-specific fields (norm_type, encoder_lr_scale)."""

    norm_type: str = "batchnorm"
    """Normalization type: "layernorm" (recommended) or "batchnorm" (legacy)."""

    encoder_lr_scale: float = 0.2
    """Encoder learning rate as fraction of base LR (used when optimizer is overridden)."""


@configclass
class ScoringActorCriticCfg(RslRlPpoActorCriticCfg):
    """Categorical-over-points AC (P3). num_points must match the env obs."""
    num_points: int = 2048
    temperature: float = 1.0


@configclass
class CranePPORunnerCfg_ScoringScratch(RslRlOnPolicyRunnerCfg):
    """PURE RL with the scoring head - no BC init, encoder trained from scratch.

    The paper-era pure-RL result (~56% clearing) used a GAUSSIAN head over absolute coordinates:
    PPO had to teach a PointNet to localize AND to regress metres, and it never did - the policy
    degenerated to aiming at memorized coordinates. The categorical head changes the problem:
    every action is "pick one of the observed points", so the policy never regresses coordinates
    and the action is grounded in the observation by construction. Whether that alone makes RL
    from scratch viable is the experiment.

    Config differs from the fine-tune in every way that matters for from-scratch learning:
    entropy bonus ON (needs exploration; the fine-tune's problem was the opposite - a bonus
    eroding a sharp BC prior), higher LR, encoder NOT frozen (perception must be learned),
    more steps per env for a stabler value estimate.
    """
    num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 50
    experiment_name = "crane_scoring_scratch"
    empirical_normalization = False
    policy = ScoringActorCriticCfg(
        class_name="rsl_rl.modules.ScoringActorCritic",
        init_noise_std=0.15,
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
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_Scoring(RslRlOnPolicyRunnerCfg):
    """P3 = BC->RL (scoring). Conservative fine-tune: small dz sigma, modest entropy so the
    categorical stays near the BC scores, low LR."""
    num_steps_per_env = 8
    max_iterations = 400
    save_interval = 50
    experiment_name = "crane_scoring_ppo"
    empirical_normalization = False
    policy = ScoringActorCriticCfg(
        class_name="rsl_rl.modules.ScoringActorCritic",   # registered in train.py
        init_noise_std=0.05,
        actor_hidden_dims=[128, 64],      # unused by the AC; kept for cfg schema
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        # NO entropy bonus (measured 2026-08-06: at 0.003 the categorical's entropy ROSE ~1 nat
        # over 100 iters while throughput fell 15.8->11.0 and stability 0.93->0.78 - the bonus
        # dominated the tiny per-point advantages and smeared the BC score sharpness. ln(2048)
        # ceiling makes the categorical FAR more entropy-sensitive than a Gaussian; sampling
        # from BC-sharp scores explores plenty on its own. Same lesson as the OG BCFinetune cfg.)
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_ScoringV2(RslRlOnPolicyRunnerCfg):
    """P3b = BC->RL (scoring), the 2026-08-08 rebuild. Same policy as _Scoring; what changed is
    everything that made the first two attempts uninterpretable.

    Post-mortem of takes 1 and 2 (both DEGRADED P2c):
      1. The reward paid for grasps that despawn nothing. `logs_grasped` is counted before the
         lift-height gate but _despawn_grasped_logs is gated ON it, so 12-18% of cycles were
         paid while clearing zero - a hackable channel that fully explains "training reward up,
         deployed argmax down". Fixed in the ENV (cfg.reward_requires_lift), not here.
      2. num_steps_per_env 8 x num_envs 4 = 32 transitions per update, minibatch 8. PPO wants
         thousands; the sweep script itself measured ~10% batch variance against a ~3% signal.
         At 16 envs this gives 16x32 = 512 per update, minibatch 128.
      3. Checkpoints were selected on training reward while the policy DEPLOYS argmax - given
         (1) that selects the most reward-hacked policy. save_interval 10 exists so
         sweep_checkpoints.sh can pick by argmax eval instead.
      4. The critic starts random against an already-good BC actor, so the first updates apply
         garbage advantages to a prior worth protecting (--critic_warmup_iters in train.py).

    Intended launch: --num_envs 16 --critic_warmup_iters 25 --freeze_encoder --anneal_sigma
    """
    num_steps_per_env = 32
    max_iterations = 400
    save_interval = 10                    # dense checkpoints -> argmax selection after the fact
    experiment_name = "crane_scoring_ppo_v2"
    empirical_normalization = False
    policy = ScoringActorCriticCfg(
        class_name="rsl_rl.modules.ScoringActorCritic",
        init_noise_std=0.05,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,                 # unchanged: see _Scoring for the measurement
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


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

    policy = PointNetActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",  # Registered in train.py
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        norm_type="batchnorm",  # Legacy: compatible with existing trained policies
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
class CranePPORunnerCfg_PointCloud_BCFinetune(RslRlOnPolicyRunnerCfg):
    """Conservative PPO config for BC→RL fine-tuning with point cloud.

    Lower LR, tighter clipping, and no entropy bonus to preserve BC knowledge
    while the fresh asymmetric critic catches up.
    """

    num_steps_per_env = 8  # 512 transitions (vs 256 default) for smoother updates
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_pointcloud_bc_finetune"
    empirical_normalization = False

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = PointNetActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        norm_type="batchnorm",  # Must match BC checkpoint
        encoder_lr_scale=0.2,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,  # 2x tighter — bound policy shift per iteration
        entropy_coef=0.0,  # Don't push toward randomness
        num_learning_epochs=3,  # Less overfitting on small batches
        num_mini_batches=4,
        learning_rate=1e-4,  # 3x lower — preserve BC knowledge
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,  # Tighter adaptive LR throttle
        max_grad_norm=1.0,
    )


@configclass
class CranePPORunnerCfg_PointCloud_PureRL(RslRlOnPolicyRunnerCfg):
    """PPO config for from-scratch RL with point cloud.

    More data per iteration, LayerNorm for batch-independent normalization,
    standard exploration pressure for from-scratch training.
    """

    num_steps_per_env = 16  # 64 envs × 16 steps = 1024 transitions per iteration
    max_iterations = 5000
    save_interval = 50  # Less frequent saves for long runs
    experiment_name = "crane_pointcloud_purerl"
    empirical_normalization = False

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = PointNetActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        norm_type="layernorm",  # Batch-independent, stable for RL
        encoder_lr_scale=0.2,
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
class CranePPORunnerCfg_PointCloud_Curriculum(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane RL with pile-size curriculum.

    Longer rollouts (num_steps_per_env=15) to collect enough data from
    short early-curriculum episodes. Higher max_iterations for curriculum ramp.
    """

    num_steps_per_env = 15
    max_iterations = 3000
    save_interval = 10
    experiment_name = "crane_pointcloud_curriculum"
    empirical_normalization = False

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = PointNetActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        norm_type="batchnorm",
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
class CranePPORunnerCfg_PointCloud_v2(RslRlOnPolicyRunnerCfg):
    """PPO configuration for crane RL with point cloud observations (v2).

    Improvements over v1:
    - LayerNorm instead of BatchNorm (better RL training stability)
    - Asymmetric actor-critic: PointNet actor + state-based MLP critic
      (requires env to return both "policy" and "critic" observation groups)

    Environment:
    - Actor obs: 1024-point cloud in base frame (3072 dims)
    - Critic obs: 128D state vector (top 32 log poses) — privileged
    - Action: 4D [x, y, z, yaw]
    """

    num_steps_per_env = 4
    max_iterations = 5000
    save_interval = 10
    experiment_name = "crane_pointcloud_v2"
    empirical_normalization = False

    # Logging
    logger = "tensorboard"
    neptune_project = None
    wandb_project = None
    resume = False
    load_run = None
    load_checkpoint = None

    policy = PointNetActorCriticCfg(
        class_name="rsl_rl.modules.PointNetActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        norm_type="layernorm",
        encoder_lr_scale=0.2,
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
