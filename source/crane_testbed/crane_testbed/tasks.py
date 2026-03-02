# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Crane log grasping task registration for Isaac Lab.
"""

import gymnasium as gym

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

# Import the crane environment - add scripts path
import sys
from pathlib import Path
crane_scripts_path = Path(__file__).resolve().parents[3] / "scripts" / "envs"
sys.path.insert(0, str(crane_scripts_path))

import crane_rl_env
from crane_rl_env import CraneDirectEnv, CraneDirectEnvCfg

import crane_rl_env_no_deposition
from crane_rl_env_no_deposition import CraneDirectEnvNoDeposition

import crane_rl_env_simplified
from crane_rl_env_simplified import CraneDirectEnvSimplified, CraneDirectEnvCfg as CraneDirectEnvCfgSimplified

import crane_rl_env_full
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull

import crane_depth_direct_env
from crane_depth_direct_env import CraneDepthDirectEnv

import crane_pointcloud_direct_env
from crane_pointcloud_direct_env import CranePointCloudDirectEnv


##
# Crane Hierarchical RL Task - Configure for hierarchical mode
##

# Create hierarchical RL config
@configclass
class CraneDirectEnvCfg_TRAIN(CraneDirectEnvCfg):
    """Crane environment configured for hierarchical RL training (with strategic state)."""
    use_hierarchical_rl: bool = True  # Enable hierarchical mode
    episode_length_s = 600.0  # Max episode time (50 cycles @ ~12s/cycle)
    # Observation includes: 32 logs × 4 features + strategic state (logs_remaining, cycle_count, etc.)
    observation_space = (32 * 4) + 4  # 132
    enable_domain_randomization: bool = False  # Keep fixed patterns for backward compatibility


@configclass
class CraneDirectEnvCfg_TRAIN_Markov(CraneDirectEnvCfg):
    """Crane environment configured for Markovian hierarchical RL (no strategic state)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    # Observation includes ONLY: 32 logs × 4 features (NO strategic state)
    observation_space = (32 * 4)  # 128
    enable_domain_randomization: bool = True  # Randomize log patterns on every reset


@configclass
class CraneDirectEnvCfg_TRAIN_Markov_NoDR(CraneDirectEnvCfg):
    """Markov config WITHOUT domain randomization - for testing observation normalization."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    observation_space = (32 * 4)  # 128 (same as Markov)
    enable_domain_randomization: bool = False  # Fixed pattern for easier learning


gym.register(
    id="Isaac-Crane-Direct-v0",
    entry_point="crane_rl_env:CraneDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg",
    },
)

# No-deposition version (original with strategic state)
gym.register(
    id="Isaac-Crane-NoDepo-v0",
    entry_point="crane_rl_env_no_deposition:CraneDirectEnvNoDeposition",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg",
    },
)

# No-deposition version (Markovian without strategic state)
gym.register(
    id="Isaac-Crane-NoDepo-Markov-v0",
    entry_point="crane_rl_env_no_deposition:CraneDirectEnvNoDeposition",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_TRAIN_Markov,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Markov",
    },
)

# No-deposition Markov WITHOUT domain randomization (for testing)
gym.register(
    id="Isaac-Crane-NoDepo-Markov-NoDR-v0",
    entry_point="crane_rl_env_no_deposition:CraneDirectEnvNoDeposition",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_TRAIN_Markov_NoDR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Markov",
    },
)


##
# Simplified environment: 3D action space [y, z, yaw], Y-binned observation
##

@configclass
class CraneDirectEnvCfg_Simplified(CraneDirectEnvCfgSimplified):
    """Simplified env: 2D action [y,z], top-32 logs observation."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    # 2D action space: [y, z] - X and yaw are fixed
    action_space = 2
    # Top 32 logs × 2 (y, z) = 64 values
    max_logs_obs: int = 32
    observation_space = 64  # 32 * 2
    enable_domain_randomization: bool = False  # Start with fixed pattern


@configclass
class CraneDirectEnvCfg_Simplified_DR(CraneDirectEnvCfgSimplified):
    """Simplified env with domain randomization."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 2
    max_logs_obs: int = 32
    observation_space = 64
    enable_domain_randomization: bool = True


# Simplified environment (no domain randomization)
gym.register(
    id="Isaac-Crane-Simplified-v0",
    entry_point="crane_rl_env_simplified:CraneDirectEnvSimplified",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Simplified,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Simplified",
    },
)

# Simplified environment with domain randomization
gym.register(
    id="Isaac-Crane-Simplified-DR-v0",
    entry_point="crane_rl_env_simplified:CraneDirectEnvSimplified",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Simplified_DR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Simplified",
    },
)


##
# Full environment: 4D action space [x, y, z, yaw], 128D observation
##

@configclass
class CraneDirectEnvCfg_Full(CraneDirectEnvCfgFull):
    """Full env: 4D action [x, y, z, yaw], top-32 logs observation (128 values)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    # 4D action space: [x, y, z, yaw] - policy controls all dimensions
    action_space = 4
    # Top 32 logs × 4 (x, y, z, yaw) = 128 values
    max_logs_obs: int = 32
    observation_space = 128  # 32 * 4
    enable_domain_randomization: bool = False  # Start with fixed pattern


# =============================================================================
# Reward formula variants (2x2 factorial: formula × normalization)
# v0: multiplicative, raw       - original design
# v1: multiplicative, normalized - normalized efficiency
# v2: additive, raw             - efficiency + alignment (raw)
# v3: additive, normalized      - efficiency + alignment (normalized)
# =============================================================================

# --- No Domain Randomization ---

# v0: multiplicative, raw (ORIGINAL)
@configclass
class CraneDirectEnvCfg_Full_v0(CraneDirectEnvCfgFull):
    """v0: Multiplicative reward, raw log count (original design)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# v1: multiplicative, normalized
@configclass
class CraneDirectEnvCfg_Full_v1(CraneDirectEnvCfgFull):
    """v1: Multiplicative reward, normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# v2: additive, raw
@configclass
class CraneDirectEnvCfg_Full_v2(CraneDirectEnvCfgFull):
    """v2: Additive reward (efficiency + alignment + stability), raw log count."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "additive"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# v3: additive, normalized
@configclass
class CraneDirectEnvCfg_Full_v3(CraneDirectEnvCfgFull):
    """v3: Additive reward (efficiency + alignment + stability), normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "additive"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# --- With Domain Randomization ---

# v0: multiplicative, raw (ORIGINAL)
@configclass
class CraneDirectEnvCfg_Full_DR_v0(CraneDirectEnvCfgFull):
    """DR v0: Multiplicative reward, raw log count (original design)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# v1: multiplicative, normalized
@configclass
class CraneDirectEnvCfg_Full_DR_v1(CraneDirectEnvCfgFull):
    """DR v1: Multiplicative reward, normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# v2: additive, raw
@configclass
class CraneDirectEnvCfg_Full_DR_v2(CraneDirectEnvCfgFull):
    """DR v2: Additive reward (efficiency + alignment + stability), raw log count."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "additive"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# v3: additive, normalized
@configclass
class CraneDirectEnvCfg_Full_DR_v3(CraneDirectEnvCfgFull):
    """DR v3: Additive reward (efficiency + alignment + stability), normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "additive"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# =============================================================================
# Progress-gated reward variants (encourage endgame pile clearing)
# v0: progress_gate (fraction-of-pile throughput) with soft alignment/stability gates
# =============================================================================

@configclass
class CraneDirectEnvCfg_Full_PG_v0(CraneDirectEnvCfgFull):
    """PG v0: Progress-gated reward (fraction of pile removed) + soft quality gates."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False

    # New reward mode implemented in crane_rl_env_full (_compute_grasp_reward)
    reward_formula: str = "progress_gate"

    # Gates + scaling (see env code; these are read via getattr so optional)
    reward_progress_scale: float = 100.0      # scales g/N0 to the ~[0,100] episode range
    reward_gate_eps_align: float = 0.10       # prevents early collapse to 0 reward
    reward_gate_eps_stab: float = 0.10
    reward_clear_bonus: float = 20.0          # bonus when the pile is fully cleared
    empty_target_penalty: float = -0.5        # extra penalty for aiming at empty space

    # Keep stability as a gate (recommended if you care about stable grasps)
    use_stability_reward: bool = True
    # Failure penalty (0-log grasp)
    failure_penalty: float = -0.5


@configclass
class CraneDirectEnvCfg_Full_DR_PG_v0(CraneDirectEnvCfgFull):
    """DR PG v0: Progress-gated reward under domain randomization."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 4
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True

    reward_formula: str = "progress_gate"
    reward_progress_scale: float = 100.0
    reward_gate_eps_align: float = 0.10
    reward_gate_eps_stab: float = 0.10
    reward_clear_bonus: float = 20.0
    empty_target_penalty: float = -0.5

    use_stability_reward: bool = True
    failure_penalty: float = -0.5

# =============================================================================
# Cos/Sin Yaw Encoding Variants (5D action space)
# Same reward formulas as 4D, but with 5D action: [x, y, z, cos(2*yaw), sin(2*yaw)]
# =============================================================================

# --- No Domain Randomization ---

# CosSin-MR: multiplicative, raw
@configclass
class CraneDirectEnvCfg_Full_CosSin_MR_v0(CraneDirectEnvCfgFull):
    """CosSin-MR v0: 5D action, multiplicative reward, raw log count."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    max_logs_obs: int = 32
    observation_space = 128  # 32 logs × 4 features (x, y, z, yaw)
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# CosSin-MN: multiplicative, normalized
@configclass
class CraneDirectEnvCfg_Full_CosSin_MN_v0(CraneDirectEnvCfgFull):
    """CosSin-MN v0: 5D action, multiplicative reward, normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# CosSin-MR-NoAlign: multiplicative raw, no alignment reward — g * s
@configclass
class CraneDirectEnvCfg_Full_CosSin_MR_NoAlign_v0(CraneDirectEnvCfgFull):
    """CosSin-MR v0 without alignment: g * s (ablation)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# CosSin-MR-NoStab: multiplicative raw, no stability reward — g * a
@configclass
class CraneDirectEnvCfg_Full_CosSin_MR_NoStab_v0(CraneDirectEnvCfgFull):
    """CosSin-MR v0 without stability: g * a (ablation)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = True
    use_stability_reward: bool = False
    failure_penalty: float = -1.0


# CosSin-MR-ThroughputOnly: just g, no alignment or stability
@configclass
class CraneDirectEnvCfg_Full_CosSin_MR_ThroughputOnly_v0(CraneDirectEnvCfgFull):
    """CosSin-MR v0 throughput only: g (ablation)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = False
    use_stability_reward: bool = False
    failure_penalty: float = -1.0


# CosSin-AR: additive, raw
@configclass
class CraneDirectEnvCfg_Full_CosSin_AR_v0(CraneDirectEnvCfgFull):
    """CosSin-AR v0: 5D action, additive reward (eff + align + stab), raw log count."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "additive"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# CosSin-AN: additive, normalized
@configclass
class CraneDirectEnvCfg_Full_CosSin_AN_v0(CraneDirectEnvCfgFull):
    """CosSin-AN v0: 5D action, additive reward (eff + align + stab), normalized efficiency."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "additive"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# CosSin-PG: progress-gated reward
@configclass
class CraneDirectEnvCfg_Full_CosSin_PG_v0(CraneDirectEnvCfgFull):
    """CosSin-PG v0: 5D action, progress-gated reward (no DR)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "progress_gate"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# --- With Domain Randomization ---

# CosSin-DR-MR: multiplicative, raw + DR
@configclass
class CraneDirectEnvCfg_Full_CosSin_DR_MR_v0(CraneDirectEnvCfgFull):
    """CosSin-DR-MR v0: 5D action, multiplicative reward, raw log count + DR."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# CosSin-DR-MN: multiplicative, normalized + DR
@configclass
class CraneDirectEnvCfg_Full_CosSin_DR_MN_v0(CraneDirectEnvCfgFull):
    """CosSin-DR-MN v0: 5D action, multiplicative reward, normalized efficiency + DR."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -1.0


# CosSin-DR-AR: additive, raw + DR
@configclass
class CraneDirectEnvCfg_Full_CosSin_DR_AR_v0(CraneDirectEnvCfgFull):
    """CosSin-DR-AR v0: 5D action, additive reward (eff + align + stab), raw + DR."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "additive"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# CosSin-DR-AN: additive, normalized + DR
@configclass
class CraneDirectEnvCfg_Full_CosSin_DR_AN_v0(CraneDirectEnvCfgFull):
    """CosSin-DR-AN v0: 5D action, additive reward (eff + align + stab), normalized + DR."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "additive"
    normalize_reward: bool = True
    use_stability_reward: bool = True
    failure_penalty: float = -0.5


# CosSin-DR-PG: progress-gated + DR
@configclass
class CraneDirectEnvCfg_Full_CosSin_DR_PG_v0(CraneDirectEnvCfgFull):
    """CosSin-DR-PG v0: 5D action, progress-gated reward + DR."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = True
    reward_formula: str = "progress_gate"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0

# =============================================================================
# Gym Task Registrations
# Naming: MR=Multiplicative Raw, MN=Multiplicative Normalized,
#         AR=Additive Raw, AN=Additive Normalized
#         CosSin=5D action space with cos/sin yaw encoding
# =============================================================================

# --- No Domain Randomization ---
gym.register(
    id="Isaac-Crane-Full-MR-v0",  # Multiplicative Raw (original)
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-MN-v0",  # Multiplicative Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_v1,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-AR-v0",  # Additive Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_v2,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-AN-v0",  # Additive Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_v3,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)


gym.register(
    id="Isaac-Crane-Full-PG-v0",  # Progress-gated reward (no DR)
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_PG_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

# --- With Domain Randomization ---
gym.register(
    id="Isaac-Crane-Full-DR-MR-v0",  # DR + Multiplicative Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_DR_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-DR-MN-v0",  # DR + Multiplicative Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_DR_v1,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-DR-AR-v0",  # DR + Additive Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_DR_v2,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-DR-AN-v0",  # DR + Additive Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_DR_v3,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)



gym.register(
    id="Isaac-Crane-Full-DR-PG-v0",  # DR + Progress-gated reward
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_DR_PG_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

# --- Cos/Sin Yaw Encoding (5D Action Space) - No DR ---
gym.register(
    id="Isaac-Crane-Full-CosSin-MR-v0",  # 5D action, Multiplicative Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MR_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-MN-v0",  # 5D action, Multiplicative Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MN_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-MR-NoAlign-v0",  # 5D action, MR without alignment
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MR_NoAlign_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-MR-NoStab-v0",  # 5D action, MR without stability
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MR_NoStab_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-MR-ThroughputOnly-v0",  # 5D action, just g
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MR_ThroughputOnly_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-AR-v0",  # 5D action, Additive Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_AR_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-AN-v0",  # 5D action, Additive Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_AN_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-PG-v0",  # 5D action, Progress-Gated
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_PG_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

# --- Cos/Sin Yaw Encoding (5D Action Space) - With DR ---
gym.register(
    id="Isaac-Crane-Full-CosSin-DR-MR-v0",  # 5D action, DR + Multiplicative Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_DR_MR_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-DR-MN-v0",  # 5D action, DR + Multiplicative Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_DR_MN_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-DR-AR-v0",  # 5D action, DR + Additive Raw
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_DR_AR_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-DR-AN-v0",  # 5D action, DR + Additive Normalized
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_DR_AN_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-DR-PG-v0",  # 5D action, DR + Progress-Gated
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_DR_PG_v0,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

##
# Full CosSin Curriculum variant (no camera, pile-size curriculum)
##

@configclass
class CraneDirectEnvCfg_Full_CosSin_MR_Curriculum(CraneDirectEnvCfgFull):
    """CosSin-MR with pile-size curriculum (no camera)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    max_logs_obs: int = 32
    observation_space = 128
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_stability_reward: bool = True
    failure_penalty: float = -1.0
    curriculum_schedule = [(0, 20), (100, 60), (200, 120), (300, 200)]

@configclass
class CraneDirectEnvCfg_Full_CosSin_ClearBonus_Ablation(CraneDirectEnvCfgFull):
    """Clearing bonus + curriculum + normalized reward, no alignment/stability. Pure poses (128D)."""
    use_hierarchical_rl: bool = True
    episode_length_s = 600.0
    action_space = 5                           # [x, y, z, cos(2*yaw), sin(2*yaw)]
    max_logs_obs: int = 32
    observation_space = 128                    # 32 logs × 4 features (pure pose, no camera)
    enable_domain_randomization: bool = False
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True              # equalize throughput
    use_alignment_reward: bool = True
    use_stability_reward: bool = True
    clearing_bonus_scale: float = 100.0        # end-of-episode clearing bonus
    curriculum_schedule = [(0, 20), (300, 60), (600, 120), (1000, 200)]

gym.register(
    id="Isaac-Crane-Full-CosSin-ClearBonus-v0",
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_ClearBonus_Ablation,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

gym.register(
    id="Isaac-Crane-Full-CosSin-Curriculum-v0",
    entry_point="crane_rl_env_full:CraneDirectEnvFull",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_Full_CosSin_MR_Curriculum,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Full",
    },
)

##
# Depth observation environment: CNN-based visual RL
##

@configclass
class CraneDepthEnvCfg(CraneDirectEnvCfgFull):
    """Crane environment with depth image observations."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 4
    # Depth observation: 256x256 = 65536
    observation_space = 65536
    enable_domain_randomization: bool = False
    # Depth-specific settings
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = True


@configclass
class CraneDepthEnvCfg_DR(CraneDirectEnvCfgFull):
    """Crane depth environment with domain randomization."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 4
    observation_space = 65536
    enable_domain_randomization: bool = True
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = True


@configclass
class CraneDepthEnvCfg_Raw(CraneDirectEnvCfgFull):
    """Crane depth environment with RAW depth (no semantic segmentation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 4
    observation_space = 65536
    enable_domain_randomization: bool = False
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = False  # No segmentation - raw depth


# --- Depth with Cos/Sin Yaw Encoding (5D Action Space) ---

@configclass
class CraneDepthEnvCfg_CosSin(CraneDirectEnvCfgFull):
    """Crane depth environment with 5D CosSin action space."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    observation_space = 65536  # 256x256 depth image
    enable_domain_randomization: bool = False
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = True


@configclass
class CraneDepthEnvCfg_CosSin_DR(CraneDirectEnvCfgFull):
    """Crane depth environment with 5D CosSin action + domain randomization."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    observation_space = 65536
    enable_domain_randomization: bool = True
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = True


@configclass
class CraneDepthEnvCfg_CosSin_Raw(CraneDirectEnvCfgFull):
    """Crane depth environment with 5D CosSin action + RAW depth (no semantic mask)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    observation_space = 65536
    enable_domain_randomization: bool = False
    depth_height: int = 256
    depth_width: int = 256
    depth_min: float = 1.5
    depth_max: float = 8.0
    use_semantic_mask: bool = False  # No segmentation - raw depth


# Depth environment (no DR)
gym.register(
    id="Isaac-Crane-Depth-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

# Depth environment with DR
gym.register(
    id="Isaac-Crane-Depth-DR-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg_DR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

# Depth environment with RAW depth (no semantic segmentation)
gym.register(
    id="Isaac-Crane-Depth-Raw-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg_Raw,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

# Depth environment with 5D CosSin action (no DR)
gym.register(
    id="Isaac-Crane-Depth-CosSin-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg_CosSin,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

# Depth environment with 5D CosSin action + DR
gym.register(
    id="Isaac-Crane-Depth-CosSin-DR-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg_CosSin_DR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

# Depth environment with 5D CosSin action + RAW depth (no semantic mask)
gym.register(
    id="Isaac-Crane-Depth-CosSin-Raw-v0",
    entry_point="crane_depth_direct_env:CraneDepthDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDepthEnvCfg_CosSin_Raw,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_Depth",
    },
)

##
# Point cloud observation environment: PointNet-based visual RL
##

@configclass
class CranePointCloudEnvCfg(CraneDirectEnvCfgFull):
    """Crane environment with point cloud observations (4D action, no DR)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 4
    # Point cloud observation: 1024 points × 3 coords = 3072
    observation_space = 3072
    enable_domain_randomization: bool = False
    # Point cloud settings
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0


@configclass
class CranePointCloudEnvCfg_DR(CraneDirectEnvCfgFull):
    """Crane point cloud environment with domain randomization (4D action)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 4
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0


##
# Point Cloud CosSin reward variants (no DR)
##

@configclass
class CranePointCloudEnvCfg_CosSin_MR(CraneDirectEnvCfgFull):
    """PCD-CosSin-MR: multiplicative reward, raw throughput."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False


@configclass
class CranePointCloudEnvCfg_CosSin_MN(CraneDirectEnvCfgFull):
    """PCD-CosSin-MN: multiplicative reward, normalized efficiency."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True


@configclass
class CranePointCloudEnvCfg_CosSin_AR(CraneDirectEnvCfgFull):
    """PCD-CosSin-AR: additive reward, raw throughput."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "additive"
    normalize_reward: bool = False


@configclass
class CranePointCloudEnvCfg_CosSin_AN(CraneDirectEnvCfgFull):
    """PCD-CosSin-AN: additive reward, normalized efficiency."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "additive"
    normalize_reward: bool = True


##
# Point Cloud CosSin reward ablation variants (masked PCD, no DR)
##

# NoAlign: multiplicative raw, no alignment reward — g * s
@configclass
class CranePointCloudEnvCfg_CosSin_MR_NoAlign(CraneDirectEnvCfgFull):
    """PCD-CosSin-MR-NoAlign: multiplicative, no alignment (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = False
    use_stability_reward: bool = True


# NoStab: multiplicative raw, no stability reward — g * a
@configclass
class CranePointCloudEnvCfg_CosSin_MR_NoStab(CraneDirectEnvCfgFull):
    """PCD-CosSin-MR-NoStab: multiplicative, no stability (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = True
    use_stability_reward: bool = False


# ThroughputOnly: just g, no alignment or stability
@configclass
class CranePointCloudEnvCfg_CosSin_MR_ThroughputOnly(CraneDirectEnvCfgFull):
    """PCD-CosSin-MR-ThroughputOnly: throughput only (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_alignment_reward: bool = False
    use_stability_reward: bool = False


##
# Point Cloud CosSin Raw PCD variants (full scene, no segmentation)
##

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR(CraneDirectEnvCfgFull):
    """PCD-CosSin-Raw-MR: raw point cloud, multiplicative reward."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_raw_pointcloud: bool = True


@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_NoAlign(CraneDirectEnvCfgFull):
    """PCD-CosSin-Raw-MR-NoAlign: raw PCD, no alignment (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_raw_pointcloud: bool = True
    use_alignment_reward: bool = False
    use_stability_reward: bool = True


@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_NoStab(CraneDirectEnvCfgFull):
    """PCD-CosSin-Raw-MR-NoStab: raw PCD, no stability (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_raw_pointcloud: bool = True
    use_alignment_reward: bool = True
    use_stability_reward: bool = False


@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_ThroughputOnly(CraneDirectEnvCfgFull):
    """PCD-CosSin-Raw-MR-ThroughputOnly: raw PCD, throughput only (ablation)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    use_raw_pointcloud: bool = True
    use_alignment_reward: bool = False
    use_stability_reward: bool = False


##
# Point Cloud CosSin Raw PCD + Reward Shaping Ablation Variants
##

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_Normalized(CranePointCloudEnvCfg_CosSin_Raw_MR):
    """Raw PCD, multiplicative normalized reward: r = (g/avail)*10*α*ς."""
    normalize_reward: bool = True
    normalized_efficiency_scale: float = 10.0

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_CurrBonus(CranePointCloudEnvCfg_CosSin_Raw_MR):
    """Raw PCD, multiplicative + curriculum + tiered clearing bonuses at 50/70/90%."""
    clearing_bonus_scale: float = 50.0
    clearing_bonus_thresholds: list[float] = [0.6, 0.75, 0.9]
    curriculum_schedule = [(0, 20), (150, 60), (300, 120), (400, 200)]


##
# Point Cloud CosSin Raw PCD + Proportional Clearing Bonus
##

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_PropClear(CranePointCloudEnvCfg_CosSin_Raw_MR):
    """Raw PCD, CosSin-MR + proportional clearing bonus (50.0 scale)."""
    clearing_bonus_scale: float = 50.0
    proportional_clearing_bonus: bool = True


##
# Point Cloud CosSin Raw PCD + Asymmetric Critic
##

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_MR_Asym(CranePointCloudEnvCfg_CosSin_Raw_MR):
    """Raw PCD, CosSin-MR + asymmetric critic (128D state for critic)."""
    asymmetric_critic: bool = True


##
# Point Cloud CosSin Curriculum variants (pile-size curriculum for exploration)
##

@configclass
class CranePointCloudEnvCfg_CosSin_MR_Curriculum(CraneDirectEnvCfgFull):
    """PCD-CosSin-MR with pile-size curriculum (from scratch)."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False
    curriculum_schedule = [(0, 20), (100, 60), (200, 120), (300, 200)]


##
# Point Cloud CosSin reward variants (with DR)
##

@configclass
class CranePointCloudEnvCfg_CosSin_DR_MR(CraneDirectEnvCfgFull):
    """PCD-CosSin-DR-MR: multiplicative reward, raw throughput + DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False


@configclass
class CranePointCloudEnvCfg_CosSin_DR_MN(CraneDirectEnvCfgFull):
    """PCD-CosSin-DR-MN: multiplicative reward, normalized efficiency + DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "multiplicative"
    normalize_reward: bool = True


@configclass
class CranePointCloudEnvCfg_CosSin_DR_AR(CraneDirectEnvCfgFull):
    """PCD-CosSin-DR-AR: additive reward, raw throughput + DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "additive"
    normalize_reward: bool = False


@configclass
class CranePointCloudEnvCfg_CosSin_DR_AN(CraneDirectEnvCfgFull):
    """PCD-CosSin-DR-AN: additive reward, normalized efficiency + DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    reward_formula: str = "additive"
    normalize_reward: bool = True


##
# Point Cloud gym registrations
##

# Legacy 4D action variants
gym.register(
    id="Isaac-Crane-PointCloud-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-DR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_DR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

# 5D CosSin action — reward ablation variants (no DR)
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-MR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-MN-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-AR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_AR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-AN-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_AN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

# 5D CosSin action — masked PCD reward ablation variants (no DR)
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-MR-NoAlign-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR_NoAlign,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-MR-NoStab-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR_NoStab,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-MR-ThroughputOnly-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR_ThroughputOnly,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

# 5D CosSin action — raw PCD variants
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-NoAlign-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_NoAlign,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-NoStab-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_NoStab,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-ThroughputOnly-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_ThroughputOnly,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-PropClear-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_PropClear,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_Asym,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-Normalized-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_Normalized,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-MR-CurrBonus-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_MR_CurrBonus,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

# 5D CosSin action — reward ablation variants (with DR)
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-DR-MR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_DR_MR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-DR-MN-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_DR_MN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-DR-AR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_DR_AR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-DR-AN-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_DR_AN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

# Backwards-compatible aliases (default to MR)
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-CosSin-DR-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_DR_MR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud",
    },
)


# Curriculum variants (pile-size curriculum)
gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Curriculum-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_MR_Curriculum,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud_Curriculum",
    },
)

##
# Point Cloud v2: LayerNorm + Asymmetric Actor-Critic
# Uses PointNet (LayerNorm) actor with state-based MLP critic
##

@configclass
class CranePointCloudEnvCfg_v2(CraneDirectEnvCfgFull):
    """PCD v2: LayerNorm + asymmetric critic, 5D CosSin action, no DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    observation_space = 3072  # 1024 points × 3 coords (actor)
    enable_domain_randomization: bool = False
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    asymmetric_critic: bool = True  # Critic sees 128D state vector
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False


@configclass
class CranePointCloudEnvCfg_v2_DR(CraneDirectEnvCfgFull):
    """PCD v2: LayerNorm + asymmetric critic, 5D CosSin action, with DR."""
    use_hierarchical_rl: bool = True
    enable_camera: bool = True
    episode_length_s = 600.0
    action_space = 5  # [x, y, z, cos(2*yaw), sin(2*yaw)]
    observation_space = 3072
    enable_domain_randomization: bool = True
    num_points: int = 1024
    depth_range_min: float = 1.0
    depth_range_max: float = 10.0
    asymmetric_critic: bool = True
    reward_formula: str = "multiplicative"
    normalize_reward: bool = False


# v2 registrations (LayerNorm + asymmetric critic, always 5D CosSin)
gym.register(
    id="Isaac-Crane-PointCloud-v2",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_v2,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud_v2",
    },
)

gym.register(
    id="Isaac-Crane-PointCloud-DR-v2",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_v2_DR,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud_v2",
    },
)


##
# Pure RL from scratch: additive reward, low exponents, curriculum, asymmetric critic
##

@configclass
class CranePointCloudEnvCfg_CosSin_Raw_PureRL(CranePointCloudEnvCfg_CosSin_Raw_MR):
    """Pure RL from scratch: additive reward, low exponents, curriculum, asymmetric critic."""
    asymmetric_critic: bool = True
    reward_formula: str = "additive"
    normalize_reward: bool = True
    alignment_exponent: int = 2
    stability_exponent: int = 2
    failure_penalty: float = -0.5
    curriculum_schedule = [(0, 20), (200, 60), (500, 120), (1000, 200)]


gym.register(
    id="Isaac-Crane-PointCloud-CosSin-Raw-PureRL-v0",
    entry_point="crane_pointcloud_direct_env:CranePointCloudDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CranePointCloudEnvCfg_CosSin_Raw_PureRL,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg_PointCloud_PureRL",
    },
)
