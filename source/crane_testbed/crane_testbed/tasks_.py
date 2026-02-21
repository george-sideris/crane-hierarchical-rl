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
    failure_penalty: float = -3.0


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
    failure_penalty: float = -3.0


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
    failure_penalty: float = -3.0


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
    failure_penalty: float = -3.0


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
# Gym Task Registrations
# Naming: MR=Multiplicative Raw, MN=Multiplicative Normalized,
#         AR=Additive Raw, AN=Additive Normalized
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
