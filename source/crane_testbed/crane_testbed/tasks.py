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


##
# Crane Hierarchical RL Task - Configure for hierarchical mode
##

# Create hierarchical RL config
@configclass
class CraneDirectEnvCfg_TRAIN(CraneDirectEnvCfg):
    """Crane environment configured for hierarchical RL training."""
    use_hierarchical_rl: bool = True  # Enable hierarchical mode
    episode_length_s = 600.0  # Max episode time (50 cycles @ ~12s/cycle)


gym.register(
    id="Isaac-Crane-Direct-v0",
    entry_point="crane_rl_env:CraneDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CraneDirectEnvCfg_TRAIN,
        "rsl_rl_cfg_entry_point": "crane_testbed.agents.rsl_rl_cfg:CranePPORunnerCfg",
    },
)
