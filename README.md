# Crane Log Grasping Environment

Isaac Lab environment for hierarchical reinforcement learning applied to crane log manipulation.

## Overview

This project implements a simulated forestry crane environment for learning grasp target selection. The crane must clear logs from a rack pile by executing pick-place cycles. The key research question is whether RL can learn better target selection than a simple heuristic baseline.

**Architecture:**
- **High-level policy**: Selects grasp target (x, y, z, yaw)
- **Low-level heuristic**: Executes pick-place cycle via finite state machine
- **Reward**: Number of logs grasped × orientation alignment (bonus for alignment > 0.7)

## Status

- **Heuristic baseline**: Working. Achieves ~10 logs/grasp with good pile management.
- **RL training**: Ongoing research. Training is unstable, likely due to sparse rewards and large action space (continuous 4D target selection).

This repository represents the current state of the work and is intended as a research portfolio piece.

## Installation

This project requires Isaac Lab with Docker. Follow these steps:

### 1. Clone Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

### 2. Set up the crane_testbed extension

Copy this `crane_testbed` directory into the Isaac Lab root:

```bash
cp -r /path/to/crane_testbed ./
```

### 3. Configure Docker to mount crane_testbed

Edit `docker/docker-compose.yaml` and add the crane_testbed mount under the `volumes:` section (around line 69):

```yaml
# Mount crane_testbed for persistence across container restarts
- type: bind
  source: ../crane_testbed
  target: /workspace/crane_testbed
```

### 4. Build and run the Docker container

```bash
cd docker
./container.sh start  # Builds base image and starts container
./container.sh enter  # Enter the container
```

Inside the container:

```bash
# Install the crane_testbed extension
cd /workspace/crane_testbed
python -m pip install -e .
```

### 5. Run the heuristic baseline

```bash
cd /workspace/isaaclab
python /workspace/crane_testbed/scripts/envs/crane_rl_env.py --num_envs 1
```

You should see the crane automatically selecting and grasping logs from the pile.

## Environment Details

- **Observation space**: 132-dim (32 logs × 4 features + 4 strategic state features)
  - Per-log: position (x, y, z), yaw orientation
  - Strategic: deposited log count, cycles remaining, previous grasp quality
- **Action space**: 4-dim continuous (target x, y, z, yaw in base frame)
- **Episode length**: 50 grasp cycles or until rack is empty
- **Parallel simulation**: Tested with 64 parallel environments

## File Structure

```
crane_testbed/
├── assets/
│   ├── urdf/                     # Crane URDF model
│   └── scenes/                   # USD scene assets
├── scripts/
│   ├── envs/
│   │   └── crane_rl_env.py       # Main environment implementation
│   └── rsl_rl/                   # Training/evaluation scripts
├── source/
│   └── crane_testbed/            # Package setup
├── README.md
└── requirements.txt
```

## Training

Training uses RSL-RL (PPO):

```bash
# Launch training with 64 parallel environments
python /workspace/crane_testbed/scripts/envs/crane_rl_env.py --num_envs 64 --use_target_selection_policy
```

**Current training challenges:**
- Sparse rewards (only at grasp completion)
- Large continuous action space
- Difficult credit assignment (action → outcome delayed by ~15 seconds of physics)

## Future Work

- Asymmetric actor-critic with privileged information (log contact forces, pile stability metrics)
- Reward shaping for intermediate phases
- Curriculum learning (start with simpler pile configurations)
- Point cloud observations for better generalization

## Contact

This work is part of a Master's thesis on hierarchical RL for forestry automation.
