# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

IROS 2026 submission: hierarchical RL for forestry crane log-pile clearing using NVIDIA Isaac Lab. A high-level RL policy selects grasp targets (5D), and a fixed 10-phase FSM controller executes each pick-place cycle (~900 physics steps). Each policy step = one complete grasp cycle, reducing episodes from ~27,000 to 30-50 steps.

Four methods are compared: heuristic expert, pure RL (PPO), behavioral cloning (BC), and BC+RL fine-tuning (headline result).

## Running Commands

All commands run inside the Isaac Lab Docker container. From the IsaacLab root (`/workspace/isaaclab`):

```bash
# Set PYTHONPATH first
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH

# RL Training (pure)
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-CosSin-MR-v0 --num_envs 4

# BC+RL Training (headline experiment)
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-CosSin-MR-v0 --num_envs 20 --headless \
    --bc_checkpoint <path_to_bc_policy>

# BC Training
./isaaclab.sh -p crane_testbed/scripts/envs/train_bc_pointcloud_cossin.py \
    --num_envs 8 --num_episodes 200 --epochs 100

# Evaluation (all support --save_metrics for JSON output)
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
    --task Isaac-Crane-PointCloud-CosSin-MR-v0 --num_envs 20 --headless --save_metrics
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
    --checkpoint <path> --num_envs 20 --headless --save_metrics
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
    --num_envs 20 --num_episodes 50 --headless --save_metrics

# TensorBoard
tensorboard --logdir /workspace/logs/rsl_rl/crane_hierarchical
```

There is no test suite, linter, or build step. The project is a pure Python research codebase running inside Isaac Lab.

## Architecture

### Grasp-and-Remove MDP

- **Observation**: Point cloud (1024×3, primary) or privileged per-object poses (128D, ablation)
- **Action**: 5D `[x, y, z, cos(2ψ), sin(2ψ)]` in crane base frame. Decode: `xyz = min + (tanh(a)+1)/2 * (max-min)`, `yaw = atan2(tanh(a4), tanh(a3)) / 2`
- **Reward**: Multiplicative `throughput × alignment × stability × 10.0`. Penalties for failures, empty targets, knocked-off logs. `r_clear` (completion bonus) is defined but NOT used.
- **Episode**: Ends at 30 cycles or empty rack

### Key Task IDs

| Task ID | Obs | Notes |
|---------|-----|-------|
| `Isaac-Crane-PointCloud-CosSin-MR-v0` | PCD 3072D | **Primary** for paper |
| `Isaac-Crane-Full-CosSin-MR-v0` | Pose 128D | Privileged-pose ablation |

Other variants (MN, AR, AN, PG reward; DR flags) exist in `tasks.py` but MR without DR is the baseline.

### Core Files

| File | Role |
|------|------|
| `scripts/envs/crane_rl_env_full.py` | Main env (~5000 lines): FSM, reward, OOB, all simulation logic |
| `source/crane_testbed/crane_testbed/tasks.py` | Gym task registration mapping IDs to env configs |
| `source/crane_testbed/crane_testbed/agents/rsl_rl_cfg.py` | PPO hyperparameters per task variant |
| `source/crane_testbed/crane_testbed/agents/pointnet_actor_critic.py` | PointNet encoder + actor-critic |
| `source/crane_testbed/crane_testbed/agents/cnn_actor_critic.py` | CNN depth encoder + actor-critic |
| `scripts/rsl_rl/train.py` | RL training (supports `--bc_checkpoint` for BC+RL) |
| `scripts/envs/train_bc_pointcloud_cossin.py` | BC training with 5D cossin actions |
| `scripts/envs/play_heuristic.py`, `play_bc_pointcloud.py`, `scripts/rsl_rl/play.py` | Evaluation (all produce identical JSON metrics) |
| `docs/iros2026_draft.tex` | IROS paper draft |

### Point Cloud Pipeline

Depth camera → semantic segmentation mask (logs only) → back-project to 3D → transform to crane base frame → Farthest Point Sampling (1024 pts) → PointNet encoder (per-point MLP 3→64→128→256 with BN + ELU, max-pool, FC 256→256) → actor/critic heads.

### FSM Controller (10 phases)

HOVER_UP → ALIGN_YAW → DESCEND → CLOSE → LIFT_HIGH → CARRY_HOME → ALIGN_HOME_YAW → LOWER_TO_DROP → OPEN → SETTLE

### BC+RL Fine-Tuning

`train.py --bc_checkpoint <path>` loads BC encoder+actor weights (`strict=False`). Critic initializes randomly. Use `sigma_init=0.3` (not 1.0) to preserve BC mean while allowing exploration.

## Critical Constraints

- **Max 20 parallel envs** — 32+ causes PhysX `Scene state is corrupted` (6400 rigid bodies exceeds GPU solver capacity)
- **Action bounds = OOB bounds**: `_action_bounds_min/max` (from rack geometry) are used for both policy scaling and out-of-bounds log despawning
- **`_logs_knocked_off` resets on episode reset** — eval scripts must capture it in the done block before reset
- **Evaluation order**: `_check_logs_out_of_bounds()` → `_check_grasped_logs()` → `_despawn_grasped_logs()` (OOB deposited first, no double-counting)

## Detailed Context

See `docs/CLAUDE.md` for extended context including full evaluation metric formats, PPO hyperparameters, reward function details, paper status, and common pitfalls.
