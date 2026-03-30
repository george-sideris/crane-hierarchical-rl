#!/usr/bin/env bash
# ─── PPO Hyperparameter Sweep for RL (Raw PCD) ─────────────────────────
#
# Vary one parameter at a time from the PureRL baseline.
# Each run trains for 1000 iterations (~2-3h on single GPU) as a screening pass.
# Promising configs can be extended to 5000 iterations later.
#
# Usage:
#   bash crane_testbed/scripts/run_ppo_sweep.sh [GROUP]
#   Groups: lr, clip, entropy, epochs, rollout, lambda, all
#
# Run from IsaacLab repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

TASK="Isaac-Crane-PointCloud-CosSin-Raw-PureRL-v0"
NUM_ENVS=64
MAX_ITER=500   # screening pass (curves separate by ~400); bump to 5000 for full runs
SEED=42

# Base command
BASE="./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task $TASK --num_envs $NUM_ENVS --seed $SEED --max_iterations $MAX_ITER --headless"

GROUP="${1:-all}"

run_header() { echo -e "\n====== $1 ======\n"; }

# ── Baseline (default PureRL config for reference) ────────────────────
run_baseline() {
    run_header "baseline (lr=3e-4, clip=0.2, ent=0.01, epochs=5, steps=16, lam=0.95)"
    $BASE --run_name sweep_baseline
}

# ── Learning Rate ─────────────────────────────────────────────────────
run_lr() {
    run_header "lr=1e-4"
    $BASE --learning_rate 1e-4 --run_name sweep_lr_1e-4

    run_header "lr=5e-4"
    $BASE --learning_rate 5e-4 --run_name sweep_lr_5e-4

    run_header "lr=1e-3"
    $BASE --learning_rate 1e-3 --run_name sweep_lr_1e-3
}

# ── Clip Range ────────────────────────────────────────────────────────
run_clip() {
    run_header "clip=0.1"
    $BASE --clip_param 0.1 --run_name sweep_clip_0.1

    run_header "clip=0.3"
    $BASE --clip_param 0.3 --run_name sweep_clip_0.3
}

# ── Entropy Coefficient ──────────────────────────────────────────────
run_entropy() {
    run_header "entropy=0.0"
    $BASE --entropy_coef 0.0 --run_name sweep_ent_0.0

    run_header "entropy=0.005"
    $BASE --entropy_coef 0.005 --run_name sweep_ent_0.005

    run_header "entropy=0.02"
    $BASE --entropy_coef 0.02 --run_name sweep_ent_0.02

    run_header "entropy=0.05"
    $BASE --entropy_coef 0.05 --run_name sweep_ent_0.05
}

# ── Learning Epochs ──────────────────────────────────────────────────
run_epochs() {
    run_header "epochs=3"
    $BASE --num_learning_epochs 3 --run_name sweep_epochs_3

    run_header "epochs=10"
    $BASE --num_learning_epochs 10 --run_name sweep_epochs_10

    run_header "epochs=15"
    $BASE --num_learning_epochs 15 --run_name sweep_epochs_15
}

# ── Rollout Length ───────────────────────────────────────────────────
run_rollout() {
    run_header "steps=8"
    $BASE --num_steps_per_env 8 --run_name sweep_steps_8

    run_header "steps=32"
    $BASE --num_steps_per_env 32 --run_name sweep_steps_32

    run_header "steps=64"
    $BASE --num_steps_per_env 64 --run_name sweep_steps_64
}

# ── GAE Lambda ───────────────────────────────────────────────────────
run_lambda() {
    run_header "lambda=0.9"
    $BASE --gae_lambda 0.9 --run_name sweep_lam_0.9

    run_header "lambda=0.99"
    $BASE --gae_lambda 0.99 --run_name sweep_lam_0.99

    run_header "lambda=1.0"
    $BASE --gae_lambda 1.0 --run_name sweep_lam_1.0
}

# ── Dispatch ─────────────────────────────────────────────────────────
case "$GROUP" in
    baseline) run_baseline ;;
    lr)       run_lr ;;
    clip)     run_clip ;;
    entropy)  run_entropy ;;
    epochs)   run_epochs ;;
    rollout)  run_rollout ;;
    lambda)   run_lambda ;;
    all)
        run_baseline
        run_lr
        run_clip
        run_entropy
        run_epochs
        run_rollout
        run_lambda
        ;;
    *)
        echo "Usage: $0 [baseline|lr|clip|entropy|epochs|rollout|lambda|all]"
        exit 1
        ;;
esac

echo -e "\n====== Sweep complete ======\n"
echo "Compare runs in TensorBoard:"
echo "  tensorboard --logdir logs/rsl_rl/crane_pointcloud_cossin_raw_purerl_v0/"
