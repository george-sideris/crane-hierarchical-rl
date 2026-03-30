#!/usr/bin/env bash
# ─── IROS 2026 Eval Runbook ─────────────────────────────────────────────
# Run from IsaacLab repo root: bash crane_testbed/scripts/run_eval_all.sh [GROUP]
# Groups: 1 (main table), 2 (RL ablation), 3 (sigma sweep), 4 (DR), 5 (noise), all
# Each run: --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

COMMON="--seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics"

# ── Checkpoints ──────────────────────────────────────────────────────────
CKPT_RL_POSE="crane_testbed/results/RL_Pose/2026-02-23_20-19-49/model_940.pt"
CKPT_RL_SEG="crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21/model_880.pt"
CKPT_RL_RAW="crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20/model_650.pt"
CKPT_BC_SEG="logs/bc_pointcloud/bc_20260222_225331/bc_pointcloud_policy_rsl_rl.pt"
CKPT_BC_RAW="crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy_rsl_rl.pt"
CKPT_BCRL="logs/rsl_rl/crane_pointcloud_cossin_raw_mr_asym_v0/2026-02-27_04-21-30/model_360.pt"
CKPT_BCRL_SIGMA="logs/rsl_rl/crane_pointcloud_cossin_raw_mr_asym_v0/2026-02-28_16-53-26/model_BEST.pt"

GROUP="${1:-all}"

run_header() { echo -e "\n====== $1 ======\n"; }

# ── Group 1: Main Table Baselines (5 runs) ───────────────────────────────
run_group1() {
    run_header "1a. Heuristic"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py $COMMON

    run_header "1b. RL Raw PCD (sym)"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
        --checkpoint "$CKPT_RL_RAW" $COMMON

    run_header "1c. BC Raw PCD"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint "$CKPT_BC_RAW" --raw_pcd $COMMON

    run_header "1d. BC+RL headline (model_360)"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
        --checkpoint "$CKPT_BCRL" $COMMON

    run_header "1e. BC Seg PCD (ablation)"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint "$CKPT_BC_SEG" $COMMON
}

# ── Group 2: RL Ablation (2 runs) ────────────────────────────────────────
run_group2() {
    run_header "2a. RL Pose (sym)"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-Full-CosSin-MR-v0 \
        --checkpoint "$CKPT_RL_POSE" $COMMON

    run_header "2b. RL Seg PCD (sym)"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-MR-v0 \
        --checkpoint "$CKPT_RL_SEG" $COMMON
}

# ── Group 3: Sigma Sweep (1 run) ─────────────────────────────────────────
run_group3() {
    if [ ! -f "$CKPT_BCRL_SIGMA" ]; then
        echo "WARNING: $CKPT_BCRL_SIGMA not found. Replace model_BEST.pt with actual best checkpoint."
        return 1
    fi
    run_header "3a. BC+RL sigma=0.1"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
        --checkpoint "$CKPT_BCRL_SIGMA" $COMMON
}

# ── Group 4: Domain Randomization (4 runs) ───────────────────────────────
run_group4() {
    run_header "4a. Heuristic + DR"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py $COMMON --domain_randomization

    run_header "4b. RL Raw PCD + DR"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
        --checkpoint "$CKPT_RL_RAW" $COMMON --domain_randomization

    run_header "4c. BC Raw PCD + DR"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint "$CKPT_BC_RAW" --raw_pcd $COMMON --domain_randomization

    run_header "4d. BC+RL + DR"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
        --checkpoint "$CKPT_BCRL" $COMMON --domain_randomization
}

# ── Group 5: Observation Noise (4 runs) ──────────────────────────────────
run_group5() {
    run_header "5a. Heuristic + noise"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py $COMMON --obs_noise 0.01

    run_header "5b. RL Raw PCD + noise"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
        --checkpoint "$CKPT_RL_RAW" $COMMON --obs_noise 0.01

    run_header "5c. BC Raw PCD + noise"
    ./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
        --checkpoint "$CKPT_BC_RAW" --raw_pcd $COMMON --obs_noise 0.01

    run_header "5d. BC+RL + noise"
    ./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
        --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
        --checkpoint "$CKPT_BCRL" $COMMON --obs_noise 0.01
}

# ── Dispatch ──────────────────────────────────────────────────────────────
case "$GROUP" in
    1)   run_group1 ;;
    2)   run_group2 ;;
    3)   run_group3 ;;
    4)   run_group4 ;;
    5)   run_group5 ;;
    all)
        run_group1
        run_group2
        run_group3
        run_group4
        run_group5
        ;;
    *)
        echo "Usage: $0 [1|2|3|4|5|all]"
        exit 1
        ;;
esac

echo -e "\n====== All requested evals complete ======\n"
