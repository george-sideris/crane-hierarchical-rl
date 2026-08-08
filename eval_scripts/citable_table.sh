#!/bin/bash
# CITABLE MAIN TABLE - 96 episodes (12 exact rounds at 8 envs), v2 metrics, one harness.
# Rows: privileged expert | deployed heuristic @1024 (its deployed config) | heuristic @2048
# (DENSITY CONTROL - the scoring row runs 2048 pts, and the heuristic is parameter-free in
# density, so a sparser cloud would be an unearned handicap) | regression BC @1024 (locked by
# its training) | P2c @2048 margin.
# Per-row VRAM guard: 8 envs, drop to 6 and retry once if the row dies early.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/citable
mkdir -p "$OUT"

run () {   # run <tag> <episodes> <envs> <args...>
  local tag=$1 eps=$2 envs=$3; shift 3
  [ -f "$OUT/$tag/.done" ] && { echo "[skip] $tag"; return; }
  echo "=== $(date +%H:%M:%S) $tag (${eps}ep, ${envs}env)"
  local COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
    --num_envs $envs --num_episodes $eps --seed 42 --save_metrics --save_decisions \
    --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
    --gripper_effort 2000 --num_logs 200"
  # shellcheck disable=SC2086
  /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
      --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  if [ $? -ne 0 ] && [ "$envs" -gt 6 ]; then
    echo "  !! $tag failed at $envs envs - retrying at 6"
    # shellcheck disable=SC2086
    COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
      --num_envs 6 --num_episodes $eps --seed 42 --save_metrics --save_decisions \
      --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
      --gripper_effort 2000 --num_logs 200"
    /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
        --output_dir "$OUT/$tag" >> "$OUT/$tag.log" 2>&1
  fi
  [ $? -eq 0 ] && { mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done"; echo "  done $(date +%H:%M:%S)"; } \
                || echo "  FAILED -> $OUT/$tag.log"
}

run expert       100 8 --policy_type expert    --expert_dig 0.25
run heuristic    100 8 --policy_type heuristic --heuristic_dig 0.25
run heuristic2048 100 8 --policy_type heuristic --heuristic_dig 0.25 --num_points 2048
run bc_reg       100 8 --policy_type bc --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt
run p2c          100 8 --policy_type scoring --crop_margin 0.5 --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
echo "=== citable table done $(date +%H:%M:%S)"
