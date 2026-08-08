#!/bin/bash
# Line-2 RL eval battery, v2 metrics, 24 episodes: P2c (control) vs P3 model_50 (plateau-edge)
# vs model_100 (post-drift). model_0 is P2c's weights round-tripped through the converter, so
# P2c IS the model_0 row. All rows: margin 0.5, 2048 pts, gate off (deployment default).
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/p3_battery
mkdir -p "$OUT"
COMMON="--policy_type scoring --gaze --raw_pcd --crop_to_bounds --crop_margin 0.5 --profile_piles \
  --headless --num_envs 4 --num_episodes 10 --seed 42 --save_metrics --save_decisions \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200"
run () {
  local tag=$1 ckpt=$2
  [ -f "$OUT/$tag/.done" ] && { echo "[skip] $tag"; return; }
  echo "=== $(date +%H:%M:%S) $tag"
  # shellcheck disable=SC2086
  /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py $COMMON \
    --checkpoint "$ckpt" --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  [ $? -eq 0 ] && { mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done"; } || echo "  FAILED -> $OUT/$tag.log"
  tail -3 "$OUT/$tag.log"
}
run p2c      /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
run p3_m50   /workspace/crane_testbed/logs/bc_pointcloud/p3_model_50/scoring_policy.pt
run p3_m100  /workspace/crane_testbed/logs/bc_pointcloud/p3_model_100/scoring_policy.pt
echo "=== battery done $(date +%H:%M:%S)"
