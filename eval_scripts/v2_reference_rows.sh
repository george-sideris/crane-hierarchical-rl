#!/bin/bash
# v2-METRICS REFERENCE ROWS - the citable comparison set, all through ONE harness
# (play_bc_pointcloud) so clearing/c95/throughput/full-clear are the MEASURED definitions.
# Previous reference numbers were 10-ep v1 screens; the expert row additionally came from the
# COLLECTION script, which never had v2 metrics at all - hence this re-run.
#
# Rows: privileged expert (dig 0.25) | deployed heuristic bridge | regression BC (tight/1024).
# P2c already has its v2 row from the P3 battery. Same seed/env count everywhere so pile
# sequences pair across rows.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/v2_reference
mkdir -p "$OUT"
COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --num_envs 4 --num_episodes 10 --seed 42 --save_metrics --save_decisions \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
  --gripper_effort 2000 --num_logs 200"
run () {
  local tag=$1; shift
  [ -f "$OUT/$tag/.done" ] && { echo "[skip] $tag"; return; }
  echo "=== $(date +%H:%M:%S) $tag"
  # shellcheck disable=SC2086
  /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
      --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  [ $? -eq 0 ] && { mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done"; } || echo "  FAILED -> $OUT/$tag.log"
  tail -3 "$OUT/$tag.log"
}
run expert     --policy_type expert    --expert_dig 0.25
run heuristic  --policy_type heuristic --heuristic_dig 0.25
run bc_reg     --policy_type bc --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt
echo "=== v2 reference rows done $(date +%H:%M:%S)"
