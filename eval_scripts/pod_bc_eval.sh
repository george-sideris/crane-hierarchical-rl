#!/bin/bash
# Deep 100-episode eval of ONE behavior-cloned checkpoint (no training stage).
#   pod_bc_eval.sh <tag> <checkpoint-path-relative-to-crane_testbed> <scoring|bc>
set -u
TAG=$1 CKPT=$2 KIND=$3
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
export PYTHONUNBUFFERED=1   # block-buffered stdout made live logs lag by ~15 min
OUT=/data/crane_testbed/logs/sim_eval/battery/deep_${TAG}
[ -f "$OUT/.done" ] && { touch /data/CHAIN_DONE; exit 0; }
mkdir -p "$OUT"
EXTRA="--policy_type scoring --crop_margin 0.5"
[ "$KIND" = bc ] && EXTRA="--policy_type bc"
/workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
  $EXTRA --checkpoint "/data/crane_testbed/$CKPT" \
  --gaze --raw_pcd --crop_to_bounds --headless \
  --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 \
  --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 \
  --num_envs 20 --num_episodes 100 --seed 42 --save_metrics --save_decisions \
  --output_dir "$OUT" > "$OUT.log" 2>&1 && touch "$OUT/.done"
touch /data/CHAIN_DONE
