#!/bin/bash
# Scoring-head rows for the observation-degradation sweep, using the IDENTICAL protocol as
# noise_sweep.sh so they drop straight into the same table.
#
# scoring_v1 vs bc_aug1v2_dig25 is a controlled head comparison - same dataset
# (bc_policy_aug1_v2), same dig 0.25, same --zed_noise augmentation, same 1024 points, 100 epochs.
# The ONLY difference is per-point argmax classification vs coordinate regression.
#
# WHAT TO EXPECT. The scoring head's advantage is mode-seeking: it cannot answer "the mean of two
# mounds", and its target is always on observed material (offline on 26 real clouds: 0% zero-support
# vs the deployed regression BC's 27%). On SIM piles that failure is rare and self-correcting - sim
# BC already clears ~99% - so a large sim win is NOT the expectation here. What this measures is
# whether the argmax constraint COSTS anything in the regime where regression is already fine, and
# how the two heads' slopes differ as the cloud degrades. A wash on the clean row with a shallower
# slope under noise would still be the result that matters.
#
# CAVEAT: scoring_v1 was trained on the TIGHT-crop dataset, so it has never seen the bed deck,
# rails or poles. The argument that motivated the architecture (structure as implicit negatives)
# is NOT tested by these rows - that needs the bc_margin05_2048 retrain.

set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs

OUT=/workspace/crane_testbed/logs/sim_eval/noise_sweep
mkdir -p "$OUT"
ISAAC=/workspace/isaaclab/isaaclab.sh
SCORING=/workspace/crane_testbed/logs/bc_pointcloud/scoring_v1/scoring_policy.pt

COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 \
  --num_logs 200 --num_envs 8 --num_episodes 10 --seed 42 --save_metrics --save_decisions"

run () {
  local tag=$1; shift
  if [ -f "$OUT/$tag/.done" ]; then echo "[skip] $tag"; return; fi
  echo "=== $(date +%H:%M:%S) START $tag"
  # shellcheck disable=SC2086
  $ISAAC -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
      --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done";
  else echo "  !! FAILED rc=$rc -> $OUT/$tag.log"; fi
  tail -4 "$OUT/$tag.log"
  echo "=== $(date +%H:%M:%S) END $tag"
}

for LEVEL in clean 1x 3x 8x; do
  case $LEVEL in
    clean) N="" ;;
    1x)    N="--zed_noise --zed_axial_coeff 0.0014" ;;
    3x)    N="--zed_noise --zed_axial_coeff 0.0042" ;;
    8x)    N="--zed_noise --zed_axial_coeff 0.0112" ;;
  esac
  # shellcheck disable=SC2086
  run "scoring_${LEVEL}" --policy_type scoring --checkpoint "$SCORING" $N
done
echo "=== scoring rows complete $(date +%H:%M:%S)"
