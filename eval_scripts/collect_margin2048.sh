#!/bin/bash
# Collect the distractor-inclusive BC dataset: widened observation crop + 2048 points.
#
# WHY (measured 2026-08-03/04):
#  * the tight crop is the action box, fitted to the rack interior, so it deletes the bed deck,
#    the rails and the pole corners BY CONSTRUCTION. Real clouds cannot exclude them, and
#    mis-targeting that structure is 22% of real cycles (67% of all failures) in the 3 baseline
#    trials -> the net has never trained on what it fails at.
#  * --crop_margin 0.5 puts them in the cloud while the expert labels stay on logs, so every
#    structure point becomes an implicit NEGATIVE. That is supervision a per-point classification
#    head (scoring_head.py) can use and a coordinate-regression head structurally cannot.
#  * --num_points 2048 because the margin grows the crop footprint 14 -> 24 m^2 AND spends
#    ~25% of the budget on deck/structure. At 1024 that leaves ~43 pts/m^2 (~12 pts per log,
#    12 cm nearest-neighbour spacing). Spacing matters doubly for the scoring head: its output IS
#    one of the input points, so point spacing is the ACTION RESOLUTION. 2048 restores ~85 pts/m^2.
#
# The dataset is deliberately a CONTROLLED VARIANT of bc_policy_aug1_v2: same 500 episodes, same
# platform-v2 profile, same clean-collection convention. Only crop_margin and num_points change.
#
# Collections are recorded CLEAN; ZED noise is a training-time augmentation (fresh draw per epoch).
# label_dig stays 0 - dig is applied at TRAIN time (--train_label_dig) so one dataset serves both
# conventions.
#
# 2026-08-04 13:10 RESTART, LEAN: the first attempt ran ~4.4 min/episode (450 eps = 31 h) because
# --dual_save added an FPS-4096 pass over the capped full scene per sample - and the FPS loop
# does a GPU sync per iteration, so 6144 synced iterations/sample vs aug1_v2's 3072. Dropped
# dual_save (the full-PCD experiment gets its own collection later) and raised NUM_ENVS 2 -> 4.
# aug1_v2 reference: 500 eps / 17.5 h at 2 envs with dual_save.

set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs

OUT=${OUT:-/workspace/crane_testbed/logs/bc_pointcloud/bc_margin05_2048}
EPISODES=${EPISODES:-500}
NUM_ENVS=${NUM_ENVS:-8}          # eval sweep held 8 envs @ 7.3 GB for 13 h with the same cameras

mkdir -p "$OUT"
/workspace/isaaclab/isaaclab.sh -p scripts/envs/train_bc_pointcloud.py \
  --collect_only \
  --gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --crop_margin 0.5 \
  --num_points 2048 \
  --save_raw_cap 16384 --raw_margin 1.0 \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
  --gripper_effort 2000 --num_logs 200 \
  --num_envs "$NUM_ENVS" --num_episodes "$EPISODES" \
  --output_dir "$OUT" \
  > "${COLLECT_LOG:-/workspace/crane_testbed/logs/collect_margin2048.log}" 2>&1

echo "collection finished rc=$? -> $OUT"
