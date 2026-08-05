#!/bin/bash
# Post-collection sequence, v2 (2026-08-04 ~23:40). Replaces chain3's tail + chain4:
#   1. wait for the RUNNING 450-ep collection python to finish (do not touch it)
#   2. top-up +50 episodes into bc_margin05_2048_b (time-seeded -> unique piles)
#   3. merge -> bc_margin05_2048_500
#   4. P2 scoring train on the FULL 500        (watcher fires the CPU gate on completion)
#   5. gate (explicit, belt+braces with the watcher)
#   6. P1 regression control on the 500
#   7. full-PCD scoring head (aug1_v2 full clouds) + its no-crop check
# Deferred to the other PC / tomorrow evening: bcrl2 rows, gated-heuristic rows.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
ISAAC_PY=/workspace/isaaclab/_isaac_sim/python.sh

echo "[chain6] waiting for the 450-ep collection to finish ..."
until ! pgrep -f "python3 scripts/envs/train_bc_pointcloud" > /dev/null; do sleep 60; done
echo "[chain6] collection done $(date +%F_%H:%M:%S) -> top-up 50"

OUT=/workspace/crane_testbed/logs/bc_pointcloud/bc_margin05_2048_b \
COLLECT_LOG=/workspace/crane_testbed/logs/collect_margin2048_b.log \
EPISODES=50 NUM_ENVS=8 bash eval_scripts/collect_margin2048.sh
echo "[chain6] top-up rc=$? $(date +%F_%H:%M:%S) -> merge"

$ISAAC_PY scripts/envs/merge_bc_datasets.py \
  --a logs/bc_pointcloud/bc_margin05_2048 \
  --b logs/bc_pointcloud/bc_margin05_2048_b \
  --out logs/bc_pointcloud/bc_margin05_2048_500 > logs/merge_500.log 2>&1
MERGE_RC=$?
DATASET=logs/bc_pointcloud/bc_margin05_2048_500
if [ $MERGE_RC -ne 0 ]; then
  echo "[chain6] MERGE FAILED rc=$MERGE_RC - training on the 450 alone"
  DATASET=logs/bc_pointcloud/bc_margin05_2048
fi

echo "[chain6] P2 on $DATASET"
$ISAAC_PY scripts/envs/train_scoring_head.py \
  --dataset "$DATASET" \
  --output_dir logs/bc_pointcloud/scoring_margin05_2048 \
  --epochs 100 --zed_noise --dig 0.25 --shift_aug_y 0.5 --shift_aug_x 0.15 \
  --device cuda:0 > logs/train_scoring_margin.log 2>&1
echo "[chain6] P2 rc=$? -> gate"

$ISAAC_PY scripts/envs/validate_margin_policy.py \
  --scoring logs/bc_pointcloud/scoring_margin05_2048/scoring_policy.pt \
  --margin 0.5 > logs/validate_margin.log 2>&1
echo "[chain6] gate rc=$? -> P1"

/workspace/isaaclab/isaaclab.sh -p scripts/envs/train_bc_pointcloud.py \
  --train_only "/workspace/crane_testbed/$DATASET" \
  --zed_noise --train_label_dig 0.25 --num_points 2048 --epochs 100 \
  --output_dir /workspace/crane_testbed/logs/bc_pointcloud/bc_margin05_2048_reg \
  > logs/train_reg_margin.log 2>&1
echo "[chain6] P1 rc=$? -> full-PCD head"

mkdir -p /tmp/full_ds logs/bc_pointcloud/scoring_full_aug1v2
ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/pointclouds_full.npy /tmp/full_ds/pointclouds.npy
ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/actions.npy /tmp/full_ds/actions.npy
ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/cam_pos_base.npy /tmp/full_ds/cam_pos_base.npy
$ISAAC_PY scripts/envs/train_scoring_head.py \
  --dataset /tmp/full_ds \
  --output_dir logs/bc_pointcloud/scoring_full_aug1v2 \
  --epochs 100 --zed_noise --dig 0.25 --label_radius 0.25 \
  --shift_aug_y 0.5 --shift_aug_x 0.15 \
  --device cuda:0 > logs/train_scoring_full.log 2>&1
echo "[chain6] full-PCD rc=$? -> no-crop check"
$ISAAC_PY scripts/envs/validate_margin_policy.py \
  --scoring logs/bc_pointcloud/scoring_full_aug1v2/scoring_policy.pt \
  --margin 25 > logs/validate_full.log 2>&1
echo "[chain6] all done $(date +%F_%H:%M:%S)"
