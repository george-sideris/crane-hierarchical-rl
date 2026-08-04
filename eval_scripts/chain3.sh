#!/bin/bash
# CRANE-DAY CHAIN (2026-08-04 evening). Priority: trained BC policies on disk by morning.
#   1. margin/2048 collection, 450 eps (~15.75 h -> ~03:30)
#   2. train P2: scoring head on the new data          (~45 min)
#   3. train P1: regression BC control on the new data (~1 h)
#   4. offline real-cloud validation of P2             (minutes)
#   5. bcrl2 sweep rows (sim-only, deferred to last)
# 450 eps not 500: buys ~1.75 h of morning buffer; collection checkpoints every 20 eps, so a
# partial run is still trainable (kill collection, run steps 2-3 by hand on what exists).
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs

echo "[chain3] collection start $(date +%F_%H:%M:%S)"
EPISODES=450 bash eval_scripts/collect_margin2048.sh
echo "[chain3] collection rc=$? $(date +%F_%H:%M:%S) -> train P2 (scoring)"

/workspace/isaaclab/_isaac_sim/python.sh scripts/envs/train_scoring_head.py \
  --dataset logs/bc_pointcloud/bc_margin05_2048 \
  --output_dir logs/bc_pointcloud/scoring_margin05_2048 \
  --epochs 100 --zed_noise --dig 0.25 --shift_aug_y 0.5 --shift_aug_x 0.15 \
  --device cuda:0 > logs/train_scoring_margin.log 2>&1
echo "[chain3] P2 rc=$? $(date +%F_%H:%M:%S) -> train P1 (regression control)"

/workspace/isaaclab/isaaclab.sh -p scripts/envs/train_bc_pointcloud.py \
  --train_only /workspace/crane_testbed/logs/bc_pointcloud/bc_margin05_2048 \
  --zed_noise --train_label_dig 0.25 --num_points 2048 --epochs 100 \
  --output_dir /workspace/crane_testbed/logs/bc_pointcloud/bc_margin05_2048_reg \
  > logs/train_reg_margin.log 2>&1
echo "[chain3] P1 rc=$? $(date +%F_%H:%M:%S) -> offline validation"

/workspace/isaaclab/_isaac_sim/python.sh scripts/envs/validate_margin_policy.py \
  --scoring logs/bc_pointcloud/scoring_margin05_2048/scoring_policy.pt \
  > logs/validate_margin.log 2>&1
echo "[chain3] validation rc=$? -> bcrl2 rows"

bash eval_scripts/bcrl_v2_sweep.sh
echo "[chain3] all done $(date +%F_%H:%M:%S)"
