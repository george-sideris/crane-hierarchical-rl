#!/bin/bash
# After chain3 (collection -> P2 -> P1 -> validation -> bcrl2):
#   1. FULL-PCD scoring head on data we ALREADY HAVE (aug1_v2's pointclouds_full.npy:
#      11,292 uncropped full-scene clouds @2048 pts, same actions, platform-v2). The
#      "no crop, no rack calibration" experiment - no new collection needed.
#      label_radius 0.25 (full-scene FPS is sparser near the label than the tight crop).
#   2. Offline check of it on the raw bag clouds with effectively NO crop (margin 25).
#   3. Gated-heuristic parity rows for the noise sweep.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
echo "[chain4] waiting for chain3 ..."
until ! pgrep -f "chain3.sh" > /dev/null; do sleep 120; done

echo "[chain4] chain3 done $(date +%F_%H:%M:%S) -> full-PCD scoring head"
mkdir -p logs/bc_pointcloud/scoring_full_aug1v2
mkdir -p /tmp/full_ds && ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/pointclouds_full.npy /tmp/full_ds/pointclouds.npy \
  && ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/actions.npy /tmp/full_ds/actions.npy \
  && ln -sf /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2/cam_pos_base.npy /tmp/full_ds/cam_pos_base.npy
/workspace/isaaclab/_isaac_sim/python.sh scripts/envs/train_scoring_head.py \
  --dataset /tmp/full_ds \
  --output_dir logs/bc_pointcloud/scoring_full_aug1v2 \
  --epochs 100 --zed_noise --dig 0.25 --label_radius 0.25 \
  --shift_aug_y 0.5 --shift_aug_x 0.15 \
  --device cuda:0 > logs/train_scoring_full.log 2>&1
echo "[chain4] full-PCD train rc=$? -> offline no-crop check"
/workspace/isaaclab/_isaac_sim/python.sh scripts/envs/validate_margin_policy.py \
  --scoring logs/bc_pointcloud/scoring_full_aug1v2/scoring_policy.pt \
  --margin 25 > logs/validate_full.log 2>&1
echo "[chain4] validation rc=$? -> gated heuristic rows"
bash eval_scripts/gated_heuristic_sweep.sh
echo "[chain4] done rc=$? $(date +%F_%H:%M:%S)"
