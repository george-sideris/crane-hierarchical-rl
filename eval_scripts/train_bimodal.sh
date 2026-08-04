#!/bin/bash
cd /workspace/crane_testbed
/workspace/isaaclab/isaaclab.sh -p scripts/envs/train_bc_pointcloud.py --train_only /workspace/crane_testbed/logs/bc_pointcloud/bc_policy_aug1_v2 --zed_noise --bimodal_aug 0.5 --train_label_dig 0.25 --epochs 100 --lr 1e-4 --output_dir /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25_bimodal 2>&1 | tee /workspace/crane_testbed/logs/train_dig25_bimodal.log
