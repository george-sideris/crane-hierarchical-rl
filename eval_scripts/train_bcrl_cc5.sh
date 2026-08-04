#!/bin/bash
cd /workspace/crane_testbed
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs:$PYTHONPATH
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-CosSin-Raw-MR-CC5-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy_rsl_rl.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 400 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 2>&1 | tee /workspace/crane_testbed/logs/bcrl_cc5_train.log
