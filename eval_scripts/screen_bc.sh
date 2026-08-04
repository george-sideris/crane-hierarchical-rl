#!/bin/bash
cd /workspace/crane_testbed
mkdir -p logs/sim_eval
/workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt --gaze --raw_pcd --crop_to_bounds --profile_piles --headless --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 --num_envs 8 --num_episodes 10 --seed 42 --save_metrics --save_decisions --output_dir /workspace/crane_testbed/logs/sim_eval/screen_bc 2>&1 | tee /workspace/crane_testbed/logs/sim_eval/screen_bc.log
