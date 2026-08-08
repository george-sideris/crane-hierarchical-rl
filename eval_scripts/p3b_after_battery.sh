#!/bin/bash
# Wait for the eval battery, then launch P3-take-2 (entropy 0, fixed lr 5e-5, P2c init, uncapped).
set -u
until ! pgrep -f p3_eval_battery.sh > /dev/null; do sleep 60; done
sleep 5
docker() { command docker "$@"; }
echo "[p3b] battery done $(date +%H:%M:%S) - launching take-2"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3b_train.log 2>&1 &'
echo "[p3b] launched"
