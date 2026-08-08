#!/bin/bash
# after the P3 battery: v2 reference rows, THEN P3 take-2 training.
set -u
until ! pgrep -f p3_eval_battery.sh > /dev/null; do sleep 60; done
echo "[chain] battery done $(date +%H:%M:%S) -> v2 reference rows"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && bash eval_scripts/v2_reference_rows.sh' > /home/george/IsaacLab/crane_testbed/logs/v2_reference.log 2>&1
echo "[chain] v2 rows done $(date +%H:%M:%S) -> P3 take-2"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3b_train.log 2>&1 &'
echo "[chain] take-2 launched $(date +%H:%M:%S)"
