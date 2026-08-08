#!/bin/bash
# Night-2 chain: v2 reference rows -> P3 take-2 (entropy 0, fixed lr).
# NOTE: unique process name; do NOT pattern-kill on generic words like "battery".
set -u
cd /home/george/IsaacLab/crane_testbed || exit 1
until ! docker exec isaac-lab-ros2 pgrep -f v2_reference_rows.sh > /dev/null 2>&1; do sleep 60; done
echo "[night2] v2 rows done $(date +%H:%M:%S) -> P3 take-2"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3b_train.log 2>&1 &'
echo "[night2] take-2 launched $(date +%H:%M:%S)"
