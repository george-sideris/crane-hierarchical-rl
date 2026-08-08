#!/bin/bash
# When the citable table finishes, hand the GPU to PURE RL (scoring, from scratch).
set -u
cd /home/george/IsaacLab/crane_testbed || exit 1
until ! docker exec isaac-lab-ros2 pgrep -f citable_table.sh > /dev/null 2>&1; do sleep 120; done
echo "[after] citable done $(date +%F_%H:%M:%S) -> pure RL"
docker restart isaac-lab-ros2 > /dev/null; sleep 14
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-v0 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3scratch_train.log 2>&1 &'
echo "[after] pure RL launched $(date +%H:%M:%S)"
