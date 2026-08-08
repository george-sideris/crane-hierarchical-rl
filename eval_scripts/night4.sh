#!/bin/bash
# Night-4: let take-2 write model_50, stop it, run the citable 96-ep table, then pure RL.
set -u
cd /home/george/IsaacLab/crane_testbed || exit 1
CK=logs/rsl_rl/crane_pointcloud_gaze_scoring_ppo_v0/2026-08-06_21-07-46
for i in $(seq 1 40); do [ -f "$CK/model_50.pt" ] && break; sleep 30; done
echo "[n4] take-2 checkpoint: $(ls $CK/model_*.pt | tr '\n' ' ')"
docker exec isaac-lab-ros2 bash -c 'pkill -f rsl_rl/train.py' 2>/dev/null; sleep 5
docker restart isaac-lab-ros2 > /dev/null; sleep 12
echo "[n4] citable table start $(date +%H:%M:%S)"
docker exec isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && bash eval_scripts/citable_table.sh' \
  > logs/citable_table.log 2>&1
echo "[n4] citable table done $(date +%H:%M:%S) -> pure RL"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-v0 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3scratch_train.log 2>&1 &'
echo "[n4] pure RL launched $(date +%H:%M:%S)"
