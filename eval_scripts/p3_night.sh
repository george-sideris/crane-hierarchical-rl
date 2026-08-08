#!/bin/bash
# P3 night chain: wait for any running train.py to exit -> clean container -> 4D smoke ->
# on success, the REAL run (400 iters, 4 envs, P2c init, frozen encoder, asym critic).
set -u
until ! pgrep -f "rsl_rl/train.py" > /dev/null; do sleep 30; done
echo "[p3] smoke3 done $(date +%H:%M:%S); container restart + 4D smoke"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 2 --max_iterations 3 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200' > /workspace/crane_testbed/logs/p3_smoke4.log 2>&1
if grep -q "Learning iteration" /workspace/crane_testbed/logs/p3_smoke4.log && ! grep -q "Traceback" /workspace/crane_testbed/logs/p3_smoke4.log; then
  echo "[p3] SMOKE PASS $(date +%H:%M:%S) -> launching the real run"
  docker restart isaac-lab-ros2 > /dev/null; sleep 12
  docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v0 --bc_checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 400 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3_train.log 2>&1 &'
  echo "[p3] real run launched"
else
  echo "[p3] SMOKE FAILED - real run NOT launched; see logs/p3_smoke4.log"
fi
