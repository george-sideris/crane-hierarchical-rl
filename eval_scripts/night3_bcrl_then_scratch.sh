#!/bin/bash
# Night-3: give BCRL-scoring (take-2) a fair window, then hand the GPU to PURE RL scoring.
# Unique name; never pattern-kill on shared words.
set -u
cd /home/george/IsaacLab/crane_testbed || exit 1
LOG=logs/p3b_train.log
# --- phase 1: watch take-2 until it has enough iterations to judge, or it dies
DEADLINE=$(( $(date +%s) + 5*3600 ))     # hard cap 5 h
while true; do
  sleep 300
  it=$(grep -oE "Learning iteration ([0-9]+)/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
  alive=$(docker exec isaac-lab-ros2 pgrep -cf rsl_rl/train.py 2>/dev/null || echo 0)
  now=$(date +%s)
  if [ "$alive" -eq 0 ]; then echo "[n3] take-2 exited at iter $it"; break; fi
  if [ "${it:-0}" -ge 150 ]; then echo "[n3] take-2 reached iter $it - enough to judge"; break; fi
  if [ "$now" -ge "$DEADLINE" ]; then echo "[n3] take-2 time cap at iter $it"; break; fi
done
docker exec isaac-lab-ros2 bash -c 'pkill -f rsl_rl/train.py' 2>/dev/null
sleep 5
echo "[n3] handing GPU to PURE RL scoring $(date +%H:%M:%S)"
docker restart isaac-lab-ros2 > /dev/null; sleep 12
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-v0 --seed 42 --num_envs 4 --max_iterations 100000 --headless --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 > logs/p3scratch_train.log 2>&1 &'
echo "[n3] pure RL launched $(date +%H:%M:%S)"
