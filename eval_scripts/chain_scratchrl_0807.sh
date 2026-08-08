#!/bin/bash
# Hand the GPU to PURE RL (scoring head, from scratch) once the chunked citable table finishes.
#
# WHY this run matters: the paper-era pure-RL result (~56% clearing) used a Gaussian head over
# absolute coordinates, so PPO had to learn perception AND metre regression at once. The
# categorical head removes the regression half - every action is "pick one of the observed
# points" - so the question is whether RL from scratch becomes viable at all. Either answer is
# a thesis result; a failure here is the control that justifies the BC->RL pipeline.
#
# Unique script name (chain_scratchrl_0807) so a later `pkill -f <name>` cannot match a
# different chain by shared substring - that mistake cost 7 GPU-hours on 2026-08-05.
set -u
cd /home/george/IsaacLab/crane_testbed || exit 1
LOG=logs/chain_scratchrl_0807.log
exec >> "$LOG" 2>&1
echo "=== chain armed $(date +%F_%H:%M:%S)"

# 1. wait for the eval driver to exit (it runs inside the container as citable_chunked.sh)
while docker exec isaac-lab-ros2 pgrep -f citable_chunked.sh > /dev/null 2>&1; do sleep 120; done
echo "=== eval chain exited $(date +%F_%H:%M:%S)"

# 2. if the eval aborted because the CUDA context died, do NOT start a 10-hour training run
#    into a dead GPU - leave it for the human, who has to rmmod/modprobe on the host anyway.
if [ -f logs/sim_eval/citable_chunk/.gpu_dead ]; then
  echo "!! .gpu_dead present - eval aborted on a lost CUDA context. NOT launching RL."
  echo "!! Host fix:  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
  exit 90
fi

# 3. fresh container: a long training run should not inherit whatever state 6 h of eval left
docker restart isaac-lab-ros2 > /dev/null; sleep 20

if ! docker exec isaac-lab-ros2 /workspace/isaaclab/_isaac_sim/python.sh \
     -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > /dev/null 2>&1; then
  echo "!! CUDA unavailable after restart - NOT launching RL. Host rmmod/modprobe needed."
  exit 90
fi
echo "=== CUDA ok $(date +%H:%M:%S)"

# 3b. Qualitative paired episode BEFORE the long RL run, because pure RL holds the GPU for
#     many hours and these four short episodes would otherwise never get a slot. ~90 min.
echo "=== paired videos start $(date +%H:%M:%S)"
docker exec isaac-lab-ros2 bash /workspace/crane_testbed/eval_scripts/paired_videos.sh \
  >> logs/paired_videos.log 2>&1
echo "=== paired videos done $(date +%H:%M:%S)"

# a fresh container again: the video run opens render pipelines the training run does not need
docker restart isaac-lab-ros2 > /dev/null; sleep 20
if ! docker exec isaac-lab-ros2 /workspace/isaaclab/_isaac_sim/python.sh \
     -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" > /dev/null 2>&1; then
  echo "!! CUDA unavailable after video run - NOT launching RL."
  exit 90
fi
echo "=== launching pure RL $(date +%H:%M:%S)"

# 4. same env knobs as every other run on this platform (platform-v2 frozen flags), so the
#    scratch row is comparable to the BC and BC->RL rows without re-deriving the setup.
docker exec -d isaac-lab-ros2 bash -c 'cd /workspace/crane_testbed && \
  export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
  env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-v0 \
    --seed 42 --num_envs 4 --max_iterations 100000 --headless \
    --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 \
    --log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200 \
    > logs/p3scratch_train.log 2>&1 &'

sleep 120
if docker exec isaac-lab-ros2 pgrep -f "Scoring-Scratch" > /dev/null 2>&1; then
  echo "=== pure RL running $(date +%H:%M:%S)"
else
  echo "!! pure RL did not survive startup - see logs/p3scratch_train.log"
fi
