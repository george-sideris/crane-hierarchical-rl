#!/bin/bash
# Continue an existing run to a target iteration, then evaluate it.
#   pod_extend.sh <arm> <task> <target_iter> <freeze:yes|no> <seed>
# Waits for any training already running on this pod, resumes from the newest
# checkpoint (RSL-RL takes max_iterations RELATIVE to the resume point), then
# runs the wave grid and the deep row for the arm.
set -u
ARM=$1 TASK=$2 TARGET=$3 FREEZE=$4 SEED=$5
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
export PYTHONUNBUFFERED=1

while pgrep -f "rsl_rl/train[.]py" >/dev/null; do sleep 120; done   # let the current run finish

RUN=$(ls -dt logs/rsl_rl/*/20* 2>/dev/null | head -1)
[ -z "$RUN" ] && { echo "no run dir to extend"; exit 1; }
LAST=$(ls "$RUN"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
[ -z "$LAST" ] && { echo "no checkpoint in $RUN"; exit 1; }
LEFT=$((TARGET - LAST))
echo "extending $ARM: $RUN at iter $LAST -> $TARGET ($LEFT more)"
if [ "$LEFT" -gt 0 ]; then
  FLAGS=""
  [ "$FREEZE" = yes ] && FLAGS="--freeze_encoder"
  /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task "$TASK" $FLAGS \
    --seed "$SEED" --num_envs 40 --max_iterations "$LEFT" --headless \
    --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
    --gripper_effort 2000 --num_logs 200 agent.num_steps_per_env=8 agent.save_interval=10 \
    agent.resume=True agent.load_run="$(basename "$RUN")" agent.load_checkpoint="model_${LAST}.pt" \
    >> logs/p3v2_train.log 2>&1
  echo "extend exit $?"
fi
bash eval_scripts/pod_eval_own.sh "$ARM"
