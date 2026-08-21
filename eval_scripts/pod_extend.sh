#!/bin/bash
# Continue an existing run to a target GLOBAL iteration, then evaluate it.
#   pod_extend.sh <arm> <task> <target_iter> <freeze:yes|no> <seed>
#
# The container is capped at 55 GB and Isaac Sim's footprint grows over hours, so a
# long run is OOM-killed after roughly eight hours. Training is therefore supervised:
# if the process dies before the target, it resumes from the newest checkpoint and
# continues. Each resume starts checkpoint numbering at zero, so progress is tracked
# cumulatively in /data/extend_progress.
set -u
ARM=$1 TASK=$2 TARGET=$3 FREEZE=$4 SEED=$5
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
export PYTHONUNBUFFERED=1

while pgrep -f "rsl_rl/train[.]py" >/dev/null; do sleep 120; done

# global progress is the SUM of the highest checkpoint in each run directory:
# every resume starts its own numbering at zero
BASE=0
for R in $(ls -dtr logs/rsl_rl/*/20* 2>/dev/null); do
  hi=$(ls "$R"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  BASE=$((BASE + ${hi:-0}))
done
[ "$BASE" -eq 0 ] && { echo "no checkpoints to extend from"; exit 1; }
echo "$BASE" > /data/extend_progress

for attempt in 1 2 3 4 5; do
  DONE=$(cat /data/extend_progress)
  LEFT=$((TARGET - DONE))
  [ "$LEFT" -le 0 ] && { echo "reached target $TARGET"; break; }
  RUN=$(ls -dt logs/rsl_rl/*/20* | head -1)
  LAST=$(ls "$RUN"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  echo "attempt $attempt: global $DONE/$TARGET, resuming $RUN at model_${LAST}.pt for $LEFT more"
  FLAGS=""
  [ "$FREEZE" = yes ] && FLAGS="--freeze_encoder"
  /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py --task "$TASK" $FLAGS \
    --seed "$SEED" --num_envs 40 --max_iterations "$LEFT" --headless \
    --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
    --gripper_effort 2000 --num_logs 200 agent.num_steps_per_env=8 agent.save_interval=10 \
    agent.resume=True agent.load_run="$(basename "$RUN")" agent.load_checkpoint="model_${LAST}.pt" \
    >> logs/p3v2_train.log 2>&1
  rc=$?
  NEW=$(ls -dt logs/rsl_rl/*/20* | head -1)
  GOT=$(ls "$NEW"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
  echo $((DONE + ${GOT:-0})) > /data/extend_progress
  echo "attempt $attempt ended rc=$rc, advanced to $(cat /data/extend_progress)"
  [ "$rc" -eq 0 ] && break
  sleep 30
done

bash eval_scripts/pod_eval_own.sh "$ARM"
