#!/bin/bash
# Drive one arm to its target and then evaluate it, entirely from the pod.
# The laptop that launched the run may sleep or disconnect, so nothing here depends on an
# external supervisor. It only ever LAUNCHES work, and only when the pod is idle, so it is
# safe to start alongside a chain that is already running.
#   pod_selfdrive.sh <arm> <task> <target_iters> <freeze:yes|no> <seed>
set -u
ARM=$1 TASK=$2 TARGET=$3 FREEZE=$4 SEED=$5
SAVE_EVERY=${SAVE_EVERY:-10}
cd /data/crane_testbed || exit 1
LOG=/data/selfdrive.log
echo "$(date +%F\ %H:%M) selfdrive up: $ARM -> $TARGET iters" >> "$LOG"
for i in $(seq 1 300); do
  busy=$(ps -eo cmd | grep -cE "[r]sl_rl/train\.py|[p]od_extend|[p]lay_bc_pointcloud\.py|[t]hesis_battery")
  if [ "$busy" -eq 0 ]; then
    g=0
    for r in logs/rsl_rl/*/20*; do
      h=$(ls "$r"/model_*.pt 2>/dev/null | sed 's/.*model_//;s/\.pt//' | sort -n | tail -1)
      g=$((g + ${h:-0}))
    done
    # Progress is read from the highest checkpoint number, and checkpoints are written
    # every SAVE_EVERY iterations, so a run shorter than that cannot move the counter. Asking
    # for fewer than SAVE_EVERY more iterations therefore relaunches forever: the run does the
    # work, saves nothing new, and the next tick asks for it again. Treat that as done.
    if [ $((TARGET - g)) -ge "$SAVE_EVERY" ]; then
      echo "$(date +%H:%M) global $g/$TARGET, (re)starting extension" >> "$LOG"
      rm -f /data/CHAIN_DONE /data/extend_progress
      setsid nohup bash eval_scripts/pod_extend.sh "$ARM" "$TASK" "$TARGET" "$FREEZE" "$SEED" \
        > /data/extend.log 2>&1 < /dev/null &
    elif [ ! -f /data/CHAIN_DONE ]; then
      echo "$(date +%H:%M) within $SAVE_EVERY of target ($g/$TARGET), starting evaluation" >> "$LOG"
      setsid nohup bash eval_scripts/pod_eval_own.sh "$ARM" > /data/chain.log 2>&1 < /dev/null &
    else
      echo "$(date +%H:%M) arm complete and evaluated" >> "$LOG"
      break
    fi
  fi
  sleep 600
done
