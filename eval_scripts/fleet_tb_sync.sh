#!/bin/bash
# Mirror every pod's evidence home: TensorBoard events, params, training log, eval jsons
# and decision dumps. Reads the live pod list, so a newly registered pod is covered at once.
# Destination: logs/fleet_tb_live/<pod>/ - never deleted, only overwritten with newer.
set -u
cd "$(dirname "$0")/.." || exit 1
DEST=logs/fleet_tb_live
mkdir -p "$DEST"
while read -r port host name; do
  [ -z "$port" ] && continue
  mkdir -p "$DEST/$name"
  timeout 180 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" \
    'cd /data/crane_testbed && tar czf - \
       logs/rsl_rl/*/2026-*/events.out.tfevents.* \
       logs/rsl_rl/*/2026-*/params \
       logs/p3v2_train.log \
       logs/sim_eval/battery/*/eval_metrics_*.json \
       logs/sim_eval/battery/*/decisions_*.npz \
       logs/sim_eval/battery/*/.done 2>/dev/null' \
    | tar xzf - -C "$DEST/$name" 2>/dev/null
  ev=$(find "$DEST/$name" -name 'events.out.tfevents.*' 2>/dev/null | wc -l)
  js=$(find "$DEST/$name" -name 'eval_metrics_*.json' 2>/dev/null | wc -l)
  echo "$name: $ev event files, $js eval rows"
done < logs/fleet/pods.txt
