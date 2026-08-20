#!/bin/bash
# Continuously mirror training evidence home from every running pod:
# TensorBoard event files, params/, the training log, and battery eval jsons.
# One pass per invocation; loop it from a watchdog or a while loop.
# Destination: logs/fleet_tb_live/<arm>/ - never deleted, only overwritten with newer.
set -u
cd "$(dirname "$0")/.." || exit 1
DEST=logs/fleet_tb_live
mkdir -p "$DEST"

# port host arm  (training pods: rsl_rl run dirs; battery pods: eval jsons)
TRAIN_PODS="
22159 69.30.85.211 S43_rerun
22005 69.30.85.193 SCC43_rerun
22149 63.141.33.5 SYMCRITIC_rerun
22128 63.141.33.10 UNFROZ_rerun
22020 194.68.245.149 GS_rerun
"
BATTERY_PODS="
22123 69.30.85.67 b1
22124 69.30.85.67 b2
"

while read -r port host arm; do
  [ -z "$port" ] && continue
  mkdir -p "$DEST/$arm"
  timeout 120 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" \
    'cd /data/crane_testbed && tar czf - logs/rsl_rl/*/2026-*/events.out.tfevents.* logs/rsl_rl/*/2026-*/params logs/p3v2_train.log 2>/dev/null' \
    | tar xzf - -C "$DEST/$arm" 2>/dev/null \
    && echo "synced $arm: $(find $DEST/$arm -name 'events.*' | wc -l) event files"
done <<< "$TRAIN_PODS"

while read -r port host arm; do
  [ -z "$port" ] && continue
  mkdir -p "$DEST/$arm"
  timeout 120 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" \
    'cd /data/crane_testbed/logs/sim_eval && tar czf - battery/*/eval_metrics_*.json battery/*/.done 2>/dev/null' \
    | tar xzf - -C "$DEST/$arm" 2>/dev/null \
    && echo "synced $arm: $(find $DEST/$arm -name 'eval_metrics_*.json' | wc -l) rows"
done <<< "$BATTERY_PODS"
