#!/bin/bash
# One supervision pass over the extension fleet.
#   - relaunch an arm whose training died and is still short of the target
#   - mirror tensorboard events, params and checkpoints home
# Launching only: this script never kills a process, so a live run is never disturbed.
set -u
cd "$(dirname "$0")/.." || exit 1
# Per-arm targets in CHECKPOINT iterations. They are not equal, because the arms did
# not all collect the same rollout per iteration: the original scratch runs logged 640
# env steps per iteration and the extensions log 320, so a common iteration count would
# mean different amounts of experience. Each target below is the iteration count that
# brings that arm to the 50k env-step mark used as the common x-limit of the curves.
LOG=logs/fleet/supervise.log
CK=logs/fleet_ckpt_live
mkdir -p logs/fleet "$CK"
stamp() { date +%H:%M; }

# arm | ssh port | host | task | freeze | seed | target_iters
ARMS='
scc43train 22005 69.30.85.193 Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-CC0-v0 no 43 88
unfroz 22128 63.141.33.10 Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2-CC0 no 42 146
symcritic 22149 63.141.33.5 Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2-CC0-SymCritic yes 42 156
scc44 22124 69.30.85.67 Isaac-Crane-PointCloud-Gaze-Scoring-Scratch-CC0-v0 no 44 110
'

echo "--- pass $(date +%F\ %H:%M) ---" >> "$LOG"
while read -r arm port host task freeze seed TARGET; do
  [ -z "$arm" ] && continue
  info=$(timeout 40 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" '
    tr=$(ps -eo cmd | grep -c "[t]rain.py")
    ex=$(ps -eo cmd | grep -c "[p]od_extend")
    ev=$(ps -eo cmd | grep -c "[p]lay_bc_pointcloud.py")
    g=0
    for d in /data/crane_testbed/logs/rsl_rl/*/20*; do
      hi=$(ls $d/model_*.pt 2>/dev/null | sed "s/.*model_//;s/\.pt//" | sort -n | tail -1)
      g=$((g + ${hi:-0}))
    done
    echo "$tr $ex $ev $g"' 2>/dev/null)
  if [ -z "$info" ]; then
    echo "$(stamp) $arm UNREACHABLE" >> "$LOG"; continue
  fi
  set -- $info; tr=$1 ex=$2 ev=$3 glob=$4
  echo "$(stamp) $arm train=$tr extend=$ex eval=$ev global=$glob/$TARGET" >> "$LOG"
  if [ "$tr" -eq 0 ] && [ "$ex" -eq 0 ] && [ "$ev" -eq 0 ] && [ "$glob" -lt "$TARGET" ]; then
    echo "$(stamp) $arm IDLE at $glob, relaunching extension" >> "$LOG"
    f=$freeze
    timeout 40 ssh -n -o StrictHostKeyChecking=no -p "$port" "root@$host" \
      "cd /data/crane_testbed && rm -f /data/CHAIN_DONE /data/extend_progress && \
       setsid nohup bash eval_scripts/pod_extend.sh $arm $task $TARGET $f $seed \
       > /data/extend.log 2>&1 < /dev/null & sleep 2; echo ok" >> "$LOG" 2>&1
  fi
done <<< "$ARMS"

# mirror evidence home (events, params, eval rows, checkpoints)
while read -r port host name; do
  [ -z "$port" ] && continue
  mkdir -p "$CK/$name"
  timeout 300 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" \
    'cd /data/crane_testbed && tar czf - logs/rsl_rl/*/2026-*/model_*.pt \
       logs/rsl_rl/*/2026-*/events.out.tfevents.* logs/rsl_rl/*/2026-*/params 2>/dev/null' \
    | tar xzf - --no-same-owner -C "$CK/$name" 2>/dev/null
done < logs/fleet/pods.txt
bash eval_scripts/fleet_tb_sync.sh >> "$LOG" 2>&1
echo "$(stamp) sync done: $(find "$CK" -name 'model_*.pt' | wc -l) checkpoints mirrored" >> "$LOG"
