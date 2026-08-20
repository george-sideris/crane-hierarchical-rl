#!/bin/bash
# Fleet watchdog: emit a line only on failure, stall, or exit. Loops until killed.
cd "$(dirname "$0")/.." || exit 1
declare -A last_prog last_err seen
while true; do
  while read -r port host name; do
    [ -z "$port" ] && continue
    out=$(timeout 60 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" \
      'busy=$({ pgrep -c -f "rsl_rl/train[.]py|thesis_battery[.]sh|play_bc_pointcloud[.]py" 2>/dev/null | head -1; });
       cyc=$(grep -ac CYCLE /data/crane_testbed/logs/p3v2_train.log 2>/dev/null || echo 0);
       rows=$(find /data/crane_testbed/logs/sim_eval/battery -name .done 2>/dev/null | wc -l);
       # eval jobs write no CYCLE count and no .done until they finish, so growing log
       # bytes is the progress signal that covers train and eval alike
       kb=$(cat /data/crane_testbed/logs/p3v2_train.log /data/crane_testbed/logs/sim_eval/battery/*.log 2>/dev/null | wc -c);
       kb=$((kb/1000));
       # CPU time of the simulator process: the one liveness signal that survives
       # block-buffered logs and long episodes with no prints
       pid=$(pgrep -f "kit/python/bin/python3" | head -1);
       cpu=$(awk "{print int((\$14+\$15)/100)}" /proc/$pid/stat 2>/dev/null || echo 0);
       err=$(grep -aE "Traceback|CUDA error|out of memory|ERROR_DEVICE_LOST" /data/crane_testbed/logs/p3v2_train.log 2>/dev/null | grep -cv "Warp CUDA error");
       echo "$busy $((cyc+rows+kb+${cpu:-0})) $err"' 2>/dev/null)
    [ -z "$out" ] && { echo "$name: UNREACHABLE"; continue; }
    read -r busy prog err <<< "$out"
    if [ -z "${seen[$name]:-}" ]; then
      # first sight of this pod: record baselines, alert on nothing. Historical error
      # lines from an old incident are not news every time the watchdog restarts.
      seen[$name]=1; last_err[$name]=$err; last_prog[$name]=$prog; continue
    fi
    if [ "${err:-0}" -gt "${last_err[$name]:-0}" ]; then echo "$name: error lines grew to $err"; fi
    last_err[$name]=$err
    if [ "${busy:-0}" -eq 0 ]; then echo "$name: IDLE (work finished or died)"; fi
    if [ -n "${last_prog[$name]:-}" ] && [ "$prog" = "${last_prog[$name]}" ] && [ "${busy:-0}" -gt 0 ]; then
      echo "$name: STALL? progress frozen at $prog for 45 min"
    fi
    last_prog[$name]=$prog
  done < logs/fleet/pods.txt
  sleep 2700
done
