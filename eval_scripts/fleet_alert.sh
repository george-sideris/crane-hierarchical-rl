#!/bin/bash
# Watch the live fleet and print one line whenever a pod CHANGES state. Silence means
# every pod is still working; a repeated condition is not repeated on the wire, so the
# only lines that appear are transitions worth acting on.
set -u
cd "$(dirname "$0")/.." || exit 1
STATE=logs/fleet/alert_state
mkdir -p logs/fleet; touch "$STATE"

prev() { grep -m1 "^$1 " "$STATE" 2>/dev/null | cut -d' ' -f2; }
setst() { grep -v "^$1 " "$STATE" > "$STATE.t" 2>/dev/null; echo "$1 $2" >> "$STATE.t"; mv "$STATE.t" "$STATE"; }

while true; do
  while read -r port host name; do
    [ -z "$port" ] && continue
    out=$(timeout 45 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$port" "root@$host" '
      b=$(ps -eo cmd | grep -cE "[r]sl_rl/train\.py|[p]lay_bc_pointcloud\.py|[p]od_extend|[t]hesis_battery")
      d=$(test -f /data/CHAIN_DONE && echo DONE || echo -)
      g=0
      for r in /data/crane_testbed/logs/rsl_rl/*/20*; do
        h=$(ls $r/model_*.pt 2>/dev/null | sed "s/.*model_//;s/\.pt//" | sort -n | tail -1)
        g=$((g + ${h:-0}))
      done
      echo "$b $d $g"' 2>/dev/null)

    if [ -z "$out" ]; then now=UNREACHABLE; detail="ssh failed"
    else
      set -- $out; busy=$1 done=$2 glob=$3
      if [ "${busy:-0}" -gt 0 ]; then now=BUSY; detail="global $glob"
      elif [ "$done" = DONE ]; then now=DONE; detail="chain complete, idle at global $glob"
      else now=DEAD; detail="nothing running, no CHAIN_DONE, global $glob"
      fi
    fi

    was=$(prev "$name")
    if [ "$now" != "${was:-}" ]; then
      setst "$name" "$now"
      case "$now" in
        DEAD)        echo "DIED: $name - $detail" ;;
        UNREACHABLE) echo "UNREACHABLE: $name - $detail" ;;
        DONE)        echo "FINISHED: $name - $detail" ;;
        BUSY)        [ -n "${was:-}" ] && echo "RECOVERED: $name working again ($detail)" ;;
      esac
    fi
  done < logs/fleet/pods.txt
  sleep 600
done
