#!/bin/bash
# One-screen campaign state: pods, queue, and evidence collected so far.
cd "$(dirname "$0")/.." || exit 1
echo "=== PODS ==="
while read -r port host name; do
  [ -z "$port" ] && continue
  out=$(timeout 30 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$port" "root@$host" \
    'b=$(pgrep -c -f "rsl_rl/train.py|thesis_battery[.]sh|play_bc_pointcloud[.]py" 2>/dev/null || echo 0);
     it=$(grep -a "Learning iteration" /data/crane_testbed/logs/p3v2_train.log 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+");
     rows=$(find /data/crane_testbed/logs/sim_eval/battery -name .done 2>/dev/null | wc -l);
     d=$(test -f /data/CHAIN_DONE && echo DONE || echo -);
     echo "busy=$b iter=${it:-.} rows=$rows chain=$d"' 2>/dev/null)
  printf "  %-12s %s\n" "$name" "${out:-UNREACHABLE}"
done < logs/fleet/pods.txt
echo "=== QUEUE ($(wc -l < logs/fleet/seed_queue.txt) jobs) ==="
head -4 logs/fleet/seed_queue.txt | sed 's/^/  next: /'
echo "=== EVIDENCE (synced home) ==="
python3 - <<'PY'
import glob, json, re
from collections import defaultdict
waves, deeps = defaultdict(set), {}
for f in glob.glob("logs/fleet_tb_live/*/battery/*/eval_metrics_*.json"):
    tag = f.split("/")[-2]
    m = re.match(r"wave_(.+)_(\d+)$", tag)
    if m:
        waves[m.group(1)].add(int(m.group(2))); continue
    if tag.startswith("deep_"):
        s = json.load(open(f))["summary"]
        deeps[tag[5:]] = (s["full_clear_rate"], s["cycles"]["mean"])
print("  wave grids: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(waves.items())) or "  none")
if deeps:
    for k, v in sorted(deeps.items()):
        print(f"  DEEP {k:18s} full {v[0]:5.1f}%  cycles {v[1]:5.2f}")
else:
    print("  deep rows: none yet")
tb = glob.glob("logs/fleet_tb_live/*/logs/rsl_rl/*/*/events.out.tfevents.*")
print(f"  training curves synced: {len(tb)} runs")
PY
