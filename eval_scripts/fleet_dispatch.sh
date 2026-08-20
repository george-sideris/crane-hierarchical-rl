#!/bin/bash
# Launch queued seed runs onto idle pods. One pass per invocation; prints a line per launch.
# State (survives session restarts): logs/fleet/seed_queue.txt, logs/fleet/pods.txt
# Queue line: <arm> <task> <iters> <freeze:yes|no> <seed>. Launched lines are removed.
cd "$(dirname "$0")/.." || exit 1
# single-instance: concurrent passes raced and double-launched a job
exec 9> logs/fleet/.dispatch.lock 2>/dev/null || true
flock -n 9 || { echo "dispatch already running"; exit 0; }
Q=logs/fleet/seed_queue.txt
PODS=logs/fleet/pods.txt
CODE=logs/fleet/crane_code.tgz
ROS2=logs/fleet/ros2copy.tgz
[ -s "$Q" ] || exit 0
# pods created but never registered bill while idle and invisible: warn loudly.
if [ -s logs/fleet/pending_pods.txt ]; then
  echo "WARNING: $(wc -l < logs/fleet/pending_pods.txt) pod(s) in logs/fleet/pending_pods.txt are NOT in pods.txt - register or delete them"
fi
# always rebuild: a stale tar silently ships pods a script that does not exist yet
tar czf "$CODE" --exclude='__pycache__' --exclude='*.pyc' \
  assets source scripts eval_scripts runpod_setup.sh fpi_crane_rl/fpi_crane_rl \
  fpi_crane_rl/package.xml logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt \
  logs/bc_pointcloud/scoring_margin05_2048_c/config.json 2>/dev/null
tar czf "$ROS2" --exclude='__pycache__' fpi_crane_ros2/fpi_crane_rl/fpi_crane_rl 2>/dev/null

while read -r port host name; do
  [ -z "$port" ] && continue
  [ -s "$Q" ] || break
  busy=$(timeout 40 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p "$port" "root@$host" \
    '{ pgrep -c -f "rsl_rl/train[.]py|thesis_battery[.]sh|play_bc_pointcloud[.]py" 2>/dev/null | head -1; }' 2>/dev/null)
  [ -z "$busy" ] && continue          # unreachable this pass
  [ "$busy" -gt 0 ] && continue       # still working
  free=$(timeout 40 ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p "$port" "root@$host" \
    'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits' 2>/dev/null | head -1)
  if [ -z "$free" ] || [ "$free" -lt 35000 ]; then
    echo "$name: SKIP, only ${free:-?} MiB free"; continue
  fi
  read -r kind a1 a2 a3 a4 a5 < "$Q"
  [ -z "$kind" ] && break
  scp -o StrictHostKeyChecking=no -P "$port" -q "$CODE" "$ROS2" "root@$host:/data/" 2>/dev/null
  [ "$kind" = bceval ] && scp -o StrictHostKeyChecking=no -P "$port" -q logs/fleet/bc_seed_ckpts.tgz "root@$host:/data/" 2>/dev/null
  timeout 90 ssh -n -o StrictHostKeyChecking=no -p "$port" "root@$host" \
    'mkdir -p /data/crane_testbed && cd /data/crane_testbed && tar xzf /data/crane_code.tgz 2>/dev/null; tar xzf /data/ros2copy.tgz 2>/dev/null; tar xzf /data/bc_seed_ckpts.tgz 2>/dev/null; ln -sfn /data/crane_testbed /workspace/crane_testbed; rm -f /data/CHAIN_DONE /data/train_finished' 2>/dev/null
  if [ "$kind" = bceval ]; then
    cmd="bash eval_scripts/pod_bc_eval.sh $a1 $a2 $a3"
    desc="bceval $a1"
  else
    cmd="bash eval_scripts/pod_chain.sh $a1 $a2 $a3 $a4 $a5"
    desc="$a1 seed $a5"
  fi
  timeout 40 ssh -n -o StrictHostKeyChecking=no -p "$port" "root@$host" \
    "cd /data/crane_testbed && setsid nohup $cmd > /data/chain.log 2>&1 < /dev/null & true" 2>/dev/null
  sed -i '1d' "$Q"
  echo "LAUNCHED $desc on $name; $(wc -l < "$Q") queued"
done < "$PODS"
