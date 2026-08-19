#!/bin/bash
# Ship the thesis battery to a pod and launch it. Run from the laptop.
#   ./eval_scripts/thesis_battery_upload.sh <host> <port>
# Pod image ghcr.io/george-sideris/isaac-lab-crane:5.0.0-ssh (template db1ouvi1ka),
# persistent volume at /data. The ghcr image ships Isaac Lab at /workspace/isaaclab.
set -eu
HOST=$1; PORT=$2
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
cd "$(dirname "$0")/.."   # crane_testbed root

# code + assets (13 MB class) and the staged checkpoints (~215 MB) in one stream.
# patches/ excluded: the ghcr image already carries a patched Isaac Lab.
tar czf - assets source scripts eval_scripts runpod_setup.sh \
    fpi_crane_rl/fpi_crane_rl fpi_crane_rl/package.xml \
    logs/battery_ckpts \
  | $SSH 'mkdir -p /data/crane_testbed && tar xzf - -C /data/crane_testbed'

$SSH 'cd /data/crane_testbed && nohup bash eval_scripts/thesis_battery.sh waves \
    > /data/battery_waves.log 2>&1 & echo "battery launched, pid $!"'
echo "follow with: ssh -p $PORT root@$HOST tail -f /data/battery_waves.log"
