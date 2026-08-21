#!/bin/bash
# Wave grid + deep row for an arm whose checkpoints were uploaded to this pod.
# For arms whose training already happened elsewhere (rescued or archived runs).
#   pod_wavedeep.sh <arm> [stride]
set -u
ARM=$1 ST=${2:-30}
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
export PYTHONUNBUFFERED=1
n=$(ls logs/battery_ckpts/"$ARM"/model_*.pt 2>/dev/null | grep -vc bc_format)
[ "${n:-0}" -eq 0 ] && { echo "no checkpoints staged for $ARM"; exit 1; }
echo "$ARM: $n checkpoints, wave stride $ST"
STRIDE=$ST bash eval_scripts/thesis_battery.sh waves "$ARM" > /data/battery_waves.log 2>&1
bash eval_scripts/thesis_battery.sh deep "$ARM" > /data/battery_deep.log 2>&1
touch /data/CHAIN_DONE
