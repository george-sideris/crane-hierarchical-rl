#!/bin/bash
# Wave + deep evaluation of the checkpoints THIS pod just trained.
# For runs launched outside pod_chain.sh (the rerun arms).
#   pod_eval_own.sh <arm>
set -u
ARM=$1
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
RUN=$(ls -dt logs/rsl_rl/*/20* 2>/dev/null | head -1)
[ -z "$RUN" ] && { echo "no run dir"; exit 1; }
mkdir -p logs/battery_ckpts/"$ARM"
cp "$RUN"/model_*.pt logs/battery_ckpts/"$ARM"/ 2>/dev/null
rm -f logs/battery_ckpts/"$ARM"/model_0.pt
echo "staged $(ls logs/battery_ckpts/$ARM | wc -l) checkpoints from $RUN"
STRIDE=20 bash eval_scripts/thesis_battery.sh waves "$ARM" > /data/battery_waves.log 2>&1
bash eval_scripts/thesis_battery.sh deep "$ARM" > /data/battery_deep.log 2>&1
touch /data/CHAIN_DONE
