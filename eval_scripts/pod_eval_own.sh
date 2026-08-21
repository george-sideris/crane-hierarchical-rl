#!/bin/bash
# Wave + deep evaluation of every checkpoint this pod trained, across all run
# directories (a resumed run creates a new one, numbering from zero again, so
# checkpoints are renumbered onto a single global axis before evaluation).
#   pod_eval_own.sh <arm>
set -u
ARM=$1
cd /data/crane_testbed || exit 1
export PYTHONPATH=/data/crane_testbed/source/crane_testbed:/data/crane_testbed/scripts/envs
export PYTHONUNBUFFERED=1
# the conversion needs a metadata donor
[ -f logs/battery_ckpts/template_scoring_policy.pt ] || {
  mkdir -p logs/battery_ckpts
  cp logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt \
     logs/battery_ckpts/template_scoring_policy.pt 2>/dev/null; }
mkdir -p logs/battery_ckpts/"$ARM"
off=0
for RUN in $(ls -dtr logs/rsl_rl/*/20* 2>/dev/null); do
  hi=0
  for f in "$RUN"/model_*.pt; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .pt); n=${n#model_}
    case "$n" in *[!0-9]*) continue ;; esac
    [ "$n" -eq 0 ] && continue
    cp -n "$f" logs/battery_ckpts/"$ARM"/model_$((off + n)).pt
    [ "$n" -gt "$hi" ] && hi=$n
  done
  off=$((off + hi))
done
echo "staged $(ls logs/battery_ckpts/$ARM/model_*.pt 2>/dev/null | grep -vc bc_format) checkpoints on a global axis"
STRIDE=20 bash eval_scripts/thesis_battery.sh waves "$ARM" > /data/battery_waves.log 2>&1
bash eval_scripts/thesis_battery.sh deep "$ARM" > /data/battery_deep.log 2>&1
touch /data/CHAIN_DONE
