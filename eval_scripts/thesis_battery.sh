#!/bin/bash
# THESIS ABLATION + SEED BATTERY - runs ON a RunPod pod (no docker; land inside the container).
#
# Inputs (shipped by tar from the laptop, see eval_scripts/thesis_battery_upload.sh):
#   $CRANE/logs/battery_ckpts/<arm>/model_<globaliter>.pt   RSL-RL checkpoints, renumbered
#   $CRANE/logs/battery_ckpts/template_scoring_policy.pt    BC template (metadata donor)
#
# Stage 1  waves: convert each checkpoint and run the clean-wave argmax protocol
#          (40 envs x 1 episode, seed 42, platform-v2). One row per checkpoint.
# Stage 2  deep rows: for each arm, pick the wave-best checkpoint (full%, cycles tiebreak)
#          and run the deep protocol (20 envs, 100 episodes, seed 42). Citable rows.
#
# Re-runnable: every row writes a .done marker and is skipped when present.
#   ./thesis_battery.sh waves            # all arms, default strides
#   ./thesis_battery.sh waves s44 unfroz # subset
#   ./thesis_battery.sh deep             # stage 2 (needs stage-1 rows)
set -u
PERSIST=${PERSIST:-/data}
CRANE=${CRANE:-$PERSIST/crane_testbed}
ISAACLAB=${ISAACLAB:-/workspace/isaaclab}
cd "$CRANE" || exit 1
export PYTHONPATH=$CRANE/source/crane_testbed:$CRANE/scripts/envs
export PYTHONUNBUFFERED=1   # block-buffered stdout made live logs lag by ~15 min
PLATFORM_V2="--profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 \
--log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200"
TPL=$CRANE/logs/battery_ckpts/template_scoring_policy.pt
OUT=$CRANE/logs/sim_eval/battery
mkdir -p "$OUT"

# arm -> wave stride (sig03fix is a 35-checkpoint archive; every 20 is plenty)
stride() { [ -n "${STRIDE:-}" ] && { echo "$STRIDE"; return; }; case "$1" in sig03fix) echo 20 ;; *) echo 10 ;; esac; }

convert() {  # convert <ckpt.pt> -> path of _bc_format.pt (cached)
  local c=$1 out=${1%.pt}_bc_format.pt
  [ -f "$out" ] || $ISAACLAB/_isaac_sim/python.sh scripts/envs/rsl_rl_to_scoring.py \
      --rsl_rl "$c" --template "$TPL" >/dev/null || return 1
  echo "$out"
}

wave_row() {  # wave_row <arm> <iter>
  local arm=$1 it=$2 tag=wave_${1}_${2} d
  d=$OUT/$tag
  [ -f "$d/.done" ] && { echo "[skip] $tag"; return; }
  local bc; bc=$(convert logs/battery_ckpts/$arm/model_$it.pt) || { echo "[convert FAIL] $tag"; return; }
  echo "=== $(date -u +%H:%M:%S) $tag"
  mkdir -p "$d"
  $ISAACLAB/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
    --policy_type scoring --crop_margin 0.5 --checkpoint "$bc" \
    --gaze --raw_pcd --crop_to_bounds --headless $PLATFORM_V2 \
    --num_envs 40 --num_episodes 40 --seed 42 --save_metrics \
    --output_dir "$d" > "$d.log" 2>&1 && touch "$d/.done" || echo "  FAILED -> $d.log"
}

deep_row() {  # deep_row <arm> <iter>
  local arm=$1 it=$2 tag=deep_${1}_${2} d
  d=$OUT/$tag
  [ -f "$d/.done" ] && { echo "[skip] $tag"; return; }
  local bc; bc=$(convert logs/battery_ckpts/$arm/model_$it.pt) || { echo "[convert FAIL] $tag"; return; }
  echo "=== $(date -u +%H:%M:%S) $tag (deep, ~2h)"
  mkdir -p "$d"
  $ISAACLAB/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
    --policy_type scoring --crop_margin 0.5 --checkpoint "$bc" \
    --gaze --raw_pcd --crop_to_bounds --headless $PLATFORM_V2 \
    --num_envs 20 --num_episodes 100 --seed 42 --save_metrics --save_decisions \
    --output_dir "$d" > "$d.log" 2>&1 && touch "$d/.done" || echo "  FAILED -> $d.log"
}

best_iter() {  # best wave checkpoint of an arm: max full_clear_rate, then min cycles
  # pure shell: the pods carry only Isaac Sim's bundled python, no system python3
  local arm=$1 d f it fc cy
  for d in "$OUT"/wave_"${arm}"_*/; do
    [ -d "$d" ] || continue
    it=${d%/}; it=${it##*_}
    f=$(ls "$d"/eval_metrics_*.json 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    fc=$(grep -o '"full_clear_rate": [0-9.]*' "$f" | head -1 | grep -o '[0-9.]*$')
    cy=$(grep -A1 '"cycles": {' "$f" | grep -o '"mean": [0-9.]*' | head -1 | grep -o '[0-9.]*$')
    [ -z "$fc" ] && continue
    echo "$fc ${cy:-999} $it"
  done | sort -k1,1gr -k2,2g | head -1 | awk '{print $3}'
}

ARMS_DEFAULT="symcritic unfroz s44 scc44 mn an nq sig03fix"
cmd=${1:-waves}; shift || true
ARMS=${*:-$ARMS_DEFAULT}

case "$cmd" in
  waves)
    for arm in $ARMS; do
      st=$(stride $arm)
      for f in $(ls logs/battery_ckpts/$arm/model_*.pt | grep -v bc_format | sort -t_ -k2 -n); do
        it=$(basename $f .pt | sed 's/model_//')
        [ $((it % st)) -ne 0 ] && continue
        wave_row $arm $it
      done
    done
    ;;
  deep)
    for arm in $ARMS; do
      it=$(best_iter $arm)
      [ -z "$it" ] && { echo "[deep] no wave rows for $arm yet"; continue; }
      deep_row $arm $it
    done
    ;;
  rank)
    printf "%-28s %6s %8s\n" row full% cycles
    for f in "$OUT"/*/eval_metrics_*.json; do
      [ -f "$f" ] || continue
      d=$(basename "$(dirname "$f")")
      fc=$(grep -o '"full_clear_rate": [0-9.]*' "$f" | head -1 | grep -o '[0-9.]*$')
      cy=$(grep -A1 '"cycles": {' "$f" | grep -o '"mean": [0-9.]*' | head -1 | grep -o '[0-9.]*$')
      printf "%-28s %6s %8s\n" "$d" "${fc:-?}" "${cy:-?}"
    done
    ;;
  *) echo "usage: thesis_battery.sh waves|deep|rank [arms...]"; exit 1 ;;
esac
