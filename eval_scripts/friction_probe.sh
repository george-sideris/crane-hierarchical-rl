#!/bin/bash
# Does anything actually PRESERVE the spawned two-mound profile through settling?
#
# Baseline (default friction 0.5, ang_damping 3.0) relaxes a 3.1 m-separated / 0.24-0.38 m-deep
# double mound down to 1.96 m / 0.05 m - essentially flat. The --log_friction help claims
# bark-like friction "raises the angle of repose so height profiles persist", but that is an
# untested assertion, and sliding friction is the wrong lever if the logs are ROLLING: mu=0.5
# already implies ~27 deg repose, above the 15-22 deg slope the spawn code limits itself to,
# yet the profile still collapsed. Angular damping is what resists rolling.
#
# One cycle each, no video - we only need the settled cloud from cycle 0.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/friction_probe
mkdir -p "$OUT"

probe () {   # probe <tag> <extra args...>
  local tag=$1; shift
  local d="$OUT/$tag"
  [ -f "$d/.done" ] && { echo "[skip] $tag"; return; }
  mkdir -p "$d"
  echo "=== $(date +%H:%M:%S) $tag :: $*"
  /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
    --gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
    --num_envs 1 --num_episodes 1 --seed 42 --save_decisions \
    --log_scale_mean 1.0 --log_scale_jitter 0.10 \
    --gripper_effort 2000 --num_logs 200 --max_grasp_cycles 1 \
    --policy_type heuristic --heuristic_dig 0.25 \
    --force_pile_profile two_mounds "$@" \
    --output_dir "$d" > "$d.log" 2>&1
  if ls "$d"/decisions_*.npz >/dev/null 2>&1; then
    touch "$d/.done"; echo "  ok $(date +%H:%M:%S)"
  else
    echo "  FAILED $tag (see $d.log)"
  fi
}

probe fr05_ad3   --log_ang_damping 3.0                        # baseline, reproduces the flat result
probe fr09_ad3   --log_ang_damping 3.0  --log_friction 0.9    # the claim under test
probe fr09_ad20  --log_ang_damping 20.0 --log_friction 0.9    # resist ROLLING, the real suspect
probe fr05_ad20  --log_ang_damping 20.0                       # isolates damping from friction

echo "=== friction probe complete $(date +%H:%M:%S)"
