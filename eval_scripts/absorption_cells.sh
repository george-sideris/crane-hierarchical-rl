#!/bin/bash
# ABSORPTION CELLS - the six eval rows that complete the escape-vs-absorption analysis.
#
# WHY. The existing rows cannot test the field claim end to end:
#   - noise_sweep scoring_* rows are the TIGHT-crop ungated variant (its collapse is the
#     support-gate motivation, not the deployed policy);
#   - the sim heuristic's tight crop contains NO structure, so the real structure-lock
#     mechanism does not exist in those rows (0/17 absorbed at 1x-3x; locks only at 8x
#     from persistent noise).
# These cells run (a) the DEPLOYED margin scoring policy under characterized noise, and
# (b) the heuristic with structure visible in its crop (margin band), gated and ungated.
# The heurM rows are a MECHANISM DEMONSTRATION, not deployed parity: the real node also
# carries corner filters the sim path does not reproduce; --support_gate 0.25 is the
# documented parity layer for outlier handling only.
#
# Analysis: scripts/envs/analyze_absorption.py (episode-level lock/recovery endpoints;
# lock = >=3 consecutive empty cycles with successive targets within 0.25 m, starting
# with >5% inventory).
#
# Protocol matches noise_sweep.sh / citable_table.sh (platform-v2 flags, seed 42,
# 96 episodes at 8 envs, per-row .done markers, resumable).
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/absorption_cells
mkdir -p "$OUT"
ISAAC=${ISAAC:-/workspace/isaaclab/isaaclab.sh}
CKPT=/workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
ENVS=${ENVS:-8}
EPS=${EPS:-96}

COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --num_envs $ENVS --num_episodes $EPS --seed 42 --save_metrics --save_decisions \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
  --gripper_effort 2000 --num_logs 200"
N1X="--zed_noise --zed_axial_coeff 0.0014"
N3X="--zed_noise --zed_axial_coeff 0.0042"

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  [ -f "$OUT/$tag/.done" ] && { echo "[skip] $tag"; return; }
  echo "=== $(date +%H:%M:%S) $tag"
  # shellcheck disable=SC2086
  $ISAAC -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
      --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1 \
    && { mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done"; echo "  done $(date +%H:%M:%S)"; } \
    || echo "  FAILED -> $OUT/$tag.log"
}

# deployed margin scoring under characterized noise
run p2c_1x      --policy_type scoring --crop_margin 0.5 --checkpoint "$CKPT" $N1X
run p2c_3x      --policy_type scoring --crop_margin 0.5 --checkpoint "$CKPT" $N3X
# heuristic with structure visible in the crop (mechanism demonstration)
run heurM_clean --policy_type heuristic --heuristic_dig 0.25 --num_points 2048 --crop_margin 0.5
run heurM_1x    --policy_type heuristic --heuristic_dig 0.25 --num_points 2048 --crop_margin 0.5 $N1X
# same, with the documented outlier parity layer
run heurMG_clean --policy_type heuristic --heuristic_dig 0.25 --num_points 2048 --crop_margin 0.5 --support_gate 0.25
run heurMG_1x    --policy_type heuristic --heuristic_dig 0.25 --num_points 2048 --crop_margin 0.5 --support_gate 0.25 $N1X

echo "=== absorption cells done $(date +%H:%M:%S)"
echo "analyze with:"
echo "  python3 scripts/envs/analyze_absorption.py \\"
echo "    'p2c_clean=logs/sim_eval/citable_chunk/p2c_s4*/decisions*.npz' \\"
echo "    'p2c_1x=logs/sim_eval/absorption_cells/p2c_1x/decisions*.npz' \\"
echo "    'p2c_3x=logs/sim_eval/absorption_cells/p2c_3x/decisions*.npz' \\"
echo "    'heurM_clean=logs/sim_eval/absorption_cells/heurM_clean/decisions*.npz' \\"
echo "    'heurM_1x=logs/sim_eval/absorption_cells/heurM_1x/decisions*.npz' \\"
echo "    'heurMG_clean=logs/sim_eval/absorption_cells/heurMG_clean/decisions*.npz' \\"
echo "    'heurMG_1x=logs/sim_eval/absorption_cells/heurMG_1x/decisions*.npz'"
