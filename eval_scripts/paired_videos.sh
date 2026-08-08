#!/bin/bash
# PAIRED qualitative episode: one episode of the SAME pile, four policies, video + decisions.
#
# WHY paired: _rebuild_log_origins_world seeds each pile with cfg.seed + episodes*num_envs +
# env_id. With --num_envs 1 and episode 0 that is exactly cfg.seed for every policy, so all
# four see an IDENTICAL starting pile and anything you see differ is the policy, not the pile.
# (This pile is NOT the same as any chunked-eval pile - those run num_envs 8 - which does not
# matter here: this run is qualitative, the numbers come from the 100-episode rows.)
#
# Two complementary artifacts per policy:
#   video     - frames captured INSIDE the FSM physics loop (every 4th step), so it shows the
#               actual motion: approach, close, lift, carry, deposit. ~2-4 min per episode.
#   decisions - one record per grasp CYCLE (input cloud + decoded target). This is where the
#               regression-vs-scoring difference is legible: the regression head averages modes
#               and can aim into the gap between two mounds; the scoring head must return an
#               observed point.
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/paired_video
MED=/workspace/crane_testbed/logs/sim_eval/paired_video/media
mkdir -p "$OUT" "$MED"
SEED=42

vid () {   # vid <tag> <args...>
  local tag=$1; shift
  local d="$OUT/$tag"
  [ -f "$d/.done" ] && { echo "[skip] $tag"; return; }
  mkdir -p "$d"
  echo "=== $(date +%H:%M:%S) $tag"
  # --num_envs 1 is REQUIRED by --record_video (asserted in play_bc_pointcloud.py)
  /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
    --gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
    --num_envs 1 --num_episodes 1 --seed "$SEED" --save_metrics --save_decisions \
    --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
    --gripper_effort 2000 --num_logs 200 \
    --record_video \
    --video_out    "$MED/${tag}_policyview.mp4" \
    --overview_out "$MED/${tag}_overview.mp4" \
    --sideview_out "$MED/${tag}_sideview.mp4" \
    "$@" --output_dir "$d" > "$d.log" 2>&1
  if ls "$d"/eval_metrics_*.json >/dev/null 2>&1; then
    touch "$d/.done"; echo "  ok $(date +%H:%M:%S)"
  else
    echo "  FAILED $tag (see $d.log)"
  fi
}

EXPERT_ARGS="--policy_type expert --expert_dig 0.25"
HEUR_ARGS="--policy_type heuristic --heuristic_dig 0.25"
BCREG_ARGS="--policy_type bc --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt"
P2C_ARGS="--policy_type scoring --crop_margin 0.5 --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt"

# PASS 1 (run FIRST - this is the diagnostic): forced two-mound pile, the geometry that broke
# the regression policy on the real rack. A random pile shows two_mounds only ~1 episode in 8,
# so without pinning it the comparison might never present the failure case at all.
FORCE="--force_pile_profile two_mounds"
vid tm_expert    $EXPERT_ARGS $FORCE
vid tm_heuristic $HEUR_ARGS   $FORCE
vid tm_bc_reg    $BCREG_ARGS  $FORCE
vid tm_p2c       $P2C_ARGS    $FORCE

# A random-pile pass was dropped on purpose (2026-08-08): a random pile shows the double-mound
# case only ~1 episode in 8, so it mostly duplicates what the 100-episode rows already report,
# and the ~90 min it costs is worth more as pure-RL training time.

# Burn the per-cycle accounting onto each overview video, so the counters can be checked
# against what the grapple visibly does (REPORTED vs MEASURED vs TRUE, phantom successes in
# red). Non-fatal: a failed overlay must not cost us the raw videos.
for tag in tm_expert tm_heuristic tm_bc_reg tm_p2c; do
  [ -f "$OUT/$tag/.done" ] || continue
  echo "--- overlay $tag"
  /workspace/isaaclab/_isaac_sim/python.sh scripts/envs/overlay_metrics_video.py \
    --video "$MED/${tag}_overview.mp4" \
    --decisions "$OUT/$tag" \
    --out "$MED/${tag}_overview_annotated.mp4" 2>&1 | tail -3 || echo "  overlay FAILED for $tag"
done

echo "=== paired videos complete $(date +%H:%M:%S)"
ls -la "$MED"
