#!/bin/bash
# BCRL RE-RUN. The original bcrl rows used 2026-07-15/model_380, trained BEFORE platform-v2
# landed (tapered log asset 07-31, density-850, effort 2000). Those rows evaluate a policy on
# physics it never trained on - a domain shift, not a test of RL. This uses the 2026-08-02 CC run,
# which IS platform-v2 and was BC-initialised from the exact bc_aug1v2_dig25 it is compared against.
# Caveats: only 100 iterations, and cycle_cost=1.0 (previously measured SUB-THRESHOLD, ~3% of
# return vs ~10% batch variance, so effectively the same task as -MR-v0).
# Observation-degradation sweep: does the BC->RL margin GROW as the cloud gets worse?
#
# PREMISE (the RL chapter's core claim). The privileged expert is optimal under FULL observability;
# a cloud-conditioned policy acts under PARTIAL observability, and those are different optima. BC is
# trained to MATCH the expert, so it inherits the expert's confidence without the expert's
# information. RL optimises outcome under the observations it actually gets, so it can hedge (aim
# where a perception error still lands on logs instead of at one log's exact centre).
#
# PREDICTION
#   expert     FLAT by construction - it reads ground-truth log poses and never touches the cloud.
#              Deliberately NOT run: it would only re-measure physics variance. It is the ceiling.
#   heuristic  degrades fastest - pure argmax-highest, so one bad point decides the grasp
#   BC         degrades faster than BC->RL
#   BC->RL     degrades slowest  <-- if this ordering does NOT appear, the premise is wrong and the
#              RL chapter needs rethinking BEFORE more compute goes into it.
#
# x-axis is the PHYSICAL degradation axis: axial sigma = coeff * depth^2 applied to the depth image
# before unprojection (along the camera ray, like real stereo error), plus pixel dropout. 0.0014 is
# the characterised real ZED X, so the levels read as "how many times worse than our sensor".
#
# 12 runs, sequential (one GPU). Re-runnable: finished rows are skipped via .done markers.

set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs

OUT=/workspace/crane_testbed/logs/sim_eval/noise_sweep
mkdir -p "$OUT"
ISAAC=/workspace/isaaclab/isaaclab.sh

BC=/workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt
# BC->RL on the SAME task as BC (-MR-v0, no cycle_cost) so the only difference is the RL stage.
BCRL=/workspace/crane_testbed/logs/rsl_rl/crane_pointcloud_gaze_cossin_raw_mr_cc_v0/2026-08-02_22-19-46/model_100_bc_format.pt

# platform-v2 frozen flags - must match the reference eval protocol or rows are not comparable
COMMON="--gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 --gripper_effort 2000 \
  --num_logs 200 --num_envs 8 --num_episodes 10 --seed 42 --save_metrics --save_decisions"

run () {  # run <tag> <extra args...>
  local tag=$1; shift
  if [ -f "$OUT/$tag/.done" ]; then echo "[skip] $tag"; return; fi
  echo "=== $(date +%H:%M:%S) START $tag"
  # shellcheck disable=SC2086
  $ISAAC -p scripts/envs/play_bc_pointcloud.py $COMMON "$@" \
      --output_dir "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mkdir -p "$OUT/$tag"; touch "$OUT/$tag/.done";
  else echo "  !! FAILED rc=$rc -> $OUT/$tag.log"; fi
  tail -4 "$OUT/$tag.log"
  echo "=== $(date +%H:%M:%S) END $tag"
}

for LEVEL in clean 1x 3x 8x; do
  case $LEVEL in
    clean) N="" ;;
    1x)    N="--zed_noise --zed_axial_coeff 0.0014" ;;
    3x)    N="--zed_noise --zed_axial_coeff 0.0042" ;;
    8x)    N="--zed_noise --zed_axial_coeff 0.0112" ;;
  esac
  # shellcheck disable=SC2086
  
  # shellcheck disable=SC2086
  
  # shellcheck disable=SC2086
  run "bcrl2_${LEVEL}"      --policy_type bc --checkpoint "$BCRL" $N
done

echo "=== sweep complete $(date +%H:%M:%S)"
