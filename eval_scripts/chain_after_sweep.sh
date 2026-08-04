#!/bin/bash
# Overnight chain, in order:
#   1. wait for noise_sweep.sh (heuristic/BC/BCRL x 4 degradation levels) to finish
#   2. scoring-head rows on the SAME protocol -> 4th curve in the same table
#   3. the margin/2048 collection (~17.5 h) so the GPU is not idle
# Cancel:  docker exec isaac-lab-ros2 pkill -f chain_after_sweep
set -u
cd /workspace/crane_testbed || exit 1
echo "[chain] waiting for noise_sweep.sh to exit ..."
until ! pgrep -f "noise_sweep.sh" > /dev/null; do sleep 60; done
echo "[chain] sweep done $(date +%F_%H:%M:%S) -> scoring rows"
bash eval_scripts/scoring_sweep.sh
echo "[chain] scoring rows returned $? $(date +%F_%H:%M:%S) -> collection"
bash eval_scripts/collect_margin2048.sh
echo "[chain] collection returned $? $(date +%F_%H:%M:%S)"
