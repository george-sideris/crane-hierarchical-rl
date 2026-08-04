#!/bin/bash
# Chain v2: wait for the scoring rows, then re-run BCRL on a platform-v2 checkpoint, then collect.
# The original bcrl rows are INVALID (pre-platform-v2 checkpoint on post-platform-v2 physics).
set -u
cd /workspace/crane_testbed || exit 1
echo "[chain2] waiting for scoring_sweep.sh to exit ..."
until ! pgrep -f "scoring_sweep.sh" > /dev/null; do sleep 60; done
echo "[chain2] scoring done $(date +%F_%H:%M:%S) -> bcrl re-run (platform-v2 ckpt)"
bash eval_scripts/bcrl_v2_sweep.sh
echo "[chain2] bcrl2 returned $? $(date +%F_%H:%M:%S) -> collection"
bash eval_scripts/collect_margin2048.sh
echo "[chain2] collection returned $? $(date +%F_%H:%M:%S)"
