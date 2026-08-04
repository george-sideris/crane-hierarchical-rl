#!/bin/bash
# Runs the three reference screens sequentially (each ~15-25 min at 8 envs).
# All rows: seed 42, 8 envs, 10 episodes, platform-v2 flags -> paired piles across rows.
set -e
cd /workspace/crane_testbed
echo "=== [1/3] expert (dig 0.25) ==="
bash /workspace/crane_testbed/eval_scripts/screen_expert.sh
echo "=== [2/3] heuristic bridge (dig 0.25) ==="
bash /workspace/crane_testbed/eval_scripts/screen_heuristic.sh
echo "=== [3/3] BC aug1v2 (dig 0.25) ==="
bash /workspace/crane_testbed/eval_scripts/screen_bc.sh
echo "=== ALL SCREENS DONE ==="
grep -a -A 8 "COLLECTION SUMMARY" /workspace/crane_testbed/logs/sim_eval/screen_expert.log | tail -8
grep -a -A 10 "RESULTS" /workspace/crane_testbed/logs/sim_eval/screen_heuristic.log | tail -10
grep -a -A 10 "RESULTS" /workspace/crane_testbed/logs/sim_eval/screen_bc.log | tail -10
