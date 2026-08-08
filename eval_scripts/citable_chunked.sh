#!/bin/bash
# CHUNKED citable rows: 4 chunks x 25 episodes per policy, seeds 42/43/44/45.
# WHY: eval writes metrics+decisions only after the LAST episode, so a mid-row crash loses
# everything (happened twice on 2026-08-07: display power transitions reset the CUDA context
# ~70 min into 3.5 h rows). Chunking caps the loss at one 50-min chunk and makes the row
# resumable via per-chunk .done markers. All policies run the SAME four seeds, so the paired
# comparison across rows is preserved; the pooled 100 episodes are then equivalent to the
# single-run version, with seed diversity as a bonus.
# Auto-retries a chunk ONCE (a crash usually needs a container restart, not different settings).
set -u
cd /workspace/crane_testbed || exit 1
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs
OUT=/workspace/crane_testbed/logs/sim_eval/citable_chunk
mkdir -p "$OUT"

# A lost CUDA context (display power transition on this hybrid-graphics laptop) cannot be
# recovered from inside the container - it needs `sudo rmmod nvidia_uvm && sudo modprobe
# nvidia_uvm` on the host. Retrying through it just burns hours (2026-08-07: 4 h lost to
# retries that could never succeed). So: verify CUDA before every chunk and ABORT the whole
# script the moment it is gone, leaving a clear marker for the human.
gpu_ok () {
  docker_out=$(/workspace/isaaclab/_isaac_sim/python.sh -c \
    "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null)
  return $?
}

chunk () {   # chunk <tag> <seed> <args...>
  local tag=$1 seed=$2; shift 2
  local d="$OUT/${tag}_s${seed}"
  [ -f "$d/.done" ] && { echo "[skip] ${tag}_s${seed}"; return; }
  if ! gpu_ok; then
    echo "!! CUDA UNAVAILABLE before ${tag}_s${seed} - ABORTING."
    echo "!! Fix on the HOST:  sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
    echo "!! Then rerun this script; completed chunks are skipped."
    touch "$OUT/.gpu_dead"
    exit 90
  fi
  for attempt in 1 2; do
    echo "=== $(date +%H:%M:%S) ${tag}_s${seed} (attempt $attempt)"
    /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
      --gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
      --num_envs 8 --num_episodes 25 --seed "$seed" --save_metrics --save_decisions \
      --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
      --gripper_effort 2000 --num_logs 200 "$@" \
      --output_dir "$d" > "$d.log" 2>&1
    if ls "$d"/eval_metrics_*.json >/dev/null 2>&1; then
      touch "$d/.done"; echo "  ok $(date +%H:%M:%S)"; return
    fi
    echo "  chunk failed (attempt $attempt)"; sleep 5
    if ! gpu_ok; then
      echo "!! CUDA died during ${tag}_s${seed} - ABORTING (needs host rmmod/modprobe)."
      touch "$OUT/.gpu_dead"
      exit 90
    fi
  done
  echo "  GAVE UP on ${tag}_s${seed}"
}

# one policy at a time, all four seeds. SCORING (p2c) FIRST - it is the headline row.

for SEED in 42 43 44 45; do
  chunk p2c "$SEED" --policy_type scoring --crop_margin 0.5 \
      --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt
done
for SEED in 42 43 44 45; do
  chunk bc_reg "$SEED" --policy_type bc \
      --checkpoint /workspace/crane_testbed/logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt
done
echo "=== chunked rows done $(date +%H:%M:%S)"
