#!/bin/bash
# Bring a rented GPU box from bare metal to a running BC->RL fine-tune.
#
# UNTESTED ON A CLOUD BOX - written from the working laptop setup, so treat the first run as a
# dry run and expect to fix a step. Every stage is separately re-runnable and idempotent.
#
#   ./cloud_setup.sh preflight    # is this machine even usable (RTX? docker? nvidia runtime?)
#   ./cloud_setup.sh setup        # clone IsaacLab, apply patches, build the container
#   ./cloud_setup.sh ladder       # measure the REAL env-count ceiling on this card
#   ./cloud_setup.sh train N      # launch the PPO-v2 fine-tune with N envs
#   ./cloud_setup.sh sweep 50 10  # argmax-eval every 50th checkpoint, 10 eps, and rank them
#   ./cloud_setup.sh fetch NAME   # tar the artifacts - RUN BEFORE TERMINATING THE INSTANCE
#
# Container providers (RunPod, Vast): use runpod_setup.sh instead - NONE of the stages below
# work there. You land INSIDE the GPU container, and every stage drives it from outside via
# `docker exec`. This script is for VM providers (Lambda, GCP), where you get a docker daemon.
#
# HARD REQUIREMENT: an RTX card. Isaac Sim renders through RTX/Vulkan ray tracing and this env
# renders a camera to build its observation, so A100/H100 (no RT cores) will fail with
# "Your GPUs do not support RayTracing". Want RTX 4090 / A40 / L40S / RTX 6000 Ada / A10G.
set -u

ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
BASE_COMMIT="4f81564f3cd2266459f45a35154cd85eaa3d9b4b"   # patch was taken against this
REPO_URL="${REPO_URL:-git@github.com:george-sideris/RLCraneTestbed.git}"
CONTAINER="${CONTAINER:-isaac-lab-base}"
# Platform-v2 physics flags - FROZEN for comparability with every existing row.
PLATFORM_V2="--profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 \
--log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200"
CKPT="/workspace/crane_testbed/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt"

say () { echo -e "\n=== $* "; }

# ---------------------------------------------------------------- preflight
preflight () {
  local ok=0
  say "GPU"
  if ! command -v nvidia-smi >/dev/null; then echo "  NO nvidia-smi - no driver"; return 1; fi
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  local vram; vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  echo "  $name  ($vram)"
  # RT cores: RTX/L40/A40/A10 have them; A100/H100/V100/T4 do not (T4 has RT but is tiny).
  if echo "$name" | grep -qiE "A100|H100|V100|K80|P100"; then
    echo "  !! This is a datacenter card with NO RT cores. Isaac Sim rendering will FAIL."
    echo "  !! Pick RTX 4090 / A40 / L40S / RTX 6000 Ada / A10G instead."
    ok=1
  fi
  say "docker + nvidia runtime"
  if ! command -v docker >/dev/null; then
    echo "  NO docker. On a VM provider:"
    echo "    curl -fsSL https://get.docker.com | sh"
    echo "    # then the NVIDIA Container Toolkit (nvidia-ctk) per NVIDIA's install docs"
    ok=1
  elif ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "  docker present but NVIDIA runtime not registered:"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    ok=1
  else
    echo "  ok"
  fi
  say "disk (image + build needs ~60 GB)"
  df -h / | tail -1
  [ $ok -eq 0 ] && echo -e "\npreflight OK" || echo -e "\npreflight FAILED - fix the above first"
  return $ok
}

# ---------------------------------------------------------------- setup
setup () {
  # SHORTCUT WORTH TRYING FIRST: NVIDIA publishes prebuilt Isaac Lab images. If one runs, you
  # skip the clone/patch/build entirely (~45 min of paid GPU time). The patches below exist to
  # fix a `packaging` conflict during `isaaclab.sh --install`; a recent official image may
  # already be clean. Test with:
  #   docker run --rm --gpus all nvcr.io/nvidia/isaac-lab:latest python -c "import isaaclab"
  # If that works, clone this repo into the image and skip to `ladder`.
  if [ ! -d "$ISAACLAB_DIR" ]; then
    say "cloning IsaacLab at the patch base commit"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" || return 1
    git -C "$ISAACLAB_DIR" checkout "$BASE_COMMIT" || {
      echo "  base commit missing; staying on default branch and using 3-way apply"; }
  fi
  if [ ! -d "$ISAACLAB_DIR/crane_testbed" ]; then
    say "cloning crane_testbed (assets + P2c checkpoint travel with it)"
    git clone "$REPO_URL" "$ISAACLAB_DIR/crane_testbed" || return 1
  fi

  say "applying IsaacLab-side patches"
  cd "$ISAACLAB_DIR" || return 1
  # The COPY target must exist BEFORE the build or docker fails with "file not found".
  mkdir -p docker/patches
  cp crane_testbed/isaaclab_patches/_structures.py docker/patches/_structures.py
  if git apply --check crane_testbed/isaaclab_patches/isaaclab_local_changes.patch 2>/dev/null; then
    git apply crane_testbed/isaaclab_patches/isaaclab_local_changes.patch
    echo "  applied cleanly"
  elif git apply -3 crane_testbed/isaaclab_patches/isaaclab_local_changes.patch 2>/dev/null; then
    echo "  applied with 3-way merge (upstream moved) - check docker/Dockerfile.base"
  else
    echo "  !! patch did not apply. The ONLY hunk that matters for sim-only training is the"
    echo "  !! packaging==23.0 pin in Dockerfile.base; apply it by hand if needed."
  fi

  say "building the container (base profile - ros2 is deployment-only, skip it)"
  echo "  this pulls ~20-30 GB and takes 30-60 min; it is billed GPU time"
  python3 docker/container.py start base || return 1
  echo "  container up"
}

# ---------------------------------------------------------------- ladder
# The old note "20 envs max, 32 corrupts the scene (6400 rigid bodies)" is PAPER-ERA, measured
# on an 8 GB laptop with untuned PhysX buffers and cylinder logs. It may not hold here. But more
# VRAM alone may not lift it either: "Scene state is corrupted" is usually a FIXED-SIZE PhysX
# GPU buffer overflowing, which stays the same size on a bigger card. If a rung dies with that
# error, the fix is a PhysxCfg on SimulationCfg in crane_rl_env_gaze.py (~line 1230) raising
# gpu_max_rigid_contact_count / gpu_found_lost_pairs_capacity / gpu_collision_stack_size.
#
# Measure THROUGHPUT, not just "did it launch". A rung that runs 64 envs at the same cycles/min
# as 16 is render- or Python-bound (14 per-env python loops in the hot path, and each env
# renders 1920x1080). Note that even a throughput-neutral rung is still useful: PPO batch =
# envs x steps, and too-small batches are exactly what broke the earlier fine-tunes.
ladder () {
  local rungs="${*:-8 16 32 64}"
  local secs="${LADDER_SECS:-360}"
  say "env-count ladder: $rungs  (${secs}s per rung)"
  printf "%6s %10s %12s %14s %s\n" "envs" "VRAM_MiB" "cycles" "cycles/min" "result"
  for n in $rungs; do
    local log="/workspace/crane_testbed/logs/ladder_${n}.log"
    docker exec -d "$CONTAINER" bash -c "cd /workspace/crane_testbed && \
      export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
      env PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
        --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2 --bc_checkpoint $CKPT \
        --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs $n --max_iterations 3 \
        --headless $PLATFORM_V2 agent.num_steps_per_env=8 > $log 2>&1"
    local peak=0 t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "$secs" ]; do
      local m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$m" -gt "$peak" ] && peak=$m
      docker exec "$CONTAINER" pgrep -f "num_envs $n " >/dev/null 2>&1 || break
      sleep 10
    done
    docker exec "$CONTAINER" bash -c "pkill -f 'num_envs $n '" 2>/dev/null
    sleep 5
    local c res
    c=$(docker exec "$CONTAINER" bash -c "grep -c CYCLE $log" 2>/dev/null || echo 0)
    if docker exec "$CONTAINER" bash -c "grep -qi 'state is corrupted\|corrupted' $log" 2>/dev/null; then
      res="PHYSX CORRUPTED -> raise PhysxCfg buffers"
    elif docker exec "$CONTAINER" bash -c "grep -qi 'out of memory\|OUT_OF_DEVICE_MEMORY' $log" 2>/dev/null; then
      res="OOM"
    elif [ "${c:-0}" -gt 0 ]; then res="ok"; else res="no cycles - see $log"; fi
    printf "%6s %10s %12s %14s %s\n" "$n" "$peak" "$c" \
      "$(python3 -c "print(f'{${c:-0}/($secs/60):.1f}')" 2>/dev/null || echo '?')" "$res"
  done
  echo
  echo "Pick the largest rung that is 'ok' AND still gaining cycles/min. If cycles/min has"
  echo "flattened, prefer the larger rung anyway for the bigger PPO batch, but know that you"
  echo "are buying batch size, not speed."
}

# ---------------------------------------------------------------- train
train () {
  local n="${1:-16}"
  say "launching PPO-v2 BC->RL fine-tune with $n envs (batch = ${n} x 8)"
  docker exec -d "$CONTAINER" bash -c "cd /workspace/crane_testbed && \
    export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
    env PYTHONUNBUFFERED=1 nohup /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2 --bc_checkpoint $CKPT \
      --freeze_encoder --sigma_init 0.05 \
      --critic_warmup_iters 25 --anneal_sigma_iters 100 --anneal_sigma_to 0.01 \
      --seed 42 --num_envs $n --max_iterations 400 --headless \
      $PLATFORM_V2 agent.num_steps_per_env=8 \
      > logs/p3v2_train.log 2>&1 &"
  sleep 90
  docker exec "$CONTAINER" bash -c "grep -c CYCLE /workspace/crane_testbed/logs/p3v2_train.log" 2>/dev/null
  cat <<'NOTE'

WATCH THESE, in order of how much they have cost before:
 1. logs/reward_audit/audit_<pid>.jsonl - after ~200 cycles check reward tracks the ACTUAL
    rack decrease. If it does not, kill the run: that is reward hacking, and it is exactly
    what invalidated the two earlier fine-tunes.
 2. Mean episode length - max_grasp_cycles is 30, so ~29.5 means piles are never being
    cleared no matter what the reward curve says.
 3. Do NOT judge by training reward. Checkpoints land every 10 iterations; pick the best by
    ARGMAX eval (play_bc_pointcloud.py --policy_type scoring), never by the reward curve.
NOTE
}

# ---------------------------------------------------------------- sweep
# Rank checkpoints by ARGMAX eval, which is the whole point: the policy trains stochastic and
# DEPLOYS argmax, so the reward curve does not tell you which checkpoint to keep. Selecting on
# training reward is one of the four things that made the earlier fine-tunes uninterpretable.
# Short rows (10 episodes) are for RANKING only - re-run the winner chunked at 100 episodes
# before putting a number in the thesis.
sweep () {
  local every="${1:-50}" eps="${2:-10}"
  local rundir
  rundir=$(docker exec "$CONTAINER" bash -c \
    "ls -dt /workspace/crane_testbed/logs/rsl_rl/crane_scoring_ppo_v2/*/ 2>/dev/null | head -1" | tr -d '\r')
  [ -z "$rundir" ] && { echo "no crane_scoring_ppo_v2 run found"; return 1; }
  say "argmax sweep over $rundir (every ${every} iters, ${eps} episodes each)"
  local ckpts
  ckpts=$(docker exec "$CONTAINER" bash -c "ls ${rundir}model_*.pt 2>/dev/null | grep -v _bc_format" | tr -d '\r')
  for c in $ckpts; do
    local n; n=$(basename "$c" .pt | sed 's/model_//')
    [ $((n % every)) -ne 0 ] && continue
    local out="/workspace/crane_testbed/logs/sim_eval/sweep_iter${n}"
    docker exec "$CONTAINER" bash -c "test -f ${out}/.done" 2>/dev/null && { echo "  [skip] iter $n"; continue; }
    echo "  -- iter $n"
    docker exec "$CONTAINER" bash -c "cd /workspace/crane_testbed && \
      export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
      /workspace/isaaclab/_isaac_sim/python.sh scripts/envs/rsl_rl_to_scoring.py \
        --rsl_rl $c --template $CKPT" >/dev/null 2>&1 || { echo "     convert FAILED"; continue; }
    docker exec "$CONTAINER" bash -c "cd /workspace/crane_testbed && \
      export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
      mkdir -p $out && /workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
        --policy_type scoring --crop_margin 0.5 --checkpoint ${c%.pt}_bc_format.pt \
        --gaze --raw_pcd --crop_to_bounds --headless --num_envs 8 --num_episodes $eps \
        --seed 42 --save_metrics --save_decisions $PLATFORM_V2 \
        --output_dir $out > ${out}.log 2>&1 && touch ${out}/.done"
  done
  say "ranking (TRUE columns are the citable ones)"
  docker exec "$CONTAINER" bash -c "cd /workspace/crane_testbed && \
    python3 scripts/envs/summarize_eval_rows.py --dir logs/sim_eval \
      --rows \$(ls -d logs/sim_eval/sweep_iter*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  echo
  echo "Compare against the P2c BC init (78.07 TRUE success, 23.09 cycles, 96% full clears)."
  echo "If no checkpoint beats it, that is the result - report it rather than hunting seeds."
}

# ---------------------------------------------------------------- fetch
# Rented instances are usually WIPED on terminate, and logs/ is gitignored so nothing comes
# home on its own. Run this BEFORE stopping the pod.
fetch () {
  local stamp="${1:-run}"
  local tar="/workspace/crane_testbed/logs/cloud_${stamp}.tgz"
  say "packing artifacts"
  docker exec "$CONTAINER" bash -c "cd /workspace/crane_testbed && tar czf $tar \
    logs/rsl_rl/crane_scoring_ppo_v2 \
    logs/reward_audit \
    logs/p3v2_train.log \
    logs/sim_eval/sweep_iter* \
    logs/ladder_*.log 2>/dev/null; ls -lh $tar"
  echo
  echo "Copy it home BEFORE terminating the instance:"
  echo "  rsync -avz <pod>:${tar} ~/IsaacLab/crane_testbed/logs/"
  echo
  echo "Then on the laptop:  tar xzf logs/cloud_${stamp}.tgz -C ."
}

case "${1:-}" in
  preflight) preflight ;;
  setup)     setup ;;
  ladder)    shift; ladder "$@" ;;
  train)     shift; train "$@" ;;
  sweep)     shift; sweep "$@" ;;
  fetch)     shift; fetch "$@" ;;
  *) sed -n '2,16p' "$0" ;;
esac
