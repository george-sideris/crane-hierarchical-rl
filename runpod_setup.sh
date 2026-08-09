#!/bin/bash
# Bring a RunPod GPU pod from first login to a running BC->RL fine-tune.
#
# UNTESTED ON A POD - written from cloud_setup.sh and the working laptop setup, so treat the
# first run as a dry run and expect to fix a step. Every stage is separately re-runnable.
#
#   ./runpod_setup.sh preflight    # is this pod usable (RTX? isaaclab visible? volume sane?)
#   ./runpod_setup.sh bootstrap    # clone crane_testbed onto persistent storage
#   ./runpod_setup.sh ladder       # measure the REAL env-count ceiling on this card
#   ./runpod_setup.sh train N      # launch the PPO-v2 fine-tune with N envs
#   ./runpod_setup.sh sweep 50 10  # argmax-eval every 50th checkpoint, 10 eps, and rank them
#   ./runpod_setup.sh fetch NAME   # tar the artifacts - RUN BEFORE TERMINATING THE POD
#
# WHY THIS EXISTS SEPARATELY FROM cloud_setup.sh
# RunPod is a container provider: you land INSIDE the GPU container, so there is no docker
# daemon and nothing to `docker exec` into. Every command here runs in the current shell.
# cloud_setup.sh remains the VM-provider path (Lambda, GCP) and is unchanged. There is no
# `setup` verb here because you cannot build the image from inside it - see POD IMAGE below.
#
# POD IMAGE: bring an image that ALREADY has Isaac Lab installed. Either push your local
# isaac-lab-base to a registry, or try NVIDIA's prebuilt one first:
#   docker run --rm --gpus all nvcr.io/nvidia/isaac-lab:latest python -c "import isaaclab"
# Installing Isaac Lab in-pod costs 30-60 min of billed GPU time and walks straight into the
# `packaging` conflict that isaaclab_patches/ exists to fix.
#
# STORAGE - THE ONE THAT SILENTLY EATS POD-MINUTES: RunPod mounts both the volume disk and a
# network volume at /workspace, and a network volume REPLACES whatever the image had there.
# Isaac Lab lives at /workspace/isaaclab, so a volume at the default mount path HIDES it and
# every command below dies at launch. Set Volume Mount Path to /data when creating the pod.
# preflight detects this. Also note the container disk is wiped on stop/restart and editing a
# running pod erases anything outside the volume, so artifacts must live on /data.
#
# ENV: set ACCEPT_EULA=Y and PRIVACY_CONSENT=Y as pod environment variables. Nothing reads
# docker/.env.base here, and Isaac Sim will block on the EULA without them.
#
# HARD REQUIREMENT: an RTX card. Isaac Sim renders through RTX/Vulkan ray tracing and this env
# renders a camera to build its observation, so A100/H100 (no RT cores) will fail with
# "Your GPUs do not support RayTracing". On RunPod that means A40, L40S, RTX 4090, or
# RTX 6000 Ada.
set -u

ISAACLAB_DIR="${ISAACLAB_DIR:-/workspace/isaaclab}"
# Persistent storage. /data is the mount path preflight expects; falls back to /workspace so
# the script still runs on a volume-less pod (everything is then lost on stop - use fetch).
if [ -z "${PERSIST:-}" ]; then
  if [ -d /data ] && [ -w /data ]; then PERSIST=/data; else PERSIST=/workspace; fi
fi
CRANE_DIR="${CRANE_DIR:-$PERSIST/crane_testbed}"
REPO_URL="${REPO_URL:-git@github.com:george-sideris/RLCraneTestbed.git}"
# Platform-v2 physics flags - FROZEN for comparability with every existing row.
PLATFORM_V2="--profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 \
--log_ang_damping 3.0 --gripper_effort 2000 --num_logs 200"
CKPT="$CRANE_DIR/logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt"
PYPATH="$CRANE_DIR/source/crane_testbed:$CRANE_DIR/scripts/envs"

say () { echo -e "\n=== $* "; }

# Run a crane command in the pod, from the repo root, with PYTHONPATH set. Replaces the
# `docker exec "$CONTAINER" bash -c` wrapper that cloud_setup.sh uses.
crane () { ( cd "$CRANE_DIR" && export PYTHONPATH="$PYPATH" PYTHONUNBUFFERED=1 && eval "$*" ); }

# ---------------------------------------------------------------- preflight
preflight () {
  local ok=0
  say "GPU"
  if ! command -v nvidia-smi >/dev/null; then echo "  NO nvidia-smi - no driver"; return 1; fi
  local name; name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  local vram; vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
  echo "  $name  ($vram)"
  if echo "$name" | grep -qiE "A100|H100|V100|K80|P100"; then
    echo "  !! This is a datacenter card with NO RT cores. Isaac Sim rendering will FAIL."
    echo "  !! Pick A40 / L40S / RTX 4090 / RTX 6000 Ada instead."
    ok=1
  fi

  say "container provider sanity"
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    echo "  a working docker daemon is visible - this looks like a VM, not a pod."
    echo "  cloud_setup.sh is the better script for this machine."
  else
    echo "  inside a container, no docker daemon (expected on RunPod)"
  fi

  say "Isaac Lab at $ISAACLAB_DIR"
  if [ -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo "  ok"
  else
    echo "  !! isaaclab.sh NOT FOUND."
    if grep -q " /workspace " /proc/mounts 2>/dev/null; then
      echo "  !! /workspace is a MOUNT POINT, so a volume is shadowing the image's Isaac Lab."
      echo "  !! This is the documented RunPod default and it is the cause. Recreate the pod"
      echo "  !! with Volume Mount Path = /data (it cannot be changed on a running pod)."
    else
      echo "  !! /workspace is not a mount, so the image itself has no Isaac Lab."
      echo "  !! Use an Isaac Lab image, not a bare Isaac Sim one."
    fi
    ok=1
  fi

  say "persistent storage"
  echo "  using PERSIST=$PERSIST"
  if [ "$PERSIST" = "/workspace" ]; then
    echo "  !! No /data volume. The container disk is WIPED on stop/restart and on any pod"
    echo "  !! edit, so a long run can vanish. Attach a volume at /data, or accept the risk"
    echo "  !! and run fetch often."
    ok=1
  fi
  if [ -d "$PERSIST" ]; then df -h "$PERSIST" | tail -1; else echo "  !! $PERSIST does not exist"; ok=1; fi

  say "EULA"
  if [ "${ACCEPT_EULA:-}" = "Y" ]; then echo "  ACCEPT_EULA=Y"; else
    echo "  !! ACCEPT_EULA is not Y - Isaac Sim will refuse to start."
    echo "  !! Set ACCEPT_EULA=Y and PRIVACY_CONSENT=Y in the pod environment variables."
    ok=1
  fi

  say "crane_testbed at $CRANE_DIR"
  if [ -f "$CKPT" ]; then echo "  repo + BC checkpoint present"
  elif [ -d "$CRANE_DIR" ]; then echo "  !! repo present but BC checkpoint missing at $CKPT"; ok=1
  else echo "  not cloned yet - run: $0 bootstrap"; ok=1; fi

  [ $ok -eq 0 ] && echo -e "\npreflight OK" || echo -e "\npreflight FAILED - fix the above first"
  return $ok
}

# ---------------------------------------------------------------- bootstrap
# Replaces cloud_setup.sh's `setup`. No image build, no IsaacLab clone, no patching: the pod
# image already carries all three. This only puts the repo on persistent storage.
bootstrap () {
  if [ ! -d "$CRANE_DIR" ]; then
    say "cloning crane_testbed to $CRANE_DIR (assets + P2c checkpoint travel with it, ~1.7 GB)"
    git clone "$REPO_URL" "$CRANE_DIR" || {
      echo "  clone failed. If this is an SSH-key problem, either add a deploy key to the pod"
      echo "  or re-run with an https REPO_URL and a token."
      return 1; }
  else
    say "crane_testbed already at $CRANE_DIR"
  fi

  say "verifying the toolchain can import isaaclab"
  if "$ISAACLAB_DIR/isaaclab.sh" -p -c "import isaaclab; print('isaaclab ok')" 2>&1 | tail -3; then
    :
  else
    echo "  !! import failed - the pod image is not a working Isaac Lab image"
    return 1
  fi

  [ -f "$CKPT" ] && echo "  BC checkpoint present" || echo "  !! BC checkpoint MISSING at $CKPT"
  mkdir -p "$CRANE_DIR/logs"
  echo "  bootstrap done - next: $0 ladder"
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
    local log="$CRANE_DIR/logs/ladder_${n}.log"
    crane "$ISAACLAB_DIR/isaaclab.sh -p scripts/rsl_rl/train.py \
        --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2 --bc_checkpoint $CKPT \
        --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs $n --max_iterations 3 \
        --headless $PLATFORM_V2 agent.num_steps_per_env=8 > $log 2>&1" &
    local pid=$!
    local peak=0 t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "$secs" ]; do
      local m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$m" -gt "$peak" ] && peak=$m
      kill -0 "$pid" 2>/dev/null || break
      sleep 10
    done
    # The wrapper shell is $pid; the python process under it needs killing too.
    pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
    pkill -f "num_envs $n " 2>/dev/null
    wait "$pid" 2>/dev/null
    sleep 5
    local c res
    c=$(grep -c CYCLE "$log" 2>/dev/null || echo 0)
    if grep -qi 'state is corrupted\|corrupted' "$log" 2>/dev/null; then
      res="PHYSX CORRUPTED -> raise PhysxCfg buffers"
    elif grep -qi 'out of memory\|OUT_OF_DEVICE_MEMORY' "$log" 2>/dev/null; then
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
  crane "nohup $ISAACLAB_DIR/isaaclab.sh -p scripts/rsl_rl/train.py \
      --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2 --bc_checkpoint $CKPT \
      --freeze_encoder --sigma_init 0.05 \
      --critic_warmup_iters 25 --anneal_sigma_iters 100 --anneal_sigma_to 0.01 \
      --seed 42 --num_envs $n --max_iterations 400 --headless \
      $PLATFORM_V2 agent.num_steps_per_env=8 \
      > logs/p3v2_train.log 2>&1 &"
  sleep 90
  grep -c CYCLE "$CRANE_DIR/logs/p3v2_train.log" 2>/dev/null
  cat <<NOTE

Detached, but NOT immune to the pod dying: on RunPod a stop, a restart, or any edit to the
running pod kills this and wipes anything outside $PERSIST. Run fetch periodically, not just
at the end.

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
  rundir=$(ls -dt "$CRANE_DIR"/logs/rsl_rl/crane_scoring_ppo_v2/*/ 2>/dev/null | head -1)
  [ -z "$rundir" ] && { echo "no crane_scoring_ppo_v2 run found"; return 1; }
  say "argmax sweep over $rundir (every ${every} iters, ${eps} episodes each)"
  local ckpts
  ckpts=$(ls "${rundir}"model_*.pt 2>/dev/null | grep -v _bc_format)
  for c in $ckpts; do
    local n; n=$(basename "$c" .pt | sed 's/model_//')
    [ $((n % every)) -ne 0 ] && continue
    local out="$CRANE_DIR/logs/sim_eval/sweep_iter${n}"
    [ -f "${out}/.done" ] && { echo "  [skip] iter $n"; continue; }
    echo "  -- iter $n"
    crane "$ISAACLAB_DIR/_isaac_sim/python.sh scripts/envs/rsl_rl_to_scoring.py \
        --rsl_rl $c --template $CKPT" >/dev/null 2>&1 || { echo "     convert FAILED"; continue; }
    mkdir -p "$out"
    crane "$ISAACLAB_DIR/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
        --policy_type scoring --crop_margin 0.5 --checkpoint ${c%.pt}_bc_format.pt \
        --gaze --raw_pcd --crop_to_bounds --headless --num_envs 8 --num_episodes $eps \
        --seed 42 --save_metrics --save_decisions $PLATFORM_V2 \
        --output_dir $out > ${out}.log 2>&1" && touch "${out}/.done"
  done
  say "ranking (TRUE columns are the citable ones)"
  crane "python3 scripts/envs/summarize_eval_rows.py --dir logs/sim_eval \
      --rows \$(ls -d logs/sim_eval/sweep_iter*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
  echo
  echo "Compare against the P2c BC init (78.07 TRUE success, 23.09 cycles, 96% full clears)."
  echo "If no checkpoint beats it, that is the result - report it rather than hunting seeds."
}

# ---------------------------------------------------------------- fetch
# Pods are WIPED on terminate, and logs/ is gitignored so nothing comes home on its own.
# Run this BEFORE stopping the pod - and periodically during a long run.
fetch () {
  local stamp="${1:-run}"
  local tar="$PERSIST/cloud_${stamp}.tgz"
  say "packing artifacts"
  ( cd "$CRANE_DIR" && tar czf "$tar" \
    logs/rsl_rl/crane_scoring_ppo_v2 \
    logs/reward_audit \
    logs/p3v2_train.log \
    logs/sim_eval/sweep_iter* \
    logs/ladder_*.log 2>/dev/null; ls -lh "$tar" )
  echo
  echo "Get it off the pod BEFORE terminating. Either:"
  echo "  runpodctl send $tar          # prints a one-time code; on the laptop: runpodctl receive CODE"
  echo "or, if you exposed SSH on the pod:"
  echo "  rsync -avz <pod>:${tar} ~/IsaacLab/crane_testbed/logs/"
  echo
  echo "Then on the laptop:  tar xzf logs/cloud_${stamp}.tgz -C ."
}

case "${1:-}" in
  preflight) preflight ;;
  bootstrap) bootstrap ;;
  ladder)    shift; ladder "$@" ;;
  train)     shift; train "$@" ;;
  sweep)     shift; sweep "$@" ;;
  fetch)     shift; fetch "$@" ;;
  *) sed -n '2,16p' "$0" ;;
esac
