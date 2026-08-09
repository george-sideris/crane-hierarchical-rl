# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project Summary

Autonomous log-pile clearing with a forestry forwarder crane. A high-level policy selects grasp
targets (5D) from point clouds; a fixed FSM controller executes each pick-place cycle. Sim
(Isaac Lab "gaze" env) transfers to a real instrumented crane (ROS2, ZED X). Masters thesis +
paper work. Methods compared: deployed heuristic baseline, BC (regression and per-point scoring
heads), BC+RL fine-tuning.

## Machine roles

- **Laptop** (primary): real-crane deployment (fpi_crane_ros2), collections, the authoritative
  logs/ tree, real bags. Anything involving "the crane" or real clouds lives there.
- **Other PC** (if you are reading this from a fresh clone of RLCraneTestbed, branch
  `master` - there is no `dev` branch on that remote): sim-only
  jobs - RL training, eval sweeps, collections. Real data only if rsync'd over.

## Running commands

Everything runs in the Isaac Lab docker container with this repo at `/workspace/crane_testbed`
and Isaac at `/workspace/isaaclab`.

```bash
# docker exec does NOT source .bashrc - ALWAYS set this first (a missing PYTHONPATH
# once killed a training launch silently):
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs

# Platform-v2 physics flags - FROZEN for comparability, use on every sim run:
#   --profile_piles --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0
#   --gripper_effort 2000 --num_logs 200

# BC+RL training (gen-1 init committed in-repo):
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-Gaze-CosSin-Raw-MR-v0 \
    --bc_checkpoint logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy_rsl_rl.pt \
    --freeze_encoder --sigma_init 0.05 --seed 42 --num_envs 4 --max_iterations 400 \
    --headless <platform-v2 flags>

# BC->RL REBUILD (2026-08-09) - this is the run to make on the other PC. See the
# "BC->RL rebuild" section below for WHY each flag is there; do not drop any of them.
/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-Gaze-Scoring-PPO-v2 \
    --bc_checkpoint logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt \
    --freeze_encoder --sigma_init 0.05 \
    --critic_warmup_iters 25 --anneal_sigma_iters 100 --anneal_sigma_to 0.01 \
    --seed 42 --num_envs 4 --max_iterations 400 --headless \
    agent.num_steps_per_env=8 <platform-v2 flags> \
    > logs/p3v2_train.log 2>&1
# num_envs x num_steps_per_env IS the PPO batch. 4x8=32 is what the two FAILED attempts
# used - raise --num_envs to whatever the GPU fits (8 -> 64, 16 -> 128) before raising
# num_steps_per_env, because envs cost wall-clock once and steps cost it every iteration.

# Eval (see eval_scripts/ for the sweep runners - ALWAYS use run scripts, never paste
# long one-liners; terminal hard-wrap has eaten launches):
/workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
    --policy_type {bc|heuristic|scoring} --checkpoint <pt> --gaze --raw_pcd --crop_to_bounds \
    --headless --num_envs 8 --num_episodes 10 --seed 42 --save_metrics --save_decisions \
    <platform-v2 flags>

# Plain-torch scripts (train_scoring_head.py, validators) need Isaac's interpreter -
# the system python3's torch may be built for a newer CUDA than the driver and silently
# runs on CPU (the trainer hard-fails on this):
/workspace/isaaclab/_isaac_sim/python.sh scripts/envs/train_scoring_head.py ...
```

No test suite or linter; pure research codebase.

## Architecture

- **Observation**: raw (unsegmented) point cloud in crane base frame; tight crop = the action
  box; `--crop_margin` widens the OBSERVATION crop (bed/rails/poles enter the cloud as learned
  negatives) while the action box stays fixed.
- **Action**: 5D `[x, y, z, cos(2yaw), sin(2yaw)]`, arctanh-encoded into the action box
  x[-5.364,-3.364] y[-1.684,5.316] z[-1.30,0.10]. 4D direct-yaw saturates - do not use.
- **Bed floor**: bed_z -1.30 + margin 0.10 -> every policy type clamps at z = -1.20.
- **Dig**: real heuristic targets surface-0.25/0.30; sim expert labels log CENTERS
  (surface-0.056). `--train_label_dig 0.25` converts at train time; scoring ckpts carry `dig`.
- **FSM phases**: GAZE -> ALIGN_YAW -> DESCEND -> CLOSE -> LIFT_HIGH -> (despawn) -> loop.
- **BC+RL**: `train.py --bc_checkpoint` loads encoder+actor (`strict=False`), critic random,
  small `--sigma_init` to preserve the BC mean.

### Core files

| File | Role |
|---|---|
| `scripts/envs/crane_rl_env_gaze.py` | THE env (~5500 lines): gaze camera, FSM, rewards, platform-v2 physics, ZED noise model, accounting counters |
| `scripts/envs/train_bc_pointcloud.py` | collection + BC training; `--save_raw_cap` stores PRE-FPS raw clouds (fp16) so num_points/crop/margin are post-hoc choices |
| `scripts/envs/scoring_head.py` | per-point scoring policy + support gate (see below) |
| `scripts/envs/play_bc_pointcloud.py` | sim eval; heuristic row = the DEPLOYED baseline imported verbatim (replay-validated bridge) |
| `scripts/envs/summarize_noise_sweep.py` | sweep tables; warns on mixed metric versions |
| `fpi_crane_ros2/fpi_crane_rl/` | vendored deployment node + policy_loader (sim eval imports from here - keep in tree) |
| `source/crane_testbed/.../tasks.py` | gym task registrations (`...Gaze-CosSin-Raw-MR-v0` primary; `-CC-v0` adds cycle_cost) |
| `eval_scripts/` | sweep/collection/chain run scripts |

### Scoring head (why it exists)

Regression BC under MSE answers bimodal piles with the MEAN of two mounds = the empty gap
(observed on the real crane 2026-08-03: 27% zero-support targets). The scoring head scores
every observed point and grasps at the argmax - mode-averaging is unrepresentable. Its own
weakness: a single noise-displaced point can win the argmax (93->50% grasp success at 1x ZED
noise). Fix: the SUPPORT GATE (default on, `support_frac 0.25`) excludes low-support points
from selection. The gate is processing, not learning - on real clouds it moved 0/26 deployed-
heuristic targets; the LEARNED claim is discrimination (rack vs logs), which needs the
margin dataset (structure as negatives).

## Metrics - critical

`eval_config.metrics_version: 2` = clearing/c95/throughput/full-clear from the MEASURED
post-despawn rack count (`logs_in_rack` in decisions npz). Version 1 (older rows) used the
cumulative grasp tally - it can exceed 100% (216/200 observed) and its c95 is optimistic.
NEVER mix v1 and v2 on those metrics. Grasp success, stability, alignment, cycles are
comparable across both. `accounting_mismatch_cycles > 0` in eval_config means that row's
clearing numbers are suspect.

## Traps (each of these has cost real time)

- All parsers use `parse_known_args`: a mistyped flag is SILENTLY ignored. grep the parser
  before trusting a new flag.
- NEVER copy code trees; symlink. A stale February copy of the env once silently collected a
  20k-sample garbage dataset.
- FPS at collection time is a one-way door - derive datasets from `pointclouds_raw.npy`, not
  from FPS'd files.
- Checkpoints carry `num_points`; the cloud pipeline must match (deployment auto-adopts).
- Collections have NO `--seed` flag yet: same config on two machines = overlapping piles.
- ZED noise: collections are CLEAN; noise is train-time augmentation (`--zed_noise`,
  sigma = 0.0014*z^2 = the characterised real sensor; eval sweeps use multiples of it).
- PhysX: high env counts corrupt the scene (6400 rigid bodies); 8 envs is the proven max on
  8 GB for eval/collection, 4 for RL training.
- Commit style: plain ASCII (no em dashes/box-drawing/emoji) and never any AI-assistant
  attribution or Co-Authored-By in commits/PRs/docs.

## Policy zoo (committed under logs/ as force-added exceptions; logs/ is otherwise ignored)

| checkpoint | what |
|---|---|
| `logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy.pt` | regression BC, tight crop, 1024 pts, dig 0.25 |
| `logs/bc_pointcloud/bc_aug1v2_dig25/bc_pointcloud_policy_rsl_rl.pt` | same in RSL-RL format = BCRL init |
| `logs/bc_pointcloud/scoring_v1/scoring_policy.pt` | scoring head, SAME dataset as the BC above (controlled head comparison) |
| `logs/bc_pointcloud/scoring_margin05_2048_c/scoring_policy.pt` | **P2c** - margin 0.5, 2048 pts, stub_aug 0.7 + stub_neg 1.0. The deployed/headline policy and the BC->RL init. 100-ep sim: 78.1% TRUE success, 23.1 cycles, 96% full clears |
| `logs/rsl_rl/.../model_100_bc_format.pt` | BCRL-CC 100 iters, platform-v2 |

Datasets are not in git; rsync from the laptop: `logs/bc_pointcloud/bc_policy_aug1_v2/`
(tight 1024 + full-scene 2048) and `logs/bc_pointcloud/bc_margin05_2048/` (margin 0.5,
2048 pts, + pre-FPS raw).

## BC->RL rebuild (2026-08-09)

Two scoring-head fine-tunes (entropy 0.003, then 0) both DEGRADED P2c, with the signature
"training reward rises while deployed argmax performance falls". Four causes were found; all
four are fixed in the repo, and `-PPO-v2` exists so the old settings stay reproducible.

1. **The reward paid for grasps that clear nothing.** `logs_grasped` is counted by proximity +
   per-log rise at line ~5075 of `crane_rl_env_gaze.py`, but `_despawn_grasped_logs` is gated on
   a SEPARATE lift-height check. Cycles passing the first and failing the second removed nothing
   yet were still paid - 12-18% of cycles, measured. That is a reward-hacking channel and it
   alone explains the signature. Fixed by `cfg.reward_requires_lift` (default True). Reported
   metrics keep the ungated count so older rows stay comparable.
2. **32 transitions per update** (8 steps x 4 envs), minibatch 8, against a 2048-way categorical.
   The old sweep script measured ~10% batch variance against a ~3% signal.
3. **Checkpoints selected on training reward** while the policy DEPLOYS argmax - with (1) that
   selects the most reward-hacked policy. `save_interval` is now 10; pick by argmax eval after
   the fact, never by the reward curve.
4. **Critic initialised random against an already-good actor**, so the first updates apply
   garbage advantages to a prior worth protecting. `--critic_warmup_iters 25` trains the critic
   alone first, then restores each parameter's ORIGINAL trainability (so `--freeze_encoder`
   survives the unfreeze).

`--anneal_sigma_iters/--anneal_sigma_to` closes the train-stochastic/deploy-argmax gap.

**Guardrail:** every cycle appends to `logs/reward_audit/audit_<pid>.jsonl` (reward, raw vs
lift-gated grasp count, rack count). After ~200 cycles, check reward correlates with actual rack
decrease. If it does not, kill the run - that is reward hacking, not learning, and it is what
cost the two previous attempts.

**Scratch RL is NOT plateaued at 150 iterations.** The paper-era 1000-iteration run was at its
WORST between iterations 75-150 (mean reward 76) and then climbed to 183 by iteration ~680.
Judging a scratch run before ~400 iterations reproduces exactly the wrong conclusion. Also watch
`Mean episode length`: `max_grasp_cycles` is 30, so ~29.5 means the pile is never being cleared
regardless of what the reward says.

## Standing questions

1. Does training WITH structure in the cloud (margin dataset) stop the rack/end-board
   mis-targeting that dominates real failures (22% of cycles, endgame-concentrated)?
2. Does RL fine-tuning beat its own BC init under a FAIR test (same platform, BC-initialised,
   matched clean rows), and degrade slower as observations degrade? STILL UNRESOLVED, but the
   two prior "no" results are now explained rather than trusted: the reward was payable without
   clearing anything (see "BC->RL rebuild"). `-PPO-v2` is the fair retest; it has never run.
3. Can a full-scene (no-crop) policy remove the rack-calibration dependency?

`docs/CLAUDE.md` has extended paper-era context (reward details, PPO hyperparameters); treat
its file references as historical where they conflict with this file.

## Thesis

The master's thesis lives in `docs/` (root `thesis_main.tex`, 8 chapters, build with
`pdflatex + bibtex + pdflatex x2` from that directory). Chapter map, notation conventions,
the two stability definitions, the algorithm blocks and the figure provenance are documented
in `docs/CLAUDE.md` - read that before editing any chapter, because the notation was unified
across chapters and it is easy to reintroduce a symbol collision.

Built artifacts (`thesis_main.pdf`, `thesis_overleaf.zip`, LaTeX aux files) are gitignored.

## IsaacLab-side changes

The surrounding IsaacLab checkout has its own local edits (`docker/Dockerfile.base`,
`docker/Dockerfile.ros2`, `source/isaaclab/setup.py`). That checkout's only remote is upstream
`isaac-sim/IsaacLab`, so the edits cannot be pushed with it. They are exported to
`isaaclab_patches/` - apply them after cloning IsaacLab on a new machine.
