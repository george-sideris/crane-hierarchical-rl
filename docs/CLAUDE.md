# Crane Testbed - Project Context for Claude

This document provides full context for working on the crane_testbed project. It covers the project structure, architecture, current state, and active work items so a new Claude session can pick up where the previous one left off.

## Project Overview

This is an **IROS 2026 submission** for learning grasp targets in a forestry log-pile clearing task. The paper is `docs/iros2026_draft.tex`. The deadline is imminent and results are being collected now.

The core idea: a crane grapple must repeatedly pick logs from a rack of up to 200 logs in frictional contact. We compare four methods:
1. **Heuristic expert** (built-in FSM targets top log with optimal yaw)
2. **Pure RL** (PPO from scratch)
3. **Behavioral Cloning (BC)** from heuristic demonstrations
4. **BC+RL** (BC-initialized RL fine-tuning) -- **the headline result**

**Narrative (updated):** "Raw point clouds (no segmentation) are sufficient for learning sequential grasp targeting. BC provides coverage; BC+RL combines coverage with per-grasp optimization to achieve the best overall performance. RL from scratch fails regardless of observation type."

Key findings:
- **BC+RL** is the strongest method: 87.0% grasp success, 0.950 stability, 98.3% clearing
- **Raw (unsegmented) PCD** works as well as segmented PCD — no semantic segmentation needed
- **RL from scratch fails** at ~58-60% clearing regardless of obs type (pose, seg PCD, raw PCD) or architecture (sym, asym)
- **BC** achieves 98.6% clearing from raw PCD alone

## Environment Architecture

### Grasp-and-Remove MDP

Each env step = one complete pick attempt. The policy outputs a grasp target, a fixed FSM executes approach/lift, logs within proximity of the grapple after lifting are despawned.

- **State**: Point cloud (primary) or privileged per-object poses (ablation)
- **Action**: 5D `[x, y, z, cos(2*yaw), sin(2*yaw)]` in crane base frame (cosine-sine encoding following Morrison et al.)
- **Action decode**: `xyz = min + (tanh(a) + 1)/2 * (max - min)`, `yaw = atan2(tanh(a4), tanh(a3)) / 2`
- **Reward**: Per-grasp only (no episode completion bonus). Multiplicative: `g * a * s * scale` where g=throughput, a=alignment, s=stability
- **Episode**: Ends at 30 cycles or empty rack

### Key Files

| File | Purpose |
|------|---------|
| `scripts/envs/crane_rl_env_full.py` | Main environment (~5000 lines). FSM, reward, OOB check, all logic |
| `source/.../tasks.py` | Task registration. Maps gym IDs to env configs |
| `source/.../agents/rsl_rl_cfg.py` | PPO hyperparameters for all task variants |
| `source/.../agents/pointnet_actor_critic.py` | PointNet encoder + actor-critic for PCD tasks |
| `scripts/rsl_rl/train.py` | RL training (supports `--bc_checkpoint` for BC+RL) |
| `scripts/rsl_rl/play.py` | RL evaluation |
| `scripts/envs/play_bc_pointcloud.py` | BC evaluation |
| `scripts/envs/play_heuristic.py` | Heuristic evaluation |
| `scripts/envs/train_bc_pointcloud_cossin.py` | BC training (5D cossin actions) |
| `scripts/envs/collect_bc_data.py` | Expert data collection |
| `docs/iros2026_draft.tex` | IROS paper draft |
| `docs/references.bib` | Bibliography |

### Task IDs (the ones that matter)

The primary tasks used for the paper:

| Task ID | Obs | Action | Arch | Notes |
|---------|-----|--------|------|-------|
| `Isaac-Crane-PointCloud-CosSin-MR-v0` | Seg PCD 3072D | 5D | Sym | Segmented PCD, multiplicative reward |
| `Isaac-Crane-PointCloud-CosSin-Raw-MR-v0` | Raw PCD 3072D | 5D | Sym | **Primary for BC/RL**. Raw (unsegmented) PCD |
| `Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0` | Raw PCD 3072D | 5D | **Asym** | **Primary for BC+RL and RL ablation**. Asymmetric AC: PCD actor + 128D state critic |
| `Isaac-Crane-Full-CosSin-MR-v0` | Pose 128D | 5D | Sym | Privileged-pose ablation |
| `Isaac-Crane-PointCloud-CosSin-Raw-MR-ThroughputOnly-v0` | Raw PCD | 5D | Sym | Reward ablation: throughput only |
| `Isaac-Crane-PointCloud-CosSin-Raw-MR-NoAlign-v0` | Raw PCD | 5D | Sym | Reward ablation: no alignment |
| `Isaac-Crane-PointCloud-CosSin-Raw-MR-NoStab-v0` | Raw PCD | 5D | Sym | Reward ablation: no stability |

**MR = multiplicative raw reward** (`g * α * s * scale`). This is the baseline reward for all methods.

### Symmetric vs Asymmetric Architecture

- **Symmetric (Sym):** Actor and critic both use PointNet encoder on PCD input (3072D)
- **Asymmetric (Asym):** Actor uses PointNet on PCD, critic uses MLP on privileged 128D pose state. Env returns `{"policy": pcd, "critic": 128D_state}`.
- Asym tasks have `asymmetric_critic: bool = True` in their config.

### PointNet Architecture

```
Input: (batch, 1024, 3) point cloud in base frame
  -> Per-point MLP: Linear(3,64) -> BN -> ELU -> Linear(64,128) -> BN -> ELU -> Linear(128,256) -> BN -> ELU
  -> MaxPool across points -> (batch, 256)
  -> FC: Linear(256, 256) -> BN -> ELU
  -> Actor MLP: Linear(256,128) -> ELU -> Linear(128,64) -> ELU -> Linear(64, 5)
```

v1 uses BatchNorm, v2 uses LayerNorm + asymmetric critic (state-based critic, PCD actor).

### Point Cloud Pipeline

**Raw PCD (primary):** Depth camera -> back-project ALL depth pixels in range to 3D -> transform to crane base frame -> FPS to 1024 points -> PointNet encoder.

**Segmented PCD (ablation):** Depth camera -> semantic segmentation mask (log class only) -> back-project masked pixels to 3D -> transform to crane base frame -> FPS to 1024 points -> PointNet encoder.

Results show raw PCD matches or exceeds segmented PCD — segmentation provides no significant benefit.

### PPO Hyperparameters

lr=3e-4, clip=0.2, entropy=0.01, gamma=0.99, lambda=0.95, num_learning_epochs=5, mini_batches=4, schedule=adaptive, desired_kl=0.01, max_grad_norm=1.0, num_steps_per_env=4, activation=ELU, sigma_init=1.0.

### BC+RL Fine-Tuning

`train.py --bc_checkpoint <path>` loads BC encoder+actor weights with `strict=False`. Critic initializes randomly, optimizer starts fresh.

**Headline config (BCFinetune):** σ_init=0.05, lr=1e-4, ε=0.1, entropy=0, encoder_lr_scale=0.2, num_steps_per_env=8. Conservative to preserve BC knowledge.

**σ_init** is the initial std dev of the Gaussian policy noise. Higher σ = more exploration but risks overwriting the BC-learned policy. σ=0.05 preserves BC policy best; σ≥0.1 degrades performance.

### Action Bounds = OOB Bounds

`_action_bounds_min/max` (computed from rack geometry) serve double duty:
1. Policy action scaling: `tanh -> [min, max]`
2. Out-of-bounds check: logs outside these bounds are despawned before next target selection

This means the heuristic (which bypasses action scaling) is still a fair comparison -- any log it could reach outside action bounds would already be despawned.

### Evaluation Order (no double counting)

Each step: `_check_logs_out_of_bounds()` -> `_check_grasped_logs()` -> `_despawn_grasped_logs()`. OOB logs are deposited first, so they can't also be counted as grasped.

### Episode Reset Behavior

`_logs_knocked_off[i]` resets to 0 on episode reset. Eval scripts must capture it BEFORE the env resets. All three eval scripts now do this correctly by accumulating `total_knocked_off` per-episode in the done block.

## Evaluation Scripts - Consistent Metrics

All three eval scripts (`play_heuristic.py`, `play_bc_pointcloud.py`, `play.py`) now use identical metric collection:

### Per-Episode Tracking

Every metric is computed per-episode, then reported as **mean +/- std across episodes**. Per-env tensors track within each episode and reset on done:

- `ep_successful_grasps[i]`, `ep_failed_grasps[i]` -- grasp outcome counts
- `ep_alignment_sum[i]` -- sum of (alignment * logs_grasped), for weighted avg
- `ep_stability_sum[i]` -- sum of (stability * logs_grasped), for weighted avg
- `episode_logs_cleared[i]` -- total logs grasped this episode
- `episode_rewards[i]` -- cumulative reward

On episode done, per-episode values are computed and appended to lists:
- `per_ep_success_rates` -- n_success / n_total * 100
- `per_ep_throughputs` -- logs_cleared / n_success
- `per_ep_alignments` -- alignment_sum / logs_cleared (weighted by logs per grasp)
- `per_ep_stabilities` -- stability_sum / logs_cleared
- `knocked_off_per_episode` -- raw count, converted to % using starting_logs
- `clearing_percentages` -- logs_cleared / starting_logs * 100

### Console Output Format

```
[Eval] ====== RESULTS (N episodes) ======
[Eval] Episode Reward:      X.XX +/- X.XX
[Eval] Pile Cleared:        X.X +/- X.X%
[Eval] Full Clear Rate:     X.X% (n/N)
[Eval] Grasp Success Rate:  X.X +/- X.X%
[Eval] Throughput:          X.XX +/- X.XX logs/grasp
[Eval] Alignment:           X.XXX +/- X.XXX
[Eval] Stability:           X.XXX +/- X.XXX
[Eval] Knocked Off:         X.X +/- X.X%
[Eval] Avg Cycles:          X.X +/- X.X
[Eval] Cycles to 95%:       X.X +/- X.X
[Eval] Total Logs Grasped:  XXXX
```

### JSON Structure (identical across all scripts)

All three eval scripts (`play_heuristic.py`, `play_bc_pointcloud.py`, `play.py`) now produce identical JSON with `clearing_curves`, `cycles`, and `cycles_to_95pct` fields.

```json
{
  "eval_config": { "method": "...", "checkpoint": "...", ... },
  "episodes": { "total": N, "logs_per_pile_avg/min/max": ... },
  "summary": {
    "reward": {"mean": X, "std": X},
    "pile_cleared_pct": {"mean": X, "std": X},
    "full_clear_rate": X,
    "grasp_success_pct": {"mean": X, "std": X},
    "throughput": {"mean": X, "std": X},
    "alignment": {"mean": X, "std": X},
    "stability": {"mean": X, "std": X},
    "knocked_off_pct": {"mean": X, "std": X},
    "cycles": {"mean": X, "std": X},
    "cycles_to_95pct": {"mean": X, "std": X},
    "total_logs_grasped": X,
    "total_grasps": X,
    "total_successful_grasps": X,
    "total_failed_grasps": X
  },
  "per_episode": {
    "rewards": [...], "clearing_pcts": [...], "success_rates": [...],
    "throughputs": [...], "alignments": [...], "stabilities": [...],
    "knocked_off_counts": [...], "knocked_off_pcts": [...],
    "logs_cleared": [...], "logs_per_pile": [...],
    "cycles": [...], "cycles_to_95pct": [...],
    "clearing_curves": [[...], ...]
  }
}
```

**Note:** `clearing_curves` is a list of lists — each inner list is the clearing % after each grasp cycle for one episode. This is the data source for the clearing progression figure.

## Eval Commands

**Important**: 20 envs is the safe maximum. 32 envs causes PhysX `Scene state is corrupted` errors (6400 rigid bodies overwhelms GPU solver).

### BC+RL Training (the headline experiment)
```bash
# BC+RL with asymmetric AC, σ=0.05 (headline result)
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
    --num_envs 20 --headless \
    --bc_checkpoint <bc_checkpoint_rsl_rl.pt>
# Uses CranePPORunnerCfg_PointCloud_BCFinetune (σ=0.05, lr=1e-4, ε=0.1)

# BC+RL σ sweep (Ablation 3) — modify sigma_init in rsl_rl_cfg.py before running
# σ=0.1: change init_noise_std=0.1 in BCFinetune config
# σ=0.3: change init_noise_std=0.3 in BCFinetune config
```

### Pure RL Training (from scratch)
```bash
# RL (Raw PCD, symmetric) — main table fallback
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
    --num_envs 20 --headless
# No --bc_checkpoint → trains from scratch

# RL (Raw PCD, asymmetric) — main table + Ablation 1
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/train.py \
    --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
    --num_envs 20 --headless
```

### BC Eval (Raw PCD)
```bash
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
    --checkpoint <bc_checkpoint.pt> \
    --num_envs 20 --headless --save_metrics --raw_pcd
```

### RL Eval (Raw PCD)
```bash
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
    --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
    --num_envs 20 --headless --save_metrics
```

### BC+RL Eval (Raw PCD, Asym)
```bash
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
    --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
    --num_envs 20 --headless --save_metrics
```

### Pure RL (Pose Ablation) Eval
```bash
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
    --task Isaac-Crane-Full-CosSin-MR-v0 \
    --num_envs 20 --headless --save_metrics
```

### Heuristic Eval
```bash
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
    --num_envs 20 --num_episodes 50 --headless --save_metrics
```

## IsaacLab Setup

- IsaacLab version: v2.2.1, commit `0f00ca2b4b2`
- Isaac Sim: 5.0.0
- Docker-based setup (see `docker/docker-compose.yaml`)
- `crane_testbed/` is bind-mounted into the container

## Paper Status (iros2026_draft.tex)

### Narrative Pivot (DONE)
- **BC+RL is the headline result** (no longer future work)
- **Raw PCD is the primary pipeline** (no segmentation needed)
- **4-method comparison**: Heuristic / RL / BC / BC+RL
- Updated contributions:
  1. Grasp-and-remove crane env in Isaac Lab for dense pile clearing at scale (200 logs)
  2. BC→RL pipeline using PointNet on raw PCD with asymmetric actor-critic
  3. Four-method comparison showing RL fails, BC achieves near-expert, BC+RL improves per-grasp metrics

### Sections Updated
- Abstract, intro, related work → pivoted to raw PCD + BC+RL
- Observations → raw PCD primary, segmented as ablation
- Learning Methods → added BC+RL subsection (Sec V-D)
- Results → 4-row main table, BC+RL Performance subsection, ablation tables
- Discussion → BC+RL analysis, raw PCD sufficiency, removed "future work" framing
- Conclusion → 4-method comparison, BC+RL strongest, raw PCD sufficient
- Architecture Table → added BC+RL asymmetric column
- references.bib → added asymmetric AC citation (Pinto et al. 2018)

### TODOs in Paper
- Fill in RL (Raw PCD, asym) numbers when eval completes
- Generate clearing progression figure (CRITICAL)
- Generate σ sweep bar chart
- Insert all TODO figures
- Final number consistency check

## Experiment Matrix

### Main Table (4 rows, all on raw PCD)

| Method | Obs | Arch | Status | Clear% |
|--------|-----|------|--------|--------|
| Heuristic | Pose | — | Done | 97.9±4.4 |
| RL | Raw PCD | Asym | **Needs run** (fallback: sym ~58-60%) | TBD |
| BC | Raw PCD | — | Done | 98.6±2.2 |
| BC+RL | Raw PCD | Asym (σ=0.05) | Done (headline) | 98.3±1.6 |

### Ablation 1: Obs+Arch for RL (all show ~58-62% clearing)

| Obs | Arch | Status |
|-----|------|--------|
| Pose 128D | Sym | Done — extract numbers |
| Seg PCD | Sym | Done — 58.6% |
| Raw PCD | Sym | Running / done |
| Raw PCD | Asym | **Needs run** (= main table row) |

### Ablation 2: Reward shaping for RL (all Raw PCD, asym, P2 best-effort)

| Reward | Status |
|--------|--------|
| Multiplicative (g·α·s) | = main RL row |
| Throughput-only (g) | **Needs run** |
| Normalized | **Needs run** |
| Curriculum + multiplicative | **Needs run** |

### Ablation 3: BC+RL σ_init sweep (all Raw PCD, asym)

| σ_init | Status |
|--------|--------|
| 0.3 | **Needs run** |
| 0.1 | **Needs run** |
| 0.05 | Done (headline) |

### Inference-only robustness tests (no training)

- **Noise robustness:** Add `--obs_noise` flag, sweep σ_noise = {0, 0.01, 0.05, 0.1}
- **Pile size generalization:** Eval on {50, 100, 150, 200, 250} logs

## Common Pitfalls

1. **Wrong script**: `[BC-PointCloud]` prefix in output means you're running `train_bc_pointcloud.py`, not `play_bc_pointcloud.py`. Play scripts print `[Play]`.
2. **Action dim**: The cossin pipeline uses 5D actions. Non-cossin BC uses 4D. `play_bc_pointcloud.py` auto-detects from checkpoint.
3. **PhysX corruption at 32 envs**: Use 20 envs max.
4. **Debug action print**: `action=[min, max]` shows min/max across the 5D vector, not a 2D action.
5. **`_logs_knocked_off` resets on episode reset**: Must capture in done block, not at end of eval.
6. **Reward ablation**: `use_alignment_reward` and `use_stability_reward` config flags. Throughput is always the base.
7. **3D action ablation** = fix yaw AND x. Only learn y (left/right) and z (height). Actually 2D.

## Reward Function Details

Located in `crane_rl_env_full.py:_compute_grasp_reward()`. Multiplicative formula (default):

```
reward = throughput * alignment * stability * scale
```

Where:
- `throughput` = logs_grasped (raw count, or normalized by logs_available_at_target)
- `alignment` = mean of (dot_product^8) across grasped logs -- sharp penalty for yaw misalignment
- `stability` = grapple stability score (1.0 = stable lift)
- `scale` = 10.0 (default)

Config flags: `use_alignment_reward`, `use_stability_reward` (can disable for ablation, keeping throughput as base).

Penalties: `failure_penalty` (0 logs grasped), `empty_target_penalty`, `knocked_off_penalty * count`.

**r_clear (sparse completion bonus) is NOT used** despite being defined in config. Do not mention it in the paper.
