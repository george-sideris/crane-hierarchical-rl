# BC→RL Pipeline Architecture Diagram Reference

Reference for creating the pipeline figure (TODO at line 429 of `iros2026_draft_new.tex`).

## Overview

Three panels side by side: **BC Training** → **RL Fine-Tuning** → **Inference**.
Use snowflake for frozen, flame for trainable.

---

## Panel 1: Behavior Cloning (Stage 1)

```
┌─────────────────────────────────────────────┐
│           BEHAVIOR CLONING (Stage 1)        │
│                                             │
│  ┌───────────────────┐                      │
│  │  Expert Heuristic │  (privileged poses)  │
│  │  top-of-pile      │                      │
│  │  targeting        │──── a* ∈ ℝ⁵         │
│  └───────────────────┘     │                │
│                            │  MSE Loss      │
│                            ▼                │
│  Raw PCD                ┌─────┐             │
│  o_t ∈ ℝ^{1024×3}      │  L  │             │
│      │                  └──┬──┘             │
│      ▼                     ▲                │
│  ┌──────────────────┐      │                │
│  │ PointNet Encoder │ 🔥   │                │
│  │ per-point MLP:   │      │                │
│  │ 3→64→128→256     │      │                │
│  │ BN + ELU         │      │                │
│  │ max-pool → FC    │      │                │
│  │ → ℝ²⁵⁶           │      │                │
│  └────────┬─────────┘      │                │
│           │ z ∈ ℝ²⁵⁶       │                │
│           ▼                │                │
│  ┌──────────────────┐      │                │
│  │   Actor MLP      │ 🔥   │                │
│  │ 256→128→64→5     │      │                │
│  │ ELU activations  │──────┘                │
│  │ dropout=0.2      │  ã_t ∈ ℝ⁵            │
│  └──────────────────┘                       │
│                                             │
│  Loss: L_BC = ||μ_θ(z) - a*||²             │
│  Optimizer: AdamW (lr=1e-4, wd=1e-4)       │
│  Dataset: ~20,700 successful grasps         │
│  Best model: epoch 190, val MSE=0.101       │
└─────────────────────────────────────────────┘
```

**Key details:**
- Entire network trained end-to-end (encoder + actor MLP)
- Only successful grasps from heuristic (g > 0) used as training data
- Expert actions: 5D (x, y, z, cos2ψ, sin2ψ) with tanh pre-squashing
- Dropout (0.2) applied after encoder output during training
- No critic in BC stage
- Early stopping: patience 30 epochs

---

## Panel 2: RL Fine-Tuning (Asymmetric Actor-Critic)

```
┌──────────────────────────────────────────────────────────────┐
│              RL FINE-TUNING (Stage 2)                        │
│                                                              │
│        ACTOR (vision)              CRITIC (privileged)       │
│                                                              │
│  Raw PCD                     Privileged State                │
│  o_t ∈ ℝ^{1024×3}           s_t ∈ ℝ¹²⁸                    │
│      │                       (top-32 log poses,              │
│      ▼                        sorted by height)              │
│  ┌──────────────────┐            │                           │
│  │ PointNet Encoder │ ❄          ▼                           │
│  │ (init from BC)   │    ┌──────────────────┐                │
│  │ 3→64→128→256     │    │  Critic Encoder  │ 🔥             │
│  │ BN locked (eval) │    │  128→256→256     │                │
│  │ max-pool+FC→ℝ²⁵⁶ │    │  ELU             │                │
│  └────────┬─────────┘    └────────┬─────────┘                │
│           │ z                     │ ℝ²⁵⁶                    │
│           ▼                       ▼                          │
│  ┌──────────────────┐    ┌──────────────────┐                │
│  │   Actor MLP      │ 🔥 │   Critic MLP     │ 🔥             │
│  │ (init from BC)   │    │ 256→128→64→1     │                │
│  │ 256→128→64→5     │    │ ELU              │                │
│  │ ELU              │    │ → V(s) scalar    │                │
│  └────────┬─────────┘    └────────┬─────────┘                │
│           │ μ_θ(z)                │ V_φ(s)                   │
│           ▼                       │                          │
│  ┌──────────────────┐             │                          │
│  │ Gaussian Policy  │             │                          │
│  │ π = N(μ, σ²I)   │             │                          │
│  │ σ₀ = 0.05  🔥    │             │                          │
│  └────────┬─────────┘             │                          │
│           │ a_t                   │                          │
│           ▼                       ▼                          │
│       ┌───────────────────────────────┐                      │
│       │          PPO Update           │                      │
│       │  ε=0.2, γ=0.99, λ=0.95       │                      │
│       │  lr=3e-4, KL target=0.01      │                      │
│       │  grad clip=1.0                │                      │
│       └───────────────────────────────┘                      │
│                                                              │
│  Reward: r = g · α · ς  (success), r = -1 (failure)         │
│  Frozen: PointNet encoder (weights + BN running stats)       │
│  Trainable: actor MLP, σ, entire critic                      │
│  No gradient path: critic → encoder                          │
└──────────────────────────────────────────────────────────────┘
```

**Key details:**
- **Asymmetry**: actor sees PCD, critic sees privileged 128D pose vector
- **Frozen encoder**: all weights locked, BatchNorm in eval mode (running stats frozen)
- **Why freeze**: critic is state-based, no gradient flows through encoder; unfrozen encoder drifts without constraint and causes training collapse
- **σ₀ = 0.05** (vs 1.0 from scratch): small perturbations around BC mean
- **Critic architecture**: encoder (128→256→256, ELU) + MLP head (256→128→64→1, ELU)
- Actor MLP initialized from BC weights; critic initialized randomly
- Optimizer starts fresh (no momentum carried from BC)

---

## Panel 3: Inference (Deployment)

```
┌─────────────────────────────────────────────┐
│            INFERENCE (Stage 3)              │
│                                             │
│  Raw PCD                                    │
│  o_t ∈ ℝ^{1024×3}                          │
│      │                                      │
│      ▼                                      │
│  ┌──────────────────┐                       │
│  │ PointNet Encoder │ ❄                     │
│  │ (from BC)        │                       │
│  │ 3→64→128→256     │                       │
│  │ max-pool+FC→ℝ²⁵⁶ │                       │
│  └────────┬─────────┘                       │
│           │ z                               │
│           ▼                                 │
│  ┌──────────────────┐                       │
│  │   Actor MLP      │ ❄                     │
│  │ (from RL)        │                       │
│  │ 256→128→64→5     │                       │
│  └────────┬─────────┘                       │
│           │ μ_θ(z) (deterministic)          │
│           ▼                                 │
│  ┌──────────────────┐                       │
│  │  tanh squash +   │                       │
│  │  workspace scale │                       │
│  └────────┬─────────┘                       │
│           │                                 │
│           ▼                                 │
│  (x, y, z, ψ) grasp target                 │
│           │                                 │
│           ▼                                 │
│  ┌──────────────────┐                       │
│  │   FSM Controller │                       │
│  │ HOVER → ALIGN →  │                       │
│  │ DESCEND → CLOSE  │                       │
│  │ → LIFT           │                       │
│  └──────────────────┘                       │
│                                             │
│  Critic discarded.                          │
│  Deterministic: a_t = μ_θ(z), no noise.    │
└─────────────────────────────────────────────┘
```

**Key details:**
- Mean action only (no sampling, σ discarded)
- 5D output → 3D position via tanh + workspace scaling, yaw via atan2(sin2ψ, cos2ψ)/2
- FSM controller is fixed (not learned), handles IK + PD servo + gripper

---

## Summary Table (for diagram legend)

| Component        | BC (Stage 1)  | RL (Stage 2)            | Inference (Stage 3) |
|------------------|---------------|-------------------------|---------------------|
| PointNet Encoder | 🔥 Train       | ❄ Frozen (from BC)      | ❄ Frozen             |
| Actor MLP        | 🔥 Train       | 🔥 Train (init from BC)  | ❄ Frozen (from RL)   |
| σ (noise)        | —             | 🔥 Train (init 0.05)     | — (deterministic)   |
| Critic Encoder   | —             | 🔥 Train (random init)   | — (discarded)       |
| Critic MLP       | —             | 🔥 Train (random init)   | — (discarded)       |

---

## Exact Dimensions (from code)

Source: `crane_testbed/source/crane_testbed/crane_testbed/agents/pointnet_actor_critic.py`

- **Input PCD**: 1024 points × 3 (XYZ in crane base frame), FPS-downsampled
- **PointNet per-point MLP**: Linear(3,64)→BN→ELU → Linear(64,128)→BN→ELU → Linear(128,256)→BN→ELU
- **PointNet aggregation**: max-pool across 1024 points → ℝ²⁵⁶
- **PointNet FC projection**: Linear(256,256)→BN→ELU → ℝ²⁵⁶ (post-pool, same dim)
- **Actor MLP**: Linear(256,128)→ELU → Linear(128,64)→ELU → Linear(64,5)
- **Critic encoder (RL, asymmetric)**: Linear(128,256)→ELU → Linear(256,256)→ELU
- **Critic MLP (RL)**: Linear(256,128)→ELU → Linear(128,64)→ELU → Linear(64,1)
- **Full critic path**: 128 → 256 → 256 → 128 → 64 → 1
- **Action**: 5D → (x, y, z) position + (cos2ψ, sin2ψ) yaw
- **Privileged state**: 32 logs × 4 (x, y, z, ψ) = 128D, sorted by height
