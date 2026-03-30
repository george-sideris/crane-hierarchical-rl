# IROS 2026 Supplementary Video — Assembly Guide

Total duration: ~2:00

---

## Slide 1 — Title Card (5s)

**Title:** Clearing Dense Log Piles with Imitation-Bootstrapped Reinforcement Learning
**Subtitle:** IROS 2026 — Supplementary Video

Source: static text slide (no video)

---

## Slide 2 — Task Overview (20s)

**Label:** Task: Sequential pile clearing (200 logs)

**Caption:**
A log loader crane removes 200 cylindrical logs from a rack one at a time.
At each step, the policy selects a grasp target from a raw point cloud.
A fixed finite-state machine (FSM) executes the pick-and-place sequence.

Source: `media/bcrl_overview.mp4`, first 40s at 2x

---

## Slide 3 — Observation Pipeline (15s)

**Label:** Observation Pipeline (all methods)

**Caption:**
Each step: 1024 points sampled from depth camera -> FPS downsampling
-> coordinate transform to crane frame -> policy input.
Shown: raw point cloud evolving as logs are removed from the pile.

Source: pipeline step PNGs (3s each), located at:
```
results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/paper_viz/pipeline_step_0000.png
results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/paper_viz/pipeline_step_0005.png
results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/paper_viz/pipeline_step_0010.png
results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/paper_viz/pipeline_step_0015.png
results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/paper_viz/pipeline_step_0020.png
```

---

## Slide 3a — Architecture Diagram (8s)

**Label:** BC -> RL Pipeline Architecture

**Caption:**
(a) PointNet encoder + actor trained on expert demos with MSE loss.
(b) Encoder frozen; actor fine-tuned with PPO + asymmetric state-based critic.
(c) At inference, critic discarded; encoder + actor run deterministically.

Source: `media/architecture_diagram.png`

---

## Slide 3b — Reward Function (5s)

**Label:** Reward Function

**Caption:**
Reward per grasp: R = n_grasped x alignment x stability. Alignment measures grapple-to-log angular match (top-down). Stability measures grapple levelness after lift (side view).

Source: `media/reward_diagram.png` (exists)

---

## Slide 3c — Training Curves (8s)

**Label:** Training Reward Curves

**Caption:**
Three observation types tested for from-scratch RL: privileged pose, segmented point cloud, and raw point cloud. All plateau well below BC->RL (red), which starts from the BC-pretrained encoder.

Source: `media/training_curves.png`

---

## Slide 4a — RL from Scratch (10s)

**Label:** From-scratch RL (Raw Point Cloud)

**Caption:**
RL trained from random initialization with three observation types: privileged pose (73% clearing), segmented PC (53%), and raw PC (53%). None approach BC or BC->RL performance.

Source: `media/rl_overview.mp4`, 10s from middle (~60s-70s)

---

## Slide 4b — Behavior Cloning (10s)

**Label:** Behavior Cloning (Raw Point Cloud)

**Caption:**
Trained on ~20k demonstrations from a heuristic expert that targets
the highest log using privileged pose state. Achieves 98.5% clearing
directly from unsegmented point clouds.

Source: `media/bc_overview.mp4`, 10s (~30s-40s)

---

## Slide 4c — BC->RL (10s)

**Label:** BC -> RL (Raw Point Cloud)

**Caption:**
BC-initialized policy fine-tuned with PPO: 98.2% clearing, 86.3% success.
BC->RL consistently achieves grasps that result in a level grapple post-lift.

Source: `media/bcrl_overview.mp4`, 10s from later section (~60s-70s)

---

## Slide 5 — Stability Comparison (15s)

**Left label:** BC Policy
**Right label:** BC -> RL Policy

**Caption:**
Side view: BC grapple frequently tilts during lift (stability 0.824).
BC -> RL consistently achieves level grapple post-lift (stability 0.947).

Source: split screen
- Left: `media/bc_sideview_20260308_021705.mp4` (BC)
- Right: **BCRL sideview needed** — no bcrl_sideview video found in media/
  - Candidates in ~/Videos/: `bc_stability.webm`, `rl_stability.webm`
  - May need to record a new BCRL sideview capture

---

## Slide 6 — Results Summary (5s)

Source: `media/results_table.png` (7-row table with all obs variants, metric legend, and bolding note)

No overlay text needed — the PNG contains the complete table and metric legend.

---

## Asset Checklist

| Asset | Status | Path |
|-------|--------|------|
| Architecture diagram PNG | OK | `media/architecture_diagram.png` |
| Reward diagram PNG | OK | `media/reward_diagram.png` |
| Training curves PNG | OK | `media/training_curves.png` |
| Results table PNG | OK | `media/results_table.png` |
| Pipeline step PNGs (5x) | OK | `results/BCRL_.../paper_viz/pipeline_step_00{00,05,10,15,20}.png` |
| BCRL overview video | OK | `media/bcrl_overview.mp4` |
| RL overview video | OK | `media/rl_overview.mp4` |
| BC overview video | OK | `media/bc_overview.mp4` |
| BC sideview video | OK | `media/bc_sideview_20260308_021705.mp4` |
| BCRL sideview video | MISSING | Need to record or locate |
