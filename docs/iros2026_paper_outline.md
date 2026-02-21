# IROS 2026 Paper Outline

## Working Title
**"Learning Grasp Target Selection for Forestry Crane Manipulation via Hierarchical Reinforcement Learning"**

Alternative titles:
- "Hierarchical BC+RL for Log Pile Grasping with Forestry Cranes"
- "From Heuristics to Learning: Target Selection for Crane-Based Log Manipulation"

---

## Abstract (150 words)

Forestry cranes must efficiently grasp logs from cluttered piles while maintaining pile stability. Traditional approaches rely on hand-crafted heuristics that struggle to generalize across varying pile configurations. We present a hierarchical reinforcement learning framework that decomposes the task into (1) a learned target selection policy and (2) a finite state machine for low-level grapple control. Our approach uses behavioral cloning to initialize the policy from a heuristic expert, followed by RL fine-tuning with a normalized reward that accounts for local grasp opportunity. Experiments in a high-fidelity Isaac Lab simulation demonstrate that our BC+RL approach outperforms the expert heuristic by X% in grasp success rate and Y% in throughput, while generalizing across pile sizes from 50-200 logs. We provide ablation studies on reward formulation and domain randomization, offering insights for reward shaping in manipulation tasks with variable object densities.

---

## 1. Introduction (1.5 pages)

### Motivation
- Forestry industry context: log handling is labor-intensive, dangerous, increasingly automated
- Challenge: grasping from unstructured log piles requires reasoning about:
  - Which logs are accessible (not buried)
  - Grapple-log alignment (parallel grasp = clean lift)
  - Pile stability (disturbing pile = cascading failures)
- Current industrial practice: teleoperation or simple heuristics

### Problem Statement
- Given: RGB-D observation of log pile, crane kinematic state
- Goal: Select grasp target (x, y, z, yaw) that maximizes logs grasped while maintaining good alignment
- Challenges: variable pile density, partially occluded logs, continuous action space

### Contributions
1. **Hierarchical framework** decomposing crane grasping into learned target selection + FSM execution
2. **BC+RL training pipeline** that outperforms hand-crafted heuristics
3. **Normalized reward formulation** enabling generalization across pile densities
4. **Systematic ablation study** on reward shaping (multiplicative vs additive, raw vs normalized)
5. **Open-source simulation environment** in NVIDIA Isaac Lab

### Paper Organization
Brief roadmap of sections

---

## 2. Related Work (1 page)

### 2.1 Forestry Automation
- Existing work on forestry crane control
- Harvester head manipulation
- Cite: La Hera et al., Ortiz Morales et al., Lindroos et al.

### 2.2 Robotic Grasping in Clutter
- Bin picking literature
- Grasp point detection (GPD, GraspNet, etc.)
- Differs from our work: we reason about which object, not just where to grasp

### 2.3 Hierarchical Reinforcement Learning
- Options framework, feudal RL
- Task decomposition in manipulation
- Our contribution: specific decomposition for crane grasping

### 2.4 Imitation Learning + RL
- BC, DAgger, BC+RL fine-tuning
- Residual policy learning
- Our approach: BC initialization + PPO fine-tuning

### 2.5 Reward Shaping for Manipulation
- Sparse vs dense rewards
- Curriculum learning
- Our contribution: normalization for variable-density scenarios

---

## 3. Problem Formulation (0.75 pages)

### 3.1 Task Description
- Crane configuration: 6-DOF hydraulic arm + rotating grapple
- Log pile: N cylindrical logs in rack, randomly oriented
- Episode: Clear pile via repeated grasp-lift-deposit cycles
- Success metrics: grasp success rate, throughput (logs/grasp), alignment, clearing percentage

### 3.2 Observation Space
- Pile state: positions and orientations of visible logs (top surface)
- Crane state: joint positions, grapple pose
- History: previous grasp outcomes (optional)

### 3.3 Action Space
- 4D continuous: target position (x, y, z) + yaw angle
- Interpreted as grapple descent target in base frame

### 3.4 Reward Formulation
- Grasp evaluation: count logs within proximity radius after lift
- Alignment: average |dot(grapple_forward, log_forward)| over grasped logs
- Reward variants explored in ablation (Section 5)

---

## 4. Method (2 pages)

### 4.1 Hierarchical Decomposition

```
┌─────────────────────────────────────────────────┐
│           High-Level Policy (Learned)           │
│   Observation → Target Selection (x,y,z,yaw)    │
└─────────────────────┬───────────────────────────┘
                      │ target pose
                      ▼
┌─────────────────────────────────────────────────┐
│         Low-Level FSM (Hand-designed)           │
│  HOVER → DESCEND → CLOSE → LIFT → DEPOSIT       │
└─────────────────────────────────────────────────┘
```

**Rationale**:
- Low-level control is well-understood (position control, gripper actuation)
- High-level target selection is where intelligence matters
- Reduces RL sample complexity significantly

### 4.2 Low-Level Finite State Machine
- HOVER_UP: Move grapple above target XY position
- HOVER_OVER: Align yaw with target orientation
- DESCEND: Lower grapple to target Z height
- CLOSE_GRAPPLE: Actuate gripper fingers
- LIFT_HIGH: Raise grapple with payload
- DEPOSIT: Move to deposit zone, open grapple
- Each phase uses proportional control to joint targets

### 4.3 High-Level Target Selection Policy

**Network Architecture**:
- Input: 45D observation (pile features + crane state)
- Hidden: [256, 128, 64] with ELU activations
- Output: 4D action (x, y, z, yaw) with tanh squashing

**Observation Features**:
- Per-log: relative position, orientation, height ranking, accessibility score
- Global: pile centroid, log count, action bounds
- Crane: current grapple pose, joint positions

### 4.4 Training Pipeline

#### Stage 1: Behavioral Cloning
- Expert heuristic: select highest accessible log + compute optimal yaw alignment
- Collect N episodes of (observation, expert_action) pairs
- Filter to successful grasps only
- Train via supervised MSE loss

#### Stage 2: RL Fine-tuning
- Initialize policy from BC weights
- PPO with normalized reward (Section 4.5)
- Domain randomization: pile sizes 50-200 logs
- Training: 500 iterations, 4096 steps/iteration

### 4.5 Reward Formulation

**Normalized Efficiency Reward** (recommended):
```
efficiency = logs_grasped / min(logs_available, max_capacity)
reward = efficiency × 10 × alignment + bonus(alignment)
```

Where:
- `logs_available`: logs in cylinder below grapple at hover
- `max_capacity = 20`: physical grapple limit
- `bonus(a) = efficiency × 10 × (a - 0.7) × 5` if alignment > 0.7

**Key insight**: Normalizing by local opportunity prevents reward scale from varying with pile density

---

## 5. Experiments (2.5 pages)

### 5.1 Experimental Setup

**Simulation Environment**:
- NVIDIA Isaac Lab (Isaac Sim 4.x)
- PhysX 5 rigid body dynamics
- 200 logs per pile (default), 50-200 with DR
- Log dimensions: 3m length, 0.3m diameter
- 32 parallel environments for training

**Baselines**:
| Method | Description |
|--------|-------------|
| Heuristic | Select highest log + optimal yaw alignment |
| BC | Behavioral cloning from heuristic (no RL) |
| Pure RL | PPO from scratch (no BC init) |
| BC+RL | BC initialization + PPO fine-tuning (ours) |

**Metrics**:
| Metric | Description |
|--------|-------------|
| Grasp Success Rate | % of grasps with ≥1 log |
| Throughput | Average logs per successful grasp |
| Alignment | Average grapple-log orientation similarity |
| Clearing % | % of pile cleared per episode |
| Full Clear Rate | % of episodes clearing entire pile |

### 5.2 Main Results

**Table 1: Method Comparison (200-log piles, no DR)**

| Method | Grasp Success ↑ | Throughput ↑ | Alignment ↑ | Clearing ↑ |
|--------|-----------------|--------------|-------------|------------|
| Heuristic | 91.3% | 10.3 | 0.64 | 95.6% |
| BC | 81.5% | 7.4 | 0.65 | 94.9% |
| Pure RL (MR) | X% | X | X | X% |
| BC+RL (ours) | **Y%** | **Y** | **Y** | **Y%** |

**Table 2: Generalization with Domain Randomization (50-200 logs)**

| Method | Grasp Success ↑ | Throughput ↑ | Clearing ↑ |
|--------|-----------------|--------------|------------|
| Heuristic | X% | X | X% |
| BC+RL (MN) | **Y%** | **Y** | **Y%** |

### 5.3 Reward Ablation Study

**Table 3: Reward Formulation Comparison**

| Variant | Formula | Normalization | Final Reward | Grasp Success |
|---------|---------|---------------|--------------|---------------|
| MR | n × a + bonus | Raw | 99.5 | X% |
| MN | η × 10 × a + bonus | Normalized | 71.4 | X% |
| AR | n/20 + a | Raw | 17.2 | X% |
| AN | η + a | Normalized | 22.3 | X% |

**Key Findings**:
- Multiplicative rewards converge faster than additive
- Normalization essential for domain randomization
- Alignment bonus (a > 0.7) improves grasp quality

**Figure: Learning curves for reward variants (TensorBoard plots)**

### 5.4 BC Data Scaling

**Table 4: Effect of Demonstration Data Amount**

| Episodes | Samples | BC Val Loss | BC+RL Grasp Success |
|----------|---------|-------------|---------------------|
| 100 | ~1,500 | X | X% |
| 500 | ~7,500 | X | X% |
| 1000 | ~15,000 | X | X% |

### 5.5 Ablation Studies

**Hierarchy Ablation**: Compare hierarchical (FSM + learned) vs end-to-end RL

**BC Initialization Ablation**: BC+RL vs pure RL from scratch

**Noise Injection**: Effect of position/yaw noise during BC collection

### 5.6 Qualitative Analysis

**Figure: Visualization of grasp selection**
- Heuristic vs BC+RL target selection on same pile
- Show cases where BC+RL makes better decisions

**Failure Mode Analysis**:
- When does BC+RL fail? (edge cases, sparse piles, etc.)

---

## 6. Discussion (0.5 pages)

### Limitations
- Simulation only (no real robot validation yet)
- Simplified log geometry (uniform cylinders)
- No visual input (assumes known log poses)

### Future Work
- Sim-to-real transfer with domain randomization
- Visual observations (point cloud or RGB-D)
- Multi-crane coordination
- Deformable/varying log shapes

---

## 7. Conclusion (0.25 pages)

We presented a hierarchical RL framework for forestry crane log grasping that decomposes the task into learned target selection and FSM execution. Our BC+RL approach, trained with a normalized reward formulation, outperforms hand-crafted heuristics while generalizing across variable pile densities. Ablation studies reveal design principles for reward shaping in manipulation tasks with variable object counts.

---

## Figures Needed

1. **System overview** - Crane + log pile + hierarchical decomposition diagram
2. **FSM diagram** - State machine phases with transitions
3. **Method pipeline** - BC collection → training → RL fine-tuning
4. **Reward formulation** - Visual explanation of alignment + efficiency
5. **Learning curves** - Reward vs iterations for different variants
6. **Qualitative results** - Screenshots of successful grasps
7. **Failure cases** - When/why methods fail

---

## Experiments To Run (Priority Order)

### Week 1: Core Results
1. [ ] Train better BC (500 episodes, with noise)
2. [ ] BC+RL fine-tuning with MN reward
3. [ ] Evaluate BC+RL vs heuristic (main result)
4. [ ] Run all variants through standardized evaluation

### Week 2: Ablations + Writing
5. [ ] BC data scaling experiment (100/500/1000 episodes)
6. [ ] Pure RL baseline (if time permits)
7. [ ] Generate figures and tables
8. [ ] Write first draft

### Week 3: Polish
9. [ ] Revise based on feedback
10. [ ] Final experiments if gaps identified
11. [ ] Camera-ready formatting

---

## Key Claims to Support

| Claim | Evidence Needed |
|-------|-----------------|
| BC+RL beats heuristic | Table 1: ≥5% improvement on key metrics |
| Normalization enables DR generalization | Table 2: MN works with variable piles |
| Hierarchy reduces sample complexity | Ablation: hierarchical vs end-to-end |
| BC initialization helps | Ablation: BC+RL vs pure RL |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| BC+RL doesn't beat heuristic | Medium | More BC data, longer training, tune hyperparams |
| Not enough time for all experiments | Medium | Prioritize main result + 1-2 ablations |
| Reviewers want real robot | High | Acknowledge as limitation, emphasize sim fidelity |

---

## Related Work References (to gather)

- Forestry automation: La Hera, Ortiz Morales, Lindroos, Ringdahl
- Grasping in clutter: Mahler (Dex-Net), Fang (GraspNet), ten Pas (GPD)
- Hierarchical RL: Sutton (options), Vezhnevets (feudal), Nachum (HIRO)
- BC+RL: Ross (DAgger), Rajeswaran (demo-augmented), Nair (AWAC)
- Isaac Lab/Sim: Mittal et al., Makoviychuk et al.
