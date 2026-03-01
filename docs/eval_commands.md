# Remaining Eval Commands

Standard protocol: `--seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics`

Laptop 2 (Docker) prefix:
```
PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH
```
Laptop 1 prefix:
```
PYTHONPATH=crane_testbed/source/crane_testbed:$PYTHONPATH
```

---

## Group 1: RL Ablation Table (Table 2, lines 545-547)

### 1a. RL Seg PCD (already running)
```bash
./isaaclab.sh -p /workspace/crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-MR-v0 \
  --checkpoint /workspace/crane_testbed/results/RL_Seg_PCD/2026-02-22_11-02-21/model_530.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --headless --save_metrics
```

### 1b. RL Raw PCD (already running)
```bash
./isaaclab.sh -p /workspace/crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-v0 \
  --checkpoint /workspace/crane_testbed/results/RL_Raw_PCD/2026-02-26_23-18-20/model_650.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --headless --save_metrics
```

### 1c. RL Pose
```bash
./isaaclab.sh -p /workspace/crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-Full-CosSin-MR-v0 \
  --checkpoint /workspace/crane_testbed/results/RL_Pose/2026-02-23_20-19-49/model_940.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --headless --save_metrics
```

---

## Group 2: Re-run Nominal Evals (for clearing curves data)

Existing JSONs lack per-grasp `clearing_curves`. Re-run to get data for the clearing progression figure.

### 2a. Heuristic
```bash
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
  --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics
```

### 2b. BC Raw PCD
```bash
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics
```

### 2c. BC->RL
```bash
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --checkpoint crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --headless --save_metrics
```

---

## Group 3: Robustness Table (Table 3, lines 568-580)

### Heuristic
```bash
# Obs noise
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
  --seed 42 --num_envs 20 --num_episodes 100 --obs_noise 0.01 --headless --save_metrics

# DR
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
  --seed 42 --num_envs 20 --num_episodes 100 --domain_randomization --headless --save_metrics

# DR + obs noise
./isaaclab.sh -p crane_testbed/scripts/envs/play_heuristic.py \
  --seed 42 --num_envs 20 --num_episodes 100 --domain_randomization --obs_noise 0.01 --headless --save_metrics
```

### BC
```bash
# DR
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --domain_randomization --headless --save_metrics

# Obs noise
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --obs_noise 0.01 --headless --save_metrics

# DR + obs noise
./isaaclab.sh -p crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --domain_randomization --obs_noise 0.01 --headless --save_metrics
```

### BC->RL
```bash
# DR
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --checkpoint crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --domain_randomization --headless --save_metrics

# Obs noise
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --checkpoint crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --obs_noise 0.01 --headless --save_metrics

# DR + obs noise
./isaaclab.sh -p crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --checkpoint crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt \
  --num_envs 20 --num_episodes 100 --seed 42 --domain_randomization --obs_noise 0.01 --headless --save_metrics
```

---

## Summary

| # | Run | Fills | Status |
|---|-----|-------|--------|
| 1a | RL Seg PCD | Table 2 row 2 | Running |
| 1b | RL Raw PCD | Table 2 row 3 | Running |
| 1c | RL Pose | Table 2 row 1 + Table 1 RL row | TODO |
| 2a | Heuristic nominal | Clearing curves fig | TODO |
| 2b | BC nominal | Clearing curves fig | TODO |
| 2c | BCRL nominal | Clearing curves fig | TODO |
| 3.1-3 | Heuristic robust (x3) | Table 3 | TODO |
| 3.4-6 | BC robust (x3) | Table 3 | TODO |
| 3.7-9 | BCRL robust (x3) | Table 3 | TODO |

**Total: 15 runs** (2 running + 13 to go)

After evals: create clearing progression plot + per-remaining-logs plot, fill table numbers, expand robustness table to include all metrics.
