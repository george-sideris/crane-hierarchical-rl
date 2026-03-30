#!/bin/bash
# Run all 5 remaining obs noise evals sequentially
# Each uses: --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics
set -e

cd /workspace/isaaclab

echo "=========================================="
echo "Starting BC sigma=0.05 eval"
echo "=========================================="
./isaaclab.sh -p /workspace/crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint /workspace/crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --raw_pcd --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics \
  --obs_noise 0.05

echo "=========================================="
echo "Starting BC sigma=0.1 eval"
echo "=========================================="
./isaaclab.sh -p /workspace/crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint /workspace/crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --raw_pcd --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics \
  --obs_noise 0.1

echo "=========================================="
echo "Starting BC sigma=0.2 eval"
echo "=========================================="
./isaaclab.sh -p /workspace/crane_testbed/scripts/envs/play_bc_pointcloud.py \
  --checkpoint /workspace/crane_testbed/results/BC_Raw_PCD/bc_pointcloud_policy.pt \
  --raw_pcd --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics \
  --obs_noise 0.2

echo "=========================================="
echo "Starting BC->RL sigma=0.05 eval"
echo "=========================================="
./isaaclab.sh -p /workspace/crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --load_run 2026-02-27_04-21-30 --checkpoint model_360.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics \
  --obs_noise 0.05

echo "=========================================="
echo "Starting BC->RL sigma=0.2 eval"
echo "=========================================="
./isaaclab.sh -p /workspace/crane_testbed/scripts/rsl_rl/play.py \
  --task Isaac-Crane-PointCloud-CosSin-Raw-MR-Asym-v0 \
  --load_run 2026-02-27_04-21-30 --checkpoint model_360.pt \
  --seed 42 --num_envs 20 --num_episodes 100 --headless --save_metrics \
  --obs_noise 0.2

echo "=========================================="
echo "All 5 noise evals complete!"
echo "=========================================="
