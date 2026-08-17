# Source clouds for the observation-pipeline figure (thesis Fig 3.3)

`episode_001_cloud_quad_gaze.npz` holds the clouds behind
`docs/thesis/figures/episode_001_pipeline_quad_gaze.png`, so the figure can be restyled
without a simulator run. It carries `base_points`, `fps_points`, `bounds_min`, `bounds_max`,
`target`, `yaw`, `logs_grasped` and `step_idx`.

It lives here rather than next to the run because `logs/` is gitignored. Two earlier
versions of this figure could not be improved after the fact for exactly that reason: the
paper_viz directory was gone and only the PNG survived, so every restyle needed a fresh run.

## How the PNG was produced (2026-08-17)

Deployed margin scoring policy, the `p2c` row from `eval_scripts/absorption_cells.sh`:

```bash
docker exec isaac-lab-base bash -c 'cd /workspace/crane_testbed && \
export PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:/workspace/crane_testbed/scripts/envs && \
/workspace/isaaclab/isaaclab.sh -p scripts/envs/play_bc_pointcloud.py \
  --gaze --raw_pcd --crop_to_bounds --profile_piles --headless \
  --num_envs 1 --num_episodes 1 --seed 42 \
  --log_scale_mean 1.0 --log_scale_jitter 0.10 --log_ang_damping 3.0 \
  --gripper_effort 2000 --num_logs 200 \
  --policy_type scoring --crop_margin 0.5 \
  --checkpoint .../scoring_margin05_2048_c/scoring_policy.pt \
  --zed_noise --zed_axial_coeff 0.0014 \
  --paper_viz --viz_dir <out>'
```

Traps:

- `--crop_margin` is **0.5**, not 0.05. The `margin05` in the dataset and checkpoint names
  reads like 0.05, but every script in `eval_scripts/` uses 0.5, and the value has to match
  what the policy was trained with.
- The builder writes the file as `episode_001_pipeline_quad_single.png` (the default suffix
  of `save_paper_pipeline_quadrant_single`). The thesis references it as
  `..._quad_gaze.png`, so the copy into `docs/thesis/figures/` renames it.
- Panels (c) and (d) must share one height scale (`_shared_zlim` in `eval_viz.py`).
  Independent autoscaling put the same scene on two different colour scales, which is the
  defect the pre-2026-08-17 version of this figure shipped with.
