"""Standalone: render the sim basemast (zed_0) camera at the GAZE pose and save its RGB.

Mirrors the real gaze view (calibration/out/real_gaze_zed0_rgb.png) for the calibration slides.
Run inside the Isaac container:
  /workspace/isaaclab/isaaclab.sh -p /workspace/crane_testbed/calibration/capture_sim_gaze_rgb.py
Saves:
  calibration/out/sim_gaze_zed0_rgb.png    (basemast RGB at the gaze pose)
  calibration/out/sim_gaze_zed0_depth.png  (colorized depth, bonus)
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3, help="full FSM step() calls before capture (settle at gaze)")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import os, sys
import numpy as np
import torch
sys.path.insert(0, "/workspace/crane_testbed/scripts/envs")
from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull

OUT = "/workspace/crane_testbed/calibration/out"
os.makedirs(OUT, exist_ok=True)

cfg = CraneDirectEnvCfgFull()
cfg.scene.num_envs = 1
cfg.enable_camera = True
env = CraneDirectEnvFull(cfg)
env.reset()
env._compute_action_space_bounds()

zero = torch.zeros((env.num_envs, int(env.cfg.action_space)), device=env.device)
for k in range(args_cli.steps):
    env.step(zero)
    print(f"[capture] step {k+1}/{args_cli.steps} done")

# force a full render pass so RGB is populated at the settled gaze pose
for _ in range(3):
    env.sim.render()

out = env._camera.data.output
rgb = out["rgb"][0].detach().cpu().numpy()          # (H, W, 3 or 4) uint8
if rgb.shape[-1] == 4:
    rgb = rgb[..., :3]
import cv2
cv2.imwrite(f"{OUT}/sim_gaze_zed0_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print(f"[capture] saved sim_gaze_zed0_rgb.png  {rgb.shape[1]}x{rgb.shape[0]}")

if "depth" in out:
    d = out["depth"][0].detach().cpu().numpy().squeeze()
    finite = np.isfinite(d) & (d > 0)
    if finite.any():
        lo, hi = np.percentile(d[finite], 2), np.percentile(d[finite], 98)
        dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        dn[~finite] = 0
        dc = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        cv2.imwrite(f"{OUT}/sim_gaze_zed0_depth.png", dc)
        print(f"[capture] saved sim_gaze_zed0_depth.png")

# sim point cloud DIRECTLY in the crane BASE frame (optical->base, same as the real ZED cloud)
pc_b = env.get_pointcloud_base(0, max_points=30000, depth_range=(0.5, 12.0))
if pc_b.shape[0] > 0:
    np.save(f"{OUT}/sim_full_pcd.npy", pc_b.detach().cpu().numpy())
    print(f"[capture] saved sim_full_pcd.npy  {pc_b.shape[0]} pts (base frame)"
          f"  zed_noise={getattr(env.cfg, 'zed_noise', False)}")
else:
    print("[capture] WARN: empty sim cloud (camera/depth?)")

env.close()
simulation_app.close()
