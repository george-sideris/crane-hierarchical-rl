"""Quick camera angle test — renders one frame from overview camera and saves as PNG.

Usage:
  PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH \
    ./isaaclab.sh -p /workspace/crane_testbed/scripts/test_camera_angle.py \
      --num_envs 4 --headless
"""
import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test camera angles")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "envs"))
from crane_rl_env_full import CraneDirectEnvFull, CraneDirectEnvCfgFull
from PIL import Image

cfg = CraneDirectEnvCfgFull()
cfg.record_video = True
cfg.enable_camera = True
cfg.scene.num_envs = args_cli.num_envs
if "rgb" not in cfg.camera_cfg.data_types:
    cfg.camera_cfg.data_types = list(cfg.camera_cfg.data_types) + ["rgb"]

print(f"[Camera] overview pos: {cfg.overview_camera_cfg.offset.pos}")
print(f"[Camera] overview rot: {cfg.overview_camera_cfg.offset.rot}")
print(f"[Camera] overview convention: {cfg.overview_camera_cfg.offset.convention}")

env = CraneDirectEnvFull(cfg, render_mode=None)
env.reset()

for _ in range(10):
    env.sim.step(render=True)
    env.scene.update(env.physics_dt)

out_dir = "/workspace/crane_testbed/media"
os.makedirs(out_dir, exist_ok=True)

if env._overview_camera is not None:
    env._overview_camera.update(env.physics_dt)
    rgb = env._overview_camera.data.output.get("rgb")
    if rgb is not None and rgb.shape[0] > 0:
        frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
        path = f"{out_dir}/camera_test_overview.png"
        Image.fromarray(frame).save(path)
        print(f"[OK] Overview saved -> {path} ({frame.shape[1]}x{frame.shape[0]})")

if env._camera is not None:
    env._camera.update(env.physics_dt)
    rgb = env._camera.data.output.get("rgb")
    if rgb is not None and rgb.shape[0] > 0:
        frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
        path = f"{out_dir}/camera_test_policy.png"
        Image.fromarray(frame).save(path)
        print(f"[OK] Policy saved -> {path} ({frame.shape[1]}x{frame.shape[0]})")

env.close()
simulation_app.close()
