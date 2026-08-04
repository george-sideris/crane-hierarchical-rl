#!/usr/bin/env python3
"""Probe: create the gaze env with --log_scale_jitter and print the ACTUAL spawned log prim
scales from the USD stage. Ground-truth check that MultiAssetSpawnerCfg variants take effect
(the viewport makes a 2 cm diameter difference hard to judge by eye).

Run: ./isaaclab.sh -p probe_log_scales.py --gaze --profile_piles --log_scale_jitter 0.2 \
         --spacing_y 0.115 --spacing_z 0.10 --rows 60 --headless ... (same flags as collection)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull

cfg = CraneDirectEnvCfgFull()
cfg.scene.num_envs = 2
cfg.enable_camera = True
cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
cfg.camera_cfg.width = 640
cfg.camera_cfg.height = 360

print(f"[PROBE] scene.replicate_physics = {cfg.scene.replicate_physics}")
env = CraneDirectEnvFull(cfg)

import isaacsim.core.utils.prims as prim_utils
from pxr import UsdGeom, Usd
import omni.usd
from collections import Counter

stage = omni.usd.get_context().get_stage()
paths = prim_utils.find_matching_prim_paths("/World/envs/env_.*/Rack/LogsAnchor/Origin.*/Log")
print(f"[PROBE] matched {len(paths)} log prims")
scales = []
for p in paths[:600]:
    prim = stage.GetPrimAtPath(p)
    xf = UsdGeom.Xformable(prim)
    s = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s = tuple(round(float(v), 3) for v in op.Get())
    scales.append(s)
cnt = Counter(scales)
print(f"[PROBE] distinct scales: {len(cnt)}")
for s, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:10]:
    print(f"  scale={s}  count={n}")
env.close()
simulation_app.close()
