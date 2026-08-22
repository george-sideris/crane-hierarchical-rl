#!/usr/bin/env python3
"""Record the grapple pose through one grasp cycle, for the task-space figure.

The state machine of the simulation chapter is described in words; this logs what it
actually commands. One environment, one policy, a handful of cycles: at every control
step it stores the phase, the grapple reference point in the crane base frame, the
grapple yaw, and the tong opening. The companion plotter turns the trace into a figure.

The policy input is built here exactly as play_bc_pointcloud.py builds it (camera cloud
in the base frame, optional crop to the action box, then farthest-point sampling). The
env's own observation vector is state, not points, so it is not what the policy reads.

    python3 scripts/envs/log_fsm_trace.py --checkpoint <scoring_policy.pt> --cycles 3 \
        --raw_pcd --crop_to_bounds --crop_margin 0.5 --num_points 2048
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--out", type=str, default="logs/fsm_trace.npz")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--raw_pcd", action="store_true")
parser.add_argument("--crop_to_bounds", action="store_true")
parser.add_argument("--crop_margin", type=float, default=0.0)
parser.add_argument("--num_points", type=int, default=2048)
args_cli, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch
from crane_rl_env_gaze import CraneDirectEnvFull, CraneDirectEnvCfgFull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "fpi_crane_ros2", "fpi_crane_rl", "fpi_crane_rl"))
from policy_loader import ScoringHeadPolicy


def yaw_of(qw, qx, qy, qz):
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """FPS downsampling, identical to the evaluation path."""
    device = points.device
    N = points.shape[0]
    if N <= num_samples:
        if N == 0:
            return torch.zeros((num_samples, 3), device=device)
        return torch.cat([points, torch.zeros((num_samples - N, 3), device=device)], dim=0)
    idx = torch.zeros(num_samples, dtype=torch.long, device=device)
    dist = torch.full((N,), float("inf"), device=device)
    cur = torch.randint(0, N, (1,), device=device).item()
    for i in range(num_samples):
        idx[i] = cur
        d = torch.norm(points - points[cur:cur + 1], dim=1)
        dist = torch.minimum(dist, d)
        cur = torch.argmax(dist).item()
    return points[idx]


def get_cloud(env, env_idx, num_points, depth_range=(1.0, 10.0)):
    """Camera cloud in the crane base frame, cropped and sampled as at evaluation."""
    pc = env.get_pointcloud_base(env_idx, max_points=50000, depth_range=depth_range)
    if pc.shape[0] == 0:
        return torch.zeros((num_points, 3), device=env.device)
    if args_cli.crop_to_bounds and getattr(env, "_action_bounds_min", None) is not None:
        bmin = env._action_bounds_min[env_idx]
        bmax = env._action_bounds_max[env_idx]
        g = float(args_cli.crop_margin)
        m = ((pc[:, 0] >= bmin[0] - g) & (pc[:, 0] <= bmax[0] + g) &
             (pc[:, 1] >= bmin[1] - g) & (pc[:, 1] <= bmax[1] + g) &
             (pc[:, 2] >= bmin[2] - g) & (pc[:, 2] <= bmax[2]))
        pc = pc[m]
        if pc.shape[0] == 0:
            return torch.zeros((num_points, 3), device=env.device)
    return farthest_point_sampling(pc, num_points)


def main():
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = 1
    cfg.use_hierarchical_rl = True
    cfg.action_space = 5
    cfg.enable_camera = True
    cfg.camera_cfg.width, cfg.camera_cfg.height = 1280, 720
    cfg.camera_cfg.data_types = ["depth", "semantic_segmentation"]
    cfg.seed = 42
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.enable_stabilization = True
    env = CraneDirectEnvFull(cfg)

    obs, _ = env.reset()

    # action box from the env itself, so the policy decodes into the same frame it trained in
    if getattr(env, "_action_bounds_min", None) is not None:
        bmin = env._action_bounds_min[0].detach().cpu().numpy().astype(np.float32)
        bmax = env._action_bounds_max[0].detach().cpu().numpy().astype(np.float32)
    else:
        bmin = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
        bmax = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
    print(f"[FSM trace] action box {bmin} -> {bmax}", flush=True)

    # dig=0: the scoring head's dz already carries the digging depth (see play_bc_pointcloud.py)
    pol = ScoringHeadPolicy(args_cli.checkpoint, bmin, bmax, cossin=True,
                            device="cuda:0", dig=0.0)

    def enc(v, lo, hi):
        n = np.clip(2.0 * (v - lo) / (hi - lo) - 1.0, -0.999, 0.999)
        return float(np.arctanh(n))

    # The env attaches nothing by default; giving it a sink turns on the in-cycle recorder
    env._trace_sink = []
    # The policy's target is expressed in the crane base frame, so the logged grapple
    # pose has to be carried into the same frame before the two can be drawn together.
    _bp = env.crane.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    _bq = env.crane.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    _w, _x, _y, _z = _bq
    _R = np.array([
        [1 - 2*_y*_y - 2*_z*_z, 2*_x*_y - 2*_w*_z, 2*_x*_z + 2*_w*_y],
        [2*_x*_y + 2*_w*_z, 1 - 2*_x*_x - 2*_z*_z, 2*_y*_z - 2*_w*_x],
        [2*_x*_z - 2*_w*_y, 2*_y*_z + 2*_w*_x, 1 - 2*_x*_x - 2*_y*_y]])
    _base_yaw = yaw_of(_w, _x, _y, _z)
    print(f"[FSM trace] crane base at {_bp} yaw {_base_yaw:.3f}", flush=True)
    rec = {k: [] for k in ("t", "phase", "x", "y", "z", "yaw", "tong", "tx", "ty", "tz", "tyaw")}
    step = 0
    for cycle in range(args_cli.cycles):
        pc = get_cloud(env, 0, args_cli.num_points)
        x, y, z, yaw = pol.get_target(pc)
        print(f"[FSM trace] cycle {cycle}: target {x:.2f} {y:.2f} {z:.2f} yaw {yaw:.2f}", flush=True)
        a = torch.zeros((1, 5), device=env.device)
        a[0, 0] = enc(x, bmin[0], bmax[0]); a[0, 1] = enc(y, bmin[1], bmax[1])
        a[0, 2] = enc(z, bmin[2], bmax[2])
        a[0, 3] = float(np.cos(2 * yaw)); a[0, 4] = float(np.sin(2 * yaw))

        # one step() is one whole grasp cycle; the sink fills from inside it
        env._trace_sink.clear()
        obs, _, term, trunc, _ = env.step(a)
        samples = list(env._trace_sink)
        for (_sc, ph, px, py, pz, qw, qx, qy, qz, tong) in samples:
            pb = (np.array([px, py, pz], dtype=np.float64) - _bp) @ _R
            rec["t"].append(step); step += 1
            rec["phase"].append(ph)
            rec["x"].append(float(pb[0]))
            rec["y"].append(float(pb[1]))
            rec["z"].append(float(pb[2]))
            yb = yaw_of(qw, qx, qy, qz) - _base_yaw
            rec["yaw"].append(float(np.arctan2(np.sin(yb), np.cos(yb))))
            rec["tong"].append(tong)
            rec["tx"].append(x); rec["ty"].append(y); rec["tz"].append(z); rec["tyaw"].append(yaw)
        print(f"[FSM trace] cycle {cycle}: {len(samples)} in-cycle samples, "
              f"phases {sorted(set(s[1] for s in samples))}", flush=True)
        if bool(term[0]) or bool(trunc[0]):
            obs, _ = env.reset()

    np.savez_compressed(args_cli.out, **{k: np.asarray(v) for k, v in rec.items()},
                        phase_names=np.array([env.PHASE_NAMES.get(i, str(i)) for i in range(12)]),
                        rate_hz=np.array(1.0 / float(env.cfg.sim.dt)))
    print(f"[FSM trace] wrote {args_cli.out}: {len(rec['t'])} control steps, "
          f"{len(set(rec['phase']))} distinct phases")
    simulation_app.close()


if __name__ == "__main__":
    main()
