#!/usr/bin/env python3
"""Gaze FK probe — checks whether the stick-mounted ZED2 camera can be driven
to (approximately) match the fixed training camera's pose by joint commands
alone.

Phase 1 (in-sim): launch the env, read fixed cam pose, crane base pose, stick
link world pose, and camera_zed2 xform world pose. Compute the constant
offset stick→camera_zed2 (in stick local frame).

Phase 2 (offline FK): sweep (q1, q2, q3) over the joint limits, compute stick
cam world pose using URDF joint origins, score against fixed cam target.
Reports best joint config + position/rotation error.

Run with sim's python (headless):
    isaaclab -p /workspace/crane_testbed/scripts/ros2/tools/gaze_fk_probe.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Crane-PointCloud-CosSin-Raw-MR-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--grid", type=int, default=25, help="Grid resolution per joint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import omni
from pxr import UsdGeom, Usd, Gf

import isaaclab_tasks  # noqa: F401
import crane_testbed.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "envs"))
from crane_pointcloud_direct_env import CranePointCloudDirectEnv  # noqa: E402

# ── Phase 1: launch env, read transforms ──────────────────────────────
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env = CranePointCloudDirectEnv(env_cfg)
env.reset()
base_env = env._base_env
base_env._camera.update(dt=base_env.cfg.sim.dt)

fixed_cam_pos_w = base_env._camera.data.pos_w[0].cpu().numpy()
fixed_cam_quat_w_ros = base_env._camera.data.quat_w_ros[0].cpu().numpy()  # wxyz
base_pos_w = base_env.crane.data.root_pos_w[0].cpu().numpy()
base_quat_w = base_env.crane.data.root_quat_w[0].cpu().numpy()  # wxyz

stage = omni.usd.get_context().get_stage()
xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def usd_world_pose(prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {prim_path}")
    M = xform_cache.GetLocalToWorldTransform(prim)
    pos = np.array([M[3][0], M[3][1], M[3][2]])
    R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
    return pos, R


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


# Stick and camera_zed2 world poses
stick_pos_w, stick_R_w = usd_world_pose("/World/envs/env_0/Crane/stick")
zed2_pos_w, zed2_R_w = usd_world_pose("/World/envs/env_0/Crane/stick/camera_zed2")

# stick → zed2 in stick local frame
stick_to_zed2_pos = stick_R_w.T @ (zed2_pos_w - stick_pos_w)
stick_to_zed2_R = stick_R_w.T @ zed2_R_w

# fixed cam in basemast (crane base) frame
base_R_w = quat_wxyz_to_R(base_quat_w)
target_pos_b = base_R_w.T @ (fixed_cam_pos_w - base_pos_w)
target_R_b = base_R_w.T @ quat_wxyz_to_R(fixed_cam_quat_w_ros)

# Verify URDF basemast frame == articulation root frame: probe basemast world pose
basemast_pos_w, basemast_R_w = usd_world_pose("/World/envs/env_0/Crane/basemast")
basemast_in_base_pos = base_R_w.T @ (basemast_pos_w - base_pos_w)
basemast_in_base_R = base_R_w.T @ basemast_R_w

print("\n========== PROBE RESULTS ==========")
print(f"fixed_cam_world pos = {fixed_cam_pos_w}")
print(f"fixed_cam_world quat (wxyz, ROS conv) = {fixed_cam_quat_w_ros}")
print(f"crane_base_world pos = {base_pos_w}, quat = {base_quat_w}")
print(f"target (fixed cam in basemast frame): pos = {target_pos_b}")
print(f"basemast offset from articulation root: pos = {basemast_in_base_pos}")
print(f"basemast rotation from articulation root:\n{basemast_in_base_R}")
print(f"stick→zed2 offset (stick local frame): pos = {stick_to_zed2_pos}")
print(f"stick→zed2 R (stick local frame):\n{stick_to_zed2_R}")

# ── Phase 2: FK sweep using URDF joint origins ────────────────────────
# URDF joints (parent_to_child static origin, then revolute around axis):
#   basemast_to_mast:    origin (0, 0, 0.26881),       axis (0,0,1) Z
#   mast_to_mainboom:    origin (0, -0.0601374, 1.09553), axis (1,0,0) X
#   mainboom_to_stick:   origin (0, 3.04904, -0.00649),   axis (1,0,0) X
JOINT_ORIGINS = [
    np.array([0.0, 0.0, 0.26881]),
    np.array([0.0, -0.0601374, 1.09553]),
    np.array([0.0, 3.04904, -0.00649]),
]
JOINT_AXES = [
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([1.0, 0.0, 0.0]),
]
JOINT_LIMITS = [
    (-1.74533, 1.74533),
    (-0.383972, 1.309),
    (-3.08574, 0.035),
]

# The Camera prim under camera_zed2 has +45° X rotation (from sim_ros2_env.py:179)
THETA = np.deg2rad(45.0)
ZED2_TO_CAMERA_R = np.array([
    [1, 0, 0],
    [0, np.cos(THETA), -np.sin(THETA)],
    [0, np.sin(THETA),  np.cos(THETA)],
])


def axis_angle_R(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def fk_basemast_to_stick(q1, q2, q3):
    """Returns (pos, R) of stick link in basemast frame."""
    T = np.eye(4)
    for q, origin, axis in zip([q1, q2, q3], JOINT_ORIGINS, JOINT_AXES):
        Tj = np.eye(4)
        Tj[:3, 3] = origin
        Tj[:3, :3] = axis_angle_R(axis, q)
        T = T @ Tj
    return T[:3, 3], T[:3, :3]


def rotation_angle(R1, R2):
    R = R1.T @ R2
    cos_t = (np.trace(R) - 1.0) / 2.0
    return np.arccos(np.clip(cos_t, -1.0, 1.0))


# basemast frame in articulation-root (= crane base) frame is approximately identity,
# but we apply it explicitly to be safe.
def stick_cam_in_base(q1, q2, q3):
    stick_pos_bm, stick_R_bm = fk_basemast_to_stick(q1, q2, q3)
    zed2_pos_bm = stick_pos_bm + stick_R_bm @ stick_to_zed2_pos
    zed2_R_bm = stick_R_bm @ stick_to_zed2_R
    cam_R_bm = zed2_R_bm @ ZED2_TO_CAMERA_R
    # basemast → articulation-root (= "base" frame used for target)
    pos_b = basemast_in_base_pos + basemast_in_base_R @ zed2_pos_bm
    R_b = basemast_in_base_R @ cam_R_bm
    return pos_b, R_b


N = args.grid
qs1 = np.linspace(*JOINT_LIMITS[0], N)
qs2 = np.linspace(*JOINT_LIMITS[1], N)
qs3 = np.linspace(*JOINT_LIMITS[2], N)

best_pos = (np.inf, None, None)        # (pos_err, joints, cam_pos_b)
best_combined = (np.inf, None, None, None, None)  # (score, joints, cam_pos_b, pos_err, rot_err)

for q1 in qs1:
    for q2 in qs2:
        for q3 in qs3:
            pos_b, R_b = stick_cam_in_base(q1, q2, q3)
            pos_err = np.linalg.norm(pos_b - target_pos_b)
            rot_err = rotation_angle(R_b, target_R_b)
            score = pos_err + 0.5 * rot_err  # 1 rad ~ 0.5 m equiv
            if pos_err < best_pos[0]:
                best_pos = (pos_err, (q1, q2, q3), pos_b)
            if score < best_combined[0]:
                best_combined = (score, (q1, q2, q3), pos_b, pos_err, rot_err)

print("\n========== FK SWEEP RESULTS ==========")
print(f"Grid: {N}^3 = {N**3} configurations")
print(f"Target cam pos in base = {target_pos_b}")
print()
print(f"[best position-only] q=({best_pos[1][0]:.3f}, {best_pos[1][1]:.3f}, {best_pos[1][2]:.3f}) rad")
print(f"  cam_pos_b = {best_pos[2]}")
print(f"  pos_err = {best_pos[0]*100:.1f} cm")
print()
b_score, b_joints, b_pos_b, b_pos_err, b_rot_err = best_combined
print(f"[best combined (pos+0.5*rot)] q=({b_joints[0]:.3f}, {b_joints[1]:.3f}, {b_joints[2]:.3f}) rad")
print(f"  cam_pos_b = {b_pos_b}")
print(f"  pos_err = {b_pos_err*100:.1f} cm   rot_err = {np.degrees(b_rot_err):.1f}°")

env.close()
simulation_app.close()
