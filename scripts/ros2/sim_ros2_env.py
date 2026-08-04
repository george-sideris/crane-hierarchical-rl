#!/usr/bin/env python3
"""Isaac Sim crane scene with ROS2 action graphs.

Creates the scene using the existing RL env (crane + logs + camera setup),
then adds OmniGraph ROS2 action graphs for native topic publishing.
Steps physics directly without running the RL policy or FSM.

Published topics:
    /zedx/depth          sensor_msgs/Image
    /zedx/rgb            sensor_msgs/Image
    /zedx/camera_info    sensor_msgs/CameraInfo
    /crane/joint_states  sensor_msgs/JointState

Subscribed topics:
    /crane/joint_command sensor_msgs/JointState

Usage:
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/humble/lib
    ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/sim_ros2_env.py
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim crane with ROS2 action graphs.")
parser.add_argument("--task", type=str, default="Isaac-Crane-PointCloud-CosSin-Raw-MR-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True  # always needed for depth publishing

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after app launch."""

import gymnasium as gym
import torch
import numpy as np
import omni
import omni.graph.core as og
import usdrt.Sdf
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, PointStamped
from isaacsim.core.utils import extensions
from pxr import Gf, UsdGeom, UsdPhysics, Usd

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
import crane_testbed.tasks  # noqa: F401

print("=" * 60)
print("Isaac Sim Crane — ROS2 Environment")
print("=" * 60)

# ── Create env using existing isaac lab scene ─────────────────────────────
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = args_cli.seed

# Create PointCloud env directly (same as train.py / play.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "envs"))
from crane_pointcloud_direct_env import CranePointCloudDirectEnv
env = CranePointCloudDirectEnv(env_cfg)
obs, _ = env.reset()

base_env = env._base_env
print(f"[INFO] Scene created: {args_cli.num_envs} env(s), task={args_cli.task}")

# Keep original actuator config (high stiffness for position control)

# Print exact transforms the sim uses for point cloud pipeline
# These can be used as static params in the policy node
# Read camera/base transforms BEFORE enabling ROS2 (avoids CUDA conflicts)
SIM_CAM_POS = None
SIM_CAM_QUAT_ROS = None
SIM_BASE_POS = None
SIM_BASE_QUAT = None
if hasattr(base_env, '_camera') and base_env._camera is not None:
    base_env._camera.update(dt=base_env.cfg.sim.dt)
    SIM_CAM_POS = base_env._camera.data.pos_w[0].cpu().numpy()
    SIM_CAM_QUAT_ROS = base_env._camera.data.quat_w_ros[0].cpu().numpy()
    SIM_BASE_POS = base_env.crane.data.root_pos_w[0].cpu().numpy()
    SIM_BASE_QUAT = base_env.crane.data.root_quat_w[0].cpu().numpy()
    print(f"[SIM TRANSFORMS] cam_pos={SIM_CAM_POS.tolist()}")
    print(f"[SIM TRANSFORMS] cam_quat_ros={SIM_CAM_QUAT_ROS.tolist()}")
    print(f"[SIM TRANSFORMS] base_pos={SIM_BASE_POS.tolist()}")
    print(f"[SIM TRANSFORMS] base_quat={SIM_BASE_QUAT.tolist()}")

    # One-time diagnostic: compute PCD the Isaac Lab way and print ranges
    try:
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth as cpfd
        _depth = base_env._camera.data.output["depth"][0].squeeze(-1)
        _intr = base_env._camera.data.intrinsic_matrices[0]
        _cam_pos_t = base_env._camera.data.pos_w[0]
        _cam_quat_t = base_env._camera.data.quat_w_ros[0]
        _base_pos_t = base_env.crane.data.root_pos_w[0]
        _d_mask = (_depth >= 1.0) & (_depth <= 10.0) & (~torch.isinf(_depth))
        _fd = torch.where(_d_mask, _depth, torch.tensor(float('inf'), device=_depth.device))
        _pc_w = cpfd(intrinsic_matrix=_intr, depth=_fd, keep_invalid=False,
                     position=_cam_pos_t, orientation=_cam_quat_t, device=_depth.device)
        _pc_b = _pc_w - _base_pos_t
        print(f"[SIM PCD] world: n={_pc_w.shape[0]}, "
              f"x=[{_pc_w[:,0].min():.2f},{_pc_w[:,0].max():.2f}] "
              f"y=[{_pc_w[:,1].min():.2f},{_pc_w[:,1].max():.2f}] "
              f"z=[{_pc_w[:,2].min():.2f},{_pc_w[:,2].max():.2f}]")
        print(f"[SIM PCD] base: n={_pc_b.shape[0]}, "
              f"x=[{_pc_b[:,0].min():.2f},{_pc_b[:,0].max():.2f}] "
              f"y=[{_pc_b[:,1].min():.2f},{_pc_b[:,1].max():.2f}] "
              f"z=[{_pc_b[:,2].min():.2f},{_pc_b[:,2].max():.2f}]")
    except Exception as e:
        print(f"[SIM PCD] diagnostic failed: {e}")

    # Print action space bounds
    try:
        if hasattr(base_env, '_action_bounds_min'):
            ab_min = base_env._action_bounds_min[0].cpu().numpy()
            ab_max = base_env._action_bounds_max[0].cpu().numpy()
            print(f"[SIM BOUNDS] action_bounds_min={ab_min.tolist()}")
            print(f"[SIM BOUNDS] action_bounds_max={ab_max.tolist()}")
    except Exception as e:
        print(f"[SIM BOUNDS] failed: {e}")

# ── Find articulation prim path ───────────────────────────────────────
usd_stage = omni.usd.get_context().get_stage()
CRANE_PATH = "/World/envs/env_0/Crane"
ARTICULATION_PATH = None
crane_prim = usd_stage.GetPrimAtPath(CRANE_PATH)
if crane_prim.IsValid():
    for prim in Usd.PrimRange(crane_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            ARTICULATION_PATH = str(prim.GetPath())
            break
if ARTICULATION_PATH is None:
    ARTICULATION_PATH = CRANE_PATH
print(f"[INFO] Articulation path: {ARTICULATION_PATH}")
# Debug: list all children of crane prim to find correct robot path
if crane_prim.IsValid():
    print("[DEBUG] Crane children:")
    for child in crane_prim.GetAllChildren():
        has_art = child.HasAPI(UsdPhysics.ArticulationRootAPI)
        print(f"  {child.GetPath()} type={child.GetTypeName()} art={has_art}")
        for grandchild in child.GetChildren()[:3]:
            print(f"    {grandchild.GetPath()} type={grandchild.GetTypeName()}")

# ── Create ROS2 camera prim ──────────────────────────────────────────
# Separate from TiledCamera (which uses tiled render products incompatible with ROS2 helper)
ROS2_CAMERA_PATH = "/World/ROS2Camera"
ros2_cam = UsdGeom.Camera(usd_stage.DefinePrim(ROS2_CAMERA_PATH, "Camera"))
xform_api = UsdGeom.XformCommonAPI(ros2_cam)
xform_api.SetTranslate(Gf.Vec3d(5.0, -1.0, 3.0))
xform_api.SetRotate((45, 0, 135), UsdGeom.XformCommonAPI.RotationOrderXYZ)
ros2_cam.GetFocalLengthAttr().Set(2.2)
ros2_cam.GetHorizontalApertureAttr().Set(5.8)
ros2_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.3, 20.0))
print(f"[INFO] Created ROS2 camera at {ROS2_CAMERA_PATH}")

# ── Create stick-mounted ZED2 camera ─────────────────────────────────
# Spawn a Camera prim under the existing camera_zed2 Xform on the stick.
ZED2_CAMERA_PATH = "/World/envs/env_0/Crane/stick/camera_zed2/Camera"
zed2_cam = UsdGeom.Camera(usd_stage.DefinePrim(ZED2_CAMERA_PATH, "Camera"))
zed2_xform = UsdGeom.XformCommonAPI(zed2_cam)
zed2_xform.SetRotate((45, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)  # 45deg down from forward
zed2_cam.GetFocalLengthAttr().Set(2.2)
zed2_cam.GetHorizontalApertureAttr().Set(5.8)
zed2_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.3, 20.0))
print(f"[INFO] Created stick ZED2 camera at {ZED2_CAMERA_PATH}")

simulation_app.update()

# ── Enable ROS2 bridge extension ──────────────────────────────────────
print("\n[ROS2] Enabling ROS2 bridge extension...")
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

# ── Action graphs ─────────────────────────────────────────────────────
keys = og.Controller.Keys

# 1. Camera graph (depth + RGB + camera_info)
print("[ROS2] Creating camera graph...")
(cam_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": "/ROS2/CameraGraph",
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
            ("depthHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("rgbHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("infoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut", "depthHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "rgbHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "infoHelper.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "depthHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "rgbHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "infoHelper.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 1),  # Use viewport 1 to not hijack the GUI
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path("/World/envs/env_0/Camera")]),  # Use existing env camera
            ("depthHelper.inputs:type", "depth"),
            ("depthHelper.inputs:topicName", "/zedx/depth"),
            ("depthHelper.inputs:frameId", "zedx_depth"),
            ("rgbHelper.inputs:type", "rgb"),
            ("rgbHelper.inputs:topicName", "/zedx/rgb"),
            ("rgbHelper.inputs:frameId", "zedx_rgb"),
            ("infoHelper.inputs:topicName", "/zedx/camera_info"),
            ("infoHelper.inputs:frameId", "zedx_depth"),
        ],
    },
)
og.Controller.evaluate_sync(cam_graph)
simulation_app.update()
print("[ROS2] Camera graph → /zedx/depth, /zedx/rgb, /zedx/camera_info")

# 1b. Stick ZED2 camera graph
print("[ROS2] Creating stick ZED2 camera graph...")
(zed2_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": "/ROS2/Zed2CameraGraph",
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("createViewport", "isaacsim.core.nodes.IsaacCreateViewport"),
            ("getRenderProduct", "isaacsim.core.nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "isaacsim.core.nodes.IsaacSetCameraOnRenderProduct"),
            ("depthHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("rgbHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("infoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut", "depthHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "rgbHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "infoHelper.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "depthHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "rgbHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "infoHelper.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 2),
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(ZED2_CAMERA_PATH)]),
            ("depthHelper.inputs:type", "depth"),
            ("depthHelper.inputs:topicName", "/zed2/depth"),
            ("depthHelper.inputs:frameId", "zed2_depth"),
            ("rgbHelper.inputs:type", "rgb"),
            ("rgbHelper.inputs:topicName", "/zed2/rgb"),
            ("rgbHelper.inputs:frameId", "zed2_rgb"),
            ("infoHelper.inputs:topicName", "/zed2/camera_info"),
            ("infoHelper.inputs:frameId", "zed2_depth"),
        ],
    },
)
og.Controller.evaluate_sync(zed2_graph)
simulation_app.update()
print("[ROS2] Stick ZED2 graph → /zed2/depth, /zed2/rgb, /zed2/camera_info")

# 2. Joint command subscriber: OmniGraph articulation controller
# This uses Isaac Sim's built-in ROS2 bridge to subscribe to /crane/joint_command
# and apply position/velocity targets through the sim's implicit actuator
# Joint command subscriber via rclpy (OmniGraph articulation controller has CUDA
# compatibility issues with Isaac Lab's GPU pipeline, so we apply targets manually)
latest_joint_cmd = {"pos": None, "vel": None, "names": None}

def _joint_cmd_cb(msg):
    latest_joint_cmd["names"] = list(msg.name)
    latest_joint_cmd["pos"] = list(msg.position) if msg.position else None
    latest_joint_cmd["vel"] = list(msg.velocity) if msg.velocity else None

# 3. Joint state publisher + EE position — created after rclpy init (see below)
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Float32MultiArray, Float64

# 4. Clock publisher (suppresses "Timestamp 0" warnings)
print("[ROS2] Creating clock publisher...")
og.Controller.edit(
    {"graph_path": "/ROS2/ClockPublisher", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("publishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
        ],
    },
)
print("[ROS2] Clock → /clock")

# 5. TF publisher using rclpy (publishes ROS-convention camera quaternions)
print("[ROS2] Initializing rclpy TF publisher...")
rclpy.init()
tf_node = Node("sim_tf_publisher")
tf_static_pub = tf_node.create_publisher(TFMessage, "/tf_static", rclpy.qos.QoSProfile(
    depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
joint_state_pub = tf_node.create_publisher(JointStateMsg, "/crane/joint_states", 10)
ee_pos_pub = tf_node.create_publisher(PointStamped, "/crane/ee_pos_base", 10)
# Current basegrapple yaw in crane_base frame (radians). Used by policy node
# to convert policy's basegrapple-yaw target → yaw-joint target (matches
# crane_rl_env_full.py:_convert_grapple_yaw_to_joint_position).
grapple_yaw_pub = tf_node.create_publisher(Float64, "/crane/grapple_yaw_base", 10)
joint_cmd_sub = tf_node.create_subscription(JointStateMsg, "/crane/joint_command", _joint_cmd_cb, 10)
print("[ROS2] Joint states → /crane/joint_states (via rclpy)")
print("[ROS2] Joint commands ← /crane/joint_command (via rclpy)")
print("[ROS2] EE position → /crane/ee_pos_base (via rclpy)")
print("[ROS2] Joint commands ← /crane/joint_command (via rclpy)")

# Get controlled joint indices (same as RL env uses)
CTRL_JOINT_NAMES = ["basemast_to_mast", "mast_to_mainboom", "mainboom_to_stick", "stick_to_telescope"]
YAW_JOINT_NAME = "lowerpassive_to_basegrapple"
GRIPPER_NAMES = ["basegrapple_to_gripperleft", "basegrapple_to_gripperright"]
all_crane_joint_names = [j.strip() for j in base_env.crane.joint_names]
ctrl_joint_ids = [all_crane_joint_names.index(n) for n in CTRL_JOINT_NAMES]
yaw_joint_id = all_crane_joint_names.index(YAW_JOINT_NAME)
gripper_joint_ids = [all_crane_joint_names.index(n) for n in GRIPPER_NAMES]
print(f"[INFO] Controlled arm joint ids: {ctrl_joint_ids}, yaw: {yaw_joint_id}, grippers: {gripper_joint_ids}")

# Get basegrapple body index for EE position tracking
bg_ids, _ = base_env.crane.find_bodies(["basegrapple"])
BG_BODY_ID = bg_ids[0] if bg_ids else None
print(f"[INFO] Basegrapple body id: {BG_BODY_ID}")

# Publish static transforms using values read BEFORE ROS2 was enabled

def make_tf_msg(frame_id, child_id, pos, quat_wxyz):
    t = TransformStamped()
    t.header.frame_id = frame_id
    t.child_frame_id = child_id
    t.header.stamp = tf_node.get_clock().now().to_msg()
    t.transform.translation.x = float(pos[0])
    t.transform.translation.y = float(pos[1])
    t.transform.translation.z = float(pos[2])
    t.transform.rotation.w = float(quat_wxyz[0])
    t.transform.rotation.x = float(quat_wxyz[1])
    t.transform.rotation.y = float(quat_wxyz[2])
    t.transform.rotation.z = float(quat_wxyz[3])
    return t

cam_tf = make_tf_msg("world", "zedx_camera", SIM_CAM_POS, SIM_CAM_QUAT_ROS)
base_tf = make_tf_msg("world", "crane_base", SIM_BASE_POS, SIM_BASE_QUAT)
tf_msg = TFMessage(transforms=[cam_tf, base_tf])
tf_static_pub.publish(tf_msg)
print(f"[ROS2] Static TF: zedx_camera pos={SIM_CAM_POS.round(3)}, crane_base pos={SIM_BASE_POS.round(3)}")

# Publish action space bounds (latched, so policy node gets them on connect)
bounds_pub = tf_node.create_publisher(Float32MultiArray, "/crane/action_bounds",
    rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL))
if hasattr(base_env, '_action_bounds_min') and base_env._action_bounds_valid[0]:
    ab_min = base_env._action_bounds_min[0].cpu().numpy()
    ab_max = base_env._action_bounds_max[0].cpu().numpy()
    bounds_msg = Float32MultiArray()
    bounds_msg.data = [float(v) for v in ab_min] + [float(v) for v in ab_max]
    bounds_pub.publish(bounds_msg)
    print(f"[ROS2] Action bounds: min={ab_min.round(3).tolist()}, max={ab_max.round(3).tolist()}")
else:
    print("[WARN] Action bounds not computed — policy node will use defaults")

# ── Debug marker: policy target sphere ────────────────────────────────
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
_policy_target_world = None

def _policy_target_cb(msg):
    global _policy_target_world
    # msg is in base frame, convert to world
    bx = float(msg.pose.position.x) + SIM_BASE_POS[0]
    by = float(msg.pose.position.y) + SIM_BASE_POS[1]
    bz = float(msg.pose.position.z) + SIM_BASE_POS[2]
    _policy_target_world = (bx, by, bz)

tf_node.create_subscription(PoseStampedMsg, "/crane/policy_target", _policy_target_cb, 10)

# Create a visible sphere prim for the target marker
MARKER_PATH = "/World/PolicyTargetMarker"
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
sphere = UsdGeom.Sphere.Define(stage, MARKER_PATH)
sphere.GetRadiusAttr().Set(0.15)
sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])  # red
print("[ROS2] Debug marker: /World/PolicyTargetMarker (red sphere at policy target)")

print("\n[ROS2] All topics ready:")
print("  Published: /zedx/depth, /zedx/rgb, /zedx/camera_info, /zed2/depth, /zed2/rgb, /zed2/camera_info, /crane/joint_states, /crane/ee_pos_base, /tf_static, /crane/action_bounds")
print("  Subscribed: /crane/joint_command, /crane/policy_target")
print("\n[INFO] Stepping simulation. Press Ctrl+C to stop.\n")

# ── Get joint names for publishing ─────────────────────────────────────
JOINT_NAMES = [
    "basemast_to_mast", "mast_to_mainboom", "mainboom_to_stick",
    "stick_to_telescope", "telescope_to_upperpassive", "upperpassive_to_lowerpassive",
    "lowerpassive_to_basegrapple", "basegrapple_to_gripperleft", "basegrapple_to_gripperright",
]

# ── Main loop: step physics directly, no RL/FSM ─────────────────────
frame = 0
try:
    while simulation_app.is_running():
        # Apply joint commands from JV controller
        # Arm/yaw: position targets (high-stiffness actuators)
        # Grippers: velocity targets (stiffness=0, damping=500)
        GRIPPER_NAMES_SET = {"basegrapple_to_gripperleft", "basegrapple_to_gripperright"}
        cmd = latest_joint_cmd
        if cmd["names"] is not None and cmd["pos"] is not None and len(cmd["pos"]) > 0:
            try:
                cmd_names = cmd["names"]
                pos_target = base_env.crane.data.joint_pos[0].clone().unsqueeze(0)
                vel_target = torch.zeros(1, base_env.crane.num_joints, device=base_env.device)
                for j, name in enumerate(cmd_names):
                    if name in all_crane_joint_names:
                        idx = all_crane_joint_names.index(name)
                        if name in GRIPPER_NAMES_SET:
                            # Run gripper PD here at physics rate
                            if cmd["pos"] is not None and j < len(cmd["pos"]):
                                q_des = float(cmd["pos"][j])
                                q_cur = float(base_env.crane.data.joint_pos[0, idx].item())
                                err = q_des - q_cur
                                # Match RL env: KP_GRIP=8, MAX_VEL_GRIP=10
                                vel = max(-10.0, min(10.0, 8.0 * err))
                                vel_target[0, idx] = vel
                        else:
                            # Arm/yaw: use position target (q_des)
                            if j < len(cmd["pos"]):
                                pos_target[0, idx] = cmd["pos"][j]
                base_env.crane.set_joint_position_target(pos_target)
                base_env.crane.set_joint_velocity_target(vel_target)
                base_env.crane.write_data_to_sim()
            except Exception as e:
                if frame % 600 == 1:
                    print(f"[WARN] Joint cmd error: {e}")

        base_env.sim.step(render=True)
        base_env.scene.update(base_env.physics_dt)
        frame += 1

        # Publish joint states from Isaac Lab articulation data
        if frame % 6 == 0:  # ~10Hz at 60fps
            try:
                joint_pos = base_env.crane.data.joint_pos[0].cpu().numpy()
                joint_vel = base_env.crane.data.joint_vel[0].cpu().numpy()
                js_msg = JointStateMsg()
                js_msg.header.stamp = tf_node.get_clock().now().to_msg()
                js_msg.name = list(JOINT_NAMES)
                js_msg.position = joint_pos[:len(JOINT_NAMES)].tolist()
                js_msg.velocity = joint_vel[:len(JOINT_NAMES)].tolist()
                js_msg.effort = [0.0] * len(JOINT_NAMES)
                joint_state_pub.publish(js_msg)

                # Publish basegrapple position AND yaw in crane base frame
                if BG_BODY_ID is not None:
                    bg_pose_w = base_env.crane.data.body_pose_w[:, BG_BODY_ID]
                    root_pose_w = base_env.crane.data.root_pose_w
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                        bg_pose_w[:, 0:3], bg_pose_w[:, 3:7],
                    )
                    p = ee_pos_b[0].cpu().numpy()
                    ee_msg = PointStamped()
                    ee_msg.header.stamp = js_msg.header.stamp
                    ee_msg.header.frame_id = "crane_base"
                    ee_msg.point.x = float(p[0])
                    ee_msg.point.y = float(p[1])
                    ee_msg.point.z = float(p[2])
                    ee_pos_pub.publish(ee_msg)

                    # Yaw of basegrapple in base frame (z-axis rotation from quat wxyz)
                    qw, qx, qy, qz = ee_quat_b[0].cpu().numpy()
                    bg_yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                                        1.0 - 2.0 * (qy * qy + qz * qz))
                    yaw_msg = Float64()
                    yaw_msg.data = float(bg_yaw)
                    grapple_yaw_pub.publish(yaw_msg)
            except Exception:
                pass  # skip if data not ready

            rclpy.spin_once(tf_node, timeout_sec=0)

        # Update policy target marker position
        if _policy_target_world is not None:
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(MARKER_PATH))
            if xform:
                ops = xform.GetOrderedXformOps()
                if not ops:
                    xform.AddTranslateOp()
                    ops = xform.GetOrderedXformOps()
                ops[0].Set(Gf.Vec3d(*_policy_target_world))

        if frame % 600 == 0:
            ee_str = ""
            if BG_BODY_ID is not None:
                try:
                    bg_pose_w = base_env.crane.data.body_pose_w[:, BG_BODY_ID]
                    root_pose_w = base_env.crane.data.root_pose_w
                    ee_pos_b, _ = subtract_frame_transforms(
                        root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                        bg_pose_w[:, 0:3], bg_pose_w[:, 3:7],
                    )
                    p = ee_pos_b[0].cpu().numpy()
                    ee_str = f"  EE base frame: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
                except Exception:
                    pass
            print(f"[INFO] Frame {frame}, sim time {frame * base_env.physics_dt:.1f}s{ee_str}")
except KeyboardInterrupt:
    print("\n[INFO] Stopping...")

tf_node.destroy_node()
rclpy.shutdown()
env.close()


if __name__ == "__main__":
    simulation_app.close()