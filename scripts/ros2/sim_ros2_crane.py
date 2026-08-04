#!/usr/bin/env python3
"""Isaac Sim scene that acts as a ROS2 crane.

Creates the crane scene (URDF + log pile + ZED X camera) and connects it
to ROS2 via OmniGraph action graphs. No RL env, no policy, no FSM — just
the sim as a "body" on the ROS2 network.

Published topics:
    /zedx/depth          (sensor_msgs/Image)       — depth from ZED X camera
    /zedx/rgb            (sensor_msgs/Image)       — RGB from ZED X camera
    /zedx/camera_info    (sensor_msgs/CameraInfo)  — camera intrinsics
    /crane/joint_states  (sensor_msgs/JointState)  — joint positions + velocities

Subscribed topics:
    /crane/joint_command (sensor_msgs/JointState)  — joint position/velocity targets

Usage (inside Docker):
    ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/sim_ros2_crane.py

Then in another terminal:
    source /opt/ros/humble/setup.bash
    ros2 topic list
    ros2 topic echo /zedx/depth
"""

import argparse
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Isaac Sim crane with ROS2 action graphs.")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--num_logs", type=int, default=200)
parser.add_argument("--crane_usd", type=str,
                    default="/workspace/crane_testbed/assets/scenes/crane.usd")
parser.add_argument("--rack_usd", type=str,
                    default="/workspace/crane_testbed/assets/scenes/short_rack.usdc")
parser.add_argument("--log_usd", type=str,
                    default="/workspace/crane_testbed/assets/scenes/testbed_log.usd")
args = parser.parse_args()

CONFIG = {
    "renderer": "RaytracedLighting",
    "headless": args.headless,
}
simulation_app = SimulationApp(CONFIG)

# ── Imports (after SimulationApp) ─────────────────────────────────────
import carb
import numpy as np
import omni
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext, World
from isaacsim.core.utils import extensions, stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# Enable ROS2 bridge extension
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

print("=" * 60)
print("Isaac Sim Crane — ROS2 Bridge Mode")
print("=" * 60)

# ── Create simulation context ─────────────────────────────────────────
sim_context = SimulationContext(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 60.0,
)

# ── Load crane USD ────────────────────────────────────────────────────
CRANE_PRIM_PATH = "/World/Crane"
print(f"[INFO] Loading crane from {args.crane_usd}")
stage.add_reference_to_stage(args.crane_usd, CRANE_PRIM_PATH)

# Set crane position (matching env defaults)
crane_xform = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(CRANE_PRIM_PATH))
if crane_xform:
    crane_xform.ClearXformOpOrder()
    crane_xform.AddTranslateOp().Set(Gf.Vec3d(6.0, -1.52, 1.423))

# ── Load rack ─────────────────────────────────────────────────────────
RACK_PRIM_PATH = "/World/Rack"
print(f"[INFO] Loading rack from {args.rack_usd}")
stage.add_reference_to_stage(args.rack_usd, RACK_PRIM_PATH)
rack_xform = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(RACK_PRIM_PATH))
if rack_xform:
    rack_xform.ClearXformOpOrder()
    rack_xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))

# ── Spawn logs ────────────────────────────────────────────────────────
LOGS_ROOT = "/World/Logs"
omni.usd.get_context().get_stage().DefinePrim(LOGS_ROOT, "Scope")

# Simple grid layout (matching env's log spawning approximately)
rows = 10
layers = 20
spacing_y = 0.304
spacing_z = 0.22
base_z = 0.20
center_y = 0.0
num_spawned = 0

print(f"[INFO] Spawning {args.num_logs} logs...")
for layer in range(layers):
    for row in range(rows):
        if num_spawned >= args.num_logs:
            break
        y = center_y + (row - rows / 2 + 0.5) * spacing_y
        z = base_z + layer * spacing_z
        x = 2.0  # rack center X
        log_path = f"{LOGS_ROOT}/log_{num_spawned:04d}"
        stage.add_reference_to_stage(args.log_usd, log_path)
        log_xform = UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(log_path))
        if log_xform:
            log_xform.ClearXformOpOrder()
            log_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        num_spawned += 1
    if num_spawned >= args.num_logs:
        break

print(f"[INFO] Spawned {num_spawned} logs")

# ── Create ZED X camera ──────────────────────────────────────────────
CAMERA_PATH = "/World/ZedXCamera"
camera_prim = UsdGeom.Camera(
    omni.usd.get_context().get_stage().DefinePrim(CAMERA_PATH, "Camera"))
xform_api = UsdGeom.XformCommonAPI(camera_prim)
# Position matching sim: (5.0, -1.0, 3.0) looking at rack
xform_api.SetTranslate(Gf.Vec3d(5.0, -1.0, 3.0))
# Rotation to look at rack (approximately matching sim quaternion)
xform_api.SetRotate((45, 0, 135), UsdGeom.XformCommonAPI.RotationOrderXYZ)
# ZED X lens parameters
camera_prim.GetFocalLengthAttr().Set(2.2)  # mm
camera_prim.GetHorizontalApertureAttr().Set(5.8)  # mm
camera_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.3, 20.0))

simulation_app.update()

# ── Create ROS2 action graphs ────────────────────────────────────────
keys = og.Controller.Keys

# --- Graph 1: Camera (depth + RGB + camera_info) → ROS2 ---
print("[INFO] Creating camera ROS2 graph...")
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
            # Wire render product to all camera helpers
            ("setCamera.outputs:execOut", "depthHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "rgbHelper.inputs:execIn"),
            ("setCamera.outputs:execOut", "infoHelper.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "depthHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "rgbHelper.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "infoHelper.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(CAMERA_PATH)]),
            # Depth
            ("depthHelper.inputs:type", "depth"),
            ("depthHelper.inputs:topicName", "/zedx/depth"),
            ("depthHelper.inputs:frameId", "zedx_depth"),
            # RGB
            ("rgbHelper.inputs:type", "rgb"),
            ("rgbHelper.inputs:topicName", "/zedx/rgb"),
            ("rgbHelper.inputs:frameId", "zedx_rgb"),
            # Camera info
            ("infoHelper.inputs:topicName", "/zedx/camera_info"),
            ("infoHelper.inputs:frameId", "zedx_depth"),
        ],
    },
)
# Evaluate once to create the SDG pipeline publishers
og.Controller.evaluate_sync(cam_graph)
simulation_app.update()

# --- Graph 2: Joint State Publisher (sim → ROS2) ---
print("[INFO] Creating joint state publisher graph...")
# Find the articulation root prim path
ARTICULATION_PATH = CRANE_PRIM_PATH + "/fpiforwarder"

(joint_pub_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": "/ROS2/JointStatePublisher",
        "evaluator_name": "execution",
    },
    {
        keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnTick"),
            ("publishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        keys.SET_VALUES: [
            ("publishJointState.inputs:targetPrim", ARTICULATION_PATH),
            ("publishJointState.inputs:topicName", "/crane/joint_states"),
        ],
        keys.CONNECT: [
            ("OnTick.outputs:tick", "publishJointState.inputs:execIn"),
        ],
    },
)

# --- Graph 3: Joint Command Subscriber (ROS2 → sim) ---
print("[INFO] Creating joint command subscriber graph...")
(joint_sub_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": "/ROS2/JointCommandSubscriber",
        "evaluator_name": "execution",
    },
    {
        keys.CREATE_NODES: [
            ("subscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("articulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.SET_VALUES: [
            ("subscribeJointState.inputs:topicName", "/crane/joint_command"),
            ("articulationController.inputs:robotPath", ARTICULATION_PATH),
            ("articulationController.inputs:usePath", True),
        ],
        keys.CONNECT: [
            ("subscribeJointState.outputs:jointNames", "articulationController.inputs:jointNames"),
            ("subscribeJointState.outputs:positionCommand", "articulationController.inputs:positionCommand"),
            ("subscribeJointState.outputs:velocityCommand", "articulationController.inputs:velocityCommand"),
            ("subscribeJointState.outputs:execOut", "articulationController.inputs:execIn"),
        ],
    },
)

print("[INFO] Action graphs created:")
print("  /zedx/depth          — depth image (sensor_msgs/Image)")
print("  /zedx/rgb            — RGB image (sensor_msgs/Image)")
print("  /zedx/camera_info    — camera intrinsics (sensor_msgs/CameraInfo)")
print("  /crane/joint_states  — joint feedback (sensor_msgs/JointState)")
print("  /crane/joint_command — joint commands (sensor_msgs/JointState)")

# ── Initialize and run ────────────────────────────────────────────────
sim_context.initialize_physics()
sim_context.play()

print("\n[INFO] Simulation running. ROS2 topics are live.")
print("[INFO] Press Ctrl+C to stop.\n")

frame = 0
try:
    while simulation_app.is_running():
        sim_context.step(render=True)
        frame += 1
        if frame % 600 == 0:  # Status every ~10s
            print(f"[INFO] Frame {frame}, sim time {frame / 60.0:.1f}s")
except KeyboardInterrupt:
    print("\n[INFO] Stopping...")

sim_context.stop()
simulation_app.close()
