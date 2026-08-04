#!/usr/bin/env python3.8
"""Patch FPI's partner USD scene so it publishes the topics our pipeline needs.

What it changes (idempotent — safe to re-run):

1. Adds a `ros2_depth_helper` ROS2CameraHelper next to the existing RGB
   helper, publishing 32FC1 depth on `/zedx/depth` (frame `zedx_camera`).
2. Updates the existing `ros2_camera_info_helper` to publish on
   `/zedx/camera_info` with frame `zedx_camera`.
3. Updates the existing `ros2_camera_helper` (RGB) to publish on
   `/zedx/rgb` with frame `zedx_camera` (cosmetic — the policy doesn't
   use RGB but consistency keeps debugging easier).
4. Adds a fixed world-frame camera at `/World/zedx_camera` matching the
   trained sim's camera spec (ZED X intrinsics, 640x360), positioned at
   the same offset to crane base as training (-1, 1.92, 1.577). Repoints
   isaac_create_render_product to render from this fixed camera and adds
   the new prim to the transform tree publisher's targets so /tf has a
   `zedx_camera` frame.

Run from the host (needs the local pxr install at
/home/george/.local/lib/python3.8/site-packages):

    PYTHONPATH=/home/george/.local/lib/python3.8/site-packages \\
        /usr/bin/python3.8 \\
        /home/george/IsaacLab/crane_testbed/scripts/ros2/tools/patch_partner_usd.py
"""

import argparse
import math
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

DEFAULT_USD = (
    "/home/george/IsaacLab/crane_testbed/Omniverse-Simulation/"
    "Omniverse/Projects/crane_lab/log_loader_crane_lab_grasping.usd"
)
SENSORS_GRAPH = "/ActionGraph_Sensors"
CONTROL_GRAPH = "/ActionGraph_Control"
VIZ_GRAPH = "/ActionGraph_Viz"
DEPTH_HELPER_NAME = "ros2_depth_helper"
DEPTH_HELPER_TYPE = "isaacsim.ros2.bridge.ROS2CameraHelper"

# Inline Python embedded in the viz ScriptNode. Reads marker positions
# from a JSON file the policy node writes each cycle, then updates the
# Sphere prims' translates. File-based bridge sidesteps the broken
# ROS2SubscribeTransformTree node in Isaac Sim 4.5.
VIZ_SCRIPT = """
import json, os
import omni.usd
from pxr import Gf, UsdGeom

STATE_FILE = "/tmp/crane_marker_state.json"
PATHS = {
    "policy": "/World/Visuals/PolicyTarget",
    "ee":     "/World/Visuals/EECommandTarget",
}


def setup(db):
    pass


def cleanup(db):
    pass


def compute(db):
    if not os.path.exists(STATE_FILE):
        return True
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return True
    stage = omni.usd.get_context().get_stage()
    for key, prim_path in PATHS.items():
        if key not in state:
            continue
        xyz = state[key]
        if len(xyz) != 3:
            continue
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == 'xformOp:translate':
                op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
                break
    return True
"""

# Fixed camera matching trained sim (ZED X, 640x360, base-relative offset)
FIXED_CAM_PATH = "/World/zedx_camera"
# (x, y, z) world position. Partner's crane base sits at world (0,0,0),
# so this is also the base-relative offset that training used.
FIXED_CAM_POS = (-1.0, 1.92, 1.577)
# Quaternion (wxyz, OpenGL convention) — same as training. UsdGeomCamera
# uses OpenGL convention natively (looks down -Z) so no remap needed.
FIXED_CAM_QUAT_WXYZ = (0.6124, 0.3536, 0.3536, 0.6124)
# ZED X intrinsics matching training:
FIXED_CAM_FOCAL_LENGTH = 2.2          # mm
FIXED_CAM_HORIZONTAL_APERTURE = 5.8   # mm  (sensor width, gives ~110° HFOV)
FIXED_CAM_CLIPPING = (0.3, 20.0)      # m, ZED X depth range
FIXED_CAM_FOCUS = 5.0                 # m, focus distance
FIXED_CAM_WIDTH = 640
FIXED_CAM_HEIGHT = 360
# Vertical aperture must scale by aspect ratio to keep square pixels
# (i.e. fy == fx). Isaac Lab's PinholeCameraCfg auto-computes this; USD
# does NOT — without setting it, USD defaults to ~15 mm and depth points
# come out stretched 5× vertically.
FIXED_CAM_VERTICAL_APERTURE = (
    FIXED_CAM_HORIZONTAL_APERTURE * FIXED_CAM_HEIGHT / FIXED_CAM_WIDTH
)

# Rack/logs target world position. Training had rack at world (2.0, -1.0, 0.0)
# and crane base at world (6.0, -2.92, 1.423), so rack-relative-to-base was
# (-4.0, 1.92, -1.423). Partner's crane base sits at world origin, so the
# rack must land at world ≈ (-4.0, 1.92, -1.423) to match the trained
# distribution. We compute the delta from the rack's current world pose
# and apply it to both `small_rack` and `Logs` so logs stay on the rack.
TRAINED_RACK_REL_TO_BASE = (-4.0, 1.92, -1.423)
RACK_PATH = "/World/Environment/small_rack"
LOGS_PATH = "/World/Environment/Logs"
PARTNER_BASE_PATH = "/World/fpiforwarder/base_link"

# Action bounds (base frame, meters) — what the trained policy decodes
# tanh outputs into. Drawn as a green wireframe AABB in the viewport
# so we can see whether the rack lands inside it.
ACTION_BOUNDS_MIN = (-5.0, -0.75, -1.373)
ACTION_BOUNDS_MAX = (-3.0, 4.59, -0.373)
VIZ_ROOT = "/World/Visuals"
ACTION_BBOX_PATH = f"{VIZ_ROOT}/ActionBoundsAABB"
TARGET_SPHERE_PATH = f"{VIZ_ROOT}/PolicyTarget"


def _set_input(prim, attr_name, value, sdf_type):
    """Create or update an inputs:* attribute on an OmniGraphNode prim."""
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        a = prim.CreateAttribute(attr_name, sdf_type, custom=False)
    a.Set(value)
    return a


def _connect(stage, prim, attr_name, source_path, sdf_type):
    """Add a connection on attr_name -> source_path (idempotent)."""
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        a = prim.CreateAttribute(attr_name, sdf_type, custom=False)
    targets = list(a.GetConnections())
    src = Sdf.Path(source_path)
    if src not in targets:
        targets.append(src)
        a.SetConnections(targets)
    return a


def patch(usd_path: str, save: bool = True) -> bool:
    print(f"Opening {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("ERROR: could not open stage", file=sys.stderr)
        return False

    sensors = stage.GetPrimAtPath(SENSORS_GRAPH)
    if not sensors:
        print(f"ERROR: {SENSORS_GRAPH} not found", file=sys.stderr)
        return False

    rgb_helper = stage.GetPrimAtPath(f"{SENSORS_GRAPH}/ros2_camera_helper")
    info_helper = stage.GetPrimAtPath(f"{SENSORS_GRAPH}/ros2_camera_info_helper")
    render_product = stage.GetPrimAtPath(f"{SENSORS_GRAPH}/isaac_create_render_product")
    if not all([rgb_helper, info_helper, render_product]):
        print("ERROR: expected RGB/info/render-product prims missing", file=sys.stderr)
        return False

    # ── 1. Update RGB helper (cosmetic — match naming convention) ────
    print("Updating ros2_camera_helper (RGB) → /zedx/rgb, frame=zedx_camera")
    _set_input(rgb_helper, "inputs:topicName", "/zedx/rgb", Sdf.ValueTypeNames.String)
    _set_input(rgb_helper, "inputs:frameId", "zedx_camera", Sdf.ValueTypeNames.String)
    _set_input(rgb_helper, "inputs:type", "rgb", Sdf.ValueTypeNames.Token)

    # ── 2. Update camera_info helper ─────────────────────────────────
    print("Updating ros2_camera_info_helper → /zedx/camera_info, frame=zedx_camera")
    _set_input(info_helper, "inputs:topicName", "/zedx/camera_info",
               Sdf.ValueTypeNames.String)
    _set_input(info_helper, "inputs:frameId", "zedx_camera",
               Sdf.ValueTypeNames.String)

    # ── 3. Add a depth helper alongside the RGB helper ───────────────
    depth_path = Sdf.Path(f"{SENSORS_GRAPH}/{DEPTH_HELPER_NAME}")
    depth_helper = stage.GetPrimAtPath(depth_path)
    if not depth_helper:
        print(f"Creating {depth_path}  (type={DEPTH_HELPER_TYPE})")
        depth_helper = stage.DefinePrim(depth_path, "OmniGraphNode")
        depth_helper.CreateAttribute("node:type", Sdf.ValueTypeNames.Token,
                                     custom=False).Set(DEPTH_HELPER_TYPE)
    else:
        print(f"{depth_path} already exists — updating in place")

    _set_input(depth_helper, "inputs:type", "depth", Sdf.ValueTypeNames.Token)
    _set_input(depth_helper, "inputs:topicName", "/zedx/depth", Sdf.ValueTypeNames.String)
    _set_input(depth_helper, "inputs:frameId", "zedx_camera", Sdf.ValueTypeNames.String)

    # ROS2CameraHelper schema: inputs:execIn is `uint` (omnigraph execution),
    # inputs:renderProductPath is `token`. The previous version used Token
    # for both — Isaac Sim rejected the node at load time. If a wrong-typed
    # attribute already exists, drop it before recreating with the right type.
    for attr_name, sdf_type in [
        ("inputs:execIn", Sdf.ValueTypeNames.UInt),
        ("inputs:renderProductPath", Sdf.ValueTypeNames.Token),
    ]:
        a = depth_helper.GetAttribute(attr_name)
        if a and a.IsValid() and a.GetTypeName() != sdf_type:
            print(f"   dropping wrong-typed {attr_name} ({a.GetTypeName()} -> {sdf_type})")
            depth_helper.RemoveProperty(attr_name)

    _connect(stage, depth_helper, "inputs:execIn",
             f"{render_product.GetPath()}.outputs:execOut",
             Sdf.ValueTypeNames.UInt)
    _connect(stage, depth_helper, "inputs:renderProductPath",
             f"{render_product.GetPath()}.outputs:renderProductPath",
             Sdf.ValueTypeNames.Token)

    # Register the new node in the OmniGraph's children list. Defining the
    # prim under the graph already does that — verify by checking presence.
    children = [c.GetName() for c in sensors.GetChildren()]
    if DEPTH_HELPER_NAME not in children:
        print(f"WARN: {DEPTH_HELPER_NAME} not in {SENSORS_GRAPH} children "
              f"({children}) — graph may not pick it up.")

    # ── 4. Fixed world-frame camera matching trained sim ─────────────
    print(f"Defining {FIXED_CAM_PATH}  (fixed-pose ZED X camera)")
    fixed_cam_prim = stage.GetPrimAtPath(FIXED_CAM_PATH)
    if not fixed_cam_prim:
        fixed_cam_prim = stage.DefinePrim(FIXED_CAM_PATH, "Camera")
    cam = UsdGeom.Camera(fixed_cam_prim)
    if not cam:
        print(f"   WARN: could not get Camera schema for {FIXED_CAM_PATH}")
    else:
        cam.GetFocalLengthAttr().Set(float(FIXED_CAM_FOCAL_LENGTH))
        cam.GetHorizontalApertureAttr().Set(float(FIXED_CAM_HORIZONTAL_APERTURE))
        cam.GetVerticalApertureAttr().Set(float(FIXED_CAM_VERTICAL_APERTURE))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(*FIXED_CAM_CLIPPING))
        cam.GetFocusDistanceAttr().Set(float(FIXED_CAM_FOCUS))

    # Apply translate + orient as xformOps. Clear any existing op order so
    # we don't accumulate translates on re-run.
    xformable = UsdGeom.Xformable(fixed_cam_prim)
    xformable.SetXformOpOrder([])
    xformable.ClearXformOpOrder()
    # Strip any leftover xformOp:* attributes from previous runs.
    for attr in list(fixed_cam_prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            fixed_cam_prim.RemoveProperty(attr.GetName())

    t_op = xformable.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*FIXED_CAM_POS))
    o_op = xformable.AddOrientOp()
    qw, qx, qy, qz = FIXED_CAM_QUAT_WXYZ
    o_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    print(f"   pos={FIXED_CAM_POS}, quat_wxyz={FIXED_CAM_QUAT_WXYZ}")
    print(f"   focal={FIXED_CAM_FOCAL_LENGTH}mm, aperture={FIXED_CAM_HORIZONTAL_APERTURE}mm, "
          f"clipping={FIXED_CAM_CLIPPING}, fov_h≈{2*57.3*math.atan(FIXED_CAM_HORIZONTAL_APERTURE/(2*FIXED_CAM_FOCAL_LENGTH)):.0f}°")

    # ── 5. Repoint the render product to the fixed camera ────────────
    print(f"Repointing {render_product.GetPath()}.inputs:cameraPrim -> {FIXED_CAM_PATH}")
    cam_rel = render_product.GetRelationship("inputs:cameraPrim")
    if not cam_rel:
        cam_rel = render_product.CreateRelationship("inputs:cameraPrim",
                                                    custom=False)
    cam_rel.SetTargets([Sdf.Path(FIXED_CAM_PATH)])

    # Match training resolution exactly (640x360).
    _set_input(render_product, "inputs:width", FIXED_CAM_WIDTH,
               Sdf.ValueTypeNames.UInt)
    _set_input(render_product, "inputs:height", FIXED_CAM_HEIGHT,
               Sdf.ValueTypeNames.UInt)

    # ── 5b. Move rack + logs to match trained-sim rack-relative-to-base ──
    base_prim = stage.GetPrimAtPath(PARTNER_BASE_PATH)
    rack_prim = stage.GetPrimAtPath(RACK_PATH)
    logs_prim = stage.GetPrimAtPath(LOGS_PATH)
    if not all([base_prim, rack_prim, logs_prim]):
        print(f"   WARN: missing rack/logs/base prim, skipping rack move")
    else:
        xfc = UsdGeom.XformCache()
        base_w = xfc.GetLocalToWorldTransform(base_prim).ExtractTranslation()
        rack_w = xfc.GetLocalToWorldTransform(rack_prim).ExtractTranslation()
        target_rack_w = Gf.Vec3d(
            base_w[0] + TRAINED_RACK_REL_TO_BASE[0],
            base_w[1] + TRAINED_RACK_REL_TO_BASE[1],
            base_w[2] + TRAINED_RACK_REL_TO_BASE[2],
        )
        delta = Gf.Vec3d(target_rack_w[0] - rack_w[0],
                         target_rack_w[1] - rack_w[1],
                         target_rack_w[2] - rack_w[2])
        print(f"Rack: world {tuple(round(c, 3) for c in rack_w)} -> "
              f"{tuple(round(c, 3) for c in target_rack_w)}  "
              f"(delta {tuple(round(c, 3) for c in delta)})")

        for path in (RACK_PATH, LOGS_PATH):
            p = stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(p)
            t_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    t_op = op
                    break
            if t_op is None:
                t_op = xf.AddTranslateOp()
                cur = Gf.Vec3d(0, 0, 0)
            else:
                cur = t_op.Get() or Gf.Vec3d(0, 0, 0)
            new_t = Gf.Vec3d(cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2])
            t_op.Set(new_t)
            print(f"   {path}.xformOp:translate: "
                  f"{tuple(round(c, 3) for c in cur)} -> "
                  f"{tuple(round(c, 3) for c in new_t)}")

    # ── 5c. Visualization markers (green AABB + red target sphere) ──
    # Both prims live under /World/Visuals so they don't interfere with
    # physics/articulation. The crane base is at world origin in the
    # partner's USD, so base-frame coords == world coords for these.
    viz_root = stage.GetPrimAtPath(VIZ_ROOT)
    if not viz_root:
        viz_root = UsdGeom.Xform.Define(stage, VIZ_ROOT).GetPrim()

    # Wireframe AABB: 12 thin cuboid edges (matches crane_rl_env_full.py
    # `bbox_cfg`). Lets you see through the box without relying on
    # displayOpacity (which Isaac Sim's viewport renders as fully opaque
    # without an MDL material). Single solid prim was unusable.
    extent = (
        ACTION_BOUNDS_MAX[0] - ACTION_BOUNDS_MIN[0],
        ACTION_BOUNDS_MAX[1] - ACTION_BOUNDS_MIN[1],
        ACTION_BOUNDS_MAX[2] - ACTION_BOUNDS_MIN[2],
    )
    EDGE_T = 0.03  # edge thickness, meters

    # Wipe the old solid-cube prim if present (from earlier patch revs).
    old = stage.GetPrimAtPath(ACTION_BBOX_PATH)
    if old:
        stage.RemovePrim(ACTION_BBOX_PATH)

    bbox_root = stage.GetPrimAtPath(ACTION_BBOX_PATH)
    if not bbox_root:
        bbox_root = UsdGeom.Xform.Define(stage, ACTION_BBOX_PATH).GetPrim()

    # Build 12 edges — 4 along x, 4 along y, 4 along z.
    edges = []
    xmin, ymin, zmin = ACTION_BOUNDS_MIN
    xmax, ymax, zmax = ACTION_BOUNDS_MAX
    cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax); cz = 0.5 * (zmin + zmax)
    for ay in (ymin, ymax):
        for az in (zmin, zmax):
            edges.append(("x", (cx, ay, az), (extent[0], EDGE_T, EDGE_T)))
    for ax in (xmin, xmax):
        for az in (zmin, zmax):
            edges.append(("y", (ax, cy, az), (EDGE_T, extent[1], EDGE_T)))
    for ax in (xmin, xmax):
        for ay in (ymin, ymax):
            edges.append(("z", (ax, ay, cz), (EDGE_T, EDGE_T, extent[2])))

    for i, (axis, pos, scale) in enumerate(edges):
        ep_path = f"{ACTION_BBOX_PATH}/edge_{axis}_{i}"
        ep = stage.GetPrimAtPath(ep_path)
        if not ep:
            ep = UsdGeom.Cube.Define(stage, ep_path).GetPrim()
        UsdGeom.Cube(ep).GetSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(ep)
        xf.SetXformOpOrder([]); xf.ClearXformOpOrder()
        for attr in list(ep.GetAttributes()):
            if attr.GetName().startswith("xformOp:"):
                ep.RemoveProperty(attr.GetName())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdGeom.Gprim(ep).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.95, 0.1)])
    center = (cx, cy, cz)
    print(f"Action-bounds wireframe AABB at {ACTION_BBOX_PATH}  "
          f"({len(edges)} edges, t={EDGE_T}m)  "
          f"center={tuple(round(c, 3) for c in center)}  extent={extent}")

    # Blue sphere for FSM-stepped EE command target. Updated by viz graph.
    ee_path = f"{VIZ_ROOT}/EECommandTarget"
    ee_prim = stage.GetPrimAtPath(ee_path)
    if not ee_prim:
        ee_prim = UsdGeom.Sphere.Define(stage, ee_path).GetPrim()
    UsdGeom.Sphere(ee_prim).GetRadiusAttr().Set(0.12)
    UsdGeom.Gprim(ee_prim).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.3, 1.0)])
    xf = UsdGeom.Xformable(ee_prim)
    xf.SetXformOpOrder([])
    xf.ClearXformOpOrder()
    for attr in list(ee_prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            ee_prim.RemoveProperty(attr.GetName())
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))

    # Red sphere placeholder for the policy target. Updated dynamically
    # by the viz OmniGraph (added below).
    sphere_prim = stage.GetPrimAtPath(TARGET_SPHERE_PATH)
    if not sphere_prim:
        sphere_prim = UsdGeom.Sphere.Define(stage, TARGET_SPHERE_PATH).GetPrim()
    sp = UsdGeom.Sphere(sphere_prim)
    sp.GetRadiusAttr().Set(0.15)
    UsdGeom.Gprim(sphere_prim).GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
    xf = UsdGeom.Xformable(sphere_prim)
    xf.SetXformOpOrder([])
    xf.ClearXformOpOrder()
    for attr in list(sphere_prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            sphere_prim.RemoveProperty(attr.GetName())
    # Start at the rack center (base frame) so it doesn't sit at origin.
    initial = (
        0.5 * (ACTION_BOUNDS_MIN[0] + ACTION_BOUNDS_MAX[0]),
        0.5 * (ACTION_BOUNDS_MIN[1] + ACTION_BOUNDS_MAX[1]),
        0.5 * (ACTION_BOUNDS_MIN[2] + ACTION_BOUNDS_MAX[2]),
    )
    xf.AddTranslateOp().Set(Gf.Vec3d(*initial))
    print(f"Policy-target sphere at {TARGET_SPHERE_PATH}  initial={initial}")

    # ── 5d. Viz OmniGraph: ScriptNode reads /tmp/crane_marker_state.json
    # each tick and writes the marker prims' translates. The standalone
    # policy node writes that file each cycle. File I/O is the bridge.
    viz_graph = stage.GetPrimAtPath(VIZ_GRAPH)
    if not viz_graph:
        viz_graph = stage.DefinePrim(VIZ_GRAPH, "OmniGraph")
    for name, sdf_type, val in [
        ("evaluationMode", Sdf.ValueTypeNames.Token, "Automatic"),
        ("evaluator:type", Sdf.ValueTypeNames.Token, "execution"),
        ("fabricCacheBacking", Sdf.ValueTypeNames.Token, "Shared"),
        ("pipelineStage", Sdf.ValueTypeNames.Token, "pipelineStageSimulation"),
    ]:
        a = viz_graph.GetAttribute(name)
        if not a or not a.IsValid():
            a = viz_graph.CreateAttribute(name, sdf_type, custom=False)
        a.Set(val)

    # Drop any stale child nodes from earlier patch revisions.
    for stale_child in ("ros2_subscribe_tf_markers",):
        stale = stage.GetPrimAtPath(f"{VIZ_GRAPH}/{stale_child}")
        if stale:
            stage.RemovePrim(stale.GetPath())
            print(f"   removed stale {stale_child}")

    # OnPlaybackTick: trigger node fires every sim frame.
    tick_path = f"{VIZ_GRAPH}/on_playback_tick"
    tick = stage.GetPrimAtPath(tick_path)
    if not tick:
        tick = stage.DefinePrim(tick_path, "OmniGraphNode")
    a = tick.GetAttribute("node:type")
    if not a or not a.IsValid():
        a = tick.CreateAttribute("node:type", Sdf.ValueTypeNames.Token, custom=False)
    a.Set("omni.graph.action.OnPlaybackTick")
    a = tick.GetAttribute("node:typeVersion")
    if not a or not a.IsValid():
        a = tick.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int, custom=False)
    a.Set(2)

    # ScriptNode: reads the JSON state file and writes marker prim transforms.
    script_path = f"{VIZ_GRAPH}/viz_script"
    script_node = stage.GetPrimAtPath(script_path)
    if not script_node:
        script_node = stage.DefinePrim(script_path, "OmniGraphNode")
    a = script_node.GetAttribute("node:type")
    if not a or not a.IsValid():
        a = script_node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token, custom=False)
    a.Set("omni.graph.scriptnode.ScriptNode")
    a = script_node.GetAttribute("node:typeVersion")
    if not a or not a.IsValid():
        a = script_node.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int, custom=False)
    a.Set(2)
    a = script_node.GetAttribute("inputs:script")
    if not a or not a.IsValid():
        a = script_node.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String, custom=False)
    a.Set(VIZ_SCRIPT)
    a = script_node.GetAttribute("inputs:usePath")
    if not a or not a.IsValid():
        a = script_node.CreateAttribute("inputs:usePath", Sdf.ValueTypeNames.Bool, custom=False)
    a.Set(False)

    a = script_node.GetAttribute("inputs:execIn")
    if a and a.IsValid() and a.GetTypeName() != Sdf.ValueTypeNames.UInt:
        script_node.RemoveProperty("inputs:execIn")
    _connect(stage, script_node, "inputs:execIn",
             f"{tick_path}.outputs:tick", Sdf.ValueTypeNames.UInt)
    print(f"Viz OmniGraph at {VIZ_GRAPH}  (tick → ScriptNode → marker prims)")

    # ── 6. Add fixed camera to TF tree publisher ─────────────────────
    tt_path = f"{CONTROL_GRAPH}/ros2_publish_transform_tree"
    tt = stage.GetPrimAtPath(tt_path)
    if tt:
        rel = tt.GetRelationship("inputs:targetPrims")
        if rel:
            targets = list(rel.GetTargets())
            new_target = Sdf.Path(FIXED_CAM_PATH)
            if new_target not in targets:
                targets.append(new_target)
                rel.SetTargets(targets)
                print(f"Added {FIXED_CAM_PATH} to {tt_path}.targetPrims")
            else:
                print(f"{FIXED_CAM_PATH} already in {tt_path}.targetPrims")

    if save:
        stage.GetRootLayer().Save()
        print(f"Saved {usd_path}")
    else:
        print("Skipping save (dry-run)")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=DEFAULT_USD,
                    help="Partner USD path (default: %(default)s)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Don't save changes — print what would be done.")
    args = ap.parse_args()
    if not Path(args.usd).exists():
        print(f"ERROR: {args.usd} not found", file=sys.stderr)
        return 1
    return 0 if patch(args.usd, save=not args.dry_run) else 2


if __name__ == "__main__":
    sys.exit(main())
