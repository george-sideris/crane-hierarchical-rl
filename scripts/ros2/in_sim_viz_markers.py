"""Run inside Isaac Sim's Script Editor to update the red policy-target sphere.

The static markers (green action-bounds AABB + red placeholder sphere) are
baked into the USD by tools/patch_partner_usd.py. This script keeps the
red sphere tracking the policy's per-cycle grasp target in real time —
same look as crane_rl_env_full.py's --viz_markers in the original env.

How to use:
1. Open Isaac Sim, load log_loader_crane_lab.usd, press Play.
2. Window → Script Editor.
3. File → Open → /workspace/crane_testbed/scripts/ros2/in_sim_viz_markers.py
4. Press Run.

Stop with the Stop button in the Script Editor when done. The script
runs in Isaac Sim's bundled Python (3.11), not system Python — uses
Isaac Sim's bundled rclpy, so it doesn't need env_setup.sh sourced.

Subscribes to:
  /crane/policy_target  (geometry_msgs/PoseStamped)  — what the policy chose
  /crane/ee_command     (geometry_msgs/PoseStamped)  — where the FSM is steering

Updates:
  /World/Visuals/PolicyTarget        translation (red, policy decision)
  /World/Visuals/EECommandTarget     translation (blue, FSM-stepped, created if missing)
"""

import omni.usd
import rclpy
from geometry_msgs.msg import PoseStamped
from pxr import Gf, Sdf, UsdGeom


VIZ_ROOT = "/World/Visuals"
POLICY_TARGET_PATH = f"{VIZ_ROOT}/PolicyTarget"
EE_TARGET_PATH = f"{VIZ_ROOT}/EECommandTarget"


def _ensure_translatable(stage, path, default_pos, color):
    """Create the prim if missing, ensure it has a translate xformOp, return the op."""
    prim = stage.GetPrimAtPath(path)
    if not prim:
        prim = UsdGeom.Sphere.Define(stage, path).GetPrim()
        UsdGeom.Sphere(prim).GetRadiusAttr().Set(0.15)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    op = xf.AddTranslateOp()
    op.Set(Gf.Vec3d(*default_pos))
    return op


def main():
    if not rclpy.ok():
        rclpy.init()

    stage = omni.usd.get_context().get_stage()
    policy_op = _ensure_translatable(
        stage, POLICY_TARGET_PATH, (-4.0, 1.92, -0.873), (1.0, 0.0, 0.0))
    ee_op = _ensure_translatable(
        stage, EE_TARGET_PATH, (-4.0, 1.92, -0.873), (0.1, 0.3, 1.0))

    node = rclpy.create_node("in_sim_viz_markers")

    def _on_policy(msg: PoseStamped):
        p = msg.pose.position
        policy_op.Set(Gf.Vec3d(p.x, p.y, p.z))

    def _on_ee(msg: PoseStamped):
        p = msg.pose.position
        ee_op.Set(Gf.Vec3d(p.x, p.y, p.z))

    node.create_subscription(PoseStamped, "/crane/policy_target", _on_policy, 10)
    node.create_subscription(PoseStamped, "/crane/ee_command", _on_ee, 10)
    print("[viz] Subscribed: /crane/policy_target (red), /crane/ee_command (blue)")
    print("[viz] Spinning. Stop the script in the editor to halt.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


main()
