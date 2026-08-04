#!/usr/bin/env python3
"""Bridge between FPI's partner USD scene and our deployment nodes.

The partner's USD (log_loader_crane_lab.usd) ships its own action graphs
that publish ROS2 topics directly from Isaac Sim — no Isaac Lab needed.
This node fills the gaps between what the USD publishes and what
crane_policy_node / jv_controller_node expect:

    USD                                this bridge                                    our nodes
    ───                                ───────────                                    ─────────
    /joint_states_sim    ─►   rename joints, retime    ─►   /crane/joint_states   (sub by jv_controller, policy)
    /crane/joint_command ◄─   rename joints back        ◄─   /crane/joint_command  (pub by jv_controller)
                              published as /joint_command  (USD subscribes)

    /tf (live)           ─►   compute basegrapple in    ─►   /crane/ee_pos_base
                              base frame, publish yaw   ─►   /crane/grapple_yaw_base

                              static frame aliases      ─►   /tf_static
                              (base_link↔crane_base,
                               CameraLeft↔zedx_camera,
                               world↔base_link)

                              latched, hardcoded        ─►   /crane/action_bounds

Run after starting Isaac Sim with the patched USD:

    source /workspace/crane_testbed/scripts/ros2/env_setup.sh
    /usr/bin/python3 /workspace/crane_testbed/scripts/ros2/usd_bridge_node.py

Then start the controller and policy nodes as usual — no extra remappings
needed; this node makes the USD scene look identical to sim_ros2_env.py
from the consumer side.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64
from tf2_msgs.msg import TFMessage

from tf2_ros import Buffer, TransformException, TransformListener


# ── Joint name maps (USD ↔ ours) ──────────────────────────────────────
# Active joints (controlled or feedback). Anything not in here is
# forwarded with its original name.
USD_TO_OURS = {
    "slew_joint": "basemast_to_mast",
    "boom_joint": "mast_to_mainboom",
    "stick_joint": "mainboom_to_stick",
    "telescope_joint": "stick_to_telescope",
    "grapplecarrier_joint": "lowerpassive_to_basegrapple",
    "grappletong1_joint": "basegrapple_to_gripperleft",
    "grappletong2_joint": "basegrapple_to_gripperright",
}
OURS_TO_USD = {v: k for k, v in USD_TO_OURS.items()}


def _quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Extract yaw (rotation about Z) from a wxyz quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class UsdBridgeNode(Node):
    def __init__(self):
        super().__init__("usd_bridge_node")

        # ── Parameters ────────────────────────────────────────────────
        # Frame names in the partner's USD vs. ours. Defaults match the
        # patched USD (post tools/patch_partner_usd.py) and our policy
        # node defaults.
        self.declare_parameter("usd_base_frame", "base_link")
        self.declare_parameter("our_base_frame", "crane_base")
        self.declare_parameter("usd_camera_frame", "CameraLeft")
        self.declare_parameter("our_camera_frame", "zedx_camera")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("basegrapple_frame", "basegrapple")

        # Action bounds the deployed policy was trained against. Override
        # via ROS params if the rack location changes.
        self.declare_parameter("action_bounds_min", [-5.0, -0.75, -1.373])
        self.declare_parameter("action_bounds_max", [-3.0, 4.59, -0.373])

        self.usd_base = self.get_parameter("usd_base_frame").value
        self.our_base = self.get_parameter("our_base_frame").value
        self.usd_cam = self.get_parameter("usd_camera_frame").value
        self.our_cam = self.get_parameter("our_camera_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        self.basegrapple_frame = self.get_parameter("basegrapple_frame").value
        bounds_min = list(self.get_parameter("action_bounds_min").value)
        bounds_max = list(self.get_parameter("action_bounds_max").value)

        # ── TF listener for /crane/ee_pos_base + /crane/grapple_yaw_base ──
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Joint state remap ─────────────────────────────────────────
        self.create_subscription(JointState, "/joint_states_sim",
                                 self._joint_state_cb, 10)
        self.pub_joint_states = self.create_publisher(
            JointState, "/crane/joint_states", 10)

        # ── Joint command remap ───────────────────────────────────────
        # jv_controller publishes our names on /crane/joint_command;
        # we rename to USD names and republish on /joint_command (which
        # the USD's ros2_subscribe_joint_state listens on by default).
        self.create_subscription(JointState, "/crane/joint_command",
                                 self._joint_command_cb, 10)
        self.pub_joint_command = self.create_publisher(
            JointState, "/joint_command", 10)

        # ── Synthesized topics for the policy node ────────────────────
        self.pub_ee_pos = self.create_publisher(
            PointStamped, "/crane/ee_pos_base", 10)
        self.pub_grapple_yaw = self.create_publisher(
            Float64, "/crane/grapple_yaw_base", 10)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_action_bounds = self.create_publisher(
            Float32MultiArray, "/crane/action_bounds", latched)
        self.pub_tf_static = self.create_publisher(
            TFMessage, "/tf_static", latched)
        # Live /tf for the policy_target_marker / ee_command_marker frames
        # consumed by the in-USD ROS2SubscribeTransformTree node (drives
        # the red/blue marker spheres). Standard reliable QoS — matches /tf.
        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.create_subscription(PoseStamped, "/crane/policy_target",
                                 self._policy_target_cb, 10)
        self.create_subscription(PoseStamped, "/crane/ee_command",
                                 self._ee_command_cb, 10)

        # ── Publish action bounds once (latched) ──────────────────────
        msg = Float32MultiArray()
        msg.data = [float(v) for v in bounds_min + bounds_max]
        self.pub_action_bounds.publish(msg)
        self.get_logger().info(
            f"Latched /crane/action_bounds: min={bounds_min}, max={bounds_max}")

        # ── Publish frame aliases on /tf_static ───────────────────────
        # base_link → crane_base (identity), CameraLeft → zedx_camera
        # (identity), and world → base_link (identity) so the policy
        # node's lookup_transform(world, ...) works without modification.
        now = self.get_clock().now().to_msg()
        tf_msg = TFMessage()
        for parent, child in [
            (self.usd_base, self.our_base),
            (self.usd_cam, self.our_cam),
            (self.world_frame, self.usd_base),
        ]:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.rotation.w = 1.0
            tf_msg.transforms.append(t)
        self.pub_tf_static.publish(tf_msg)
        self.get_logger().info(
            f"Latched /tf_static aliases: "
            f"{self.world_frame}->{self.usd_base}, "
            f"{self.usd_base}->{self.our_base}, "
            f"{self.usd_cam}->{self.our_cam}")

        # ── Periodic poll for basegrapple pose (TF lookup) ────────────
        # The USD publishes /tf at sim tick rate; we just consume it.
        self.create_timer(0.05, self._publish_basegrapple_pose)  # 20 Hz

    # ── Joint name remap ──────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        out.name = [USD_TO_OURS.get(n, n) for n in msg.name]
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub_joint_states.publish(out)

    def _joint_command_cb(self, msg: JointState):
        out = JointState()
        out.header = msg.header
        out.name = [OURS_TO_USD.get(n, n) for n in msg.name]
        out.position = list(msg.position)
        out.velocity = list(msg.velocity)
        out.effort = list(msg.effort)
        self.pub_joint_command.publish(out)

    # ── Marker TF broadcasts ──────────────────────────────────────────
    # The OmniGraph ROS2SubscribeTransformTree node in the patched USD
    # listens for these frames and writes them to /World/Visuals/* prims.

    def _publish_marker_tf(self, child_frame: str, x: float, y: float, z: float):
        # Publish under both `world` and `base_link` so Isaac Sim's
        # ROS2SubscribeTransformTree can resolve regardless of which root
        # it's anchored on. Marker prims sit under /World, and partner's
        # base is at world origin, so world-relative xyz == base-relative.
        msg = TFMessage()
        for parent in (self.world_frame, self.usd_base):
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent
            t.child_frame_id = child_frame
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z)
            t.transform.rotation.w = 1.0
            msg.transforms.append(t)
        self.pub_tf.publish(msg)

    def _policy_target_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._publish_marker_tf("policy_target_marker", p.x, p.y, p.z)

    def _ee_command_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._publish_marker_tf("ee_command_marker", p.x, p.y, p.z)

    # ── Basegrapple pose from TF ──────────────────────────────────────

    def _publish_basegrapple_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self.our_base, self.basegrapple_frame, rclpy.time.Time())
        except TransformException:
            return  # not available yet
        pt = PointStamped()
        pt.header.stamp = t.header.stamp
        pt.header.frame_id = self.our_base
        pt.point.x = t.transform.translation.x
        pt.point.y = t.transform.translation.y
        pt.point.z = t.transform.translation.z
        self.pub_ee_pos.publish(pt)

        q = t.transform.rotation
        yaw_msg = Float64()
        yaw_msg.data = _quat_to_yaw(q.w, q.x, q.y, q.z)
        self.pub_grapple_yaw.publish(yaw_msg)


def main():
    rclpy.init()
    node = UsdBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
