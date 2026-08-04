#!/usr/bin/env python3
"""Crane grasping policy node.

Subscribes to a depth camera + joint states, runs a learned point-cloud
policy, drives a 10-phase pick-and-place FSM, and forwards EE targets
to the joint-velocity controller via the SetTarget service.

Subscriptions:
    /zedx/depth         (sensor_msgs/Image, 32FC1, meters)
    /zedx/camera_info   (sensor_msgs/CameraInfo)
    /joint_states_sim   (sensor_msgs/JointState)
    /tf, /tf_static     (tf2_msgs/TFMessage)
    /crane/cycle_complete (std_msgs/Bool, optional)

Publications:
    /crane/policy_target (geometry_msgs/PoseStamped, once per cycle)
    /crane/ee_command   (geometry_msgs/PoseStamped, every tick)
    /crane/gripper_command (std_msgs/Float64)
    /crane/fsm_state    (std_msgs/String)

Service client:
    /crane/set_target   (sim_interface/SetTarget)

Run:
    python3 crane_policy_node.py \\
        --policy_type bcrl \\
        --checkpoint /path/to/model.pt \\
        --ros-args -p use_sim_time:=true
"""

import argparse
import json
import math
import os
import tempfile

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64, String, Bool
from tf2_ros import TransformBroadcaster

from pointcloud_pipeline import process_depth
from fsm import CraneFSM, FSMConfig, Phase
from policy_loader import load_policy
from sim_interface.srv import SetTarget, CheckTargetReached

# Default frame and joint names (override via ROS params)
WORLD_FRAME = "base_link"
BASE_FRAME = "base_link"
CAMERA_FRAME = "zedx_camera"
BASEGRAPPLE_FRAME = "basegrapple"
YAW_JOINT_NAME = "grapplecarrier_joint"


def _quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class CranePolicyNode(Node):
    def __init__(self, policy_type_override=None, checkpoint_override=None,
                 dry_run_override=False):
        super().__init__("crane_policy_node")

        # Parameters
        self.declare_parameter("policy_type", policy_type_override or "heuristic")
        self.declare_parameter("checkpoint_path", checkpoint_override or "")
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("depth_range_min", 1.0)
        self.declare_parameter("depth_range_max", 10.0)
        self.declare_parameter("cossin", True)
        self.declare_parameter("device",
                               "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("tick_rate", 10.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("max_cycles", 30)
        self.declare_parameter("bounds_min", [-5.0, -0.75, -1.373])
        self.declare_parameter("bounds_max", [-3.0, 4.59, -0.373])
        self.declare_parameter("hover_clear", 2.5)
        self.declare_parameter("approach_above", 0.6)
        self.declare_parameter("ee_tolerance", 0.10)
        self.declare_parameter("drop_position", [0.0, 2.2, 3.0])
        self.declare_parameter("camera_frame", CAMERA_FRAME)
        self.declare_parameter("base_frame", BASE_FRAME)
        self.declare_parameter("world_frame", WORLD_FRAME)
        self.declare_parameter("basegrapple_frame", BASEGRAPPLE_FRAME)
        self.declare_parameter("yaw_joint_name", YAW_JOINT_NAME)

        policy_type = self.get_parameter("policy_type").value
        checkpoint = self.get_parameter("checkpoint_path").value
        self.num_points = self.get_parameter("num_points").value
        self.depth_range = (self.get_parameter("depth_range_min").value,
                            self.get_parameter("depth_range_max").value)
        cossin = self.get_parameter("cossin").value
        device = self.get_parameter("device").value
        tick_rate = self.get_parameter("tick_rate").value
        self.dry_run = self.get_parameter("dry_run").value
        self.max_cycles = self.get_parameter("max_cycles").value
        bounds_min = np.array(self.get_parameter("bounds_min").value, dtype=np.float32)
        bounds_max = np.array(self.get_parameter("bounds_max").value, dtype=np.float32)

        self._camera_frame = self.get_parameter("camera_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._world_frame = self.get_parameter("world_frame").value
        self._basegrapple_frame = self.get_parameter("basegrapple_frame").value
        self._yaw_joint_name = self.get_parameter("yaw_joint_name").value

        # Camera/base extrinsics (overridden by TF lookup once available)
        self.cam_pos = np.array([-1.0, 1.92, 1.577], dtype=np.float32)
        self.cam_quat = np.array([0.6124, 0.3536, 0.3536, 0.6124], dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Policy
        ckpt = checkpoint if checkpoint else None
        self.policy = load_policy(policy_type, ckpt, bounds_min, bounds_max,
                                  cossin, device)
        self.get_logger().info(
            f"Policy: {policy_type}" +
            (f" from {checkpoint}" if checkpoint else "") +
            f"  bounds_min={bounds_min.round(3).tolist()} "
            f"bounds_max={bounds_max.round(3).tolist()}")

        # FSM
        fsm_cfg = FSMConfig(
            hover_clear=self.get_parameter("hover_clear").value,
            approach_above=self.get_parameter("approach_above").value,
            ee_tolerance=self.get_parameter("ee_tolerance").value,
            drop_position=self.get_parameter("drop_position").value,
        )
        self.fsm = CraneFSM(fsm_cfg)

        # State
        self.intrinsics = None
        self.latest_depth = None
        self.current_ee_pos = [0.0, 0.0, 0.0]
        self.current_ee_yaw = 0.0
        self.current_yaw_joint = 0.0
        self._last_cmd_yaw = None
        self._cached_joint_yaw = 0.0
        self.current_gripper = 0.0
        self.waiting_for_target = True
        self._tf_initialized = False

        # cv_bridge for depth decode
        try:
            from cv_bridge import CvBridge
            self._cv_bridge = CvBridge()
        except ImportError:
            self._cv_bridge = None

        # Subscribers
        self.create_subscription(Image, "/zedx/depth", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/zedx/camera_info",
                                 self._caminfo_cb, 10)
        self.create_subscription(JointState, "/joint_states_sim",
                                 self._joint_states_cb, 10)
        self.create_subscription(Bool, "/crane/cycle_complete",
                                 self._cycle_complete_cb, 10)

        # TF (basegrapple pose plus camera/base extrinsics)
        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        # TF broadcaster for the policy_target and ee_command marker
        # frames consumed by the viz graph.
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.pub_ee_cmd = self.create_publisher(PoseStamped, "/crane/ee_command", 10)
        self.pub_gripper = self.create_publisher(Float64, "/crane/gripper_command", 10)
        self.pub_fsm_state = self.create_publisher(String, "/crane/fsm_state", 10)
        self.pub_policy_target = self.create_publisher(PoseStamped,
                                                       "/crane/policy_target", 10)

        # Service client to JV controller
        self.set_target_client = self.create_client(SetTarget, "/crane/set_target")
        self.check_target_client = self.create_client(CheckTargetReached,
                                                      "/crane/is_target_reached")

        self.create_timer(1.0 / tick_rate, self._tick)
        self.get_logger().info(
            f"Started (tick={tick_rate}Hz, dry_run={self.dry_run})")

    # Callbacks

    def _depth_cb(self, msg):
        if self._cv_bridge:
            self.latest_depth = self._cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding="32FC1")
        else:
            self.latest_depth = np.frombuffer(msg.data, dtype=np.float32) \
                .reshape(msg.height, msg.width)

    def _caminfo_cb(self, msg):
        if self.intrinsics is None:
            K = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.intrinsics = K
            self.get_logger().info(
                f"Camera intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    def _joint_states_cb(self, msg):
        # Yaw joint position used by _grapple_yaw_to_joint
        try:
            idx = list(msg.name).index(self._yaw_joint_name)
            self.current_yaw_joint = float(msg.position[idx])
        except (ValueError, IndexError):
            pass
        # Gripper state from one of the tong joints (USD names)
        for n in ("grappletong1_joint", "grappletong2_joint"):
            try:
                idx = list(msg.name).index(n)
                self.current_gripper = float(msg.position[idx])
                break
            except ValueError:
                continue

    def _cycle_complete_cb(self, msg):
        if msg.data:
            self.waiting_for_target = True

    # TF helpers

    def _update_extrinsics_from_tf(self):
        """Look up cam/base extrinsics in world_frame once TF is available."""
        if self._tf_initialized:
            return
        try:
            t_cam = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, rclpy.time.Time())
            p, q = t_cam.transform.translation, t_cam.transform.rotation
            self.cam_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.cam_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            t_base = self._tf_buffer.lookup_transform(
                self._world_frame, self._base_frame, rclpy.time.Time())
            p, q = t_base.transform.translation, t_base.transform.rotation
            self.base_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.base_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            self._tf_initialized = True
            self.get_logger().info(
                f"TF loaded: cam_pos={self.cam_pos.round(3).tolist()}, "
                f"base_pos={self.base_pos.round(3).tolist()}")
        except Exception:
            pass

    def _update_basegrapple_from_tf(self):
        """Refresh basegrapple position + yaw in base frame via TF."""
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame, self._basegrapple_frame, rclpy.time.Time())
        except Exception:
            return
        p = t.transform.translation
        q = t.transform.rotation
        self.current_ee_pos = [p.x, p.y, p.z]
        self.current_ee_yaw = _quat_to_yaw(q.w, q.x, q.y, q.z)

    # Yaw conversion (basegrapple frame to yaw joint)

    def _grapple_yaw_to_joint(self, target_grapple_yaw_b: float) -> float:
        def _wrap(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        rotation_diff = _wrap(target_grapple_yaw_b - self.current_ee_yaw)
        base = self.current_yaw_joint + rotation_diff
        candidates = [base, base + math.pi, base - math.pi]
        return min(candidates, key=lambda c: abs(c - self.current_yaw_joint))

    # Main tick

    def _tick(self):
        if self.fsm.cycle_count >= self.max_cycles:
            return
        self._update_extrinsics_from_tf()
        self._update_basegrapple_from_tf()

        if self.waiting_for_target:
            if self.latest_depth is None or self.intrinsics is None:
                return
            obs = process_depth(
                self.latest_depth, self.intrinsics,
                self.cam_pos, self.cam_quat,
                self.base_pos, self.base_quat,
                self.num_points, self.depth_range,
            )
            x, y, z, yaw = self.policy.get_target(obs)
            self.get_logger().info(
                f"Cycle {self.fsm.cycle_count + 1}: target=({x:.2f}, {y:.2f}, "
                f"{z:.2f}), yaw={math.degrees(yaw):.1f}deg")
            self._publish_target(x, y, z, yaw)
            self.fsm.set_target(x, y, z, yaw)
            self.waiting_for_target = False

        pre_phase = self.fsm.phase
        cmd = self.fsm.tick(self.current_ee_pos, self.current_ee_yaw,
                            self.current_gripper)

        if not self.dry_run:
            phase_changed = (pre_phase != self.fsm.phase)
            yaw_changed = (self._last_cmd_yaw is None
                           or abs(cmd.yaw - self._last_cmd_yaw) > 1e-4)
            if phase_changed or yaw_changed:
                self._cached_joint_yaw = self._grapple_yaw_to_joint(cmd.yaw)
                self._last_cmd_yaw = cmd.yaw
            self._publish_ee_command(cmd.position, cmd.yaw)
            self._publish_gripper(cmd.gripper)
            self._send_target_to_jv(cmd.position, self._cached_joint_yaw,
                                    cmd.gripper)

        state_msg = String()
        state_msg.data = f"{cmd.phase.name} (cycle {self.fsm.cycle_count})"
        self.pub_fsm_state.publish(state_msg)

        if cmd.cycle_complete:
            self.get_logger().info(f"Cycle {self.fsm.cycle_count} complete")
            self.waiting_for_target = True

    # Service / publisher helpers

    def _send_target_to_jv(self, pos, yaw, gripper):
        if not self.set_target_client.service_is_ready():
            if not hasattr(self, '_jv_warn_printed'):
                self.get_logger().warn(
                    "JV service /crane/set_target not available. "
                    "Is jv_controller_node running?")
                self._jv_warn_printed = True
            return
        new_target = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                      round(yaw, 3), round(gripper, 3))
        if hasattr(self, '_last_jv_target') and self._last_jv_target == new_target:
            return
        self._last_jv_target = new_target
        req = SetTarget.Request()
        req.target_xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
        req.target_yaw = float(yaw)
        req.grapple_opening = float(gripper)
        self.set_target_client.call_async(req)

    def _broadcast_marker_tf(self, child_frame, x, y, z, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._base_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.w = math.cos(yaw / 2)
        t.transform.rotation.z = math.sin(yaw / 2)
        self._tf_broadcaster.sendTransform(t)

    def _write_marker_state(self, key, x, y, z):
        # Marker positions for the viz graph, written via an atomic
        # rename so the reader never sees a half-written file.
        path = "/tmp/crane_marker_state.json"
        try:
            state = {}
            if os.path.exists(path):
                with open(path) as f:
                    state = json.load(f)
        except Exception:
            state = {}
        state[key] = [float(x), float(y), float(z)]
        fd, tmp = tempfile.mkstemp(prefix=".crane_marker_", dir="/tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def _publish_ee_command(self, pos, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_ee_cmd.publish(msg)
        self._broadcast_marker_tf("ee_command_marker", pos[0], pos[1], pos[2], yaw)
        self._write_marker_state("ee", pos[0], pos[1], pos[2])

    def _publish_gripper(self, opening):
        msg = Float64()
        msg.data = float(opening)
        self.pub_gripper.publish(msg)

    def _publish_target(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_policy_target.publish(msg)
        self._broadcast_marker_tf("policy_target_marker", x, y, z, yaw)
        self._write_marker_state("policy", x, y, z)

def main():
    parser = argparse.ArgumentParser(description="Crane policy node")
    parser.add_argument("--policy_type", default="heuristic")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dry_run", action="store_true")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CranePolicyNode(
        policy_type_override=args.policy_type,
        checkpoint_override=args.checkpoint,
        dry_run_override=args.dry_run,
    )
    if args.dry_run:
        node.dry_run = True
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Completed {node.fsm.cycle_count} cycles")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
