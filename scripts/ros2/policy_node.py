#!/usr/bin/env python3
"""Policy node (v2) — perception + policy only, publishes grasp targets.

No FSM, no control logic. Works with sequencer_node.py + jv_controller_node.py.

Subscribes:
    /zedx/depth            (Image)         — depth from camera
    /zedx/camera_info      (CameraInfo)    — camera intrinsics
    /crane/cycle_complete   (Bool)          — sequencer signals cycle done
    /tf_static                              — camera + base transforms

Publishes:
    /crane/grasp_target    (PoseStamped)   — policy's chosen grasp target

Usage:
    python3 policy_node.py --policy_type heuristic
    python3 policy_node.py --policy_type bcrl --checkpoint /path/to/model.pt
"""

import argparse
import math
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from pointcloud_pipeline import process_depth
from policy_loader import load_policy


class PolicyNode(Node):
    def __init__(self, policy_type="heuristic", checkpoint=None, device="auto"):
        super().__init__("policy_node")

        # Parameters
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("depth_range_min", 1.0)
        self.declare_parameter("depth_range_max", 10.0)
        self.declare_parameter("cossin", True)
        self.declare_parameter("max_cycles", 30)
        self.declare_parameter("bounds_min", [-2.0, -3.0, -0.5])
        self.declare_parameter("bounds_max", [2.0, 3.0, 1.5])
        self.declare_parameter("cam_pos", [5.0, -1.0, 3.0])
        self.declare_parameter("cam_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("base_pos", [0.0, 0.0, 0.0])
        self.declare_parameter("base_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("camera_frame", "zedx_camera")
        self.declare_parameter("base_frame", "crane_base")
        self.declare_parameter("world_frame", "world")

        self.num_points = self.get_parameter("num_points").value
        self.depth_range = (self.get_parameter("depth_range_min").value,
                           self.get_parameter("depth_range_max").value)
        self.max_cycles = self.get_parameter("max_cycles").value

        bounds_min = np.array(self.get_parameter("bounds_min").value, dtype=np.float32)
        bounds_max = np.array(self.get_parameter("bounds_max").value, dtype=np.float32)

        # Camera and base transforms (defaults, overridden by TF)
        cam_pos = np.array(self.get_parameter("cam_pos").value, dtype=np.float32)
        cam_rpy = np.deg2rad(self.get_parameter("cam_rpy").value)
        self.cam_pos = cam_pos
        self.cam_rot = self._rpy_to_matrix(cam_rpy)
        base_pos = np.array(self.get_parameter("base_pos").value, dtype=np.float32)
        base_rpy = np.deg2rad(self.get_parameter("base_rpy").value)
        self.base_pos = base_pos
        self.base_rot = self._rpy_to_matrix(base_rpy)

        # Policy
        cossin = self.get_parameter("cossin").value
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = load_policy(policy_type, checkpoint, bounds_min, bounds_max, cossin, device)
        self.get_logger().info(f"Policy: {policy_type}" +
                               (f" from {checkpoint}" if checkpoint else ""))

        # State
        self.intrinsics = None
        self.latest_depth = None
        self.waiting_for_target = True
        self.cycle_count = 0
        self._tf_initialized = False

        # Subscribers
        self.create_subscription(Image, "/zedx/depth", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/zedx/camera_info", self._caminfo_cb, 10)
        self.create_subscription(Bool, "/crane/cycle_complete", self._cycle_complete_cb, 10)

        # TF listener
        try:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_available = True
            self.get_logger().info("TF listener started")
        except ImportError:
            self._tf_available = False

        self._camera_frame = self.get_parameter("camera_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._world_frame = self.get_parameter("world_frame").value

        # Publisher
        self.pub_target = self.create_publisher(PoseStamped, "/crane/grasp_target", 10)

        # Timer
        self.create_timer(1.0 / 10.0, self._tick)  # 10Hz is enough for perception
        self.get_logger().info("Policy node started, waiting for depth + TF")

    # ── Callbacks ─────────────────────────────────────────────────────

    def _depth_cb(self, msg):
        self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    def _caminfo_cb(self, msg):
        if self.intrinsics is None:
            K = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.intrinsics = K
            self.get_logger().info(f"Camera intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    def _cycle_complete_cb(self, msg):
        if msg.data:
            self.cycle_count += 1
            self.waiting_for_target = True
            self.get_logger().info(f"Cycle {self.cycle_count} complete, getting next target")

    def _update_tf(self):
        if not self._tf_available or self._tf_initialized:
            return
        try:
            t_cam = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, rclpy.time.Time())
            p = t_cam.transform.translation
            q = t_cam.transform.rotation
            self.cam_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.cam_rot = self._quat_to_matrix(
                np.array([q.w, q.x, q.y, q.z], dtype=np.float32))

            t_base = self._tf_buffer.lookup_transform(
                self._world_frame, self._base_frame, rclpy.time.Time())
            p2 = t_base.transform.translation
            q2 = t_base.transform.rotation
            self.base_pos = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
            self.base_rot = self._quat_to_matrix(
                np.array([q2.w, q2.x, q2.y, q2.z], dtype=np.float32))

            self._tf_initialized = True
            self.get_logger().info(f"TF loaded: cam_pos={self.cam_pos}, base_pos={self.base_pos}")
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _tick(self):
        if self.cycle_count >= self.max_cycles:
            return

        self._update_tf()

        if not self.waiting_for_target:
            return

        # Wait for TF + depth + intrinsics before first target
        if self.latest_depth is None or self.intrinsics is None or not self._tf_initialized:
            return

        # Run point cloud pipeline
        obs = process_depth(
            self.latest_depth, self.intrinsics,
            self.cam_pos, self.cam_rot,
            self.base_pos, self.base_rot,
            self.num_points, self.depth_range,
        )

        # Run policy
        x, y, z, yaw = self.policy.get_target(obs)
        self.get_logger().info(
            f"Cycle {self.cycle_count + 1}: target=({x:.2f}, {y:.2f}, {z:.2f}), "
            f"yaw={math.degrees(yaw):.1f}deg")

        # Publish grasp target
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "crane_base"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_target.publish(msg)

        self.waiting_for_target = False  # wait for cycle_complete

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_matrix(quat_wxyz):
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    @staticmethod
    def _rpy_to_matrix(rpy):
        r, p, y = rpy
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Policy node (v2)")
    parser.add_argument("--policy_type", default="heuristic")
    parser.add_argument("--checkpoint", default=None)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PolicyNode(
        policy_type=args.policy_type,
        checkpoint=args.checkpoint,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Completed {node.cycle_count} cycles")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
