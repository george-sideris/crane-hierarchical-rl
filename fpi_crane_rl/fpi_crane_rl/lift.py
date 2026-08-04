#!/usr/bin/env python3
"""Lift the end-effector straight up by a given Z distance.

Reads the current basegrapple TF (in base_link frame), solves IK for
(x, y, z + dz) with the current grapplecarrier yaw held fixed, and calls
/relative_joint_move so the upstream planner generates a smooth trajectory
through the PLC.

Usage:  ros2 run fpi_crane_rl lift DZ [VELOCITY] [MIN_DUR]
"""
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from fpi_crane_msgs.srv import RelativeJointMove
from fpi_crane_rl import ik


ARM_JOINTS = ("slew_joint", "boom_joint", "stick_joint", "telescope_joint")
YAW_JOINT = "grapplecarrier_joint"
BASE_FRAME = "base_link"
EE_FRAME = "basegrapple"


def _resolve_urdf_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("fpi_crane_rl"),
                            "urdf", "fpiforwarder-upperpassive.urdf")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "urdf", "fpiforwarder-upperpassive.urdf")


class Lift(Node):
    def __init__(self, dz, velocity, min_dur):
        super().__init__("lift")
        self.ik = ik.CraneIK(_resolve_urdf_path())
        self.dz = float(dz)
        self.velocity = float(velocity)
        self.min_dur = float(min_dur)
        self.done = False
        self.client = self.create_client(RelativeJointMove, "relative_joint_move")
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info(
            f"Lift ready: dz={self.dz} velocity={self.velocity} min_dur={self.min_dur}")

    def on_joints(self, msg):
        if self.done:
            return
        try:
            arm = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]
            yaw_cur = float(msg.position[msg.name.index(YAW_JOINT)])
        except ValueError:
            return  # joint state not yet complete

        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, EE_FRAME, rclpy.time.Time())
        except Exception:
            return  # TF not ready yet

        cur_xyz = np.array([tf.transform.translation.x,
                            tf.transform.translation.y,
                            tf.transform.translation.z], dtype=float)
        target_xyz = cur_xyz + np.array([0.0, 0.0, self.dz])

        q = self.ik.solve(target_xyz.tolist(), arm)
        if q is None:
            self.get_logger().error(
                f"IK failed for target {target_xyz.tolist()} "
                f"(current {cur_xyz.tolist()}, dz={self.dz})")
            rclpy.shutdown()
            return

        deltas = [float(q[i] - arm[i]) for i in range(4)]
        deltas.append(0.0)  # hold current grapplecarrier yaw

        self.get_logger().info(
            f"Current xyz: ({cur_xyz[0]:+.3f}, {cur_xyz[1]:+.3f}, "
            f"{cur_xyz[2]:+.3f})")
        self.get_logger().info(
            f"Target xyz:  ({target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, "
            f"{target_xyz[2]:+.3f})")
        self.get_logger().info(
            f"Deltas:      {[f'{d:+.3f}' for d in deltas]}")

        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/relative_joint_move service not available")
            rclpy.shutdown()
            return

        req = RelativeJointMove.Request()
        req.sequence_id = 99
        req.joint_names = list(ARM_JOINTS) + [YAW_JOINT]
        req.delta_positions = deltas
        req.velocity = self.velocity
        req.min_duration = self.min_dur

        self.done = True
        future = self.client.call_async(req)
        future.add_done_callback(self.on_response)

    def on_response(self, future):
        r = future.result()
        self.get_logger().info(
            f"Sent: success={r.success}, duration={r.duration:.2f}s, "
            f"targets={[f'{p:.3f}' for p in r.target_positions]}, "
            f"message={r.message}")
        rclpy.shutdown()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dz = float(sys.argv[1])
    velocity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    min_dur = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    rclpy.init()
    node = Lift(dz, velocity, min_dur)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
