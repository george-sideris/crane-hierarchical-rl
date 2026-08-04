#!/usr/bin/env python3
"""Minimal ROS2 publisher: send known joint positions to /crane/joint_command.

Run alongside sim_ros2_env.py to test if the crane moves to target positions.

Usage (system Python, separate terminal):
    source /opt/ros/humble/setup.bash
    /usr/bin/python3 /workspace/crane_testbed/scripts/ros2/test_joint_pub.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time


class JointTestPublisher(Node):
    def __init__(self):
        super().__init__("joint_test_publisher")

        self.pub = self.create_publisher(JointState, "/crane/joint_command", 10)

        # Same joint names as the RL env
        self.joint_names = [
            "basemast_to_mast",
            "mast_to_mainboom",
            "mainboom_to_stick",
            "stick_to_telescope",
        ]

        # Target positions — moderate offsets from default
        # Default positions are roughly [0, 0.5, -0.8, 0] (depends on URDF)
        # We'll send a sequence of targets to see the crane move
        self.targets = [
            [0.2, 0.6, -0.5, 0.3],   # move all joints a bit
            [0.0, 0.3, -1.0, 0.0],   # move back toward defaults
            [-0.2, 0.8, -0.3, 0.5],  # different pose
        ]
        self.current_target_idx = 0
        self.steps_at_target = 0
        self.hold_steps = 300  # hold each target for ~10s at 30Hz

        self.timer = self.create_timer(1.0 / 30.0, self.publish_cmd)
        self.get_logger().info(f"Publishing joint position targets to /crane/joint_command")
        self.get_logger().info(f"Target 0: {self.targets[0]}")

    def publish_cmd(self):
        target = self.targets[self.current_target_idx]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(p) for p in target]
        msg.velocity = []  # empty — sim should use position
        msg.effort = []
        self.pub.publish(msg)

        self.steps_at_target += 1
        if self.steps_at_target % 30 == 0:
            self.get_logger().info(
                f"Target {self.current_target_idx}: {target}  "
                f"({self.steps_at_target}/{self.hold_steps})"
            )

        if self.steps_at_target >= self.hold_steps:
            self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
            self.steps_at_target = 0
            self.get_logger().info(
                f"Switching to target {self.current_target_idx}: "
                f"{self.targets[self.current_target_idx]}"
            )


def main():
    rclpy.init()
    node = JointTestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
