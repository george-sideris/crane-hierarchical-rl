#!/usr/bin/env python3
"""ROS2 bridge receiver: reads sim data from TCP socket and publishes to ROS2 topics.

Runs with system Python 3.10 + rclpy (NOT Isaac Lab's Python 3.11).

Usage:
    source /opt/ros/humble/setup.bash
    python3 sim_bridge_receiver.py
"""

import math
import socket
import struct
import pickle
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64


class SimBridgeReceiver(Node):
    def __init__(self, host="127.0.0.1", port=9876):
        super().__init__("sim_bridge_receiver")
        self.pub_depth = self.create_publisher(Image, "/zedx/depth", 10)
        self.pub_cam_info = self.create_publisher(CameraInfo, "/zedx/camera_info", 10)
        self.pub_ee_state = self.create_publisher(PoseStamped, "/crane/ee_state", 10)
        self.pub_gripper = self.create_publisher(Float64, "/crane/gripper_state", 10)

        self.host = host
        self.port = port
        self.running = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
        self.get_logger().info(f"Connecting to sim bridge at {host}:{port}")

    def _recv_loop(self):
        """Connect to sim sender and receive data in a loop."""
        while self.running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                self.get_logger().info("Connected to sim bridge")

                while self.running:
                    # Read 4-byte length header
                    header = self._recv_exact(sock, 4)
                    if header is None:
                        break
                    length = struct.unpack("!I", header)[0]

                    # Read payload
                    payload = self._recv_exact(sock, length)
                    if payload is None:
                        break

                    data = pickle.loads(payload)
                    self._publish(data)

            except ConnectionRefusedError:
                import time
                time.sleep(1.0)  # retry connection
            except Exception as e:
                self.get_logger().warn(f"Bridge error: {e}")
                import time
                time.sleep(1.0)

    def _recv_exact(self, sock, n):
        """Receive exactly n bytes."""
        buf = b""
        while len(buf) < n:
            try:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except socket.timeout:
                continue
        return buf

    def _publish(self, data):
        """Publish received sim data to ROS2 topics."""
        now = self.get_clock().now().to_msg()

        # Depth image
        depth = data["depth"]
        depth_msg = Image()
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "zedx_depth"
        depth_msg.height, depth_msg.width = depth.shape
        depth_msg.encoding = "32FC1"
        depth_msg.is_bigendian = 0
        depth_msg.step = depth.shape[1] * 4
        depth_msg.data = depth.tobytes()
        self.pub_depth.publish(depth_msg)

        # Camera info
        K = data["intrinsics"].flatten()
        ci = CameraInfo()
        ci.header = depth_msg.header
        ci.height = depth.shape[0]
        ci.width = depth.shape[1]
        ci.k = K.tolist()
        self.pub_cam_info.publish(ci)

        # EE state
        ee_pos = data["ee_pos"]
        ee_quat = data["ee_quat"]  # wxyz
        ee_msg = PoseStamped()
        ee_msg.header.stamp = now
        ee_msg.header.frame_id = "crane_base"
        ee_msg.pose.position.x = float(ee_pos[0])
        ee_msg.pose.position.y = float(ee_pos[1])
        ee_msg.pose.position.z = float(ee_pos[2])
        ee_msg.pose.orientation.w = float(ee_quat[0])
        ee_msg.pose.orientation.x = float(ee_quat[1])
        ee_msg.pose.orientation.y = float(ee_quat[2])
        ee_msg.pose.orientation.z = float(ee_quat[3])
        self.pub_ee_state.publish(ee_msg)

        # Gripper
        grip_msg = Float64()
        grip_msg.data = data["gripper"]
        self.pub_gripper.publish(grip_msg)

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = SimBridgeReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
