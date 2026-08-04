"""Lightweight sender: publishes sim data over TCP socket.

Called from within Isaac Lab's play.py (Python 3.11, no rclpy).
Sends depth image, camera info, EE state, and gripper state as
msgpack-encoded dicts over a TCP socket.
"""

import socket
import struct
import numpy as np
import pickle


class SimBridgeSender:
    """Sends sim data over TCP to the ROS2 bridge receiver."""

    def __init__(self, host="127.0.0.1", port=9876):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        self.sock.settimeout(0.1)  # non-blocking accept
        self.conn = None
        self.host = host
        self.port = port
        print(f"[SimBridge] Listening on {host}:{port}")

    def accept(self):
        """Try to accept a connection (non-blocking)."""
        if self.conn is None:
            try:
                self.conn, addr = self.sock.accept()
                self.conn.settimeout(0.5)
                print(f"[SimBridge] Client connected from {addr}")
            except socket.timeout:
                pass

    def send(self, depth, intrinsics, ee_pos, ee_quat, gripper,
             cam_pos=None, cam_quat_ros=None, base_pos=None, base_quat=None):
        """Send one frame of sim data.

        Args:
            depth: (H, W) float32 numpy array.
            intrinsics: (3, 3) float32 numpy array.
            ee_pos: (3,) float32 — EE position in base frame.
            ee_quat: (4,) float32 — EE orientation (wxyz).
            gripper: float — gripper opening.
            cam_pos: (3,) float32 — camera position in world frame.
            cam_quat_ros: (4,) float32 — camera orientation (wxyz, ROS convention).
            base_pos: (3,) float32 — crane base position in world frame.
            base_quat: (4,) float32 — crane base orientation (wxyz).
        """
        self.accept()
        if self.conn is None:
            return  # no client connected

        data = {
            "depth": depth.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "gripper": float(gripper),
        }
        if cam_pos is not None:
            data["cam_pos"] = cam_pos.astype(np.float32)
        if cam_quat_ros is not None:
            data["cam_quat_ros"] = cam_quat_ros.astype(np.float32)
        if base_pos is not None:
            data["base_pos"] = base_pos.astype(np.float32)
        if base_quat is not None:
            data["base_quat"] = base_quat.astype(np.float32)
        payload = pickle.dumps(data)
        header = struct.pack("!I", len(payload))
        try:
            self.conn.sendall(header + payload)
        except (BrokenPipeError, ConnectionResetError):
            print("[SimBridge] Client disconnected")
            self.conn = None

    def close(self):
        if self.conn:
            self.conn.close()
        self.sock.close()
        print("[SimBridge] Closed")
