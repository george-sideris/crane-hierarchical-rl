#!/usr/bin/env python3
"""Send the crane to the gaze pose via /relative_joint_move.

Reads the current /joint_states, computes per-joint deltas from the current
pose to the gaze target, and calls /relative_joint_move so the upstream
planner generates a smooth trajectory through the PLC. Safe to run on
hardware: the PLC sees deltas, which is the wire convention the firmware
expects.

The gaze target is the crane configuration from the 17:39 rack-calibration
bag. Only joints that the trajectory pipeline actually drives are commanded:
slew, boom, stick, telescope, grapplecarrier. The grippers and passive
joints (hanger, bearingfork) are not included.

Usage:  python3 send_gaze_pose.py [--velocity 0.15] [--min_dur 12] [--seq 2]
"""
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from fpi_crane_msgs.srv import RelativeJointMove


JOINT_NAMES = ['slew_joint', 'boom_joint', 'stick_joint', 'telescope_joint',
               'grapplecarrier_joint']
# Absolute joint targets at the gaze pose, same order as JOINT_NAMES.
GAZE = {
    'slew_joint':           0.9913,
    'boom_joint':           0.6667,
    'stick_joint':         -0.7592,
    'telescope_joint':      0.2795,
    'grapplecarrier_joint': 3.3318,
}


class GazeSender(Node):
    def __init__(self, velocity, min_dur, seq):
        super().__init__('send_gaze_pose')
        self.velocity = float(velocity)
        self.min_dur = float(min_dur)
        self.seq = int(seq)
        self.done = False
        self.client = self.create_client(RelativeJointMove, 'relative_joint_move')
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.get_logger().info(
            f"Gaze sender ready: velocity={self.velocity} min_dur={self.min_dur}")

    def on_joints(self, msg):
        if self.done:
            return
        try:
            cur = {n: float(msg.position[msg.name.index(n)]) for n in JOINT_NAMES}
        except ValueError:
            return  # joint state not yet complete

        deltas = [GAZE[n] - cur[n] for n in JOINT_NAMES]
        self.get_logger().info(
            f"Current: {[f'{cur[n]:+.3f}' for n in JOINT_NAMES]}")
        self.get_logger().info(
            f"Target:  {[f'{GAZE[n]:+.3f}' for n in JOINT_NAMES]}")
        self.get_logger().info(
            f"Deltas:  {[f'{d:+.3f}' for d in deltas]}")

        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/relative_joint_move service not available")
            rclpy.shutdown()
            return

        req = RelativeJointMove.Request()
        req.sequence_id = self.seq
        req.joint_names = list(JOINT_NAMES)
        req.delta_positions = [float(d) for d in deltas]
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
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--velocity', type=float, default=0.15,
                    help='Peak joint velocity in rad/s (default 0.15)')
    ap.add_argument('--min_dur', type=float, default=12.0,
                    help='Minimum trajectory duration in seconds (default 12)')
    ap.add_argument('--seq', type=int, default=2,
                    help='Trajectory sequence id (default 2)')
    args = ap.parse_args()

    rclpy.init()
    node = GazeSender(args.velocity, args.min_dur, args.seq)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
