#!/usr/bin/env python3
"""Send a task-space target via IK + /relative_joint_move and report errors.

Usage:
    ros2 run fpi_crane_rl test_target x y z yaw [velocity=0.15] [min_dur=2.0]

xyz is the end-effector target in base_link frame (meters).
yaw is the grapplecarrier joint target in base frame (radians).
velocity is peak joint speed (rad/s).
min_dur is the floor on segment duration (seconds).

After the move completes, prints joint-space error per joint and the
task-space Cartesian error computed by running FK on the final joints.
"""
import os
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from fpi_crane_msgs.msg import PlcStatus
from fpi_crane_msgs.srv import RelativeJointMove
from fpi_crane_rl import ik


ARM_JOINTS = ("slew_joint", "boom_joint", "stick_joint", "telescope_joint")
YAW_JOINT = "grapplecarrier_joint"
HANGER_JOINT = "hanger_joint"

PLC_HOLD = 1
PLC_EXECUTE = 2
PLC_END = 3

STATE_WAITING_FOR_JOINTS = "waiting_for_joints"
STATE_WAITING_FOR_EXECUTE = "waiting_for_execute"
STATE_WAITING_FOR_HOLD = "waiting_for_hold"
STATE_DONE = "done"


def _resolve_urdf_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("fpi_crane_rl"),
                            "urdf", "fpiforwarder-upperpassive.urdf")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "urdf", "fpiforwarder-upperpassive.urdf")


class OneShot(Node):
    def __init__(self, x, y, z, yaw, velocity, min_dur):
        super().__init__("test_target")
        self.ik = ik.CraneIK(_resolve_urdf_path())
        self.target_xyz = np.array([x, y, z], dtype=float)
        self.target_yaw = float(yaw)
        self.velocity = float(velocity)
        self.min_dur = float(min_dur)

        self.state = STATE_WAITING_FOR_JOINTS
        self.initial_arm = None
        self.initial_yaw = None
        self.target_arm = None
        self.duration_s = 0.0
        self.deadline_ns = None
        self.latest_joints_msg = None

        self.client = self.create_client(RelativeJointMove, "relative_joint_move")
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(PlcStatus, "/plc_status", self.on_plc_status, 10)

        self.get_logger().info(
            f"Target xyz={self.target_xyz.tolist()}, yaw={self.target_yaw:.3f}, "
            f"velocity={self.velocity}, min_dur={self.min_dur}")

    def on_joints(self, msg):
        self.latest_joints_msg = msg
        if self.state == STATE_WAITING_FOR_JOINTS:
            self._send_command(msg)

    def _send_command(self, msg):
        try:
            arm = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]
            yaw_cur = msg.position[msg.name.index(YAW_JOINT)]
        except ValueError:
            return

        q = self.ik.solve(self.target_xyz.tolist(), arm)
        if q is None:
            self.get_logger().error(f"IK failed for {self.target_xyz.tolist()}")
            rclpy.shutdown()
            return

        self.initial_arm = np.array(arm)
        self.initial_yaw = float(yaw_cur)
        self.target_arm = np.array(q)

        deltas = [float(q[i] - arm[i]) for i in range(4)]
        deltas.append(float(self.target_yaw - yaw_cur))

        self.get_logger().info(
            f"IK solution q={[f'{v:.3f}' for v in q]}, "
            f"deltas={[f'{d:+.3f}' for d in deltas]}")

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

        self.state = STATE_WAITING_FOR_EXECUTE
        future = self.client.call_async(req)
        future.add_done_callback(self.on_send_response)

    def on_send_response(self, future):
        r = future.result()
        if not r.success:
            self.get_logger().error(f"Service rejected: {r.message}")
            rclpy.shutdown()
            return
        self.duration_s = float(r.duration)
        timeout_s = max(self.duration_s * 3.0, 30.0)
        self.deadline_ns = self.get_clock().now().nanoseconds + int(timeout_s * 1e9)
        self.get_logger().info(
            f"Service accepted: duration={self.duration_s:.2f}s, "
            f"timeout={timeout_s:.1f}s")

    def on_plc_status(self, msg):
        if self.state == STATE_DONE:
            return

        if self.deadline_ns is not None and \
                self.get_clock().now().nanoseconds > self.deadline_ns:
            self.get_logger().warn(
                "Timeout waiting for trajectory; computing errors anyway")
            self._compute_and_exit()
            return

        state = int(msg.plc_crane_state)
        if self.state == STATE_WAITING_FOR_EXECUTE and state == PLC_EXECUTE:
            self.state = STATE_WAITING_FOR_HOLD
            self.get_logger().info("Trajectory executing...")
        elif self.state == STATE_WAITING_FOR_HOLD and state in (PLC_HOLD, PLC_END):
            self.get_logger().info("Trajectory complete; computing errors")
            self._compute_and_exit()

    def _compute_and_exit(self):
        if self.state == STATE_DONE or self.latest_joints_msg is None:
            return
        self.state = STATE_DONE

        msg = self.latest_joints_msg
        try:
            final_arm = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]
            final_yaw = msg.position[msg.name.index(YAW_JOINT)]
        except ValueError:
            self.get_logger().error("Joint names missing from final /joint_states")
            rclpy.shutdown()
            return

        try:
            hanger = msg.position[msg.name.index(HANGER_JOINT)]
        except ValueError:
            hanger = 0.0

        # FK from final joints. Chain links: [base, slew, boom, stick,
        # telescope, hanger]. Result is upperpassive frame; subtract
        # EE_OFFSET_Z to get the grapple base.
        full_joints = np.zeros(6)
        full_joints[1] = final_arm[0]
        full_joints[2] = final_arm[1]
        full_joints[3] = final_arm[2]
        full_joints[4] = final_arm[3]
        full_joints[5] = float(hanger)
        upperpassive_xyz = self.ik.chain.forward_kinematics(full_joints)[:3, 3]
        actual_xyz = upperpassive_xyz + np.array([0.0, 0.0, ik.EE_OFFSET_Z])
        task_error = actual_xyz - self.target_xyz
        task_error_norm = float(np.linalg.norm(task_error))

        print("")
        print("Move complete.")
        print(f"  Duration commanded: {self.duration_s:.2f} s")
        print("")
        print("  Joint errors (final - target):")
        units = {"telescope_joint": "m"}
        for i, n in enumerate(ARM_JOINTS):
            err = final_arm[i] - float(self.target_arm[i])
            u = units.get(n, "rad")
            print(f"    {n:22s} {err:+.4f} {u}")
        yaw_err = final_yaw - self.target_yaw
        print(f"    {YAW_JOINT:22s} {yaw_err:+.4f} rad")
        print("")
        print("  Task-space error:")
        print(f"    target xyz: ({self.target_xyz[0]:+.3f}, "
              f"{self.target_xyz[1]:+.3f}, {self.target_xyz[2]:+.3f}) m")
        print(f"    actual xyz: ({actual_xyz[0]:+.3f}, "
              f"{actual_xyz[1]:+.3f}, {actual_xyz[2]:+.3f}) m")
        print(f"    delta:      ({task_error[0]:+.3f}, "
              f"{task_error[1]:+.3f}, {task_error[2]:+.3f}) m")
        print(f"    norm:        {task_error_norm:.4f} m")
        print("")

        rclpy.shutdown()


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    x, y, z, yaw = map(float, sys.argv[1:5])
    velocity = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    min_dur = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0

    rclpy.init()
    node = OneShot(x, y, z, yaw, velocity, min_dur)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
