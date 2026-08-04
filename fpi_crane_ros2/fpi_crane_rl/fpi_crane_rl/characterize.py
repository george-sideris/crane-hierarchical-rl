#!/usr/bin/env python3
"""Run multiple targets through IK + /relative_joint_move and log the
joint-space and task-space error for each.

Useful for characterizing tracking accuracy across the workspace. Each
target is sent once, the resulting error is appended to a CSV, then the
next target fires. After all targets, prints summary statistics.

Usage:
  ros2 run fpi_crane_rl characterize \\
      [--targets "x,y,z,yaw" "x,y,z,yaw" ...] \\
      [--velocity 0.15] [--min_dur 2.0] \\
      [--repeat 1] [--out characterize.csv]

If no --targets given, uses a safe default sweep of 5 points with z in
[1.5, 2.5] and yaw spanning +-pi/2.

CSV columns:
  run_id, target_x, target_y, target_z, target_yaw,
  slew_err, boom_err, stick_err, telescope_err, yaw_err,
  actual_x, actual_y, actual_z, task_err_norm, duration_s
"""
import argparse
import csv
import os
from datetime import datetime

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

# Safe default targets: 10 points spread across the reachable workspace in
# both +x and -x halves, z in [1.5, 2.7] (clears the rack), yaw spanning the
# full +-pi range to exercise slew and grapplecarrier.
DEFAULT_TARGETS = [
    (-4.0,  2.0, 1.5,  0.00),
    (+4.0,  2.0, 2.5,  1.57),    # +pi/2
    (-3.0, -1.5, 2.0, -1.57),    # -pi/2
    (+3.0, -2.5, 2.0,  2.36),    # +3pi/4
    (-3.5,  3.0, 2.5, -2.36),    # -3pi/4
    (+4.5, -1.0, 1.5,  0.78),    # +pi/4
    (-3.5, -2.5, 2.7, -0.78),    # -pi/4
    (+3.5,  0.0, 2.0,  3.00),
    (-3.0,  3.0, 2.5, -3.00),
    ( 0.0,  4.0, 2.0,  0.00),
]


def _resolve_urdf_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("fpi_crane_rl"),
                            "urdf", "fpiforwarder-upperpassive.urdf")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "urdf", "fpiforwarder-upperpassive.urdf")


class Characterize(Node):
    """One node spans the full target queue. State machine advances after
    each target's HOLD transition, with a settle pause between."""

    STATE_WAIT_JOINTS = "wait_joints"
    STATE_WAIT_EXECUTE = "wait_execute"
    STATE_WAIT_HOLD = "wait_hold"
    STATE_SETTLE = "settle"
    STATE_DONE = "done"

    def __init__(self, targets, velocity, min_dur, settle):
        super().__init__("characterize")
        self.ik = ik.CraneIK(_resolve_urdf_path())
        self.targets = list(targets)
        self.velocity = float(velocity)
        self.min_dur = float(min_dur)
        self.settle = float(settle)

        self.idx = 0
        self.rows = []
        self.state = self.STATE_WAIT_JOINTS
        self.target_arm = None
        self.duration_s = 0.0
        self.deadline_ns = None
        self.settle_until_ns = None
        self.latest_joints_msg = None
        self.seen_execute = False

        self.client = self.create_client(RelativeJointMove, "relative_joint_move")
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(PlcStatus, "/plc_status", self._on_plc, 10)
        self.create_timer(0.2, self._tick)

        self.get_logger().info(
            f"Characterize: {len(self.targets)} target(s) at "
            f"velocity={self.velocity}, min_dur={self.min_dur}")
        for i, t in enumerate(self.targets, 1):
            self.get_logger().info(
                f"  target {i}: xyz=({t[0]:+.2f}, {t[1]:+.2f}, "
                f"{t[2]:+.2f}) yaw={t[3]:+.2f}")

    @property
    def current_target(self):
        return self.targets[self.idx]

    def _on_joints(self, msg):
        self.latest_joints_msg = msg
        if self.state == self.STATE_WAIT_JOINTS:
            self._send_current(msg)

    def _send_current(self, msg):
        try:
            arm = [msg.position[msg.name.index(n)] for n in ARM_JOINTS]
            yaw_cur = float(msg.position[msg.name.index(YAW_JOINT)])
        except ValueError:
            return

        x, y, z, target_yaw = self.current_target
        target_xyz = np.array([x, y, z], dtype=float)

        q = self.ik.solve(target_xyz.tolist(), arm)
        if q is None:
            self.get_logger().error(
                f"target {self.idx + 1}/{len(self.targets)}: IK failed")
            self._record_failure("IK")
            self._advance()
            return

        self.target_arm = np.array(q)
        deltas = [float(q[i] - arm[i]) for i in range(4)]
        deltas.append(float(target_yaw - yaw_cur))

        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/relative_joint_move service not available")
            self._record_failure("no_service")
            self._advance()
            return

        req = RelativeJointMove.Request()
        req.sequence_id = 1000 + self.idx + 1
        req.joint_names = list(ARM_JOINTS) + [YAW_JOINT]
        req.delta_positions = deltas
        req.velocity = self.velocity
        req.min_duration = self.min_dur

        self.seen_execute = False
        self.state = self.STATE_WAIT_EXECUTE
        self.get_logger().info(
            f"--- run {self.idx + 1}/{len(self.targets)}: target=({x:+.2f}, "
            f"{y:+.2f}, {z:+.2f}) yaw={target_yaw:+.2f} ---")
        self.get_logger().info(
            f"  deltas={[f'{d:+.3f}' for d in deltas]}")
        f = self.client.call_async(req)
        f.add_done_callback(self._on_send)

    def _on_send(self, future):
        r = future.result()
        if not r.success:
            self.get_logger().error(f"service rejected: {r.message}")
            self._record_failure(r.message)
            self._advance()
            return
        self.duration_s = float(r.duration)
        timeout = max(self.duration_s * 3.0, 30.0)
        self.deadline_ns = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        self.get_logger().info(
            f"  service accepted: duration={self.duration_s:.2f}s")

    def _on_plc(self, msg):
        if self.state not in (self.STATE_WAIT_EXECUTE, self.STATE_WAIT_HOLD):
            return
        s = int(msg.plc_crane_state)
        if s == PLC_EXECUTE:
            self.seen_execute = True
            self.state = self.STATE_WAIT_HOLD
        elif s in (PLC_HOLD, PLC_END) and self.seen_execute:
            self._finalize()

    def _tick(self):
        now_ns = self.get_clock().now().nanoseconds

        # Timeout in wait_execute / wait_hold
        if self.state in (self.STATE_WAIT_EXECUTE, self.STATE_WAIT_HOLD) and \
                self.deadline_ns is not None and now_ns > self.deadline_ns:
            self.get_logger().warn(
                "timeout reached; finalizing with current joint state")
            self._finalize()
            return

        # End of settle pause
        if self.state == self.STATE_SETTLE and \
                self.settle_until_ns is not None and now_ns > self.settle_until_ns:
            self.state = self.STATE_WAIT_JOINTS
            if self.latest_joints_msg is not None:
                self._send_current(self.latest_joints_msg)

    def _finalize(self):
        if self.latest_joints_msg is None:
            self._record_failure("no_joints")
            self._advance()
            return
        msg = self.latest_joints_msg
        try:
            final_arm = [float(msg.position[msg.name.index(n)]) for n in ARM_JOINTS]
            final_yaw = float(msg.position[msg.name.index(YAW_JOINT)])
        except ValueError:
            self._record_failure("missing_final_joints")
            self._advance()
            return
        try:
            hanger = float(msg.position[msg.name.index(HANGER_JOINT)])
        except ValueError:
            hanger = 0.0

        x, y, z, target_yaw = self.current_target
        joint_err = [final_arm[i] - float(self.target_arm[i]) for i in range(4)]
        yaw_err = final_yaw - target_yaw

        full = np.zeros(6)
        full[1] = final_arm[0]; full[2] = final_arm[1]
        full[3] = final_arm[2]; full[4] = final_arm[3]
        full[5] = hanger
        upperpassive = self.ik.chain.forward_kinematics(
            [float(v) for v in full])[:3, 3]
        actual_xyz = upperpassive + np.array([0.0, 0.0, ik.EE_OFFSET_Z])
        task_err = actual_xyz - np.array([x, y, z])
        task_err_norm = float(np.linalg.norm(task_err))

        self.rows.append({
            "run_id": self.idx + 1,
            "target_x": x, "target_y": y, "target_z": z, "target_yaw": target_yaw,
            "slew_err": joint_err[0],
            "boom_err": joint_err[1],
            "stick_err": joint_err[2],
            "telescope_err": joint_err[3],
            "yaw_err": yaw_err,
            "actual_x": float(actual_xyz[0]),
            "actual_y": float(actual_xyz[1]),
            "actual_z": float(actual_xyz[2]),
            "task_err_norm": task_err_norm,
            "duration_s": self.duration_s,
        })
        self.get_logger().info(
            f"  joint err: {[f'{e:+.4f}' for e in joint_err]}, "
            f"yaw {yaw_err:+.4f}")
        self.get_logger().info(
            f"  task err norm: {task_err_norm:.4f} m  "
            f"actual=({actual_xyz[0]:+.3f}, {actual_xyz[1]:+.3f}, "
            f"{actual_xyz[2]:+.3f})")
        self._advance()

    def _record_failure(self, reason):
        x, y, z, target_yaw = self.current_target
        self.rows.append({
            "run_id": self.idx + 1,
            "target_x": x, "target_y": y, "target_z": z, "target_yaw": target_yaw,
            "slew_err": float("nan"), "boom_err": float("nan"),
            "stick_err": float("nan"), "telescope_err": float("nan"),
            "yaw_err": float("nan"),
            "actual_x": float("nan"), "actual_y": float("nan"),
            "actual_z": float("nan"),
            "task_err_norm": float("nan"),
            "duration_s": float("nan"),
        })
        self.get_logger().error(f"  recorded failure: {reason}")

    def _advance(self):
        self.idx += 1
        self.target_arm = None
        self.duration_s = 0.0
        self.deadline_ns = None
        if self.idx >= len(self.targets):
            self.state = self.STATE_DONE
            self.get_logger().info("All targets complete.")
            return
        self.state = self.STATE_SETTLE
        self.settle_until_ns = (
            self.get_clock().now().nanoseconds + int(self.settle * 1e9))


def _parse_target(s):
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"target '{s}' must be 'x,y,z,yaw' (got {len(parts)} values)")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", nargs="+", type=_parse_target,
                    default=DEFAULT_TARGETS,
                    help='Targets as "x,y,z,yaw" strings. Default: 5-point sweep.')
    ap.add_argument("--velocity", type=float, default=0.15,
                    help="Peak joint velocity rad/s (default 0.15)")
    ap.add_argument("--min_dur", type=float, default=2.0,
                    help="Minimum duration seconds (default 2.0)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="How many times to send each target (default 1)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV file (default characterize_<timestamp>.csv)")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="Seconds to wait between runs (default 2.0)")
    args = ap.parse_args()

    if args.out is None:
        args.out = f"characterize_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    targets = list(args.targets) * args.repeat
    rclpy.init()
    node = Characterize(targets, args.velocity, args.min_dur, args.settle)
    try:
        while rclpy.ok() and node.state != node.STATE_DONE:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    rows = node.rows
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if not rows:
        print("\nNo runs recorded.")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    arr = lambda k: np.array([r[k] for r in rows if not np.isnan(r[k])])
    print(f"\n=== summary over {len(rows)} run(s) ===")
    norm = arr("task_err_norm")
    if len(norm):
        print(f"  task_err_norm:  mean={norm.mean():.4f}  std={norm.std():.4f}  "
              f"max={norm.max():.4f} m")
    for j in ("slew_err", "boom_err", "stick_err", "telescope_err", "yaw_err"):
        a = arr(j)
        if len(a):
            print(f"  {j:14s}: mean={a.mean():+.4f}  std={a.std():.4f}  "
                  f"abs_max={np.abs(a).max():.4f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
