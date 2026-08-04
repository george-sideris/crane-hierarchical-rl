#!/usr/bin/env python3
"""Sequencer node — runs the 10-phase FSM, calls controller services for each phase.

Receives a grasp target from the policy node, sequences through all FSM phases
by calling SetTarget on the controller and waiting for CheckTargetReached.

Subscribes:
    /crane/grasp_target    (PoseStamped)   — from policy node
    /crane/ee_state        (PoseStamped)   — current EE pose (monitoring)

Publishes:
    /crane/fsm_state       (String)        — current phase name
    /crane/cycle_complete  (Bool)          — signals policy node to get next target

Service clients:
    /crane/set_target          (SetTarget)
    /crane/is_target_reached   (CheckTargetReached)

Usage:
    source /workspace/ros2_ws/install/setup.bash
    python3 sequencer_node.py
"""

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool
from sim_interface.srv import SetTarget, CheckTargetReached


class Phase:
    HOVER_UP = 0
    ALIGN_YAW = 1
    DESCEND = 2
    CLOSE = 3
    LIFT_HIGH = 4
    CARRY_HOME = 5
    ALIGN_HOME_YAW = 6
    LOWER_TO_DROP = 7
    OPEN = 8
    SETTLE = 9

PHASE_NAMES = {
    0: "HOVER_UP", 1: "ALIGN_YAW", 2: "DESCEND", 3: "CLOSE",
    4: "LIFT_HIGH", 5: "CARRY_HOME", 6: "ALIGN_HOME_YAW",
    7: "LOWER_TO_DROP", 8: "OPEN", 9: "SETTLE",
}


class SequencerNode(Node):
    def __init__(self):
        super().__init__("sequencer")

        # FSM config
        self.hover_clear = 0.8       # meters above target (limited by IK workspace)
        self.approach_above = 0.15   # meters above target for approach
        self.grip_open = 0.05        # rad
        self.grip_closed = 2.23      # rad
        self.drop_position = [2.1, 0.0, 2.0]  # base frame
        self.drop_yaw = 0.0
        self.check_rate = 10.0       # Hz for polling target reached
        self.safety_timeout = 60.0   # seconds max per phase
        self.settle_time = 2.0       # seconds for settle phase

        # State
        self.phase = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0
        self.cycle_count = 0
        self.busy = False

        # Subscribers
        self.create_subscription(PoseStamped, "/crane/grasp_target", self._grasp_target_cb, 10)

        # Publishers
        self.fsm_pub = self.create_publisher(String, "/crane/fsm_state", 10)
        self.complete_pub = self.create_publisher(Bool, "/crane/cycle_complete", 10)

        # Service clients
        self.set_target_client = self.create_client(SetTarget, "/crane/set_target")
        self.check_reached_client = self.create_client(CheckTargetReached, "/crane/is_target_reached")

        self.get_logger().info("Sequencer started, waiting for /crane/grasp_target")

    def _grasp_target_cb(self, msg):
        """Receive grasp target from policy node and execute full cycle."""
        if self.busy:
            self.get_logger().warn("Still executing previous cycle, ignoring new target")
            return

        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        # Extract yaw from quaternion
        q = msg.pose.orientation
        self.target_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self.get_logger().info(
            f"Cycle {self.cycle_count + 1}: target=({self.target_x:.2f}, {self.target_y:.2f}, "
            f"{self.target_z:.2f}), yaw={math.degrees(self.target_yaw):.1f}deg")

        # Execute cycle in a separate thread to not block the callback
        import threading
        threading.Thread(target=self._execute_cycle, daemon=True).start()

    def _execute_cycle(self):
        """Execute full 10-phase pick-and-place cycle."""
        self.busy = True
        hover_z = self.target_z + self.hover_clear
        approach_z = self.target_z + self.approach_above

        phases = [
            # (phase, xyz, yaw, gripper, description)
            (Phase.HOVER_UP, [self.target_x, self.target_y, hover_z],
             None, self.grip_open, "Hover above target"),
            (Phase.ALIGN_YAW, [self.target_x, self.target_y, hover_z],
             self.target_yaw, self.grip_open, "Align yaw"),
            (Phase.DESCEND, [self.target_x, self.target_y, approach_z],
             self.target_yaw, self.grip_open, "Descend to approach"),
            (Phase.CLOSE, [self.target_x, self.target_y, approach_z],
             self.target_yaw, self.grip_closed, "Close gripper"),
            (Phase.LIFT_HIGH, [self.target_x, self.target_y, hover_z],
             self.target_yaw, self.grip_closed, "Lift"),
            (Phase.CARRY_HOME, [self.drop_position[0], self.drop_position[1], hover_z],
             self.target_yaw, self.grip_closed, "Carry to drop"),
            (Phase.ALIGN_HOME_YAW, [self.drop_position[0], self.drop_position[1], hover_z],
             self.drop_yaw, self.grip_closed, "Align drop yaw"),
            (Phase.LOWER_TO_DROP, self.drop_position,
             self.drop_yaw, self.grip_closed, "Lower to drop"),
            (Phase.OPEN, self.drop_position,
             self.drop_yaw, self.grip_open, "Open gripper"),
            (Phase.SETTLE, self.drop_position,
             self.drop_yaw, self.grip_open, "Settle"),
        ]

        for phase_id, xyz, yaw, gripper, desc in phases:
            self.phase = phase_id
            self._publish_state(f"{PHASE_NAMES[phase_id]} (cycle {self.cycle_count + 1})")
            self.get_logger().info(f"  Phase: {PHASE_NAMES[phase_id]} — {desc}")

            if phase_id == Phase.SETTLE:
                time.sleep(self.settle_time)
                continue

            # Use current yaw if None (HOVER_UP keeps existing yaw)
            if yaw is None:
                yaw = 0.0

            # Call SetTarget service
            success = self._call_set_target(xyz, yaw, gripper)
            if not success:
                self.get_logger().error(f"SetTarget failed at {PHASE_NAMES[phase_id]}, aborting cycle")
                break

            # Wait for target reached
            reached = self._wait_for_target(timeout=self.safety_timeout)
            if not reached:
                self.get_logger().warn(f"Timeout at {PHASE_NAMES[phase_id]}, continuing anyway")

        self.cycle_count += 1
        self.busy = False
        self.get_logger().info(f"Cycle {self.cycle_count} complete")

        # Signal policy node
        msg = Bool()
        msg.data = True
        self.complete_pub.publish(msg)

    def _call_set_target(self, xyz, yaw, gripper):
        """Call /crane/set_target service."""
        if not self.set_target_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("set_target service not available")
            return False

        req = SetTarget.Request()
        req.target_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        req.target_yaw = float(yaw)
        req.grapple_opening = float(gripper)

        future = self.set_target_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            return future.result().target_set
        return False

    def _wait_for_target(self, timeout=60.0):
        """Poll /crane/is_target_reached until True or timeout."""
        if not self.check_reached_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("is_target_reached service not available")
            return False

        start = time.time()
        while time.time() - start < timeout:
            req = CheckTargetReached.Request()
            req.tolerance = 0.05
            future = self.check_reached_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

            if future.result() is not None and future.result().target_reached:
                return True
            time.sleep(1.0 / self.check_rate)

        return False

    def _publish_state(self, state_str):
        msg = String()
        msg.data = state_str
        self.fsm_pub.publish(msg)


def main():
    rclpy.init()
    node = SequencerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
