#!/usr/bin/env python3
"""Joint Velocity Controller for crane — ROS2 node with SetTarget/CheckTargetReached services.

Receives EE targets via SetTarget service, runs IK + PD velocity control, publishes joint commands.

Services:
    /crane/set_target          (SetTarget)           — set EE target xyz + yaw + gripper
    /crane/is_target_reached   (CheckTargetReached)  — check if arm is at target

Subscribes:
    /crane/joint_states        (JointState)          — joint feedback from sim/PLC

Publishes:
    /crane/joint_command       (JointState)          — velocity commands to sim/PLC
    /crane/ee_state            (PoseStamped)         — current EE position (for policy node)
    /crane/gripper_state       (Float64)             — current gripper opening

Usage:
    source /workspace/ros2_ws/install/setup.bash
    python3 jv_controller_node.py
"""

import math
import numpy as np
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from sim_interface.srv import SetTarget, CheckTargetReached

# ── IKPy setup ────────────────────────────────────────────────────────
# Patch deprecated NumPy aliases (needed by ikpy)
for _alias, _real in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

from ikpy.chain import Chain

URDF_PATH = "/workspace/isaaclab/scripts/crane_controller/urdf/fpiforwarder-upperpassive.urdf"

ACTIVE_MASK = [False, True, True, True, True, True]  # 6 links: base(fixed) + 5 active

ARM_CHAIN = Chain.from_urdf_file(
    URDF_PATH,
    active_links_mask=ACTIVE_MASK,
    base_elements=["basemast"]
)

# Joint names for our crane
CONTROLLED_ARM_JOINTS = ["basemast_to_mast", "mast_to_mainboom", "mainboom_to_stick", "stick_to_telescope"]
YAW_JOINT = "lowerpassive_to_basegrapple"
GRIPPER_JOINTS = ["basegrapple_to_gripperleft", "basegrapple_to_gripperright"]
ALL_JOINTS = CONTROLLED_ARM_JOINTS + [YAW_JOINT] + GRIPPER_JOINTS
N_ARM = len(CONTROLLED_ARM_JOINTS)

# Upperpassive to basegrapple offset
EE_OFFSET_Z = -0.415

# PD gains.
# Arm/yaw values match original ROS1 controller (real-crane values need tuning).
KP = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 8.0, 8.0])
KD = np.array([1.2, 1.2, 1.2, 1.2, 0.3, 1.0, 1.0])
MAX_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
LOOP_HZ = 10.0  # matches original ROS1 controller
DEFAULT_TOLERANCE = 0.05  # radians


def ik_xyz(xyz, seed):
    """IKPy helper: basemast frame to joint angles.
    """
    UPPERPASSIVE_TO_BASEGRAPPLE_OFFSET = np.array([0.0, 0.0, -0.415])
    target_basegrapple = xyz
    target_upperpassive = target_basegrapple - UPPERPASSIVE_TO_BASEGRAPPLE_OFFSET

    T = np.eye(4)
    T[:3, 3] = [target_upperpassive[0], target_upperpassive[1], target_upperpassive[2]]  # no swap

    # Create full initial position array for all 6 links (hardcoded size)
    full_initial = np.zeros(6)
    active_idx = 0
    for i, is_active in enumerate(ACTIVE_MASK):
        if is_active and i > 0:
            if active_idx < len(seed):
                full_initial[i] = seed[active_idx]
                active_idx += 1

    best_solution = None
    best_error = float('inf')

    for iteration in range(3):
        if iteration == 0:
            initial_pos = full_initial.copy()
        else:
            initial_pos = full_initial + np.random.normal(0, 0.05, 6)

        # Clamp to joint bounds to prevent scipy error
        for i, link in enumerate(ARM_CHAIN.links):
            if i < len(initial_pos) and link.bounds is not None:
                lo, hi = link.bounds
                if np.isfinite(lo) and np.isfinite(hi):
                    initial_pos[i] = np.clip(initial_pos[i], lo + 0.001, hi - 0.001)

        try:
            full = ARM_CHAIN.inverse_kinematics_frame(T, initial_position=initial_pos)
        except (ValueError, Exception):
            continue

        # Extract first 4 controlled arm joints
        q = []
        active_count = 0
        for i, is_active in enumerate(ACTIVE_MASK):
            if is_active and i > 0:
                if active_count < 4:
                    q.append(full[i])
                active_count += 1
        q = np.array(q)

        if np.any(np.isnan(q)):
            continue

        # FK check
        full_joints_test = np.zeros(6)
        controlled_idx = 0
        active_idx = 0
        for i, is_active in enumerate(ACTIVE_MASK):
            if is_active and i > 0:
                if active_idx < 4:
                    full_joints_test[i] = q[controlled_idx]
                    controlled_idx += 1
                else:
                    full_joints_test[i] = full[i]
                active_idx += 1

        fk_test = ARM_CHAIN.forward_kinematics(full_joints_test)[:3, 3]
        error = np.linalg.norm(fk_test - [target_upperpassive[0], target_upperpassive[1], target_upperpassive[2]])

        if error < best_error:
            best_error = error
            best_solution = q.copy()
        if error < 0.001:
            break

    if best_solution is None:
        return None

    # Joint limits
    joint_limits = [
        (-1.74533, 1.74533),
        (-0.383972, 1.309),
        (-3.08574, 0.035),
        (0.13, 1.8),
    ]
    for i, (q_val, (q_min, q_max)) in enumerate(zip(best_solution, joint_limits)):
        if q_val < q_min or q_val > q_max:
            return None

    return best_solution


class JVControllerNode(Node):
    def __init__(self):
        super().__init__("jv_controller")

        self.q = np.zeros(len(ALL_JOINTS))
        self.qdot = np.zeros_like(self.q)
        self.q_des = self.q.copy()
        self.prev_error = np.zeros_like(self.q)
        self.dt = 1.0 / LOOP_HZ
        self.has_target = False

        # Joint indices
        idx = {n: i for i, n in enumerate(ALL_JOINTS)}
        self.i_yaw = idx[YAW_JOINT]
        self.i_L = idx[GRIPPER_JOINTS[0]]
        self.i_R = idx[GRIPPER_JOINTS[1]]

        # Subscribers
        self.create_subscription(JointState, "/crane/joint_states", self._joint_state_cb, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(JointState, "/crane/joint_command", 10)
        self.ee_pub = self.create_publisher(PoseStamped, "/crane/ee_state", 10)
        self.gripper_pub = self.create_publisher(Float64, "/crane/gripper_state", 10)

        # Services (same interface as FPI's sim_interface)
        self.create_service(SetTarget, "/crane/set_target", self._set_target_cb)
        self.create_service(CheckTargetReached, "/crane/is_target_reached", self._check_target_cb)

        # Control loop
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(f"JV Controller started ({LOOP_HZ}Hz, {len(ALL_JOINTS)} joints)")

    def _joint_state_cb(self, msg):
        m = {n: i for i, n in enumerate(msg.name)}
        try:
            self.q = np.array([msg.position[m[n]] for n in ALL_JOINTS])
            self.qdot = np.array([msg.velocity[m[n]] for n in ALL_JOINTS])
        except KeyError:
            pass

        # Publish EE state (basegrapple position from FK)
        # Simplified: use first 4 joints for FK, add EE offset
        ee_msg = PoseStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = "crane_base"
        # FK to get approximate EE position
        full_joints = np.zeros(len(ARM_CHAIN.links))
        for j, (i, a) in enumerate([(i, a) for i, a in enumerate(ACTIVE_MASK) if a and i > 0]):
            if j < N_ARM:
                full_joints[i] = self.q[j]
        fk = ARM_CHAIN.forward_kinematics(full_joints)
        # ikpy returns [y, x, z] for our crane, convert back
        ee_pos = np.array([fk[1, 3], fk[0, 3], fk[2, 3] + EE_OFFSET_Z])
        ee_msg.pose.position.x = float(ee_pos[0])
        ee_msg.pose.position.y = float(ee_pos[1])
        ee_msg.pose.position.z = float(ee_pos[2])
        yaw = float(self.q[self.i_yaw])
        ee_msg.pose.orientation.w = math.cos(yaw / 2)
        ee_msg.pose.orientation.z = math.sin(yaw / 2)
        self.ee_pub.publish(ee_msg)

        # Publish gripper state
        grip_msg = Float64()
        grip_msg.data = float(self.q[self.i_L])
        self.gripper_pub.publish(grip_msg)

    def _set_target_cb(self, request, response):
        """SetTarget service: EE xyz + yaw + gripper → run IK → set q_des."""
        xyz = np.array(request.target_xyz)
        yaw = request.target_yaw
        gripper = request.grapple_opening

        q_arm = ik_xyz(xyz, self.q[:N_ARM])
        if q_arm is None:
            self.get_logger().warn(f"IK failed for target {xyz}")
            response.target_set = False
            return response

        self.q_des[:N_ARM] = q_arm
        self.q_des[self.i_yaw] = yaw
        self.q_des[self.i_L] = gripper
        self.q_des[self.i_R] = gripper
        self.has_target = True

        self.get_logger().info(
            f"Target set: xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}), "
            f"yaw={math.degrees(yaw):.1f}deg, grip={gripper:.2f}")
        self.get_logger().info(
            f"  IK q_des: arm={q_arm.round(3)}, current q={self.q[:N_ARM].round(3)}, "
            f"error={np.abs(q_arm - self.q[:N_ARM]).round(3)}")

        # Debug: FK of current joints vs FK of IK solution
        full_cur = np.zeros(6)
        full_ik = np.zeros(6)
        for i, (idx, active) in enumerate([(i, a) for i, a in enumerate(ACTIVE_MASK) if a and i > 0]):
            if i < N_ARM:
                full_cur[idx] = self.q[i]
                full_ik[idx] = q_arm[i]
        fk_cur = ARM_CHAIN.forward_kinematics(full_cur)[:3, 3]
        fk_ik = ARM_CHAIN.forward_kinematics(full_ik)[:3, 3]
        self.get_logger().info(
            f"  FK current (ikpy coords): {fk_cur.round(3)}")
        self.get_logger().info(
            f"  FK IK solution (ikpy coords): {fk_ik.round(3)}")
        self.get_logger().info(
            f"  IK target (ikpy coords): [{xyz[1]:.3f}, {xyz[0]:.3f}, {xyz[2]+0.415:.3f}]")
        self.get_logger().info(
            f"  IK target (no swap): [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]+0.415:.3f}]")
        response.target_set = True
        return response

    def _check_target_cb(self, request, response):
        """CheckTargetReached service: check if joints are within tolerance."""
        if not self.has_target:
            response.target_reached = False
            return response

        tolerance = request.tolerance if request.tolerance > 0.0 else DEFAULT_TOLERANCE
        error = np.abs(self.q_des - self.q)

        arm_ok = bool(np.all(error[:N_ARM] < tolerance))
        yaw_ok = bool(error[self.i_yaw] < tolerance)
        grip_ok = bool(np.all(error[self.i_L:] < tolerance * 2))  # looser for grippers

        response.target_reached = arm_ok and yaw_ok and grip_ok
        return response

    def _control_loop(self):
        """PD velocity control at LOOP_HZ."""
        if not self.has_target:
            return

        self._loop_count = getattr(self, '_loop_count', 0) + 1

        err = self.q_des - self.q
        err_dot = (err - self.prev_error) * LOOP_HZ
        vel = KP * err + KD * err_dot
        vel = np.clip(vel, -MAX_VEL, MAX_VEL)
        self.prev_error = err.copy()

        # Log every 2 seconds
        if self._loop_count % int(LOOP_HZ * 2) == 0:
            arm_err = np.abs(err[:N_ARM])
            self.get_logger().info(
                f"[PD] arm_err={arm_err.round(3)}, max_vel={np.abs(vel[:N_ARM]).max():.3f}, "
                f"q={self.q[:N_ARM].round(3)}, q_des={self.q_des[:N_ARM].round(3)}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ALL_JOINTS)
        msg.position = self.q_des.tolist()
        msg.velocity = vel.tolist()
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = JVControllerNode()
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
