#!/usr/bin/env python3
"""Joint-velocity controller for the crane.

Subscribes to joint states, runs IK on EE targets supplied via the
SetTarget service, and publishes joint position + velocity commands.

Subscriptions:
    /joint_states_sim        (sensor_msgs/JointState)

Publications:
    /joint_command           (sensor_msgs/JointState)
    /crane/ee_state          (geometry_msgs/PoseStamped)
    /crane/gripper_state     (std_msgs/Float64)

Services:
    /crane/set_target        (sim_interface/SetTarget)
    /crane/is_target_reached (sim_interface/CheckTargetReached)

Run:
    python3 jv_controller_node.py --ros-args -p use_sim_time:=true
"""

import math
import os
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from sim_interface.srv import SetTarget, CheckTargetReached

# IKPy needs deprecated NumPy aliases.
for _alias, _real in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)
from ikpy.chain import Chain

URDF_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "assets", "urdf", "fpiforwarder-upperpassive.urdf",
)
ACTIVE_MASK = [False, True, True, True, True, True]
ARM_CHAIN = Chain.from_urdf_file(
    URDF_PATH,
    active_links_mask=ACTIVE_MASK,
    base_elements=["basemast"],
)

# Joint names
CONTROLLED_ARM_JOINTS = [
    "slew_joint",
    "boom_joint",
    "stick_joint",
    "telescope_joint",
]
YAW_JOINT = "grapplecarrier_joint"
GRIPPER_JOINTS = ["grappletong1_joint", "grappletong2_joint"]
ALL_JOINTS = CONTROLLED_ARM_JOINTS + [YAW_JOINT] + GRIPPER_JOINTS
N_ARM = len(CONTROLLED_ARM_JOINTS)

EE_OFFSET_Z = -0.415

KP = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 8.0, 8.0])
KD = np.array([1.2, 1.2, 1.2, 1.2, 0.3, 1.0, 1.0])
MAX_VEL = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 10.0])
LOOP_HZ = 10.0
DEFAULT_TOLERANCE = 0.05


def ik_xyz(xyz, seed):
    target_basegrapple = xyz
    target_upperpassive = target_basegrapple - np.array([0.0, 0.0, EE_OFFSET_Z])

    T = np.eye(4)
    T[:3, 3] = target_upperpassive

    full_initial = np.zeros(6)
    active_idx = 0
    for i, is_active in enumerate(ACTIVE_MASK):
        if is_active and i > 0:
            if active_idx < len(seed):
                full_initial[i] = seed[active_idx]
                active_idx += 1

    best_solution = None
    best_error = float("inf")
    for iteration in range(3):
        initial_pos = (full_initial.copy() if iteration == 0
                       else full_initial + np.random.normal(0, 0.05, 6))
        for i, link in enumerate(ARM_CHAIN.links):
            if i < len(initial_pos) and link.bounds is not None:
                lo, hi = link.bounds
                if np.isfinite(lo) and np.isfinite(hi):
                    initial_pos[i] = np.clip(initial_pos[i], lo + 0.001, hi - 0.001)
        try:
            full = ARM_CHAIN.inverse_kinematics_frame(T, initial_position=initial_pos)
        except Exception:
            continue
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
        error = np.linalg.norm(fk_test - target_upperpassive)
        if error < best_error:
            best_error = error
            best_solution = q.copy()
        if error < 0.001:
            break
    if best_solution is None:
        return None
    joint_limits = [
        (-1.74533, 1.74533),
        (-0.383972, 1.309),
        (-3.08574, 0.035),
        (0.13, 1.8),
    ]
    for q_val, (q_min, q_max) in zip(best_solution, joint_limits):
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

        idx = {n: i for i, n in enumerate(ALL_JOINTS)}
        self.i_yaw = idx[YAW_JOINT]
        self.i_L = idx[GRIPPER_JOINTS[0]]
        self.i_R = idx[GRIPPER_JOINTS[1]]

        self.create_subscription(JointState, "/joint_states_sim",
                                 self._joint_state_cb, 10)
        self.cmd_pub = self.create_publisher(JointState, "/joint_command", 10)
        self.ee_pub = self.create_publisher(PoseStamped, "/crane/ee_state", 10)
        self.gripper_pub = self.create_publisher(Float64, "/crane/gripper_state", 10)

        self.create_service(SetTarget, "/crane/set_target", self._set_target_cb)
        self.create_service(CheckTargetReached, "/crane/is_target_reached",
                            self._check_target_cb)

        self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f"JV Controller started ({LOOP_HZ}Hz, {len(ALL_JOINTS)} joints)")

    def _joint_state_cb(self, msg):
        m = {n: i for i, n in enumerate(msg.name)}
        try:
            self.q = np.array([msg.position[m[n]] for n in ALL_JOINTS])
            self.qdot = np.array([msg.velocity[m[n]] for n in ALL_JOINTS])
        except KeyError:
            return

        ee_msg = PoseStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = "base_link"
        full_joints = np.zeros(len(ARM_CHAIN.links))
        for j, (i, _a) in enumerate(
                [(i, a) for i, a in enumerate(ACTIVE_MASK) if a and i > 0]):
            if j < N_ARM:
                full_joints[i] = self.q[j]
        fk = ARM_CHAIN.forward_kinematics(full_joints)
        ee_pos = np.array([fk[1, 3], fk[0, 3], fk[2, 3] + EE_OFFSET_Z])
        ee_msg.pose.position.x = float(ee_pos[0])
        ee_msg.pose.position.y = float(ee_pos[1])
        ee_msg.pose.position.z = float(ee_pos[2])
        yaw = float(self.q[self.i_yaw])
        ee_msg.pose.orientation.w = math.cos(yaw / 2)
        ee_msg.pose.orientation.z = math.sin(yaw / 2)
        self.ee_pub.publish(ee_msg)

        grip_msg = Float64()
        grip_msg.data = float(self.q[self.i_L])
        self.gripper_pub.publish(grip_msg)

    def _set_target_cb(self, request, response):
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
        response.target_set = True
        return response

    def _check_target_cb(self, request, response):
        if not self.has_target:
            response.target_reached = False
            return response
        tol = request.tolerance if request.tolerance > 0.0 else DEFAULT_TOLERANCE
        error = np.abs(self.q_des - self.q)
        arm_ok = bool(np.all(error[:N_ARM] < tol))
        yaw_ok = bool(error[self.i_yaw] < tol)
        grip_ok = bool(np.all(error[self.i_L:] < tol * 2))
        response.target_reached = arm_ok and yaw_ok and grip_ok
        return response

    def _control_loop(self):
        if not self.has_target:
            return
        err = self.q_des - self.q
        err_dot = (err - self.prev_error) * LOOP_HZ
        vel = KP * err + KD * err_dot
        vel = np.clip(vel, -MAX_VEL, MAX_VEL)
        self.prev_error = err.copy()
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
