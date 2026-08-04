#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from sim_interface.srv import SetTarget, CheckTargetReached
import os
import numpy as np
from ikpy.chain import Chain
import traceback


URDF_PATH = os.path.join(get_package_share_directory('sim_interface'), 'urdf', 'log_loader_kinematics.urdf')
BASE_FRAME = "base_link"
ARM_JOINTS = ['slew_joint', 'boom_joint', 'stick_joint', 'grapplecarrier_joint']
GRAPPLE_JOINTS = ['grappletong1_joint', 'grappletong2_joint']
JOINT_STATES_TOPIC = "/joint_states_sim"
JOINT_COMMAND_TOPIC = "/joint_command"
KP = [7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
KD = [2, 2, 2.5, 1, 1, 1]
CONTROLLER_RATE = 10 # Hz
DEFAULT_TOLERANCE = 0.035 # radians


class JointVelocityControllerNode(Node):
    def __init__(self, urdf_path=URDF_PATH, base_frame=BASE_FRAME, arm_joints = ARM_JOINTS, grapple_joints = GRAPPLE_JOINTS, joint_states_topic = JOINT_STATES_TOPIC, joint_command_topic=JOINT_COMMAND_TOPIC, p_gains=KP, d_gains=KD, controller_rate=CONTROLLER_RATE, default_tolerance=DEFAULT_TOLERANCE):
        super().__init__('joint_planner')
        self.current_position = None
        self.target_position = None
        self.position_error = None
        self.p_gains=p_gains
        self.d_gains = d_gains
        self.controller_rate = controller_rate
        self.default_tolerance = default_tolerance
        self.arm_joint_names = arm_joints
        self.grapple_joint_names = grapple_joints
        self.robot_chain = Chain.from_urdf_file(
            urdf_path,
            base_elements=[base_frame],
            active_links_mask=[False, True, True, True, True]
        )
        self.joint_state_sub = self.create_subscription(JointState, joint_states_topic, self.joint_state_callback, 10)
        self.velocity_pub = self.create_publisher(JointState, joint_command_topic, 10)
        self.set_target_srv = self.create_service(SetTarget, 'set_target', self.set_target_handler)
        self.check_target_reached_srv = self.create_service(CheckTargetReached, 'is_target_reached', self.check_target_reached_handler)
        self.create_timer(1/self.controller_rate, self.control_loop)

    def joint_state_callback(self, msg):
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        try:
            self.current_position = [msg.position[name_to_index[j]] for j in self.arm_joint_names + self.grapple_joint_names]
        except KeyError:
            self.get_logger().warn("Joint names in joint states topic do not match expected names")
    
    def check_target_reached_handler(self, request, response):
        if self.current_position is None or self.target_position is None:
            response.target_reached = False
            return response
        error = np.abs(np.array(self.current_position) - np.array(self.target_position))
        tolerance = request.tolerance if request.tolerance > 0.0 else self.default_tolerance
        arm_target_reached = bool(np.all(error[:4] < tolerance))
        grapple_target_reached = bool(np.all(error[4:6] < tolerance)) or self.check_for_grapple_stall()
        response.target_reached = arm_target_reached and grapple_target_reached
        return response

    def set_target_handler(self, request, response):
        try:
            target_tf = np.eye(4)
            target_tf[:3, 3] = np.array(request.target_xyz) 
            ik_result = self.robot_chain.inverse_kinematics_frame(target_tf)
            if not self.is_ik_solution_valid(ik_result, target_tf):
                self.get_logger().warn("Inverse Kinematics solution is invalid.")
                response.target_set = False
                return response
            self.target_position = np.append(
                np.append(ik_result[1:4], np.deg2rad(-request.target_yaw) - ik_result[1]),
                [self.map_grapple_opening(request.grapple_opening), -self.map_grapple_opening(request.grapple_opening)]
            )
            response.target_set = True
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f'Exception occurred:\n{tb_str}')
            response.target_set = False
        return response
    
    def check_for_grapple_stall(self, stall_margin=0.05):
        return bool(np.abs(self.current_position[4] + self.current_position[5]) >= stall_margin)

    def is_ik_solution_valid(self, ik_result, target_tf, position_threshold=0.1):
        achieved_tf = self.robot_chain.forward_kinematics(ik_result)
        achieved_pos = achieved_tf[:3, 3]
        target_pos = target_tf[:3, 3]
        position_error = np.linalg.norm(achieved_pos - target_pos)
        return position_error <= position_threshold

    def send_velocity_command(self, joint_velocities):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.arm_joint_names + self.grapple_joint_names
        msg.velocity = [float(v) for v in joint_velocities]
        self.velocity_pub.publish(msg)
    
    def map_grapple_opening(self, value, min_angle=-45, max_angle=60):
        value = np.clip(value, 0.0, 1.0)
        return np.deg2rad(min_angle + value * (max_angle - min_angle))
    
    def control_loop(self):
        if self.current_position is None or self.target_position is None:
            return
        error = np.array(self.target_position) - np.array(self.current_position)
        error_derivative = (error - (np.zeros_like(error) if self.position_error is None else self.position_error))*self.controller_rate
        self.position_error = error
        velocity = self.p_gains * error + self.d_gains * error_derivative
        self.send_velocity_command(velocity)


def main(args=None):
    rclpy.init(args=args)
    node = JointVelocityControllerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
