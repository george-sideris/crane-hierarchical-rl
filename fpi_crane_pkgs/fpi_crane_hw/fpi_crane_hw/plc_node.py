#!/usr/bin/env python3
import socket
import threading
import time
from typing import Iterable, Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import JointState, Joy

try:
    # Recommended ROS 2 interface names.
    from fpi_crane_msgs.msg import PlcStatus, RobotTrajectoryInfo, PlcGripper, PlcStop
    from fpi_crane_msgs.srv import RequestGrappleMove
except ImportError:  # Compatibility shim for older/lowercase generated Python names.
    from fpi_crane_msgs.msg import plc_status as PlcStatus
    from fpi_crane_msgs.msg import RobotTrajectoryInfo, plc_gripper as PlcGripper, plc_stop as PlcStop
    from fpi_crane_msgs.srv import request_grapple_move as RequestGrappleMove

from .robot_globals import GripperDirection, PlcState, RobotLinks, UdpMessage, RRCAxes
from .udp_handler import UDPHandler

DEFAULT_JOINT_NAMES = [
    "slew_joint",
    "boom_joint",
    "stick_joint",
    "telescope_joint",
    "hanger_joint",
    "bearingfork_joint",
    "grapplecarrier_joint",
    "grappletong1_joint",
    "grappletong2_joint",
]
DEFAULT_PASSIVE_JOINT_NAMES = ["hanger_joint", "bearingfork_joint"]

JOINT_SENSOR_OFFSETS = [-0.07330376666666666,
0.005235983333333333,
0.003490655555555555,
0.0,
-0.12915425555555554,
-0.04188786666666666,
-0.21991129999999998,
0.0,
0.0]

MAX_TRAJECTORY_POINTS = 3000
PLC_UPDATE_RATE_SIM = 50.0  # Hz
NUM_RRC_AXES = 6
RAD_TO_DEG = 180.0 / 3.14159
DEG_TO_RAD = 3.14159 / 180.0
M_TO_MM = 1000.0
MS_TO_S = 1000.0


def get_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def set_attr_if_present(obj, names: Iterable[str], value) -> None:
    for name in names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return
    # As a last resort, set the first name. This is useful during early bring-up when
    # generated message classes are not available in the editor.
    setattr(obj, next(iter(names)), value)


def duration_to_nanoseconds(duration_msg) -> int:
    sec = get_attr(duration_msg, "sec", "secs", default=0)
    nanosec = get_attr(duration_msg, "nanosec", "nsec", "nsecs", default=0)
    return int(sec) * 1_000_000_000 + int(nanosec)


class PLCNode(Node):
    def __init__(self) -> None:
        super().__init__("plc_node")
        self.callback_group = ReentrantCallbackGroup()
        self.lock = threading.RLock()

        self.declare_parameter("use_simulation", False)
        self.declare_parameter("udp_recv_ip_address", "172.20.230.162")
        self.declare_parameter("udp_recv_port", 30305)
        self.declare_parameter("udp_send_ip_address", "172.20.230.120")
        self.declare_parameter("udp_send_port", 30310)
        self.declare_parameter("udp_recv_timeout", 0.005)
        self.declare_parameter("udp_send_timeout", 0.2)
        self.declare_parameter("hardware_poll_rate", 100.0)
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("passive_joint_names", DEFAULT_PASSIVE_JOINT_NAMES)
        self.declare_parameter("min_grapple_angle", 0.0)
        self.declare_parameter("max_grapple_angle", 2.23402)
        self.declare_parameter("grapple_cmd_time_ms_sim", 4000.0)
        self.declare_parameter("grapple_velocity_sim", 5)
        self.declare_parameter("telescope_scale_factor", 1.0)
        # Telescope handling. Default False: the real telescope position sensor is used
        # and the joint is commanded normally. Set telescope_static:=true to hold it
        # static when the sensor is unreliable: its reported position is forced to
        # telescope_static_value (m) with zero velocity instead of the decoded sensor
        # value, and it is never commanded to move (~0.13 m matches the retracted lock).
        # Only affects the real-hardware UDP path (sim feeds /joint_states separately).
        self.declare_parameter("telescope_static", False)
        self.declare_parameter("telescope_static_value", 0.13)
        self.declare_parameter("plc_update_rate_sim", 50.0)
        self.declare_parameter("teleop_scale_vel", 5.0)
        # Local sandbox patch: lets the trajectory commander drive grapplecarrier
        # yaw directly. When False the autolevel `-stick - boom` override is
        # skipped. Upstream currently hardcodes the override on; flip this to
        # True to restore the legacy teleop leveling behavior.
        self.declare_parameter("autolevel_grapple", False)

        self.simulation_mode = bool(self.get_parameter("use_simulation").value)
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.passive_joint_names = list(self.get_parameter("passive_joint_names").value)
        self.min_grapple_angle = float(self.get_parameter("min_grapple_angle").value)
        self.max_grapple_angle = float(self.get_parameter("max_grapple_angle").value)
        self.grapple_cmd_time_ms_sim = float(self.get_parameter("grapple_cmd_time_ms_sim").value)
        self.grapple_velocity_sim = float(self.get_parameter("grapple_velocity_sim").value)
        self.telescope_scale_factor = float(self.get_parameter("telescope_scale_factor").value)
        self.telescope_static = bool(self.get_parameter("telescope_static").value)
        self.telescope_static_value = float(self.get_parameter("telescope_static_value").value)
        self.plc_update_rate_sim = float(self.get_parameter("plc_update_rate_sim").value)
        self.teleop_scale_vel = float(self.get_parameter("teleop_scale_vel").value)
        self.autolevel_grapple = bool(self.get_parameter("autolevel_grapple").value)

        qos = QoSProfile(depth=10)
        self.joint_state = self.create_joint_state(self.joint_names)
        self.passive_joint_state = self.create_joint_state(self.passive_joint_names)
        self.joint_command = self.create_joint_state(self.joint_names)

        self.plc_status = PlcStatus()
        set_attr_if_present(self.plc_status, ["plc_trajectory_count_in_queue"], 0)
        self.rrc = Joy()
        self.rrc.axes = [0.0] * NUM_RRC_AXES

        self.stop_simulation_execution = False
        self.udp: Optional[UDPHandler] = None

        # Publishers.
        self.joint_state_pub = self.create_publisher(JointState, "/joint_states", qos)
        self.plc_status_pub = self.create_publisher(PlcStatus, "/plc_status", qos)
        self.rrc_pub = self.create_publisher(Joy, "/rrc", qos)
        self.passive_joint_state_pub = self.create_publisher(JointState, "/joint_states_passive", qos)

        # Subscribers.
        self.create_subscription(
            RobotTrajectoryInfo,
            "/RobotTrajectoryInfo",
            self.robot_trajectory_callback,
            qos,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PlcStop,
            "/plc_stop",
            self.plc_stop_trajectory_execution_callback,
            qos,
            callback_group=self.callback_group,
        )

        # Service.
        self.create_service(
            RequestGrappleMove,
            "request_grapple_move",
            self.grapple_move_callback,
            callback_group=self.callback_group,
        )

        if self.simulation_mode:
            set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.HOLD))
            self.joint_command_pub = self.create_publisher(JointState, "/joint_command", qos)
            self.create_subscription(
                JointState,
                "/joint_states_sim",
                self.simulator_joint_states_callback,
                qos,
                callback_group=self.callback_group,
            )
            self.joy_sub = self.create_subscription(
                Joy,
                "/joy",
                self.simulate_rrc_callback,
                10,
            )
            self.publish_timer = self.create_timer(1.0 / PLC_UPDATE_RATE_SIM, self.publish_sim_state)
            self.get_logger().info("PLC node initialized in SIMULATION mode")
        else:
            recv_ip = str(self.get_parameter("udp_recv_ip_address").value)
            recv_port = int(self.get_parameter("udp_recv_port").value)
            send_ip = str(self.get_parameter("udp_send_ip_address").value)
            send_port = int(self.get_parameter("udp_send_port").value)
            recv_timeout = float(self.get_parameter("udp_recv_timeout").value)
            send_timeout = float(self.get_parameter("udp_send_timeout").value)
            self.udp = UDPHandler(recv_ip, recv_port, send_ip, send_port, recv_timeout, send_timeout)
            poll_rate = float(self.get_parameter("hardware_poll_rate").value)
            self.hardware_timer = self.create_timer(1.0 / poll_rate, self.hardware_poll_once)
            self.get_logger().info(
                f"PLC node initialized in HARDWARE mode. Receiving {recv_ip}:{recv_port}; "
                f"sending {send_ip}:{send_port}"
            )

    def create_joint_state(self, names: Iterable[str]) -> JointState:
        msg = JointState()
        msg.name = list(names)
        msg.position = [0.0] * len(msg.name)
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        return msg

    def now_msg(self):
        return self.get_clock().now().to_msg()

    def publish_sim_state(self) -> None:
        with self.lock:
            self.joint_state.header.stamp = self.now_msg()
            self.plc_status_pub.publish(self.plc_status)
            self.joint_state_pub.publish(self.joint_state)
            self.rrc_pub.publish(self.rrc)
            self.simulate_teleop()

    def hardware_poll_once(self) -> None:
        if self.udp is None:
            return
        try:
            msg_id, msg_data, _ = self.udp.receive_message()
        except socket.timeout:
            return
        except ValueError as exc:
            self.get_logger().warn(f"Error receiving UDP message: {exc}")
            return

        if msg_id != int(UdpMessage.MSG_CURENT_PLC_VALUES):
            return

        with self.lock:
            self.udp_receive_callback(msg_data)
            self.joint_state_pub.publish(self.joint_state)
            self.plc_status_pub.publish(self.plc_status)
            self.rrc_pub.publish(self.rrc)
            self.passive_joint_state_pub.publish(self.passive_joint_state)

    def udp_receive_callback(self, packet: bytes) -> None:
        self.udp_receive_update_joint_states(packet)
        self.udp_receive_update_plc_status(packet)
        self.udp_receive_update_rrc(packet)
        self.udp_receive_update_passive_joint_states(packet)

    def udp_receive_update_joint_states(self, msg_data: bytes) -> None:
        if self.udp is None:
            return
        pos, vel, eff = self.udp.unpack_joint_states(msg_data)

        for i in (RobotLinks.MAST, RobotLinks.BOOM, RobotLinks.STICK, RobotLinks.GRAPPLE_BASE):
            self.joint_state.position[i] = pos[i] / 10.0 * DEG_TO_RAD - JOINT_SENSOR_OFFSETS[i]
            self.joint_state.velocity[i] = vel[i] / 10.0 * DEG_TO_RAD
            self.joint_state.effort[i] = float(eff[i])

            # if i is RobotLinks.GRAPPLE_BASE:
                # self.joint_state.position[i] = self.joint_state.position[i] + 0.2

        if self.telescope_static:
            # Sensor unreliable: report the mechanical-lock position with no motion.
            self.joint_state.position[RobotLinks.TELESCOPE] = self.telescope_static_value
            self.joint_state.velocity[RobotLinks.TELESCOPE] = 0.0
            self.joint_state.effort[RobotLinks.TELESCOPE] = 0.0
        else:
            self.joint_state.position[RobotLinks.TELESCOPE] = pos[RobotLinks.TELESCOPE] / 10000.0
            self.joint_state.velocity[RobotLinks.TELESCOPE] = vel[RobotLinks.TELESCOPE] / 10000.0
            self.joint_state.effort[RobotLinks.TELESCOPE] = float(eff[RobotLinks.TELESCOPE])
        self.joint_state.header.stamp = self.now_msg()

    def udp_receive_update_plc_status(self, msg_data: bytes) -> None:
        if self.udp is None:
            return
        plc_state, plc_seq_id, plc_move_point = self.udp.unpack_plc_states(msg_data)
        grapple_move_completeness, grapple_angle, grapple_state = self.udp.unpack_grapple_states(msg_data)

        set_attr_if_present(self.plc_status, ["plc_crane_state"], plc_state)
        set_attr_if_present(self.plc_status, ["plc_crane_sequence_id", "plc_crane_sequenceID"], plc_seq_id)
        set_attr_if_present(self.plc_status, ["plc_crane_move_point_id"], plc_move_point)
        set_attr_if_present(self.plc_status, ["plc_gripper_move_completeness"], grapple_move_completeness)
        set_attr_if_present(self.plc_status, ["plc_gripper_move_angle"], grapple_angle)
        set_attr_if_present(self.plc_status, ["plc_gripper_state"], grapple_state)

    def udp_receive_update_rrc(self, msg_data: bytes) -> None:
        if self.udp is None:
            return
        rrc_raw = self.udp.unpack_rrc_values(msg_data)

        for i in range(NUM_RRC_AXES):
            positive = rrc_raw[2 * i]
            negative = rrc_raw[2 * i + 1]
            self.rrc.axes[i] = float(positive if positive > negative else -negative)

        self.rrc.header.stamp = self.now_msg()

    def udp_receive_update_passive_joint_states(self, msg_data: bytes) -> None:
        if self.udp is None:
            return
        pos, vel = self.udp.unpack_passive_joint_states(msg_data)

        self.passive_joint_state.position = [x / 10.0 * DEG_TO_RAD for x in pos]
        self.passive_joint_state.velocity = [x / 10.0 * DEG_TO_RAD for x in vel]

        self.joint_state.position[RobotLinks.UPPERPASSIVE] = pos[0] / 10.0 * DEG_TO_RAD - JOINT_SENSOR_OFFSETS[RobotLinks.UPPERPASSIVE]
        self.joint_state.velocity[RobotLinks.UPPERPASSIVE] = vel[0] / 10.0 * DEG_TO_RAD
        self.joint_state.position[RobotLinks.LOWERPASSIVE] = pos[1] / 10.0 * DEG_TO_RAD - JOINT_SENSOR_OFFSETS[RobotLinks.LOWERPASSIVE]
        self.joint_state.velocity[RobotLinks.LOWERPASSIVE] = vel[1] / 10.0 * DEG_TO_RAD

        self.passive_joint_state.header.stamp = self.now_msg()

    def grapple_move_callback(self, request, response):
        grapple = get_attr(request, "grapple_move", "grappleMove")
        sequence_id = int(get_attr(grapple, "sequence_id", "sequenceID", default=0))
        set_attr_if_present(self.plc_status, ["plc_gripper_move_completeness"], 0)
        set_attr_if_present(self.plc_status, ["plc_gripper_sequence_id", "plc_gripper_sequenceID"], sequence_id)

        set_attr_if_present(response, ["request_id", "requestID"], sequence_id)
        set_attr_if_present(response, ["success", "succes"], False)

        if self.simulation_mode:
            set_attr_if_present(
                self.plc_status,
                ["plc_gripper_state"],
                int(get_attr(grapple, "gripper_move_to", default=GripperDirection.NONE)),
            )
            set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.EXECUTE_SEQUENCE))
            self.simulate_grapple(grapple)
            set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.HOLD))
            set_attr_if_present(self.plc_status, ["plc_gripper_state"], int(GripperDirection.NONE))
            set_attr_if_present(response, ["success", "succes"], True)
            return response

        if self.udp_send_grapple_command_message(grapple):
            set_attr_if_present(response, ["success", "succes"], True)
        return response

    def udp_send_grapple_command_message(self, grapple: PlcGripper) -> bool:
        if self.udp is None:
            return False
        gripper_move_to = int(get_attr(grapple, "gripper_move_to", default=0))
        delta_angle = int(get_attr(grapple, "delta_angle", default=0))
        delta_time = int(get_attr(grapple, "delta_time", default=0))
        grapple_data = (
            gripper_move_to.to_bytes(1, "big", signed=False)
            + delta_angle.to_bytes(2, "big", signed=False)
            + delta_time.to_bytes(2, "big", signed=False)
        )
        self.get_logger().info("Sending grapple command packet to PLC")
        ok = self.udp.send_message(int(UdpMessage.MSG_SET_GRAPPLE_POS), grapple_data)
        if not ok:
            self.get_logger().error("Unable to send grapple command packet")
        return ok
    
    def simulate_rrc_callback(self, joystick: Joy):
        joy_axes = joystick.axes
        joy_buttons = joystick.buttons

        # Guard against controllers with fewer axes/buttons than expected.
        if len(joy_axes) < 5 or len(joy_buttons) < 6:
            self.get_logger().warn(
                "Received /joy message with fewer axes/buttons than expected. "
                f"axes={len(joy_axes)}, buttons={len(joy_buttons)}"
            )
            return

        # Left joystick:
        # LB modifies horizontal left stick from mast control to telescope control.
        if joy_buttons[4] and (abs(joy_axes[0]) > abs(joy_axes[1])):
            self.rrc.axes[RRCAxes.MAST] = 0.0
            self.rrc.axes[RRCAxes.BOOM] = 0.0
            self.rrc.axes[RRCAxes.TELESCOPE] = joy_axes[0]

        elif not joy_buttons[4] and (abs(joy_axes[0]) > abs(joy_axes[1])):
            self.rrc.axes[RRCAxes.MAST] = joy_axes[0]
            self.rrc.axes[RRCAxes.BOOM] = 0.0
            self.rrc.axes[RRCAxes.TELESCOPE] = 0.0

        elif not joy_buttons[4] and (abs(joy_axes[0]) < abs(joy_axes[1])):
            self.rrc.axes[RRCAxes.MAST] = 0.0
            self.rrc.axes[RRCAxes.BOOM] = joy_axes[1]
            self.rrc.axes[RRCAxes.TELESCOPE] = 0.0

        else:
            self.rrc.axes[RRCAxes.MAST] = 0.0
            self.rrc.axes[RRCAxes.BOOM] = 0.0
            self.rrc.axes[RRCAxes.TELESCOPE] = 0.0

        # Right joystick:
        # RB modifies horizontal right stick from grapple rotation to grapple open/close.
        if joy_buttons[5] and (abs(joy_axes[3]) > abs(joy_axes[4])):
            self.rrc.axes[RRCAxes.GRAPPLE_ROTATION] = 0.0
            self.rrc.axes[RRCAxes.STICK] = 0.0
            self.rrc.axes[RRCAxes.GRAPPLE_OPEN] = joy_axes[3]

        elif not joy_buttons[5] and (abs(joy_axes[3]) > abs(joy_axes[4])):
            self.rrc.axes[RRCAxes.GRAPPLE_ROTATION] = joy_axes[3]
            self.rrc.axes[RRCAxes.STICK] = 0.0
            self.rrc.axes[RRCAxes.GRAPPLE_OPEN] = 0.0

        elif not joy_buttons[5] and (abs(joy_axes[3]) < abs(joy_axes[4])):
            self.rrc.axes[RRCAxes.GRAPPLE_ROTATION] = 0.0
            self.rrc.axes[RRCAxes.STICK] = joy_axes[4]
            self.rrc.axes[RRCAxes.GRAPPLE_OPEN] = 0.0

        else:
            self.rrc.axes[RRCAxes.GRAPPLE_ROTATION] = 0.0
            self.rrc.axes[RRCAxes.STICK] = 0.0
            self.rrc.axes[RRCAxes.GRAPPLE_OPEN] = 0.0

            self.rrc.header.stamp = self.get_clock().now().to_msg()

    def simulate_teleop(self):
        # Local sandbox patch: only override joint_command on active joystick
        # input. Without this guard, the upstream loop writes
        # joint_command.position = current_pos to every joint at 50 Hz during
        # HOLD, which clobbers any trajectory's final goal and freezes the
        # drive wherever it had gotten to. The guard lets simulate_trajectory
        # and simulate_grapple terminal targets persist into HOLD so the drive
        # keeps converging while the policy waits for the joints to settle.
        scale_vel = self.teleop_scale_vel

        if self.plc_status.plc_crane_state != PlcState.HOLD:
            return

        current_pos = self.joint_state.position
        dt = 1.0 / float(self.plc_update_rate_sim)

        def drive(joint_idx, axis_idx):
            v = self.rrc.axes[axis_idx] * scale_vel
            if v == 0.0:
                return
            self.joint_command.velocity[joint_idx] = v
            self.joint_command.position[joint_idx] = current_pos[joint_idx] + v * dt

        drive(RobotLinks.MAST, RRCAxes.MAST)
        drive(RobotLinks.BOOM, RRCAxes.BOOM)
        drive(RobotLinks.STICK, RRCAxes.STICK)
        drive(RobotLinks.TELESCOPE, RRCAxes.TELESCOPE)
        drive(RobotLinks.GRAPPLE_BASE, RRCAxes.GRAPPLE_ROTATION)

        gripper_vel = self.rrc.axes[RRCAxes.GRAPPLE_OPEN] * scale_vel
        if gripper_vel != 0.0:
            self.joint_command.velocity[RobotLinks.GRIPPER_LEFT] = gripper_vel
            self.joint_command.velocity[RobotLinks.GRIPPER_RIGHT] = gripper_vel
            self.joint_command.position[RobotLinks.GRIPPER_LEFT] = (
                current_pos[RobotLinks.GRIPPER_LEFT] + gripper_vel * dt
            )
            self.joint_command.position[RobotLinks.GRIPPER_RIGHT] = (
                current_pos[RobotLinks.GRIPPER_RIGHT] + gripper_vel * dt
            )

        self.joint_command.header.stamp = self.get_clock().now().to_msg()
        self.joint_command_pub.publish(self.joint_command)

    def simulate_grapple(self, gripper: PlcGripper) -> None:
        self.stop_simulation_execution = False
        gripper_move_to = int(get_attr(gripper, "gripper_move_to", default=GripperDirection.NONE))
        if gripper_move_to == int(GripperDirection.NONE):
            return

        start_angle = self.joint_state.position[RobotLinks.GRIPPER_LEFT]
        velocity = self.grapple_velocity_sim
        if gripper_move_to == int(GripperDirection.OPEN):
            velocity = -self.grapple_velocity_sim

        start_time = self.get_clock().now()
        duration_s = self.grapple_cmd_time_ms_sim / MS_TO_S
        period = Duration(seconds=1.0 / PLC_UPDATE_RATE_SIM)

        while rclpy.ok():
            if self.stop_simulation_execution:
                return
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= duration_s:
                break

            angle = velocity * elapsed + start_angle
            angle = min(max(angle, self.min_grapple_angle), self.max_grapple_angle)
            completeness = int(elapsed / duration_s * 100) + 1
            set_attr_if_present(self.plc_status, ["plc_gripper_move_completeness"], completeness)

            with self.lock:
                self.joint_command.position[RobotLinks.GRIPPER_LEFT] = angle
                self.joint_command.position[RobotLinks.GRIPPER_RIGHT] = angle
                self.joint_command.velocity[RobotLinks.GRIPPER_LEFT] = velocity
                self.joint_command.velocity[RobotLinks.GRIPPER_RIGHT] = velocity
                self.joint_command.header.stamp = self.now_msg()
                self.joint_command_pub.publish(self.joint_command)

            # The executor is already spinning this node, so do not call spin_once here.
            # Sleep briefly while letting the MultiThreadedExecutor handle callbacks.
            target = self.get_clock().now() + period
            while rclpy.ok() and self.get_clock().now() < target:
                time.sleep(0.001)

    def simulator_joint_states_callback(self, joint_state_from_sim: JointState) -> None:
        with self.lock:
            self.joint_state = joint_state_from_sim
            self.joint_state.header.stamp = self.now_msg()

    def robot_trajectory_callback(self, current_trajectory: RobotTrajectoryInfo) -> None:
        joint_trajectory = get_attr(current_trajectory, "joint_trajectory")
        points = list(joint_trajectory.points)
        sequence_id = int(get_attr(current_trajectory, "sequence_id", "sequenceID", default=0))
        nb_points = len(points)

        if nb_points > MAX_TRAJECTORY_POINTS:
            self.get_logger().error("Trajectory exceeds maximum number of points")
            return
        if nb_points == 0:
            self.get_logger().error("Trajectory contains zero points")
            return
        if int(get_attr(self.plc_status, "plc_trajectory_count_in_queue", default=0)):
            self.get_logger().error("There is already a trajectory in queue")
            return
        if int(get_attr(self.plc_status, "plc_crane_state", default=PlcState.HOLD)) != int(PlcState.HOLD):
            self.get_logger().error("PLC is not in HOLD state. Trajectory cannot be sent")
            return

        set_attr_if_present(self.plc_status, ["plc_trajectory_count_in_queue"], 1)
        
        if self.simulation_mode:
            set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.EXECUTE_SEQUENCE))
            thread = threading.Thread(
                target=self._simulate_trajectory_thread,
                args=(points, sequence_id),
                daemon=True,
            )
            thread.start()
        else:
            if not self.execute_plc_trajectory(points, sequence_id, nb_points):
                self.get_logger().error("Trajectory execution failed")
                set_attr_if_present(self.plc_status, ["plc_trajectory_count_in_queue"], 0)

    def _simulate_trajectory_thread(self, points, sequence_id: int) -> None:
        try:
            self.simulate_trajectory(points, sequence_id)
        finally:
            set_attr_if_present(self.plc_status, ["plc_trajectory_count_in_queue"], 0)
            if not self.stop_simulation_execution:
                set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.HOLD))

    def execute_plc_trajectory(self, points, sequence_id: int, nb_points: int) -> bool:
        self.get_logger().info(f"Processing {nb_points} trajectory points to send to PLC")

        if not self.udp_send_trajectory_header_message(sequence_id, nb_points):
            return False

        points_buffer = self.convert_trajectory_to_bytes_buffer(points)
        if self.udp is None or not self.udp.send_message(int(UdpMessage.MSG_TRAJ), points_buffer):
            self.get_logger().error("Failed to send all trajectory points to PLC")
            return False

        if not self.udp_send_plc_start_execution_message():
            return False

        self.get_logger().info("Starting trajectory execution. Pay attention to the crane")

        set_attr_if_present(self.plc_status, ["plc_trajectory_count_in_queue"], 0)
        return True

    def convert_trajectory_to_bytes_buffer(self, points) -> bytes:
        data = bytearray()
        actuated_joints = (
            RobotLinks.MAST,
            RobotLinks.BOOM,
            RobotLinks.STICK,
            RobotLinks.TELESCOPE,
            RobotLinks.GRAPPLE_BASE,
        )

        for point_id, point in enumerate(points):
            positions = list(point.positions)
            velocities = list(point.velocities)
            accelerations = list(point.accelerations)

            data.extend(int(point_id).to_bytes(2, "big", signed=True))
            for joint_index in actuated_joints:
                if joint_index == RobotLinks.TELESCOPE:
                    scale = M_TO_MM * 10.0
                    if self.telescope_static:
                        # Hold the telescope at its locked position; never command motion.
                        pos_j, vel_j, acc_j = self.telescope_static_value, 0.0, 0.0
                    else:
                        pos_j, vel_j, acc_j = (positions[joint_index],
                                               velocities[joint_index],
                                               accelerations[joint_index])
                else:
                    scale = RAD_TO_DEG * 10.0
                    pos_j, vel_j, acc_j = (positions[joint_index],
                                           velocities[joint_index],
                                           accelerations[joint_index])
                data.extend(int(pos_j * scale).to_bytes(2, "big", signed=True))
                data.extend(int(vel_j * scale).to_bytes(2, "big", signed=True))
                data.extend(int(acc_j * scale).to_bytes(2, "big", signed=True))

        self.get_logger().info(f"Formatted trajectory buffer with {len(points)} points")
        return bytes(data)

    def udp_send_trajectory_header_message(self, sequence_id: int, num_points: int) -> bool:
        if self.udp is None:
            return False
        traj_header_data = int(sequence_id).to_bytes(3, "big", signed=False) + int(num_points).to_bytes(
            2, "big", signed=False
        )
        ok = self.udp.send_message(int(UdpMessage.MSG_TRAJ_HEADER), traj_header_data)
        if ok:
            self.get_logger().info(f"Sent trajectory header with {num_points} points")
        else:
            self.get_logger().error("Failed to send trajectory header message to PLC")
        return ok

    def simulate_trajectory(self, points, sequence_id: int) -> None:
        self.stop_simulation_execution = False
        previous_time_ns = 0
        set_attr_if_present(self.plc_status, ["plc_crane_sequence_id", "plc_crane_sequenceID"], sequence_id)

        for i, point in enumerate(points):
            if self.stop_simulation_execution:
                return

            set_attr_if_present(self.plc_status, ["plc_crane_move_point_id"], i + 1)
            time_offset_ns = duration_to_nanoseconds(point.time_from_start)
            delta_ns = max(0, time_offset_ns - previous_time_ns)

            if delta_ns > 0:
                sleep_s = delta_ns / 1e9
                end_time = self.get_clock().now() + Duration(seconds=sleep_s)
                while rclpy.ok() and self.get_clock().now() < end_time:
                    if self.stop_simulation_execution:
                        return
                    time.sleep(0.001)

            with self.lock:
                for joint_index in (
                    RobotLinks.MAST,
                    RobotLinks.BOOM,
                    RobotLinks.STICK,
                    RobotLinks.GRAPPLE_BASE,
                ):
                    self.joint_command.position[joint_index] = point.positions[joint_index]
                    self.joint_command.velocity[joint_index] = point.velocities[joint_index]

                self.joint_command.position[RobotLinks.TELESCOPE] = (
                    point.positions[RobotLinks.TELESCOPE] * self.telescope_scale_factor
                )
                self.joint_command.velocity[RobotLinks.TELESCOPE] = (
                    point.velocities[RobotLinks.TELESCOPE] * self.telescope_scale_factor
                )

                # Local sandbox patch: skip autolevel by default so trajectories
                # can command grapplecarrier yaw directly. Flip autolevel_grapple
                # on to restore the legacy teleop fake-joint behavior.
                if self.autolevel_grapple:
                    self.joint_command.position[RobotLinks.GRAPPLE_BASE] = (
                        -self.joint_command.position[RobotLinks.STICK]
                        - self.joint_command.position[RobotLinks.BOOM]
                    )

                self.joint_command.header.stamp = self.now_msg()
                set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.EXECUTE_SEQUENCE))
                self.joint_command_pub.publish(self.joint_command)

            previous_time_ns = time_offset_ns

        set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.END_SEQUENCE))
        self.plc_status_pub.publish(self.plc_status)

    def plc_stop_trajectory_execution_callback(self, _msg: PlcStop) -> None:
        if self.simulation_mode:
            self.stop_simulation_execution = True
            set_attr_if_present(self.plc_status, ["plc_crane_state"], int(PlcState.HOLD))
        else:
            if not self.udp_send_plc_stop_execution_message():
                self.get_logger().error("Failed to send PLC stop message")

    def udp_send_plc_stop_execution_message(self) -> bool:
        if self.udp is None:
            return False
        ok = self.udp.send_message(int(UdpMessage.MSG_START_STOP_EXECUTION), b"\x01")
        if ok:
            self.get_logger().info("Stop message sent to PLC")
        else:
            self.get_logger().error("Unable to send STOP message to PLC")
        return ok

    def udp_send_plc_start_execution_message(self) -> bool:
        if self.udp is None:
            return False
        ok = self.udp.send_message(int(UdpMessage.MSG_START_STOP_EXECUTION), b"\x02")
        if ok:
            self.get_logger().info("START message sent to PLC")
        else:
            self.get_logger().error("Unable to send START message to PLC")
        return ok

    def destroy_node(self) -> bool:
        if self.udp is not None:
            self.udp.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PLCNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
