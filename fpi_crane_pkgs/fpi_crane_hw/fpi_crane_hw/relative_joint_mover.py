#!/usr/bin/env python3

import math
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from fpi_crane_msgs.msg import RobotTrajectoryInfo
from fpi_crane_msgs.srv import RelativeJointMove


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


def seconds_to_duration_msg(seconds: float) -> Duration:
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1e9))

    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000

    return Duration(sec=sec, nanosec=nanosec)


def generate_hydraulic_smooth_trajectory_points(
    q_start: List[float],
    q_target: List[float],
    duration_sec: float,
    sample_period_sec: float,
    include_velocities: bool = False,
    include_accelerations: bool = False,
) -> List[JointTrajectoryPoint]:

    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be greater than 0")

    if sample_period_sec <= 0.0:
        raise ValueError("sample_period_sec must be greater than 0")

    if len(q_start) != len(q_target):
        raise ValueError("q_start and q_target must have the same length")

    dq = [
        qf - q0
        for q0, qf in zip(q_start, q_target)
    ]

    num_samples = int(math.ceil(duration_sec / sample_period_sec))
    points: List[JointTrajectoryPoint] = []

    for i in range(1, num_samples + 1):
        t = min(i * sample_period_sec, duration_sec)
        tau = t / duration_sec

        # 7th-order smoothstep
        s = (
            35.0 * tau**4
            - 84.0 * tau**5
            + 70.0 * tau**6
            - 20.0 * tau**7
        )

        # First derivative ds/dt
        ds_dt = (
            140.0 * tau**3
            - 420.0 * tau**4
            + 420.0 * tau**5
            - 140.0 * tau**6
        ) / duration_sec

        # Second derivative d2s/dt2
        dds_dt2 = (
            420.0 * tau**2
            - 1680.0 * tau**3
            + 2100.0 * tau**4
            - 840.0 * tau**5
        ) / (duration_sec**2)

        positions = [
            q0 + s * delta
            for q0, delta in zip(q_start, dq)
        ]

        velocities = [
            ds_dt * delta
            for delta in dq
        ]

        accelerations = [
            dds_dt2 * delta
            for delta in dq
        ]

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = seconds_to_duration_msg(t)

        if include_velocities:
            point.velocities = velocities

        if include_accelerations:
            point.accelerations = accelerations

        points.append(point)

    return points


class RelativeJointMover(Node):
    def __init__(self):
        super().__init__("relative_joint_mover")

        self.declare_parameter("trajectory_topic", "/RobotTrajectoryInfo")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("trajectory_joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("initial_sequence_id", 1)
        self.declare_parameter("max_joint_state_age_sec", 1.0)
        self.declare_parameter("sample_period_sec", 0.02)
        self.declare_parameter("include_velocities", True)
        self.declare_parameter("include_accelerations", True)

        self.trajectory_topic = (
            self.get_parameter("trajectory_topic")
            .get_parameter_value()
            .string_value
        )

        self.joint_states_topic = (
            self.get_parameter("joint_states_topic")
            .get_parameter_value()
            .string_value
        )

        self.trajectory_joint_names = list(
            self.get_parameter("trajectory_joint_names").value
        )
        self.next_sequence_id = int(self.get_parameter("initial_sequence_id").value)

        self.max_joint_state_age_sec = (
            self.get_parameter("max_joint_state_age_sec")
            .get_parameter_value()
            .double_value
        )

        self.sample_period_sec = (
            self.get_parameter("sample_period_sec")
            .get_parameter_value()
            .double_value
        )

        self.include_velocities = (
            self.get_parameter("include_velocities")
            .get_parameter_value()
            .bool_value
        )

        self.include_accelerations = (
            self.get_parameter("include_accelerations")
            .get_parameter_value()
            .bool_value
        )

        self.latest_joint_positions: Dict[str, float] = {}
        self.last_joint_state_receive_time = None

        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            10,
        )

        self.trajectory_pub = self.create_publisher(
            RobotTrajectoryInfo,
            self.trajectory_topic,
            10,
        )

        self.service = self.create_service(
            RelativeJointMove,
            "relative_joint_move",
            self.handle_relative_joint_move,
        )

        self.get_logger().info(
            f"Relative joint mover ready.\n"
            f"  Listening to: {self.joint_states_topic}\n"
            f"  Publishing to: {self.trajectory_topic}\n"
            f"  Service: /relative_joint_move\n"
            f"  Sample period: {self.sample_period_sec:.3f} s\n"
            f"  Trajectory joints: {self.trajectory_joint_names}"
        )

    def joint_state_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            self.latest_joint_positions[name] = position

        self.last_joint_state_receive_time = self.get_clock().now()

    def joint_states_are_fresh(self) -> bool:
        if self.last_joint_state_receive_time is None:
            return False

        age = self.get_clock().now() - self.last_joint_state_receive_time
        return age < RclpyDuration(seconds=self.max_joint_state_age_sec)

    def allocate_sequence_id(self, requested_sequence_id: int) -> int:
        if requested_sequence_id > 0:
            return requested_sequence_id

        sequence_id = self.next_sequence_id
        self.next_sequence_id += 1
        if self.next_sequence_id > 0xFFFFFFFF:
            self.next_sequence_id = 1
        return sequence_id

    def handle_relative_joint_move(self, request, response):
        requested_sequence_id = int(request.sequence_id)
        joint_names: List[str] = list(request.joint_names)
        delta_positions: List[float] = list(request.delta_positions)
        velocity: float = float(request.velocity)
        min_duration: float = float(request.min_duration)

        if not self.joint_states_are_fresh():
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "No recent /joint_states received. "
                "Cannot compute relative move."
            )
            response.duration = 0.0
            response.target_positions = []
            return response

        if len(joint_names) == 0:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = "joint_names cannot be empty."
            response.duration = 0.0
            response.target_positions = []
            return response

        if len(joint_names) != len(delta_positions):
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "joint_names and delta_positions must have the same length. "
                f"Got {len(joint_names)} names and {len(delta_positions)} deltas."
            )
            response.duration = 0.0
            response.target_positions = []
            return response

        if velocity <= 0.0:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = "velocity must be greater than 0."
            response.duration = 0.0
            response.target_positions = []
            return response

        if self.sample_period_sec <= 0.0:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = "sample_period_sec parameter must be greater than 0."
            response.duration = 0.0
            response.target_positions = []
            return response

        missing_joints = [
            name for name in self.trajectory_joint_names
            if name not in self.latest_joint_positions
        ]

        if missing_joints:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "Could not find these joints in /joint_states: "
                + ", ".join(missing_joints)
            )
            response.duration = 0.0
            response.target_positions = []
            return response

        unknown_joints = [
            name for name in joint_names
            if name not in self.trajectory_joint_names
        ]

        if unknown_joints:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "Cannot command joints outside trajectory_joint_names: "
                + ", ".join(unknown_joints)
            )
            response.duration = 0.0
            response.target_positions = []
            return response

        current_positions = [
            self.latest_joint_positions[name]
            for name in self.trajectory_joint_names
        ]
        target_positions = list(current_positions)
        joint_index_by_name = {
            name: index
            for index, name in enumerate(self.trajectory_joint_names)
        }

        for name, delta in zip(joint_names, delta_positions):
            target_positions[joint_index_by_name[name]] += delta

        max_abs_delta = max(abs(delta) for delta in delta_positions)

        if max_abs_delta < 1e-9:
            duration_sec = max(min_duration, 0.1)
        else:
            # For quintic time-scaling, peak velocity is 1.875 times average velocity.
            # This keeps the peak joint velocity approximately <= requested velocity.
            duration_sec = max(
                1.875 * max_abs_delta / velocity,
                min_duration,
                self.sample_period_sec,
            )

        points = generate_hydraulic_smooth_trajectory_points(
            q_start=current_positions,
            q_target=target_positions,
            duration_sec=duration_sec,
            sample_period_sec=self.sample_period_sec,
            include_velocities=self.include_velocities,
            include_accelerations=self.include_accelerations,
        )

        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = self.trajectory_joint_names
        traj_msg.points = points

        sequence_id = self.allocate_sequence_id(requested_sequence_id)

        trajectory_info = RobotTrajectoryInfo()
        trajectory_info.sequence_id = sequence_id
        trajectory_info.joint_trajectory = traj_msg

        self.trajectory_pub.publish(trajectory_info)

        response.sequence_id = sequence_id
        response.success = True
        response.message = (
            f"Published {len(points)} trajectory points to {self.trajectory_topic}"
        )
        response.duration = duration_sec
        response.target_positions = [
            target_positions[joint_index_by_name[name]]
            for name in joint_names
        ]

        self.get_logger().info(
            f"Published relative move:\n"
            f"  sequence_id: {sequence_id}\n"
            f"  joints: {joint_names}\n"
            f"  deltas: {delta_positions}\n"
            f"  targets: {response.target_positions}\n"
            f"  duration: {duration_sec:.3f} s\n"
            f"  points: {len(points)}"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = RelativeJointMover()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
