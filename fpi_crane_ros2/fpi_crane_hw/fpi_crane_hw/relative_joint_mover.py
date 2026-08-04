#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration as RclpyDuration

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from fpi_crane_msgs.msg import RobotTrajectoryInfo
from fpi_crane_msgs.srv import RelativeJointMove

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


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

DEFAULT_LIMITED_JOINT_NAMES = [
    "slew_joint",
    "boom_joint",
    "stick_joint",
    "telescope_joint",
    "grapplecarrier_joint",
]

ALWAYS_ENFORCED_JOINT_MINIMUMS = {
    "telescope_joint": 0.13,
}

DEFAULT_UNCOMMANDED_JOINT_POSITIONS = {
    "telescope_joint": 0.13,
}

JOINT_LIMIT_TOLERANCE = 1e-9


def find_default_robot_description_file() -> str:
    if get_package_share_directory is None:
        return ""

    try:
        description_share = Path(get_package_share_directory("fpi_crane_description"))
    except Exception:
        return ""

    urdf_path = description_share / "urdf" / "fpi_crane.urdf.xacro"
    if not urdf_path.exists():
        return ""
    return str(urdf_path)


def parse_joint_limits_from_urdf(
    robot_description: str,
    joint_names: List[str],
) -> Dict[str, Dict[str, float]]:
    root = ElementTree.fromstring(robot_description)
    requested_names = set(joint_names)
    joint_limits: Dict[str, Dict[str, float]] = {}

    for joint_element in root.findall(".//joint"):
        joint_name = joint_element.attrib.get("name", "")
        if joint_name not in requested_names:
            continue

        limit_element = joint_element.find("limit")
        if limit_element is None:
            continue

        lower = limit_element.attrib.get("lower")
        upper = limit_element.attrib.get("upper")
        if lower is None or upper is None:
            continue

        joint_limits[joint_name] = {
            "lower": float(lower),
            "upper": float(upper),
        }

    return joint_limits


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


def generate_step_trajectory_points(
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

    num_samples = int(math.ceil(duration_sec / sample_period_sec))
    points: List[JointTrajectoryPoint] = []
    zero_values = [0.0 for _ in q_target]

    for i in range(1, num_samples + 1):
        t = min(i * sample_period_sec, duration_sec)

        point = JointTrajectoryPoint()
        point.positions = list(q_target)
        point.time_from_start = seconds_to_duration_msg(t)

        if include_velocities:
            point.velocities = list(zero_values)

        if include_accelerations:
            point.accelerations = list(zero_values)

        points.append(point)

    return points


def generate_ramp_trajectory_points(
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
    velocities = [
        delta / duration_sec
        for delta in dq
    ]
    zero_values = [0.0 for _ in q_target]

    num_samples = int(math.ceil(duration_sec / sample_period_sec))
    points: List[JointTrajectoryPoint] = []

    for i in range(1, num_samples + 1):
        t = min(i * sample_period_sec, duration_sec)
        alpha = t / duration_sec

        point = JointTrajectoryPoint()
        point.positions = [
            q0 + alpha * delta
            for q0, delta in zip(q_start, dq)
        ]
        point.time_from_start = seconds_to_duration_msg(t)

        if include_velocities:
            point.velocities = list(velocities)

        if include_accelerations:
            point.accelerations = list(zero_values)

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
        self.declare_parameter("enforce_joint_limits", True)
        self.declare_parameter("joint_limit_names", DEFAULT_LIMITED_JOINT_NAMES)
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_description_file", find_default_robot_description_file())

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

        self.enforce_joint_limits = (
            self.get_parameter("enforce_joint_limits")
            .get_parameter_value()
            .bool_value
        )

        self.joint_limit_names = list(self.get_parameter("joint_limit_names").value)
        self.robot_description = (
            self.get_parameter("robot_description")
            .get_parameter_value()
            .string_value
        )
        self.robot_description_file = (
            self.get_parameter("robot_description_file")
            .get_parameter_value()
            .string_value
        )
        self.joint_limits: Dict[str, Dict[str, float]] = {}
        if self.enforce_joint_limits:
            self.joint_limits = self.load_joint_limits()

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
            f"  Trajectory joints: {self.trajectory_joint_names}\n"
            f"  Joint limits: {self.format_loaded_joint_limits()}"
        )

    def load_joint_limits(self) -> Dict[str, Dict[str, float]]:
        robot_description = self.robot_description
        source = "robot_description parameter"

        if not robot_description:
            if not self.robot_description_file:
                self.get_logger().warn(
                    "Joint limit enforcement is enabled, but no robot_description "
                    "or robot_description_file is available."
                )
                return {}

            source = self.robot_description_file
            try:
                robot_description = Path(self.robot_description_file).read_text()
            except OSError as exc:
                self.get_logger().error(
                    f"Could not read robot_description_file "
                    f"'{self.robot_description_file}': {exc}"
                )
                return {}

        try:
            joint_limits = parse_joint_limits_from_urdf(
                robot_description,
                self.joint_limit_names,
            )
        except (ElementTree.ParseError, ValueError) as exc:
            self.get_logger().error(f"Could not parse joint limits from {source}: {exc}")
            return {}

        missing_limit_names = [
            name for name in self.joint_limit_names if name not in joint_limits
        ]
        if missing_limit_names:
            self.get_logger().warn(
                "No URDF lower/upper limits found for: "
                + ", ".join(missing_limit_names)
            )

        return joint_limits

    def format_loaded_joint_limits(self) -> str:
        if not self.enforce_joint_limits:
            return "disabled"
        if not self.joint_limits:
            return "enabled, no limits loaded"

        return ", ".join(
            f"{name}=[{limits['lower']:.6f}, {limits['upper']:.6f}]"
            for name, limits in self.joint_limits.items()
        )

    def find_joint_limit_violations(
        self,
        target_positions: List[float],
        joint_index_by_name: Dict[str, int],
    ) -> List[str]:
        if not self.enforce_joint_limits:
            return []

        violations = []
        for name in self.joint_limit_names:
            limits = self.joint_limits.get(name)
            joint_index = joint_index_by_name.get(name)
            if joint_index is None:
                continue

            target = target_positions[joint_index]
            lower = limits["lower"] if limits is not None else None
            upper = limits["upper"] if limits is not None else None

            minimum_lower = ALWAYS_ENFORCED_JOINT_MINIMUMS.get(name)
            if minimum_lower is not None:
                lower = minimum_lower if lower is None else max(lower, minimum_lower)

            below_lower = (
                lower is not None
                and target < lower - JOINT_LIMIT_TOLERANCE
            )
            above_upper = (
                upper is not None
                and target > upper + JOINT_LIMIT_TOLERANCE
            )
            if not below_lower and not above_upper:
                continue

            if lower is not None and upper is not None:
                limit_text = f"[{lower:.6f}, {upper:.6f}]"
            elif lower is not None:
                limit_text = f">= {lower:.6f}"
            else:
                limit_text = f"<= {upper:.6f}"

            violations.append(
                f"{name} target {target:.6f} is outside "
                f"{limit_text}"
            )

        return violations

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
        trajectory_type = (
            str(getattr(request, "trajectory_type", "smooth")).strip().lower()
            or "smooth"
        )

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

        if trajectory_type not in ("smooth", "step", "ramp"):
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "trajectory_type must be 'smooth', 'step', or 'ramp'. "
                f"Got '{trajectory_type}'."
            )
            response.duration = 0.0
            response.target_positions = []
            return response

        if velocity <= 0.0 and trajectory_type in ("smooth", "ramp"):
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "velocity must be greater than 0 for smooth or ramp trajectories."
            )
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

        commanded_joint_names = set(joint_names)
        for name, default_position in DEFAULT_UNCOMMANDED_JOINT_POSITIONS.items():
            if name in commanded_joint_names:
                continue

            joint_index = joint_index_by_name.get(name)
            if joint_index is None:
                continue

            if target_positions[joint_index] < default_position:
                target_positions[joint_index] = default_position

        limit_violations = self.find_joint_limit_violations(
            target_positions,
            joint_index_by_name,
        )
        if limit_violations:
            response.sequence_id = requested_sequence_id
            response.success = False
            response.message = (
                "Requested relative move exceeds URDF joint limits: "
                + "; ".join(limit_violations)
            )
            response.duration = 0.0
            response.target_positions = [
                target_positions[joint_index_by_name[name]]
                for name in joint_names
            ]
            return response

        max_abs_delta = max(abs(delta) for delta in delta_positions)

        if trajectory_type == "step":
            duration_sec = max(min_duration, self.sample_period_sec)
        elif trajectory_type == "ramp":
            duration_sec = max(
                max_abs_delta / velocity,
                min_duration,
                self.sample_period_sec,
            )
        elif max_abs_delta < 1e-9:
            duration_sec = max(min_duration, 0.1)
        else:
            # For smooth time-scaling, peak velocity is 1.875 times average velocity.
            # This keeps the peak joint velocity approximately <= requested velocity.
            duration_sec = max(
                1.875 * max_abs_delta / velocity,
                min_duration,
                self.sample_period_sec,
            )

        if trajectory_type == "step":
            points = generate_step_trajectory_points(
                q_target=target_positions,
                duration_sec=duration_sec,
                sample_period_sec=self.sample_period_sec,
                include_velocities=self.include_velocities,
                include_accelerations=self.include_accelerations,
            )
        elif trajectory_type == "ramp":
            points = generate_ramp_trajectory_points(
                q_start=current_positions,
                q_target=target_positions,
                duration_sec=duration_sec,
                sample_period_sec=self.sample_period_sec,
                include_velocities=self.include_velocities,
                include_accelerations=self.include_accelerations,
            )
        else:
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
            f"  trajectory_type: {trajectory_type}\n"
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
