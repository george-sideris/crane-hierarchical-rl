#!/usr/bin/env python3

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import JointState, Joy

from fpi_crane_hw.robot_globals import PlcState, RRCAxes

try:
    from fpi_crane_msgs.msg import PlcStatus, RobotTrajectoryInfo
except ImportError:
    from fpi_crane_msgs.msg import plc_status as PlcStatus
    from fpi_crane_msgs.msg import RobotTrajectoryInfo


def get_attr(obj, *names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


JOINT_TO_RRC_AXIS = {
    "slew_joint": RRCAxes.MAST,
    "boom_joint": RRCAxes.BOOM,
    "stick_joint": RRCAxes.STICK,
    "telescope_joint": RRCAxes.TELESCOPE,
    "grapplecarrier_joint": RRCAxes.GRAPPLE_ROTATION,
}


class TrajectoryTrackingMonitor(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_tracking_monitor")

        self.declare_parameter("trajectory_topic", "/RobotTrajectoryInfo")
        self.declare_parameter("plc_status_topic", "/plc_status")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("rrc_topic", "/rrc")
        self.declare_parameter("joint_setpoint_topic", "/joint_setpoint")
        self.declare_parameter("joint_tracking_error_topic", "/jooint_tracking_error")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("csv_directory", "/workspace/bags/errors")
        self.declare_parameter("csv_flush_on_write", True)

        self.trajectory_topic = str(self.get_parameter("trajectory_topic").value)
        self.plc_status_topic = str(self.get_parameter("plc_status_topic").value)
        self.joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self.rrc_topic = str(self.get_parameter("rrc_topic").value)
        self.joint_setpoint_topic = str(
            self.get_parameter("joint_setpoint_topic").value
        )
        self.joint_tracking_error_topic = str(
            self.get_parameter("joint_tracking_error_topic").value
        )
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.csv_directory = str(self.get_parameter("csv_directory").value)
        self.csv_flush_on_write = bool(self.get_parameter("csv_flush_on_write").value)

        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be greater than 0")

        qos = QoSProfile(depth=10)
        self.latest_trajectory: Optional[RobotTrajectoryInfo] = None
        self.latest_plc_status: Optional[PlcStatus] = None
        self.latest_joint_positions: Dict[str, float] = {}
        self.latest_joint_velocities: Dict[str, float] = {}
        self.latest_joint_efforts: Dict[str, float] = {}
        self.latest_rrc_axes: List[float] = []
        self.last_active_point_id: Optional[int] = None
        self.warned_missing_joints: Tuple[str, ...] = ()
        self.csv_file = None
        self.csv_writer = None
        self.csv_path: Optional[Path] = None
        self.csv_start_time_sec: Optional[float] = None
        self.active_csv_sequence_id: Optional[int] = None
        self.csv_has_rows = False

        self.joint_setpoint_pub = self.create_publisher(
            JointState,
            self.joint_setpoint_topic,
            qos,
        )
        self.joint_tracking_error_pub = self.create_publisher(
            JointState,
            self.joint_tracking_error_topic,
            qos,
        )

        self.create_subscription(
            RobotTrajectoryInfo,
            self.trajectory_topic,
            self.trajectory_callback,
            qos,
        )
        self.create_subscription(
            PlcStatus,
            self.plc_status_topic,
            self.plc_status_callback,
            qos,
        )
        self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            qos,
        )
        self.create_subscription(
            Joy,
            self.rrc_topic,
            self.rrc_callback,
            qos,
        )

        self.publish_timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_tracking,
        )

        self.get_logger().info(
            f"Trajectory tracking monitor ready.\n"
            f"  Trajectory: {self.trajectory_topic}\n"
            f"  PLC status: {self.plc_status_topic}\n"
            f"  Joint states: {self.joint_states_topic}\n"
            f"  RRC: {self.rrc_topic}\n"
            f"  Setpoint: {self.joint_setpoint_topic}\n"
            f"  Tracking error: {self.joint_tracking_error_topic}\n"
            f"  CSV directory: {self.csv_directory or 'disabled'}"
        )

    def open_csv(self, sequence_id: int) -> None:
        self.close_csv()
        if not self.csv_directory:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = (
            Path(self.csv_directory).expanduser()
            / f"{timestamp}_tracking_errors.csv"
        )
        try:
            if path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = path.open("w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(
                [
                    "time_sec",
                    "elapsed_sec",
                    "sequence_id",
                    "point_id",
                    "joint_name",
                    "target_position",
                    "actual_position",
                    "position_error",
                    "actual_rrc",
                    "target_velocity",
                    "actual_velocity",
                    "velocity_error",
                    "target_acceleration",
                    "actual_effort",
                ]
            )
            self.csv_file.flush()
            self.csv_path = path
            self.csv_start_time_sec = None
            self.active_csv_sequence_id = sequence_id
            self.csv_has_rows = False
            self.get_logger().info(f"Logging tracking errors to {path}")
        except OSError as exc:
            self.csv_file = None
            self.csv_writer = None
            self.csv_path = None
            self.active_csv_sequence_id = None
            self.csv_has_rows = False
            self.get_logger().error(f"Could not open CSV file '{path}': {exc}")

    def close_csv(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.csv_start_time_sec = None
        self.active_csv_sequence_id = None
        self.csv_has_rows = False

    def trajectory_callback(self, msg: RobotTrajectoryInfo) -> None:
        joint_trajectory = get_attr(msg, "joint_trajectory")
        points = list(get_attr(joint_trajectory, "points", default=[]))

        if not points:
            self.get_logger().warn("Ignoring trajectory with no points")
            return

        self.latest_trajectory = msg
        self.last_active_point_id = None
        self.warned_missing_joints = ()

        sequence_id = int(get_attr(msg, "sequence_id", "sequenceID", default=0))
        self.open_csv(sequence_id)
        self.get_logger().info(
            f"Stored trajectory sequence {sequence_id} with {len(points)} points"
        )

    def plc_status_callback(self, msg: PlcStatus) -> None:
        self.latest_plc_status = msg

    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_positions = {
            name: position
            for name, position in zip(msg.name, msg.position)
        }
        self.latest_joint_velocities = {
            name: velocity
            for name, velocity in zip(msg.name, msg.velocity)
        }
        self.latest_joint_efforts = {
            name: effort
            for name, effort in zip(msg.name, msg.effort)
        }

    def rrc_callback(self, msg: Joy) -> None:
        self.latest_rrc_axes = list(msg.axes)

    def publish_tracking(self) -> None:
        active_point = self.get_active_point()
        if active_point is None:
            return

        joint_names, point, sequence_id, point_id = active_point
        stamp = self.get_clock().now().to_msg()

        setpoint_msg = JointState()
        setpoint_msg.header.stamp = stamp
        setpoint_msg.name = joint_names
        setpoint_msg.position = list(point.positions)
        setpoint_msg.velocity = list(point.velocities)
        setpoint_msg.effort = []
        self.joint_setpoint_pub.publish(setpoint_msg)

        if not self.latest_joint_positions:
            return

        missing_joints = tuple(
            name for name in joint_names if name not in self.latest_joint_positions
        )
        if missing_joints:
            if missing_joints != self.warned_missing_joints:
                self.get_logger().warn(
                    "Cannot publish complete tracking error. Missing joints in "
                    f"{self.joint_states_topic}: {list(missing_joints)}"
                )
                self.warned_missing_joints = missing_joints
            return

        error_msg = JointState()
        error_msg.header.stamp = stamp
        error_msg.name = joint_names
        error_msg.position = [
            self.latest_joint_positions[name] - setpoint
            for name, setpoint in zip(joint_names, point.positions)
        ]
        error_msg.velocity = self.compute_velocity_error(joint_names, point.velocities)
        error_msg.effort = []
        self.joint_tracking_error_pub.publish(error_msg)
        self.write_csv_rows(
            stamp,
            sequence_id,
            point_id,
            joint_names,
            point.positions,
            point.velocities,
            point.accelerations,
            error_msg.position,
            error_msg.velocity,
        )

    def get_active_point(self):
        if self.latest_trajectory is None or self.latest_plc_status is None:
            return None

        plc_state = int(
            get_attr(self.latest_plc_status, "plc_crane_state", default=PlcState.HOLD)
        )
        if plc_state != int(PlcState.EXECUTE_SEQUENCE):
            if self.csv_writer is not None and self.csv_has_rows:
                self.close_csv()
            return None

        trajectory_sequence_id = int(
            get_attr(self.latest_trajectory, "sequence_id", "sequenceID", default=0)
        )
        plc_sequence_id = int(
            get_attr(
                self.latest_plc_status,
                "plc_crane_sequence_id",
                "plc_crane_sequenceID",
                default=0,
            )
        )
        if trajectory_sequence_id and plc_sequence_id:
            if trajectory_sequence_id != plc_sequence_id:
                return None

        joint_trajectory = get_attr(self.latest_trajectory, "joint_trajectory")
        points = list(get_attr(joint_trajectory, "points", default=[]))
        joint_names = list(get_attr(joint_trajectory, "joint_names", default=[]))

        if not points or not joint_names:
            return None

        point_id = int(
            get_attr(self.latest_plc_status, "plc_crane_move_point_id", default=0)
        )
        if point_id <= 0:
            return None

        point_index = point_id - 1
        if point_index >= len(points):
            self.close_csv()
            if point_id != self.last_active_point_id:
                self.get_logger().warn(
                    f"PLC point id {point_id} is outside stored trajectory with "
                    f"{len(points)} points"
                )
                self.last_active_point_id = point_id
            return None

        if len(points[point_index].positions) != len(joint_names):
            if point_id != self.last_active_point_id:
                self.get_logger().warn(
                    f"Trajectory point {point_id} has "
                    f"{len(points[point_index].positions)} positions for "
                    f"{len(joint_names)} joints"
                )
                self.last_active_point_id = point_id
            return None

        if point_id != self.last_active_point_id:
            self.last_active_point_id = point_id

        return joint_names, points[point_index], trajectory_sequence_id, point_id

    def compute_velocity_error(
        self,
        joint_names: List[str],
        setpoint_velocities: List[float],
    ) -> List[float]:
        if len(setpoint_velocities) != len(joint_names):
            return []

        if any(name not in self.latest_joint_velocities for name in joint_names):
            return []

        return [
            self.latest_joint_velocities[name] - setpoint_velocity
            for name, setpoint_velocity in zip(joint_names, setpoint_velocities)
        ]

    def write_csv_rows(
        self,
        stamp,
        sequence_id: int,
        point_id: int,
        joint_names: List[str],
        target_positions: List[float],
        target_velocities: List[float],
        target_accelerations: List[float],
        position_errors: List[float],
        velocity_errors: List[float],
    ) -> None:
        if self.csv_writer is None:
            return
        if (
            self.active_csv_sequence_id is not None
            and sequence_id != self.active_csv_sequence_id
        ):
            return

        time_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.csv_start_time_sec is None:
            self.csv_start_time_sec = time_sec
        elapsed_sec = time_sec - self.csv_start_time_sec

        has_target_velocities = len(target_velocities) == len(joint_names)
        has_target_accelerations = len(target_accelerations) == len(joint_names)
        has_velocity_errors = len(velocity_errors) == len(joint_names)

        for index, name in enumerate(joint_names):
            target_position = target_positions[index]
            actual_position = self.latest_joint_positions[name]
            actual_velocity = self.latest_joint_velocities.get(name, "")
            actual_effort = self.latest_joint_efforts.get(name, "")
            target_velocity = target_velocities[index] if has_target_velocities else ""
            target_acceleration = (
                target_accelerations[index] if has_target_accelerations else ""
            )
            velocity_error = velocity_errors[index] if has_velocity_errors else ""
            actual_rrc = self.get_actual_rrc(name)

            self.csv_writer.writerow(
                [
                    time_sec,
                    elapsed_sec,
                    sequence_id,
                    point_id,
                    name,
                    target_position,
                    actual_position,
                    position_errors[index],
                    actual_rrc,
                    target_velocity,
                    actual_velocity,
                    velocity_error,
                    target_acceleration,
                    actual_effort,
                ]
            )

        self.csv_has_rows = True
        if self.csv_flush_on_write and self.csv_file is not None:
            self.csv_file.flush()

    def get_actual_rrc(self, joint_name: str):
        axis = JOINT_TO_RRC_AXIS.get(joint_name)
        if axis is None or int(axis) >= len(self.latest_rrc_axes):
            return ""
        return self.latest_rrc_axes[int(axis)]

    def destroy_node(self) -> bool:
        self.close_csv()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryTrackingMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
