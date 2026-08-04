from __future__ import annotations

import copy
import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from fpi_crane_msgs.srv import GrappleFK
from crane_fk import CraneFK


class FKServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("crane_fk_service")

        self.declare_parameter("urdf_path", "/home/fpiadmin/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_kinematics/urdf/fpi_crane_fk_minimal.urdf")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("target_frame", "grapplecarrier")
        self.declare_parameter("max_joint_state_age_sec", np.inf)

        urdf_path = (
            self.get_parameter("urdf_path")
            .get_parameter_value()
            .string_value
        )

        joint_states_topic = (
            self.get_parameter("joint_states_topic")
            .get_parameter_value()
            .string_value
        )

        self._base_frame = (
            self.get_parameter("base_frame")
            .get_parameter_value()
            .string_value
        )

        self._target_frame = (
            self.get_parameter("target_frame")
            .get_parameter_value()
            .string_value
        )

        self._max_joint_state_age_sec = (
            self.get_parameter("max_joint_state_age_sec")
            .get_parameter_value()
            .double_value
        )

        self.get_logger().info(f"Loading FK model from {urdf_path}")
        self._fk_model = CraneFK(urdf_path)

        self._joint_state_lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None

        self._joint_state_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self._joint_state_callback,
            10,
        )

        self._service = self.create_service(
            GrappleFK,
            "/get_grapple_fk",
            self._service_callback,
        )

        self.get_logger().info(
            "FK service ready: /get_grapple_fk"
        )

    def _joint_state_callback(self, msg: JointState) -> None:
        # Copy the message so a service callback always works with a stable
        # snapshot.
        with self._joint_state_lock:
            self._latest_joint_state = copy.deepcopy(msg)

    def _get_latest_joint_state(self) -> Optional[JointState]:
        with self._joint_state_lock:
            if self._latest_joint_state is None:
                return None

            return copy.deepcopy(self._latest_joint_state)

    def _joint_state_age_sec(self, msg: JointState) -> float:
        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

        now_ns = self.get_clock().now().nanoseconds
        return (now_ns - stamp_ns) * 1e-9

    def _service_callback(
        self,
        request: GrappleFK.Request,
        response: GrappleFK.Response,
    ) -> GrappleFK.Response:
        try:
            if request.use_latest_joint_state:
                joint_state = self._get_latest_joint_state()

                if joint_state is None:
                    raise RuntimeError(
                        "No /joint_states message has been received yet"
                    )

                age_sec = self._joint_state_age_sec(joint_state)

                # Avoid this check when bag timestamps and system time differ
                # unless use_sim_time is configured.
                if (
                    self._max_joint_state_age_sec > 0.0
                    and age_sec > self._max_joint_state_age_sec
                ):
                    raise RuntimeError(
                        "Latest JointState is stale: "
                        f"{age_sec:.3f} s old"
                    )
            else:
                joint_state = copy.deepcopy(request.joint_state)

            fk_result = self._fk_model.forward_from_joint_state(
                joint_state
            )

            response.pose = self._matrix_to_pose_stamped(
                fk_result.transform,
                joint_state,
            )

            response.z_axis = Vector3(
                x=float(fk_result.z_axis[0]),
                y=float(fk_result.z_axis[1]),
                z=float(fk_result.z_axis[2]),
            )

            response.joint_state_used = joint_state
            response.success = True
            response.message = (
                f"Computed {self._base_frame} -> "
                f"{self._target_frame}"
            )

        except Exception as exc:
            response.success = False
            response.message = str(exc)

            self.get_logger().error(
                f"FK request failed: {exc}"
            )

        return response

    def _matrix_to_pose_stamped(
        self,
        transform: np.ndarray,
        joint_state: JointState,
    ) -> PoseStamped:
        pose_msg = PoseStamped()

        # Use the acquisition time of the joint state, not the time at which
        # the client happened to call the service.
        pose_msg.header.stamp = joint_state.header.stamp
        pose_msg.header.frame_id = self._base_frame

        pose_msg.pose.position.x = float(transform[0, 3])
        pose_msg.pose.position.y = float(transform[1, 3])
        pose_msg.pose.position.z = float(transform[2, 3])

        quaternion_xyzw = Rotation.from_matrix(
            transform[:3, :3]
        ).as_quat()

        pose_msg.pose.orientation.x = float(quaternion_xyzw[0])
        pose_msg.pose.orientation.y = float(quaternion_xyzw[1])
        pose_msg.pose.orientation.z = float(quaternion_xyzw[2])
        pose_msg.pose.orientation.w = float(quaternion_xyzw[3])

        return pose_msg


def main(args=None) -> None:
    rclpy.init(args=args)

    node = FKServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()