#!/usr/bin/env python3

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Vector3
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformException, TransformListener

from fpi_crane_msgs.msg import StabilityMetric
from fpi_crane_msgs.srv import Stability


MODE_BASE = 0
MODE_GAZE = 1
MODE_IMU = 2
MODE_IMU_ORIENTATION = 3
MODE_ALL = 4


@dataclass
class ImuSample:
    stamp: TimeMsg
    frame_id: str
    acceleration: np.ndarray | None
    angular_velocity: np.ndarray | None
    orientation: np.ndarray | None


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot normalize zero or non-finite vector")

    return vector / norm


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float):
    quaternion = np.array([x, y, z, w], dtype=float)
    quaternion = normalize(quaternion)

    x, y, z, w = quaternion

    return np.array([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])


def vector3_from_numpy(vector: np.ndarray) -> Vector3:
    return Vector3(
        x=float(vector[0]),
        y=float(vector[1]),
        z=float(vector[2]),
    )


def stamp_to_nanoseconds(stamp: TimeMsg) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def has_acceleration_gravity_data(sample: ImuSample) -> bool:
    return (
        sample.acceleration is not None
        and sample.angular_velocity is not None
    )


def has_orientation_gravity_data(sample: ImuSample) -> bool:
    return sample.orientation is not None


class StabilityServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("stability_service")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("grapple_frame", "grapplecarrier")
        self.declare_parameter(
            "imu_topic",
            "/zed_0/zed_node/imu/data",
        )
        self.declare_parameter("imu_frame_override", "")

        self.declare_parameter("default_exponent", 4.0)

        # Free-hanging gaze result obtained from your original bag.
        # Store it as a parameter even if you consider it a fixed calibration.
        self.declare_parameter(
            "gaze_up_base",
            [
                0.011373999980936902,
                -0.0367031558336826,
                0.9992614825341174,
            ],
        )

        # The ZED sample you tested appeared to report approximately +g
        # along the upward sensor axis. Confirm this experimentally.
        self.declare_parameter("imu_acceleration_sign", 1.0)

        # Use recent samples instead of one instantaneous accelerometer value.
        self.declare_parameter("imu_window_sec", 0.25)
        self.declare_parameter("imu_max_age_sec", np.inf)

        # Quasi-static validity limits.
        self.declare_parameter("gravity_magnitude", 9.80665)
        self.declare_parameter("accel_norm_tolerance", 0.75)
        self.declare_parameter("max_angular_speed", 0.10)

        # Reject an average when the individual gravity estimates disagree.
        self.declare_parameter("minimum_vector_concentration", 0.98)

        self.base_frame = (
            self.get_parameter("base_frame")
            .get_parameter_value()
            .string_value
        )
        self.grapple_frame = (
            self.get_parameter("grapple_frame")
            .get_parameter_value()
            .string_value
        )
        imu_topic = (
            self.get_parameter("imu_topic")
            .get_parameter_value()
            .string_value
        )
        self.imu_frame_override = (
            self.get_parameter("imu_frame_override")
            .get_parameter_value()
            .string_value
            .strip()
        )

        self.default_exponent = (
            self.get_parameter("default_exponent")
            .get_parameter_value()
            .double_value
        )

        gaze_values = (
            self.get_parameter("gaze_up_base")
            .get_parameter_value()
            .double_array_value
        )
        self.gaze_up_base = normalize(np.array(gaze_values))

        self.imu_acceleration_sign = (
            self.get_parameter("imu_acceleration_sign")
            .get_parameter_value()
            .double_value
        )
        self.imu_window_sec = (
            self.get_parameter("imu_window_sec")
            .get_parameter_value()
            .double_value
        )
        self.imu_max_age_sec = (
            self.get_parameter("imu_max_age_sec")
            .get_parameter_value()
            .double_value
        )
        self.gravity_magnitude = (
            self.get_parameter("gravity_magnitude")
            .get_parameter_value()
            .double_value
        )
        self.accel_norm_tolerance = (
            self.get_parameter("accel_norm_tolerance")
            .get_parameter_value()
            .double_value
        )
        self.max_angular_speed = (
            self.get_parameter("max_angular_speed")
            .get_parameter_value()
            .double_value
        )
        self.minimum_vector_concentration = (
            self.get_parameter("minimum_vector_concentration")
            .get_parameter_value()
            .double_value
        )

        self.callback_group = ReentrantCallbackGroup()

        self.tf_buffer = Buffer(
            cache_time=Duration(seconds=10.0)
        )
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            spin_thread=False,
        )

        self.imu_lock = threading.Lock()

        # Large enough for several seconds at typical IMU rates.
        self.imu_samples: deque[ImuSample] = deque(maxlen=1000)

        self.imu_subscription = self.create_subscription(
            Imu,
            imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group,
        )

        self.service = self.create_service(
            Stability,
            "/get_stability",
            self.service_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"Stability service ready: "
            f"{self.base_frame} -> {self.grapple_frame}"
        )
        if self.imu_frame_override:
            self.get_logger().info(
                "Using IMU TF frame override: "
                f"{self.imu_frame_override}"
            )

    # ------------------------------------------------------------------
    # IMU input
    # ------------------------------------------------------------------

    def imu_callback(self, msg: Imu) -> None:
        """
        Cache structurally valid IMU samples.

        Motion/stationarity checks are performed later when the service is
        called. A moving sample is not malformed; it is simply unsuitable
        for gravity estimation from acceleration. A zero acceleration vector
        also indicates the fused orientation should not be trusted.
        """
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)

        if stamp_ns == 0:
            self.get_logger().warning(
                "Discarding zero-stamped IMU message",
                throttle_duration_sec=2.0,
            )
            return

        acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])

        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])

        orientation = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        ])

        if (
            not np.all(np.isfinite(acceleration))
            or np.linalg.norm(acceleration) < 1e-6
        ):
            self.get_logger().warning(
                "Discarding IMU message with invalid acceleration",
                throttle_duration_sec=2.0,
            )
            return

        if not np.all(np.isfinite(angular_velocity)):
            angular_velocity = None

        if (
            msg.orientation_covariance[0] == -1.0
            or not np.all(np.isfinite(orientation))
            or np.linalg.norm(orientation) < 1e-6
        ):
            orientation = None

        sample = ImuSample(
            stamp=msg.header.stamp,
            frame_id=self.imu_frame_override or msg.header.frame_id,
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
        )

        with self.imu_lock:
            self.imu_samples.append(sample)

    def get_imu_snapshot(self) -> list[ImuSample]:
        with self.imu_lock:
            return list(self.imu_samples)

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------

    def lookup_rotation(
        self,
        target_frame: str,
        source_frame: str,
        query_time: Time,
    ) -> tuple[np.ndarray, object]:
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            query_time,
            timeout=Duration(seconds=0.25),
        )

        q = transform.transform.rotation

        rotation = quaternion_to_rotation_matrix(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        return rotation, transform

    def get_grapple_z(
        self,
        query_time: Time,
    ) -> tuple[np.ndarray, object]:
        rotation, transform = self.lookup_rotation(
            self.base_frame,
            self.grapple_frame,
            query_time,
        )

        z_grapple_base = normalize(rotation[:, 2])

        return z_grapple_base, transform

    # ------------------------------------------------------------------
    # IMU gravity estimation
    # ------------------------------------------------------------------

    def estimate_imu_up_base(
        self,
        samples: list[ImuSample],
    ) -> tuple[np.ndarray, int, str]:
        usable_samples = [
            sample
            for sample in samples
            if has_acceleration_gravity_data(sample)
        ]

        if not usable_samples:
            raise RuntimeError(
                "No IMU messages with valid acceleration and angular "
                "velocity have been received"
            )

        latest_stamp_ns = max(
            stamp_to_nanoseconds(sample.stamp)
            for sample in usable_samples
        )
        latest_sample = max(
            usable_samples,
            key=lambda sample: stamp_to_nanoseconds(sample.stamp),
        )
        latest_sample_time = Time.from_msg(
            latest_sample.stamp
        )
        sample_age = (
            self.get_clock().now() - latest_sample_time
        ).nanoseconds / 1e9

        if sample_age > self.imu_max_age_sec:
            raise RuntimeError(
                "Latest IMU sample is stale: "
                f"age={sample_age:.3f}s > {self.imu_max_age_sec:.3f}s"
            )

        window_ns = int(self.imu_window_sec * 1e9)

        candidates = [
            sample
            for sample in usable_samples
            if latest_stamp_ns - stamp_to_nanoseconds(sample.stamp)
            <= window_ns
        ]

        up_vectors_base = []

        for sample in candidates:
            accel_norm = float(np.linalg.norm(sample.acceleration))
            angular_speed = float(
                np.linalg.norm(sample.angular_velocity)
            )

            # Acceleration magnitude must be close to g.
            if abs(accel_norm - self.gravity_magnitude) \
                    > self.accel_norm_tolerance:
                continue

            # A static inclinometer assumption is questionable while rotating.
            if angular_speed > self.max_angular_speed:
                continue

            z_up_imu = (
                self.imu_acceleration_sign
                * normalize(sample.acceleration)
            )

            try:
                rotation_base_imu, _ = self.lookup_rotation(
                    self.base_frame,
                    sample.frame_id,
                    Time.from_msg(sample.stamp),
                )
            except TransformException:
                continue

            z_up_base = normalize(
                rotation_base_imu @ z_up_imu
            )
            up_vectors_base.append(z_up_base)

        if not up_vectors_base:
            raise RuntimeError(
                "No recent IMU samples passed the quasi-static "
                "acceleration and angular-rate checks"
            )

        mean_vector = np.mean(up_vectors_base, axis=0)

        # For unit vectors, the magnitude of their mean indicates how closely
        # the samples agree. One means perfect agreement.
        concentration = float(np.linalg.norm(mean_vector))

        if concentration < self.minimum_vector_concentration:
            raise RuntimeError(
                "IMU gravity estimates disagree: "
                f"concentration={concentration:.3f}"
            )

        z_up_base = normalize(mean_vector)

        status = (
            f"IMU estimate from {len(up_vectors_base)} samples; "
            f"concentration={concentration:.4f}"
        )

        return z_up_base, len(up_vectors_base), status

    def estimate_imu_orientation_up_base(
        self,
        samples: list[ImuSample],
    ) -> tuple[np.ndarray, int, str]:
        usable_samples = [
            sample
            for sample in samples
            if has_orientation_gravity_data(sample)
        ]

        if not usable_samples:
            raise RuntimeError(
                "No IMU messages with valid fused orientation have been "
                "received"
            )

        latest_stamp_ns = max(
            stamp_to_nanoseconds(sample.stamp)
            for sample in usable_samples
        )
        latest_sample = max(
            usable_samples,
            key=lambda sample: stamp_to_nanoseconds(sample.stamp),
        )
        latest_sample_time = Time.from_msg(
            latest_sample.stamp
        )
        sample_age = (
            self.get_clock().now() - latest_sample_time
        ).nanoseconds / 1e9

        if sample_age > self.imu_max_age_sec:
            raise RuntimeError(
                "Latest IMU sample is stale: "
                f"age={sample_age:.3f}s > {self.imu_max_age_sec:.3f}s"
            )

        window_ns = int(self.imu_window_sec * 1e9)

        candidates = [
            sample
            for sample in usable_samples
            if latest_stamp_ns - stamp_to_nanoseconds(sample.stamp)
            <= window_ns
        ]

        up_vectors_base = []
        z_up_earth = np.array([0.0, 0.0, 1.0])

        for sample in candidates:
            try:
                rotation_earth_imu = quaternion_to_rotation_matrix(
                    sample.orientation[0],
                    sample.orientation[1],
                    sample.orientation[2],
                    sample.orientation[3],
                )
            except ValueError:
                continue

            # The fused orientation maps IMU-frame vectors into the Earth
            # frame. Transpose it to express Earth-up in the IMU frame.
            z_up_imu = rotation_earth_imu.T @ z_up_earth

            try:
                rotation_base_imu, _ = self.lookup_rotation(
                    self.base_frame,
                    sample.frame_id,
                    Time.from_msg(sample.stamp),
                )
            except TransformException:
                continue

            z_up_base = normalize(
                rotation_base_imu @ z_up_imu
            )
            up_vectors_base.append(z_up_base)

        if not up_vectors_base:
            raise RuntimeError(
                "No recent IMU samples had a valid fused orientation and "
                "base-frame transform"
            )

        mean_vector = np.mean(up_vectors_base, axis=0)
        concentration = float(np.linalg.norm(mean_vector))

        if concentration < self.minimum_vector_concentration:
            raise RuntimeError(
                "IMU orientation gravity estimates disagree: "
                f"concentration={concentration:.3f}"
            )

        z_up_base = normalize(mean_vector)

        status = (
            f"IMU orientation estimate from {len(up_vectors_base)} samples; "
            f"concentration={concentration:.4f}"
        )

        return z_up_base, len(up_vectors_base), status

    # ------------------------------------------------------------------
    # Metric calculation
    # ------------------------------------------------------------------

    def calculate_metric(
        self,
        reference_name: str,
        z_grapple_base: np.ndarray,
        z_reference_base: np.ndarray,
        exponent: float,
        samples_used: int = 0,
        status: str = "OK",
    ) -> StabilityMetric:
        result = StabilityMetric()
        result.reference_name = reference_name
        result.reference_z_base = vector3_from_numpy(
            normalize(z_reference_base)
        )
        result.samples_used = samples_used

        try:
            z_grapple_base = normalize(z_grapple_base)
            z_reference_base = normalize(z_reference_base)

            dot = float(
                np.dot(z_grapple_base, z_reference_base)
            )
            dot = float(np.clip(dot, -1.0, 1.0))

            tilt_rad = math.acos(dot)
            reward = max(0.0, dot) ** exponent

            result.valid = True
            result.status = status
            result.dot_product = dot
            result.tilt_rad = tilt_rad
            result.tilt_deg = math.degrees(tilt_rad)
            result.reward = reward

        except Exception as exc:
            result.valid = False
            result.status = str(exc)

        return result

    def make_invalid_metric(
        self,
        reference_name: str,
        status: str,
    ) -> StabilityMetric:
        result = StabilityMetric()
        result.reference_name = reference_name
        result.valid = False
        result.status = status
        return result

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def service_callback(
        self,
        request: Stability.Request,
        response: Stability.Response,
    ) -> Stability.Response:
        if request.mode not in {
            MODE_BASE,
            MODE_GAZE,
            MODE_IMU,
            MODE_IMU_ORIENTATION,
            MODE_ALL,
        }:
            response.success = False
            response.message = (
                f"Unknown stability mode: {request.mode}"
            )
            return response

        exponent = (
            request.exponent
            if request.exponent > 0.0
            else self.default_exponent
        )

        imu_samples = self.get_imu_snapshot()

        # For IMU comparisons, query the grapple transform at the latest
        # relevant IMU timestamp. Otherwise, query the latest available TF.
        imu_query_samples = imu_samples

        if request.mode == MODE_IMU:
            imu_query_samples = [
                sample
                for sample in imu_samples
                if has_acceleration_gravity_data(sample)
            ]
        elif request.mode == MODE_IMU_ORIENTATION:
            imu_query_samples = [
                sample
                for sample in imu_samples
                if has_orientation_gravity_data(sample)
            ]

        if request.mode in {
            MODE_IMU,
            MODE_IMU_ORIENTATION,
            MODE_ALL,
        } and imu_query_samples:
            latest_sample = max(
                imu_query_samples,
                key=lambda sample:
                    stamp_to_nanoseconds(sample.stamp),
            )
            grapple_query_time = Time.from_msg(
                latest_sample.stamp
            )
        else:
            grapple_query_time = Time()

        try:
            z_grapple_base, grapple_transform = (
                self.get_grapple_z(grapple_query_time)
            )
        except TransformException as exc:
            response.success = False
            response.message = (
                "Could not obtain base-to-grapple transform: "
                f"{exc}"
            )
            return response

        response.stamp = grapple_transform.header.stamp
        response.grapple_z_base = vector3_from_numpy(
            z_grapple_base
        )

        metrics = []

        if request.mode in {MODE_BASE, MODE_ALL}:
            metrics.append(
                self.calculate_metric(
                    reference_name="base_link_z",
                    z_grapple_base=z_grapple_base,
                    z_reference_base=np.array([0.0, 0.0, 1.0]),
                    exponent=exponent,
                )
            )

        if request.mode in {MODE_GAZE, MODE_ALL}:
            metrics.append(
                self.calculate_metric(
                    reference_name="free_hanging_gaze",
                    z_grapple_base=z_grapple_base,
                    z_reference_base=self.gaze_up_base,
                    exponent=exponent,
                )
            )

        if request.mode in {MODE_IMU, MODE_ALL}:
            try:
                z_imu_up_base, samples_used, imu_status = (
                    self.estimate_imu_up_base(imu_samples)
                )

                metrics.append(
                    self.calculate_metric(
                        reference_name="zed_imu_acceleration",
                        z_grapple_base=z_grapple_base,
                        z_reference_base=z_imu_up_base,
                        exponent=exponent,
                        samples_used=samples_used,
                        status=imu_status,
                    )
                )

            except Exception as exc:
                metrics.append(
                    self.make_invalid_metric(
                        reference_name="zed_imu_acceleration",
                        status=str(exc),
                    )
                )

        if request.mode in {MODE_IMU_ORIENTATION, MODE_ALL}:
            try:
                z_imu_up_base, samples_used, imu_status = (
                    self.estimate_imu_orientation_up_base(imu_samples)
                )

                metrics.append(
                    self.calculate_metric(
                        reference_name="zed_imu_orientation",
                        z_grapple_base=z_grapple_base,
                        z_reference_base=z_imu_up_base,
                        exponent=exponent,
                        samples_used=samples_used,
                        status=imu_status,
                    )
                )

            except Exception as exc:
                metrics.append(
                    self.make_invalid_metric(
                        reference_name="zed_imu_orientation",
                        status=str(exc),
                    )
                )

        response.metrics = metrics

        valid_count = sum(metric.valid for metric in metrics)

        # For MODE_ALL, base and gaze can still be returned even when IMU is
        # temporarily invalid.
        response.success = valid_count > 0
        response.message = (
            f"Computed {valid_count}/{len(metrics)} requested "
            f"stability metrics using exponent {exponent}"
        )

        return response


def main(args=None) -> None:
    rclpy.init(args=args)

    node = StabilityServiceNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
