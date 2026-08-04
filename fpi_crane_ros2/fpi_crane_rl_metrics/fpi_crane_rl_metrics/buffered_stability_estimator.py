from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformException, TransformListener

from .settling_detector import clipped_dot, normalize
from .stability_service_node import (
    quaternion_to_rotation_matrix,
    stamp_to_nanoseconds,
)


@dataclass(frozen=True)
class BufferedStabilitySample:
    stamp_ns: int
    z_grapple_base: np.ndarray
    z_up_base: np.ndarray
    theta_rad: float
    dot_product: float
    reward: float

    def to_detector_sample(self) -> tuple[int, np.ndarray, np.ndarray]:
        return self.stamp_ns, self.z_grapple_base, self.z_up_base


class BufferedStabilityEstimator:
    """Continuously buffers synchronized grapple/gravity stability samples."""

    def __init__(
        self,
        node,
        *,
        base_frame: str,
        grapple_frame: str,
        imu_topic: str,
        imu_frame_override: str,
        buffer_duration_sec: float,
        tf_lookup_timeout_sec: float,
        reward_exponent: float,
        callback_group,
        gravity_source: str = "imu",
        base_sample_rate_hz: float = 20.0,
    ) -> None:
        self.node = node
        self.base_frame = base_frame
        self.grapple_frame = grapple_frame
        self.imu_frame_override = imu_frame_override.strip()
        self.gravity_source = gravity_source.strip().lower()
        if self.gravity_source not in ("imu", "base"):
            node.get_logger().warn(
                f"gravity_source must be 'imu' or 'base', got '{gravity_source}'; "
                "using 'imu'")
            self.gravity_source = "imu"
        self.buffer_duration_ns = int(round(buffer_duration_sec * 1_000_000_000))
        self.tf_lookup_timeout = Duration(seconds=tf_lookup_timeout_sec)
        self.reward_exponent = float(reward_exponent)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=max(10.0, buffer_duration_sec)))
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self.node,
            spin_thread=False,
        )

        self._lock = threading.Lock()
        self._samples: deque[BufferedStabilitySample] = deque()
        self._last_error = ""

        if self.gravity_source == "base":
            # Base mode needs nothing from the IMU: up is base_frame +z and the
            # grapple axis comes from TF. Sample on a local timer instead of on
            # IMU arrival; message pacing proved unusable in the field (bursty
            # best-effort transport -> excessive receive-time gaps, and zero or
            # clock-skewed header stamps).
            self.subscription = None
            self.timer = self.node.create_timer(
                1.0 / max(base_sample_rate_hz, 1e-3),
                self.timer_callback,
                callback_group=callback_group,
            )
        else:
            self.timer = None
            self.subscription = self.node.create_subscription(
                Imu,
                imu_topic,
                self.imu_callback,
                qos_profile_sensor_data,
                callback_group=callback_group,
            )

    def snapshot_since(self, start_ns: int) -> list[BufferedStabilitySample]:
        with self._lock:
            return [sample for sample in self._samples if sample.stamp_ns >= start_ns]

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def timer_callback(self) -> None:
        # Base mode: base_frame +z as "up", pure encoder/geometry tilt relative
        # to the (assumed plumb) crane base, sampled from the latest TF.
        stamp_ns = self.node.get_clock().now().nanoseconds
        z_up_base = np.array([0.0, 0.0, 1.0])
        try:
            z_grapple_base = self._grapple_z_base(Time())
        except Exception as exc:
            self._set_error(str(exc))
            return
        self._append_sample(stamp_ns, z_grapple_base, z_up_base)

    def imu_callback(self, msg: Imu) -> None:
        stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        if stamp_ns == 0:
            self._set_error("zero-stamped IMU message")
            return

        linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ], dtype=float)
        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ], dtype=float)
        if (
            np.all(np.isfinite(linear_acceleration))
            and np.all(np.isfinite(angular_velocity))
            and np.allclose(linear_acceleration, 0.0)
            and np.allclose(angular_velocity, 0.0)
        ):
            self._set_error(
                "IMU message has zero linear acceleration and angular velocity"
            )
            return

        query_time = Time.from_msg(msg.header.stamp)
        try:
            orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ], dtype=float)
            if (
                msg.orientation_covariance[0] == -1.0
                or not np.all(np.isfinite(orientation))
                or np.linalg.norm(orientation) < 1e-6
            ):
                self._set_error("IMU message has no valid fused orientation")
                return

            imu_frame = self.imu_frame_override or msg.header.frame_id
            if not imu_frame:
                self._set_error("IMU frame_id is empty")
                return

            z_up_base = self._orientation_up_base(
                orientation, imu_frame, query_time)
            z_grapple_base = self._grapple_z_base(query_time)
        except Exception as exc:
            self._set_error(str(exc))
            return

        self._append_sample(stamp_ns, z_grapple_base, z_up_base)

    def _append_sample(
        self,
        stamp_ns: int,
        z_grapple_base: np.ndarray,
        z_up_base: np.ndarray,
    ) -> None:
        dot = clipped_dot(z_grapple_base, z_up_base)
        theta = float(np.arccos(dot))
        reward = max(0.0, dot) ** self.reward_exponent

        with self._lock:
            if self._samples and stamp_ns <= self._samples[-1].stamp_ns:
                # Callbacks in a reentrant group can interleave; the settling
                # detector requires strictly increasing stamps.
                stamp_ns = self._samples[-1].stamp_ns + 1
            sample = BufferedStabilitySample(
                stamp_ns=stamp_ns,
                z_grapple_base=z_grapple_base,
                z_up_base=z_up_base,
                theta_rad=theta,
                dot_product=dot,
                reward=reward,
            )
            self._samples.append(sample)
            cutoff = stamp_ns - self.buffer_duration_ns
            while self._samples and self._samples[0].stamp_ns < cutoff:
                self._samples.popleft()
            self._last_error = ""

    def _orientation_up_base(
        self,
        orientation_xyzw: np.ndarray,
        imu_frame: str,
        query_time: Time,
    ) -> np.ndarray:
        rotation_earth_imu = quaternion_to_rotation_matrix(
            orientation_xyzw[0],
            orientation_xyzw[1],
            orientation_xyzw[2],
            orientation_xyzw[3],
        )
        z_up_earth = np.array([0.0, 0.0, 1.0])
        z_up_imu = rotation_earth_imu.T @ z_up_earth

        rotation_base_imu = self._lookup_rotation(
            self.base_frame,
            imu_frame,
            query_time,
        )
        return normalize(rotation_base_imu @ z_up_imu)

    def _grapple_z_base(self, query_time: Time) -> np.ndarray:
        rotation_base_grapple = self._lookup_rotation(
            self.base_frame,
            self.grapple_frame,
            query_time,
        )
        return normalize(rotation_base_grapple[:, 2])

    def _lookup_rotation(
        self,
        target_frame: str,
        source_frame: str,
        query_time: Time,
    ) -> np.ndarray:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                query_time,
                timeout=self.tf_lookup_timeout,
            )
        except TransformException as exc:
            raise RuntimeError(
                f"tf {target_frame}<-{source_frame} unavailable: {exc}"
            ) from exc

        q = transform.transform.rotation
        return quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
