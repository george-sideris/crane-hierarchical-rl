from __future__ import annotations

import math
import threading

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from fpi_crane_msgs.action import MeasureStability

from .buffered_stability_estimator import BufferedStabilityEstimator
from .settling_detector import (
    NSEC_PER_SEC,
    DetectorUpdate,
    SettlingConfig,
    SettlingDetector,
    SettlingResult,
)
from .stability_service_node import stamp_to_nanoseconds


def ns_to_time_msg(stamp_ns: int) -> TimeMsg:
    msg = TimeMsg()
    msg.sec = int(stamp_ns // NSEC_PER_SEC)
    msg.nanosec = int(stamp_ns % NSEC_PER_SEC)
    return msg


def vector3_from_numpy(vector: np.ndarray) -> Vector3:
    return Vector3(
        x=float(vector[0]),
        y=float(vector[1]),
        z=float(vector[2]),
    )


class MeasureStabilityActionNode(Node):
    def __init__(self) -> None:
        super().__init__("measure_stability_action")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("grapple_frame", "grapplecarrier")
        self.declare_parameter("imu_topic", "/zed_0/zed_node/imu/data")
        self.declare_parameter("imu_frame_override", "")
        # Up-vector source. "base": use base_frame +z (pure encoder/geometry tilt,
        # assumes the crane base is plumb; IMU messages only pace the sampling).
        # "imu": rotate true gravity from the fused IMU orientation into base_frame
        # (needs the imu_link TF and quiet boom accelerations).
        self.declare_parameter("gravity_source", "base")
        # Sampling rate in base mode (timer-paced TF sampling; imu mode paces
        # on IMU arrival instead).
        self.declare_parameter("base_sample_rate_hz", 20.0)
        self.declare_parameter("buffer_duration_sec", 30.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.05)
        self.declare_parameter("default_timeout_sec", 20.0)
        self.declare_parameter("required_window_sec", 1.0)
        self.declare_parameter("required_dwell_sec", 0.5)
        self.declare_parameter("min_samples", 20)
        self.declare_parameter("max_sample_gap_sec", 0.15)
        self.declare_parameter("stability_variance_threshold_rad2", 1.0e-4)
        self.declare_parameter("gravity_variance_threshold_rad2", 1.0e-5)
        self.declare_parameter("grapple_variance_threshold_rad2", 1.0e-4)
        self.declare_parameter("reward_exponent", 4.0)
        self.declare_parameter("feedback_period_sec", 0.2)
        self.declare_parameter("log_degrees", True)

        self.callback_group = ReentrantCallbackGroup()
        self._active_goal_lock = threading.Lock()
        self._active_goal = None
        self._goal_reserved = False

        reward_exponent = float(self.get_parameter("reward_exponent").value)
        self.estimator = BufferedStabilityEstimator(
            self,
            base_frame=self.get_parameter("base_frame").value,
            grapple_frame=self.get_parameter("grapple_frame").value,
            imu_topic=self.get_parameter("imu_topic").value,
            imu_frame_override=self.get_parameter("imu_frame_override").value,
            buffer_duration_sec=float(self.get_parameter("buffer_duration_sec").value),
            tf_lookup_timeout_sec=float(self.get_parameter("tf_lookup_timeout_sec").value),
            reward_exponent=reward_exponent,
            gravity_source=str(self.get_parameter("gravity_source").value),
            base_sample_rate_hz=float(self.get_parameter("base_sample_rate_hz").value),
            callback_group=self.callback_group,
        )

        self.action_server = ActionServer(
            self,
            MeasureStability,
            "measure_stability",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            "MeasureStability action ready: measure_stability "
            f"({self.get_parameter('base_frame').value} -> "
            f"{self.get_parameter('grapple_frame').value})"
        )

    def goal_callback(self, goal_request):
        with self._active_goal_lock:
            if self._goal_reserved or self._active_goal is not None:
                self.get_logger().warn("Rejecting MeasureStability goal: another goal is active")
                return GoalResponse.REJECT
        with self._active_goal_lock:
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        with self._active_goal_lock:
            self._active_goal = goal_handle
            self._goal_reserved = False
        try:
            result = self._execute_goal(goal_handle)
            return self._to_action_result(result)
        finally:
            with self._active_goal_lock:
                if self._active_goal is goal_handle:
                    self._active_goal = None
                self._goal_reserved = False

    def _execute_goal(self, goal_handle) -> SettlingResult:
        request = goal_handle.request
        config = self._config_from_goal(request)
        measurement_start_ns = stamp_to_nanoseconds(request.measurement_start)
        timeout_ns = int(round(request.timeout_sec * NSEC_PER_SEC))
        feedback_period_ns = int(round(
            float(self.get_parameter("feedback_period_sec").value)
            * NSEC_PER_SEC
        ))
        detector = SettlingDetector(config, measurement_start_ns)

        processed_stamps: set[int] = set()
        last_feedback_ns = 0
        last_window_status = "no samples processed"
        wait_event = threading.Event()

        self.get_logger().info(
            "MeasureStability goal accepted: "
            f"window={config.required_window_sec:.3f}s dwell={config.required_dwell_sec:.3f}s "
            f"timeout={request.timeout_sec:.3f}s"
        )

        while rclpy.ok():
            now_ns = self.get_clock().now().nanoseconds
            if goal_handle.is_cancel_requested:
                result = detector.make_failure_result("cancelled", now_ns)
                goal_handle.canceled()
                self.get_logger().warn("MeasureStability cancelled")
                return result

            if now_ns - measurement_start_ns > timeout_ns:
                result = detector.make_failure_result("timeout", now_ns)
                goal_handle.abort()
                self.get_logger().warn(
                    "MeasureStability timeout (did not settle): "
                    f"samples={result.measurement_sample_count} "
                    f"(window {result.accepted_window_sample_count}), "
                    f"best_effort_reward={result.settle_reward:.4f}, "
                    f"best_effort_theta={math.degrees(result.theta_settle_rad):.3f}deg, "
                    f"stability_var={result.final_stability_variance_rad2:.2e}rad2, "
                    f"gravity_var={result.final_gravity_variance_rad2:.2e}rad2, "
                    f"grapple_var={result.final_grapple_variance_rad2:.2e}rad2, "
                    f"last_window_status='{last_window_status}', "
                    f"last_error={self.estimator.last_error()}"
                )
                return result

            latest_update = None
            for sample in self.estimator.snapshot_since(measurement_start_ns):
                if sample.stamp_ns in processed_stamps:
                    continue
                processed_stamps.add(sample.stamp_ns)
                latest_update = detector.add_sample(
                    sample.stamp_ns,
                    sample.z_grapple_base,
                    sample.z_up_base,
                )
                last_window_status = latest_update.status
                if latest_update.accepted:
                    result = detector.make_result(sample.stamp_ns)
                    goal_handle.succeed()
                    self._log_success(result)
                    return result

            if latest_update is not None and (
                latest_update.sample is not None
                and latest_update.sample.stamp_ns - last_feedback_ns >= feedback_period_ns
            ):
                goal_handle.publish_feedback(self._to_feedback(latest_update))
                last_feedback_ns = latest_update.sample.stamp_ns

            wait_event.wait(0.02)

        result = detector.make_failure_result("shutdown", self.get_clock().now().nanoseconds)
        goal_handle.abort()
        return result

    def _config_from_goal(self, request) -> SettlingConfig:
        return SettlingConfig(
            required_window_sec=float(request.required_window_sec),
            required_dwell_sec=float(request.required_dwell_sec),
            min_samples=int(request.min_samples),
            max_sample_gap_sec=float(request.max_sample_gap_sec),
            stability_variance_threshold_rad2=float(
                request.stability_variance_threshold_rad2),
            gravity_variance_threshold_rad2=float(
                request.gravity_variance_threshold_rad2),
            grapple_variance_threshold_rad2=float(
                request.grapple_variance_threshold_rad2),
            reward_exponent=float(request.reward_exponent),
        )

    def _to_feedback(self, update: DetectorUpdate):
        msg = MeasureStability.Feedback()
        stamp_ns = update.sample.stamp_ns if update.sample is not None else 0
        msg.stamp = ns_to_time_msg(stamp_ns)
        msg.status = update.status
        msg.elapsed_sec = float(update.elapsed_sec)
        msg.current_theta_rad = (
            float(update.sample.theta_rad) if update.sample is not None else math.nan
        )
        msg.theta_max_rad = float(update.theta_max_rad)
        msg.theta_rms_rad = float(update.theta_rms_rad)
        msg.current_stability_variance_rad2 = float(update.window.stability_variance_rad2)
        msg.current_gravity_variance_rad2 = float(update.window.gravity_variance_rad2)
        msg.current_grapple_variance_rad2 = float(update.window.grapple_variance_rad2)
        msg.candidate_dwell_sec = float(update.candidate_dwell_sec)
        msg.window_sample_count = int(update.window.sample_count)
        msg.window_start = ns_to_time_msg(update.window.start_ns)
        msg.window_end = ns_to_time_msg(update.window.end_ns)
        return msg

    def _to_action_result(self, result: SettlingResult):
        msg = MeasureStability.Result()
        msg.success = bool(result.success)
        msg.failure_reason = result.failure_reason
        msg.measurement_start = ns_to_time_msg(result.measurement_start_ns)
        msg.measurement_end = ns_to_time_msg(result.measurement_end_ns)
        msg.t_settle_sec = float(result.t_settle_sec)
        msg.theta_max_rad = float(result.theta_max_rad)
        msg.theta_rms_rad = float(result.theta_rms_rad)
        msg.theta_settle_rad = float(result.theta_settle_rad)
        msg.theta_from_mean_vectors_rad = float(result.theta_from_mean_vectors_rad)
        msg.settle_z_grapple_base = vector3_from_numpy(result.settle_z_grapple_base)
        msg.settle_z_up_base = vector3_from_numpy(result.settle_z_up_base)
        msg.settle_dot_product = float(result.settle_dot_product)
        msg.settle_reward = float(result.settle_reward)
        msg.reward_exponent = float(result.reward_exponent)
        msg.accepted_window_start = ns_to_time_msg(result.accepted_window_start_ns)
        msg.accepted_window_end = ns_to_time_msg(result.accepted_window_end_ns)
        msg.measurement_sample_count = int(result.measurement_sample_count)
        msg.accepted_window_sample_count = int(result.accepted_window_sample_count)
        msg.final_stability_variance_rad2 = float(result.final_stability_variance_rad2)
        msg.final_gravity_variance_rad2 = float(result.final_gravity_variance_rad2)
        msg.final_grapple_variance_rad2 = float(result.final_grapple_variance_rad2)
        msg.sample_offsets_sec = [float(v) for v in result.sample_offsets_sec]
        msg.sample_theta_rad = [float(v) for v in result.sample_theta_rad]
        return msg

    def _log_success(self, result: SettlingResult) -> None:
        if bool(self.get_parameter("log_degrees").value):
            self.get_logger().info(
                "MeasureStability succeeded: "
                f"t_settle={result.t_settle_sec:.3f}s "
                f"theta_max={math.degrees(result.theta_max_rad):.3f}deg "
                f"theta_rms={math.degrees(result.theta_rms_rad):.3f}deg "
                f"theta_settle={math.degrees(result.theta_settle_rad):.3f}deg"
            )
        else:
            self.get_logger().info(
                "MeasureStability succeeded: "
                f"t_settle={result.t_settle_sec:.3f}s "
                f"theta_max={result.theta_max_rad:.6f}rad "
                f"theta_rms={result.theta_rms_rad:.6f}rad "
                f"theta_settle={result.theta_settle_rad:.6f}rad"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MeasureStabilityActionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        # try_shutdown: plain shutdown() raises if the SIGINT handler already
        # shut the context down (seen on Ctrl-C)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
