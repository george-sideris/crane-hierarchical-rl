from __future__ import annotations

import argparse
import math

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time
from rclpy.action import ActionClient
from rclpy.node import Node

from fpi_crane_msgs.action import MeasureStability


NSEC_PER_SEC = 1_000_000_000


def ns_to_time_msg(stamp_ns: int) -> Time:
    msg = Time()
    msg.sec = int(stamp_ns // NSEC_PER_SEC)
    msg.nanosec = int(stamp_ns % NSEC_PER_SEC)
    return msg


def time_msg_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * NSEC_PER_SEC + int(stamp.nanosec)


class MeasureStabilityClient(Node):
    def __init__(self, args) -> None:
        super().__init__("measure_stability_client")
        self.args = args
        self.client = ActionClient(self, MeasureStability, args.action_name)

    def send_goal(self) -> bool:
        if not self.client.wait_for_server(timeout_sec=self.args.server_timeout_sec):
            self.get_logger().error(
                f"Action server '{self.args.action_name}' not available"
            )
            return False

        goal = MeasureStability.Goal()
        goal.measurement_start = self._measurement_start()
        goal.timeout_sec = self.args.timeout_sec
        goal.required_window_sec = self.args.required_window_sec
        goal.required_dwell_sec = self.args.required_dwell_sec
        goal.min_samples = self.args.min_samples
        goal.max_sample_gap_sec = self.args.max_sample_gap_sec
        goal.stability_variance_threshold_rad2 = (
            self.args.stability_variance_threshold_rad2
        )
        goal.gravity_variance_threshold_rad2 = (
            self.args.gravity_variance_threshold_rad2
        )
        goal.grapple_variance_threshold_rad2 = (
            self.args.grapple_variance_threshold_rad2
        )
        goal.reward_exponent = self.args.reward_exponent

        start_ns = time_msg_to_ns(goal.measurement_start)
        self.get_logger().info(
            "Sending MeasureStability goal: "
            f"measurement_start={start_ns} ns, "
            f"timeout={goal.timeout_sec:.3f}s, "
            f"window={goal.required_window_sec:.3f}s, "
            f"dwell={goal.required_dwell_sec:.3f}s"
        )

        send_future = self.client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error("Goal send failed")
            return False
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            self.get_logger().error("Result failed")
            return False

        self.print_result(wrapped.status, wrapped.result)
        return wrapped.status == GoalStatus.STATUS_SUCCEEDED and wrapped.result.success

    def _measurement_start(self) -> Time:
        if self.args.start_ns is not None:
            return ns_to_time_msg(self.args.start_ns)
        if self.args.start_sec is not None:
            nanosec = self.args.start_nanosec or 0
            return ns_to_time_msg(self.args.start_sec * NSEC_PER_SEC + nanosec)
        return self.get_clock().now().to_msg()

    def feedback_callback(self, feedback_msg) -> None:
        feedback = feedback_msg.feedback
        self.get_logger().info(
            "feedback: "
            f"status={feedback.status}, "
            f"elapsed={feedback.elapsed_sec:.3f}s, "
            f"theta={math.degrees(feedback.current_theta_rad):.3f}deg, "
            f"theta_max={math.degrees(feedback.theta_max_rad):.3f}deg, "
            f"theta_rms={math.degrees(feedback.theta_rms_rad):.3f}deg, "
            f"dwell={feedback.candidate_dwell_sec:.3f}s, "
            f"window_samples={feedback.window_sample_count}"
        )

    def print_result(self, status: int, result) -> None:
        status_name = {
            GoalStatus.STATUS_UNKNOWN: "unknown",
            GoalStatus.STATUS_ACCEPTED: "accepted",
            GoalStatus.STATUS_EXECUTING: "executing",
            GoalStatus.STATUS_CANCELING: "canceling",
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "canceled",
            GoalStatus.STATUS_ABORTED: "aborted",
        }.get(status, f"status_{status}")

        if not result.success:
            self.get_logger().error(
                f"MeasureStability failed: status={status_name}, "
                f"reason={result.failure_reason}, "
                f"samples={result.measurement_sample_count}"
            )
            return

        self.get_logger().info(
            "MeasureStability result: "
            f"status={status_name}, "
            f"t_settle={result.t_settle_sec:.3f}s, "
            f"theta_max={math.degrees(result.theta_max_rad):.3f}deg, "
            f"theta_rms={math.degrees(result.theta_rms_rad):.3f}deg, "
            f"theta_settle={math.degrees(result.theta_settle_rad):.3f}deg, "
            f"theta_mean_vectors="
            f"{math.degrees(result.theta_from_mean_vectors_rad):.3f}deg, "
            f"dot={result.settle_dot_product:.6f}, "
            f"reward={result.settle_reward:.6f}, "
            f"samples={result.measurement_sample_count}, "
            f"accepted_window_samples={result.accepted_window_sample_count}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Send a MeasureStability action goal without running the policy node."
    )
    parser.add_argument("--action-name", default="measure_stability")
    parser.add_argument("--server-timeout-sec", type=float, default=5.0)
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--start-sec", type=int, default=None)
    parser.add_argument("--start-nanosec", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--required-window-sec", type=float, default=1.0)
    parser.add_argument("--required-dwell-sec", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-sample-gap-sec", type=float, default=0.15)
    parser.add_argument(
        "--stability-variance-threshold-rad2",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--gravity-variance-threshold-rad2",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--grapple-variance-threshold-rad2",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument("--reward-exponent", type=float, default=4.0)
    return parser.parse_known_args(argv)


def main(args=None) -> None:
    parsed, ros_args = parse_args(args)
    rclpy.init(args=ros_args)
    node = MeasureStabilityClient(parsed)
    try:
        ok = node.send_goal()
        if not ok:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
