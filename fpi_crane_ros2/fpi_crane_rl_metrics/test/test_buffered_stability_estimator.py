import threading
from collections import deque

import numpy as np
from sensor_msgs.msg import Imu

from fpi_crane_rl_metrics.buffered_stability_estimator import (
    BufferedStabilityEstimator,
)


def make_estimator():
    estimator = BufferedStabilityEstimator.__new__(BufferedStabilityEstimator)
    estimator._lock = threading.Lock()
    estimator._samples = deque()
    estimator._last_error = ""
    estimator.buffer_duration_ns = 10_000_000_000
    estimator.reward_exponent = 4.0
    estimator.imu_frame_override = ""
    estimator.gravity_source = "imu"
    estimator._orientation_up_base = lambda *args: np.array([0.0, 0.0, 1.0])
    estimator._grapple_z_base = lambda *args: np.array([0.0, 0.0, 1.0])
    return estimator


def imu_msg():
    msg = Imu()
    msg.header.stamp.sec = 1
    msg.header.frame_id = "zed_0_imu_link"
    msg.orientation.w = 1.0
    return msg


def test_rejects_imu_with_zero_acceleration_and_zero_angular_velocity():
    estimator = make_estimator()
    msg = imu_msg()

    estimator.imu_callback(msg)

    assert estimator.snapshot_since(0) == []
    assert estimator.last_error() == (
        "IMU message has zero linear acceleration and angular velocity"
    )


def test_does_not_reject_when_only_linear_acceleration_is_zero():
    estimator = make_estimator()
    msg = imu_msg()
    msg.angular_velocity.z = 0.1

    estimator.imu_callback(msg)

    assert len(estimator.snapshot_since(0)) == 1
    assert estimator.last_error() == ""
