from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NSEC_PER_SEC = 1_000_000_000


@dataclass(frozen=True)
class SettlingConfig:
    required_window_sec: float
    required_dwell_sec: float
    min_samples: int
    max_sample_gap_sec: float
    stability_variance_threshold_rad2: float
    gravity_variance_threshold_rad2: float
    grapple_variance_threshold_rad2: float
    reward_exponent: float


@dataclass(frozen=True)
class StabilitySample:
    stamp_ns: int
    z_grapple_base: np.ndarray
    z_up_base: np.ndarray
    theta_rad: float
    dot_product: float
    reward: float


@dataclass
class WindowMetrics:
    valid: bool
    status: str
    start_ns: int = 0
    end_ns: int = 0
    sample_count: int = 0
    stability_variance_rad2: float = math.nan
    gravity_variance_rad2: float = math.nan
    grapple_variance_rad2: float = math.nan
    theta_mean_rad: float = math.nan
    mean_z_grapple_base: Optional[np.ndarray] = None
    mean_z_up_base: Optional[np.ndarray] = None


@dataclass
class DetectorUpdate:
    accepted: bool
    status: str
    sample: Optional[StabilitySample]
    window: WindowMetrics
    elapsed_sec: float
    candidate_dwell_sec: float
    theta_max_rad: float
    theta_rms_rad: float
    measurement_sample_count: int


@dataclass
class SettlingResult:
    success: bool
    failure_reason: str
    measurement_start_ns: int
    measurement_end_ns: int
    t_settle_sec: float
    theta_max_rad: float
    theta_rms_rad: float
    theta_settle_rad: float
    theta_from_mean_vectors_rad: float
    settle_z_grapple_base: np.ndarray
    settle_z_up_base: np.ndarray
    settle_dot_product: float
    settle_reward: float
    reward_exponent: float
    accepted_window_start_ns: int
    accepted_window_end_ns: int
    measurement_sample_count: int
    accepted_window_sample_count: int
    final_stability_variance_rad2: float
    final_gravity_variance_rad2: float
    final_grapple_variance_rad2: float
    # Full measurement time series (offsets relative to measurement_start)
    # for plotting/diagnostics; empty when no samples arrived.
    sample_offsets_sec: list = field(default_factory=list)
    sample_theta_rad: list = field(default_factory=list)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Cannot normalize zero or non-finite vector")
    return vector / norm


def clipped_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(float(np.dot(a, b)), -1.0, 1.0))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(clipped_dot(a, b))


class SettlingDetector:
    """Pure settling detector for synchronized grapple/gravity unit vectors.

    The whole-measurement accumulator is never truncated and is used for
    theta_max_rad and theta_rms_rad. The rolling deque is only used for
    settled-window decisions and accepted-window result fields.
    """

    def __init__(self, config: SettlingConfig, measurement_start_ns: int) -> None:
        self.config = config
        self.measurement_start_ns = int(measurement_start_ns)
        self.window_ns = int(round(config.required_window_sec * NSEC_PER_SEC))
        self.max_gap_ns = int(round(config.max_sample_gap_sec * NSEC_PER_SEC))
        self.dwell_ns = int(round(config.required_dwell_sec * NSEC_PER_SEC))

        self._window_samples: deque[StabilitySample] = deque()
        self._theta_series: list = []   # (stamp_ns, theta) over the whole measurement
        self._theta_max_rad = -math.inf
        self._theta_sq_sum = 0.0
        self._sample_count = 0
        self._last_stamp_ns: Optional[int] = None

        self._candidate_settled_start_ns: Optional[int] = None
        self._candidate_first_pass_ns: Optional[int] = None
        self._accepted_window: Optional[WindowMetrics] = None

    @property
    def measurement_sample_count(self) -> int:
        return self._sample_count

    @property
    def theta_max_rad(self) -> float:
        return self._theta_max_rad if self._sample_count else math.nan

    @property
    def theta_rms_rad(self) -> float:
        if self._sample_count == 0:
            return math.nan
        return math.sqrt(self._theta_sq_sum / self._sample_count)

    def add_sample(
        self,
        stamp_ns: int,
        z_grapple_base: np.ndarray,
        z_up_base: np.ndarray,
    ) -> DetectorUpdate:
        stamp_ns = int(stamp_ns)
        if stamp_ns < self.measurement_start_ns:
            return self._invalid_update("sample before measurement_start", stamp_ns)

        try:
            z_grapple = normalize(z_grapple_base)
            z_up = normalize(z_up_base)
            dot = clipped_dot(z_grapple, z_up)
            theta = math.acos(dot)
        except Exception as exc:
            return self._invalid_update(f"invalid sample: {exc}", stamp_ns)

        if not np.isfinite(theta):
            return self._invalid_update("invalid sample: non-finite theta", stamp_ns)

        reward = max(0.0, dot) ** self.config.reward_exponent
        sample = StabilitySample(
            stamp_ns=stamp_ns,
            z_grapple_base=z_grapple,
            z_up_base=z_up,
            theta_rad=theta,
            dot_product=dot,
            reward=reward,
        )

        self._sample_count += 1
        self._theta_sq_sum += theta * theta
        self._theta_max_rad = max(self._theta_max_rad, theta)
        self._last_stamp_ns = stamp_ns
        self._theta_series.append((stamp_ns, theta))

        self._window_samples.append(sample)
        cutoff_ns = stamp_ns - self.window_ns
        # Keep the newest sample at or before the cutoff so the retained span can
        # reach required_window_sec. Trimming everything older than the cutoff
        # leaves a span strictly shorter than the window, and the span check in
        # _compute_window then only passes when a stamp lands exactly on the
        # cutoff (true in fixed-step sim, never on hardware).
        while (
            len(self._window_samples) >= 2
            and self._window_samples[1].stamp_ns <= cutoff_ns
        ):
            self._window_samples.popleft()

        window = self._compute_window()
        accepted = False
        status = window.status
        candidate_dwell_sec = 0.0

        if window.valid:
            if self._candidate_settled_start_ns is None:
                self._candidate_settled_start_ns = window.start_ns
                self._candidate_first_pass_ns = window.end_ns

            candidate_dwell_sec = max(
                0.0,
                (window.end_ns - (self._candidate_first_pass_ns or window.end_ns))
                / NSEC_PER_SEC,
            )
            if (
                self._candidate_first_pass_ns is not None
                and window.end_ns - self._candidate_first_pass_ns >= self.dwell_ns
            ):
                self._accepted_window = window
                accepted = True
                status = "settled"
            else:
                status = "candidate_settled"
        else:
            self._candidate_settled_start_ns = None
            self._candidate_first_pass_ns = None

        return DetectorUpdate(
            accepted=accepted,
            status=status,
            sample=sample,
            window=window,
            elapsed_sec=(stamp_ns - self.measurement_start_ns) / NSEC_PER_SEC,
            candidate_dwell_sec=candidate_dwell_sec,
            theta_max_rad=self.theta_max_rad,
            theta_rms_rad=self.theta_rms_rad,
            measurement_sample_count=self._sample_count,
        )

    def _invalid_update(self, status: str, stamp_ns: int) -> DetectorUpdate:
        return DetectorUpdate(
            accepted=False,
            status=status,
            sample=None,
            window=(
                self._compute_window()
                if self._window_samples
                else WindowMetrics(False, status)
            ),
            elapsed_sec=max(0.0, (stamp_ns - self.measurement_start_ns) / NSEC_PER_SEC),
            candidate_dwell_sec=0.0,
            theta_max_rad=self.theta_max_rad,
            theta_rms_rad=self.theta_rms_rad,
            measurement_sample_count=self._sample_count,
        )

    def _compute_window(self) -> WindowMetrics:
        samples = list(self._window_samples)
        count = len(samples)
        if count == 0:
            return WindowMetrics(False, "no valid samples")

        start_ns = samples[0].stamp_ns
        end_ns = samples[-1].stamp_ns
        span_ns = end_ns - start_ns

        base = WindowMetrics(
            valid=False,
            status="",
            start_ns=start_ns,
            end_ns=end_ns,
            sample_count=count,
        )

        if span_ns < self.window_ns:
            base.status = "window too short"
            return base
        if count < int(self.config.min_samples):
            base.status = "insufficient samples"
            return base

        stamps = [sample.stamp_ns for sample in samples]
        gaps = [b - a for a, b in zip(stamps[:-1], stamps[1:])]
        if any(gap <= 0 for gap in gaps):
            base.status = "non-increasing sample timestamps"
            return base
        if any(gap > self.max_gap_ns for gap in gaps):
            base.status = "excessive timestamp gap"
            return base

        theta = np.array([sample.theta_rad for sample in samples], dtype=float)
        stability_mean = float(np.mean(theta))
        stability_variance = float(np.mean((theta - stability_mean) ** 2))

        z_up = np.array([sample.z_up_base for sample in samples], dtype=float)
        z_grapple = np.array([sample.z_grapple_base for sample in samples], dtype=float)

        try:
            mean_z_up = normalize(np.mean(z_up, axis=0))
            mean_z_grapple = normalize(np.mean(z_grapple, axis=0))
        except ValueError as exc:
            base.status = str(exc)
            return base

        gravity_deviation = np.array(
            [angle_between(v, mean_z_up) for v in z_up],
            dtype=float,
        )
        grapple_deviation = np.array(
            [angle_between(v, mean_z_grapple) for v in z_grapple],
            dtype=float,
        )
        gravity_variance = float(np.mean(gravity_deviation ** 2))
        grapple_variance = float(np.mean(grapple_deviation ** 2))

        base.theta_mean_rad = stability_mean
        base.stability_variance_rad2 = stability_variance
        base.gravity_variance_rad2 = gravity_variance
        base.grapple_variance_rad2 = grapple_variance
        base.mean_z_up_base = mean_z_up
        base.mean_z_grapple_base = mean_z_grapple

        checks = [
            (
                stability_variance
                <= self.config.stability_variance_threshold_rad2,
                "stability variance above threshold",
            ),
            (
                gravity_variance
                <= self.config.gravity_variance_threshold_rad2,
                "gravity variance above threshold",
            ),
            (
                grapple_variance
                <= self.config.grapple_variance_threshold_rad2,
                "grapple variance above threshold",
            ),
        ]
        for passed, reason in checks:
            if not passed:
                base.status = reason
                return base

        base.valid = True
        base.status = "window settled"
        return base

    def make_result(self, measurement_end_ns: Optional[int] = None) -> SettlingResult:
        if self._accepted_window is None:
            raise RuntimeError("Cannot make a successful result before settling")
        window = self._accepted_window
        if window.mean_z_grapple_base is None or window.mean_z_up_base is None:
            raise RuntimeError("Accepted window is missing representative vectors")

        measurement_end_ns = (
            int(measurement_end_ns)
            if measurement_end_ns is not None
            else int(window.end_ns)
        )
        settle_dot = clipped_dot(window.mean_z_grapple_base, window.mean_z_up_base)
        theta_from_mean = math.acos(settle_dot)
        settle_reward = max(0.0, settle_dot) ** self.config.reward_exponent
        settled_start_ns = (
            self._candidate_settled_start_ns
            if self._candidate_settled_start_ns is not None
            else window.start_ns
        )
        t_settle_sec = (settled_start_ns - self.measurement_start_ns) / NSEC_PER_SEC

        return SettlingResult(
            success=True,
            failure_reason="",
            measurement_start_ns=self.measurement_start_ns,
            measurement_end_ns=measurement_end_ns,
            t_settle_sec=t_settle_sec,
            theta_max_rad=self.theta_max_rad,
            theta_rms_rad=self.theta_rms_rad,
            theta_settle_rad=window.theta_mean_rad,
            theta_from_mean_vectors_rad=theta_from_mean,
            settle_z_grapple_base=window.mean_z_grapple_base,
            settle_z_up_base=window.mean_z_up_base,
            settle_dot_product=settle_dot,
            settle_reward=settle_reward,
            reward_exponent=self.config.reward_exponent,
            accepted_window_start_ns=window.start_ns,
            accepted_window_end_ns=window.end_ns,
            measurement_sample_count=self._sample_count,
            accepted_window_sample_count=window.sample_count,
            final_stability_variance_rad2=window.stability_variance_rad2,
            final_gravity_variance_rad2=window.gravity_variance_rad2,
            final_grapple_variance_rad2=window.grapple_variance_rad2,
            sample_offsets_sec=self._series_offsets_sec(),
            sample_theta_rad=[theta for _, theta in self._theta_series],
        )

    def make_failure_result(
        self,
        reason: str,
        measurement_end_ns: Optional[int] = None,
    ) -> SettlingResult:
        end_ns = (
            int(measurement_end_ns)
            if measurement_end_ns is not None
            else (self._last_stamp_ns or self.measurement_start_ns)
        )
        nan_vec = np.array([math.nan, math.nan, math.nan], dtype=float)
        # Best-effort estimate over whatever is in the rolling window so a
        # timeout still reports a number instead of NaN. success=False and
        # failure_reason mark it as not properly settled; t_settle stays NaN.
        theta_settle = math.nan
        theta_from_mean = math.nan
        settle_z_grapple = nan_vec
        settle_z_up = nan_vec
        settle_dot = math.nan
        settle_reward = math.nan
        window_start_ns = 0
        window_end_ns = 0
        window_count = 0
        stability_variance = math.nan
        gravity_variance = math.nan
        grapple_variance = math.nan
        samples = list(self._window_samples)
        if samples:
            try:
                mean_z_up = normalize(np.mean(
                    [sample.z_up_base for sample in samples], axis=0))
                mean_z_grapple = normalize(np.mean(
                    [sample.z_grapple_base for sample in samples], axis=0))
                theta = np.array(
                    [sample.theta_rad for sample in samples], dtype=float)
                theta_settle = float(np.mean(theta))
                stability_variance = float(np.mean((theta - theta_settle) ** 2))
                gravity_variance = float(np.mean(
                    [angle_between(s.z_up_base, mean_z_up) ** 2 for s in samples]))
                grapple_variance = float(np.mean(
                    [angle_between(s.z_grapple_base, mean_z_grapple) ** 2
                     for s in samples]))
                # Per-sample averages, NOT the mean-vector reward: averaging
                # the vectors of a swinging grapple cancels the oscillation
                # and overstates stability. theta_from_mean keeps the
                # mean-vector angle as a diagnostic of the swing center.
                settle_dot = float(np.mean(
                    [sample.dot_product for sample in samples]))
                settle_reward = float(np.mean(
                    [sample.reward for sample in samples]))
                theta_from_mean = angle_between(mean_z_grapple, mean_z_up)
                settle_z_grapple = mean_z_grapple
                settle_z_up = mean_z_up
                window_start_ns = samples[0].stamp_ns
                window_end_ns = samples[-1].stamp_ns
                window_count = len(samples)
            except ValueError:
                pass
        return SettlingResult(
            success=False,
            failure_reason=reason,
            measurement_start_ns=self.measurement_start_ns,
            measurement_end_ns=end_ns,
            t_settle_sec=math.nan,
            theta_max_rad=self.theta_max_rad,
            theta_rms_rad=self.theta_rms_rad,
            theta_settle_rad=theta_settle,
            theta_from_mean_vectors_rad=theta_from_mean,
            settle_z_grapple_base=settle_z_grapple,
            settle_z_up_base=settle_z_up,
            settle_dot_product=settle_dot,
            settle_reward=settle_reward,
            reward_exponent=self.config.reward_exponent,
            accepted_window_start_ns=window_start_ns,
            accepted_window_end_ns=window_end_ns,
            measurement_sample_count=self._sample_count,
            accepted_window_sample_count=window_count,
            final_stability_variance_rad2=stability_variance,
            final_gravity_variance_rad2=gravity_variance,
            final_grapple_variance_rad2=grapple_variance,
            sample_offsets_sec=self._series_offsets_sec(),
            sample_theta_rad=[theta for _, theta in self._theta_series],
        )

    def _series_offsets_sec(self) -> list:
        return [(stamp_ns - self.measurement_start_ns) / NSEC_PER_SEC
                for stamp_ns, _ in self._theta_series]
