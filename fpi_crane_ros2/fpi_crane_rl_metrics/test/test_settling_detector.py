import math

import numpy as np
import pytest

from fpi_crane_rl_metrics.settling_detector import (
    NSEC_PER_SEC,
    SettlingConfig,
    SettlingDetector,
    normalize,
)


def cfg(**overrides):
    values = dict(
        required_window_sec=1.0,
        required_dwell_sec=0.2,
        min_samples=6,
        max_sample_gap_sec=0.25,
        stability_variance_threshold_rad2=1.0e-4,
        gravity_variance_threshold_rad2=1.0e-5,
        grapple_variance_threshold_rad2=1.0e-4,
        reward_exponent=4.0,
    )
    values.update(overrides)
    return SettlingConfig(**values)


def ns(t):
    return int(round(t * NSEC_PER_SEC))


def tilted(theta, phi=math.pi / 2.0):
    return normalize(np.array([
        math.sin(theta) * math.cos(phi),
        math.sin(theta) * math.sin(phi),
        math.cos(theta),
    ]))


def feed(detector, entries):
    last = None
    for t, zg, zu in entries:
        last = detector.add_sample(ns(t), zg, zu)
    return last


def constant_entries(times, theta=0.1):
    up = np.array([0.0, 0.0, 1.0])
    return [(t, tilted(theta), up) for t in times]


def test_constant_settled_angle_with_small_noise_succeeds():
    detector = SettlingDetector(cfg(), ns(0.0))
    times = [i * 0.1 for i in range(15)]
    entries = [
        (float(t), tilted(0.1 + 1.0e-4 * math.sin(20.0 * t)), np.array([0.0, 0.0, 1.0]))
        for t in times
    ]
    update = feed(detector, entries)
    assert update.accepted
    result = detector.make_result(update.sample.stamp_ns)
    assert result.success
    assert result.t_settle_sec == pytest.approx(0.0)
    assert result.theta_settle_rad == pytest.approx(0.1, abs=2.0e-4)


def test_damped_oscillation_succeeds_only_after_dwell():
    detector = SettlingDetector(cfg(stability_variance_threshold_rad2=2.0e-5), ns(0.0))
    accepted_at = None
    for t in np.arange(0.0, 4.01, 0.1):
        amp = 0.08 * math.exp(-1.5 * t)
        update = detector.add_sample(
            ns(float(t)),
            tilted(0.1 + amp * math.sin(12.0 * t)),
            [0, 0, 1],
        )
        if update.accepted:
            accepted_at = t
            break
    assert accepted_at is not None
    assert accepted_at > 1.0


def test_undamped_oscillation_times_out_by_never_accepting():
    detector = SettlingDetector(cfg(stability_variance_threshold_rad2=1.0e-5), ns(0.0))
    updates = [
        detector.add_sample(ns(float(t)), tilted(0.1 + 0.05 * math.sin(20.0 * t)), [0, 0, 1])
        for t in np.arange(0.0, 5.0, 0.1)
    ]
    assert not any(update.accepted for update in updates)


def test_constant_stability_angle_with_grapple_cone_motion_is_rejected():
    detector = SettlingDetector(cfg(grapple_variance_threshold_rad2=1.0e-5), ns(0.0))
    updates = [
        detector.add_sample(ns(float(t)), tilted(0.1, phi=8.0 * t), [0, 0, 1])
        for t in np.arange(0.0, 3.0, 0.1)
    ]
    assert not any(update.accepted for update in updates)
    assert updates[-1].window.status == "grapple variance above threshold"


def test_stable_grapple_with_unstable_gravity_is_rejected():
    detector = SettlingDetector(cfg(
        stability_variance_threshold_rad2=1.0,
        grapple_variance_threshold_rad2=1.0,
        gravity_variance_threshold_rad2=1.0e-5,
    ), ns(0.0))
    updates = [
        detector.add_sample(ns(float(t)), tilted(0.1), tilted(0.04, phi=8.0 * t))
        for t in np.arange(0.0, 3.0, 0.1)
    ]
    assert not any(update.accepted for update in updates)
    assert updates[-1].window.status == "gravity variance above threshold"


def test_slow_drift_does_not_falsely_settle():
    detector = SettlingDetector(cfg(stability_variance_threshold_rad2=1.0e-5), ns(0.0))
    updates = [
        detector.add_sample(ns(float(t)), tilted(0.02 + 0.08 * t), [0, 0, 1])
        for t in np.arange(0.0, 3.0, 0.1)
    ]
    assert not any(update.accepted for update in updates)


def test_insufficient_samples_do_not_settle():
    detector = SettlingDetector(cfg(min_samples=20), ns(0.0))
    update = feed(detector, constant_entries(np.arange(0.0, 1.21, 0.2)))
    assert not update.accepted
    assert update.window.status == "insufficient samples"


def test_excessive_timestamp_gaps_do_not_settle():
    detector = SettlingDetector(cfg(max_sample_gap_sec=0.15), ns(0.0))
    update = feed(detector, constant_entries([0.0, 0.1, 0.2, 0.4, 0.5, 1.0, 1.1, 1.2]))
    assert not update.accepted
    assert update.window.status == "excessive timestamp gap"


def test_invalid_or_nonfinite_vectors_are_rejected():
    detector = SettlingDetector(cfg(), ns(0.0))
    update = detector.add_sample(ns(0.0), [math.nan, 0, 1], [0, 0, 1])
    assert update.sample is None
    assert update.measurement_sample_count == 0
    assert "invalid sample" in update.status


def test_exact_threshold_boundaries_pass_deterministically():
    detector = SettlingDetector(cfg(
        required_window_sec=1.0,
        required_dwell_sec=0.0,
        min_samples=2,
        max_sample_gap_sec=1.0,
        stability_variance_threshold_rad2=np.nextafter(0.01, 1.0),
        gravity_variance_threshold_rad2=0.0,
        grapple_variance_threshold_rad2=0.011,
    ), ns(0.0))
    detector.add_sample(ns(0.0), tilted(0.0), [0, 0, 1])
    update = detector.add_sample(ns(1.0), tilted(0.2), [0, 0, 1])
    assert update.accepted
    assert update.window.stability_variance_rad2 == pytest.approx(0.01)


def test_t_settle_refers_to_continuously_passing_interval_start():
    detector = SettlingDetector(cfg(), ns(0.5))
    update = feed(detector, constant_entries(np.arange(0.5, 1.91, 0.1)))
    result = detector.make_result(update.sample.stamp_ns)
    assert result.t_settle_sec == pytest.approx(0.0)
    assert result.measurement_end_ns > result.accepted_window_start_ns


def test_theta_max_includes_early_transient_before_settling():
    detector = SettlingDetector(cfg(stability_variance_threshold_rad2=5.0e-5), ns(0.0))
    entries = [(0.0, tilted(0.5), [0, 0, 1])]
    entries += constant_entries(np.arange(0.1, 2.1, 0.1), theta=0.1)
    update = feed(detector, entries)
    assert update.accepted
    result = detector.make_result(update.sample.stamp_ns)
    assert result.theta_max_rad == pytest.approx(0.5)


def test_theta_rms_includes_entire_period_and_matches_hand_computation():
    detector = SettlingDetector(cfg(required_dwell_sec=0.0), ns(0.0))
    angles = [0.3] + [0.1] * 11
    updates = [
        detector.add_sample(ns(i * 0.1), tilted(theta), [0, 0, 1])
        for i, theta in enumerate(angles)
    ]
    result = detector.make_result(updates[-1].sample.stamp_ns)
    expected = math.sqrt(sum(theta * theta for theta in angles) / len(angles))
    assert result.theta_rms_rad == pytest.approx(expected)


def test_failed_candidate_does_not_reset_theta_max_or_rms():
    detector = SettlingDetector(cfg(stability_variance_threshold_rad2=1.0e-5), ns(0.0))
    angles = [0.4] + [0.1] * 11 + [0.25] + [0.1] * 14
    updates = [
        detector.add_sample(ns(i * 0.1), tilted(theta), [0, 0, 1])
        for i, theta in enumerate(angles)
    ]
    assert updates[-1].accepted
    result = detector.make_result(updates[-1].sample.stamp_ns)
    expected = math.sqrt(sum(theta * theta for theta in angles) / len(angles))
    assert result.theta_max_rad == pytest.approx(0.4)
    assert result.theta_rms_rad == pytest.approx(expected)


def test_theta_settle_is_final_window_arithmetic_mean_only():
    detector = SettlingDetector(cfg(
        required_dwell_sec=0.0,
        stability_variance_threshold_rad2=1.0e-3,
        grapple_variance_threshold_rad2=1.0e-3,
    ), ns(0.0))
    angles = [0.5, 0.4] + [0.12] * 5 + [0.08] * 6
    updates = [
        detector.add_sample(ns(i * 0.1), tilted(theta), [0, 0, 1])
        for i, theta in enumerate(angles)
    ]
    result = detector.make_result(updates[-1].sample.stamp_ns)
    assert result.theta_settle_rad == pytest.approx(np.mean(angles[-11:]))
    assert result.theta_settle_rad != pytest.approx(np.mean(angles))


def test_required_dwell_does_not_change_theta_settle_definition():
    angles = [0.1] * 20
    results = []
    for dwell in (0.0, 0.5):
        detector = SettlingDetector(cfg(required_dwell_sec=dwell), ns(0.0))
        updates = [
            detector.add_sample(ns(i * 0.1), tilted(theta), [0, 0, 1])
            for i, theta in enumerate(angles)
        ]
        results.append(detector.make_result(updates[-1].sample.stamp_ns))
    assert results[0].accepted_window_start_ns == results[1].accepted_window_start_ns
    assert results[0].theta_settle_rad == pytest.approx(results[1].theta_settle_rad)


def test_representative_vectors_are_normalized_means():
    detector = SettlingDetector(cfg(required_dwell_sec=0.0), ns(0.0))
    update = feed(detector, constant_entries(np.arange(0.0, 1.11, 0.1), theta=0.1))
    result = detector.make_result(update.sample.stamp_ns)
    assert np.linalg.norm(result.settle_z_grapple_base) == pytest.approx(1.0)
    assert np.linalg.norm(result.settle_z_up_base) == pytest.approx(1.0)
    assert result.settle_z_grapple_base[2] == pytest.approx(math.cos(0.1))


def test_settle_dot_product_and_reward_match_mean_vectors_and_exponent():
    exponent = 2.5
    detector = SettlingDetector(cfg(required_dwell_sec=0.0, reward_exponent=exponent), ns(0.0))
    update = feed(detector, constant_entries(np.arange(0.0, 1.11, 0.1), theta=0.2))
    result = detector.make_result(update.sample.stamp_ns)
    expected_dot = float(np.dot(result.settle_z_grapple_base, result.settle_z_up_base))
    assert result.settle_dot_product == pytest.approx(expected_dot)
    assert result.settle_reward == pytest.approx(max(0.0, expected_dot) ** exponent)


def test_theta_settle_can_differ_from_theta_from_mean_vectors():
    detector = SettlingDetector(cfg(
        required_dwell_sec=0.0,
        stability_variance_threshold_rad2=1.0,
        grapple_variance_threshold_rad2=1.0,
    ), ns(0.0))
    entries = []
    for i, phi in enumerate(np.linspace(0.0, 2.0 * math.pi, 11, endpoint=False)):
        entries.append((i * 0.1, tilted(0.3, phi=phi), np.array([0.0, 0.0, 1.0])))
    update = feed(detector, entries)
    result = detector.make_result(update.sample.stamp_ns)
    assert result.theta_settle_rad == pytest.approx(0.3)
    assert result.theta_from_mean_vectors_rad == pytest.approx(0.0, abs=1.0e-12)
    assert result.theta_settle_rad != pytest.approx(result.theta_from_mean_vectors_rad)
