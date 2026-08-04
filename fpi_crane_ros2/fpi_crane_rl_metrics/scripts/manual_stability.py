#!/usr/bin/env python3
"""Measure grapple stability for a manually-completed grasp and log it into a
policy run directory as if the policy node had recorded it.

Use when a run had to be interrupted and the last grasp was finished by hand:
grasp, lift, hold still, then run this. It sends the same MeasureStability
action goal the policy node sends (same defaults), appends a node-schema
record (plus "manual_grasp": true) to <run_dir>/stability.jsonl, and renders
the same angle-vs-time PNG into the run directory.

Run inside the ros2 container with the sandbox workspace sourced:

  python3 manual_stability.py --run-dir /workspace/crane_testbed/logs/<...>/policy_debug/run_<stamp>

The cycle number defaults to the cycle the run was interrupted in: the last
started cycle in decisions.jsonl when that exceeds the last cycle in
stability.jsonl, otherwise last stability cycle plus one. Override with
--cycle (e.g. to re-measure a cycle whose measurement failed). If no stability action server is running
(the policy node's autostarted one dies with it), one is spawned for the
duration of the measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node

from fpi_crane_msgs.action import MeasureStability


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True,
                        help="Policy run directory holding stability.jsonl")
    parser.add_argument("--cycle", type=int, default=None,
                        help="Cycle number to attribute (default: inferred from "
                             "stability.jsonl, see module docstring)")
    parser.add_argument("--action-name", default="measure_stability")
    parser.add_argument("--server-timeout-sec", type=float, default=3.0)
    # Goal defaults mirror crane_policy_node's measure_stability_* parameters.
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--required-window-sec", type=float, default=1.0)
    parser.add_argument("--required-dwell-sec", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--max-sample-gap-sec", type=float, default=0.6)
    parser.add_argument("--stability-variance-threshold-rad2", type=float, default=1.0e-4)
    parser.add_argument("--gravity-variance-threshold-rad2", type=float, default=1.0e-5)
    parser.add_argument("--grapple-variance-threshold-rad2", type=float, default=1.0e-4)
    parser.add_argument("--reward-exponent", type=float, default=4.0)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt for the chosen cycle")
    known, ros_args = parser.parse_known_args(argv)
    return known, ros_args


def read_stability_log(path):
    records = []
    if not os.path.isfile(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def last_started_cycle(run_dir):
    """Highest cycle number the policy started, from the 'cycle' field of
    decisions.jsonl records (a cycle can log several decisions on rescans, so
    counting records over-counts). None when the file is missing/unreadable."""
    path = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.isfile(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cycle = int(json.loads(line).get("cycle", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if last is None or cycle > last:
                last = cycle
    return last


def infer_cycle(records, n_started):
    """Note: a trailing valid=false entry alone does NOT imply a re-measure;
    failed measurements of completed cycles look the same. decisions.jsonl is
    the tiebreaker: a started-but-never-measured cycle is the manual one."""
    last_meas = max((int(r.get("cycle", 0)) for r in records), default=0)
    if n_started is not None and n_started > last_meas:
        return n_started, (
            f"decisions.jsonl last started cycle is {n_started} but stability "
            f"log ends at cycle {last_meas} -> the unmeasured cycle {n_started}")
    if records and not records[-1].get("valid", False):
        last = records[-1]
        last_cycle = int(last.get("cycle", last_meas))
        return last_cycle + 1, (
            f"last stability entry is cycle {last_cycle} "
            f"(valid=false, reason={last.get('reason', '?')}); no newer started "
            f"cycle in decisions.jsonl -> assuming a NEW cycle {last_cycle + 1}. "
            f"Pass --cycle {last_cycle} instead if you are re-measuring that "
            f"failed attempt")
    if last_meas:
        return last_meas + 1, (
            f"last stability entry is cycle {last_meas} with valid=true "
            f"-> cycle {last_meas + 1}")
    return 1, "stability.jsonl is empty/missing -> cycle 1"


def ensure_server(node, client, timeout_sec):
    """Return a Popen handle if we had to spawn the action server, else None."""
    if client.wait_for_server(timeout_sec=timeout_sec):
        return None
    node.get_logger().info(
        "measure_stability server not found; spawning measure_stability_action_node")
    proc = subprocess.Popen(
        ["ros2", "run", "fpi_crane_rl_metrics", "measure_stability_action_node"])
    if not client.wait_for_server(timeout_sec=10.0):
        proc.terminate()
        raise RuntimeError("spawned measure_stability_action_node but the "
                           "action server never came up")
    return proc


class ManualStabilityClient(Node):
    def __init__(self, args):
        super().__init__("manual_stability_client")
        self.args = args
        self.client = ActionClient(self, MeasureStability, args.action_name)

    def measure(self):
        goal = MeasureStability.Goal()
        goal.measurement_start = self.get_clock().now().to_msg()
        goal.timeout_sec = self.args.timeout_sec
        goal.required_window_sec = self.args.required_window_sec
        goal.required_dwell_sec = self.args.required_dwell_sec
        goal.min_samples = self.args.min_samples
        goal.max_sample_gap_sec = self.args.max_sample_gap_sec
        goal.stability_variance_threshold_rad2 = self.args.stability_variance_threshold_rad2
        goal.gravity_variance_threshold_rad2 = self.args.gravity_variance_threshold_rad2
        goal.grapple_variance_threshold_rad2 = self.args.grapple_variance_threshold_rad2
        goal.reward_exponent = self.args.reward_exponent

        self.get_logger().info(
            f"Measuring: timeout={goal.timeout_sec:.1f}s "
            f"window={goal.required_window_sec:.2f}s dwell={goal.required_dwell_sec:.2f}s")
        send_future = self.client.send_goal_async(goal, feedback_callback=self._feedback)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("MeasureStability goal rejected or send failed")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("MeasureStability returned no result")
        settled = (wrapped.status == GoalStatus.STATUS_SUCCEEDED
                   and wrapped.result.success)
        return wrapped.result, settled

    def _feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"  {fb.status}: elapsed={fb.elapsed_sec:.1f}s "
            f"theta={math.degrees(fb.current_theta_rad):.2f}deg "
            f"dwell={fb.candidate_dwell_sec:.2f}s window={fb.window_sample_count}")


def render_plot(result, settled, cycle, run_dir, window_sec, stability_var_rad2):
    """Same figure as crane_policy_node._render_stability_plot, manual variant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.asarray(result.sample_offsets_sec, dtype=np.float64)
        th = np.degrees(np.asarray(result.sample_theta_rad, dtype=np.float64))
        if t.size < 2:
            return None
        meas0 = result.measurement_start.sec + result.measurement_start.nanosec * 1e-9
        w0 = (result.accepted_window_start.sec
              + result.accepted_window_start.nanosec * 1e-9 - meas0)
        w1 = (result.accepted_window_end.sec
              + result.accepted_window_end.nanosec * 1e-9 - meas0)
        thr_deg = math.degrees(math.sqrt(stability_var_rad2))
        std = np.full(t.shape, np.nan)
        for i in range(t.size):
            j = int(np.searchsorted(t, t[i] - window_sec))
            if i + 1 - j >= 2 and t[i] - t[j] >= window_sec * 0.9:
                std[i] = np.std(th[j:i + 1])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                       gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(t, th, lw=0.8, color="tab:blue")
        if w1 > w0 > 0.0:
            ax1.axvspan(w0, w1, color="tab:green" if settled else "tab:orange",
                        alpha=0.25,
                        label="settle window" if settled else "last window (unsettled)")
        if math.isfinite(result.theta_settle_rad):
            ax1.axhline(math.degrees(result.theta_settle_rad), ls="--", lw=0.8,
                        color="tab:gray", label="theta_settle")
        if settled and math.isfinite(result.t_settle_sec):
            ax1.axvline(result.t_settle_sec, ls=":", lw=1.0, color="tab:green",
                        label="t_settle")
        status = "settled" if settled else "NOT settled"
        ax1.set_title(f"cycle {cycle} LIFT_HIGH stability (manual grasp): {status}")
        ax1.set_ylabel("grapple tilt theta [deg]")
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.6)
        lines = []
        if math.isfinite(result.settle_reward):
            lines.append(f"score  {result.settle_reward:.4f}"
                         + ("" if settled else "  (best-effort)"))
        if math.isfinite(result.theta_settle_rad):
            lines.append(f"theta_settle  {math.degrees(result.theta_settle_rad):.2f} deg")
        if math.isfinite(result.theta_max_rad):
            lines.append(f"theta_max  {math.degrees(result.theta_max_rad):.2f} deg")
        if settled and math.isfinite(result.t_settle_sec):
            lines.append(f"t_settle  {result.t_settle_sec:.2f} s")
        if lines:
            ax1.text(0.02, 0.97, "\n".join(lines), transform=ax1.transAxes,
                     va="top", ha="left", fontsize=9, family="monospace",
                     bbox={"boxstyle": "round,pad=0.4", "alpha": 0.85,
                           "facecolor": "honeydew" if settled else "seashell",
                           "edgecolor": "tab:green" if settled else "tab:orange"})

        ax2.plot(t, std, lw=0.8, color="tab:purple")
        ax2.axhline(thr_deg, ls="--", lw=0.8, color="tab:red",
                    label=f"collect threshold ({thr_deg:.2f} deg std)")
        ax2.set_yscale("log")
        ax2.margins(y=0.2)
        ax2.set_ylabel(f"rolling std over {window_sec:.1f}s [deg]")
        ax2.set_xlabel("time since measurement start [s]")
        ax2.legend(loc="best", fontsize=8, framealpha=0.6)
        fig.tight_layout()
        fn = f"stability_{cycle:03d}_manual.png"
        fig.savefig(os.path.join(run_dir, fn), dpi=110)
        plt.close(fig)
        return fn
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: stability plot failed: {exc}", file=sys.stderr)
        return None


def build_record(result, settled, cycle, png):
    rec = {
        "cycle": cycle,
        "phase": "LIFT_HIGH",
        "valid": bool(settled),
        "manual_grasp": True,
        "t_wall": time.time(),
    }
    if settled:
        rec.update({
            "t_settle_sec": float(result.t_settle_sec),
            "theta_max_rad": float(result.theta_max_rad),
            "theta_rms_rad": float(result.theta_rms_rad),
            "theta_settle_rad": float(result.theta_settle_rad),
            "theta_from_mean_vectors_rad": float(result.theta_from_mean_vectors_rad),
            "settle_dot_product": float(result.settle_dot_product),
            "settle_reward": float(result.settle_reward),
            "measurement_sample_count": int(result.measurement_sample_count),
            "accepted_window_sample_count": int(result.accepted_window_sample_count),
        })
    else:
        rec["reason"] = result.failure_reason or "did not settle"
        if math.isfinite(result.settle_reward):
            rec.update({
                "best_effort": True,
                "theta_max_rad": float(result.theta_max_rad),
                "theta_rms_rad": float(result.theta_rms_rad),
                "theta_settle_rad": float(result.theta_settle_rad),
                "theta_from_mean_vectors_rad": float(result.theta_from_mean_vectors_rad),
                "settle_dot_product": float(result.settle_dot_product),
                "settle_reward": float(result.settle_reward),
                "measurement_sample_count": int(result.measurement_sample_count),
                "window_sample_count": int(result.accepted_window_sample_count),
                "stability_variance_rad2": float(result.final_stability_variance_rad2),
                "gravity_variance_rad2": float(result.final_gravity_variance_rad2),
                "grapple_variance_rad2": float(result.final_grapple_variance_rad2),
            })
    # Full raw progression so the metric can be recomputed offline (matches the
    # policy node's stability.jsonl records)
    if result.sample_offsets_sec:
        rec["sample_offsets_sec"] = [round(float(t), 4) for t in result.sample_offsets_sec]
        rec["sample_theta_rad"] = [round(float(th), 5) for th in result.sample_theta_rad]
    if png:
        rec["png"] = png
    return rec


def main(argv=None):
    args, ros_args = parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"run dir does not exist: {run_dir}")
    log_path = os.path.join(run_dir, "stability.jsonl")

    records = read_stability_log(log_path)
    if args.cycle is not None:
        cycle, why = args.cycle, "given on the command line"
    else:
        cycle, why = infer_cycle(records, last_started_cycle(run_dir))
    print(f"Attributing to cycle {cycle} ({why})")
    for rec in records[-2:]:
        print(f"  tail: {json.dumps(rec)[:120]}")
    if not args.yes:
        answer = input(f"Measure now and append as cycle {cycle}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            sys.exit("aborted")

    rclpy.init(args=ros_args)
    spawned = None
    try:
        node = ManualStabilityClient(args)
        spawned = ensure_server(node, node.client, args.server_timeout_sec)
        result, settled = node.measure()
    finally:
        if spawned is not None:
            spawned.terminate()
        rclpy.shutdown()

    png = render_plot(result, settled, cycle, run_dir,
                      args.required_window_sec,
                      args.stability_variance_threshold_rad2)
    rec = build_record(result, settled, cycle, png)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    verdict = "settled" if settled else f"NOT settled ({rec.get('reason', '?')})"
    print(f"\n{verdict}: theta_settle={math.degrees(result.theta_settle_rad):.2f}deg "
          f"reward={result.settle_reward:.4f} "
          f"samples={result.measurement_sample_count}")
    print(f"appended cycle {cycle} record to {log_path}")
    if png:
        print(f"plot: {os.path.join(run_dir, png)}")


if __name__ == "__main__":
    main()
