#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


DEFAULT_URDF_PATH = (
    PACKAGE_DIR / "urdf" / "fpi_crane_fk_minimal.urdf"
)


def read_joint_states_csv(csv_path: Path) -> OrderedDict[tuple[str, str], dict]:
    samples = OrderedDict()

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"t_ns", "t_offset_s", "joint_name", "position"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        for row in reader:
            key = (row["t_ns"], row["t_offset_s"])
            sample = samples.setdefault(
                key,
                {
                    "t_ns": int(row["t_ns"]),
                    "t_offset_s": float(row["t_offset_s"]),
                    "positions": {},
                },
            )
            sample["positions"][row["joint_name"]] = float(row["position"])

    return samples


def make_joint_state(sample: dict, required_joints: tuple[str, ...]):
    try:
        from sensor_msgs.msg import JointState
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing ROS Python package 'sensor_msgs'. Source the ROS 2 "
            "workspace before running this script."
        ) from exc

    missing_joints = [
        joint_name
        for joint_name in required_joints
        if joint_name not in sample["positions"]
    ]
    if missing_joints:
        raise ValueError(
            f"Timestamp {sample['t_ns']} is missing required joints: "
            f"{missing_joints}"
        )

    msg = JointState()
    msg.header.stamp.sec = sample["t_ns"] // 1_000_000_000
    msg.header.stamp.nanosec = sample["t_ns"] % 1_000_000_000
    msg.name = list(required_joints)
    msg.position = [
        sample["positions"][joint_name]
        for joint_name in required_joints
    ]

    return msg


def output_fieldnames() -> list[str]:
    transform_columns = [
        f"T_base_grapple_{row}{col}"
        for row in range(4)
        for col in range(4)
    ]

    return [
        "t_ns",
        "t_offset_s",
        *transform_columns,
        "z_axis_x",
        "z_axis_y",
        "z_axis_z",
    ]


def make_output_row(sample: dict, transform, z_axis) -> dict[str, float]:
    row = {
        "t_ns": sample["t_ns"],
        "t_offset_s": sample["t_offset_s"],
        "z_axis_x": float(z_axis[0]),
        "z_axis_y": float(z_axis[1]),
        "z_axis_z": float(z_axis[2]),
    }

    for transform_row in range(4):
        for transform_col in range(4):
            row[f"T_base_grapple_{transform_row}{transform_col}"] = float(
                transform[transform_row, transform_col]
            )

    return row


def solve_fk_for_csv(
    joint_states_csv: Path,
    output_csv: Path,
    urdf_path: Path,
) -> None:
    try:
        from fpi_crane_kinematics.crane_fk import CraneFK
    except ModuleNotFoundError as exc:
        if exc.name == "ikpy":
            raise RuntimeError(
                "Missing Python dependency 'ikpy', which is required by "
                "CraneFK. Install it or run in the environment where the FK "
                "service already works."
            ) from exc
        raise

    fk_model = CraneFK(str(urdf_path))
    samples = read_joint_states_csv(joint_states_csv)

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=output_fieldnames(),
        )
        writer.writeheader()

        for sample in samples.values():
            joint_state = make_joint_state(
                sample,
                fk_model.REQUIRED_JOINTS,
            )
            fk_result = fk_model.forward_from_joint_state(joint_state)
            writer.writerow(
                make_output_row(
                    sample,
                    fk_result.transform,
                    fk_result.z_axis,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Solve T_base_grapple and grapple z-axis for each timestamp in a "
            "long-format /joint_states CSV."
        )
    )
    parser.add_argument(
        "joint_states_csv",
        type=Path,
        help="CSV with t_ns, t_offset_s, joint_name, and position columns",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"URDF used by CraneFK. Defaults to {DEFAULT_URDF_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input stem>_fk.csv next to input.",
    )
    args = parser.parse_args()

    output_csv = (
        args.output
        if args.output is not None
        else args.joint_states_csv.with_name(
            f"{args.joint_states_csv.stem}_fk.csv"
        )
    )

    try:
        solve_fk_for_csv(
            joint_states_csv=args.joint_states_csv,
            output_csv=output_csv,
            urdf_path=args.urdf,
        )
    except Exception as exc:
        raise SystemExit(f"FK solve failed: {exc}") from exc

    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
