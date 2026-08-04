#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_joint_positions(csv_path):
    times_by_joint = defaultdict(list)
    positions_by_joint = defaultdict(list)

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"t_offset_s", "joint_name", "position"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        for row in reader:
            joint_name = row["joint_name"]
            times_by_joint[joint_name].append(float(row["t_offset_s"]))
            positions_by_joint[joint_name].append(float(row["position"]))

    return times_by_joint, positions_by_joint


def main():
    parser = argparse.ArgumentParser(
        description="Plot /joint_states joint positions from a long-format CSV."
    )
    parser.add_argument(
        "joint_states_csv",
        nargs="?",
        default="lift_high_extract_0/joint_states_lift_high_long.csv",
        help="CSV containing t_offset_s, joint_name, and position columns",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to joint_positions_lift_high.png next to the CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    args = parser.parse_args()

    csv_path = Path(args.joint_states_csv)
    output_path = (
        Path(args.output)
        if args.output is not None
        else csv_path.with_name("joint_positions_lift_high.png")
    )

    times_by_joint, positions_by_joint = load_joint_positions(csv_path)

    plt.figure(figsize=(12, 7))
    for joint_name in positions_by_joint:
        plt.plot(
            times_by_joint[joint_name],
            positions_by_joint[joint_name],
            label=joint_name,
        )

    plt.xlabel("Bag offset time [s]")
    plt.ylabel("Joint position [rad or m]")
    plt.title("/joint_states positions during LIFT_HIGH")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
