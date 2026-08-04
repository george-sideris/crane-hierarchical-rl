#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_quaternion(csv_path):
    t = []
    quaternion = {
        "qx": [],
        "qy": [],
        "qz": [],
        "qw": [],
    }

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"t_offset_s", *quaternion.keys()}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        for row in reader:
            t.append(float(row["t_offset_s"]))
            for component in quaternion:
                quaternion[component].append(float(row[component]))

    return t, quaternion


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot base_link -> zed_0_left_camera_frame quaternion components "
            "from compose_base_to_zed_from_tf_csv.py output."
        )
    )
    parser.add_argument(
        "base_to_zed_csv",
        nargs="?",
        default="lift_high_extract_0/base_to_zed_0_left_camera_frame.csv",
        help="CSV containing t_offset_s, qx, qy, qz, and qw columns",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "base_to_zed_quaternion_lift_high.png next to the CSV."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    args = parser.parse_args()

    csv_path = Path(args.base_to_zed_csv)
    output_path = (
        Path(args.output)
        if args.output is not None
        else csv_path.with_name("base_to_zed_quaternion_lift_high.png")
    )

    t, quaternion = load_quaternion(csv_path)

    plt.figure(figsize=(12, 7))
    plt.plot(t, quaternion["qx"], label="qx")
    plt.plot(t, quaternion["qy"], label="qy")
    plt.plot(t, quaternion["qz"], label="qz")
    plt.plot(t, quaternion["qw"], label="qw")

    plt.xlabel("Bag offset time [s]")
    plt.ylabel("Quaternion component")
    plt.title("base_link -> zed_0_left_camera_frame rotation during LIFT_HIGH")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")

    if not args.no_show:
        plt.show()

    # extract metrics for quaternion components
    qx_mean = np.mean(quaternion["qx"])
    qy_mean = np.mean(quaternion["qy"])
    qz_mean = np.mean(quaternion["qz"])
    qw_mean = np.mean(quaternion["qw"])

    print(f"Quaternion means - qx: {qx_mean}, qy: {qy_mean}, qz: {qz_mean}, qw: {qw_mean}")


if __name__ == "__main__":
    main()
