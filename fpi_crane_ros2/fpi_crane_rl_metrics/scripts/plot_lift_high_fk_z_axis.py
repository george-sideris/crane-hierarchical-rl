#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_z_axis(csv_path):
    t = []
    z_axis = {
        "z_axis_x": [],
        "z_axis_y": [],
        "z_axis_z": [],
    }

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"t_offset_s", *z_axis.keys()}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        for row in reader:
            t.append(float(row["t_offset_s"]))
            for axis_name in z_axis:
                z_axis[axis_name].append(float(row[axis_name]))

    return t, z_axis


def main():
    parser = argparse.ArgumentParser(
        description="Plot grapple z-axis components from solve_fk.py output."
    )
    parser.add_argument(
        "fk_csv",
        nargs="?",
        default="lift_high_extract_0/joint_states_lift_high_long_fk.csv",
        help="CSV containing t_offset_s, z_axis_x, z_axis_y, and z_axis_z columns",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to fk_z_axis_lift_high.png next to the CSV.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    args = parser.parse_args()

    csv_path = Path(args.fk_csv)
    output_path = (
        Path(args.output)
        if args.output is not None
        else csv_path.with_name("fk_z_axis_lift_high.png")
    )

    t, z_axis = load_z_axis(csv_path)

    plt.figure(figsize=(12, 7))
    plt.plot(t, z_axis["z_axis_x"], label="z_axis_x")
    plt.plot(t, z_axis["z_axis_y"], label="z_axis_y")
    plt.plot(t, z_axis["z_axis_z"], label="z_axis_z")

    plt.xlabel("Bag offset time [s]")
    plt.ylabel("Grapple z-axis component in base_link")
    plt.title("Grapple z-axis during LIFT_HIGH")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved {output_path}")

    if not args.no_show:
        plt.show()

    # extract metrics for z-axis components
    z_x_mean = np.mean(z_axis["z_axis_x"])
    z_y_mean = np.mean(z_axis["z_axis_y"])
    z_z_mean = np.mean(z_axis["z_axis_z"])

    print(f"Z-axis means - X: {z_x_mean}, Y: {z_y_mean}, Z: {z_z_mean}")

if __name__ == "__main__":
    main()
