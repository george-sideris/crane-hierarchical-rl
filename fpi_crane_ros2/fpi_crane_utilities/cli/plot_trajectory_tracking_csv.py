#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List


ALL_JOINTS_NAME = "all_joints"
ALL_JOINT_ERROR_PLOTS = (
    ("slew_joint", "Slew"),
    ("boom_joint", "Boom"),
    ("stick_joint", "Stick"),
    ("telescope_joint", "Telescope"),
    ("grapplecarrier_joint", "Grapplecarrier"),
)

REQUIRED_COLUMNS = (
    "time_sec",
    "joint_name",
    "target_position",
    "actual_position",
    "position_error",
    "target_velocity",
    "actual_velocity",
    "velocity_error",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot target, actual, and tracking error for one joint from a "
            "trajectory_tracking_monitor CSV file."
        )
    )
    parser.add_argument("csv_file", help="Path to trajectory tracking CSV file")
    parser.add_argument(
        "joint_name",
        help=(
            "Joint name to plot. Use 'all_joints' to plot slew, boom, stick, "
            "telescope, and grapplecarrier position errors plus setpoint/actual "
            "joint states, velocity tracking, actual effort, and actual RRC."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        help=(
            "Optional output image path. With 'all_joints', related figures are "
            "saved beside it with '_states', '_velocity_errors', and "
            "'_velocity_states', '_effort', and '_rrc' appended. If omitted, the "
            "plot window is shown."
        ),
    )
    parser.add_argument(
        "--title",
        help="Optional figure title. Defaults to the joint name.",
    )
    return parser.parse_args()


def load_csv_rows(csv_file: Path) -> List[Dict[str, str]]:
    with csv_file.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_file} has no CSV header")

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"{csv_file} is missing required columns: {missing_columns}"
            )

        return list(reader)


def filter_joint_rows(
    rows: List[Dict[str, str]],
    joint_name: str,
    csv_file: Path,
) -> List[Dict[str, str]]:
    joint_rows = [row for row in rows if row.get("joint_name") == joint_name]

    if not joint_rows:
        raise ValueError(f"No rows found for joint '{joint_name}' in {csv_file}")

    return joint_rows


def load_joint_rows(csv_file: Path, joint_name: str) -> List[Dict[str, str]]:
    return filter_joint_rows(load_csv_rows(csv_file), joint_name, csv_file)


def to_float_series(
    rows: List[Dict[str, str]],
    column: str,
    allow_empty: bool = False,
) -> List[float]:
    values = []
    for row_index, row in enumerate(rows, start=2):
        try:
            value = row[column]
            if allow_empty and value == "":
                values.append(float("nan"))
            else:
                values.append(float(value))
        except KeyError as exc:
            raise ValueError(f"CSV is missing required column '{column}'") from exc
        except ValueError as exc:
            raise ValueError(
                f"Could not parse column '{column}' as float on CSV row {row_index}: "
                f"{row[column]!r}"
            ) from exc
    return values


def plot_joint(rows: List[Dict[str, str]], joint_name: str, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    time_sec = to_float_series(rows, "time_sec")
    target_position = to_float_series(rows, "target_position")
    actual_position = to_float_series(rows, "actual_position")
    position_error = to_float_series(rows, "position_error")

    start_time = time_sec[0]
    time_from_start = [time - start_time for time in time_sec]

    figure, (trajectory_axis, error_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 7),
        constrained_layout=True,
    )

    trajectory_axis.plot(
        time_from_start,
        target_position,
        label="Setpoint",
        linewidth=1.8,
    )
    trajectory_axis.plot(
        time_from_start,
        actual_position,
        label="Actual",
        linewidth=1.5,
    )
    trajectory_axis.set_title(title)
    trajectory_axis.set_ylabel("Joint Position")
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.legend()

    error_axis.plot(
        time_from_start,
        position_error,
        label="Error",
        color="tab:red",
        linewidth=1.5,
    )
    error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    error_axis.set_xlabel("Time From Start (s)")
    error_axis.set_ylabel("Position Error")
    error_axis.grid(True, alpha=0.3)
    error_axis.legend()

    return figure


def plot_all_joint_errors(rows: List[Dict[str, str]], csv_file: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        position_error = to_float_series(joint_rows, "position_error")
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            position_error,
            label=f"{label} Error",
            color="tab:red",
            linewidth=1.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(label)
        axis.set_ylabel("Position Error")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def plot_all_joint_states(rows: List[Dict[str, str]], csv_file: Path, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        target_position = to_float_series(joint_rows, "target_position")
        actual_position = to_float_series(joint_rows, "actual_position")
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            target_position,
            label="Setpoint",
            linewidth=1.8,
        )
        axis.plot(
            time_from_start,
            actual_position,
            label="Actual",
            linewidth=1.5,
        )
        axis.set_title(label)
        axis.set_ylabel("Joint Position")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def plot_all_joint_velocity_errors(
    rows: List[Dict[str, str]],
    csv_file: Path,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        velocity_error = to_float_series(joint_rows, "velocity_error")
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            velocity_error,
            label=f"{label} Velocity Error",
            color="tab:red",
            linewidth=1.5,
        )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(label)
        axis.set_ylabel("Velocity Error")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def plot_all_joint_velocity_states(
    rows: List[Dict[str, str]],
    csv_file: Path,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        target_velocity = to_float_series(joint_rows, "target_velocity")
        actual_velocity = to_float_series(joint_rows, "actual_velocity")
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            target_velocity,
            label="Setpoint",
            linewidth=1.8,
        )
        axis.plot(
            time_from_start,
            actual_velocity,
            label="Actual",
            linewidth=1.5,
        )
        axis.set_title(label)
        axis.set_ylabel("Joint Velocity")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def plot_all_joint_actual_efforts(
    rows: List[Dict[str, str]],
    csv_file: Path,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        actual_effort = to_float_series(
            joint_rows,
            "actual_effort",
            allow_empty=True,
        )
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            actual_effort,
            label=f"{label} Actual Effort",
            linewidth=1.5,
        )
        axis.set_title(label)
        axis.set_ylabel("Actual Effort")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def plot_all_joint_actual_rrcs(
    rows: List[Dict[str, str]],
    csv_file: Path,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Could not import matplotlib. Install a matplotlib/numpy combination "
            "that is compatible with this Python environment."
        ) from exc

    figure, axes = plt.subplots(
        len(ALL_JOINT_ERROR_PLOTS),
        1,
        sharex=True,
        figsize=(11, 10),
        constrained_layout=True,
    )
    figure.suptitle(title)

    for axis, (joint_name, label) in zip(axes, ALL_JOINT_ERROR_PLOTS):
        joint_rows = filter_joint_rows(rows, joint_name, csv_file)
        time_sec = to_float_series(joint_rows, "time_sec")
        actual_rrc = to_float_series(
            joint_rows,
            "actual_rrc",
            allow_empty=True,
        )
        start_time = time_sec[0]
        time_from_start = [time - start_time for time in time_sec]

        axis.plot(
            time_from_start,
            actual_rrc,
            label=f"{label} Actual RRC",
            linewidth=1.5,
        )
        axis.set_title(label)
        axis.set_ylabel("Actual RRC")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time From Start (s)")
    return figure


def add_filename_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def main() -> None:
    args = parse_args()
    csv_file = Path(args.csv_file).expanduser()

    if args.joint_name == ALL_JOINTS_NAME:
        rows = load_csv_rows(csv_file)
        error_figure = plot_all_joint_errors(
            rows,
            csv_file,
            args.title or "Joint Position Errors",
        )
        state_figure = plot_all_joint_states(
            rows,
            csv_file,
            "Joint Setpoint And Actual Positions",
        )
        velocity_error_figure = plot_all_joint_velocity_errors(
            rows,
            csv_file,
            "Joint Velocity Errors",
        )
        velocity_state_figure = plot_all_joint_velocity_states(
            rows,
            csv_file,
            "Joint Setpoint And Actual Velocities",
        )
        effort_figure = plot_all_joint_actual_efforts(
            rows,
            csv_file,
            "Joint Actual Efforts",
        )
        rrc_figure = plot_all_joint_actual_rrcs(
            rows,
            csv_file,
            "Joint Actual RRC",
        )
        figures = [
            error_figure,
            state_figure,
            velocity_error_figure,
            velocity_state_figure,
            effort_figure,
            rrc_figure,
        ]
    else:
        rows = load_joint_rows(csv_file, args.joint_name)
        figures = [plot_joint(rows, args.joint_name, args.title or args.joint_name)]

    if args.output:
        output = Path(args.output).expanduser()
        if output.parent:
            output.parent.mkdir(parents=True, exist_ok=True)
        figures[0].savefig(output, dpi=150)
        if args.joint_name == ALL_JOINTS_NAME:
            figures[1].savefig(add_filename_suffix(output, "_states"), dpi=150)
            figures[2].savefig(
                add_filename_suffix(output, "_velocity_errors"),
                dpi=150,
            )
            figures[3].savefig(
                add_filename_suffix(output, "_velocity_states"),
                dpi=150,
            )
            figures[4].savefig(add_filename_suffix(output, "_effort"), dpi=150)
            figures[5].savefig(add_filename_suffix(output, "_rrc"), dpi=150)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
