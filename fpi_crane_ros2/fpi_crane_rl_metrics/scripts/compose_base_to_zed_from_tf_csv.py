#!/usr/bin/env python3
"""
Compose base_link -> zed_0_left_camera_frame transforms from an extracted TF CSV.

Assumes CSV columns:
t_ns,t_offset_s,parent_frame,child_frame,x,y,z,qx,qy,qz,qw,is_static
"""

import argparse
import math
import numpy as np
import pandas as pd


CHAIN = [
    ("base_link", "mast"),
    ("mast", "zed_0_camera_link"),
    ("zed_0_camera_link", "zed_0_camera_center"),
    ("zed_0_camera_center", "zed_0_left_camera_frame"),
]


def quat_to_rot(qx, qy, qz, qw):
    """Return 3x3 rotation matrix for ROS quaternion x,y,z,w."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
    return np.array([
        [1.0 - (yy + zz), xy - wz,         xz + wy],
        [xy + wz,         1.0 - (xx + zz), yz - wx],
        [xz - wy,         yz + wx,         1.0 - (xx + yy)],
    ])


def rot_to_quat(R):
    """Return ROS quaternion x,y,z,w from a 3x3 rotation matrix."""
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return q


def row_to_T(row):
    """Convert one TF row to a 4x4 homogeneous transform T_parent_child."""
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(row.qx, row.qy, row.qz, row.qw)
    T[:3, 3] = [row.x, row.y, row.z]
    return T


def latest_edge(df, parent, child, t_ns):
    """
    Pick the transform for parent -> child at query time.

    Dynamic: latest row with timestamp <= t_ns.
    Static: valid for all times, used if no dynamic row is available.
    If both static and dynamic exist for the same edge, dynamic wins once it exists.
    """
    edge = df[(df.parent_frame == parent) & (df.child_frame == child)]
    if edge.empty:
        raise KeyError(f"No TF edge found for {parent} -> {child}")

    dyn = edge[(edge.is_static == False) & (edge.t_ns <= t_ns)].sort_values("t_ns")
    if not dyn.empty:
        return dyn.iloc[-1]

    static = edge[edge.is_static == True].sort_values("t_ns")
    if not static.empty:
        return static.iloc[-1]

    # Fall back to the first future dynamic row if query time is before that edge starts.
    # This should not normally happen for the base->camera query if query times come from TF rows.
    future_dyn = edge[edge.is_static == False].sort_values("t_ns")
    if not future_dyn.empty:
        return future_dyn.iloc[0]

    raise KeyError(f"No usable TF edge found for {parent} -> {child} at {t_ns}")


def compose_at_time(df, t_ns):
    T = np.eye(4)
    used_edges = []
    for parent, child in CHAIN:
        row = latest_edge(df, parent, child, t_ns)
        T = T @ row_to_T(row)
        used_edges.append((parent, child, int(row.t_ns), float(row.t_offset_s), bool(row.is_static)))
    return T, used_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tf_csv", help="Path to tf_lift_high_long.csv")
    parser.add_argument("--out", default="base_to_zed_0_left_camera_frame.csv")
    parser.add_argument(
        "--query-source",
        choices=["base_mast", "all_tf"],
        default="base_mast",
        help="Timestamps to evaluate. base_mast is compact and matches the crane dynamic TF rate.",
    )
    parser.add_argument(
        "--print-one",
        type=float,
        default=None,
        help="Optional bag offset time in seconds. Prints the closest composed transform.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.tf_csv)

    # Make sure booleans are actual bools even if CSV is read as strings on some systems.
    if df["is_static"].dtype != bool:
        df["is_static"] = df["is_static"].astype(str).str.lower().isin(["true", "1"])

    if args.query_source == "base_mast":
        qdf = df[(df.parent_frame == "base_link") & (df.child_frame == "mast") & (df.is_static == False)]
    else:
        qdf = df[df.is_static == False]

    query_times = qdf[["t_ns", "t_offset_s"]].drop_duplicates().sort_values("t_ns")

    rows = []
    for qr in query_times.itertuples(index=False):
        # Use itertuples rather than iterrows so nanosecond int64 timestamps do not lose precision.
        t_ns = int(qr.t_ns)
        T, _ = compose_at_time(df, t_ns)
        qx, qy, qz, qw = rot_to_quat(T[:3, :3])
        rows.append({
            "t_ns": t_ns,
            "t_offset_s": float(qr.t_offset_s),
            "parent_frame": "base_link",
            "child_frame": "zed_0_left_camera_frame",
            "x": T[0, 3],
            "y": T[1, 3],
            "z": T[2, 3],
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
            "camera_x_in_base_x": T[0, 0],
            "camera_x_in_base_y": T[1, 0],
            "camera_x_in_base_z": T[2, 0],
            "camera_y_in_base_x": T[0, 1],
            "camera_y_in_base_y": T[1, 1],
            "camera_y_in_base_z": T[2, 1],
            "camera_z_in_base_x": T[0, 2],
            "camera_z_in_base_y": T[1, 2],
            "camera_z_in_base_z": T[2, 2],
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} composed transforms to {args.out}")

    if args.print_one is not None and not out_df.empty:
        idx = (out_df.t_offset_s - args.print_one).abs().idxmin()
        r = out_df.loc[idx]
        T, used = compose_at_time(df, int(r.t_ns))
        print("\nClosest transform:")
        print(r[["t_ns", "t_offset_s", "x", "y", "z", "qx", "qy", "qz", "qw"]].to_string())
        print("\nEdges used:")
        for e in used:
            print(f"  {e[0]} -> {e[1]}  row_offset={e[3]:.6f}s  is_static={e[4]}")


if __name__ == "__main__":
    main()
