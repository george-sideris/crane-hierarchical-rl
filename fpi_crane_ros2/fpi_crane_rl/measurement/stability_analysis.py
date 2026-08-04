#!/usr/bin/env python3

import argparse
import math
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


CHAIN = [
    ("base_link", "mast"),
    ("mast", "mainboom"),
    ("mainboom", "stick"),
    ("stick", "telescope"),
    ("telescope", "upperpassive"),
    ("upperpassive", "lowerpassive"),
    ("lowerpassive", "grapplecarrier"),
]


def normalize_frame(name: str) -> str:
    return name[1:] if name.startswith("/") else name


def quat_to_R_xyzw(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ])


def transform_msg_to_matrix(tr):
    t = tr.transform.translation
    q = tr.transform.rotation

    T = np.eye(4)
    T[:3, :3] = quat_to_R_xyzw(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def compose_chain(edges, chain):
    T = np.eye(4)

    for parent, child in chain:
        key = (parent, child)
        if key not in edges:
            return None
        T = T @ edges[key]

    return T


def stability_from_T(T_base_grapple, z_ref_base, exponent=4.0):
    R = T_base_grapple[:3, :3]

    z_g_base = R[:, 2]
    z_g_base = z_g_base / np.linalg.norm(z_g_base)

    z_ref_base = np.asarray(z_ref_base, dtype=float)
    z_ref_base = z_ref_base / np.linalg.norm(z_ref_base)

    d = float(np.dot(z_g_base, z_ref_base))
    d = float(np.clip(d, -1.0, 1.0))

    theta_rad = math.acos(d)
    theta_deg = math.degrees(theta_rad)

    reward = max(0.0, d) ** exponent

    return d, theta_deg, reward, z_g_base


def read_bag_tf_metrics(bag_path, z_ref_base, storage_id="sqlite3"):
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topic_types}
    msg_type_cache = {}

    edges = {}
    rows = []

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()

        if topic not in ["/tf", "/tf_static"]:
            continue

        if topic not in msg_type_cache:
            msg_type_cache[topic] = get_message(type_map[topic])

        msg = deserialize_message(data, msg_type_cache[topic])

        for tr in msg.transforms:
            parent = normalize_frame(tr.header.frame_id)
            child = normalize_frame(tr.child_frame_id)
            edges[(parent, child)] = transform_msg_to_matrix(tr)

        T = compose_chain(edges, CHAIN)

        if T is None:
            continue

        d, theta_deg, reward, z_g_base = stability_from_T(
            T,
            z_ref_base=z_ref_base,
            exponent=4.0,
        )

        rows.append({
            "time_sec": bag_time_ns * 1e-9,
            "dot": d,
            "theta_deg": theta_deg,
            "reward": reward,
            "zg_x": z_g_base[0],
            "zg_y": z_g_base[1],
            "zg_z": z_g_base[2],
        })

    return rows


def summarize(rows, label):
    if not rows:
        print(f"\n{label}: no transform samples found.")
        return

    theta = np.array([r["theta_deg"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    dot = np.array([r["dot"] for r in rows])
    zg = np.array([[r["zg_x"], r["zg_y"], r["zg_z"]] for r in rows])

    print(f"\n{label}")
    print("-" * len(label))
    print(f"samples:          {len(rows)}")
    print(f"dot mean:         {dot.mean():.6f}")
    print(f"theta mean deg:   {theta.mean():.3f}")
    print(f"theta std deg:    {theta.std():.6f}")
    print(f"reward mean:      {reward.mean():.6f}")
    print(f"reward std:       {reward.std():.6f}")
    print(f"mean z_g_base:    [{zg[:,0].mean():.6f}, {zg[:,1].mean():.6f}, {zg[:,2].mean():.6f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument(
        "--zref",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 1.0],
        help="Reference up vector expressed in base_link.",
    )
    args = parser.parse_args()

    rows = read_bag_tf_metrics(
        args.bag_path,
        z_ref_base=np.array(args.zref),
        storage_id=args.storage_id,
    )

    summarize(rows, label=args.bag_path)


if __name__ == "__main__":
    main()