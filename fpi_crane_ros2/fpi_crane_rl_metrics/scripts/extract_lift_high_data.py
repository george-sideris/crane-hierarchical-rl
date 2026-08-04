#!/usr/bin/env python3

import argparse
import csv
import os
from datetime import datetime, timezone

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


FSM_TOPIC = "/crane/fsm_state"
IMU_TOPIC = "/zed_0/zed_node/imu/data"
JOINT_TOPIC = "/joint_states"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TARGET_STATE = "LIFT_HIGH"


def open_reader(bag_path, topics=None):
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))

    return reader, type_map


def get_bag_start_ns(bag_path):
    reader, _ = open_reader(bag_path)
    if not reader.has_next():
        raise RuntimeError("Bag is empty")
    _, _, t_ns = reader.read_next()
    return t_ns


def normalize_state(value):
    if value is None:
        return ""
    return value.strip().split(" (cycle", 1)[0]


def find_intervals(bag_path, target_state):
    bag_start_ns = get_bag_start_ns(bag_path)

    reader, type_map = open_reader(bag_path, topics=[FSM_TOPIC])
    msg_type = get_message(type_map[FSM_TOPIC])

    intervals = []
    in_target = False
    target_start_ns = None
    last_t_ns = None

    normalized_target_state = normalize_state(target_state)

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, msg_type)
        state = normalize_state(msg.data)
        last_t_ns = t_ns

        if state == normalized_target_state and not in_target:
            target_start_ns = t_ns
            in_target = True

        elif state != normalized_target_state and in_target:
            intervals.append((target_start_ns, t_ns))
            in_target = False

    if in_target and last_t_ns is not None:
        intervals.append((target_start_ns, last_t_ns))

    return bag_start_ns, intervals


def in_intervals(t_ns, intervals):
    return any(start_ns <= t_ns <= end_ns for start_ns, end_ns in intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to ROS 2 bag directory")
    parser.add_argument("--out", default="lift_high_extract")
    parser.add_argument("--target", default=TARGET_STATE)
    parser.add_argument(
        "--interval-index",
        type=int,
        default=None,
        help="Only extract one LIFT_HIGH interval by index. Default extracts all.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    bag_start_ns, intervals = find_intervals(args.bag, args.target)

    if args.interval_index is not None:
        intervals = [intervals[args.interval_index]]

    print(f"Extracting {len(intervals)} interval(s):")
    for i, (start_ns, end_ns) in enumerate(intervals):
        print(
            f"[{i}] start_offset={(start_ns - bag_start_ns) / 1e9:.3f}s, "
            f"end_offset={(end_ns - bag_start_ns) / 1e9:.3f}s, "
            f"duration={(end_ns - start_ns) / 1e9:.3f}s"
        )

    topics = [IMU_TOPIC, JOINT_TOPIC, TF_TOPIC, TF_STATIC_TOPIC]
    reader, type_map = open_reader(args.bag, topics=topics)

    imu_type = get_message(type_map[IMU_TOPIC])
    joint_type = get_message(type_map[JOINT_TOPIC])
    tf_type = get_message(type_map[TF_TOPIC])

    imu_path = os.path.join(args.out, "imu_lift_high.csv")
    joint_path = os.path.join(args.out, "joint_states_lift_high_long.csv")
    tf_path = os.path.join(args.out, "tf_lift_high_long.csv")

    with open(imu_path, "w", newline="") as imu_f, \
         open(joint_path, "w", newline="") as joint_f, \
         open(tf_path, "w", newline="") as tf_f:

        imu_writer = csv.DictWriter(
            imu_f,
            fieldnames=[
                "t_ns", "t_offset_s",
                "ax", "ay", "az",
                "gx", "gy", "gz",
                "qx", "qy", "qz", "qw",
            ],
        )
        imu_writer.writeheader()

        joint_writer = csv.DictWriter(
            joint_f,
            fieldnames=[
                "t_ns", "t_offset_s",
                "joint_name", "position", "velocity", "effort",
            ],
        )
        joint_writer.writeheader()

        tf_writer = csv.DictWriter(
            tf_f,
            fieldnames=[
                "t_ns", "t_offset_s",
                "parent_frame", "child_frame",
                "x", "y", "z",
                "qx", "qy", "qz", "qw",
                "is_static",
            ],
        )
        tf_writer.writeheader()

        while reader.has_next():
            topic, data, t_ns = reader.read_next()

            # Always keep tf_static, because it may occur outside the LIFT_HIGH interval.
            keep = topic == TF_STATIC_TOPIC or in_intervals(t_ns, intervals)
            if not keep:
                continue

            t_offset_s = (t_ns - bag_start_ns) / 1e9

            if topic == IMU_TOPIC:
                msg = deserialize_message(data, imu_type)
                imu_writer.writerow({
                    "t_ns": t_ns,
                    "t_offset_s": t_offset_s,
                    "ax": msg.linear_acceleration.x,
                    "ay": msg.linear_acceleration.y,
                    "az": msg.linear_acceleration.z,
                    "gx": msg.angular_velocity.x,
                    "gy": msg.angular_velocity.y,
                    "gz": msg.angular_velocity.z,
                    "qx": msg.orientation.x,
                    "qy": msg.orientation.y,
                    "qz": msg.orientation.z,
                    "qw": msg.orientation.w,
                })

            elif topic == JOINT_TOPIC:
                msg = deserialize_message(data, joint_type)
                n = len(msg.name)

                for i, name in enumerate(msg.name):
                    pos = msg.position[i] if i < len(msg.position) else ""
                    vel = msg.velocity[i] if i < len(msg.velocity) else ""
                    eff = msg.effort[i] if i < len(msg.effort) else ""

                    joint_writer.writerow({
                        "t_ns": t_ns,
                        "t_offset_s": t_offset_s,
                        "joint_name": name,
                        "position": pos,
                        "velocity": vel,
                        "effort": eff,
                    })

            elif topic in [TF_TOPIC, TF_STATIC_TOPIC]:
                msg = deserialize_message(data, tf_type)

                for tr in msg.transforms:
                    tf_writer.writerow({
                        "t_ns": t_ns,
                        "t_offset_s": t_offset_s,
                        "parent_frame": tr.header.frame_id,
                        "child_frame": tr.child_frame_id,
                        "x": tr.transform.translation.x,
                        "y": tr.transform.translation.y,
                        "z": tr.transform.translation.z,
                        "qx": tr.transform.rotation.x,
                        "qy": tr.transform.rotation.y,
                        "qz": tr.transform.rotation.z,
                        "qw": tr.transform.rotation.w,
                        "is_static": topic == TF_STATIC_TOPIC,
                    })

    print("\nWrote:")
    print(f"  {imu_path}")
    print(f"  {joint_path}")
    print(f"  {tf_path}")


if __name__ == "__main__":
    main()