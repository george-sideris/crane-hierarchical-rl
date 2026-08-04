#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


FSM_TOPIC = "/crane/fsm_state"
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
    _, _, t = reader.read_next()
    return t


def ns_to_stamp(t_ns):
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).isoformat()


def normalize_state(value):
    if value is None:
        return ""
    return value.strip().split(" (cycle", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", help="Path to ROS 2 bag directory")
    parser.add_argument("--target", default=TARGET_STATE)
    args = parser.parse_args()

    bag_start_ns = get_bag_start_ns(args.bag)
    target_state = normalize_state(args.target)

    reader, type_map = open_reader(args.bag, topics=[FSM_TOPIC])
    msg_type = get_message(type_map[FSM_TOPIC])

    intervals = []
    current_state = None
    in_target = False
    target_start_ns = None
    last_t_ns = None

    print("\nFSM transitions:")
    print("offset_s, absolute_time_utc, state")

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        msg = deserialize_message(data, msg_type)

        state = normalize_state(msg.data)
        offset_s = (t_ns - bag_start_ns) / 1e9
        last_t_ns = t_ns

        if state != current_state:
            print(f"{offset_s:10.3f}, {ns_to_stamp(t_ns)}, {state}")
            current_state = state

        if state == target_state and not in_target:
            target_start_ns = t_ns
            in_target = True

        elif state != target_state and in_target:
            intervals.append((target_start_ns, t_ns))
            in_target = False

    if in_target and last_t_ns is not None:
        intervals.append((target_start_ns, last_t_ns))

    print(f"\nIntervals where /crane/fsm_state == {target_state}:")
    for i, (start_ns, end_ns) in enumerate(intervals):
        start_offset = (start_ns - bag_start_ns) / 1e9
        end_offset = (end_ns - bag_start_ns) / 1e9
        duration = (end_ns - start_ns) / 1e9

        print(
            f"[{i}] "
            f"start_offset={start_offset:.3f}s, "
            f"end_offset={end_offset:.3f}s, "
            f"duration={duration:.3f}s, "
            f"start_ns={start_ns}, "
            f"end_ns={end_ns}"
        )


if __name__ == "__main__":
    main()