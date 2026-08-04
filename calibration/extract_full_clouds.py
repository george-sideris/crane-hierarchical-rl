#!/usr/bin/env python3
"""Extract the raw (uncropped) ZED clouds at each policy-trigger moment of a crane run bag,
transformed into the policy's base frame. Companion to the cropped real_cycle{N}.npy inputs:
these are what a full-PCD policy would have seen.

For each /crane/policy_target message (one per grasp cycle) the script takes the candidate
/zed_0 cloud_registered messages in a window before it, transforms each into the frame of
/crane/policy_input using the bag's TF tree, and keeps the candidate whose points best match
the recorded policy_input cloud (the policy input was FPS-sampled from the true source cloud,
so the right candidate matches to ~mm). This validates cloud choice, TF chain and calibration
in one shot.

Run inside a ROS2 humble container with the T9 mounted (see project notes):
  docker run --rm --network host -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e ACCEPT_EULA=Y \
    --entrypoint bash -v /media/george/T9:/mnt/ssd \
    -v /home/george/IsaacLab/crane_testbed:/workspace/crane_testbed isaac-lab-ros2 \
    -c 'source /opt/ros/humble/setup.bash && python3 /workspace/crane_testbed/calibration/extract_full_clouds.py \
        --bag /mnt/ssd/crane_run_20260703_161820 --out /workspace/crane_testbed/calibration/out'
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bag", required=True, help="rosbag2 directory")
ap.add_argument("--out", required=True, help="output directory for real_cycle{N}_full.npy")
ap.add_argument("--cloud_topic", default="/zed_0/zed_node/point_cloud/cloud_registered")
ap.add_argument("--target_topic", default="/crane/policy_target")
ap.add_argument("--input_topic", default="/crane/policy_input")
ap.add_argument("--window", type=float, default=3.0, help="seconds before trigger to search for the source cloud")
ap.add_argument("--min_gap", type=float, default=5.0, help="collapse trigger msgs closer than this into one cycle")
args = ap.parse_args()

import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from rosidl_runtime_py.utilities import get_message
from tf2_ros.buffer import Buffer


def open_reader(topics):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


def topic_types():
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


TYPES = topic_types()
for t in (args.cloud_topic, args.target_topic, args.input_topic, "/tf", "/tf_static"):
    if t not in TYPES:
        raise SystemExit(f"topic {t} not in bag (have: {sorted(TYPES)})")


def msg_class(topic):
    return get_message(TYPES[topic])


def cloud_to_xyz(msg):
    # fast path: contiguous float32 x,y,z at offsets 0/4/8 (ZED layout, point_step 16 or 32)
    assert msg.point_step % 4 == 0, f"unexpected point_step {msg.point_step}"
    flat = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(-1, msg.point_step // 4)
    xyz = flat[:, :3]
    return xyz[np.isfinite(xyz).all(axis=1)]


def quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# pass 1: trigger stamps and recorded policy inputs
print("[pass 1] policy_target / policy_input ...")
triggers = []          # bag-time ns of each cycle's target msg
inputs = []            # (bag ns, frame_id, Nx3) recorded policy input clouds
reader = open_reader([args.target_topic, args.input_topic])
Target = msg_class(args.target_topic)
InputPC = msg_class(args.input_topic)
while reader.has_next():
    topic, data, t_ns = reader.read_next()
    if topic == args.target_topic:
        if not triggers or (t_ns - triggers[-1]) / 1e9 > args.min_gap:
            triggers.append(t_ns)
    else:
        msg = deserialize_message(data, InputPC)
        inputs.append((t_ns, msg.header.frame_id, cloud_to_xyz(msg)))
print(f"  {len(triggers)} grasp cycles, {len(inputs)} policy_input clouds")
if not triggers:
    raise SystemExit("no policy_target messages found")

# match each trigger to the nearest policy_input at/just before it
cycle_inputs = []
for t_ns in triggers:
    cands = [ip for ip in inputs if ip[0] <= t_ns + int(0.5e9)]
    if not cands:
        raise SystemExit(f"no policy_input before trigger at {t_ns}")
    cycle_inputs.append(min(cands, key=lambda ip: abs(ip[0] - t_ns)))
base_frame = cycle_inputs[0][1]
print(f"  base frame: {base_frame}")

# pass 2: TF tree + candidate clouds around each trigger
print("[pass 2] tf + clouds ...")
tf_buffer = Buffer(cache_time=Duration(seconds=1e6))
TFMsg = msg_class("/tf")
CloudMsg = msg_class(args.cloud_topic)
win_ns = int(args.window * 1e9)
candidates = {i: [] for i in range(len(triggers))}  # i -> [(stamp_ns, raw bytes)]
reader = open_reader(["/tf", "/tf_static", args.cloud_topic])
n_tf = 0
while reader.has_next():
    topic, data, t_ns = reader.read_next()
    if topic == "/tf_static":
        for tr in deserialize_message(data, TFMsg).transforms:
            tf_buffer.set_transform_static(tr, "bag")
    elif topic == "/tf":
        for tr in deserialize_message(data, TFMsg).transforms:
            tf_buffer.set_transform(tr, "bag")
        n_tf += 1
    else:
        for i, trig in enumerate(triggers):
            if trig - win_ns <= t_ns <= trig:
                candidates[i].append((t_ns, data))
print(f"  fed {n_tf} /tf msgs; candidates per cycle: {[len(candidates[i]) for i in range(len(triggers))]}")

# pick per cycle the candidate whose transformed points contain the recorded policy input
os.makedirs(args.out, exist_ok=True)
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
    print("  (no scipy: falling back to brute-force matching on a subsample)")

for i, (in_ns, in_frame, in_pts) in enumerate(cycle_inputs):
    best = None
    for t_ns, data in candidates[i]:
        msg = deserialize_message(data, CloudMsg)
        try:
            tf = tf_buffer.lookup_transform(base_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except Exception as e:
            print(f"  cycle{i+1}: TF lookup failed for cloud at {t_ns}: {e}")
            continue
        R = quat_to_mat(tf.transform.rotation)
        tvec = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
        xyz = cloud_to_xyz(msg) @ R.T + tvec
        probe = in_pts[:: max(1, len(in_pts) // 200)]
        if cKDTree is not None:
            d = cKDTree(xyz).query(probe)[0]
        else:
            d = np.array([np.min(np.linalg.norm(xyz - p, axis=1)) for p in probe])
        med = float(np.median(d))
        if best is None or med < best[0]:
            best = (med, t_ns, xyz, tvec)
    if best is None:
        print(f"  cycle{i+1}: NO candidate cloud transformed - skipped")
        continue
    med, t_ns, xyz, cam_origin = best
    lag = (triggers[i] - t_ns) / 1e9
    out = os.path.join(args.out, f"real_cycle{i+1}_full.npy")
    np.save(out, xyz.astype(np.float32))
    # camera origin in base frame: needed to reproduce the sim depth-range crop at eval time
    np.save(os.path.join(args.out, f"real_cycle{i+1}_cam.npy"), cam_origin.astype(np.float32))
    flag = "OK" if med < 0.01 else "SUSPECT (check TF/calibration)"
    print(f"  cycle{i+1}: {len(xyz)} pts, source cloud {lag:.2f}s before trigger, "
          f"median match to policy_input {med*1000:.1f} mm [{flag}] -> {out}")

print("done")
