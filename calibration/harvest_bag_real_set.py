#!/usr/bin/env python3
"""Harvest MANY real policy-input clouds per grasp cycle from a crane run bag.

During each gaze hold the ZED streams ~9 Hz while the pile state is frozen, so every frame in
the hold window is an independently-noised observation of the same state: natural sensor-noise
augmentation. This extracts every such frame, applies the node's exact input pipeline (base
transform -> action-box crop -> pole filters -> FPS to 1024), and writes a dataset dir with
one label per CYCLE shared by all its frames.

Labels per cycle come from a JSON spec (see --labels) so executed-success labels, geometric
labels, and shifted thin-pile labels can be mixed, matching build_real_bc_set.py conventions.

Run inside a ros2 humble container (throwaway pattern) with the bag reachable.
"""
import argparse
import json
import os

import numpy as np

BMIN = np.array([-5.364, -1.684, -1.30])
BMAX = np.array([-3.364, 5.316, 0.10])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--labels", required=True,
                    help="JSON: list of {cycle: int, label: [x,y,z,yaw]} (cycles are 1-based, "
                         "in policy_target order; omit a cycle to skip it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=3.5, help="seconds before each trigger")
    ap.add_argument("--min_gap", type=float, default=5.0)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--max_frames", type=int, default=40, help="cap frames per cycle")
    return ap.parse_args()


def main():
    args = parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rclpy.time import Time
    from rclpy.duration import Duration
    from rosidl_runtime_py.utilities import get_message
    from tf2_ros.buffer import Buffer

    spec = {int(e["cycle"]): (e if "t_wall" in e else e["label"])
            for e in json.load(open(args.labels))}

    def reader_for(topics):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
               rosbag2_py.ConverterOptions("", ""))
        r.set_filter(rosbag2_py.StorageFilter(topics=topics))
        return r

    r0 = rosbag2_py.SequentialReader()
    r0.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", ""))
    TYPES = {t.name: t.type for t in r0.get_all_topics_and_types()}
    CLOUD = "/zed_0/zed_node/point_cloud/cloud_registered"

    # cycle triggers: policy_target msgs, or (older bags) the HOVER_UP fsm transition per cycle,
    # which fires right after the capture+decision - the gaze hold is the window before it
    triggers = []
    if "/crane/policy_target" in TYPES:
        r = reader_for(["/crane/policy_target"])
        while r.has_next():
            _, _, t = r.read_next()
            if not triggers or (t - triggers[-1]) / 1e9 > args.min_gap:
                triggers.append(t)
    else:
        String = get_message(TYPES["/crane/fsm_state"])
        r = reader_for(["/crane/fsm_state"])
        last = None
        while r.has_next():
            _, data, t = r.read_next()
            s = deserialize_message(data, String).data
            if s != last and s.startswith("HOVER_UP"):
                if not triggers or (t - triggers[-1]) / 1e9 > args.min_gap:
                    triggers.append(t)
            last = s
        print("(no policy_target topic: using HOVER_UP fsm transitions as triggers)")
    print(f"{len(triggers)} cycles in bag; labels supplied for {sorted(spec)}")

    # tf + clouds
    TFMsg = get_message(TYPES["/tf"])
    Cloud = get_message(TYPES[CLOUD])
    buf = Buffer(cache_time=Duration(seconds=1e6))
    win = int(args.window * 1e9)
    cands = {i: [] for i in range(len(triggers))}
    r = reader_for(["/tf", "/tf_static", CLOUD])
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/tf_static":
            for tr in deserialize_message(data, TFMsg).transforms:
                buf.set_transform_static(tr, "bag")
        elif topic == "/tf":
            for tr in deserialize_message(data, TFMsg).transforms:
                buf.set_transform(tr, "bag")
        else:
            for i, trig in enumerate(triggers):
                if trig - win <= t <= trig:
                    cands[i].append((t, data))

    def xyz(msg):
        a = np.frombuffer(bytes(msg.data), np.float32).reshape(-1, msg.point_step // 4)[:, :3]
        return a[np.isfinite(a).all(axis=1)]

    def quatmat(q):
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def fps(P, k, seed):
        n = len(P)
        if n <= k:
            return np.vstack([P, np.zeros((k - n, 3), np.float32)])
        rng = np.random.default_rng(seed)
        if n > k * 6:
            P = P[rng.choice(n, k * 6, replace=False)]
            n = len(P)
        idx = np.zeros(k, dtype=int)
        dist = np.full(n, np.inf)
        cur = int(rng.integers(n))
        for i in range(k):
            idx[i] = cur
            d = np.linalg.norm(P - P[cur], axis=1)
            dist = np.minimum(dist, d)
            cur = int(np.argmax(dist))
        return P[idx]

    def encode(x, y, z, yaw):
        n = np.clip(2.0 * (np.array([x, y, z]) - BMIN) / (BMAX - BMIN) - 1.0, -0.999, 0.999)
        t = np.concatenate([n, [np.cos(2 * yaw), np.sin(2 * yaw)]])
        return np.arctanh(np.clip(t, -0.999, 0.999)).astype(np.float32)

    # map bag triggers to labeled cycles: by t_wall when the spec provides it (robust to fsm
    # messages lost in recorder dropouts, which silently shift ordinal numbering), else ordinal
    spec_tw = {c: e for c, e in spec.items() if isinstance(e, dict) and "t_wall" in e}
    def cycle_for(i, t_ns):
        if spec_tw:
            tw = t_ns / 1e9
            best = min(spec_tw, key=lambda c: abs(spec_tw[c]["t_wall"] - tw))
            if abs(spec_tw[best]["t_wall"] - tw) < 5.0:
                return best, spec_tw[best]["label"]
            return None, None
        c = i + 1
        return (c, spec[c]) if c in spec else (None, None)

    clouds, actions, prov = [], [], []
    for i in range(len(triggers)):
        cyc, lab = cycle_for(i, triggers[i])
        if cyc is None:
            continue
        act = encode(*lab)
        kept = 0
        for j, (t, data) in enumerate(cands[i][:args.max_frames]):
            msg = deserialize_message(data, Cloud)
            try:
                tf = buf.lookup_transform("base_link", msg.header.frame_id,
                                          Time.from_msg(msg.header.stamp))
            except Exception:
                continue
            R = quatmat(tf.transform.rotation)
            tr = tf.transform.translation
            p = xyz(msg) @ R.T + np.array([tr.x, tr.y, tr.z])
            m = np.all((p >= BMIN) & (p <= BMAX), axis=1)
            m &= ~((p[:, 0] > -3.6) & (p[:, 1] < -1.35))
            m &= ~((p[:, 0] > -3.8) & (p[:, 1] > 5.1))
            p = p[m].astype(np.float32)
            if len(p) < 200:
                continue
            clouds.append(fps(p, args.num_points, seed=cyc * 1000 + j))
            actions.append(act)
            kept += 1
        prov.append({"cycle": cyc, "frames": kept, "label": lab})
        print(f"  cycle {cyc}: {kept} frames")

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "pointclouds.npy"), np.stack(clouds))
    np.save(os.path.join(args.out, "actions.npy"), np.stack(actions))
    np.save(os.path.join(args.out, "episode_rewards.npy"), np.zeros(len(clouds)))
    md = {"num_points": args.num_points, "obs_dim": args.num_points * 3, "action_dim": 5,
          "num_samples": len(clouds), "gaze": True, "raw_pcd": True, "crop_to_bounds": True,
          "real_data": True, "bag": args.bag, "cycles": prov}
    json.dump(md, open(os.path.join(args.out, "metadata.json"), "w"), indent=2)
    print(f"\nwrote {len(clouds)} clouds ({len(prov)} cycles) -> {args.out}")


if __name__ == "__main__":
    main()
