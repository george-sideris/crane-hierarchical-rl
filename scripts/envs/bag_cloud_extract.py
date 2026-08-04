#!/usr/bin/env python3
"""Extract a raw ZED cloud from a rosbag2 .db3 and put it in base_link - WITHOUT ROS.

Reads the sqlite3 bag directly and decodes the CDR payloads of sensor_msgs/PointCloud2 and
tf2_msgs/TFMessage by hand. Motivation: the recorded policy_debug npz only keeps the TIGHT
action-box crop, so the points a widened crop would have added are not recoverable from it.
This gets the full cloud back so the proposed --crop_margin can be evaluated on real data.

    python3 scripts/envs/bag_cloud_extract.py \
        --bag /media/george/T9/crane_run_20260803_164106 --at_frac 0.9 --out /tmp/real_raw.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import struct

import numpy as np

CLOUD_TOPIC = "/zed_0/zed_node/point_cloud/cloud_registered"


class CDR:
    """Minimal little-endian CDR reader with the alignment rules ROS 2 serialization uses."""

    def __init__(self, buf: bytes):
        self.b = buf
        self.p = 4                      # skip the 4-byte encapsulation header

    def _align(self, n: int):
        pad = (-(self.p - 4)) % n
        self.p += pad

    def u8(self):
        v = self.b[self.p]; self.p += 1; return v

    def u32(self):
        self._align(4)
        v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v

    def i32(self):
        self._align(4)
        v = struct.unpack_from("<i", self.b, self.p)[0]; self.p += 4; return v

    def f64(self):
        self._align(8)
        v = struct.unpack_from("<d", self.b, self.p)[0]; self.p += 8; return v

    def string(self):
        n = self.u32()
        s = self.b[self.p:self.p + n - 1].decode("utf-8", "replace")
        self.p += n
        return s

    def header(self):
        sec, nsec = self.i32(), self.u32()
        return sec + nsec * 1e-9, self.string()

    def bytes(self, n: int):
        v = self.b[self.p:self.p + n]; self.p += n; return v


def decode_pointcloud2(buf: bytes):
    c = CDR(buf)
    stamp, frame = c.header()
    height, width = c.u32(), c.u32()
    nf = c.u32()
    fields = []
    for _ in range(nf):
        name = c.string(); off = c.u32(); dt = c.u8(); cnt = c.u32()
        fields.append((name, off, dt, cnt))
    c.u8()                                   # is_bigendian
    point_step, _row_step = c.u32(), c.u32()
    n = c.u32()
    data = c.bytes(n)
    off = {f[0]: f[1] for f in fields}
    arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, point_step)
    xyz = np.stack([arr[:, off[k]:off[k] + 4].copy().view(np.float32).ravel()
                    for k in ("x", "y", "z")], axis=1)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return stamp, frame, height, width, xyz


def decode_tf(buf: bytes):
    c = CDR(buf)
    out = []
    for _ in range(c.u32()):
        stamp, parent = c.header()
        child = c.string()
        t = np.array([c.f64(), c.f64(), c.f64()])
        q = np.array([c.f64(), c.f64(), c.f64(), c.f64()])      # x, y, z, w
        out.append((stamp, parent, child, t, q))
    return out


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def chain_to_base(tfs, target, base="base_link"):
    """Compose target -> base from the latest transform seen for each parent/child edge."""
    up = {c: (p, t, q) for (_, p, c, t, q) in tfs}        # child -> parent
    R, T, cur, hops = np.eye(3), np.zeros(3), target, []
    while cur != base:
        if cur not in up:
            raise KeyError(f"no tf path to {base}; stuck at '{cur}' (chain so far: {hops})")
        p, t, q = up[cur]
        Rp = quat_to_R(q)
        T = Rp @ T + t
        R = Rp @ R
        hops.append(cur)
        cur = p
        if len(hops) > 32:
            raise RuntimeError("tf cycle")
    return R, T, hops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--at_frac", type=float, default=0.9, help="position in the bag, 0-1")
    ap.add_argument("--at_time", type=float, default=None,
                    help="recorder wall time (s) to seek to; matches decisions.jsonl t_wall")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list_frames", action="store_true")
    a = ap.parse_args()

    db = glob.glob(os.path.join(a.bag, "*.db3"))[0]
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute("select id,name from topics")
    tid = {n: i for i, n in cur.fetchall()}

    cur.execute("select timestamp,data from messages where topic_id=? order by timestamp",
                (tid[CLOUD_TOPIC],))
    clouds = cur.fetchall()
    if a.at_time is not None:
        k = int(np.argmin([abs(t / 1e9 - a.at_time) for t, _ in clouds]))
    else:
        k = min(int(len(clouds) * a.at_frac), len(clouds) - 1)
    ts, blob = clouds[k]
    stamp, frame, h, w, xyz = decode_pointcloud2(blob)
    print(f"[bag] cloud {k}/{len(clouds)} t={ts/1e9:.1f} frame='{frame}' {h}x{w} "
          f"-> {len(xyz)} finite pts")

    # /tf carries the moving crane joints; the camera MOUNT (zed_0_camera_center -> mast etc.)
    # is latched on /tf_static, so both are needed to close the chain to base_link.
    tfs = []
    if "/tf_static" in tid:
        cur.execute("select data from messages where topic_id=?", (tid["/tf_static"],))
        for (d,) in cur.fetchall():
            tfs.extend(decode_tf(d))
    cur.execute("select data from messages where topic_id=? and timestamp<=? "
                "order by timestamp desc limit 4000", (tid["/tf"], ts))
    dyn = []
    for (d,) in cur.fetchall():
        dyn.extend(decode_tf(d))
    tfs.extend(dyn[::-1])                             # oldest first so the latest wins in the dict
    if a.list_frames:
        edges = sorted({(p, c) for (_, p, c, _, _) in tfs})
        print("[bag] tf edges:", *[f"    {p} -> {c}" for p, c in edges], sep="\n")

    R, T, hops = chain_to_base(tfs, frame)
    print(f"[bag] tf chain {frame} -> base_link via {hops}")
    pts = xyz @ R.T + T
    np.savez(a.out, points=pts.astype(np.float32), frame_id=frame, stamp=np.float64(stamp))
    print(f"[bag] wrote {a.out}: {len(pts)} pts in base_link, "
          f"x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
          f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
          f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")


if __name__ == "__main__":
    main()
