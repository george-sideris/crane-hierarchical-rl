#!/usr/bin/env python3
"""Extract raw base-frame clouds at gaze-decision times from the 2026-08-03 trial bags.

One bag open per trial (bag_cloud_extract re-opens per call, which is why this exists).
Resumable: existing outputs are skipped. Writes logs/real_raw_clouds/<tag>_cNN.npz.
"""

from __future__ import annotations

import glob
import json
import os
import sqlite3
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bag_cloud_extract import decode_pointcloud2, decode_tf, chain_to_base  # noqa: E402

RUNS = {
    "logs/crane_policy_debug/run_20260803_155559": ("/media/george/T9/crane_run_20260803_115616", "double"),
    "logs/crane_policy_debug/run_20260803_174017": ("/media/george/T9/crane_run_20260803_134017", "jagged"),
    "logs/crane_policy_debug/run_20260803_204034": ("/media/george/T9/crane_run_20260803_164106", "single"),
}
OUT = "logs/real_raw_clouds"
EVERY = 2                                  # every 2nd covered cycle


def main():
    os.makedirs(OUT, exist_ok=True)
    for run, (bag, tag) in RUNS.items():
        db = glob.glob(bag + "/*.db3")
        if not db:
            print(f"[{tag}] no bag; skipped")
            continue
        con = sqlite3.connect(db[0]); cur = con.cursor()
        cur.execute("select id,name from topics")
        tid = {n: i for i, n in cur.fetchall()}
        ct = tid.get("/zed_0/zed_node/point_cloud/cloud_registered")
        if ct is None:
            continue
        cur.execute("select timestamp from messages where topic_id=? order by timestamp", (ct,))
        stamps = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
        stat = []
        if "/tf_static" in tid:
            cur.execute("select data from messages where topic_id=?", (tid["/tf_static"],))
            for (d,) in cur.fetchall():
                stat.extend(decode_tf(d))

        gaze = [json.loads(l) for l in open(os.path.join(run, "decisions.jsonl"))
                if json.loads(l)["kind"] == "gaze"]
        cover = [g for g in gaze if stamps[0] <= g["t_wall"] * 1e9 <= stamps[-1]]
        print(f"[{tag}] {len(cover)} covered gaze cycles")
        for g in cover[::EVERY]:
            out = os.path.join(OUT, f"{tag}_c{g['cycle']:02d}.npz")
            if os.path.exists(out):
                continue
            ts = int(stamps[np.argmin(np.abs(stamps - g["t_wall"] * 1e9))])
            cur.execute("select data from messages where topic_id=? and timestamp=?", (ct, ts))
            _, frame, _, _, xyz = decode_pointcloud2(cur.fetchone()[0])
            cur.execute("select data from messages where topic_id=? and timestamp<=? "
                        "order by timestamp desc limit 400", (tid["/tf"], ts))
            tfs = list(stat)
            dyn = []
            for (d,) in cur.fetchall():
                dyn.extend(decode_tf(d))
            tfs.extend(dyn[::-1])
            try:
                R, T, _ = chain_to_base(tfs, frame)
            except Exception as e:  # noqa: BLE001
                print(f"  {out}: tf failed ({e})")
                continue
            np.savez(out, points=(xyz @ R.T + T).astype(np.float32),
                     stamp=np.float64(ts / 1e9), cycle=np.int32(g["cycle"]))
            print(f"  wrote {out} ({len(xyz)} pts)")
        con.close()


if __name__ == "__main__":
    main()
