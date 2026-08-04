#!/usr/bin/env python3
"""Extract ArUco marker poses + crane FK per frame from a ROS2 calibration bag.

For each sampled camera image we:
  - detect the requested marker ids and estimate T_cam_optical <- marker (ArUco)
  - look up the live FK transforms T_base <- grappletong1 and T_base <- grappletong2
    (composed from /tf + /tf_static at the image timestamp)
and dump everything to an npz for the solver. No point clouds are read.

NOTE on marker ids: only 4 physical 6x6 markers exist and the ids are REUSED by
context. In the cam-calibration (grapple) bags, ids 1 and 8 are on the grapple
tongs (left/right) and 3, 11 are on the rack. In the rack bags those same ids are
rack-pole markers. So --ids must be set per bag set; there is no global id->object.
We also warn if the same id is detected twice in one frame (grapple + rack in view).

Marker poses are in the camera left-optical frame (camera_info frame_id).
"""
import argparse, sys
import numpy as np
import cv2
import cv2.aruco as aruco
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)

# base_link -> frame chains. Each edge looked up from /tf(_static) at frame time.
CHAIN_T1 = ['mast', 'mainboom', 'stick', 'telescope', 'upperpassive',
            'lowerpassive', 'grapplecarrier', 'grappletong1']
CHAIN_T2 = ['mast', 'mainboom', 'stick', 'telescope', 'upperpassive',
            'lowerpassive', 'grapplecarrier', 'grappletong2']
CHAIN_MAST = ['mast']                        # zed_0 rigid parent
CHAIN_STICK = ['mast', 'mainboom', 'stick']  # zed_1 rigid parent
ROOT = 'base_link'


def quat_to_R(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def tf_to_T(tr):
    T = np.eye(4)
    q = tr.transform.rotation
    t = tr.transform.translation
    T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def bgr(m):
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
    return a if m.encoding == 'bgr8' else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--cam', default='zed_0')
    ap.add_argument('--ids', default='1,3,8,11',
                    help='comma list of marker ids to extract. Default = all 4. '
                         'Grapple vs rack is decided downstream by geometry, '
                         'NOT by id (ids are reused across grapple/rack).')
    ap.add_argument('--marker-len', type=float, default=0.13)
    ap.add_argument('--step', type=int, default=4, help='use every Nth image')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    want_ids = {int(x) for x in args.ids.split(',') if x.strip() != ''}
    img_topic = f'/{args.cam}/zed_node/rgb/color/rect/image'
    info_topic = f'/{args.cam}/zed_node/rgb/color/rect/camera_info'
    adict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    params = aruco.DetectorParameters_create()

    # ---- pass 1: collect tf edges as time series + intrinsics ----
    edges = {}          # (parent, child) -> list[(t_ns, 4x4)]
    static_edges = {}
    K = None; dist = None; cam_frame = None
    with Reader(args.bag) as r:
        conns = {c.topic: c for c in r.connections}
        for topic in ['/tf_static', '/tf', info_topic]:
            if topic not in conns:
                continue
            for _, t_ns, raw in r.messages(connections=[conns[topic]]):
                msg = TS.deserialize_cdr(raw, conns[topic].msgtype)
                if topic == info_topic:
                    if K is None:
                        K = np.array(msg.k).reshape(3, 3)
                        dist = np.array(msg.d, float)
                        if dist.size < 4:
                            dist = np.zeros(5)
                        cam_frame = msg.header.frame_id
                    continue
                for tr in msg.transforms:
                    key = (tr.header.frame_id, tr.child_frame_id)
                    T = tf_to_T(tr)
                    if topic == '/tf_static':
                        static_edges[key] = T
                    else:
                        edges.setdefault(key, []).append((t_ns, T))
        for k in edges:
            edges[k].sort(key=lambda x: x[0])

    if K is None:
        print(f'ERROR: no camera_info on {info_topic}', file=sys.stderr); sys.exit(1)
    print(f'[{args.cam}] fx={K[0,0]:.1f} cam_frame={cam_frame} '
          f'dyn_edges={len(edges)} static_edges={len(static_edges)}', file=sys.stderr)

    def lookup(parent, child, t_ns):
        key = (parent, child)
        if key in static_edges:
            return static_edges[key]
        lst = edges.get(key)
        if lst is None:
            raise KeyError(key)
        ts_arr = [e[0] for e in lst]
        i = max(0, min(np.searchsorted(ts_arr, t_ns, side='right') - 1, len(lst) - 1))
        return lst[i][1]

    def fk(chain, t_ns):
        T = np.eye(4); parent = ROOT
        for child in chain:
            T = T @ lookup(parent, child, t_ns)
            parent = child
        return T

    # ---- pass 2: images ----
    recs = []
    n_img = 0; n_det = 0; n_dup = 0; n_nofk = 0
    with Reader(args.bag) as r:
        conns = {c.topic: c for c in r.connections}
        i = 0
        for _, t_ns, raw in r.messages(connections=[conns[img_topic]]):
            i += 1
            if i % args.step:
                continue
            m = TS.deserialize_cdr(raw, conns[img_topic].msgtype)
            n_img += 1
            g = cv2.cvtColor(bgr(m), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(g, adict, parameters=params)
            if ids is None:
                continue
            ids = ids.flatten()
            sel = [j for j, mid in enumerate(ids) if int(mid) in want_ids]
            if not sel:
                continue
            # duplicate-id guard: same id detected twice (e.g. grapple + rack both visible)
            sel_ids = [int(ids[j]) for j in sel]
            dup = {x for x in sel_ids if sel_ids.count(x) > 1}
            if dup:
                n_dup += 1
                sel = [j for j in sel if int(ids[j]) not in dup]
                if not sel:
                    continue
            try:
                T_b_t1 = fk(CHAIN_T1, t_ns)
                T_b_t2 = fk(CHAIN_T2, t_ns)
                T_b_mast = fk(CHAIN_MAST, t_ns)
                T_b_stick = fk(CHAIN_STICK, t_ns)
            except KeyError:
                n_nofk += 1
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                [corners[j] for j in sel], args.marker_len, K, dist)
            for k, j in enumerate(sel):
                R, _ = cv2.Rodrigues(rvecs[k])
                T_cm = np.eye(4); T_cm[:3, :3] = R; T_cm[:3, 3] = tvecs[k].ravel()
                recs.append((t_ns, int(ids[j]), T_cm, T_b_t1, T_b_t2,
                             T_b_mast, T_b_stick, corners[j].reshape(4, 2)))
                n_det += 1

    print(f'[{args.cam}] sampled={n_img} detections={n_det} '
          f'dup_frames={n_dup} fk_miss={n_nofk}', file=sys.stderr)
    np.savez_compressed(
        args.out,
        cam=args.cam, cam_frame=cam_frame, K=K, dist=dist, marker_len=args.marker_len,
        stamp=np.array([r_[0] for r_ in recs], dtype=np.int64),
        marker_id=np.array([r_[1] for r_ in recs]),
        T_cam_marker=np.array([r_[2] for r_ in recs]) if recs else np.zeros((0, 4, 4)),
        T_base_tong1=np.array([r_[3] for r_ in recs]) if recs else np.zeros((0, 4, 4)),
        T_base_tong2=np.array([r_[4] for r_ in recs]) if recs else np.zeros((0, 4, 4)),
        T_base_mast=np.array([r_[5] for r_ in recs]) if recs else np.zeros((0, 4, 4)),
        T_base_stick=np.array([r_[6] for r_ in recs]) if recs else np.zeros((0, 4, 4)),
        corners=np.array([r_[7] for r_ in recs]) if recs else np.zeros((0, 4, 2)),
    )
    print(f'saved {len(recs)} detections -> {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
