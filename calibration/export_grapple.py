#!/usr/bin/env python3
"""Export grapple-cropped point clouds + joint values from a motion bag, per camera,
for model-based calibration. Clouds are put in the camera LEFT-OPTICAL frame and
cropped to a box around the NOMINAL grapple position (FK) so only the grapple remains.

Output: out/grapple_<cam>.npz  with object-array `clouds` (each Nx3, optical frame),
`joints` (dict of per-frame joint-value arrays), and `stamp`.
"""
import argparse, sys
import numpy as np
sys.path.insert(0, '.')
import calib_fk as fk
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
JNAMES = ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork',
          'grapplecarrier', 'grappletong1', 'grappletong2']


def q2R(x, y, z, w):
    n = (x*x+y*y+z*z+w*w)**.5; x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--cam', required=True, choices=['zed_0', 'zed_1', 'lidar_0'])
    ap.add_argument('--step', type=int, default=40, help='use every Nth point cloud')
    ap.add_argument('--half', type=float, default=1.3, help='crop half-extent (m) around grapple')
    ap.add_argument('--voxel', type=float, default=0.01)
    ap.add_argument('--min-pts', type=int, default=400)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    parent = fk.CAM_PARENT[args.cam]
    scfg = fk.SENSORS[args.cam]
    pc_topic = scfg['topic']
    opt_key = ((scfg['cloud_frame'], scfg['cloud_frame'] + '_optical')
               if scfg['is_optical'] else None)

    # joint-state time series + optical edge
    j_t = []; j_v = []; opt_edge = None
    with Reader(args.bag) as r:
        conns = {c.topic: c for c in r.connections}
        for _, _, raw in r.messages(connections=[conns['/tf_static']]):
            m = TS.deserialize_cdr(raw, conns['/tf_static'].msgtype)
            for tr in m.transforms:
                if opt_key is not None and (tr.header.frame_id, tr.child_frame_id) == opt_key:
                    M = np.eye(4); q = tr.transform.rotation; t = tr.transform.translation
                    M[:3, :3] = q2R(q.x, q.y, q.z, q.w); M[:3, 3] = [t.x, t.y, t.z]; opt_edge = M
        for _, tn, raw in r.messages(connections=[conns['/joint_states']]):
            m = TS.deserialize_cdr(raw, conns['/joint_states'].msgtype)
            name2pos = {n: p for n, p in zip(m.name, m.position)}
            j_t.append(tn); j_v.append([name2pos.get(j+'_joint', 0.0) for j in JNAMES])
    j_t = np.array(j_t); j_v = np.array(j_v)
    opt_inv = np.linalg.inv(opt_edge) if opt_key is not None else np.eye(4)   # lidar: cloud already in mount frame

    def joints_at(t):
        i = np.argmin(np.abs(j_t - t))
        return {n: j_v[i][k] for k, n in enumerate(JNAMES)}

    clouds = []; jrec = {n: [] for n in JNAMES}; stamps = []; n_skip = 0
    with Reader(args.bag) as r:
        conns = {c.topic: c for c in r.connections}
        i = 0
        for _, t_ns, raw in r.messages(connections=[conns[pc_topic]]):
            i += 1
            if i % args.step:
                continue
            pc = TS.deserialize_cdr(raw, conns[pc_topic].msgtype)
            offs = {f.name: f.offset for f in pc.fields}; ps = pc.point_step
            arr = np.frombuffer(pc.data, np.uint8).reshape(len(pc.data)//ps, ps)
            def fld(nm): o = offs[nm]; return arr[:, o:o+4].copy().view(np.float32).ravel()
            x, y, z = fld('x'), fld('y'), fld('z')
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            P = np.c_[x[m], y[m], z[m]]
            P = (opt_inv @ np.c_[P, np.ones(len(P))].T).T[:, :3]   # -> optical frame
            jv = joints_at(t_ns)
            # nominal grapplecarrier origin in optical frame
            T_opt_gc = np.linalg.inv(fk.mount_T(args.cam)) @ fk.fk_chain(parent, jv)
            ctr = T_opt_gc[:3, 3]
            sel = np.all(np.abs(P - ctr) < args.half, axis=1)
            Pg = P[sel]
            if len(Pg) < args.min_pts:
                n_skip += 1; continue
            # voxel downsample
            keys = np.floor(Pg / args.voxel).astype(np.int64)
            _, uniq = np.unique(keys, axis=0, return_index=True)
            Pg = Pg[uniq]
            clouds.append(Pg.astype(np.float32)); stamps.append(t_ns)
            for n in JNAMES:
                jrec[n].append(jv[n])

    print(f'[{args.cam}] exported {len(clouds)} grapple frames (skipped {n_skip} sparse), '
          f'median pts={int(np.median([len(c) for c in clouds])) if clouds else 0}')
    np.savez_compressed(args.out,
                        clouds=np.array(clouds, dtype=object),
                        stamp=np.array(stamps, dtype=np.int64),
                        **{f'j_{n}': np.array(jrec[n]) for n in JNAMES})
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
