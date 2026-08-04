#!/usr/bin/env python3
"""Diagnose camera-mount orientation error via the floor plane.

We KNOW the lab floor is horizontal in the base frame. So after transforming a
zed_0 cloud to base with a candidate mount, a plane fit to the floor points must
have a vertical normal. The tilt of that normal == the mount's roll/pitch error.

Usage: diag_floor.py <bag> [mount.npy ...]   (defaults to a set of candidates)
"""
import sys, glob
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import calib_fk as fk

TS = get_typestore(Stores.ROS2_FOXY)
CAM = 'zed_0'
CLOUD_FRAME = 'zed_0_left_camera_frame'
OPT_CHILD = CLOUD_FRAME + '_optical'
TOPIC = f'/{CAM}/zed_node/point_cloud/cloud_registered'


def q2R(x, y, z, w):
    n = np.sqrt(x*x+y*y+z*z+w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tf2T(tr):
    T = np.eye(4); q = tr.transform.rotation; t = tr.transform.translation
    T[:3, :3] = q2R(q.x, q.y, q.z, q.w); T[:3, 3] = [t.x, t.y, t.z]; return T


def fit_plane(P):
    """RANSAC-lite plane to the densest planar set; return (normal, inliers, rms)."""
    best = None
    rng_idx = np.linspace(0, len(P)-1, 300).astype(int)
    for i in rng_idx:
        # local neighborhood seed not needed; sample 3 random-ish via spread
        pass
    # simpler: iterative least-squares with trimming
    c = P.mean(0); Pc = P - c
    n = np.linalg.svd(Pc, full_matrices=False)[2][2]
    for _ in range(10):
        d = Pc @ n
        keep = np.abs(d - np.median(d)) < 0.05      # 5 cm band
        c = P[keep].mean(0); Pc2 = P[keep] - c
        n = np.linalg.svd(Pc2, full_matrices=False)[2][2]
        Pc = P - c
    if n[2] < 0:
        n = -n
    d = (P - c) @ n
    keep = np.abs(d) < 0.05
    rms = np.sqrt((((P[keep]-c) @ n)**2).mean())
    return n, keep.sum(), rms, c


def main():
    bag = sys.argv[1]
    mounts = sys.argv[2:] or sorted(glob.glob('out/calib_zed_0_2026*_mount.npy')) + ['out/calib_zed_0_mount.npy']

    opt_edge = None
    best_cloud = None; densest = 0; best_t = None
    jt = []; jslew = []
    with Reader(bag) as r:
        conns = {c.topic: c for c in r.connections}
        for con, t, raw in r.messages():
            if con.topic in ('/tf', '/tf_static'):
                m = TS.deserialize_cdr(raw, con.msgtype)
                for tr in m.transforms:
                    if (tr.header.frame_id, tr.child_frame_id) == (CLOUD_FRAME, OPT_CHILD):
                        opt_edge = tf2T(tr)
            elif con.topic == '/joint_states':
                m = TS.deserialize_cdr(raw, con.msgtype)
                d = dict(zip(m.name, m.position))
                jt.append(t); jslew.append(d.get('slew_joint', 0.0))
            elif con.topic == TOPIC:
                m = TS.deserialize_cdr(raw, con.msgtype)
                arr = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
                xyz = arr[:, :12].copy().view(np.float32).reshape(-1, 3)
                fin = np.isfinite(xyz).all(1).sum()
                if fin > densest:
                    densest = fin; best_cloud = xyz[np.isfinite(xyz).all(1)].astype(np.float64); best_t = t

    jt = np.array(jt); jslew = np.array(jslew)
    slew = jslew[np.argmin(np.abs(jt - best_t))]
    P = best_cloud
    opt_inv = np.linalg.inv(opt_edge)
    T_base_mast = fk.joint_T('slew', slew)
    print(f'bag densest cloud {densest} pts, slew={slew:.4f}')
    print(f'{"mount":42s} {"trans(m)":24s} {"floorN":22s} tilt°  rms(mm) inl')
    for mp in mounts:
        calib = np.load(mp)
        T = T_base_mast @ calib @ opt_inv
        Pb = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        # floor = lowest 35% in Z
        zc = np.quantile(Pb[:, 2], 0.35)
        floor = Pb[Pb[:, 2] < zc]
        n, inl, rms, c = fit_plane(floor)
        tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        name = mp.split('/')[-1].replace('calib_zed_0_', '').replace('_mount.npy', '')
        print(f'{name:42s} {str(calib[:3,3].round(3)):24s} {str(n.round(3)):22s} {tilt:5.1f}  {rms*1000:6.1f} {inl}')


if __name__ == '__main__':
    main()
