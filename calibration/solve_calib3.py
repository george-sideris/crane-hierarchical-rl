#!/usr/bin/env python3
"""Solve T_parent<-cam for a ZED, in the camera's RIGID parent frame.

zed_0 is bolted to the `mast` (slew rotates the mast), zed_1 to the `stick`. So the
constant transform is T_mast<-cam (zed_0) / T_stick<-cam (zed_1), NOT T_base<-cam.
We express the FK tong origins in the parent frame and solve the constant
parent<-cam transform from ArUco marker CENTERS only (rotation-free), jointly with
each marker's center position m_i in its tong frame. Alternating closed form,
robust rejection, seeded from the nominal static chain parent->cam in the bag.

Constraint per detection of marker i at time t:
    X @ c_cam(t) = T_parent<-tong_i(t) @ m_i,   X = T_parent<-cam (constant)
"""
import argparse
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
OFFNORM = {'LEFT': np.linalg.norm([0.114, 0.172, 0.014]),
           'RIGHT': np.linalg.norm([0.144, 0.106, 0.0])}

CAMCFG = {
    'zed_0': dict(parent='mast', base_chain=['mast'],
                  static_chain=['mast_sensor_bracket', 'lidar_0/os_sensor',
                                'zed_0_camera_link', 'zed_0_camera_center',
                                'zed_0_left_camera_frame', 'zed_0_left_camera_frame_optical']),
    'zed_1': dict(parent='stick', base_chain=['mast', 'mainboom', 'stick'],
                  static_chain=['camera_zed', 'zed_1_camera_link', 'zed_1_camera_center',
                                'zed_1_left_camera_frame', 'zed_1_left_camera_frame_optical']),
}


def q2R(x, y, z, w):
    n = (x*x+y*y+z*z+w*w)**.5; x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tfT(tr):
    M = np.eye(4); q = tr.transform.rotation; t = tr.transform.translation
    M[:3, :3] = q2R(q.x, q.y, q.z, q.w); M[:3, 3] = [t.x, t.y, t.z]; return M


def horn(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src-cs).T @ (dst-cd))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cd - R @ cs; return T


def rpy(R):
    sy = (R[0, 0]**2 + R[1, 0]**2)**.5
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy),
                       np.arctan2(R[1, 0], R[0, 0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--npz', nargs='+', required=True)
    ap.add_argument('--cam', default='zed_0')
    ap.add_argument('--id-left', type=int, default=1)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--reject', type=float, default=3.0)
    args = ap.parse_args()
    cfg = CAMCFG[args.cam]

    # ---- tf pass: T_base<-parent(t) series + nominal static parent<-cam ----
    static = {}; dyn = {}
    with Reader(args.bag) as r:
        conns = {c.topic: c for c in r.connections}
        for top in ['/tf_static', '/tf']:
            for _, tn, raw in r.messages(connections=[conns[top]]):
                m = TS.deserialize_cdr(raw, conns[top].msgtype)
                for tr in m.transforms:
                    k = (tr.header.frame_id, tr.child_frame_id)
                    if top == '/tf_static':
                        static[k] = tfT(tr)
                    else:
                        dyn.setdefault(k, []).append((tn, tfT(tr)))
    for k in dyn:
        dyn[k].sort(key=lambda x: x[0])

    def lookup(parent, child, t):
        if (parent, child) in static:
            return static[(parent, child)]
        lst = dyn[(parent, child)]
        ts = [e[0] for e in lst]
        i = max(0, min(np.searchsorted(ts, t, 'right') - 1, len(lst) - 1))
        return lst[i][1]

    def base_to_parent(t):
        T = np.eye(4); p = 'base_link'
        for c in cfg['base_chain']:
            T = T @ lookup(p, c, t); p = c
        return T

    # nominal parent<-cam (static)
    Xnom = np.eye(4); p = cfg['parent']
    for c in cfg['static_chain']:
        Xnom = Xnom @ static[(p, c)]; p = c

    # ---- load detections ----
    ids, Tcm, Tt1, Tt2, stamp = [], [], [], [], []
    for f in args.npz:
        d = np.load(f, allow_pickle=True)
        ids.append(d['marker_id']); Tcm.append(d['T_cam_marker'])
        Tt1.append(d['T_base_tong1']); Tt2.append(d['T_base_tong2']); stamp.append(d['stamp'])
    ids = np.concatenate(ids).astype(int); stamp = np.concatenate(stamp)
    Tcm = np.concatenate(Tcm); Tt1 = np.concatenate(Tt1); Tt2 = np.concatenate(Tt2)
    c_cam = Tcm[:, :3, 3]
    present = sorted(set(ids)); id_left = args.id_left
    id_right = [p for p in present if p != id_left][0]
    role = {id_left: 'LEFT', id_right: 'RIGHT'}

    # express FULL tong transforms in parent frame at each detection time
    Pt1 = np.empty((len(ids), 4, 4)); Pt2 = np.empty((len(ids), 4, 4))
    for k in range(len(ids)):
        Bp = np.linalg.inv(base_to_parent(int(stamp[k])))
        Pt1[k] = Bp @ Tt1[k]
        Pt2[k] = Bp @ Tt2[k]

    print(f'cam={args.cam} parent={cfg["parent"]} dets={len(ids)} '
          f'LEFT=id{id_left} RIGHT=id{id_right}')
    np.set_printoptions(precision=4, suppress=True)
    print('nominal parent<-cam translation:', Xnom[:3, 3].round(4))

    is_left = np.array([role[i] == 'LEFT' for i in ids])
    cc1 = np.c_[c_cam, np.ones(len(ids))]   # homogeneous marker centers
    best = None
    for left_on_t1 in (True, False):
        Pt = np.where((is_left[:, None, None] == left_on_t1), Pt1, Pt2)  # parent<-tong per det
        Ptinv = np.linalg.inv(Pt)
        keep = np.ones(len(ids), bool); X = Xnom.copy()
        for it in range(args.iters):
            cen_par = (X @ cc1.T).T            # marker centers in parent (via X), homog
            # m_i = marker center in its TONG frame (constant): avg over inlier dets
            cen_tong = np.einsum('nij,nj->ni', Ptinv, cen_par)[:, :3]
            m = {i: cen_tong[keep & (ids == i)].mean(0) for i in present}
            mh = np.array([np.append(m[ids[k]], 1.0) for k in range(len(ids))])
            dst = np.einsum('nij,nj->ni', Pt, mh)[:, :3]   # predicted centers in parent
            X = horn(c_cam[keep], dst[keep])
            pred = (X @ cc1.T).T[:, :3]
            res = np.linalg.norm(pred - dst, axis=1)
            md = np.median(res[keep]); mad = np.median(np.abs(res[keep]-md))+1e-9
            keep = res < md + args.reject*1.4826*mad
        rms = (res[keep]**2).mean()**.5
        if best is None or rms < best[0]:
            best = (rms, X.copy(), {i: m[i].copy() for i in present}, keep.copy(),
                    res.copy(), left_on_t1)

    rms, X, m, keep, res, lt1 = best
    print(f'\nLEFT(id{id_left}) rides grappletong{1 if lt1 else 2}; '
          f'inliers={keep.sum()}/{len(keep)} RMS={rms*1000:.1f}mm '
          f'med={np.median(res[keep])*1000:.1f} p95={np.percentile(res[keep],95)*1000:.1f}mm')
    print('marker-center |offset| solved vs measured:')
    for i in present:
        print(f'  id{i}({role[i]}): |m|={np.linalg.norm(m[i])*1000:.1f}mm '
              f'measured={OFFNORM[role[i]]*1000:.1f}mm '
              f'diff={abs(np.linalg.norm(m[i])-OFFNORM[role[i]])*1000:.1f}mm')
    print(f'\n=== T_{cfg["parent"]} <- {args.cam} (optical) ===')
    print(X)
    print(f'  translation (m): {X[:3,3].round(4)}   rpy ZYX(deg): {rpy(X[:3,:3]).round(2)}')
    print(f'  vs nominal trans diff (m): {(X[:3,3]-Xnom[:3,3]).round(4)}')
    np.save(f'out/T_{cfg["parent"]}_{args.cam}.npy', X)


if __name__ == '__main__':
    main()
