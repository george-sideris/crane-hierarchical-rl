#!/usr/bin/env python3
"""Localize the rack in the crane BASE frame from the two CALIBRATED ZEDs.

Method (from the original draw_rack2.py, now with calibrated extrinsics + new bag):
  1. ArUco gives rough base-frame seeds for the posts (id3/id11 near via zed_0,
     id8 far via zed_1). ArUco 3D is noisy at range -> used ONLY to seed.
  2. The zed_0 stereo point cloud (in base, via the CALIBRATED mount) is the metric
     truth. refine() snaps each seed to the real post: mean XY of nearby tall points
     + 90th-pct Z (post top). This is the vertical-column detector.
  3. 3 real corners (A=id3, B=id11, C=id8-far) -> 4th D = A + (C-B). Box extruded H.
  4. Post-top height from the ArUco markers. Reprojected on the zed_0 image to validate.
"""
import argparse
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
Z0_TOPIC = '/zed_0/zed_node/point_cloud/cloud_registered'
Z0_CF = 'zed_0_left_camera_frame'; Z0_OC = Z0_CF + '_optical'


def q2R(x, y, z, w):
    n = np.sqrt(x*x+y*y+z*z+w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def marker_centers(npz, calib, parentkey):
    d = np.load(npz, allow_pickle=True); ids = d['marker_id'].astype(int)
    out = {}
    for mid in sorted(set(ids)):
        sel = np.where(ids == mid)[0]
        pts = np.array([(d[parentkey][i] @ calib @ d['T_cam_marker'][i])[:3, 3] for i in sel])
        out[mid] = np.median(pts, 0)
    return d, out


def build_zed0_base_cloud(bag, calib, n_accum=12, voxel=0.03):
    """Accumulate zed_0 clouds into the base frame using the calibrated mount."""
    opt_edge = None; T_base_mast = None; clouds = []
    with Reader(bag) as r:
        conns = {c.topic: c for c in r.connections}
        for con, t, raw in r.messages():
            if con.topic in ('/tf', '/tf_static'):
                m = TS.deserialize_cdr(raw, con.msgtype)
                for tr in m.transforms:
                    fc = (tr.header.frame_id, tr.child_frame_id)
                    if fc == (Z0_CF, Z0_OC):
                        T = np.eye(4); qq = tr.transform.rotation; tt = tr.transform.translation
                        T[:3, :3] = q2R(qq.x, qq.y, qq.z, qq.w); T[:3, 3] = [tt.x, tt.y, tt.z]; opt_edge = T
                    elif fc == ('base_link', 'mast'):
                        T = np.eye(4); qq = tr.transform.rotation; tt = tr.transform.translation
                        T[:3, :3] = q2R(qq.x, qq.y, qq.z, qq.w); T[:3, 3] = [tt.x, tt.y, tt.z]; T_base_mast = T
            elif con.topic == Z0_TOPIC and len(clouds) < n_accum:
                m = TS.deserialize_cdr(raw, con.msgtype)
                a = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
                xyz = a[:, :12].copy().view(np.float32).reshape(-1, 3)
                clouds.append(xyz[np.isfinite(xyz).all(1)].astype(np.float64))
    P = np.vstack(clouds)
    T = T_base_mast @ calib @ np.linalg.inv(opt_edge)   # base <- cloud body frame
    Pb = (T @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    keys = np.floor(Pb / voxel).astype(np.int64)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    return Pb[uniq], T_base_mast


def refine(Pb, seed_xy, r=0.4):
    """Snap an XY seed to the real post: mean XY of nearby tall points + 90th-pct Z."""
    near = Pb[(np.linalg.norm(Pb[:, :2] - seed_xy, axis=1) < r) & (Pb[:, 2] > 0.2)]
    if len(near) < 20:
        return np.array([seed_xy[0], seed_xy[1], 1.1]), len(near)
    return np.array([near[:, 0].mean(), near[:, 1].mean(), np.percentile(near[:, 2], 90)]), len(near)


def grab_zed0_image(bag, idx=30):
    topic = '/zed_0/zed_node/rgb/color/rect/image'
    with Reader(bag) as r:
        conns = {c.topic: c for c in r.connections}
        msgs = list(r.messages(connections=[conns[topic]]))
        _, _, raw = msgs[min(idx, len(msgs)-1)]
        m = TS.deserialize_cdr(raw, conns[topic].msgtype)
    buf = np.frombuffer(m.data, np.uint8); ch = buf.size // (m.height*m.width)
    a = buf.reshape(m.height, m.width, ch)
    return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR) if ch == 4 else a.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    # KNOWN rigid rack footprint = the SAME dims the sim uses (crane_rl_env_gaze.py:2207)
    ap.add_argument('--W', type=float, default=2.16, help='rack width  (base X), sim value')
    ap.add_argument('--L', type=float, default=7.38, help='rack length (base Y), sim value')
    ap.add_argument('--H', type=float, default=2.54, help='rack height, sim value')
    args = ap.parse_args()
    C0 = np.load('out/calib_zed_0_mount.npy')   # T_mast<-zed0_optical
    C1 = np.load('out/calib_zed_1_mount.npy')   # T_stick<-zed1_optical

    d0, z0 = marker_centers('out/rack_z0.npz', C0, 'T_base_mast')
    _,  z1 = marker_centers('out/rack_z1.npz', C1, 'T_base_stick')
    seedA, seedB, seedC = z0[3][:2], z0[11][:2], z1[8][:2]   # ArUco seeds (XY)
    ztop = float(np.mean([z0[3][2], z0[11][2]]))

    print('building calibrated zed_0 base cloud ...')
    Pb, T_base_mast = build_zed0_base_cloud(args.bag, C0)
    up = np.array([0, 0, 1.0])
    def horiz(v): v = v - v.dot(up)*up; return v / np.linalg.norm(v)

    # Anchor the KNOWN rigid footprint on the two RELIABLE posts: id11 (near corner) and
    # id8 (far corner, adjacent to id11 -> the L edge). id3 only picks the width sign.
    B, nB = refine(Pb, seedB)   # id11 (near corner)
    F, nF = refine(Pb, seedC)   # id8  (far corner)
    B[2] = F[2] = ztop
    Ldir = horiz(F - B)                                   # rack length axis
    w = np.r_[seedA, ztop] - B
    Wdir = horiz(w - w.dot(Ldir)*Ldir)                    # width axis, toward id3
    c_id11 = B
    c_id3  = B + args.W*Wdir
    c_id8  = B + args.L*Ldir
    c_far  = B + args.L*Ldir + args.W*Wdir                # back corner (rigid, not inferred)
    top = np.array([c_id3, c_id11, c_id8, c_far])
    bot = top.copy(); bot[:, 2] = ztop - args.H
    corners = np.vstack([top, bot])
    print(f'anchored on id11 {B[:2].round(2)}(n{nB}) + id8 {F[:2].round(2)}(n{nF})')
    print(f'KNOWN footprint W={args.W} L={args.L} H={args.H}  (detected id11-id8 = {np.linalg.norm(F-B):.2f}m)')

    center = top.mean(0)
    yaw = np.degrees(np.arctan2(Ldir[0], Ldir[1]))        # length axis from +Y (sim convention)
    Rp = np.eye(4); Rp[:3, 0] = Ldir; Rp[:3, 1] = Wdir; Rp[:3, 2] = up; Rp[:3, 3] = center
    np.save('out/rack_corners_base.npy', corners)
    np.save('out/rack_pose_base.npy', Rp)
    print(f'\nrack center (base) = ({center[0]:.3f}, {center[1]:.3f})  yaw {yaw:.2f} deg')
    print(f'  -> sim:  RACK_BASE_X, RACK_BASE_Y, RACK_BASE_YAW_DEG = {center[0]:.3f}, {center[1]:.3f}, {yaw:.2f}')
    print('  (old/previous: -5.061, 2.527, 1.77)')

    # reproject box + anchor posts onto zed_0 image
    K = d0['K']; Tbo = T_base_mast @ C0
    img = grab_zed0_image(args.bag)
    def proj(P3):
        q = np.linalg.inv(Tbo) @ np.r_[P3, 1]
        return (K @ q[:3])[:2] / (K @ q[:3])[2], q[2] > 0
    Pc = (np.linalg.inv(Tbo) @ np.c_[corners, np.ones(8)].T).T[:, :3]
    uv = (K @ Pc.T).T; px = uv[:, :2]/uv[:, 2:3]; front = Pc[:, 2] > 0
    for a, b in EDGES:
        if front[a] and front[b]:
            cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)), (0, 255, 0), 3)
    for P, col in [(c_id3, (0,0,255)), (B, (255,0,0)), (F, (0,255,255))]:
        uvp, ok = proj(P)
        if ok: cv2.circle(img, tuple(uvp.astype(int)), 10, col, -1)
    cv2.imwrite('out/rack_calib.jpg', cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))

    # top-down with cloud
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sb = (Pb[:, 2] > -1.0) & (Pb[:, 2] < 1.6)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.scatter(Pb[sb, 0], Pb[sb, 1], c='lightgray', s=2)
    ring = np.vstack([top[:, :2], top[0, :2]])
    ax.plot(ring[:, 0], ring[:, 1], 'g-', lw=3)
    for nm, p, c in [('id3', c_id3, 'red'), ('id11', B, 'blue'), ('id8', F, 'orange'), ('back', c_far, 'purple')]:
        ax.scatter([p[0]], [p[1]], c=c, s=160, marker='*', zorder=5, label=nm)
    ax.scatter([0], [0], c='k', marker='+', s=200, label='base')
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f'rack (known {args.W}x{args.L} m) center ({center[0]:.2f},{center[1]:.2f}) yaw {yaw:.1f}')
    plt.savefig('out/rack_topdown.png', dpi=75, bbox_inches='tight')
    print('saved out/rack_calib.jpg, out/rack_topdown.png, out/rack_corners_base.npy, out/rack_pose_base.npy')


if __name__ == '__main__':
    main()
