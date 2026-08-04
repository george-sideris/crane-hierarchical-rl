#!/usr/bin/env python3
"""Rack action-bound box from the 17:39 basemast (zed_0) bag, built in the camera's
own optical frame so it never depends on the uncalibrated basemast extrinsic.

Method (see notes at bottom):
  1. id3, id11 ArUco centers (median over static frames) = the two near post tops.
  2. zed_0 point cloud in the optical frame (used ONLY to fit the ground plane).
  3. ground plane via RANSAC -> vertical normal n (no extrinsic needed).
  4. W edge = id3->id11 projected into the ground plane; L = n x W_dir (toward pile).
  5. rectangle: short edge centered on the marker midpoint at the TRUE W, extended L,
     post tops at marker height, extruded H downward along -n.
  6. reproject with K only -> green wireframe.

Outputs (in crane_testbed/calibration/out/):
  rack_camframe.jpg          wireframe reprojected on the zed_0 image
  rack_corners_optical.npy   8 corners in zed_0_left_camera_frame_optical (exact)
  rack_corners_base.npy      8 corners in crane base_link (via NOMINAL extrinsic; see caveat)
  rack_corners.txt           readable dump of both, plus provenance

CAVEAT: the optical-frame box is metric and well-aligned to the image. The base-frame
version rides the NOMINAL T_base<-zed_0 (uncalibrated lidar-bracket guess), so its
absolute placement in the crane frame is ~10-20cm / few-deg off until the basemast
cam extrinsic is properly calibrated. Shape and size are validated; absolute pose is not.
"""
import argparse
import numpy as np
import cv2
import open3d as o3d
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
BAG = '/media/george/T9/rosbag2_cranelab_calibration_2026_06_17-17_39_00'
IMG_TOPIC = '/zed_0/zed_node/rgb/color/rect/image'
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def grab_image(idx=70):
    with Reader(BAG) as r:
        conns = {c.topic: c for c in r.connections}
        _, _, raw = list(r.messages(connections=[conns[IMG_TOPIC]]))[idx]
        m = TS.deserialize_cdr(raw, conns[IMG_TOPIC].msgtype)
    buf = np.frombuffer(m.data, np.uint8); ch = buf.size // (m.height * m.width)
    a = buf.reshape(m.height, m.width, ch)
    return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR) if ch == 4 else a.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, default=7.19)
    ap.add_argument('--W', type=float, default=2.06)
    ap.add_argument('--H', type=float, default=2.54)
    args = ap.parse_args()
    L, W, H = args.L, args.W, args.H

    d0 = np.load('out/all_static_zed_0.npz', allow_pickle=True)
    K = d0['K']
    Tb_z0o = d0['T_base_mast'][0] @ np.load('out/nom_mast_zed_0.npy')   # base <- optical (nominal)

    # point cloud in the optical frame (round-trip cancels the extrinsic exactly)
    Pb = np.load('out/pcd_base.npy')
    Po = (np.linalg.inv(Tb_z0o) @ np.c_[Pb, np.ones(len(Pb))].T).T[:, :3]

    # id3, id11 marker centers in optical (= near post tops)
    ids = d0['marker_id'].astype(int)
    def center(mid):
        return np.median(d0['T_cam_marker'][ids == mid][:, :3, 3], axis=0)
    A, B = center(3), center(11)

    # ground plane -> vertical normal, oriented up toward the markers
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(Po)
    plane, inl = pc.segment_plane(0.05, 3, 1000)
    n = np.array(plane[:3]); n /= np.linalg.norm(n)
    if np.dot(A - Po[inl].mean(0), n) < 0:
        n = -n

    # W direction (near edge, in ground plane); L perpendicular to it, toward the pile
    w = (B - A); w = w - np.dot(w, n) * n; wdir = w / np.linalg.norm(w)
    Ldir = np.cross(n, wdir); Ldir /= np.linalg.norm(Ldir)
    mid = (A + B) / 2.0
    near = Po[np.linalg.norm(Po - mid, axis=1) < 5]
    if np.dot(near.mean(0) - mid, Ldir) < 0:
        Ldir = -Ldir

    # rectangle: short edge centered on midpoint at TRUE W, extend L, extrude H down
    c0 = mid - 0.5 * W * wdir
    c1 = mid + 0.5 * W * wdir
    top = np.array([c0, c1, c1 + L * Ldir, c0 + L * Ldir])
    bot = top + H * (-n)
    corners_opt = np.vstack([top, bot])                 # optical frame
    corners_base = (Tb_z0o @ np.c_[corners_opt, np.ones(8)].T).T[:, :3]   # base (nominal)

    # reproject onto image (intrinsics only)
    img = grab_image()
    uv = (K @ corners_opt.T).T; px = uv[:, :2] / uv[:, 2:3]; front = corners_opt[:, 2] > 0
    for a, b in EDGES:
        if front[a] and front[b]:
            cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)), (0, 255, 0), 3)
    for p in (A, B):
        q = K @ p; q = (q[:2] / q[2]).astype(int)
        cv2.circle(img, tuple(q), 9, (0, 0, 255), -1)
    cv2.imwrite('out/rack_camframe.jpg', cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))

    np.save('out/rack_corners_optical.npy', corners_opt)
    np.save('out/rack_corners_base.npy', corners_base)
    lbl = ['top_NL', 'top_NR', 'top_FR', 'top_FL', 'bot_NL', 'bot_NR', 'bot_FR', 'bot_FL']
    with open('out/rack_corners.txt', 'w') as f:
        f.write(f'# Rack action-bound box  L={L} W={W} H={H} m\n')
        f.write('# Source: 17:39 basemast bag (zed_0), camera-frame method (rack_camframe.py)\n')
        f.write(f'# ground normal (optical) = {n.round(4).tolist()}\n')
        f.write('# CAVEAT: base-frame coords use the NOMINAL zed_0 extrinsic (absolute pose ~10-20cm off)\n\n')
        f.write('corner        optical_x optical_y optical_z      base_x   base_y   base_z\n')
        for i, name in enumerate(lbl):
            o = corners_opt[i]; b = corners_base[i]
            f.write(f'{name:9s}  {o[0]:9.3f}{o[1]:9.3f}{o[2]:9.3f}   {b[0]:8.3f}{b[1]:8.3f}{b[2]:8.3f}\n')

    print(f'W edge (markers, depth-inflated) = {np.linalg.norm(B-A):.3f} m; box uses true W={W}')
    print('saved: out/rack_camframe.jpg, rack_corners_optical.npy, rack_corners_base.npy, rack_corners.txt')
    print('\nbase-frame corners (nominal extrinsic):')
    for name, b in zip(lbl, corners_base):
        print(f'  {name:9s} {b.round(3)}')


if __name__ == '__main__':
    main()
