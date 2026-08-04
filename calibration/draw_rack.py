#!/usr/bin/env python3
"""Build the rack action-bound box from the two solid basemast corners (id3,id11)
and the known footprint L x W + height, then render it as a green wireframe both
reprojected on the zed_0 image and in the base frame (top-down).

id3-id11 is one adjacent edge (its length ~ one of L/W). The perpendicular edge is
the other dimension, taken horizontally toward the rack interior (the log pile).
Box top = post tops (marker z); box extruded H downwards to the ground.
"""
import argparse
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
BAG = '/media/george/T9/rosbag2_cranelab_calibration_2026_06_17-17_39_00'


def grab_zed0_image(idx=70):
    topic = '/zed_0/zed_node/rgb/color/rect/image'
    with Reader(BAG) as r:
        conns = {c.topic: c for c in r.connections}
        _, _, raw = list(r.messages(connections=[conns[topic]]))[idx]
        m = TS.deserialize_cdr(raw, conns[topic].msgtype)
    buf = np.frombuffer(m.data, np.uint8); ch = buf.size // (m.height * m.width)
    a = buf.reshape(m.height, m.width, ch)
    if ch == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR if m.encoding.startswith('bgra') else cv2.COLOR_RGBA2BGR)
    return a if m.encoding == 'bgr8' else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, required=True, help='rack dimension along id3-id11 edge (m)')
    ap.add_argument('--W', type=float, required=True, help='perpendicular rack dimension (m)')
    ap.add_argument('--H', type=float, default=2.54, help='rack height (m), default 100in')
    args = ap.parse_args()

    d0 = np.load('out/all_static_zed_0.npz', allow_pickle=True)
    K = d0['K']
    Tb_z0 = d0['T_base_mast'][0] @ np.load('out/nom_mast_zed_0.npy')  # base<-optical
    Tz0_b = np.linalg.inv(Tb_z0)
    mk = np.load('out/markers_base.npy', allow_pickle=True).item()
    c3, c11 = mk['id3'], mk['id11']
    Pb = np.load('out/pcd_base.npy')

    # ground-plane edge id3->id11 and its horizontal perpendicular
    u = c11 - c3; u[2] = 0; Ledge = np.linalg.norm(u); u /= Ledge
    perp = np.array([-u[1], u[0], 0.0])
    # pick perpendicular sign toward the log pile (cloud mass near the markers)
    near = Pb[(np.linalg.norm(Pb[:, :2] - ((c3[:2]+c11[:2])/2), axis=1) < 4) & (Pb[:, 2] > -1) & (Pb[:, 2] < 1)]
    if len(near) and np.dot(near[:, :3].mean(0) - c3, perp) < 0:
        perp = -perp
    print(f'measured id3-id11 edge = {Ledge:.3f} m  (you gave L={args.L} along this edge)')

    # 4 top corners. Anchor the id3-id11 edge on the markers' midpoint (splits the
    # ArUco depth error) and use the TRUE edge length args.L, not the noisy 2.35 m.
    mid = (c3 + c11) / 2.0
    t0 = mid - 0.5 * args.L * u
    t1 = mid + 0.5 * args.L * u
    t2 = t1 + args.W * perp
    t3 = t0 + args.W * perp
    top = np.array([t0, t1, t2, t3])
    bot = top.copy(); bot[:, 2] -= args.H
    corners = np.vstack([top, bot])  # 0-3 top, 4-7 bottom
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    # reproject onto zed_0 image
    img = grab_zed0_image()
    cam = (Tz0_b @ np.c_[corners, np.ones(8)].T).T[:, :3]
    uv = (K @ cam.T).T
    px = uv[:, :2] / uv[:, 2:3]
    infront = cam[:, 2] > 0
    GREEN = (0, 255, 0)
    for a, b in edges:
        if infront[a] and infront[b]:
            cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)), GREEN, 3)
    for nm, p in [('id3', c3), ('id11', c11)]:
        q = (Tz0_b @ np.append(p, 1))[:3]
        if q[2] > 0:
            uvp = K @ q; uvp = (uvp[:2]/uvp[2]).astype(int)
            cv2.circle(img, tuple(uvp), 10, (0, 0, 255), -1)
            cv2.putText(img, nm, tuple(uvp+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    out = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.imwrite('out/rack_wireframe.jpg', out)
    print('saved out/rack_wireframe.jpg')

    # base-frame top-down with box
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sb = (Pb[:, 2] > -1.5) & (Pb[:, 2] < 2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(Pb[sb, 0], Pb[sb, 1], c=Pb[sb, 2], s=2, cmap='gray', alpha=.4)
    ring = np.vstack([top[:, :2], top[0, :2]])
    ax.plot(ring[:, 0], ring[:, 1], 'g-', lw=3, label='rack bounds')
    for nm, p, col in [('id3', c3, 'red'), ('id11', c11, 'orange')]:
        ax.scatter([p[0]], [p[1]], c=col, s=200, marker='*', zorder=5, label=nm)
    ax.scatter([Tb_z0[0,3]], [Tb_z0[1,3]], c='magenta', marker='^', s=150, label='zed_0')
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(); ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)')
    ax.set_title(f'rack action bounds {args.L}x{args.W}x{args.H} m (base frame)')
    plt.savefig('out/rack_topdown.png', dpi=70, bbox_inches='tight')
    print('saved out/rack_topdown.png')
    np.save('out/rack_corners_base.npy', corners)
    print('rack corners (base):\n', corners.round(3))


if __name__ == '__main__':
    main()
