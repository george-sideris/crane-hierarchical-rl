#!/usr/bin/env python3
"""Rebuild the rack box from 3 point-cloud-detected posts (data-driven orientation).

Posts (ground-plane, base frame) seeds from the vertical-column detector:
  A = id3 post, B = id11 post (short W edge), C = right post adjacent to id11 (long L edge).
Each seed is refined to the centroid of nearby tall cloud points; the 4th corner D is
inferred as A + (C - B) (parallelogram from the 3 real corners). Box extruded H down.
Renders green wireframe on the zed_0 image + base-frame top-down, and reports edge
lengths vs the known L/W as a calibration check.
"""
import numpy as np
import cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
BAG = '/media/george/T9/rosbag2_cranelab_calibration_2026_06_17-17_39_00'
H = 2.54
SEEDS = {'A_id3': (-5.85, -1.42), 'B_id11': (-3.81, -1.29), 'C_right': (-4.50, 5.88)}


def grab_img(idx=70):
    topic = '/zed_0/zed_node/rgb/color/rect/image'
    with Reader(BAG) as r:
        conns = {c.topic: c for c in r.connections}
        _, _, raw = list(r.messages(connections=[conns[topic]]))[idx]
        m = TS.deserialize_cdr(raw, conns[topic].msgtype)
    buf = np.frombuffer(m.data, np.uint8); ch = buf.size // (m.height * m.width)
    a = buf.reshape(m.height, m.width, ch)
    return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR) if ch == 4 else a


def refine(Pb, seed, r=0.35):
    near = Pb[(np.linalg.norm(Pb[:, :2] - seed, axis=1) < r) & (Pb[:, 2] > 0.0)]
    xy = near[:, :2].mean(0) if len(near) else np.array(seed)
    ztop = np.percentile(near[:, 2], 90) if len(near) else 1.4
    return np.array([xy[0], xy[1], ztop])


def main():
    d0 = np.load('out/all_static_zed_0.npz', allow_pickle=True)
    K = d0['K']
    Tb_z0 = d0['T_base_mast'][0] @ np.load('out/nom_mast_zed_0.npy')
    Tz0_b = np.linalg.inv(Tb_z0)
    Pb = np.load('out/pcd_base.npy')

    A = refine(Pb, SEEDS['A_id3'])
    B = refine(Pb, SEEDS['B_id11'])
    C = refine(Pb, SEEDS['C_right'])
    D = A + (C - B)
    top = np.array([A, B, C, D])
    # anchor the top plane to the marker post-tops (reliable), not the hedge behind
    mk = np.load('out/markers_base.npy', allow_pickle=True).item()
    ztop = float(np.mean([mk['id3'][2], mk['id11'][2]]))
    top[:, 2] = ztop                       # flatten the top to a common height
    bot = top.copy(); bot[:, 2] = ztop - H
    corners = np.vstack([top, bot])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    W = np.linalg.norm(A[:2]-B[:2]); L = np.linalg.norm(B[:2]-C[:2])
    ang = np.degrees(np.arccos(np.dot((A[:2]-B[:2])/W, (C[:2]-B[:2])/L)))
    print(f'detected posts: A_id3={A[:2].round(2)} B_id11={B[:2].round(2)} C_right={C[:2].round(2)}')
    print(f'edge W(id3-id11)={W:.3f}m (true 2.06)  L(id11-right)={L:.3f}m (true 7.19)  angle={ang:.1f}deg')
    print(f'top z={ztop:.2f} bottom z={ztop-H:.2f} (H={H})')

    img = grab_img()
    cam = (Tz0_b @ np.c_[corners, np.ones(8)].T).T[:, :3]
    uv = (K @ cam.T).T; px = uv[:, :2] / uv[:, 2:3]; infront = cam[:, 2] > 0
    for a, b in edges:
        if infront[a] and infront[b]:
            cv2.line(img, tuple(px[a].astype(int)), tuple(px[b].astype(int)), (0, 255, 0), 3)
    out = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
    cv2.imwrite('out/rack_wireframe2.jpg', out)

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    sb = (Pb[:, 2] > -1.3) & (Pb[:, 2] < 1.6) & (Pb[:, 0] > -10) & (Pb[:, 1] < 9)
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.scatter(Pb[sb, 0], Pb[sb, 1], c='lightgray', s=2)
    ring = np.vstack([top[:, :2], top[0, :2]])
    ax.plot(ring[:, 0], ring[:, 1], 'g-', lw=3, label='rack bounds')
    for nm, p, c in [('A id3', A, 'red'), ('B id11', B, 'orange'), ('C right', C, 'blue'), ('D (inf)', D, 'purple')]:
        ax.scatter([p[0]], [p[1]], c=c, s=160, marker='*', zorder=5, label=nm)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(); ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)')
    ax.set_title(f'rack bounds from detected posts  W={W:.2f} L={L:.2f} m')
    plt.savefig('out/rack_topdown2.png', dpi=70, bbox_inches='tight')
    np.save('out/rack_corners_base.npy', corners)
    print('saved out/rack_wireframe2.jpg, out/rack_topdown2.png, out/rack_corners_base.npy')


if __name__ == '__main__':
    main()
