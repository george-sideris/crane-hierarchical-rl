#!/usr/bin/env python3
"""Floor-constrained mount calibration (zed_0 or zed_1).

Grapple-only calibration leaves a few degrees of roll/pitch error (the grapple is
a small, near object; tilt trades off against range and stays invisible in grapple
RMS). The lab floor is a large, distant horizontal plane, so it pins roll/pitch.

Cost = grapple model->cloud cost  +  W * (floor-normal tilt from vertical, deg)^2.

The floor points are segmented ONCE from a static frame using the incoming calib,
then re-posed each eval into the BASE frame (through the camera's parent chain) so
the plane-normal tilt is measured against true vertical (gravity / base +Z).
"""
import argparse, sys, time
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
import calib_fk as fk
import model_calib as mc
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
JN = ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork',
      'grapplecarrier', 'grappletong1', 'grappletong2']
# fixed-joint chain from base_link down to each camera's rigid parent link
BASE_CHAIN = {'mast': ['slew'], 'stick': ['slew', 'boom', 'stick']}


def q2R(x, y, z, w):
    n = np.sqrt(x*x+y*y+z*z+w*w); x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tf2T(tr):
    T = np.eye(4); q = tr.transform.rotation; t = tr.transform.translation
    T[:3, :3] = q2R(q.x, q.y, q.z, q.w); T[:3, 3] = [t.x, t.y, t.z]; return T


def base_parent_T(parent, jvals):
    T = np.eye(4)
    for j in BASE_CHAIN[parent]:
        T = T @ fk.joint_T(j, jvals[j])
    return T


def load_floor_frame(bag, cam, init_mount, max_pts=3000):
    """Return (floor_opt Nx3 optical-frame floor points, T_base_parent for that frame).
    Floor = low-Z planar inliers under the initial calib, in the base frame."""
    sc = fk.SENSORS[cam]; topic = sc['topic']
    cf = sc['cloud_frame']; oc = cf + '_optical'
    parent = fk.CAM_PARENT[cam]
    opt_edge = None; best = None; densest = 0; best_t = None
    jt = []; jv = []
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic in (topic, '/joint_states', '/tf_static')]
        for con, t, raw in r.messages(connections=conns):
            if con.topic == '/tf_static':
                m = TS.deserialize_cdr(raw, con.msgtype)
                for tr in m.transforms:
                    if (tr.header.frame_id, tr.child_frame_id) == (cf, oc):
                        opt_edge = tf2T(tr)
            elif con.topic == '/joint_states':
                m = TS.deserialize_cdr(raw, con.msgtype); d = dict(zip(m.name, m.position))
                jt.append(t); jv.append({n: d.get(n + '_joint', 0.0) for n in JN})
            elif con.topic == topic:
                m = TS.deserialize_cdr(raw, con.msgtype)
                arr = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
                xyz = arr[:, :12].copy().view(np.float32).reshape(-1, 3)
                fin = np.isfinite(xyz).all(1).sum()
                if fin > densest:
                    densest = fin; best = xyz[np.isfinite(xyz).all(1)].astype(float); best_t = t
    jt = np.array(jt)
    jvals = jv[int(np.argmin(np.abs(jt - best_t)))]
    T_bp = base_parent_T(parent, jvals)
    opt_inv = np.linalg.inv(opt_edge)
    q = (opt_inv @ np.c_[best, np.ones(len(best))].T).T[:, :3]   # optical-frame pts
    Pb = (T_bp @ init_mount @ np.c_[q, np.ones(len(q))].T).T[:, :3]   # base frame
    zc = np.quantile(Pb[:, 2], 0.35)
    cand = q[Pb[:, 2] < zc]; candb = Pb[Pb[:, 2] < zc]
    c = candb.mean(0); n = np.linalg.svd(candb - c, full_matrices=False)[2][2]
    for _ in range(8):
        d = (candb - c) @ n; keep = np.abs(d) < 0.05
        c = candb[keep].mean(0); n = np.linalg.svd(candb[keep] - c, full_matrices=False)[2][2]
    keep = np.abs((candb - c) @ n) < 0.05
    floor_opt = cand[keep]
    if len(floor_opt) > max_pts:
        floor_opt = floor_opt[np.linspace(0, len(floor_opt)-1, max_pts).astype(int)]
    print(f'[{cam}] floor frame: densest {densest} pts, parent {parent}, floor inliers {len(floor_opt)}')
    return floor_opt, T_bp


def floor_tilt_deg(cam, mount_params, floor_opt, T_bp):
    T = T_bp @ fk.mount_T(cam, mount_params)
    Pb = (T[:3, :3] @ floor_opt.T).T + T[:3, 3]
    c = Pb.mean(0); n = np.linalg.svd(Pb - c, full_matrices=False)[2][2]
    if n[2] < 0:
        n = -n
    return np.degrees(np.arccos(np.clip(n[2], -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', required=True, choices=['zed_0', 'zed_1'])
    ap.add_argument('--npz', default=None, help='grapple npz (default out/grapple_<cam>.npz)')
    ap.add_argument('--floor-bag', required=True)
    ap.add_argument('--init', default=None, help='mount for floor segmentation (default out/calib_<cam>_mount.npy)')
    ap.add_argument('--w', type=float, default=400.0, help='floor weight (mm^2 per deg^2)')
    ap.add_argument('--max-frames', type=int, default=24)
    args = ap.parse_args()
    cam = args.cam
    npz = args.npz or f'out/grapple_{cam}.npz'
    init = args.init or f'out/calib_{cam}_mount.npy'

    d = np.load(npz, allow_pickle=True); clouds = d['clouds']
    idx = np.linspace(0, len(clouds)-1, min(args.max_frames, len(clouds))).astype(int)
    frames = [(np.asarray(clouds[i], float), {n: float(d[f'j_{n}'][i]) for n in JN}) for i in idx]
    base = (mc.sample_mesh('grapplecarrier.stl'), mc.sample_mesh('grappletong1.stl'),
            mc.sample_mesh('grappletong2.stl'))
    grap_cost = mc.make_cost(cam, frames, base)
    floor_opt, T_bp = load_floor_frame(args.floor_bag, cam, np.load(init))

    def cost(x):
        return grap_cost(x) + args.w * floor_tilt_deg(cam, x, floor_opt, T_bp)**2

    x0 = fk.nominal_mount_params(cam).astype(float)
    print(f'[{cam}] init nominal: grapple RMS {np.sqrt(grap_cost(x0)):.1f} mm, '
          f'floor tilt {floor_tilt_deg(cam, x0, floor_opt, T_bp):.2f} deg')
    steps = np.array([15, 15, 15, 5, 5, 5], float)
    simplex = np.vstack([x0] + [x0 + np.eye(6)[i]*steps[i] for i in range(6)])
    t0 = time.time()
    res = minimize(cost, x0, method='Nelder-Mead',
                   options={'maxiter': 8000, 'maxfev': 8000, 'xatol': 1e-3, 'fatol': 1e-2,
                            'adaptive': True, 'initial_simplex': simplex})
    g = np.sqrt(grap_cost(res.x)); tlt = floor_tilt_deg(cam, res.x, floor_opt, T_bp)
    T = fk.mount_T(cam, res.x)
    print(f'[{cam}] optimized in {time.time()-t0:.0f}s (w={args.w}): '
          f'grapple RMS {g:.1f} mm, floor tilt {tlt:.2f} deg')
    print(f'[{cam}] mount params (x,y,z mm | r,p,y deg): {res.x.round(2)}')
    print(f'[{cam}] full optical trans (m): {T[:3,3].round(4)}')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    np.save(f'out/calib_{cam}_{ts}_floor_mount.npy', T)
    np.save(f'out/calib_{cam}_{ts}_floor_params.npy', res.x)
    np.save(f'out/calib_{cam}_floor_mount.npy', T)         # latest-floor pointer
    np.save(f'out/calib_{cam}_floor_params.npy', res.x)
    print(f'[{cam}] saved -> out/calib_{cam}_{ts}_floor_mount.npy (+ latest out/calib_{cam}_floor_mount.npy)')


if __name__ == '__main__':
    main()
