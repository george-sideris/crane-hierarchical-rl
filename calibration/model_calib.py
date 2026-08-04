#!/usr/bin/env python3
"""Model-based extrinsic calibration of a ZED on the crane (rewrite of wilah's
model_based_calib.py against our URDF FK + STL meshes).

Optimizes the camera mount (6 dof: x,y,z mm + roll,pitch,yaw deg) and a few joint
angle offsets (deg) so the FK-posed grapple mesh aligns to the observed grapple
point cloud over many frames. Cost = trimmed sum of squared model->cloud nearest
neighbour distances (trimming drops occluded back-side model points).

Result for zed_0 is T_mast<-zed_0_optical; for zed_1 it is T_stick<-zed_1_optical.
"""
import argparse, sys, time
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import open3d as o3d
sys.path.insert(0, '.')
import calib_fk as fk

MESHDIR = '../fpi_crane_ros2/fpi_crane_description/meshes'
OFFSET_JOINTS = {'zed_0': ['boom', 'stick', 'hanger', 'bearingfork', 'grapplecarrier'],
                 'zed_1': ['hanger', 'bearingfork', 'grapplecarrier'],
                 'lidar_0': ['boom', 'stick', 'hanger', 'bearingfork', 'grapplecarrier']}
TRIM = 0.7          # keep best 70% of model-point distances
N_PER_LINK = 900


def sample_mesh(stl):
    m = o3d.io.read_triangle_mesh(f'{MESHDIR}/{stl}')
    return np.asarray(m.sample_points_uniformly(N_PER_LINK).points)


def model_points(cam, mount_params, offsets, jv, base_pts):
    """Pose the 3 grapple link meshes into the optical frame -> stacked Nx3."""
    parent = fk.CAM_PARENT[cam]
    Topt_parent = np.linalg.inv(fk.mount_T(cam, mount_params))
    T_gc = Topt_parent @ fk.fk_chain(parent, jv, offsets)
    tong = offsets.get('tong', 0.0)
    T_t1 = T_gc @ fk.joint_T('grappletong1', jv['grappletong1'] + tong)
    T_t2 = T_gc @ fk.joint_T('grappletong2', jv['grappletong2'] + tong)
    gc, t1, t2 = base_pts
    out = np.empty((len(gc)+len(t1)+len(t2), 3))
    out[:len(gc)] = (T_gc[:3, :3] @ gc.T).T + T_gc[:3, 3]
    out[len(gc):len(gc)+len(t1)] = (T_t1[:3, :3] @ t1.T).T + T_t1[:3, 3]
    out[len(gc)+len(t1):] = (T_t2[:3, :3] @ t2.T).T + T_t2[:3, 3]
    return out


def unpack(x, cam):
    mount = x[:6]
    if len(x) == 6:                      # mount-only: arm offsets fixed at 0
        return mount, {}
    offs = {j: np.radians(x[6 + i]) for i, j in enumerate(OFFSET_JOINTS[cam])}
    offs['tong'] = np.radians(x[6 + len(OFFSET_JOINTS[cam])])
    return mount, offs


def make_cost(cam, frames, base_pts):
    """Trimmed model->cloud cost: each model point pulled to the nearest observed point;
    trimming drops the worst 30% (occluded back-side model points). Robust to extra
    (non-grapple) cloud structure left in the crop. Frames parallel (cKDTree frees GIL)."""
    from concurrent.futures import ThreadPoolExecutor
    trees = [cKDTree(c) for c, jv in frames]
    pool = ThreadPoolExecutor(max_workers=min(len(frames), 14))

    def frame_err(args):
        tree, jv, x = args
        mount, offs = unpack(x, cam)
        mp = model_points(cam, mount, offs, jv, base_pts)
        d, _ = tree.query(mp)
        d2 = np.sort(d**2)[:int(TRIM*len(d))]
        return d2.sum(), len(d2)

    def cost(x):
        res = list(pool.map(frame_err, [(t, jv, x) for t, (c, jv) in zip(trees, frames)]))
        tot = sum(r[0] for r in res); npts = sum(r[1] for r in res)
        return tot / npts * 1e6   # mean sq dist in mm^2
    return cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', required=True, choices=['zed_0', 'zed_1', 'lidar_0'])
    ap.add_argument('--npz', required=True)
    ap.add_argument('--max-frames', type=int, default=24)
    ap.add_argument('--method', default='Nelder-Mead')
    ap.add_argument('--mount-only', action='store_true', help='fix arm offsets at 0, optimize 6-dof mount only')
    args = ap.parse_args()

    JN = ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork',
          'grapplecarrier', 'grappletong1', 'grappletong2']
    d = np.load(args.npz, allow_pickle=True)
    clouds = d['clouds']
    idx = np.linspace(0, len(clouds)-1, min(args.max_frames, len(clouds))).astype(int)
    frames = [(np.asarray(clouds[i], float),
               {n: float(d[f'j_{n}'][i]) for n in JN}) for i in idx]
    base_pts = (sample_mesh('grapplecarrier.stl'),
                sample_mesh('grappletong1.stl'), sample_mesh('grappletong2.stl'))
    print(f'{args.cam}: {len(frames)} frames, offset joints {OFFSET_JOINTS[args.cam]} + tong')

    n_off = 0 if args.mount_only else len(OFFSET_JOINTS[args.cam]) + 1
    x0 = np.concatenate([fk.nominal_mount_params(args.cam), np.zeros(n_off)])
    cost = make_cost(args.cam, frames, base_pts)
    print(f'initial cost (nominal) = {cost(x0):.2f} mm^2  (RMS {np.sqrt(cost(x0)):.1f} mm)')

    # per-dimension initial simplex: translation +/-15mm, rotation +/-5deg, offsets +/-3deg
    steps = np.array([15, 15, 15, 5, 5, 5] + [3] * n_off, float)
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])

    t0 = time.time()
    res = minimize(cost, x0, method=args.method,
                   options={'maxiter': 6000, 'maxfev': 6000, 'xatol': 1e-3, 'fatol': 1e-2,
                            'adaptive': True, 'initial_simplex': simplex})
    dt = time.time() - t0
    mount, offs = unpack(res.x, args.cam)
    print(f'\noptimized in {dt:.0f}s: cost = {res.fun:.2f} mm^2 (RMS {np.sqrt(res.fun):.1f} mm)')
    print(f'mount (x,y,z mm | r,p,y deg): {res.x[:6].round(2)}')
    print(f'   nominal was:               {fk.nominal_mount_params(args.cam).round(2)}')
    if not args.mount_only:
        print('joint offsets (deg):', {j: round(res.x[6+i], 3) for i, j in enumerate(OFFSET_JOINTS[args.cam])},
              'tong', round(res.x[6+len(OFFSET_JOINTS[args.cam])], 3))
    else:
        print('joint offsets: fixed at 0 (mount-only)')
    T = fk.mount_T(args.cam, mount)
    # Timestamped archive (never overwritten) + a 'latest' copy for downstream tools.
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch = f'out/calib_{args.cam}_{ts}'
    np.save(f'{arch}_mount.npy', T)
    np.save(f'{arch}_params.npy', res.x)
    np.save(f'out/calib_{args.cam}_mount.npy', T)
    np.save(f'out/calib_{args.cam}_params.npy', res.x)
    print(f'\nT_{fk.CAM_PARENT[args.cam]}<-{args.cam}_optical saved -> {arch}_mount.npy'
          f'  (+ latest out/calib_{args.cam}_mount.npy)')
    print('translation (m):', T[:3, 3].round(4))


if __name__ == '__main__':
    main()
