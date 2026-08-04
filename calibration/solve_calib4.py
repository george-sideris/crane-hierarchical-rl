#!/usr/bin/env python3
"""Classify each marker (grapple vs rack) by geometry, then solve T_parent<-cam.

IDs are reused across grapple and rack, so we do NOT trust them. For each marker id
we test which rigid body it actually rides:
    tong1 / tong2  -> grapple marker (rigid to a tong frame)
    world          -> rack marker (fixed in base_link)
by solving, under each hypothesis, the constant X = T_parent<-cam plus the marker's
fixed point p in that body, from marker CENTERS only (rotation-free):
        X @ c_cam(t) = F(t) @ p
where F(t) is the body's pose in the camera's parent frame:
    tong_i:  parent<-tong_i = inv(base<-parent) @ base<-tong_i
    world:   parent<-base    = inv(base<-parent)         (p is the point in base_link)
The lowest-RMS hypothesis classifies the marker. Then X is jointly re-solved using
all markers under their winning hypotheses. parent = mast (zed_0) / stick (zed_1).
"""
import argparse
import numpy as np

PARENT = {'zed_0': ('mast', 'T_base_mast', 'out/nom_mast_zed_0.npy'),
          'zed_1': ('stick', 'T_base_stick', 'out/nom_stick_zed_1.npy')}


def horn(s, d):
    cs, cd = s.mean(0), d.mean(0)
    U, _, Vt = np.linalg.svd((s - cs).T @ (d - cd))
    de = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, de]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cd - R @ cs
    return T


def rpy(R):
    sy = (R[0, 0]**2 + R[1, 0]**2)**.5
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy),
                       np.arctan2(R[1, 0], R[0, 0])])


def solve(c_cam, F_per_det, ids, id_list, X0, iters=60, reject=2.5):
    """Alternating solve of X (+ fixed point p per id) given each det's body frame F."""
    cc1 = np.c_[c_cam, np.ones(len(c_cam))]
    Finv = np.linalg.inv(F_per_det)
    keep = np.ones(len(c_cam), bool)
    X = X0.copy()
    for _ in range(iters):
        cen = (X @ cc1.T).T
        p = {}
        for i in id_list:
            sel = keep & (ids == i)
            if sel.sum() < 3:
                p[i] = np.zeros(3); continue
            p[i] = np.einsum('nij,nj->ni', Finv[sel], cen[sel])[:, :3].mean(0)
        ph = np.array([np.append(p[i], 1.0) for i in ids])
        dst = np.einsum('nij,nj->ni', F_per_det, ph)[:, :3]
        X = horn(c_cam[keep], dst[keep])
        res = np.linalg.norm((X @ cc1.T).T[:, :3] - dst, axis=1)
        md = np.median(res[keep]); mad = np.median(np.abs(res[keep] - md)) + 1e-9
        keep = res < md + reject * 1.4826 * mad
    rms = (res[keep]**2).mean()**.5
    return X, p, res, keep, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True)
    ap.add_argument('--cam', required=True)
    args = ap.parse_args()
    pname, pkey, nomf = PARENT[args.cam]
    Xseed = np.load(nomf)

    D = {k: [] for k in ['marker_id', 'T_cam_marker', 'T_base_tong1', 'T_base_tong2', pkey]}
    Xnom = None
    for f in args.npz:
        d = np.load(f, allow_pickle=True)
        for k in D:
            D[k].append(d[k])
    ids = np.concatenate(D['marker_id']).astype(int)
    Tcm = np.concatenate(D['T_cam_marker'])
    Tt1 = np.concatenate(D['T_base_tong1']); Tt2 = np.concatenate(D['T_base_tong2'])
    Bp = np.concatenate(D[pkey])               # base<-parent per det
    c_cam = Tcm[:, :3, 3]
    Bpinv = np.linalg.inv(Bp)
    # body pose in parent frame, per det, per hypothesis
    F = {'tong1': Bpinv @ Tt1, 'tong2': Bpinv @ Tt2, 'world': Bpinv}
    present = sorted(set(ids))
    print(f'cam={args.cam} parent={pname} dets={len(ids)} ids={present} '
          f'(counts={[int((ids==i).sum()) for i in present]})')

    X0 = Xseed
    print(f'seed T_{pname}<-{args.cam} translation: {X0[:3,3].round(4)}')
    print('\nper-id hypothesis RMS (mm)  [lower = the body it rides]:')
    klass = {}
    for i in present:
        row = {}
        for h in ('tong1', 'tong2', 'world'):
            sel = ids == i
            _, _, _, keep, rms = solve(c_cam[sel], F[h][sel], ids[sel], [i], X0)
            row[h] = (rms, keep.sum(), sel.sum())
        best_h = min(row, key=lambda h: row[h][0])
        klass[i] = best_h
        cells = '  '.join(f'{h}={row[h][0]*1000:7.1f}({row[h][1]}/{row[h][2]})' for h in row)
        tag = 'GRAPPLE' if best_h.startswith('tong') else 'RACK'
        print(f'  id{i:2d}: {cells}   -> {best_h} [{tag}]')

    # joint solve of X using all dets under their winning hypothesis
    Fsel = np.array([F[klass[ids[k]]][k] for k in range(len(ids))])
    X, p, res, keep, rms = solve(c_cam, Fsel, ids, present, X0, iters=80)
    np.set_printoptions(precision=4, suppress=True)
    print(f'\n=== joint T_{pname} <- {args.cam} (optical) ===  '
          f'inliers={keep.sum()}/{len(keep)} RMS={rms*1000:.1f}mm '
          f'med={np.median(res[keep])*1000:.1f} p95={np.percentile(res[keep],95)*1000:.1f}mm')
    print(X)
    print(f'  translation (m): {X[:3,3].round(4)}  rpy ZYX(deg): {rpy(X[:3,:3]).round(2)}')
    for i in present:
        print(f'  id{i} fixed-pt |p|={np.linalg.norm(p[i])*1000:.1f}mm ({klass[i]})')
    np.save(f'out/T_{pname}_{args.cam}.npy', X)


if __name__ == '__main__':
    main()
