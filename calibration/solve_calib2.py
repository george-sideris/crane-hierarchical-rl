#!/usr/bin/env python3
"""Robust solve of T_base<-cam using ArUco marker CENTERS only (no lever-arm rotation).

Single 13cm markers at ~2.7m give reliable centers (~cm) but noisy/ambiguous rotation.
So instead of applying the measured marker->tong offset through the noisy per-frame
marker rotation, we jointly estimate:
    X   = T_base<-cam            (constant; zed_0 is rigid to base)
    m_i = marker center position in its tong frame   (constant, unknown)
from the constraint, per detection of marker i at time t:
    X @ c_cam(t)  =  T_base<-tong_i(t) @ m_i
using only c_cam = marker-center translation. Solved by alternating closed form:
  - fix m_i -> X by Horn(src=c_cam, dst=T_base<-tong_i @ m_i)
  - fix X   -> m_i = mean_t [ inv(T_base<-tong_i(t)) @ X @ c_cam(t) ]
with iterative outlier rejection. |m_i| is cross-checked vs the measured offset norm.
"""
import argparse
import numpy as np

# measured grappletong-origin in marker frame -> its norm = |marker_center - tong_origin|
OFFNORM = {'LEFT': np.linalg.norm([0.114, 0.172, 0.014]),
           'RIGHT': np.linalg.norm([0.144, 0.106, 0.0])}


def horn(src, dst, w=None):
    if w is None:
        w = np.ones(len(src))
    w = w / w.sum()
    cs = (w[:, None] * src).sum(0)
    cd = (w[:, None] * dst).sum(0)
    H = ((w[:, None] * (src - cs)).T @ (dst - cd))
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cd - R @ cs
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def R_to_rpy(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]),
                       np.arctan2(-R[2, 0], sy),
                       np.arctan2(R[1, 0], R[0, 0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True)
    ap.add_argument('--id-left', type=int, default=1)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--reject', type=float, default=3.0, help='reject resid > k*MAD')
    args = ap.parse_args()

    ids, Tcm, Tt1, Tt2 = [], [], [], []
    cam = None
    for f in args.npz:
        d = np.load(f, allow_pickle=True)
        cam = str(d['cam'])
        ids.append(d['marker_id']); Tcm.append(d['T_cam_marker'])
        Tt1.append(d['T_base_tong1']); Tt2.append(d['T_base_tong2'])
    ids = np.concatenate(ids).astype(int)
    Tcm = np.concatenate(Tcm); Tt1 = np.concatenate(Tt1); Tt2 = np.concatenate(Tt2)
    present = sorted(set(ids))
    id_left = args.id_left
    id_right = [p for p in present if p != id_left][0]
    role = {id_left: 'LEFT', id_right: 'RIGHT'}
    c_cam = Tcm[:, :3, 3]
    print(f'cam={cam} dets={len(ids)} ids={present} (LEFT={id_left}, RIGHT={id_right})')

    best = None
    for left_on_t1 in (True, False):
        # which FK tong frame each detection rides
        Tt = np.where(((role_arr := np.array([role[i] for i in ids])) == 'LEFT')[:, None, None]
                      == left_on_t1, Tt1, Tt2)
        Ttinv = np.linalg.inv(Tt)
        keep = np.ones(len(ids), bool)
        X = np.eye(4)
        # init X from a robust subset using offset-free guess: align centroids+PCA later;
        # bootstrap m_i from identity X
        for it in range(args.iters):
            # update m_i
            m = {}
            for i in present:
                sel = keep & (ids == i)
                pts = np.einsum('nij,nj->ni', Ttinv[sel],
                                np.c_[(X @ np.c_[c_cam[sel], np.ones(sel.sum())].T).T[:, :3],
                                      np.ones(sel.sum())])[:, :3]
                m[i] = pts.mean(0)
            # build dst, update X
            dst = np.array([(Tt[k] @ np.append(m[ids[k]], 1))[:3] for k in range(len(ids))])
            X = horn(c_cam[keep], dst[keep])
            # residuals
            pred = (X @ np.c_[c_cam, np.ones(len(ids))].T).T[:, :3]
            res = np.linalg.norm(pred - dst, axis=1)
            med = np.median(res[keep]); mad = np.median(np.abs(res[keep] - med)) + 1e-9
            keep = res < med + args.reject * 1.4826 * mad
        rms = np.sqrt((res[keep]**2).mean())
        if best is None or rms < best[0]:
            best = (rms, X.copy(), {i: m[i].copy() for i in present}, keep.copy(),
                    res.copy(), left_on_t1)

    rms, X, m, keep, res, left_on_t1 = best
    Xinv = np.linalg.inv(X)
    np.set_printoptions(precision=4, suppress=True)
    print(f'\nLEFT(id{id_left}) rides grappletong{1 if left_on_t1 else 2}, '
          f'RIGHT(id{id_right}) rides grappletong{2 if left_on_t1 else 1}')
    print(f'inliers={keep.sum()}/{len(keep)}  RMS={rms*1000:.1f}mm  '
          f'median={np.median(res[keep])*1000:.1f}mm  p95={np.percentile(res[keep],95)*1000:.1f}mm')
    print('\nmarker-center distance to tong origin (solved vs measured):')
    for i in present:
        print(f'  id{i} ({role[i]}): |m|={np.linalg.norm(m[i])*1000:.1f}mm  '
              f'measured={OFFNORM[role[i]]*1000:.1f}mm  '
              f'diff={abs(np.linalg.norm(m[i])-OFFNORM[role[i]])*1000:.1f}mm')
    print('\n=== T_base <- cam (zed_0 optical) ===')
    print(X)
    print(f'  translation (m): {X[:3,3].round(4)}')
    print(f'  rpy ZYX (deg):   {R_to_rpy(X[:3,:3]).round(2)}')
    print(f'  cam origin in base (m): {X[:3,3].round(4)}')


if __name__ == '__main__':
    main()
