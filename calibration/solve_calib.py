#!/usr/bin/env python3
"""Solve T_base <- cam for a ZED from extracted grapple-marker detections.

Each detection gives, via ArUco, the full pose T_cam<-marker. The user measured
the grappletong origin position in each marker's frame (translation only):
    LEFT  marker: (-0.114, -0.172, +/-0.014) m
    RIGHT marker: (+0.144, +0.106, ~0)       m
Applying that offset puts the tong origin as a 3D point in the CAMERA frame.
FK gives the same tong origin in the BASE frame. Over a motion bag this yields
many camera<->base point correspondences, solved by Horn's absolute orientation
(rigid registration) for T_base<-cam.

Unknowns the user does not remember are brute-forced by min RMS residual:
  - which marker id (1 or 8) is the LEFT vs RIGHT grapple marker
  - the +/- z sign of each measured offset (origin in front of / behind marker plane)
"""
import argparse, itertools
import numpy as np

# measured grappletong origin in marker frame (x, y, |z|), metres
OFFSET = {'LEFT': np.array([-0.114, -0.172, 0.014]),
          'RIGHT': np.array([0.144, 0.106, 0.000])}


def horn(src, dst):
    """Least-squares rigid transform mapping src->dst (Nx3). Returns 4x4, rms."""
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cd - R @ cs
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    res = (R @ src.T).T + t - dst
    rms = np.sqrt((res ** 2).sum(1).mean())
    return T, rms, np.linalg.norm(res, axis=1)


def R_to_rpy(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        return np.degrees([np.arctan2(R[2, 1], R[2, 2]),
                           np.arctan2(-R[2, 0], sy),
                           np.arctan2(R[1, 0], R[0, 0])])
    return np.degrees([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True, help='one or more extract npz files')
    ap.add_argument('--id-left', type=int, default=None,
                    help='force which marker id is LEFT (else brute-force)')
    args = ap.parse_args()

    ids, Tcm, Tt1, Tt2 = [], [], [], []
    cam = None
    for f in args.npz:
        d = np.load(f, allow_pickle=True)
        cam = str(d['cam'])
        ids.append(d['marker_id']); Tcm.append(d['T_cam_marker'])
        Tt1.append(d['T_base_tong1']); Tt2.append(d['T_base_tong2'])
    ids = np.concatenate(ids)
    Tcm = np.concatenate(Tcm); Tt1 = np.concatenate(Tt1); Tt2 = np.concatenate(Tt2)
    present = sorted(set(int(i) for i in ids))
    print(f'cam={cam}  detections={len(ids)}  ids present={present}')
    if len(present) != 2:
        print('expected exactly 2 grapple ids; got', present); return

    id_choices = ([(args.id_left, [p for p in present if p != args.id_left][0])]
                  if args.id_left is not None
                  else [(present[0], present[1]), (present[1], present[0])])

    best = None
    for id_left, id_right in id_choices:
        role = {id_left: 'LEFT', id_right: 'RIGHT'}
        # tong frame each marker rides on is decided by the FK that best matches;
        # try both marker->tong pairings too (left marker on tong1 or tong2)
        for left_on_t1 in (True, False):
            for zsL in (+1, -1):
                for zsR in (+1, -1):
                    zsign = {'LEFT': zsL, 'RIGHT': zsR}
                    src = np.empty((len(ids), 3)); dst = np.empty((len(ids), 3))
                    for k in range(len(ids)):
                        r = role[int(ids[k])]
                        off = OFFSET[r].copy(); off[2] *= zsign[r]
                        # tong origin point in camera frame
                        p_cam = (Tcm[k] @ np.append(off, 1.0))[:3]
                        src[k] = p_cam
                        on_t1 = (r == 'LEFT') == left_on_t1
                        dst[k] = (Tt1[k] if on_t1 else Tt2[k])[:3, 3]
                    T, rms, per = horn(src, dst)
                    cand = (rms, id_left, id_right, left_on_t1, zsL, zsR, T, per)
                    if best is None or rms < best[0]:
                        best = cand

    rms, id_left, id_right, left_on_t1, zsL, zsR, T, per = best
    Tinv = np.linalg.inv(T)
    print('\n=== best assignment ===')
    print(f'  LEFT  marker = id {id_left}  (rides grappletong{1 if left_on_t1 else 2}), z-sign {zsL:+d}')
    print(f'  RIGHT marker = id {id_right} (rides grappletong{2 if left_on_t1 else 1}), z-sign {zsR:+d}')
    print(f'  registration RMS = {rms*1000:.2f} mm   (median {np.median(per)*1000:.2f}, '
          f'p95 {np.percentile(per,95)*1000:.2f} mm, n={len(per)})')
    print('\n=== T_base <- cam ===')
    np.set_printoptions(precision=4, suppress=True)
    print(T)
    print(f'  translation (m): {T[:3,3].round(4)}')
    print(f'  rpy (deg ZYX):   {R_to_rpy(T[:3,:3]).round(2)}')
    print('\n=== T_cam <- base (inverse) ===')
    print(f'  translation (m): {Tinv[:3,3].round(4)}')


if __name__ == '__main__':
    main()
