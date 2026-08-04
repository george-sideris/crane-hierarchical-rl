#!/usr/bin/env python3
"""Validate a model-based calibration: overlay the calibrated grapple model on the
observed cloud (nominal vs calibrated) and report per-frame RMS model->cloud distance.
"""
import argparse, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
import calib_fk as fk
from model_calib import sample_mesh, model_points, unpack, OFFSET_JOINTS

JN = ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork',
      'grapplecarrier', 'grappletong1', 'grappletong2']


def rms(cam, mount, offs, frames, base_pts):
    tot = 0.0; n = 0
    for cloud, jv in frames:
        mp = model_points(cam, mount, offs, jv, base_pts)
        d, _ = cKDTree(cloud).query(mp)
        d2 = np.sort(d**2)[:int(0.7*len(d))]
        tot += d2.sum(); n += len(d2)
    return np.sqrt(tot/n)*1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', required=True)
    ap.add_argument('--npz', required=True)
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    clouds = d['clouds']
    idx = np.linspace(0, len(clouds)-1, 20).astype(int)
    frames = [(np.asarray(clouds[i], float), {n: float(d[f'j_{n}'][i]) for n in JN}) for i in idx]
    base_pts = (sample_mesh('grapplecarrier.stl'), sample_mesh('grappletong1.stl'), sample_mesh('grappletong2.stl'))

    x_nom = np.concatenate([fk.nominal_mount_params(args.cam), np.zeros(len(OFFSET_JOINTS[args.cam])+1)])
    x_cal = np.load(f'out/calib_{args.cam}_params.npy')
    for tag, x in [('nominal', x_nom), ('calibrated', x_cal)]:
        mount, offs = unpack(x, args.cam)
        print(f'{tag:11s} RMS = {rms(args.cam, mount, offs, frames, base_pts):.1f} mm')

    # overlay one frame
    mount, offs = unpack(x_cal, args.cam)
    cloud, jv = frames[len(frames)//2]
    mp_cal = model_points(args.cam, mount, offs, jv, base_pts)
    mn, on = unpack(x_nom, args.cam); mp_nom = model_points(args.cam, mn, on, jv, base_pts)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (i, j, xl, yl) in zip(axs, [(0, 2, 'x', 'z depth'), (0, 1, 'x', 'y')]):
        ax.scatter(cloud[:, i], cloud[:, j], s=4, c='steelblue', label='observed')
        ax.scatter(mp_nom[:, i], mp_nom[:, j], s=2, c='gray', alpha=.4, label='nominal model')
        ax.scatter(mp_cal[:, i], mp_cal[:, j], s=2, c='red', alpha=.5, label='calibrated model')
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=.3)
    axs[0].set_title(f'{args.cam} calibrated grapple alignment')
    plt.savefig(f'out/calib_check_{args.cam}.png', dpi=70, bbox_inches='tight')
    print(f'saved out/calib_check_{args.cam}.png')


if __name__ == '__main__':
    main()
