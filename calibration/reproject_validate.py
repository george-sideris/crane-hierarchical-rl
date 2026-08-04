#!/usr/bin/env python3
"""Reproject the calibrated (and nominal) grapple mesh onto the camera image to
visually validate the model-based calibration. Green = calibrated, red = nominal."""
import argparse, sys
import numpy as np
import cv2
sys.path.insert(0, '.')
import calib_fk as fk
from model_calib import sample_mesh, model_points, unpack, OFFSET_JOINTS
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_FOXY)
JN = ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork',
      'grapplecarrier', 'grappletong1', 'grappletong2']
K = {'zed_0': np.array([[746.93, 0, 969.16], [0, 746.93, 543.95], [0, 0, 1]]),
     'zed_1': np.array([[728.99, 0, 962.29], [0, 728.99, 528.02], [0, 0, 1]])}


def grab_images_near(bag, cam, stamps):
    topic = f'/{cam}/zed_node/rgb/color/rect/image'
    want = sorted(stamps); got = {}
    with Reader(bag) as r:
        conns = {c.topic: c for c in r.connections}
        for _, t_ns, raw in r.messages(connections=[conns[topic]]):
            for s in want:
                if abs(t_ns - s) < 60_000_000 and s not in got:  # 60ms
                    m = TS.deserialize_cdr(raw, conns[topic].msgtype)
                    buf = np.frombuffer(m.data, np.uint8); ch = buf.size//(m.height*m.width)
                    a = buf.reshape(m.height, m.width, ch)
                    got[s] = cv2.cvtColor(a, cv2.COLOR_BGRA2BGR) if ch == 4 else a.copy()
            if len(got) == len(want):
                break
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cam', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--bag', required=True)
    ap.add_argument('--frames', type=int, nargs='+', default=[3, 8, 13])
    ap.add_argument('--params', default=None, help='override calibrated params .npy (default: active calib)')
    ap.add_argument('--out', default=None, help='override output jpg path')
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    base = (sample_mesh('grapplecarrier.stl'), sample_mesh('grappletong1.stl'), sample_mesh('grappletong2.stl'))
    x_cal = np.load(args.params or f'out/calib_{args.cam}_params.npy')
    if len(x_cal) == 6:                     # mount-only: pad joint offsets with zeros
        x_cal = np.concatenate([x_cal, np.zeros(len(OFFSET_JOINTS[args.cam])+1)])
    x_nom = np.concatenate([fk.nominal_mount_params(args.cam), np.zeros(len(OFFSET_JOINTS[args.cam])+1)])
    n = len(d['stamp'])
    fr = [i for i in args.frames if i < n] or list(np.linspace(0, n-1, min(3, n)).astype(int))
    stamps = {int(d['stamp'][i]): i for i in fr}
    imgs = grab_images_near(args.bag, args.cam, list(stamps))
    Kc = K[args.cam]
    tiles = []
    for s, i in stamps.items():
        if s not in imgs:
            continue
        img = imgs[s].copy()
        jv = {n: float(d[f'j_{n}'][i]) for n in JN}
        for x, col in [(x_nom, (0, 0, 255)), (x_cal, (0, 255, 0))]:
            mount, offs = unpack(x, args.cam)
            mp = model_points(args.cam, mount, offs, jv, base)
            uv = (Kc @ mp.T).T; px = uv[:, :2] / uv[:, 2:3]
            for p in px[mp[:, 2] > 0].astype(int):
                if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                    cv2.circle(img, tuple(p), 1, col, -1)
        tiles.append(cv2.resize(img, (img.shape[1]//2, img.shape[0]//2)))
    if tiles:
        outp = args.out or f'out/reproj_{args.cam}.jpg'
        cv2.imwrite(outp, np.vstack(tiles))
        print(f'saved {outp}  (red=nominal, green=calibrated)')


if __name__ == '__main__':
    main()
