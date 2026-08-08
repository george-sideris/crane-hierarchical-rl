#!/usr/bin/env python3
"""Open3D iso renders of stored (cloud, expert action) pairs from a BC dataset.

Same visual conventions as render_crop_iso.py: height-coloured cloud, blue action-box wireframe;
plus the DECODED stored action as a red sphere (grasp centre) with a red yaw line. This is the
ground truth the policy is trained to reproduce - what you see here is exactly what the network
sees and what it is told to answer.

    python3 scripts/envs/render_dataset_samples.py --dataset logs/bc_pointcloud/bc_policy_aug1_v2 \
        --out docs/figures/dataset_samples_aug1v2.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

B_MIN = np.array([-5.364, -1.684, -1.30])
B_MAX = np.array([-3.364, 5.316, 0.10])
W, H = 760, 620

_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material
_R = {"r": None}


def valid(p):
    return p[np.abs(p).sum(-1) > 1e-6]


def height_colors(z, lo=-1.50, hi=-0.40):
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=1)


def decode(a):
    """stored arctanh-encoded 5D action -> (x, y, z, yaw) metres/radians"""
    xyz = B_MIN + (np.tanh(a[:3]) + 1.0) / 2.0 * (B_MAX - B_MIN)
    yaw = 0.5 * np.arctan2(a[4], a[3]) if len(a) >= 5 else float(a[3])
    return xyz[0], xyz[1], xyz[2], yaw


def panel(pts, tgt, margin=0.0):
    r = _R["r"]
    if r is None:
        r = rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1.0, 1.0, 1.0, 1.0])
        _R["r"] = r
    sc = r.scene
    sc.clear_geometry()
    mp = _Mat(); mp.shader = "defaultUnlit"; mp.point_size = 3.5
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 2.0
    mm = _Mat(); mm.shader = "defaultLit"

    if len(pts):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2]))
        sc.add_geometry("pcd", pcd, mp)
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(B_MIN, B_MAX))
    box.paint_uniform_color([0.1, 0.5, 1.0])
    sc.add_geometry("box", box, ml)
    if margin > 0:
        wl, wh = B_MIN - margin, B_MAX.copy(); wh[:2] += margin
        wbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(wl, wh))
        wbox.paint_uniform_color([0.95, 0.55, 0.0])
        sc.add_geometry("wbox", wbox, ml)

    x, y, z, yaw = tgt
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    sph.translate([x, y, z]); sph.paint_uniform_color([1, 0, 0]); sph.compute_vertex_normals()
    sc.add_geometry("sph", sph, mm)
    end1 = [x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw), z]
    end2 = [x - 0.6 * np.cos(yaw), y - 0.6 * np.sin(yaw), z]
    line = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([end1, [x, y, z], end2]),
                                lines=o3d.utility.Vector2iVector([[0, 1], [1, 2]]))
    line.paint_uniform_color([1, 0, 0])
    sc.add_geometry("line", line, ml)

    c = 0.5 * (B_MIN + B_MAX); c[2] = B_MIN[2] + 0.3 * (B_MAX[2] - B_MIN[2])
    rad = float(np.linalg.norm(B_MAX - B_MIN))
    r.setup_camera(60.0, c, c + np.array([1.0, -1.0, 0.55]) * (0.62 * rad), [0.0, 0.0, 1.0])
    return np.asarray(r.render_to_image())[:, :, :3]


def label(img, text, sub):
    """Caption band ABOVE the panel. Painting it over the image would hide content now
    that panels are cropped to their drawn extent."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    out = Image.new("RGB", (im.width, im.height + 40), (255, 255, 255))
    out.paste(im, (0, 40))
    d = ImageDraw.Draw(out)
    d.text((10, 6), text, fill=(0, 0, 0)); d.text((10, 22), sub, fill=(90, 90, 90))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--margin", type=float, default=0.0, help="draw the margin box if > 0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pc = np.load(os.path.join(a.dataset, "pointclouds.npy"), mmap_mode="r")
    ac = np.load(os.path.join(a.dataset, "actions.npy"))
    # spread the picks across pile fullness: sample count in the cloud tracks clearing progress
    counts = np.array([len(valid(np.asarray(pc[i]))) for i in range(0, len(pc), 29)])
    idx0 = np.arange(0, len(pc), 29)
    targets_n = np.linspace(counts.max(), max(counts.min(), 60), a.n).astype(int)
    picks = [int(idx0[np.argmin(np.abs(counts - t))]) for t in targets_n]

    # render first, then crop every panel with ONE common bounding box of drawn content:
    # the camera is shared, so a common crop centres the content without changing scale
    raw, meta = [], []
    for i in picks:
        pts = valid(np.asarray(pc[i]).astype(np.float64))
        tgt = decode(ac[i])
        raw.append(panel(pts, tgt, a.margin))
        meta.append((i, len(pts), tgt))
    bg = raw[0][0, 0].astype(float)
    ys0, ys1, xs0, xs1 = 10**9, -1, 10**9, -1
    for img in raw:
        m = np.abs(img.astype(float) - bg).sum(axis=2) > 12
        ys, xs = np.where(m)
        if len(ys):
            ys0, ys1 = min(ys0, ys.min()), max(ys1, ys.max())
            xs0, xs1 = min(xs0, xs.min()), max(xs1, xs.max())
    pad = 12
    ys0, xs0 = max(0, ys0 - pad), max(0, xs0 - pad)
    ys1 = min(raw[0].shape[0], ys1 + pad); xs1 = min(raw[0].shape[1], xs1 + pad)

    rows, row = [], []
    for img0, (i, npts, tgt) in zip(raw, meta):
        img = label(img0[ys0:ys1, xs0:xs1],
                    f"sample {i}  -  stored cloud + stored expert action",
                    f"{npts} pts | label ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}) "
                    f"yaw {np.degrees(tgt[3]):.0f} deg")
        row.append(img)
        if len(row) == 4:
            rows.append(np.concatenate(row, axis=1)); row = []
    if row:
        while len(row) < 4:
            row.append(np.full_like(row[0], 255))
        rows.append(np.concatenate(row, axis=1))
    out = np.concatenate(rows, axis=0)
    from PIL import Image
    Image.fromarray(out).save(a.out)
    print(f"wrote {a.out} (samples {picks})")


if __name__ == "__main__":
    main()
