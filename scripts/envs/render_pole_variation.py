#!/usr/bin/env python3
"""Preview of the POLE-VARIATION augmentation exactly as train_scoring_head.py applies it.

The trainer never calls stub_aug.inject_stub: the only structure augmentation used for the
deployed checkpoint is per-epoch variation of the pole columns already present in the
collected margin clouds. Per cloud, with probability --stub_aug the variation is drawn at
all; then for each detected column u ~ U(0,1) decides
    u < 0.25          keep the column intact
    0.25 <= u <= 0.90 cut to a stub of height U(0.08, 0.6) m
    u > 0.90          cut nearly flush (0.05 m)
Whatever pole points remain afterwards form the explicit negative set for the softplus term.

Clouds are picked with the same screening as the interactive preview (pile substance and
spread), so half-captured gaze frames are excluded.

    python3 scripts/envs/render_pole_variation.py --out docs/figures/pole_variation.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_stub_aug_interactive import screened_picks  # noqa: E402
from stub_aug import find_pole_columns  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30])
B_MAX = np.array([-3.364, 5.316, 0.10])
DS = "logs/bc_pointcloud/bc_margin05_2048_500"
W, H = 1150, 780
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material
_R = {"r": None}


def vary(pts, cols, rng):
    """Trainer-exact pole variation. Returns (new_pts, removed_mask, kept_pole_mask, heights)."""
    out = pts.copy()
    removed = np.zeros(len(pts), bool)
    heights = []
    for zbase, idx in cols:
        u = rng.random()
        if u < 0.25:
            heights.append(None)                      # column kept intact
            continue
        h = 0.05 if u > 0.90 else rng.uniform(0.08, 0.6)
        heights.append(h)
        cut = idx[out[idx][:, 2] > zbase + h]
        removed[cut] = True
    out[removed] = 0.0
    pole = np.zeros(len(pts), bool)
    for _, idx in cols:
        pole[idx] = True
    kept_pole = pole & ~removed & (np.abs(out).sum(-1) > 1e-6)
    return out, removed, kept_pole, heights


def height_colors(z):
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    from matplotlib import cm
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    return cm.viridis(t)[:, :3] * 0.45 + 0.48


def panel(pts, ghost=None, negatives=None):
    r = _R["r"]
    if r is None:
        r = rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1, 1, 1, 1])
        _R["r"] = r
    sc = r.scene
    sc.clear_geometry()
    mp = _Mat(); mp.shader = "defaultUnlit"; mp.point_size = 5.5
    mg = _Mat(); mg.shader = "defaultUnlit"; mg.point_size = 4.0
    mn = _Mat(); mn.shader = "defaultUnlit"; mn.point_size = 7.0
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 2.0

    v = pts[np.abs(pts).sum(-1) > 1e-6]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(v[:, 2]))
    sc.add_geometry("pcd", pcd, mp)
    if ghost is not None and len(ghost):
        g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ghost))
        g.paint_uniform_color([0.72, 0.72, 0.74])
        sc.add_geometry("ghost", g, mg)
    if negatives is not None and len(negatives):
        n = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(negatives))
        n.paint_uniform_color([0.85, 0.10, 0.10])
        sc.add_geometry("neg", n, mn)
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(B_MIN, B_MAX))
    box.paint_uniform_color([0.35, 0.55, 0.90])
    sc.add_geometry("box", box, ml)

    allp = np.vstack([v] + ([ghost] if ghost is not None and len(ghost) else []))
    lo = np.minimum(allp.min(0), B_MIN); hi = np.maximum(allp.max(0), B_MAX)
    ctr = 0.5 * (lo + hi)
    rad = float(np.linalg.norm(hi - lo))
    d = np.array([1.00, -0.28, 0.62]); d /= np.linalg.norm(d)
    r.setup_camera(50.0, ctr, ctr + d * 0.95 * rad, [0, 0, 1])
    return np.asarray(r.render_to_image())[:, :, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DS)
    ap.add_argument("--out", default="docs/figures/pole_variation.png")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    pc = np.load(os.path.join(a.dataset, "pointclouds.npy"), mmap_mode="r")
    rng = np.random.default_rng(a.seed)
    picks = screened_picks(pc, n_want=a.rows + 4, rng=np.random.default_rng(11))
    raw, caps = [], []
    used = 0
    for i in picks:
        if used >= a.rows:
            break
        pts = np.asarray(pc[i]).astype(np.float64)
        cols = [(float(pts[m][:, 2].min()), np.where(m)[0]) for _, m in find_pole_columns(pts)]
        if not cols:
            continue
        new, removed, kept_pole, hs = vary(pts, cols, rng)
        if not removed.any():
            continue                                   # show draws where something changed
        hstr = ", ".join("kept" if h is None else f"{h:.2f} m" for h in hs)
        raw.append((panel(pts), panel(new, ghost=pts[removed], negatives=new[kept_pole])))
        caps.append((f"cloud {i}: {len(cols)} pole column(s), original",
                     f"after variation: stub heights {hstr}"))
        used += 1

    bg = raw[0][0][0, 0].astype(float)
    ys0, ys1, xs0, xs1 = 10**9, -1, 10**9, -1
    for pair in raw:
        for img in pair:
            m = np.abs(img.astype(float) - bg).sum(axis=2) > 12
            ys, xs = np.where(m)
            ys0, ys1 = min(ys0, ys.min()), max(ys1, ys.max())
            xs0, xs1 = min(xs0, xs.min()), max(xs1, xs.max())
    pad = 12
    ys0, xs0 = max(0, ys0 - pad), max(0, xs0 - pad)
    ys1 += pad; xs1 += pad

    def lab(img, text):
        im = Image.fromarray(img[ys0:ys1, xs0:xs1])
        out = Image.new("RGB", (im.width, im.height + 26), "white")
        out.paste(im, (0, 26))
        ImageDraw.Draw(out).text((10, 8), text, fill=(0, 0, 0))
        return np.asarray(out)

    rows = [np.hstack([lab(p[0], c[0]), lab(p[1], c[1])]) for p, c in zip(raw, caps)]
    Image.fromarray(np.vstack(rows)).save(a.out)
    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
