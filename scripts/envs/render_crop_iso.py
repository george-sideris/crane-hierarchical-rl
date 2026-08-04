#!/usr/bin/env python3
"""Open3D isometric renders of sim vs real clouds under the tight and widened crops.

Same visual conventions as calibration/render_run_video.py::_o3d_panel (white background, blue
action-box wireframe, base-frame axes) but the cloud is coloured by height so the bed plane, the
rails and the poles read at a glance - which is the whole point of the comparison.

    python3 scripts/envs/render_crop_iso.py --margin 0.5 --out docs/figures/crop_iso.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

B_MIN = np.array([-5.364, -1.684, -1.30])
B_MAX = np.array([-3.364, 5.316, 0.10])
RACK_Y_SHIFT = -0.55
DS = "logs/bc_pointcloud/bc_policy_aug1_v2"
W, H = 760, 620


def crop(p, g, shift=0.0):
    lo, hi = B_MIN + [0, shift, 0], B_MAX + [0, shift, 0]
    m = ((p[:, 0] >= lo[0] - g) & (p[:, 0] <= hi[0] + g) &
         (p[:, 1] >= lo[1] - g) & (p[:, 1] <= hi[1] + g) &
         (p[:, 2] >= lo[2] - g) & (p[:, 2] <= hi[2]))
    q = p[m].copy()
    q[:, 1] -= shift
    return q


def valid(p):
    return p[np.abs(p).sum(-1) > 1e-6]


def fps(p, k=1024, seed=0):
    """Farthest-point sampling, the same reduction the policy input goes through in BOTH domains.
    Comparing a stored (already-FPS'd) sim cloud against a RAW real cloud makes sim look sparse
    when the sim render is in fact ~4x denser in the crop; this makes the comparison honest."""
    if len(p) <= k:
        return p
    rng = np.random.default_rng(seed)
    idx = np.empty(k, dtype=np.int64)
    idx[0] = rng.integers(len(p))
    d = np.linalg.norm(p - p[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(p - p[idx[i]], axis=1))
    return p[idx]


def height_colors(z, lo=-1.50, hi=-0.40):
    """turbo-like ramp so the deck (dark blue) separates from poles (red) without matplotlib."""
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return np.stack([r, g, b], axis=1)


_R = {"r": None}
# open3d renamed Material -> MaterialRecord in 0.14; the host has 0.13, the container is newer
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material


def panel(pts: np.ndarray, margin: float):
    r = _R["r"]
    if r is None:
        r = rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1.0, 1.0, 1.0, 1.0])
        _R["r"] = r
    sc = r.scene
    sc.clear_geometry()

    mp = _Mat(); mp.shader = "defaultUnlit"; mp.point_size = 3.5
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 2.0

    if len(pts):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2]))
        sc.add_geometry("pcd", pcd, mp)

    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(B_MIN, B_MAX))
    box.paint_uniform_color([0.1, 0.5, 1.0])          # blue = the ACTION box (unchanged)
    sc.add_geometry("box", box, ml)
    if margin > 0:
        wl, wh = B_MIN - margin, B_MAX.copy()
        wh[:2] += margin                               # z_max is not widened
        wbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(wl, wh))
        wbox.paint_uniform_color([0.95, 0.55, 0.0])   # orange = the OBSERVATION crop
        sc.add_geometry("wbox", wbox, ml)

    c = 0.5 * (B_MIN + B_MAX)
    c[2] = B_MIN[2] + 0.3 * (B_MAX[2] - B_MIN[2])
    rad = float(np.linalg.norm(B_MAX - B_MIN))
    eye = c + np.array([1.0, -1.0, 0.55]) * (0.62 * rad)
    r.setup_camera(60.0, c, eye, [0.0, 0.0, 1.0])
    return np.asarray(r.render_to_image())[:, :, :3]


def label(img, text, sub):
    """draw a caption band using PIL (already a dependency of the video tooling)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 40], fill=(255, 255, 255))
    d.text((10, 6), text, fill=(0, 0, 0))
    d.text((10, 22), sub, fill=(90, 90, 90))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--figdir", default="docs/figures")
    ap.add_argument("--sim_raw", default=None,
                   help="raw sim render npy (calibration/out/sim_full_pcd.npy) -> like-for-like mode")
    ap.add_argument("--out", default="docs/figures/crop_iso.png")
    a = ap.parse_args()

    tight = np.load(os.path.join(DS, "pointclouds.npy"), mmap_mode="r")
    full = np.load(os.path.join(DS, "pointclouds_full.npy"), mmap_mode="r")
    n = np.array([len(valid(np.asarray(tight[i]))) for i in range(0, len(tight), 23)])
    idx = np.arange(0, len(tight), 23)
    pick = [int(idx[np.argsort(np.abs(n - t))[0]]) for t in (1024, 90)]

    if a.sim_raw:
        # LIKE-FOR-LIKE: raw sim render vs raw real cloud, then both reduced by the same FPS
        # that the policy input actually applies. Sim is ~4x DENSER in the crop than real.
        sr = np.load(a.sim_raw)
        rp = np.load(os.path.join(a.figdir, "real_full.npz"))["points"]
        cells = [(crop(sr, 0.0), 0.0, "SIM  tight crop", "raw render"),
                 (crop(sr, a.margin), a.margin, f"SIM  crop_margin {a.margin}", "raw render"),
                 (crop(rp, 0.0, RACK_Y_SHIFT), 0.0, "REAL tight crop", "raw ZED"),
                 (crop(rp, a.margin, RACK_Y_SHIFT), a.margin,
                  f"REAL crop_margin {a.margin}", "raw ZED")]
        rows = []
        for tag, red in (("RAW", lambda q: q), ("FPS 1024 (policy input)", lambda q: fps(q))):
            imgs = [label(panel(red(p), g), f"{t}  -  {tag}", f"{s} | {len(red(p))} pts")
                    for p, g, t, s in cells]
            rows.append(np.concatenate(imgs, axis=1))
        out = np.concatenate(rows, axis=0)
    else:
        rows = []
        for si, (stage, rf) in enumerate([("FULL PILE", "real_full.npz"),
                                          ("ENDGAME", "real_end.npz")]):
            rp = np.load(os.path.join(a.figdir, rf))["points"]
            cells = [
                (valid(np.asarray(tight[pick[si]])), 0.0, "SIM  tight crop",
                 "what BC trains on today"),
                (crop(valid(np.asarray(full[pick[si]])), a.margin), a.margin,
                 f"SIM  crop_margin {a.margin}", "proposed"),
                (crop(rp, 0.0, RACK_Y_SHIFT), 0.0, "REAL tight crop",
                 "what deployment feeds it"),
                (crop(rp, a.margin, RACK_Y_SHIFT), a.margin,
                 f"REAL crop_margin {a.margin}", "proposed"),
            ]
            imgs = [label(panel(p, g), f"{t}  -  {stage}", f"{s} | {len(p)} pts")
                    for p, g, t, s in cells]
            rows.append(np.concatenate(imgs, axis=1))
        out = np.concatenate(rows, axis=0)

    from PIL import Image
    Image.fromarray(out).save(a.out)
    print(f"wrote {a.out}  ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
