#!/usr/bin/env python3
"""Pile-clearing progression for one real trial: Open3D height-heatmap panels, one per
selected cycle, each with the target the policy commanded on that cycle.

Same visual language as the interactive HTML comparisons (render_policy_comparison.py):
jet-style height ramp, blue action box, red target sphere with a yaw line through it.
Two things differ, and both are required for a progression to be readable:

  * the CAMERA is identical in every panel (framed once on the union of all clouds in the
    trial), so the pile shrinking is the only thing that moves;
  * the height ramp is pinned to a FIXED absolute range, so a colour means the same height
    in cycle 1 and cycle 32.

Each panel is annotated with the fraction of observed points sitting more than 0.25 m above
the rack floor. That is a clearing measure taken from the same cloud that is drawn (the
clouds are FPS-subsampled to a fixed point count, so the fraction is a surface-occupancy
proxy, not a log count).

    python3 scripts/envs/render_clearing_progression.py --trial scoring_double
    python3 scripts/envs/render_clearing_progression.py --trial baseline_single --cols 2 --rows 4
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw

LOGS = "logs"
OUT = "docs/figures/real_trials"
W, H = 1180, 660
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material

# training-coords action box; physical box = this + rack_y_shift in y
A_MIN = np.array([-5.364, -1.684, -1.30])
A_MAX = np.array([-3.364, 5.316, 0.10])
BED = -1.30
ABOVE = 0.25           # a point this far above the rack floor counts as pile

TRIALS = {
    "baseline_double": ("Baseline, double mound", f"{LOGS}/crane_policy_debug/run_20260803_155559"),
    "baseline_jagged": ("Baseline, jagged", f"{LOGS}/crane_policy_debug/run_20260803_174017"),
    "baseline_single": ("Baseline, single mound", f"{LOGS}/crane_policy_debug/run_20260803_204034"),
    "scoring_single": ("Scoring, single mound",
                       f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_153036"),
    "scoring_double": ("Scoring, double mound",
                       f"{LOGS}/bc_pointcloud/scoring_margin05_2048_c/policy_debug/run_20260805_194008"),
}

_R = {"r": None}


def renderer():
    if _R["r"] is None:
        r = rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1, 1, 1, 1])
        _R["r"] = r
    return _R["r"]


def height_colors(z, lo, hi):
    """Jet-style ramp as used in the HTML viewers, desaturated so the target stays foreground."""
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                    np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                    np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=1)
    return rgb * 0.62 + 0.30


def load_trial(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "decisions.jsonl"))]
    gaze = [r for r in recs if r["kind"] == "gaze" and r.get("target")]
    out = []
    for r in gaze:
        z = np.load(os.path.join(d, r["npz"]))
        p = z["points"].astype(np.float64)
        v = p[np.abs(p).sum(-1) > 1e-6]
        out.append({"cycle": r["cycle"], "target": r["target"], "support": r.get("support"),
                    "pts": v, "shift": float(z["rack_y_shift"]),
                    "frac": float((v[:, 2] > BED + ABOVE).mean())})
    return out


def panel(rec, cam, lo, hi):
    ctr, eye = cam
    r = renderer()
    sc = r.scene
    sc.clear_geometry()
    pts = rec["pts"]
    mp = _Mat(); mp.shader = "defaultUnlit"
    mp.point_size = float(np.clip(11000.0 / max(len(pts), 1), 4.5, 8.0))
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 2.0
    mt = _Mat(); mt.shader = "unlitLine"; mt.line_width = 3.5
    mm = _Mat(); mm.shader = "defaultUnlit"   # flat marker; a lit sphere renders muddy under Filament tone mapping

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2], lo, hi))
    sc.add_geometry("pcd", pcd, mp)

    a_lo = A_MIN + np.array([0.0, rec["shift"], 0.0])
    a_hi = A_MAX + np.array([0.0, rec["shift"], 0.0])
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(a_lo, a_hi))
    box.paint_uniform_color([0.42, 0.58, 0.88])
    sc.add_geometry("box", box, ml)

    x, y, z, yaw = rec["target"]
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.13, resolution=16)
    s.translate([x, y, z]); s.paint_uniform_color([0.86, 0.08, 0.08]); s.compute_vertex_normals()
    sc.add_geometry("tgt", s, mm)
    e1 = [x + 0.50 * np.cos(yaw), y + 0.50 * np.sin(yaw), z]
    e2 = [x - 0.50 * np.cos(yaw), y - 0.50 * np.sin(yaw), z]
    ln = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([e1, e2]),
                              lines=o3d.utility.Vector2iVector([[0, 1]]))
    ln.paint_uniform_color([0.86, 0.08, 0.08])
    sc.add_geometry("yaw", ln, mt)

    r.setup_camera(50.0, ctr, eye, [0, 0, 1])
    return whiten_bg(np.asarray(r.render_to_image())[:, :, :3])


def whiten_bg(img, tol=10):
    """Open3D 0.13 has no post-processing toggle, so Filament tone-maps the white background
    to grey. The background is flat, so snapping pixels near the corner colour back to white
    is exact and leaves the geometry untouched."""
    out = img.copy()
    bg = out[0, 0].astype(np.int16)
    m = np.abs(out.astype(np.int16) - bg).max(axis=2) <= tol
    out[m] = 255
    return out


def label(img, text, sub):
    im = Image.fromarray(img)
    top = 40
    out = Image.new("RGB", (im.width, im.height + top), "white")
    out.paste(im, (0, top))
    d = ImageDraw.Draw(out)
    d.text((12, 8), text, fill=(0, 0, 0))
    d.text((12, 23), sub, fill=(80, 80, 80))
    return np.asarray(out)


def colorbar(w, lo, hi):
    """Shared height legend for the whole figure."""
    h = 46
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    bw, bh = min(320, w - 240), 13
    bx, by = 12, 18
    for i in range(bw):
        c = height_colors(np.array([lo + (hi - lo) * i / (bw - 1)]), lo, hi)[0]
        d.line([(bx + i, by), (bx + i, by + bh)], fill=tuple(int(255 * v) for v in c))
    d.rectangle([bx, by, bx + bw, by + bh], outline=(120, 120, 120))
    d.text((bx, by - 14), "height above rack floor", fill=(0, 0, 0))
    d.text((bx, by + bh + 3), "0.00 m", fill=(70, 70, 70))
    d.text((bx + bw - 42, by + bh + 3), f"{hi - lo:.2f} m", fill=(70, 70, 70))
    d.text((bx + bw + 34, by + 1),
           "red sphere = commanded target, line = grapple yaw, blue box = action volume",
           fill=(70, 70, 70))
    return np.asarray(im)


def grid(imgs, cols, gap=14):
    rows = []
    for i in range(0, len(imgs), cols):
        chunk = imgs[i:i + cols]
        while len(chunk) < cols:
            chunk.append(np.full_like(chunk[0], 255))
        sep = np.full((chunk[0].shape[0], gap, 3), 255, np.uint8)
        row = chunk[0]
        for c in chunk[1:]:
            row = np.hstack([row, sep, c])
        rows.append(row)
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.hstack([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)])
        out.append(r)
    sep = np.full((gap, w, 3), 255, np.uint8)
    stacked = out[0]
    for r in out[1:]:
        stacked = np.vstack([stacked, sep, r])
    return stacked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="scoring_double", choices=sorted(TRIALS))
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cycles", default="", help="comma-separated cycle numbers (overrides even spacing)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    title, d = TRIALS[a.trial]
    recs = load_trial(d)
    n = len(recs)
    if a.cycles:
        want = [int(c) for c in a.cycles.split(",")]
        sel = [r for r in recs if r["cycle"] in want]
    else:
        k = min(a.cols * a.rows, n)
        idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        sel = [recs[i] for i in idx]

    # one camera and one colour scale for the whole figure
    allp = np.vstack([r["pts"] for r in recs])
    shift = recs[0]["shift"]
    lo3 = np.minimum(allp.min(0), A_MIN + [0, shift, 0])
    hi3 = np.maximum(allp.max(0), A_MAX + [0, shift, 0])
    ctr = 0.5 * (lo3 + hi3)
    rad = float(np.linalg.norm(hi3 - lo3))
    dirv = np.array([1.00, -0.28, 0.62]); dirv /= np.linalg.norm(dirv)
    cam = (ctr, ctr + dirv * 0.92 * rad)
    z_lo, z_hi = BED, float(np.percentile(np.vstack([r["pts"] for r in recs])[:, 2], 99.5))

    raw = [panel(r, cam, z_lo, z_hi) for r in sel]

    # crop every panel to one common content bbox so the panels stay registered
    y0, y1, x0, x1 = 10**9, -1, 10**9, -1
    for img in raw:
        m = img.min(axis=2) < 245
        ys, xs = np.where(m)
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    pad = 10
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(raw[0].shape[0], y1 + pad), min(raw[0].shape[1], x1 + pad)

    panels = []
    for img, r in zip(raw, sel):
        sup = f", support {r['support']}" if r.get("support") is not None else ""
        panels.append(label(img[y0:y1, x0:x1],
                            f"cycle {r['cycle']} of {recs[-1]['cycle']}",
                            f"pile fraction {r['frac']:.2f}{sup}"))

    fig = grid(panels, a.cols)
    fig = np.vstack([fig, colorbar(fig.shape[1], z_lo, z_hi)])
    out = a.out or f"{OUT}/clearing_progression_{a.trial}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(fig).save(out)
    print(f"wrote {out} {fig.shape} from {len(panels)} of {n} cycles")
    print("  cycles:", [r["cycle"] for r in sel])
    print("  frac  :", [round(r["frac"], 2) for r in sel])


if __name__ == "__main__":
    main()
