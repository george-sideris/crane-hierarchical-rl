#!/usr/bin/env python3
"""Open3D iso renders of the real-trial commanded targets over the recorded clouds.

Replaces the matplotlib scatter version: same data, but the crane bed reads as a surface
instead of a point fog, and the camera is framed on the actual cloud bounds (the earlier
Open3D panels were framed on the nominal action box, which left them off-centre).

Coordinates: decisions.jsonl targets and policy_debug_*.npz points share ONE frame (the
recorded/physical frame). bounds_min/bounds_max in the npz are the CROP bounds with
rack_y_shift already applied, so the margin runs store a wider box than the action box;
the action box is drawn separately from the training box + shift.

Outputs (docs/figures/real_trials/):
  targets_3d_<trial>.png    per-trial overview: first-gaze cloud + every commanded target
  targets_3d.png            2 panels (baseline single, scoring double) for the chapter
  targets_3d_phases.png     early cloud+targets vs late cloud+targets, same two trials

    python3 scripts/envs/render_trial_targets_3d.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import open3d as o3d
from matplotlib import cm
from open3d.visualization import rendering
from PIL import Image, ImageDraw

LOGS = "logs"
OUT = "docs/figures/real_trials"
W, H = 1300, 900
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material

# training-coords action box (physical box = this + rack_y_shift in y)
A_MIN = np.array([-5.364, -1.684, -1.30])
A_MAX = np.array([-3.364, 5.316, 0.10])

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


def load_trial(d):
    recs = [json.loads(l) for l in open(os.path.join(d, "decisions.jsonl"))]
    gaze = [r for r in recs if r["kind"] == "gaze" and r.get("target")]
    return gaze


def cloud_of(d, rec):
    z = np.load(os.path.join(d, rec["npz"]))
    return (z["points"].astype(np.float64), z["bounds_min"].astype(np.float64),
            z["bounds_max"].astype(np.float64), float(z["rack_y_shift"]))


def height_colors(z, lo=None, hi=None):
    lo = np.percentile(z, 2) if lo is None else lo
    hi = np.percentile(z, 98) if hi is None else hi
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    # pastel height ramp: enough shading to read the pile surface, desaturated so the
    # saturated target spheres stay the foreground element
    return cm.viridis(t)[:, :3] * 0.45 + 0.48


def box_lines(lo, hi, color):
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(np.asarray(lo, float), np.asarray(hi, float)))
    ls.paint_uniform_color(color)
    return ls


def panel(pts, targets, crop_lo, crop_hi, shift, sphere_r=0.10, show_path=True):
    """One iso render: cloud + action/crop boxes + colour-coded targets."""
    r = renderer()
    sc = r.scene
    sc.clear_geometry()
    mp = _Mat(); mp.shader = "defaultUnlit"
    mp.point_size = float(np.clip(12000.0 / max(len(pts), 1), 5.0, 9.0))
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 2.0
    mlt = _Mat(); mlt.shader = "unlitLine"; mlt.line_width = 1.5
    mm = _Mat(); mm.shader = "defaultLit"

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2]))
    sc.add_geometry("pcd", pcd, mp)

    a_lo = A_MIN + np.array([0.0, shift, 0.0])
    a_hi = A_MAX + np.array([0.0, shift, 0.0])
    sc.add_geometry("abox", box_lines(a_lo, a_hi, [0.35, 0.55, 0.90]), ml)
    if np.any(np.abs(crop_lo - a_lo) > 1e-3) or np.any(np.abs(crop_hi - a_hi) > 1e-3):
        sc.add_geometry("cbox", box_lines(crop_lo, crop_hi, [0.95, 0.70, 0.25]), mlt)
    n = len(targets)
    cols = cm.plasma(np.linspace(0.06, 0.94, max(n, 2)))[:, :3]
    for k, (x, y, z, yaw) in enumerate(targets):
        c = list(cols[k])
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r, resolution=14)
        s.translate([x, y, z]); s.paint_uniform_color(c); s.compute_vertex_normals()
        sc.add_geometry(f"s{k}", s, mm)
        e1 = [x + 0.45 * np.cos(yaw), y + 0.45 * np.sin(yaw), z]
        e2 = [x - 0.45 * np.cos(yaw), y - 0.45 * np.sin(yaw), z]
        ln = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([e1, e2]),
                                  lines=o3d.utility.Vector2iVector([[0, 1]]))
        ln.paint_uniform_color(c)
        sc.add_geometry(f"y{k}", ln, mlt)
    if show_path and n > 1:
        path = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([[t[0], t[1], t[2]] for t in targets]),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(n - 1)]))
        path.paint_uniform_color([0.45, 0.45, 0.45])
        sc.add_geometry("path", path, mlt)

    # frame on the data actually drawn, not the nominal box
    allp = np.vstack([pts, np.array([[t[0], t[1], t[2]] for t in targets])])
    lo, hi = allp.min(0), allp.max(0)
    ctr = 0.5 * (lo + hi)
    rad = float(np.linalg.norm(hi - lo))
    # look down the short (x) axis so the bed's 7 m length spreads across the frame:
    # target variation is almost entirely in y, and an oblique view collapses it
    d = np.array([1.00, -0.28, 0.62]); d /= np.linalg.norm(d)
    eye = ctr + d * 0.95 * rad
    r.setup_camera(50.0, ctr, eye, [0, 0, 1])
    return np.asarray(r.render_to_image())[:, :, :3]


def crop_white(img, pad=10):
    bg = img[0, 0].astype(float)
    m = np.abs(img.astype(float) - bg).sum(axis=2) > 12
    ys, xs = np.where(m)
    if not len(ys):
        return img
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def annotate(img, title, sub, n_targets):
    """Title bar + a plasma colour ramp legend for cycle order."""
    im = Image.fromarray(img)
    top = 62
    out = Image.new("RGB", (im.width, im.height + top), "white")
    out.paste(im, (0, top))
    d = ImageDraw.Draw(out)
    d.text((12, 8), title, fill=(0, 0, 0))
    d.text((12, 24), sub, fill=(70, 70, 70))
    bar_w, bar_h, bx, by = 250, 12, im.width - 300, 16
    for i in range(bar_w):
        c = cm.plasma(0.06 + 0.88 * i / (bar_w - 1))[:3]
        d.line([(bx + i, by), (bx + i, by + bar_h)], fill=tuple(int(255 * v) for v in c))
    d.rectangle([bx, by, bx + bar_w, by + bar_h], outline=(120, 120, 120))
    d.text((bx, by + bar_h + 4), "cycle 1", fill=(70, 70, 70))
    d.text((bx + bar_w - 46, by + bar_h + 4), f"cycle {n_targets}", fill=(70, 70, 70))
    return np.asarray(out)


def stack_h(imgs, gap=16):
    h = max(i.shape[0] for i in imgs)
    ims = []
    for i in imgs:
        if i.shape[0] < h:
            pad = np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)
            i = np.vstack([i, pad])
        ims.append(i)
    sep = np.full((h, gap, 3), 255, np.uint8)
    out = ims[0]
    for i in ims[1:]:
        out = np.hstack([out, sep, i])
    return out


def stack_v(imgs, gap=16):
    w = max(i.shape[1] for i in imgs)
    ims = []
    for i in imgs:
        if i.shape[1] < w:
            pad = np.full((i.shape[0], w - i.shape[1], 3), 255, np.uint8)
            i = np.hstack([i, pad])
        ims.append(i)
    sep = np.full((gap, w, 3), 255, np.uint8)
    out = ims[0]
    for i in ims[1:]:
        out = np.vstack([out, sep, i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    overview = {}
    for key, (label, d) in TRIALS.items():
        gaze = load_trial(d)
        pts, clo, chi, shift = cloud_of(d, gaze[0])
        tg = [g["target"] for g in gaze]
        img = crop_white(panel(pts, tg, clo, chi, shift))
        img = annotate(img, f"{label}: {len(tg)} commanded targets",
                       "first-gaze cloud (full pile); blue = action box, orange = crop margin", len(tg))
        Image.fromarray(img).save(f"{a.out}/targets_3d_{key}.png")
        overview[key] = img
        print("wrote", f"{a.out}/targets_3d_{key}.png", img.shape)

    pair = ["baseline_single", "scoring_double"]
    Image.fromarray(stack_h([overview[k] for k in pair])).save(f"{a.out}/targets_3d.png")
    Image.fromarray(stack_v([overview[k] for k in ["baseline_double", "baseline_jagged"]] +
                            [overview["scoring_single"]])).save(f"{a.out}/targets_3d_all.png")

    # early vs late: does the target set migrate to structure as the pile empties?
    phases = []
    for key in pair:
        label, d = TRIALS[key]
        gaze = load_trial(d)
        n = len(gaze)
        k = max(2, n // 3)
        for tag, sel, cloud_rec in [("first third", gaze[:k], gaze[0]),
                                    ("final third", gaze[-k:], gaze[-1])]:
            pts, clo, chi, shift = cloud_of(d, cloud_rec)
            img = crop_white(panel(pts, [g["target"] for g in sel], clo, chi, shift))
            img = annotate(img, f"{label}: {tag} ({len(sel)} targets)",
                           f"cloud recorded at cycle {cloud_rec['cycle']}", len(sel))
            phases.append(img)
    Image.fromarray(stack_v([stack_h(phases[:2]), stack_h(phases[2:])])).save(
        f"{a.out}/targets_3d_phases.png")
    print("wrote", f"{a.out}/targets_3d.png and targets_3d_phases.png")


if __name__ == "__main__":
    main()
