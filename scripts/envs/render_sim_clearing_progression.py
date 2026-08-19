#!/usr/bin/env python3
"""Pile-clearing progression for one simulated episode: Open3D height-heatmap panels, one
per selected grasp cycle, each with the target the policy commanded on that cycle.

Sim counterpart of render_clearing_progression.py (the real-trial version). Input is a
decisions_*.npz written by play_bc_pointcloud.py --save_decisions: each record holds the
policy input cloud (pre-grasp, base frame), the decoded target, the action bounds, and the
measured post-cycle rack count, so the panels can be annotated with exact log counts
instead of the surface-occupancy proxy the real figure has to use.

Visual language matches the real figure (jet-style height ramp pinned to a fixed absolute
range, blue action box, red target sphere with a yaw line) but the camera is the front
elevation of the sim pile-shapes figure: straight on to the rack with a 20 degree downward
tilt, framed once on the action box so every panel and every policy is registered.

    python3 scripts/envs/render_sim_clearing_progression.py \
        --npz logs/sim_eval/thesis_progression/expert/decisions_*.npz --list
    python3 scripts/envs/render_sim_clearing_progression.py \
        --npz .../decisions_*.npz --env 0 --episode 0 --out docs/figures/sim_progression_expert.png
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw

W, H = 1180, 660
TILT_DEG = 20.0
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material

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


def load_episode(npz_path, env_id, episode):
    d = np.load(npz_path, allow_pickle=True)
    m = (d["env"] == env_id) & (d["episode"] == episode)
    idx = np.where(m)[0]
    idx = idx[np.argsort(d["cycle"][idx])]
    total = None
    recs = []
    for j in idx:
        p = d["points"][j].astype(np.float64)
        v = p[np.abs(p).sum(-1) > 1e-6]
        recs.append({
            "cycle": int(d["cycle"][j]),
            "pts": v,
            "target": d["target"][j].astype(float),
            "bmin": d["bounds_min"][j].astype(float),
            "bmax": d["bounds_max"][j].astype(float),
            "grasped": int(d["logs_grasped"][j]),
            "in_rack": int(d["logs_in_rack"][j]) if "logs_in_rack" in d.files else -1,
        })
    # pre-grasp rack count for each panel: the previous cycle's measured post-count
    if recs and recs[0]["in_rack"] >= 0:
        total = recs[0]["in_rack"] + recs[0]["grasped"]
        prev = total
        for r in recs:
            r["before"] = prev
            prev = r["in_rack"]
    return recs, total


def episode_table(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    print(f"{npz_path}")
    for e in sorted(set(d["env"].tolist())):
        for ep in sorted(set(d["episode"][d["env"] == e].tolist())):
            m = (d["env"] == e) & (d["episode"] == ep)
            cyc = d["cycle"][m]
            fin = int(d["logs_in_rack"][m][np.argmax(cyc)]) if "logs_in_rack" in d.files else -1
            print(f"  env {e} episode {ep}: {m.sum()} cycles, final logs in rack {fin}")


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

    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(rec["bmin"], rec["bmax"]))
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

    r.setup_camera(26.0, ctr, eye, [0, 0, 1])
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
    ap.add_argument("--npz", required=True, help="decisions_*.npz path (glob ok, takes the newest)")
    ap.add_argument("--env", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cycles", default="", help="comma-separated cycle numbers (overrides even spacing)")
    ap.add_argument("--zhi", type=float, default=0.0, help="fix the top of the height ramp (0 = from data); "
                    "set it when rendering a policy pair so both figures share one colour scale")
    ap.add_argument("--list", action="store_true", help="print per-episode cycle counts and exit")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = sorted(glob.glob(a.npz))
    assert paths, f"no npz matches {a.npz}"
    npz = paths[-1]

    if a.list:
        episode_table(npz)
        return

    recs, total = load_episode(npz, a.env, a.episode)
    assert recs, f"no records for env {a.env} episode {a.episode}"
    n = len(recs)
    if a.cycles:
        want = [int(c) for c in a.cycles.split(",")]
        sel = [r for r in recs if r["cycle"] in want]
    else:
        k = min(a.cols * a.rows, n)
        idx = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        sel = [recs[i] for i in idx]

    # one camera for the whole figure, framed on the action box (identical across policies
    # run on the same seed): the front elevation of the sim pile-shapes figure
    bmin, bmax = recs[0]["bmin"], recs[0]["bmax"]
    ctr = 0.5 * (bmin + bmax)
    ctr[2] = bmin[2] + 0.35   # aim just above the rack floor so the pile fills the frame
    t = np.deg2rad(TILT_DEG)
    dirv = np.array([-np.cos(t), 0.0, np.sin(t)])
    eye = ctr + dirv * 11.0
    cam = (ctr, eye)

    bed = float(bmin[2])
    z_hi = a.zhi if a.zhi > 0 else float(np.percentile(np.vstack([r["pts"] for r in recs])[:, 2], 99.5)) - bed
    lo, hi = bed, bed + z_hi

    raw = [panel(r, cam, lo, hi) for r in sel]

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
        if r.get("before") is not None and total:
            sub = f"logs in rack {r['before']} of {total}"
            if r["grasped"] == 0:
                sub += ", empty cycle"
        else:
            sub = ""
        panels.append(label(img[y0:y1, x0:x1], f"cycle {r['cycle']} of {recs[-1]['cycle']}", sub))

    fig = grid(panels, a.cols)
    fig = np.vstack([fig, colorbar(fig.shape[1], lo, hi)])
    out = a.out or f"docs/figures/sim_progression_env{a.env}_ep{a.episode}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(fig).save(out)
    print(f"wrote {out} {fig.shape} from {len(panels)} of {n} cycles (z ramp {lo:.2f}..{hi:.2f})")
    print("  cycles:", [r["cycle"] for r in sel])
    print("  before:", [r.get("before") for r in sel])


if __name__ == "__main__":
    main()
