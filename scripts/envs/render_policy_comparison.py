#!/usr/bin/env python3
"""Qualitative target comparison: P1 (regression/margin), scoring_v1 (scoring/tight),
P2 (scoring/margin) on the SAME real clouds, one Open3D iso panel per cloud, wrapped in a
self-contained HTML slider (all images inlined as base64 - works offline at the crane).

Cloud sets:
  * 26 recorded policy_debug clouds from the 2026-08-03 gap-failure BC run (node pipeline,
    tight crop @1024 - the margin policies see them zero-padded to 2048; noted in the caption)
  * 26 raw bag clouds from the 3 baseline trials; per-policy pipeline is respected
    (v1: tight crop FPS 1024, P1/P2: margin-0.5 crop FPS 2048); the DISPLAYED cloud is the
    margin crop (superset).

Colors: P2 = red, scoring_v1 = magenta, P1 = cyan. Line through each sphere = commanded yaw.

    python3 scripts/envs/render_policy_comparison.py --out docs/figures/policy_comparison.html
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import sys

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from open3d.visualization import rendering

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring_head import ScoringGraspPolicy  # noqa: E402

B_MIN = np.array([-5.364, -1.684, -1.30], dtype=np.float32)
B_MAX = np.array([-3.364, 5.316, 0.10], dtype=np.float32)
SH = -0.55
RECORDED = "logs/bc_pointcloud/bc_aug1v2_dig25/policy_debug/run_20260803_142901"
RAW = "logs/real_raw_clouds"
W, H = 1000, 720
_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material


class RegressionBC(nn.Module):
    """Minimal standalone mirror of train_bc_pointcloud's policy (encoder.mlp1/fc + actor_mlp)."""

    def __init__(self, action_dim=5):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ELU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ELU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ELU())
        self.encoder.fc = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ELU())
        self.actor_mlp = nn.Sequential(nn.Linear(256, 128), nn.ELU(),
                                       nn.Linear(128, 64), nn.ELU(), nn.Linear(64, action_dim))

    def forward(self, x):                      # (B, N, 3)
        b, n, _ = x.shape
        f = self.encoder.mlp1(x.reshape(b * n, 3)).view(b, n, -1)
        g = self.encoder.fc(f.max(dim=1).values)
        return self.actor_mlp(g)


def dec5(a):
    xyz = B_MIN + (np.tanh(a[:3]) + 1.0) / 2.0 * (B_MAX - B_MIN)
    return float(xyz[0]), float(xyz[1]), float(xyz[2]), float(0.5 * np.arctan2(a[4], a[3]))


def fps(p, k, seed=0):
    if len(p) <= k:
        out = np.zeros((k, 3), np.float32); out[:len(p)] = p; return out
    rng = np.random.default_rng(seed)
    if len(p) > 20000:
        p = p[rng.choice(len(p), 20000, replace=False)]
    idx = np.empty(k, np.int64); idx[0] = rng.integers(len(p))
    d = np.linalg.norm(p - p[idx[0]], axis=1)
    for i in range(1, k):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(p - p[idx[i]], axis=1))
    return p[idx].astype(np.float32)


def crop(p, g):
    m = ((p[:, 0] >= B_MIN[0] - g) & (p[:, 0] <= B_MAX[0] + g) &
         (p[:, 1] >= B_MIN[1] - g) & (p[:, 1] <= B_MAX[1] + g) &
         (p[:, 2] >= B_MIN[2] - g) & (p[:, 2] <= B_MAX[2]))
    return p[m]


def pad_to(p, k):
    out = np.zeros((k, 3), np.float32); n = min(len(p), k); out[:n] = p[:n]; return out


def height_colors(z, lo=-1.50, hi=-0.40):
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=1)


_R = {"r": None}
COL = {"P2": [0.85, 0.05, 0.05], "v1": [0.85, 0.05, 0.75], "P1": [0.0, 0.65, 0.85]}


def panel(pts, targets, margin):
    r = _R["r"]
    if r is None:
        r = rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1, 1, 1, 1])
        _R["r"] = r
    sc = r.scene
    sc.clear_geometry()
    mp = _Mat(); mp.shader = "defaultUnlit"; mp.point_size = 3.2
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 3.0
    mm = _Mat(); mm.shader = "defaultLit"

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2]) * 0.55 + 0.35)
    sc.add_geometry("pcd", pcd, mp)
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(B_MIN, B_MAX))
    box.paint_uniform_color([0.35, 0.55, 0.9])
    sc.add_geometry("box", box, ml)
    if margin > 0:
        wl, wh = B_MIN - margin, B_MAX.copy(); wh[:2] += margin
        wb = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(wl, wh))
        wb.paint_uniform_color([0.95, 0.7, 0.3])
        sc.add_geometry("wbox", wb, ml)

    for name, (x, y, z, yaw) in targets.items():
        c = COL[name]
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.11)
        s.translate([x, y, z]); s.paint_uniform_color(c); s.compute_vertex_normals()
        sc.add_geometry("s" + name, s, mm)
        e1 = [x + 0.55 * np.cos(yaw), y + 0.55 * np.sin(yaw), z]
        e2 = [x - 0.55 * np.cos(yaw), y - 0.55 * np.sin(yaw), z]
        ln = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector([e1, [x, y, z], e2]),
                                  lines=o3d.utility.Vector2iVector([[0, 1], [1, 2]]))
        ln.paint_uniform_color(c)
        sc.add_geometry("l" + name, ln, ml)

    c = 0.5 * (B_MIN + B_MAX); c[2] = B_MIN[2] + 0.3 * (B_MAX[2] - B_MIN[2])
    rad = float(np.linalg.norm(B_MAX - B_MIN))
    r.setup_camera(58.0, c, c + np.array([1.0, -1.0, 0.55]) * (0.6 * rad), [0, 0, 1])
    img = np.asarray(r.render_to_image())[:, :, :3]
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/policy_comparison.html")
    a = ap.parse_args()

    v1 = ScoringGraspPolicy(num_points=1024)
    v1.load_state_dict(torch.load("logs/bc_pointcloud/scoring_v1/scoring_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    p2 = ScoringGraspPolicy(num_points=2048)
    p2.load_state_dict(torch.load("logs/bc_pointcloud/scoring_margin05_2048/scoring_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    p1 = RegressionBC()
    p1.load_state_dict(torch.load("logs/bc_pointcloud/bc_margin05_2048_reg/bc_pointcloud_policy.pt",
                                  map_location="cpu", weights_only=False)["model_state_dict"])
    for m in (v1, p2, p1):
        m.eval()

    def targets_for(tight_cloud, margin_cloud):
        out = {}
        with torch.no_grad():
            t = v1.act(torch.from_numpy(pad_to(tight_cloud, 1024))[None])[0].numpy()
            out["v1"] = tuple(t)
            t = p2.act(torch.from_numpy(pad_to(margin_cloud, 2048))[None])[0].numpy()
            out["P2"] = tuple(t)
            out["P1"] = dec5(p1(torch.from_numpy(pad_to(margin_cloud, 2048))[None])[0].numpy())
        return out

    slides = []
    for f in sorted(glob.glob(os.path.join(RECORDED, "policy_debug_*.npz"))):
        d = np.load(f)
        pts = d["points"].astype(np.float32).copy(); pts[:, 1] -= float(d["rack_y_shift"])
        tg = targets_for(pts, pts)          # recorded clouds ARE the tight crop; margin unavailable
        cap = " | ".join(f"{k} y={v[1]:.2f} z={v[2]:.2f}" for k, v in tg.items())
        slides.append((f"gap-run {os.path.basename(f)[13:16]} (recorded, tight pipeline)",
                       cap, panel(pts, tg, 0.0)))
        print("done", f)
    for f in sorted(glob.glob(os.path.join(RAW, "*.npz"))):
        p = np.load(f)["points"].astype(np.float32).copy()
        p[:, 1] -= SH
        tight = fps(crop(p, 0.0), 1024)
        marg = fps(crop(p, 0.5), 2048)
        tg = targets_for(tight, marg)
        cap = " | ".join(f"{k} y={v[1]:.2f} z={v[2]:.2f}" for k, v in tg.items())
        slides.append((f"trial {os.path.basename(f)[:-4]} (raw bag, per-policy pipeline)",
                       cap, panel(crop(p, 0.5), tg, 0.5)))
        print("done", f)

    imgs = ",\n".join("{t:%r,c:%r,d:%r}" % (t, c, d) for t, c, d in slides)
    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>P1 vs scoring_v1 vs P2 - real-cloud targets</title>
<style>body{font-family:sans-serif;margin:12px;background:#fafafa}
img{max-width:100%;border:1px solid #ccc}.leg span{padding:2px 8px;margin-right:8px;color:#fff;border-radius:3px}
#cap{font-family:monospace;margin:6px 0}</style></head><body>
<h3>Target comparison on real clouds - slide or use arrow keys</h3>
<div class="leg"><span style="background:#d90d0d">P2 scoring+margin</span>
<span style="background:#d90dbf">scoring_v1 tight</span>
<span style="background:#00a6d9">P1 regression+margin</span>
blue box = action box, orange = margin crop</div>
<div id="ttl"></div><div id="cap"></div>
<input id="sl" type="range" min="0" max="NMAX" value="0" style="width:100%">
<div><img id="im"></div>
<script>const S=[SLIDES];
const sl=document.getElementById('sl'),im=document.getElementById('im');
function show(i){im.src='data:image/jpeg;base64,'+S[i].d;
document.getElementById('ttl').innerHTML='<b>'+(+i+1)+'/'+S.length+'</b>  '+S[i].t;
document.getElementById('cap').textContent=S[i].c;}
sl.oninput=e=>show(e.target.value);
document.onkeydown=e=>{if(e.key=='ArrowRight')sl.value=Math.min(+sl.value+1,NMAX);
if(e.key=='ArrowLeft')sl.value=Math.max(+sl.value-1,0);show(sl.value);};
show(0);</script></body></html>"""
            .replace("NMAX", str(len(slides) - 1))
            .replace("SLIDES", imgs))
    with open(a.out, "w") as fo:
        fo.write(html)
    print(f"wrote {a.out} ({len(slides)} slides, {os.path.getsize(a.out)//1024} KB)")


if __name__ == "__main__":
    main()
