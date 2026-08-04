#!/usr/bin/env python3
"""Make a TAPERED log USD by deforming a COPY of the existing log asset.

Real logs taper ~1 cm of diameter per metre, so a 2.4 m log is ~2 cm fatter at the butt
than at the tip. Perfect cylinders nest into a near-optimal hex packing, which is the
leading suspect for sim grasping ~22 logs per bite where the real grapple manages ~13.

This deforms the SOURCE asset rather than authoring a new mesh, so materials/textures,
physics APIs (mass, collision approximation) and prim structure are all preserved --
only the vertex positions change. Each mesh vertex has its transverse (X,Z) coordinates
scaled by a factor that varies linearly along the log's axis (local Y).

Run INSIDE the container:
    /workspace/isaaclab/isaaclab.sh -p /workspace/crane_testbed/assets/make_tapered_log.py

Then collect with:
    --log_usd /workspace/crane_testbed/assets/scenes/testbed_log_taper.usd
"""

import argparse
import os
import shutil

# pxr lives inside Isaac's Kit runtime, so the app must start before it can be imported.
from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, Gf


def taper_stage(dst: str, butt_scale: float, tip_scale: float, butt_at_plus_y: bool) -> None:
    stage = Usd.Stage.Open(dst)
    n_meshes = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts_attr = mesh.GetPointsAttr()
        pts = pts_attr.Get()
        if not pts:
            continue
        ys = [p[1] for p in pts]
        y0, y1 = min(ys), max(ys)
        span = max(y1 - y0, 1e-9)
        lo, hi = (tip_scale, butt_scale) if butt_at_plus_y else (butt_scale, tip_scale)
        new = []
        for p in pts:
            t = (p[1] - y0) / span              # 0 at -Y end, 1 at +Y end
            f = lo + (hi - lo) * t              # transverse scale at this station
            new.append(Gf.Vec3f(p[0] * f, p[1], p[2] * f))
        pts_attr.Set(new)
        # keep the bounding box honest for culling/collision cooking
        mx = max(max(abs(q[0]), abs(q[2])) for q in new)
        mesh.GetExtentAttr().Set([Gf.Vec3f(-mx, y0, -mx), Gf.Vec3f(mx, y1, mx)])
        n_meshes += 1
    stage.GetRootLayer().Save()
    print(f"  tapered {n_meshes} mesh prim(s): transverse scale {lo:.3f} (-Y end) -> {hi:.3f} (+Y end)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/crane_testbed/assets/scenes/testbed_log.usd",
                    help="source log asset to copy and deform (materials/physics preserved)")
    ap.add_argument("--out_dir", default="/workspace/crane_testbed/assets/scenes")
    ap.add_argument("--taper", type=float, default=0.18,
                    help="fractional diameter difference butt-to-tip (0.18 = butt 18%% fatter than "
                         "tip; ~1 cm/m on a 2.4 m, 11.3 cm log). Mean diameter is preserved.")
    a = ap.parse_args()

    # preserve the mean diameter: butt = 1+t/2, tip = 1-t/2
    butt, tip = 1.0 + a.taper / 2.0, 1.0 - a.taper / 2.0
    for name, plus in (("testbed_log_taper.usd", True), ("testbed_log_taper_flip.usd", False)):
        dst = os.path.join(a.out_dir, name)
        shutil.copyfile(a.src, dst)          # keeps materials, textures, physics APIs
        print(f"{dst}  (from {os.path.basename(a.src)})")
        taper_stage(dst, butt, tip, butt_at_plus_y=plus)


if __name__ == "__main__":
    main()
    _app.close()
