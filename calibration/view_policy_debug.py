#!/usr/bin/env python3
"""Offline 3D view of a live policy cycle dumped by crane_policy_node.

When the node runs with `-p debug_save_dir:=<dir>`, each cycle writes
<dir>/policy_debug_NNN.npz holding the cropped policy-input cloud + the chosen
target. This shows it the same way as test_policy_real.py: gray cloud, red target
sphere, red yaw line, blue crop box, base-frame axes.

Run on the HOST (has open3d). Accepts a single .npz OR a directory (steps through
every policy_debug_*.npz in order; with --save it writes a PNG next to each).
    python3 view_policy_debug.py <file.npz | dir> [--save out.png|--save-dir]
"""
import argparse
import glob
import os
import numpy as np


def show(npz_path, save_path=None):
    import open3d as o3d

    d = np.load(npz_path)
    pts = d["points"].astype(np.float64)
    x, y, z, yaw = [float(v) for v in d["target"]]
    bmin = d["bounds_min"].astype(np.float64)
    bmax = d["bounds_max"].astype(np.float64)
    if "max_z" in d.files:   # cap the overlay box/bins at the real fill max, not the scan ceiling
        bmax[2] = float(d["max_z"][0])
    print(f"{os.path.basename(npz_path)}: {pts.shape[0]} pts | "
          f"target x={x:.3f} y={y:.3f} z={z:.3f} yaw={np.degrees(yaw):.1f}deg")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    sph.translate([x, y, z])
    sph.paint_uniform_color([1, 0, 0])
    sph.compute_vertex_normals()

    end = [x + 0.7 * np.cos(yaw), y + 0.7 * np.sin(yaw), z]
    line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[x, y, z], end]),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    line.paint_uniform_color([1, 0, 0])

    box = o3d.geometry.AxisAlignedBoundingBox(bmin, bmax)
    box.color = (0.1, 0.5, 1.0)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # base_link origin

    geoms = [pcd, sph, line, box, frame]

    # bins-mode deposit npz: draw compartment dividers (orange) + active bin (green).
    if "bin_edges" in d.files and d["bin_edges"].size >= 2:
        edges = d["bin_edges"].astype(np.float64)
        active = int(d["active_bin"][0]) if "active_bin" in d.files else -1
        dp, dl = [], []
        for ye in edges:
            b = len(dp)
            dp += [[bmin[0], ye, bmin[2]], [bmax[0], ye, bmin[2]],
                   [bmax[0], ye, bmax[2]], [bmin[0], ye, bmax[2]]]
            dl += [[b, b + 1], [b + 1, b + 2], [b + 2, b + 3], [b + 3, b]]
        divls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(dp, np.float64)),
            lines=o3d.utility.Vector2iVector(np.array(dl, np.int32)))
        divls.paint_uniform_color([1.0, 0.5, 0.0])
        geoms.append(divls)
        if 0 <= active < edges.size - 1:
            abox = o3d.geometry.AxisAlignedBoundingBox(
                np.array([bmin[0], edges[active], bmin[2]]),
                np.array([bmax[0], edges[active + 1], bmax[2]]))
            abox.color = (0.0, 0.8, 0.1)
            geoms.append(abox)
    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path, do_render=True)
        vis.destroy_window()
        print(f"  saved -> {save_path}")
    else:
        o3d.visualization.draw_geometries(
            geoms, window_name=os.path.basename(npz_path) + " (base frame; close for next)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a .npz file OR a directory of policy_debug_*.npz")
    ap.add_argument("--save", default=None,
                    help="for a single file: write this PNG instead of a window")
    ap.add_argument("--save-dir", action="store_true",
                    help="for a directory: write a PNG next to each npz instead of windows")
    ap.add_argument("--out-dir", default=None,
                    help="for a directory: render each npz to a PNG in THIS (writable) dir. Use this when "
                         "the npz live in a root-owned logs dir (the container writes them as root).")
    ap.add_argument("--glob", default="*.npz",
                    help="glob for which npz to view in a directory (e.g. 'deposit_debug_*.npz' for the "
                         "trailer deposit dumps, 'policy_debug_*.npz' for the gaze inputs). Default: all.")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, args.glob)))
        if not files:
            print(f"no .npz files in {args.path}")
            return
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
        print(f"{len(files)} cycle(s) in {args.path}")
        for f in files:
            if args.out_dir:
                sp = os.path.join(args.out_dir, os.path.splitext(os.path.basename(f))[0] + ".png")
            elif args.save_dir:
                sp = os.path.splitext(f)[0] + ".png"
            else:
                sp = None
            show(f, save_path=sp)
    else:
        show(args.path, save_path=args.save)


if __name__ == "__main__":
    main()
