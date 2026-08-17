#!/usr/bin/env python3
"""Open3D renders of the observation-pipeline clouds, in the style of the offline-replay
figure of the field-trials chapter.

Reads the npz that --paper_viz now writes alongside the quadrant PNG and produces two
perspective panels: the raw base-frame cloud, and the cropped and FPS-resampled cloud with
the commanded grasp drawn on it. The matplotlib panels the pipeline figure shipped with are
flat top-down scatter with 0.6pt markers; at \\linewidth the mound structure and the rack
floor are both hard to read, which is the whole point of the figure.

Camera, height ramp, box colours and marker sizes are deliberately identical to
render_policy_comparison.py so the two chapters look like they came from one tool.

Runs on the HOST (needs open3d, which is not in the Isaac container):
    python3 scripts/envs/render_pipeline_clouds_3d.py \
        --npz <run>/paper_viz/episode_001_cloud_quad_single.npz \
        --out docs/thesis/figures
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image

_Mat = getattr(rendering, "MaterialRecord", None) or rendering.Material
W, H = 1400, 900


def height_colors(z, lo, hi):
    """The turbo-ish ramp used by the replay figure, so the two chapters match.

    Deliberately not viridis: that is the depth image's colormap, and reusing it here would
    make the cloud panels read as another view of the same heatmap.
    """
    t = np.clip((z - lo) / (hi - lo), 0, 1)
    return np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                     np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], axis=1)


def render(pts, bmin, bmax, zlim, target=None, yaw=None, point_size=3.2):
    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    sc = r.scene

    mp = _Mat(); mp.shader = "defaultUnlit"; mp.point_size = point_size
    ml = _Mat(); ml.shader = "unlitLine"; ml.line_width = 3.0
    # The prediction has to survive being scaled into a third of \linewidth, so it gets a
    # heavier weight than the crop box that frames it.
    mt = _Mat(); mt.shader = "unlitLine"; mt.line_width = 9.0
    mm = _Mat(); mm.shader = "defaultLit"

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(height_colors(pts[:, 2], *zlim) * 0.55 + 0.35)
    sc.add_geometry("pcd", pcd, mp)

    if bmin is not None and bmax is not None:
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox(bmin, bmax))
        box.paint_uniform_color([0.35, 0.55, 0.9])
        sc.add_geometry("box", box, ml)

    if target is not None:
        # The commanded z sits a digging depth BELOW the surface, so the marker is inside the
        # pile by construction and a small sphere is simply buried, the more so at the point
        # sizes this figure needs. A stem dropped from clear air marks the same target where
        # nothing can occlude it, and the sphere still shows the depth it commands.
        x, y, z = target
        # Black, not red: the height ramp ends in red, so a red marker is indistinguishable
        # from the rack posts and the pile crest it is drawn against. Nothing in the ramp is
        # dark, so black separates from every point in the cloud.
        col = [0.05, 0.05, 0.05]
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.20)
        s.translate([x, y, z]); s.paint_uniform_color(col); s.compute_vertex_normals()
        sc.add_geometry("tgt", s, mm)
        top = float(max(pts[:, 2].max(), z) + 0.3)
        stem = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector([[x, y, top], [x, y, z]]),
            lines=o3d.utility.Vector2iVector([[0, 1]]))
        stem.paint_uniform_color(col)
        sc.add_geometry("tgtstem", stem, mt)
        if yaw is not None:
            e1 = [x + 0.7 * np.cos(yaw), y + 0.7 * np.sin(yaw), z]
            e2 = [x - 0.7 * np.cos(yaw), y - 0.7 * np.sin(yaw), z]
            ln = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([e1, [x, y, z], e2]),
                lines=o3d.utility.Vector2iVector([[0, 1], [1, 2]]))
            ln.paint_uniform_color(col)
            sc.add_geometry("tgtyaw", ln, mt)

    # Frame the crop box when there is one, otherwise the cloud itself.
    lo = bmin if bmin is not None else pts.min(axis=0)
    hi = bmax if bmax is not None else pts.max(axis=0)
    c = 0.5 * (np.asarray(lo) + np.asarray(hi))
    c[2] = lo[2] + 0.3 * (hi[2] - lo[2])
    rad = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
    r.setup_camera(58.0, c, c + np.array([1.0, -1.0, 0.55]) * (0.6 * rad), [0, 0, 1])
    return np.asarray(r.render_to_image())[:, :, :3]


def autocrop(img, tol=6):
    """Trim the uniform border the offscreen renderer leaves around the scene.

    The renderer always returns a full 1400x900 frame, so a cloud that occupies the middle
    third of it becomes unreadable once the panel is scaled into a quarter of \\linewidth.
    """
    bg = img[0, 0].astype(int)
    diff = np.abs(img.astype(int) - bg).sum(axis=2)
    ys, xs = np.where(diff > tol)
    if len(ys) == 0:
        return img
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def fit_aspect(img, ar):
    """Pad an image out to aspect ratio ar with its own background colour.

    The camera frames are 16:9 and the cropped renders are whatever the scene happens to
    occupy, so imshow gave every panel a different size and the grid looked ragged. Padding
    rather than cropping keeps the whole scene and leaves each panel the same shape.
    """
    h, w = img.shape[:2]
    bg = img[0, 0]
    tw, th = max(w, int(round(h * ar))), max(h, int(round(w / ar)))
    out = np.empty((th, tw, img.shape[2]), dtype=img.dtype)
    out[:, :] = bg
    y0, x0 = (th - h) // 2, (tw - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def compose_quad(rgb, depth, cloud_raw, cloud_fps, out_path, depth_range=(1.0, 10.0),
                 annotation=None, drop_rgb=False, zlim=None):
    """Lay the pipeline stages out as the thesis figure.

    The cloud panels are the Open3D perspective renders rather than the flat top-down
    scatter the figure used to carry, matching the offline-replay figure of the field-trials
    chapter so the two chapters read as one tool.

    drop_rgb gives a single row of three. The colour frame is not a stage of the pipeline
    (Algorithm 3.2 starts at the depth image) and the scene is already established earlier
    in the chapter, so cutting it costs no information and buys every remaining panel about
    half again as much width, which is what makes the perspective renders readable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Every panel is shown at one shape so the grid reads as a grid. The camera frames set
    # it, since they are the only panels whose framing is not ours to choose.
    ar = (rgb.shape[1] / rgb.shape[0]) if rgb is not None else 16 / 9
    cloud_raw = fit_aspect(cloud_raw, ar)
    cloud_fps = fit_aspect(cloud_fps, ar)

    def depth_panel(ax, label):
        if depth is not None:
            dep = np.clip(depth, depth_range[0], depth_range[1])
            dep[np.isinf(depth)] = np.nan
            im = ax.imshow(dep, cmap="viridis", vmin=depth_range[0], vmax=depth_range[1])
            cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.set_label("depth (m)", fontsize=6)
        ax.set_title("%s Raw depth" % label, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    def reserve_cax(ax):
        """Take the colourbar's width out of the axes whether or not one is drawn.

        A colourbar steals width from the axes it is attached to, so a panel without one
        renders wider than a panel with one and the grid stops lining up. Panels that show
        no bar reserve the space and hide it.
        """
        cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.05)
        cax.set_axis_off()
        return cax

    def cloud_panel(ax, img, label, title, note=None, scale=False):
        ax.imshow(img)
        ax.set_title("%s %s" % (label, title), fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        # Both cloud panels share one height ramp, so one bar serves both and sits on the
        # resampled panel; putting it on each crowded the label into the neighbouring panel.
        if zlim is not None and scale:
            from matplotlib.colors import ListedColormap, Normalize
            from matplotlib.cm import ScalarMappable
            ramp = height_colors(np.linspace(zlim[0], zlim[1], 256), *zlim) * 0.55 + 0.35
            sm = ScalarMappable(norm=Normalize(*zlim), cmap=ListedColormap(ramp))
            cax = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.05)
            cb = plt.colorbar(sm, cax=cax)
            cb.ax.tick_params(labelsize=5)
            cb.set_label("height (m)", fontsize=6)
        else:
            reserve_cax(ax)
        if note:
            ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=6,
                    verticalalignment="top", color="#27ae60", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    if drop_rgb:
        fig = plt.figure(figsize=(9, 2.5))
        gs = fig.add_gridspec(1, 3, wspace=0.10)
        depth_panel(fig.add_subplot(gs[0, 0]), "(a)")
        cloud_panel(fig.add_subplot(gs[0, 1]), cloud_raw, "(b)", "3D points")
        cloud_panel(fig.add_subplot(gs[0, 2]), cloud_fps, "(c)", "FPS + prediction",
                    note=annotation, scale=True)
    else:
        fig = plt.figure(figsize=(9, 5.2))
        gs = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.22)
        ax = fig.add_subplot(gs[0, 0])
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_title("(a) RGB", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        reserve_cax(ax)
        depth_panel(fig.add_subplot(gs[0, 1]), "(b)")
        cloud_panel(fig.add_subplot(gs[1, 0]), cloud_raw, "(c)", "3D points")
        cloud_panel(fig.add_subplot(gs[1, 1]), cloud_fps, "(d)", "FPS + prediction",
                    note=annotation, scale=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="episode_XXX_cloud_*.npz from --paper_viz")
    ap.add_argument("--out", default="docs/thesis/figures")
    ap.add_argument("--prefix", default="pipeline_cloud")
    ap.add_argument("--quad", metavar="PNG", default=None,
                    help="also compose the full four-panel figure to this path "
                         "(needs rgb/depth in the npz)")
    ap.add_argument("--no-rgb", action="store_true",
                    help="with --quad, compose a single row of three and drop the colour "
                         "frame, which is not a stage of the pipeline")
    args = ap.parse_args()

    z = np.load(args.npz)
    base, fps = z["base_points"], z["fps_points"]
    bmin = z["bounds_min"] if z["bounds_min"].size == 3 else None
    bmax = z["bounds_max"] if z["bounds_max"].size == 3 else None
    if base.size == 0 or fps.size == 0:
        raise SystemExit("npz carries no cloud")

    # One height ramp across both panels, taken from the cropped cloud. The raw cloud is the
    # superset but it contains the grapple hanging metres above the rack, and scaling to that
    # squeezes the whole pile into one colour. Raw points outside the range clamp to the ends.
    zlim = (float(fps[:, 2].min()), float(fps[:, 2].max()))
    print("z range %.3f .. %.3f  raw %d pts  fps %d pts" % (*zlim, len(base), len(fps)))

    os.makedirs(args.out, exist_ok=True)
    # Point sizes are set for a panel that is roughly half of \linewidth. The 2.2/4.5 pair
    # was tuned against the full-width replay figure; scaled into a quadrant panel the cloud
    # reads as scattered specks and the mound structure disappears, which is the one thing
    # this figure has to show.
    jobs = [("raw", base, None, None, 5.5),
            ("fps", fps, z["target"], float(z["yaw"]), 10.0)]
    panels = {}
    for name, pts, tgt, yaw, ps in jobs:
        img = autocrop(render(pts, bmin, bmax, zlim, target=tgt, yaw=yaw, point_size=ps))
        panels[name] = img
        path = os.path.join(args.out, f"{args.prefix}_{name}.jpg")
        Image.fromarray(img).save(path, quality=92)
        print("wrote", path)

    if args.quad:
        keys = z.files
        if "rgb" not in keys or "depth" not in keys:
            raise SystemExit(
                "npz has no rgb/depth; re-run the eval with --paper_viz after the "
                "camera-frame dump was added to _dump_cloud_npz")
        # No outcome legend: this figure is about what the policy sees and what it commands,
        # and the drawn target already carries the prediction. Whether that grasp went on to
        # lift 17 logs is a results-chapter question.
        compose_quad(z["rgb"], z["depth"].astype(float), panels["raw"], panels["fps"],
                     args.quad, drop_rgb=args.no_rgb, zlim=zlim)


if __name__ == "__main__":
    main()
