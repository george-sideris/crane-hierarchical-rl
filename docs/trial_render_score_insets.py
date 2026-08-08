"""Render an Open3D iso view of the policy's scored cloud for chosen cycles.

Colour = per-point grasp score recomputed from the deployed checkpoint (verified to
reproduce the logged target). Context points outside the action box are grey; the
issued target is a red sphere with a short yaw line.
"""
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from matplotlib import cm

sys.path.insert(0, "/home/george/IsaacLab/crane_testbed/docs")
from trial_recompute_scores import scores_for  # noqa: E402

S = "/home/george/IsaacLab/crane_testbed/docs/figures/real_trials/_build/"
W, H = 1400, 1000
BMIN = np.array([-5.364, -1.684, -1.30])
BMAX = np.array([-3.364, 5.316, 0.10])


def mat(shader="defaultUnlit", size=5.0):
    m = rendering.Material()
    m.shader = shader
    m.point_size = size
    return m


def render(cycle, out):
    pts, sc, tgt, _ = scores_for(cycle)
    d = np.load("/home/george/IsaacLab/crane_testbed/logs/bc_pointcloud/"
                "scoring_margin05_2048_c/policy_debug/"
                f"run_20260805_194008/policy_debug_{cycle:03d}.npz")
    shift = float(d["rack_y_shift"])
    lo, hi = BMIN.copy(), BMAX.copy()
    lo[1] += shift
    hi[1] += shift

    inbox = np.all((pts >= lo) & (pts <= hi), axis=1)
    # normalise scores over the admissible set only, robust to a single outlier
    s_in = sc[inbox]
    lo_s, hi_s = np.percentile(s_in, 2), s_in.max()
    t = np.clip((sc - lo_s) / max(hi_s - lo_s, 1e-6), 0, 1)
    cols = cm.get_cmap("viridis")(t)[:, :3]
    cols[~inbox] = np.array([0.55, 0.55, 0.55])   # context: not commandable

    ren = rendering.OffscreenRenderer(W, H)
    ren.scene.set_background([1, 1, 1, 1])

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(cols)
    ren.scene.add_geometry("cloud", pc, mat(size=8.0))

    box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    box.color = (0.15, 0.35, 0.85)
    ren.scene.add_geometry("box", o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box),
                           mat("unlitLine"))

    # pin marker: small sphere at the exact target, head raised clear of the cloud
    for r, dz, name in ((0.11, 0.0, "tgt"), (0.24, 1.15, "pinhead")):
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=r*1.25)
        sph.translate(np.array(tgt[:3]) + np.array([0, 0, dz]))
        sph.paint_uniform_color([0.90, 0.05, 0.05])
        sph.compute_vertex_normals()
        ren.scene.add_geometry(name, sph, mat())

    stalk = o3d.geometry.LineSet()
    stalk.points = o3d.utility.Vector3dVector(
        [np.array(tgt[:3]), np.array(tgt[:3]) + np.array([0, 0, 1.15])])
    stalk.lines = o3d.utility.Vector2iVector([[0, 1]])
    stalk.colors = o3d.utility.Vector3dVector([[0.85, 0.05, 0.05]])
    ren.scene.add_geometry("stalk", stalk, mat("unlitLine"))

    yaw = float(tgt[3])
    a = np.array(tgt[:3]) + 0.75 * np.array([np.cos(yaw), np.sin(yaw), 0])
    b = np.array(tgt[:3]) - 0.75 * np.array([np.cos(yaw), np.sin(yaw), 0])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([a, b])
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([[0.85, 0.05, 0.05]])
    ren.scene.add_geometry("yawline", ls, mat("unlitLine"))

    ctr = (lo + hi) / 2.0
    # viewed from the timelapse camera's side of the rack (-x), not the crane's,
    # so left/right in the inset match left/right in the photograph
    eye = ctr + np.array([-6.5, -1.8, 3.4])
    ren.scene.camera.look_at(ctr, eye, [0, 0, 1])
    img = np.asarray(ren.render_to_image())
    bg = img[0, 0].astype(int)
    mask = (np.abs(img.astype(int) - bg).sum(-1) > 18)
    ys, xs = np.where(mask)
    pad = 18
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    from PIL import Image as _I
    _I.fromarray(img[y0:y1, x0:x1]).save(out)
    print(f"cycle {cycle:02d} -> {out}  ({inbox.sum()} in box / {len(pts)})")


if __name__ == "__main__":
    for c in [int(a) for a in sys.argv[1:]]:
        render(c, S + f"insets/inset_{c:03d}.png")
