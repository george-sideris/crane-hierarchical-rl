#!/usr/bin/env python3
"""Visualize a trained policy's grasp target on a real-world point cloud (PLY).

Loads a PLY, transforms to crane base frame via a configurable
camera-in-base pose, FPS-downsamples to match the training input size,
runs the policy, and renders:
  - the downsampled point cloud (gray)
  - the policy's target as a red sphere at (x, y, z)
  - a blue arrow showing the target yaw (basegrapple orientation in base frame)
  - the crane base coordinate frame at the origin

The camera→base transform defaults match `sim_ros2_env.py`'s published values.
Override via --cam_pos_b and --cam_quat_b for the real ZED X mount.

Example:
  /usr/bin/python3 visualize_policy_on_pcd.py \
      --ply ~/Documents/point_cloud_PLY_31527836_1242_08-04-2026-15-40-57.ply \
      --checkpoint ~/IsaacLab/crane_testbed/results/BCRL_Raw_PCD_Sigma_0.05/2026-02-27_04-21-30/model_350.pt
"""

import argparse
import os
import sys

import numpy as np
import torch

# Import policy_loader directly from the ros2 scripts directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROS2_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROS2_DIR)
from policy_loader import load_policy  # noqa: E402


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def farthest_point_sampling(pts: np.ndarray, n: int) -> np.ndarray:
    """Simple CPU FPS. Returns (n, 3)."""
    N = pts.shape[0]
    if N <= n:
        # Pad with first point (matches pipeline behaviour for sparse clouds)
        pad = np.tile(pts[0:1], (n - N, 1)) if N > 0 else np.zeros((n, 3))
        return np.concatenate([pts, pad], axis=0)
    sampled = np.empty((n, 3), dtype=pts.dtype)
    distances = np.full(N, np.inf)
    idx = np.random.randint(N)
    for i in range(n):
        sampled[i] = pts[idx]
        diff = pts - pts[idx]
        d = np.einsum("ij,ij->i", diff, diff)
        distances = np.minimum(distances, d)
        idx = int(np.argmax(distances))
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="Path to .ply point cloud")
    ap.add_argument("--checkpoint", required=True, help="Policy checkpoint (.pt)")
    ap.add_argument("--policy_type", default="bcrl",
                    choices=["bc", "bcrl", "rl", "sac", "heuristic"])
    ap.add_argument("--cossin", action="store_true", default=True,
                    help="5D action with cos/sin yaw encoding (default True)")
    ap.add_argument("--no_cossin", dest="cossin", action="store_false")
    # Action bounds (policy's [-1,1] output is mapped into this box, base frame)
    ap.add_argument("--bounds_min", nargs=3, type=float,
                    default=[-5.0, -0.75, -1.373])
    ap.add_argument("--bounds_max", nargs=3, type=float,
                    default=[-3.0, 4.59, -0.373])
    # Pipeline params (match crane_policy_node.py defaults)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--depth_min", type=float, default=1.0)
    ap.add_argument("--depth_max", type=float, default=10.0)
    # Camera pose in crane base frame (defaults = sim's values)
    ap.add_argument("--cam_pos_b", nargs=3, type=float,
                    default=[-1.0, 1.92, 1.577])
    ap.add_argument("--cam_quat_b", nargs=4, type=float,
                    default=[0.2594, 0.6585, 0.6585, 0.2594],
                    help="Camera quaternion in base frame (w x y z, ROS convention)")
    ap.add_argument("--ply_frame", default="camera",
                    choices=["camera", "base"],
                    help="Frame the PLY points are in (default: camera)")
    ap.add_argument("--ply_convention", default="ros",
                    choices=["ros", "opengl"],
                    help="Camera convention of the PLY points. ROS = X right, "
                         "Y down, Z forward (default). OpenGL = X right, Y up, "
                         "Z backward — typical for ZED SDK PLY exports. "
                         "OpenGL is converted to ROS by flipping Y and Z.")
    ap.add_argument("--viz_frame", default="camera",
                    choices=["camera", "base"],
                    help="Frame to visualize in (default: camera — keeps PLY "
                         "untouched, converts target back to camera frame)")
    ap.add_argument("--no_gui", action="store_true",
                    help="Skip 3D visualization (print target only)")
    ap.add_argument("--save_image", type=str, default=None,
                    help="Render to PNG at this path (offscreen, no window)")
    ap.add_argument("--recenter", action="store_true",
                    help="Detect rack centroid in base-frame PCD and shift PCD so "
                         "it lands at --sim_rack_center, preserving training-time "
                         "input distribution. Target is shifted back for viz.")
    ap.add_argument("--sim_rack_center", nargs=3, type=float,
                    default=[-4.0, 1.92, -0.873],
                    help="Where the sim rack centroid sat in BASE frame "
                         "(default = midpoint of sim's published action bounds: "
                         "x=(-5+-3)/2, y=(-0.75+4.59)/2, z=(-1.373+-0.373)/2). "
                         "Crane base sits at world (6,-2.92,1.423) and rack at "
                         "world (~2,~0,~0.1), so rack_in_base ≈ (-4, 1.92, -0.87).")
    ap.add_argument("--rack_z_min", type=float, default=0.2,
                    help="When --recenter, points with base-z above this are "
                         "considered rack content for centroid detection")
    ap.add_argument("--rack_z_max", type=float, default=1.0,
                    help="Upper z cap for centroid detection (excludes tall "
                         "poles/struts above the log pile)")
    ap.add_argument("--centroid_stat", default="median",
                    choices=["median", "mean"],
                    help="Reduction over filtered rack points (default: median, "
                         "robust to pole/strut outliers)")
    ap.add_argument("--real_rack_center", nargs=3, type=float, default=None,
                    help="Manually override the detected rack centroid (base "
                         "frame xyz). Bypasses auto-detection. Use this when "
                         "you can see the rack in the viz but auto-detection "
                         "picks the wrong spot.")
    ap.add_argument("--pick", action="store_true",
                    help="Interactive mode: open viz first to pick rack center "
                         "by Shift+Left-click on the PCD, then close to run "
                         "policy with that centroid as the recenter anchor.")
    ap.add_argument("--pick_bounds", action="store_true",
                    help="Interactive mode: also pick points to define the "
                         "action-bounds AABB. Pick ≥2 points (Shift+Click); "
                         "their min/max coords become bounds_min/max, "
                         "overriding --bounds_min/--bounds_max.")
    args = ap.parse_args()

    # Lazy import — open3d is heavyweight
    import open3d as o3d

    # ── Load PLY ──────────────────────────────────────────────────────
    print(f"Loading {args.ply} ...")
    pcd = o3d.io.read_point_cloud(args.ply)
    pts_raw = np.asarray(pcd.points, dtype=np.float32)
    if pts_raw.shape[0] == 0:
        print("ERROR: point cloud is empty")
        return 1
    print(f"  raw points: {pts_raw.shape[0]}")
    print(f"  raw range: x=[{pts_raw[:,0].min():.2f},{pts_raw[:,0].max():.2f}] "
          f"y=[{pts_raw[:,1].min():.2f},{pts_raw[:,1].max():.2f}] "
          f"z=[{pts_raw[:,2].min():.2f},{pts_raw[:,2].max():.2f}]")

    # ── Convert PLY convention if needed ─────────────────────────────
    # ZED SDK PLY exports use OpenGL convention (Y up, Z backward). The sim
    # extrinsic expects ROS convention (Y down, Z forward). Flip Y and Z to
    # convert OpenGL → ROS before applying the extrinsic.
    if args.ply_convention == "opengl":
        pts_raw[:, 1] *= -1
        pts_raw[:, 2] *= -1
        print(f"  [convention] flipped Y and Z (OpenGL → ROS)")

    # ── Filter: remove NaN/inf and out-of-range depths (magnitude) ───
    finite = np.all(np.isfinite(pts_raw), axis=1)
    pts_raw = pts_raw[finite]
    dist = np.linalg.norm(pts_raw, axis=1)
    mask = (dist >= args.depth_min) & (dist <= args.depth_max)
    pts_raw = pts_raw[mask]
    print(f"  after finite + depth [{args.depth_min}, {args.depth_max}]: {pts_raw.shape[0]}")

    if pts_raw.shape[0] == 0:
        print("ERROR: no valid points after filtering")
        return 1

    # ── Build camera→base transform ──────────────────────────────────
    cam_pos_b = np.array(args.cam_pos_b, dtype=np.float32)
    cam_quat_b = np.array(args.cam_quat_b, dtype=np.float32)
    R_c2b = quat_wxyz_to_rot(cam_quat_b).astype(np.float32)

    # Points we feed to the policy MUST be in base frame (that's where bounds live)
    if args.ply_frame == "camera":
        pts_for_policy = pts_raw @ R_c2b.T + cam_pos_b
    else:  # ply_frame == "base"
        pts_for_policy = pts_raw
    print(f"  policy-input (base) range: "
          f"x=[{pts_for_policy[:,0].min():.2f},{pts_for_policy[:,0].max():.2f}] "
          f"y=[{pts_for_policy[:,1].min():.2f},{pts_for_policy[:,1].max():.2f}] "
          f"z=[{pts_for_policy[:,2].min():.2f},{pts_for_policy[:,2].max():.2f}]")

    # ── Apply the same preprocessing the policy will see BEFORE any picker.
    # This way the picker shows the exact 1024 points the policy operates on.
    if pts_for_policy.shape[0] > 5000:
        rng_idx = np.random.permutation(pts_for_policy.shape[0])[:5000]
        pts_for_policy = pts_for_policy[rng_idx]
    sampled = farthest_point_sampling(pts_for_policy, args.num_points)
    print(f"  FPS sampled (base frame, for policy): {sampled.shape[0]} points")

    # ── Optional: recenter PCD so detected rack lands at sim's rack location ─
    # Preserves the policy's training-time input distribution: the rack appears
    # in the same base-frame location it did during training, so the learned
    # mapping (input → target → bounds-scaled output) stays consistent.
    recenter_delta = np.zeros(3, dtype=np.float32)
    real_rack_center = None
    picked_bounds = None  # set if --pick_bounds; (min[3], max[3])
    if args.recenter or args.pick_bounds:
        # 0) Pick bounds — gives both a centroid AND an AABB
        if args.pick_bounds:
            print("\n[PICK BOUNDS] Opening viz of FPS'd PCD (the exact 1024 "
                  "points the policy will see).")
            print("  Shift+Left-click ≥2 points to define an AABB around the "
                  "rack. The AABB midpoint will be the recenter centroid; the "
                  "AABB itself becomes the visualized bounds box.")
            print("  Close the window when done.")
            import open3d as o3d_pick
            pre_pcd = o3d_pick.geometry.PointCloud()
            pre_pcd.points = o3d_pick.utility.Vector3dVector(
                sampled.astype(np.float64))
            pre_pcd.paint_uniform_color([0.55, 0.55, 0.55])
            base_axes = o3d_pick.geometry.TriangleMesh.create_coordinate_frame(
                size=0.7, origin=[0, 0, 0])
            picker = o3d_pick.visualization.VisualizerWithEditing()
            picker.create_window(window_name="Pick AABB corners (Shift+Click ≥2)",
                                 width=1280, height=960)
            picker.add_geometry(pre_pcd)
            picker.add_geometry(base_axes)
            picker.run()
            picked_idx = picker.get_picked_points()
            picker.destroy_window()
            if len(picked_idx) < 2:
                print(f"  [PICK BOUNDS] need ≥2 picks, got {len(picked_idx)} — "
                      "skipping bounds override.")
            else:
                picked_pts = np.asarray(pre_pcd.points)[picked_idx]
                picked_bounds = (picked_pts.min(axis=0).astype(np.float32),
                                 picked_pts.max(axis=0).astype(np.float32))
                real_rack_center = ((picked_bounds[0] + picked_bounds[1]) / 2.0
                                    ).astype(np.float32)
                print(f"  [PICK BOUNDS] {len(picked_idx)} picks → "
                      f"AABB min={picked_bounds[0].round(3).tolist()} "
                      f"max={picked_bounds[1].round(3).tolist()}")
                print(f"  [PICK BOUNDS] midpoint (used as recenter centroid): "
                      f"{real_rack_center.round(3).tolist()}")
        # 1) Manual override wins
        elif args.real_rack_center is not None:
            real_rack_center = np.array(args.real_rack_center, dtype=np.float32)
            print(f"  [recenter] using MANUAL rack centroid (base): "
                  f"{real_rack_center.round(3).tolist()}")
        # 2) Interactive pick mode
        elif args.pick:
            print("\n[PICK MODE] Opening viz of FPS'd PCD (1024 pts the policy sees).")
            print("  Shift+Left-click to pick a point on the rack (multiple "
                  "picks averaged). Close the window when done.")
            import open3d as o3d_pick
            pre_pcd = o3d_pick.geometry.PointCloud()
            pre_pcd.points = o3d_pick.utility.Vector3dVector(
                sampled.astype(np.float64))
            pre_pcd.paint_uniform_color([0.55, 0.55, 0.55])
            base_axes = o3d_pick.geometry.TriangleMesh.create_coordinate_frame(
                size=0.7, origin=[0, 0, 0])
            picker = o3d_pick.visualization.VisualizerWithEditing()
            picker.create_window(window_name="Pick rack center (Shift+Click)",
                                 width=1280, height=960)
            picker.add_geometry(pre_pcd)
            picker.add_geometry(base_axes)
            picker.run()
            picked_idx = picker.get_picked_points()
            picker.destroy_window()
            if len(picked_idx) == 0:
                print("  [PICK] no points picked — skipping recenter.")
                real_rack_center = None
            else:
                picked_pts = np.asarray(pre_pcd.points)[picked_idx]
                real_rack_center = picked_pts.mean(axis=0).astype(np.float32)
                print(f"  [PICK] picked {len(picked_idx)} pts, "
                      f"centroid (base): {real_rack_center.round(3).tolist()}")
        # 3) Fall back to auto-detection
        else:
            z = sampled[:, 2]
            rack_pts = sampled[(z > args.rack_z_min) & (z < args.rack_z_max)]
            if rack_pts.shape[0] < 50:
                print(f"  [recenter] WARN: only {rack_pts.shape[0]} points in "
                      f"z∈({args.rack_z_min}, {args.rack_z_max}) — bad extrinsic "
                      f"or wrong z range? skipping recenter.")
                real_rack_center = None
            else:
                if args.centroid_stat == "median":
                    real_rack_center = np.median(rack_pts, axis=0).astype(np.float32)
                else:
                    real_rack_center = rack_pts.mean(axis=0).astype(np.float32)
                print(f"  [recenter] auto centroid (base, {args.centroid_stat}): "
                      f"{real_rack_center.round(3).tolist()}  "
                      f"({rack_pts.shape[0]} pts in z∈[{args.rack_z_min},{args.rack_z_max}])")

        if real_rack_center is not None:
            sim_rack_center = np.array(args.sim_rack_center, dtype=np.float32)
            recenter_delta = sim_rack_center - real_rack_center
            sampled = sampled + recenter_delta
            print(f"  [recenter] sim rack centroid (base):  "
                  f"{sim_rack_center.round(3).tolist()}")
            print(f"  [recenter] shift delta (added to PCD): "
                  f"{recenter_delta.round(3).tolist()}")

    # ── Load policy ──────────────────────────────────────────────────
    bounds_min = np.array(args.bounds_min, dtype=np.float32)
    bounds_max = np.array(args.bounds_max, dtype=np.float32)
    print(f"\nLoading policy: type={args.policy_type}, cossin={args.cossin}")
    print(f"  bounds_min={bounds_min.tolist()}, bounds_max={bounds_max.tolist()}")
    policy = load_policy(args.policy_type, args.checkpoint,
                         bounds_min, bounds_max, args.cossin, device="cpu")

    # ── Inference ────────────────────────────────────────────────────
    obs = torch.from_numpy(sampled.reshape(-1).astype(np.float32))
    x, y, z, yaw = policy.get_target(obs)
    if args.recenter:
        # Policy saw shifted PCD → output is in shifted frame. Undo for viz.
        x -= float(recenter_delta[0])
        y -= float(recenter_delta[1])
        z -= float(recenter_delta[2])
    print("\n┌─────────── Policy output (crane base frame) ────────────┐")
    print(f"│ target xyz  = ({x:+.3f}, {y:+.3f}, {z:+.3f}) m            ")
    print(f"│ target yaw  = {np.degrees(yaw):+.1f}°                      ")
    print("└──────────────────────────────────────────────────────────┘")

    if args.no_gui:
        return 0

    # ── Pick what to show in the chosen viz frame ────────────────────
    # Target from policy is in BASE frame. PCD is in whatever ply_frame says.
    # We either convert target→camera (keep raw PCD) or convert PCD→base.
    target_b = np.array([x, y, z], dtype=np.float32)
    yaw_b = float(yaw)

    # When --recenter is on, `sampled` lives in the SHIFTED frame (rack moved
    # to sim location). For viz we want everything in the REAL base frame so
    # bounds, target, centroid, and PCD all share one coordinate system. Undo
    # the shift on the FPS'd points for display purposes.
    sampled_real = sampled - recenter_delta if args.recenter else sampled

    if args.viz_frame == "camera":
        # Subsample the raw PCD (unchanged) just for display
        pts_viz = pts_raw
        if pts_viz.shape[0] > 20000:
            idx = np.random.permutation(pts_viz.shape[0])[:20000]
            pts_viz = pts_viz[idx]
        # Convert target from base → camera frame
        # p_b = R_c2b @ p_c + cam_pos_b  =>  p_c = R_c2b.T @ (p_b - cam_pos_b)
        target_viz = R_c2b.T @ (target_b - cam_pos_b)
        # Yaw arrow: rotate the base-frame yaw axis into camera frame.
        # In base frame the arrow along base +x rotated by yaw is:
        #   v_b = [cos(yaw), sin(yaw), 0]
        # In camera frame:
        v_b = np.array([np.cos(yaw_b), np.sin(yaw_b), 0.0], dtype=np.float32)
        v_c = R_c2b.T @ v_b
        axes_origin = -R_c2b.T @ cam_pos_b  # where the base-frame origin sits in cam frame
        frame_size = 0.5
        print(f"\nViz frame: camera (ZED native)")
        print(f"  target (camera frame): ({target_viz[0]:+.3f}, {target_viz[1]:+.3f}, {target_viz[2]:+.3f})")
    else:  # base
        pts_viz = sampled_real  # un-shifted to match real-base coords
        target_viz = target_b
        v_c = np.array([np.cos(yaw_b), np.sin(yaw_b), 0.0], dtype=np.float32)
        axes_origin = np.array([0, 0, 0], dtype=np.float32)
        frame_size = 0.5
        print(f"\nViz frame: base")

    # Point cloud
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pts_viz.astype(np.float64))
    vis_pcd.paint_uniform_color([0.55, 0.55, 0.55])

    # Target: red sphere
    target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.10)
    target_sphere.compute_vertex_normals()
    target_sphere.paint_uniform_color([1.0, 0.15, 0.15])
    target_sphere.translate(target_viz.astype(np.float64))

    # Yaw arrow: blue, built along +z, rotate to match v_c direction.
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02, cone_radius=0.05,
        cylinder_height=0.30, cone_height=0.10,
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color([0.15, 0.3, 1.0])
    # Rotate default-z arrow to point along v_c
    z_axis = np.array([0.0, 0.0, 1.0])
    v = v_c / (np.linalg.norm(v_c) + 1e-9)
    axis = np.cross(z_axis, v)
    s = np.linalg.norm(axis)
    c = float(np.dot(z_axis, v))
    if s < 1e-6:  # parallel or antiparallel
        if c > 0:
            R_arrow = np.eye(3)
        else:
            R_arrow = np.diag([1.0, -1.0, -1.0])
    else:
        axis = axis / s
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R_arrow = np.eye(3) + s * K + (1 - c) * K @ K
    arrow.rotate(R_arrow, center=np.array([0.0, 0.0, 0.0]))
    arrow.translate(target_viz.astype(np.float64))

    # Two coord frames: large = crane base, small = camera. Both drawn in the
    # current viz frame so they share a coordinate system with everything else.
    if args.viz_frame == "camera":
        base_origin_viz = -R_c2b.T @ cam_pos_b   # crane base in camera frame
        cam_origin_viz = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # cam at viz origin
    else:  # base
        base_origin_viz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        cam_origin_viz = cam_pos_b                # camera in base frame
    axes_base = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.7, origin=base_origin_viz.astype(np.float64).tolist(),
    )
    axes_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=cam_origin_viz.astype(np.float64).tolist(),
    )
    print(f"\nFrame markers in viz:")
    print(f"  large RGB triad (size 0.7) = CRANE BASE origin "
          f"@ {base_origin_viz.round(3).tolist()}")
    print(f"  small RGB triad (size 0.3) = CAMERA       origin "
          f"@ {cam_origin_viz.round(3).tolist()}")

    # ── Action-bounds wireframe box (green) ──────────────────────────
    # If --pick_bounds was used, draw the picked AABB directly. Otherwise draw
    # the sim-anchored bounds (and shift them to wrap the real rack if --recenter).
    if picked_bounds is not None:
        pmin, pmax = picked_bounds
        bx = [float(pmin[0]), float(pmax[0])]
        by = [float(pmin[1]), float(pmax[1])]
        bz = [float(pmin[2]), float(pmax[2])]
    else:
        bx = [bounds_min[0], bounds_max[0]]
        by = [bounds_min[1], bounds_max[1]]
        bz = [bounds_min[2], bounds_max[2]]
    corners_b = np.array([[x_, y_, z_] for x_ in bx for y_ in by for z_ in bz],
                         dtype=np.float32)
    if args.recenter and picked_bounds is None:
        corners_b = corners_b - recenter_delta  # shift sim-anchored bounds → real-rack-anchored
    if args.viz_frame == "camera":
        corners_viz = (corners_b - cam_pos_b) @ R_c2b
    else:
        corners_viz = corners_b
    bbox_lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],
                  [0,4],[1,5],[2,6],[3,7]]
    bbox = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners_viz.astype(np.float64)),
        lines=o3d.utility.Vector2iVector(bbox_lines),
    )
    bbox.colors = o3d.utility.Vector3dVector([[0.1, 0.9, 0.1]] * len(bbox_lines))

    geoms = [vis_pcd, target_sphere, arrow, axes_base, axes_cam, bbox]

    # ── Detected rack centroid (green sphere) when --recenter is on ──
    if real_rack_center is not None:
        if args.viz_frame == "camera":
            centroid_viz = R_c2b.T @ (real_rack_center - cam_pos_b)
        else:
            centroid_viz = real_rack_center
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
        centroid_sphere.compute_vertex_normals()
        centroid_sphere.paint_uniform_color([0.1, 0.9, 0.1])
        centroid_sphere.translate(centroid_viz.astype(np.float64))
        geoms.append(centroid_sphere)
        print(f"\nLegend: red=policy target, green sphere=detected rack centroid, "
              f"green wireframe=action bounds")
    else:
        print(f"\nLegend: red=policy target, green wireframe=action bounds")

    if args.save_image is not None:
        # Headless render via offscreen Visualizer (no window manager needed)
        vis = o3d.visualization.Visualizer()
        ok = vis.create_window(visible=False, width=1280, height=960)
        if not ok:
            print("ERROR: could not create offscreen window "
                  "(open3d needs an X display even for offscreen on this version).")
            print("Falling back to on-screen and saving via screenshot...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=960)
        for g in geoms:
            vis.add_geometry(g)
        # Fit the view to the scene, looking roughly along -z so the
        # camera-frame points are in front of the viewer.
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save_image, do_render=True)
        vis.destroy_window()
        print(f"\nSaved visualization to: {args.save_image}")
        return 0

    print("\nOpening 3D viewer (close window to exit)...")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Policy target ({args.viz_frame}) on {os.path.basename(args.ply)}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
