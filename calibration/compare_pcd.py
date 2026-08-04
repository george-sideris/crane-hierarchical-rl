#!/usr/bin/env python3
"""Interactive sim-vs-real point-cloud comparison (Open3D).

Loads the FULL sim cloud saved by a gaze run and the FULL real ZED cloud from the
calibration bag, both already in the crane BASE frame, and shows them overlaid:
    SIM  = red
    REAL = blue
Rotate / zoom / pan with the mouse. A coordinate frame marks the base origin.

  python3 compare_pcd.py                 # interactive overlaid view
  python3 compare_pcd.py --side          # offset real in +Y to view side by side
  python3 compare_pcd.py --ply           # export .ply files (open in CloudCompare/MeshLab)
"""
import argparse
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("--sim", default="out/sim_full_pcd.npy", help="full sim cloud (base frame)")
ap.add_argument("--real", default="out/pcd_base_gaze.npy",
                help="full real ZED cloud (base frame). Default = measured gaze-pose cloud from the "
                     "2026_06_22 20:31 static gaze bag. Use out/pcd_base_calib.npy for the old synthesized one.")
ap.add_argument("--side", action="store_true", help="offset real +Y to view side by side")
ap.add_argument("--ply", action="store_true", help="export .ply instead of opening a window")
ap.add_argument("--rack", action="store_true", help="crop to the rack region (drop background)")
ap.add_argument("--minz", type=float, default=None, help="drop points below this Z (e.g. -1.3 to remove floor)")
ap.add_argument("--yaw", type=float, default=0.0,
                help="rotate the REAL cloud by this many degrees about the VERTICAL axis (the slew axis "
                     "at x=0,y=0). Dial it until the clouds overlap. ~ -57 (slew) is the expected value.")
a = ap.parse_args()

def _filt(P):
    if a.rack:
        m = ((P[:,0]>-6.94)&(P[:,0]<-3.18)&(P[:,1]>-1.76)&(P[:,1]<6.82)&(P[:,2]>-1.55)&(P[:,2]<0.30))
        P = P[m]
    if a.minz is not None:
        P = P[P[:,2] > a.minz]
    return P

sim = np.load(a.sim)[:, :3].astype(np.float64)
real = np.load(a.real)[:, :3].astype(np.float64)
if a.yaw != 0.0:
    # PURE rotation about the vertical (slew) axis through x=0,y=0 — no tilt.
    t = np.deg2rad(a.yaw); c, s = np.cos(t), np.sin(t)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    real = (Rz @ real.T).T
sim = _filt(sim)
real = _filt(real)
print(f"sim  : {sim.shape[0]:>7d} pts  X[{sim[:,0].min():.1f},{sim[:,0].max():.1f}] "
      f"Y[{sim[:,1].min():.1f},{sim[:,1].max():.1f}] Z[{sim[:,2].min():.1f},{sim[:,2].max():.1f}]")
print(f"real : {real.shape[0]:>7d} pts  X[{real[:,0].min():.1f},{real[:,0].max():.1f}] "
      f"Y[{real[:,1].min():.1f},{real[:,1].max():.1f}] Z[{real[:,2].min():.1f},{real[:,2].max():.1f}]")

ps = o3d.geometry.PointCloud(); ps.points = o3d.utility.Vector3dVector(sim);  ps.paint_uniform_color([0.90, 0.10, 0.10])
pr = o3d.geometry.PointCloud(); pr.points = o3d.utility.Vector3dVector(real); pr.paint_uniform_color([0.10, 0.30, 0.90])

if a.side:
    pr.translate((0.0, float(np.ptp(real[:, 1])) + 2.0, 0.0))

if a.ply:
    o3d.io.write_point_cloud("out/sim_full.ply", ps)
    o3d.io.write_point_cloud("out/real_full.ply", pr)
    print("wrote out/sim_full.ply (red) and out/real_full.ply (blue)")
else:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)  # base origin
    # action-bounds wireframe (the calibrated rack region the policy acts in / crop box), green
    CX, CY, HX, HY, ZMIN, ZMAX = -5.061, 2.527, 1.00, 3.50, -1.30, 0.10
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(CX - HX, CY - HY, ZMIN), max_bound=(CX + HX, CY + HY, ZMAX))
    aabb.color = (0.0, 0.9, 0.0)
    o3d.visualization.draw_geometries([ps, pr, frame, aabb],
                                      window_name="SIM (red) vs REAL ZED (blue) + action bounds (green) - base frame")
