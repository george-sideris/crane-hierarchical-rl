#!/usr/bin/env python3
"""Bake the calibrated zed_0 mount into the URDF zed_0_joint.

Calibration is computed in the `mast` frame (T_mast<-zed_0_optical). The camera
is rigid to the mast (it slews with it), so the cleanest model parents zed_0
DIRECTLY to `mast`:

    --parent mast (default):
      JOINT(mast -> zed_0_camera_link) = T_mast<-optical @ inv(T_internal)
    where T_internal = zed_0_camera_link -> ... -> optical (zed-internal edges).

The legacy model parented zed_0 to the lidar (because it used to be mounted on
the lidar). After a remount that coupling is wrong, but keep it available:

    --parent lidar:
      JOINT(os_sensor -> zed_0_camera_link)
        = inv(T1(mast->bracket) @ T2(bracket->os_sensor)) @ T_mast<-optical @ inv(T_internal)

Prints old vs new xyz/rpy + a round-trip self-check. Does NOT write the URDF.
Run on the HOST.  python3 bake_zed0_urdf.py [--parent mast|lidar] [--calib ...]
"""
import argparse
import re
import numpy as np
import calib_fk as fk

URDF = "../fpi_crane_ros2/fpi_crane_description/urdf/fpi_crane.urdf.xacro"


def attrs(snippet):
    xyz = re.search(r'xyz="([^"]+)"', snippet).group(1)
    rpy = re.search(r'rpy="([^"]+)"', snippet).group(1)
    return [float(v) for v in xyz.split()], [float(v) for v in rpy.split()]


def T_from(snippet):
    xyz, rpy = attrs(snippet)
    return fk.rpyT(rpy, xyz)


def rpy_of(R):
    # URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="out/calib_zed_0_mount.npy",
                    help="T_mast<-zed_0_optical 4x4 (promote the floor fit first)")
    ap.add_argument("--parent", choices=["mast", "lidar"], default="mast",
                    help="parent link for zed_0 (mast = direct/clean; lidar = legacy chain)")
    ap.add_argument("--write", action="store_true",
                    help="write the new parent/xyz/rpy into the URDF in place (else just print)")
    args = ap.parse_args()

    urdf = open(URDF).read()
    # match a single self-closing tag (no '>' inside) that names zed_0 -- avoids spanning the zed_1 tag
    zed0 = re.search(r'<xacro:zed_sensor\b[^>]*name="zed_0"[^>]*/>', urdf).group(0)
    Jold = T_from(zed0)    # current zed_0 joint (whatever its parent)

    # zed_0_camera_link -> optical: the zed-internal edges (camera_link->center->left->optical)
    Tint = np.eye(4)
    for xyz, rpy in fk._ZED_INTERNAL:
        Tint = Tint @ fk.rpyT(rpy, xyz)

    Tcal = np.load(args.calib)  # T_mast<-zed_0_optical (calibrated)

    # old optical pose = T_mast<-optical from the CURRENT URDF, through zed_0's current parent
    bracket = re.search(r'<joint name="mast_sensor_bracket_joint".*?</joint>', urdf, re.S).group(0)
    ouster = re.search(r'<xacro:ouster_lidar_sensor\b[^>]*name="lidar_0"[^>]*/>', urdf).group(0)
    legacy_base = T_from(bracket) @ T_from(ouster)   # mast -> os_sensor
    cur_parent = re.search(r'parent="([^"]*)"', zed0).group(1)
    old_base = legacy_base if cur_parent == "lidar_0/os_sensor" else np.eye(4)  # mast-parented => identity
    old_optical = old_base @ Jold @ Tint             # T_mast<-optical from the current URDF

    if args.parent == "mast":
        base = np.eye(4)                     # parent IS mast
        parent_link = "mast"
    else:
        base = legacy_base                   # mast -> os_sensor (held fixed)
        parent_link = "lidar_0/os_sensor"
    Jnew = np.linalg.inv(base) @ Tcal @ np.linalg.inv(Tint)

    # round-trip self-check: chain with Jnew must reproduce the calibrated optical pose
    rt = base @ Jnew @ Tint
    err = np.abs(rt - Tcal).max()

    # how far the optical frame moves from the OLD URDF pose to the new calib
    moved = np.linalg.norm(Tcal[:3, 3] - old_optical[:3, 3])

    ox, orpy = Jold[:3, 3], rpy_of(Jold[:3, :3])
    nx, nrpy = Jnew[:3, 3], rpy_of(Jnew[:3, :3])
    print(f"calib: {args.calib}")
    print(f"round-trip self-check (should be ~0): {err:.2e}")
    print(f"optical-frame shift old->new: {moved*1000:.1f} mm\n")
    print(f"zed_0_joint ({parent_link} -> zed_0_camera_link):")
    print(f"  OLD  xyz=\"{ox[0]:.6f} {ox[1]:.6f} {ox[2]:.6f}\"  "
          f"rpy=\"{orpy[0]:.6f} {orpy[1]:.6f} {orpy[2]:.6f}\"")
    print(f"  NEW  xyz=\"{nx[0]:.6f} {nx[1]:.6f} {nx[2]:.6f}\"  "
          f"rpy=\"{nrpy[0]:.6f} {nrpy[1]:.6f} {nrpy[2]:.6f}\"")

    if not args.write:
        print("\n(dry run) re-run with --write to update the URDF in place.")
        return

    nxyz = f"{nx[0]:.6f} {nx[1]:.6f} {nx[2]:.6f}"
    nrpy_s = f"{nrpy[0]:.6f} {nrpy[1]:.6f} {nrpy[2]:.6f}"
    new_tag = zed0
    new_tag = re.sub(r'parent="[^"]*"', f'parent="{parent_link}"', new_tag)
    new_tag = re.sub(r'xyz="[^"]*"', f'xyz="{nxyz}"', new_tag)
    new_tag = re.sub(r'rpy="[^"]*"', f'rpy="{nrpy_s}"', new_tag)
    if new_tag == zed0:
        print("\nERROR: nothing changed in the zed_0 tag; not writing.")
        return
    open(URDF, "w").write(urdf.replace(zed0, new_tag, 1))
    print(f"\nwrote zed_0 ({parent_link}) into {URDF}")
    print("now rebuild fpi_crane_description on the robot and relaunch bringup.")


if __name__ == "__main__":
    main()
