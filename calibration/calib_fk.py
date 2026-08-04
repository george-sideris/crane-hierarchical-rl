#!/usr/bin/env python3
"""FPI crane forward kinematics from the URDF (fpi_crane.urdf.xacro), for model-based
camera calibration. Encodes the joint chain and the nominal camera mounts, and lets
us recompute FK at (joint_value + calibration_offset). Validated against the bag /tf.

Frames of interest:
  zed_0 (basemast) calibrates in the `mast` frame  (slew cancels in cam<-grapple)
  zed_1 (stick)    calibrates in the `stick` frame
"""
import numpy as np

DEG = np.pi / 180.0


def rpyT(rpy, xyz):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r); cp, sp = np.cos(p), np.sin(p); cy, sy = np.cos(y), np.sin(y)
    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [-sp,   cp*sr,            cp*cr]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
    return T


def axis_angle(axis, q):
    a = np.array(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
    T = np.eye(4); T[:3, :3] = R
    return T


# joint: name -> (parent, child, origin_xyz, origin_rpy, axis_xyz, axis_rpy, type)
JOINTS = {
    'slew':          ('base_link', 'mast',         (0, 0, 0.26881),                 (0,0,0), (0,0,1),               (0,0,0),        'rev'),
    'boom':          ('mast', 'mainboom',          (0, -0.0601374, 1.09553),        (0,0,0), (1,0,0),               (0,0,0),        'rev'),
    'stick':         ('mainboom', 'stick',         (0, 3.04904, -0.00649),          (0,0,0), (1,0,0),               (0,0,0),        'rev'),
    'telescope':     ('stick', 'telescope',        (0, 0.1256771629, 0.21905106705),(0,0,0), (0,0.997833966,-0.065782793),(0,0,0), 'pris'),
    'hanger':        ('telescope', 'upperpassive', (0, 2.03711, -0.20229),          (0,0,0), (1,0,0),               (0,0,0),        'rev'),
    'bearingfork':   ('upperpassive', 'lowerpassive',(0, 0, -0.154),                (0,0,0), (0,1,0),               (0,0,0),        'rev'),
    'grapplecarrier':('lowerpassive', 'grapplecarrier',(0, 0, -0.261),              (0,0,0), (0,0,1),               (0,-1.22173,0), 'rev'),
    'grappletong1':  ('grapplecarrier', 'grappletong1',(-0.275, -0.0192847891741644, -0.202),(0,0,0),(0,-1,0),      (0,0,0),        'rev'),
    'grappletong2':  ('grapplecarrier', 'grappletong2',(0.275, -0.0192847891741644, -0.202),(0,0,0),(0,1,0),        (0,0,0),        'rev'),
}

# ordered chains from a root link down to the grapple
CHAIN = {
    'mast':  ['boom', 'stick', 'telescope', 'hanger', 'bearingfork', 'grapplecarrier'],
    'stick': ['telescope', 'hanger', 'bearingfork', 'grapplecarrier'],
    'base_link': ['slew', 'boom', 'stick', 'telescope', 'hanger', 'bearingfork', 'grapplecarrier'],
}

# nominal camera mount: list of (xyz, rpy) fixed transforms from the rigid parent to the
# camera's left optical frame (parent -> ... -> camera_link -> center -> left -> optical).
# zed internal frames are the same for both (from tf_static / zed_macro).
_ZED_INTERNAL = [
    ((0, 0, 0.016), (0, 0, 0)),                 # camera_link -> camera_center
    ((-0.01, 0.06, 0), (0, 0, 0)),              # camera_center -> left_camera_frame
    ((0, 0, 0), (-np.pi/2, 0, -np.pi/2)),       # left_camera_frame -> optical (REP103)
]
NOMINAL_MOUNT = {
    # zed_1: stick -> camera_zed -> zed_1_camera_link(rpy 0,0,1.57) -> internal
    'zed_1': [((0, 0.37, 0.03), (0, 0, 0)), ((0, 0, 0), (0, 0, 1.57))] + _ZED_INTERNAL,
    # zed_0: mast -> bracket -> lidar -> zed_0_camera_link -> internal
    'zed_0': [((-0.15, 0, 0.74), (0, 0.1, -0.05)), ((0, 0, 0), (3.14, 0, 1.57)),
              ((0.05, 0.1, -0.3), (3.0, -0.2, -0.05))] + _ZED_INTERNAL,
    # lidar_0: single transform mast -> os_sensor (= bracket @ ouster, the cloud frame).
    # Parametrized as one 6-dof pose; cloud (/lidar_0/points) is already in os_sensor.
    'lidar_0': [((-0.15, 0.0, 0.74), (-3.04319, 0.00008, 1.52))],
}
CAM_PARENT = {'zed_0': 'mast', 'zed_1': 'stick', 'lidar_0': 'mast'}

# sensor point-cloud config: topic, cloud frame, and whether the cloud frame == the
# mount-output frame (lidar) or needs the optical-edge transform applied (cameras).
SENSORS = {
    'zed_0': dict(topic='/zed_0/zed_node/point_cloud/cloud_registered',
                  cloud_frame='zed_0_left_camera_frame', is_optical=True),
    'zed_1': dict(topic='/zed_1/zed_node/point_cloud/cloud_registered',
                  cloud_frame='zed_1_left_camera_frame', is_optical=True),
    'lidar_0': dict(topic='/lidar_0/points', cloud_frame='lidar_0/os_sensor', is_optical=False),
}


def joint_T(name, value):
    p, c, oxyz, orpy, axyz, arpy, typ = JOINTS[name]
    T = rpyT(orpy, oxyz)
    if typ == 'rev':
        # NOTE: standard robot_state_publisher (and the bag /tf) IGNORE the non-standard
        # <axis rpy=...> attribute and rotate about the plain xyz axis. Match that.
        T = T @ axis_angle(np.array(axyz), value)
    else:  # prismatic
        ax = np.array(axyz, float); ax /= np.linalg.norm(ax)
        Tt = np.eye(4); Tt[:3, 3] = ax * value
        T = T @ Tt
    return T


def fk_chain(root, joints_values, offsets=None):
    """T_root <- last_child, composing CHAIN[root] with given joint values (+offsets)."""
    offsets = offsets or {}
    T = np.eye(4)
    for j in CHAIN[root]:
        T = T @ joint_T(j, joints_values[j] + offsets.get(j, 0.0))
    return T


def mount_T(cam, params=None):
    """T_parent <- cam_optical. params overrides the FIRST fixed transform (xyz mm + rpy deg)
    when given as (x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg); rest of chain stays nominal."""
    chain = list(NOMINAL_MOUNT[cam])
    if params is not None:
        x, y, z, roll, pit, yaw = params
        chain[0] = ((x/1000.0, y/1000.0, z/1000.0), (roll*DEG, pit*DEG, yaw*DEG))
    T = np.eye(4)
    for xyz, rpy in chain:
        T = T @ rpyT(rpy, xyz)
    return T


def nominal_mount_params(cam):
    """Return the nominal first-transform as (x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg)."""
    (x, y, z), (r, p, yw) = NOMINAL_MOUNT[cam][0]
    return np.array([x*1000, y*1000, z*1000, r/DEG, p/DEG, yw/DEG])
