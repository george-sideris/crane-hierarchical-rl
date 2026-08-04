#!/usr/bin/env python3
"""Crane grasping policy node.

Subscribes to a depth camera and joint states, runs a learned point-cloud
policy to pick a grasp target, then sequences a pick-and-place cycle. Each
arm phase issues an IK solve, then calls the /relative_joint_move service
to ship the smooth trajectory through the PLC. Gripper phases call
/request_grapple_move directly.

Subscriptions:
    /zedx/points        (sensor_msgs/PointCloud2)  PRIMARY: ZED registered cloud -> base -> crop -> FPS
    /zedx/depth         (sensor_msgs/Image, 32FC1, meters)  fallback if a depth image is published instead
    /zedx/camera_info   (sensor_msgs/CameraInfo)            (only used by the depth path)
    /joint_states       (sensor_msgs/JointState, from the PLC node)
    /plc_status         (fpi_crane_msgs/PlcStatus)
    /tf, /tf_static     (tf2_msgs/TFMessage)

Publications:
    /crane/policy_target (geometry_msgs/PoseStamped, once per cycle)
    /crane/ee_command    (geometry_msgs/PoseStamped, per phase)
    /crane/fsm_state     (std_msgs/String)

Service clients:
    /relative_joint_move   (fpi_crane_msgs/RelativeJointMove)
    /request_grapple_move  (fpi_crane_msgs/RequestGrappleMove)

Run:
    ros2 run fpi_crane_rl crane_policy_node --ros-args \\
        -p policy_type:=bcrl -p checkpoint_path:=<path>/model_350.pt
"""

import argparse
import json
import math
import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField, CompressedImage
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

from fpi_crane_msgs.msg import PlcStatus
from fpi_crane_msgs.srv import RequestGrappleMove, RelativeJointMove

from fpi_crane_rl.pointcloud_pipeline import process_depth, process_cloud, transform_points
from fpi_crane_rl.fsm import FSMConfig, Phase
from fpi_crane_rl.policy_loader import load_policy
from fpi_crane_rl import ik

# PLC crane states (fpi_crane_hw robot_globals.PlcState).
PLC_STATE_HOLD = 1
PLC_STATE_EXECUTE = 2
PLC_STATE_END = 3
# Gripper directions (fpi_crane_hw robot_globals.GripperDirection).
GRIPPER_OPEN = 1
GRIPPER_CLOSE = 2

# Sequencer states.
SEQ_IDLE = "idle"
SEQ_GAZE = "gaze"          # driving to / settling at the gaze pose before capturing the cloud
SEQ_SEND = "send"
SEQ_WAIT_TRAJ = "wait_traj"
SEQ_WAIT_GRIP = "wait_grip"

# Default frame and joint names (override via ROS params).
WORLD_FRAME = "base_link"
BASE_FRAME = "base_link"
CAMERA_FRAME = "zedx_camera"
BASEGRAPPLE_FRAME = "grapplecarrier"   # URDF link for the grapple yaw (sim USD body == "basegrapple")
YAW_JOINT_NAME = "grapplecarrier_joint"
GRIPPER_JOINT_NAMES = ("grappletong1_joint", "grappletong2_joint")
ARM_JOINT_NAMES = ("slew_joint", "boom_joint", "stick_joint", "telescope_joint")


@dataclass
class Move:
    """One step of the pick-and-place cycle."""
    phase: Phase
    kind: str                      # "traj" (cartesian IK), "slew" (joint-space slew), or "grip"
    position: list = None          # EE target (base frame) for "traj"
    yaw: float = None              # base-frame grapple yaw; None = hold current
    gripper: int = None            # GRIPPER_OPEN / GRIPPER_CLOSE for "grip"
    slew: float = None             # absolute slew joint target (rad) for "slew"; arm held otherwise
    velocity: float = None         # per-move joint velocity override (rad/s); None -> traj_velocity
    min_duration: float = None     # per-move min-duration floor (s) override; None -> traj_min_duration


def _quat_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _resolve_urdf_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("fpi_crane_rl"),
                            "urdf", "fpiforwarder-upperpassive.urdf")
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "urdf", "fpiforwarder-upperpassive.urdf")


class CranePolicyNode(Node):
    def __init__(self, policy_type_override=None, checkpoint_override=None,
                 dry_run_override=False):
        super().__init__("crane_policy_node")

        # Parameters
        self.declare_parameter("policy_type", policy_type_override or "heuristic")
        self.declare_parameter("checkpoint_path", checkpoint_override or "")
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("depth_range_min", 1.0)
        self.declare_parameter("depth_range_max", 10.0)
        self.declare_parameter("cossin", True)
        self.declare_parameter("device",
                               "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("tick_rate", 10.0)
        self.declare_parameter("dry_run", dry_run_override)
        self.declare_parameter("max_cycles", 30)
        # Crop box = the sim ACTION bounds (must match training: crane_rl_env_gaze.py action box).
        # 2026-06-28 calibrated rack: center (-4.364, 1.816), HALF_X=1.0 HALF_Y=3.5, Z [-1.30, 0.10].
        # (prev/stale old-rack: min [-5.0,-0.75,-1.373] max [-3.0,4.59,-0.373])
        self.declare_parameter("bounds_min", [-5.364, -1.684, -1.30])
        self.declare_parameter("bounds_max", [-3.364, 5.316, 0.10])
        self.declare_parameter("hover_clear", 2.5)
        self.declare_parameter("approach_above", 0.6)
        self.declare_parameter("drop_position", [0.0, 3.0, 3.0])
        # pi/2 so the grapple opening is PARALLEL to the trailer (0.0 came out perpendicular to the bed).
        self.declare_parameter("drop_yaw", 1.5708)
        # Deposit scan: after CARRY_HOME (at the home x,y + yaw aligned), scan the basemast cloud for the
        # highest log in the trailer box and LOWER to that + offset before releasing (descend-only, capped
        # at the home z). Off by default -> old behavior (drop straight from drop_position). Bounds are in
        # base_link frame and match the sim --trailer_box_* viz.
        self.declare_parameter("deposit_scan", False)
        self.declare_parameter("deposit_offset", 0.5)          # m above the detected pile top
        self.declare_parameter("deposit_min_points", 30)       # min in-box points to trust the scan
        self.declare_parameter("trailer_bounds_min", [-0.7, 0.2, 0.0])
        # z ceiling 1.5 (was 2.0) so the scan doesn't catch the held logs / trailer rim up near the grapple
        # at the home pose (that inflated the pile top and kept the deposit high).
        self.declare_parameter("trailer_bounds_max", [0.7, 5.302, 1.5])
        # If the scan finds nothing in the box, still lower to this fixed z (base_link) before releasing
        # instead of dropping from full home height. Clamped to <= home z (descend-only).
        self.declare_parameter("deposit_fallback_z", 2.0)
        # If true, run the scan + publish the deposit target/markers but SKIP the actual descend (watch
        # where it WOULD lower without moving the arm down). The rest of the cycle still runs.
        self.declare_parameter("deposit_dry_run", False)
        # The registered-cloud pipeline lags: a cloud that ARRIVES right after settle still holds data
        # captured ~this many seconds earlier (mid-motion). Require the cloud to arrive this long AFTER
        # the settle dwell so its CONTENT is genuinely post-settle. Applies to gaze + deposit scans.
        self.declare_parameter("cloud_capture_latency_s", 0.6)
        # Gaze pose: drive here (joint targets) before every cloud capture so the basemast cam
        # sees the rack from the SAME viewpoint the policy was trained on (sim PH_GAZE). Defaults
        # = crane_rl_env_gaze.py GAZE_* (slew aims at the rack; arm tucked so it doesn't occlude).
        self.declare_parameter("gaze_slew", 1.1205)
        self.declare_parameter("gaze_boom", 0.7871)
        self.declare_parameter("gaze_stick", -0.9198)
        self.declare_parameter("gaze_telescope", 0.1300)
        self.declare_parameter("gaze_dwell_s", 0.5)   # settle time at gaze before capturing
        # If true, the GAZE positioning move runs even in dry_run (so you can watch it reach the
        # gaze viewpoint + see the target), while the pick/grasp stays dry. Gaze is benign motion.
        self.declare_parameter("gaze_in_dry_run", False)
        # DEBUG: if set, dump each cycle's cropped policy-input cloud + target to <dir>/policy_debug_NNN.npz
        # for an offline Open3D view (calibration/view_policy_debug.py). "" disables it. Pass the literal
        # "auto" to write to <checkpoint_dir>/policy_debug/run_<stamp>/ -- i.e. a fresh per-run dir kept
        # WITH that policy's logs, so runs never overwrite each other. Any other value is an explicit path.
        self.declare_parameter("debug_save_dir", "")
        # Also render each cycle's decision as an iso-view JPG next to the NPZ (needs matplotlib).
        self.declare_parameter("debug_save_jpg", True)
        # Record the basemast RGB stream to <debug_save_dir>/basemast_run.mp4 so a run can be reviewed
        # as video alongside the per-cycle decision dumps. Rate-limited to video_fps. Needs
        # imageio(+ffmpeg); on failure it warns once and disables (use `ros2 bag record` instead).
        # video_topic may be raw (sensor_msgs/Image) or compressed (sensor_msgs/CompressedImage) -- a
        # topic ending in "compressed" is auto-detected and JPEG-decoded. The ZED publishes compressed.
        self.declare_parameter("record_video", False)
        self.declare_parameter("video_topic", "/zed_0/zed_node/rgb/color/rect/image/compressed")
        self.declare_parameter("video_fps", 15.0)
        self.declare_parameter("video_swap_rb", False)   # force R<->B for compressed (default off is correct for cv2 jpegs)
        # When record_video is on, the node also auto-stitches <debug_save_dir>/review.mp4 at shutdown
        # (per-cycle policy input/target panels over the basemast stream) via render_run_video.py.
        # Needs open3d+imageio in this env. render_video_script = where that script lives.
        self.declare_parameter("render_video_script",
                               "/workspace/crane_testbed/calibration/render_run_video.py")
        self.declare_parameter("camera_frame", CAMERA_FRAME)
        self.declare_parameter("base_frame", BASE_FRAME)
        self.declare_parameter("world_frame", WORLD_FRAME)
        self.declare_parameter("basegrapple_frame", BASEGRAPPLE_FRAME)
        # Constant correction added to the measured grapple yaw, for any fixed convention
        # difference between the URDF grapplecarrier frame and the sim basegrapple body
        # (set e.g. 1.5708 if a residual 90deg offset remains after the frame fix).
        self.declare_parameter("grapple_yaw_offset", 0.0)
        self.declare_parameter("yaw_joint_name", YAW_JOINT_NAME)
        self.declare_parameter("urdf_path", "")
        # Peak joint velocity sent to /relative_joint_move. The trajectory
        # planner derives segment duration from this and the largest joint
        # delta.
        # The telescope (slow hydraulic prismatic) can't track a fast trajectory: too-short moves
        # leave it short and blow the phase timeout. Instead of slowing EVERY joint (glacial), we
        # size each move's min_duration to its telescope delta at telescope_speed -> revolute-only
        # moves stay quick, only telescope-heavy moves take longer.
        self.declare_parameter("traj_velocity", 0.1)
        self.declare_parameter("traj_min_duration", 5.0)
        # ALIGN_YAW is a near-pure grapple rotation (arm held), but at traj_velocity/min_duration it took
        # ~15s. Give the yaw-align phase its own faster velocity + lower floor.
        self.declare_parameter("yaw_velocity", 0.4)
        self.declare_parameter("yaw_min_duration", 2.0)
        self.declare_parameter("traj_timeout", 60.0)
        self.declare_parameter("telescope_speed", 0.04)   # m/s; lower => more time for the telescope
        self.declare_parameter("grip_timeout", 15.0)
        # CLOSE drive time (ms). The grapple is a slow hydraulic: 4000 ms expired before it reached full
        # close, so the jaws stopped partway (loose -> logs fell out on lift). Give it longer to finish.
        self.declare_parameter("grapple_delta_time_ms", 8000)
        # Relative close angle (deg). 120 is ~full mechanical close; don't drive past it (strains the
        # hard stop). If still loose at full close, it's a time/depth issue, not angle.
        self.declare_parameter("grapple_delta_angle_deg", 120)
        # Convergence tolerances for the phase-advance gate: only advance the
        # FSM when the crane has actually reached the commanded joint config,
        # not just when the PLC reports the trajectory finished.
        self.declare_parameter("arm_tolerance", 0.05)        # rad
        self.declare_parameter("telescope_tolerance", 0.03)  # m
        self.declare_parameter("yaw_tolerance", 0.08)        # rad
        # Telescope handling. Default False: the telescope is active (real encoder used,
        # optimized in IK, commanded normally). Set telescope_static:=true to hold it
        # fixed when the encoder is unreliable: feedback is treated as the constant
        # telescope_static_value, it is masked inactive in IK, and it is never commanded
        # to move. Must match the PLC node's telescope_static / telescope_static_value.
        self.declare_parameter("telescope_static", False)
        self.declare_parameter("telescope_static_value", 0.13)  # m

        policy_type = self.get_parameter("policy_type").value
        checkpoint = self.get_parameter("checkpoint_path").value
        self.num_points = self.get_parameter("num_points").value
        self.depth_range = (self.get_parameter("depth_range_min").value,
                            self.get_parameter("depth_range_max").value)
        cossin = self.get_parameter("cossin").value
        device = self.get_parameter("device").value
        tick_rate = self.get_parameter("tick_rate").value
        self.dry_run = self.get_parameter("dry_run").value
        self.max_cycles = self.get_parameter("max_cycles").value
        bounds_min = np.array(self.get_parameter("bounds_min").value, dtype=np.float32)
        bounds_max = np.array(self.get_parameter("bounds_max").value, dtype=np.float32)
        self._bounds_min = bounds_min   # also used to crop the cloud (must match training)
        self._bounds_max = bounds_max
        self._gaze_joints = np.array([
            self.get_parameter("gaze_slew").value,
            self.get_parameter("gaze_boom").value,
            self.get_parameter("gaze_stick").value,
            self.get_parameter("gaze_telescope").value], dtype=np.float64)
        self._gaze_dwell_s = float(self.get_parameter("gaze_dwell_s").value)
        self._gaze_in_dry_run = bool(self.get_parameter("gaze_in_dry_run").value)
        self._deposit_scan = bool(self.get_parameter("deposit_scan").value)
        self._deposit_offset = float(self.get_parameter("deposit_offset").value)
        self._deposit_min_points = int(self.get_parameter("deposit_min_points").value)
        self._deposit_dry_run = bool(self.get_parameter("deposit_dry_run").value)
        self._trailer_bounds_min = np.array(self.get_parameter("trailer_bounds_min").value, dtype=np.float32)
        self._trailer_bounds_max = np.array(self.get_parameter("trailer_bounds_max").value, dtype=np.float32)
        self._deposit_fallback_z = float(self.get_parameter("deposit_fallback_z").value)
        self._cloud_latency_s = float(self.get_parameter("cloud_capture_latency_s").value)
        self._deposit_start_time = None
        self._deposit_waiting_fresh = False
        self._last_deposit_pts = None   # in-box trailer points from the latest scan (for the npz dump)
        self._debug_save_dir = self.get_parameter("debug_save_dir").value
        if self._debug_save_dir == "auto":
            from datetime import datetime
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Keep debug dumps WITH the policy: <checkpoint_dir>/policy_debug/run_<stamp>/, so each
            # run of a given policy gets its own dir under that policy's logs (never overwritten).
            # Fall back to ~ only when there is no checkpoint (e.g. heuristic policy).
            if checkpoint:
                base = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "policy_debug")
            else:
                base = os.path.join(os.path.expanduser("~"), "crane_policy_debug")
            self._debug_save_dir = os.path.join(base, f"run_{stamp}")
        if self._debug_save_dir:
            self.get_logger().info(f"Debug NPZ dumps -> {self._debug_save_dir}")
        self._debug_save_jpg = bool(self.get_parameter("debug_save_jpg").value)
        # Basemast video recording state (writer opened lazily on the first frame).
        self._record_video = bool(self.get_parameter("record_video").value) and bool(self._debug_save_dir)
        self._video_topic = self.get_parameter("video_topic").value
        self._video_compressed = self._video_topic.rstrip("/").endswith("compressed")
        self._video_swap_rb = bool(self.get_parameter("video_swap_rb").value)
        self._video_fps = float(self.get_parameter("video_fps").value)
        self._render_video_script = self.get_parameter("render_video_script").value
        self._video_writer = None
        self._video_frames = 0
        self._last_video_stamp = None
        self._gaze_reached_time = None
        self._gaze_waiting_fresh = False

        self._camera_frame = self.get_parameter("camera_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._world_frame = self.get_parameter("world_frame").value
        self._basegrapple_frame = self.get_parameter("basegrapple_frame").value
        self._grapple_yaw_offset = float(self.get_parameter("grapple_yaw_offset").value)
        self._yaw_joint_name = self.get_parameter("yaw_joint_name").value
        self._traj_velocity = float(self.get_parameter("traj_velocity").value)
        self._traj_min_duration = float(self.get_parameter("traj_min_duration").value)
        self._yaw_velocity = float(self.get_parameter("yaw_velocity").value)
        self._yaw_min_duration = float(self.get_parameter("yaw_min_duration").value)
        self._traj_timeout = self.get_parameter("traj_timeout").value
        self._telescope_speed = float(self.get_parameter("telescope_speed").value)
        self._grip_timeout = self.get_parameter("grip_timeout").value
        self._grapple_dt_ms = self.get_parameter("grapple_delta_time_ms").value
        self._grapple_da_deg = self.get_parameter("grapple_delta_angle_deg").value
        self._arm_tol = float(self.get_parameter("arm_tolerance").value)
        self._telescope_tol = float(self.get_parameter("telescope_tolerance").value)
        self._yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        self._telescope_static = bool(self.get_parameter("telescope_static").value)
        self._telescope_static_value = float(self.get_parameter("telescope_static_value").value)

        # Camera/base extrinsics (overridden by TF lookup once available)
        self.cam_pos = np.array([-1.0, 1.92, 1.577], dtype=np.float32)
        self.cam_quat = np.array([0.6124, 0.3536, 0.3536, 0.6124], dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Policy
        ckpt = checkpoint if checkpoint else None
        self.policy = load_policy(policy_type, ckpt, bounds_min, bounds_max,
                                  cossin, device)
        self.get_logger().info(
            f"Policy: {policy_type}" +
            (f" from {checkpoint}" if checkpoint else "") +
            f"  bounds_min={bounds_min.round(3).tolist()} "
            f"bounds_max={bounds_max.round(3).tolist()}")

        # FSM config (phase geometry) and IK / planner
        self.fsm_cfg = FSMConfig(
            hover_clear=self.get_parameter("hover_clear").value,
            approach_above=self.get_parameter("approach_above").value,
            drop_position=self.get_parameter("drop_position").value,
            drop_yaw=self.get_parameter("drop_yaw").value,
        )
        urdf_path = self.get_parameter("urdf_path").value or _resolve_urdf_path()
        # Optimize the telescope in IK only when it is not held static (encoder working).
        self.ik = ik.CraneIK(urdf_path, telescope_active=not self._telescope_static)
        self.get_logger().info(
            f"IK loaded from {urdf_path} "
            f"(telescope {'STATIC/locked' if self._telescope_static else 'ACTIVE'})")

        # State
        self.intrinsics = None
        self.latest_depth = None
        self.latest_cloud = None       # (N,3) ZED registered cloud in its own frame
        self._cloud_frame = None       # frame_id of the cloud (e.g. zed_0_left_camera_frame)
        self._cloud_recv_time = None   # node-clock time the latest_cloud arrived (for post-settle freshness)
        self.current_ee_pos = [0.0, 0.0, 0.0]
        self.current_ee_yaw = 0.0
        self.current_arm = np.zeros(4)
        self.current_yaw_joint = 0.0
        self.current_gripper = 0.0
        self._joints_received = False
        self._tf_initialized = False

        # PLC status
        self.plc_state = PLC_STATE_HOLD
        self.plc_queue = 0
        self._plc_received = False

        # Sequencer
        self.moves = []
        self.move_idx = 0
        self.seq_state = SEQ_IDLE
        self.cycle_count = 0
        self._seq_id = 0
        self._seen_executing = False
        self._wait_start = None
        self._grapple_future = None
        self._traj_future = None
        self._goal_joints = None

        # Subscribers. Primary input is the ZED registered cloud (matches what the ZED publishes);
        # depth-image path kept for setups that publish a depth image instead.
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(PointCloud2, "/zedx/points", self._cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Image, "/zedx/depth", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/zedx/camera_info",
                                 self._caminfo_cb, 10)
        self.create_subscription(JointState, "/joint_states",
                                 self._joint_states_cb, 10)
        self.create_subscription(PlcStatus, "/plc_status",
                                 self._plc_status_cb, 10)
        if self._record_video:
            vid_type = CompressedImage if self._video_compressed else Image
            self.create_subscription(vid_type, self._video_topic, self._video_cb,
                                     qos_profile_sensor_data)
            self.get_logger().info(
                f"Recording basemast video: {self._video_topic} "
                f"({'compressed' if self._video_compressed else 'raw'}) -> "
                f"{self._debug_save_dir}/basemast_run.mp4 @ {self._video_fps:.0f} fps")

        # TF (basegrapple pose plus camera/base extrinsics)
        from tf2_ros import Buffer, TransformListener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publishers
        self.pub_fsm_state = self.create_publisher(String, "/crane/fsm_state", 10)
        self.pub_policy_target = self.create_publisher(PoseStamped,
                                                       "/crane/policy_target", 10)
        self.pub_ee_cmd = self.create_publisher(PoseStamped, "/crane/ee_command", 10)
        # DEBUG: the exact cropped+FPS'd cloud the policy sees (base frame) -> view in RViz.
        self.pub_policy_input = self.create_publisher(PointCloud2, "/crane/policy_input", 1)
        # DEBUG: the action-box crop region (base frame) as a translucent cube -> view in RViz.
        self.pub_crop_box = self.create_publisher(Marker, "/crane/crop_box", 1)
        # DEBUG (deposit): trailer scan box (cube), the in-box sampled points, and the chosen deposit
        # target (sphere at [home_x, home_y, deposit_z]) -> view in RViz.
        self.pub_deposit_box = self.create_publisher(Marker, "/crane/trailer_box", 1)
        self.pub_deposit_scan = self.create_publisher(PointCloud2, "/crane/deposit_scan", 1)
        self.pub_deposit_target = self.create_publisher(Marker, "/crane/deposit_target", 1)

        # Service clients: relative-joint trajectory planner and grapple action
        self.move_client = self.create_client(RelativeJointMove,
                                              "relative_joint_move")
        self.grapple_client = self.create_client(RequestGrappleMove,
                                                 "request_grapple_move")

        self.create_timer(1.0 / tick_rate, self._tick)
        self.get_logger().info(
            f"Started (tick={tick_rate}Hz, dry_run={self.dry_run})")

    # Callbacks

    def _cloud_cb(self, msg):
        # Parse a PointCloud2 into (N,3) XYZ in the cloud's own frame. Manual field-offset
        # decode (no sensor_msgs_py dependency); keeps only finite points.
        offs = {f.name: f.offset for f in msg.fields}
        if not all(k in offs for k in ("x", "y", "z")):
            return
        ps = msg.point_step
        arr = np.frombuffer(msg.data, np.uint8).reshape(len(msg.data) // ps, ps)

        def fld(name):
            o = offs[name]
            return arr[:, o:o + 4].copy().view(np.float32).ravel()

        x, y, z = fld("x"), fld("y"), fld("z")
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        self.latest_cloud = np.c_[x[m], y[m], z[m]].astype(np.float32)
        self._cloud_frame = msg.header.frame_id
        self._cloud_recv_time = self.get_clock().now()

    def _depth_cb(self, msg):
        # Manual decode based on encoding. Avoids a cv_bridge SystemError
        # (Boost.Python / NumPy 2.x ABI mismatch) that fires inside
        # cvtColor2 when imgmsg_to_cv2 is called.
        if msg.encoding == "32FC1":
            depth = np.frombuffer(
                msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding in ("16UC1", "mono16"):
            depth = np.frombuffer(
                msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            self.get_logger().warn(
                f"Unsupported depth encoding: {msg.encoding}; skipping frame")
            return
        self.latest_depth = depth

    def _caminfo_cb(self, msg):
        if self.intrinsics is None:
            K = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.intrinsics = K
            self.get_logger().info(
                f"Camera intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    def _joint_states_cb(self, msg):
        name_to_pos = {n: msg.position[i] for i, n in enumerate(msg.name)
                       if i < len(msg.position)}
        try:
            self.current_arm = np.array([name_to_pos[n] for n in ARM_JOINT_NAMES])
        except KeyError:
            return
        if self._telescope_static:
            # Ignore the unreliable telescope sensor; treat it as locked. This keeps the
            # IK seed, commanded delta, and convergence check consistent with a held joint.
            self.current_arm[3] = self._telescope_static_value
        if self._yaw_joint_name in name_to_pos:
            self.current_yaw_joint = float(name_to_pos[self._yaw_joint_name])
        for n in GRIPPER_JOINT_NAMES:
            if n in name_to_pos:
                self.current_gripper = float(name_to_pos[n])
                break
        self._joints_received = True

    def _plc_status_cb(self, msg):
        self.plc_state = int(msg.plc_crane_state)
        self.plc_queue = int(msg.plc_trajectory_count_in_queue)
        self._plc_received = True

    # TF helpers

    def _update_extrinsics_from_tf(self):
        if self._tf_initialized:
            return
        try:
            t_cam = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, rclpy.time.Time())
            p, q = t_cam.transform.translation, t_cam.transform.rotation
            self.cam_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.cam_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            t_base = self._tf_buffer.lookup_transform(
                self._world_frame, self._base_frame, rclpy.time.Time())
            p, q = t_base.transform.translation, t_base.transform.rotation
            self.base_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.base_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            self._tf_initialized = True
            self.get_logger().info(
                f"TF loaded: cam_pos={self.cam_pos.round(3).tolist()}, "
                f"base_pos={self.base_pos.round(3).tolist()}")
        except Exception:
            pass

    def _update_basegrapple_from_tf(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._base_frame, self._basegrapple_frame, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            if not hasattr(self, "_bg_tf_warn"):
                self.get_logger().warn(
                    f"tf {self._base_frame}<-{self._basegrapple_frame} unavailable ({e}); "
                    f"current grapple yaw stays 0 -> yaw command WILL be wrong. "
                    f"Set -p basegrapple_frame:=<the real grapple frame>.")
                self._bg_tf_warn = True
            return
        p = t.transform.translation
        q = t.transform.rotation
        self.current_ee_pos = [p.x, p.y, p.z]
        self.current_ee_yaw = _quat_to_yaw(q.w, q.x, q.y, q.z) - self._grapple_yaw_offset

    def _grapple_yaw_to_joint(self, target_grapple_yaw_b: float) -> float:
        def _wrap(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        rotation_diff = _wrap(target_grapple_yaw_b - self.current_ee_yaw)
        base = self.current_yaw_joint + rotation_diff
        candidates = [base, base + math.pi, base - math.pi]
        return min(candidates, key=lambda c: abs(c - self.current_yaw_joint))

    # Main sequencer tick

    def _tick(self):
        if self.cycle_count >= self.max_cycles:
            return
        self._update_extrinsics_from_tf()
        self._update_basegrapple_from_tf()

        if self.seq_state == SEQ_IDLE:
            self._start_gaze()
        elif self.seq_state == SEQ_GAZE:
            self._check_gaze_done()
        elif self.seq_state == SEQ_SEND:
            self._send_current_move()
        elif self.seq_state == SEQ_WAIT_TRAJ:
            self._check_traj_done()
        elif self.seq_state == SEQ_WAIT_GRIP:
            self._check_grip_done()

    def _start_gaze(self):
        """Drive to the gaze pose (joint targets) so the cloud is captured from the trained
        viewpoint. Mirrors the IK trajectory send but commands the gaze joints directly."""
        if not self._joints_received or not self._plc_received:
            return
        if not (self.plc_state == PLC_STATE_HOLD and self.plc_queue == 0):
            return
        g = self._gaze_joints.copy()
        if self._telescope_static:
            g[3] = self._telescope_static_value   # hold telescope; never move it at gaze
        self._goal_joints = np.array([g[0], g[1], g[2], g[3], self.current_yaw_joint])
        self._gaze_reached_time = None
        self._gaze_waiting_fresh = False
        self._deposit_start_time = None
        self._deposit_waiting_fresh = False
        m = String(); m.data = f"GAZE (cycle {self.cycle_count})"; self.pub_fsm_state.publish(m)
        self._append_phase_log("GAZE")

        if self.dry_run and not self._gaze_in_dry_run:
            self.get_logger().info(
                f"GAZE: dry_run (no motion); capturing from current pose. "
                f"gaze target slew={g[0]:.3f} boom={g[1]:.3f} stick={g[2]:.3f} tele={g[3]:.3f}")
            self._seen_executing = False
            self._wait_start = self.get_clock().now()
            self.seq_state = SEQ_GAZE
            return
        if not self.move_client.service_is_ready():
            if not hasattr(self, "_move_warn"):
                self.get_logger().warn("/relative_joint_move not available (relative_joint_mover running?)")
                self._move_warn = True
            return
        delta = [float(g[i] - self.current_arm[i]) for i in range(4)]
        self._seq_id += 1
        req = RelativeJointMove.Request()
        req.sequence_id = self._seq_id
        req.joint_names = ["slew_joint", "boom_joint", "stick_joint", "telescope_joint"]
        req.delta_positions = delta
        req.velocity = self._traj_velocity
        req.min_duration = max(self._traj_min_duration,
                               abs(delta[3]) / max(self._telescope_speed, 1e-3))
        self._traj_future = self.move_client.call_async(req)
        self._seen_executing = False
        self._wait_start = self.get_clock().now()
        self.seq_state = SEQ_GAZE
        self.get_logger().info(f"GAZE: drive to gaze pose, max_delta={max(abs(d) for d in delta):.3f} rad")

    def _check_gaze_done(self):
        """Wait until the crane reaches + settles at the gaze pose, then capture + plan."""
        if not self._seen_executing and (
                self.plc_state == PLC_STATE_EXECUTE or self.plc_queue > 0):
            self._seen_executing = True
        plc_idle = (self._seen_executing
                    and self.plc_state in (PLC_STATE_HOLD, PLC_STATE_END)
                    and self.plc_queue == 0)
        reached = (plc_idle and self._converged_to_goal()) or (self.dry_run and not self._gaze_in_dry_run)
        if not reached:
            if self._waited_longer_than(self._traj_timeout):
                self.get_logger().warn("GAZE: timeout reaching gaze pose; capturing anyway")
            else:
                return
        # settle dwell at the gaze pose so the cloud is from the held viewpoint
        if self._gaze_reached_time is None:
            self._gaze_reached_time = self.get_clock().now()
            return
        deadline = self._gaze_reached_time + Duration(seconds=self._gaze_dwell_s)
        if self.get_clock().now() < deadline:
            return
        # the ZED registered-cloud pipeline lags: a cloud that ARRIVES right after settle still holds data
        # captured ~cloud_latency earlier (mid-slew). Require arrival >= settle + latency so its CONTENT is
        # genuinely post-settle, not just its receive time.
        fresh_after = deadline + Duration(seconds=self._cloud_latency_s)
        if self._cloud_recv_time is None or self._cloud_recv_time < fresh_after:
            if not self._gaze_waiting_fresh:
                self.get_logger().info("GAZE: settled, waiting for a fresh post-settle cloud...")
                self._gaze_waiting_fresh = True
            return
        self._gaze_reached_time = None
        self._gaze_waiting_fresh = False
        self._capture_and_plan()

    def _capture_and_plan(self):
        if self.latest_cloud is None or self._cloud_frame is None:
            return
        # base_link <- cloud_frame in one tf hop (mirrors training get_pointcloud_base).
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._cloud_frame, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(
                f"tf {self._base_frame}<-{self._cloud_frame} unavailable: {e}")
            return
        tr, q = tf.transform.translation, tf.transform.rotation
        bfc_pos = np.array([tr.x, tr.y, tr.z], dtype=np.float32)
        bfc_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)  # wxyz
        obs = process_cloud(
            self.latest_cloud, bfc_pos, bfc_quat,
            self._bounds_min, self._bounds_max, self.num_points,
        )
        self._publish_input_cloud(obs)   # DEBUG: what the policy actually sees
        self._publish_crop_box()         # DEBUG: where the crop region sits
        x, y, z, yaw = self.policy.get_target(obs)
        self.get_logger().info(
            f"Cycle {self.cycle_count + 1}: target=({x:.2f}, {y:.2f}, "
            f"{z:.2f}), yaw={math.degrees(yaw):.1f}deg")
        self._publish_target(x, y, z, yaw)
        self._save_debug(obs, x, y, z, yaw)
        self.moves = self._build_moves(x, y, z, yaw)
        self.move_idx = 0
        self.seq_state = SEQ_SEND

    def _build_moves(self, x, y, z, yaw):
        cfg = self.fsm_cfg
        hover = [x, y, z + cfg.hover_clear]
        approach = [x, y, z + cfg.approach_above]
        drop = list(cfg.drop_position)
        dyaw = cfg.drop_yaw
        moves = [
            Move(Phase.HOVER_UP, "traj", hover, None),
            Move(Phase.ALIGN_YAW, "traj", hover, yaw,
                 velocity=self._yaw_velocity, min_duration=self._yaw_min_duration),
            Move(Phase.DESCEND, "traj", approach, yaw),
            Move(Phase.CLOSE, "grip", gripper=GRIPPER_CLOSE),
            Move(Phase.LIFT_HIGH, "traj", hover, yaw),
            Move(Phase.CARRY_HOME, "traj", drop, dyaw),
        ]
        if self._deposit_scan:
            # At home (yaw already aligned by CARRY_HOME), scan the trailer for the pile top and descend
            # to it (+offset) before releasing. yaw=None => HOLD the current yaw (don't re-command it, so
            # the descend doesn't turn the grapple). The z here is a placeholder; _resolve_deposit fills it.
            moves.append(Move(Phase.LOWER_TO_DROP, "deposit", list(drop), None))
        moves.append(Move(Phase.OPEN, "grip", gripper=GRIPPER_OPEN))
        return moves

    def _send_current_move(self):
        move = self.moves[self.move_idx]
        self._publish_fsm_state(move.phase)
        if move.kind == "deposit":
            # Resolve the descend z from a fresh trailer scan first; once set, send it as a normal traj.
            if not self._resolve_deposit(move):
                return
            if self._deposit_dry_run:
                self.get_logger().info(
                    f"DEPOSIT: dry_run -> NOT descending; target z={move.position[2]:.2f} published "
                    f"(/crane/deposit_target). Skipping the lower, going to OPEN.")
                self._advance_move()
                return
            move.kind = "traj"  # resolved -> behave as a cartesian move for the rest of this cycle
        if move.kind == "traj":
            if self._send_trajectory(move):
                self._seen_executing = False
                self._wait_start = self.get_clock().now()
                self.seq_state = SEQ_WAIT_TRAJ
        else:
            if self._send_grapple(move.gripper):
                self._wait_start = self.get_clock().now()
                self.seq_state = SEQ_WAIT_GRIP

    def _resolve_deposit(self, move) -> bool:
        """At home, settle + wait for a fresh basemast cloud, scan the trailer box for the pile top,
        and set move.position[2] = min(pile_top + offset, home_z). Returns True once the z is set."""
        now = self.get_clock().now()
        if self._deposit_start_time is None:
            self._deposit_start_time = now
            self._deposit_waiting_fresh = False
            return False
        deadline = self._deposit_start_time + Duration(seconds=self._gaze_dwell_s)
        if now < deadline:
            return False
        # same freshness gate as gaze: the registered cloud lags, so require arrival >= settle + latency.
        fresh_after = deadline + Duration(seconds=self._cloud_latency_s)
        if (self.latest_cloud is None or self._cloud_frame is None
                or self._cloud_recv_time is None or self._cloud_recv_time < fresh_after):
            if not self._deposit_waiting_fresh:
                self.get_logger().info("DEPOSIT: at home, waiting for a fresh trailer cloud...")
                self._deposit_waiting_fresh = True
            return False
        self._publish_deposit_box()
        home_z = float(self.fsm_cfg.drop_position[2])
        pile_top = self._scan_trailer_top()   # also publishes the in-box sampled points
        if pile_top is None:
            deposit_z = min(self._deposit_fallback_z, home_z)  # trailer not seen -> lower to a safe fixed z
            self.get_logger().warn("DEPOSIT: no points in trailer box; lowering to fallback z "
                                   f"{deposit_z:.2f} (check the cam sees the trailer / bounds).")
        else:
            deposit_z = min(pile_top + self._deposit_offset, home_z)
            self.get_logger().info(
                f"DEPOSIT: pile_top={pile_top:.2f} + off {self._deposit_offset:.2f} "
                f"-> descend to z={deposit_z:.2f} (home_z={home_z:.2f})")
        tx, ty = float(move.position[0]), float(move.position[1])
        move.position = [tx, ty, float(deposit_z)]
        self._publish_deposit_target(tx, ty, deposit_z)
        self._save_deposit_debug(tx, ty, deposit_z)
        self._deposit_start_time = None
        self._deposit_waiting_fresh = False
        return True

    def _save_deposit_debug(self, x, y, z):
        """Dump the trailer scan (in-box cloud + chosen deposit target + bounds) to
        <debug_save_dir>/deposit_debug_NNN.npz -- same schema as policy_debug so view_policy_debug.py
        and render_run_video.py render it with no extra code (yaw = drop_yaw)."""
        if not self._debug_save_dir:
            return
        try:
            os.makedirs(self._debug_save_dir, exist_ok=True)
            pts = self._last_deposit_pts if self._last_deposit_pts is not None else np.zeros((0, 3), np.float32)
            yaw = float(self.fsm_cfg.drop_yaw)
            fn = os.path.join(self._debug_save_dir, f"deposit_debug_{self.cycle_count + 1:03d}.npz")
            np.savez(fn, points=np.asarray(pts, np.float32),
                     target=np.array([x, y, z, yaw], dtype=np.float32),
                     bounds_min=np.asarray(self._trailer_bounds_min, np.float32),
                     bounds_max=np.asarray(self._trailer_bounds_max, np.float32))
            self.get_logger().info(f"saved deposit debug -> {fn} ({np.asarray(pts).shape[0]} pts)")
            self._append_decision_log(x, y, z, yaw, kind="deposit", npz=os.path.basename(fn))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"deposit debug save failed: {e}")

    def _scan_trailer_top(self):
        """Transform the latest basemast cloud to base_link, crop to the trailer box, and return the
        max z of the in-box points (the pile top), or None if too few points land in the box."""
        try:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, self._cloud_frame, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"DEPOSIT: tf {self._base_frame}<-{self._cloud_frame} unavailable: {e}")
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        pos = torch.tensor([tr.x, tr.y, tr.z], dtype=torch.float32)
        quat = torch.tensor([q.w, q.x, q.y, q.z], dtype=torch.float32)  # wxyz
        pts = torch.from_numpy(np.ascontiguousarray(self.latest_cloud, dtype=np.float32))
        pts_b = transform_points(pts, pos, quat).numpy()
        mn, mx = self._trailer_bounds_min, self._trailer_bounds_max
        m = ((pts_b[:, 0] >= mn[0]) & (pts_b[:, 0] <= mx[0])
             & (pts_b[:, 1] >= mn[1]) & (pts_b[:, 1] <= mx[1])
             & (pts_b[:, 2] >= mn[2]) & (pts_b[:, 2] <= mx[2]))
        n = int(m.sum())
        self._last_deposit_pts = pts_b[m].astype(np.float32)   # for the npz dump
        self._publish_deposit_scan(pts_b[m])   # show exactly what got sampled (base frame)
        if n < self._deposit_min_points:
            self.get_logger().warn(f"DEPOSIT: only {n} pts in trailer box (< {self._deposit_min_points}).")
            return None
        return float(pts_b[m, 2].max())

    def _publish_deposit_box(self):
        """Publish the trailer scan box (base frame) as a translucent orange cube for RViz."""
        bmin = np.asarray(self._trailer_bounds_min, float)
        bmax = np.asarray(self._trailer_bounds_max, float)
        c = 0.5 * (bmin + bmax); size = bmax - bmin
        m = Marker()
        m.header.frame_id = self._base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "trailer_box"; m.id = 0; m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position.x = float(c[0]); m.pose.position.y = float(c[1]); m.pose.position.z = float(c[2])
        m.pose.orientation.w = 1.0
        m.scale.x = float(size[0]); m.scale.y = float(size[1]); m.scale.z = float(size[2])
        m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; m.color.a = 0.15
        self.pub_deposit_box.publish(m)

    def _publish_deposit_scan(self, pts):
        """Publish the in-box sampled points (base frame) as a PointCloud2 for RViz."""
        pts = np.ascontiguousarray(pts, dtype=np.float32).reshape(-1, 3)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.height = 1; msg.width = int(pts.shape[0])
        msg.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(("x", "y", "z"))]
        msg.is_bigendian = False; msg.point_step = 12; msg.row_step = 12 * int(pts.shape[0])
        msg.is_dense = True; msg.data = pts.tobytes()
        self.pub_deposit_scan.publish(msg)

    def _publish_deposit_target(self, x, y, z):
        """Publish the chosen deposit point (base frame) as a green sphere for RViz."""
        m = Marker()
        m.header.frame_id = self._base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "deposit_target"; m.id = 0; m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.25
        m.color.r = 0.1; m.color.g = 1.0; m.color.b = 0.2; m.color.a = 0.9
        self.pub_deposit_target.publish(m)

    def _send_trajectory(self, move) -> bool:
        if not self._joints_received or not self._plc_received:
            return False
        if not (self.plc_state == PLC_STATE_HOLD and self.plc_queue == 0):
            return False

        goal_xyz = list(move.position)
        if move.yaw is None:
            goal_gc = self.current_yaw_joint
            marker_yaw = self.current_ee_yaw
        else:
            goal_gc = self._grapple_yaw_to_joint(move.yaw)
            marker_yaw = move.yaw

        q_target = self.ik.solve(goal_xyz, self.current_arm)
        if q_target is None:
            self.get_logger().warn(
                f"{move.phase.name}: IK failed for goal {np.round(goal_xyz, 3).tolist()}")
            return False
        if self._telescope_static:
            # Hold the telescope fixed (IK already keeps it inactive); guarantees a zero delta.
            q_target[3] = self._telescope_static_value

        # Goal joint vector (slew, boom, stick, telescope, grapplecarrier).
        # Kept here so the convergence gate can compare against the commanded
        # arm + yaw target, not just the IK output.
        self._goal_joints = np.array([
            q_target[0], q_target[1], q_target[2], q_target[3], goal_gc])

        if self.dry_run:
            self.get_logger().info(
                f"{move.phase.name}: dry_run, IK ok for goal "
                f"{np.round(goal_xyz, 3).tolist()}")
            self._publish_ee_command(goal_xyz, marker_yaw)
            return True

        if not self.move_client.service_is_ready():
            if not hasattr(self, "_move_warn"):
                self.get_logger().warn(
                    "/relative_joint_move service not available. "
                    "Is relative_joint_mover running?")
                self._move_warn = True
            return False

        joint_names = ["slew_joint", "boom_joint", "stick_joint",
                       "telescope_joint", "grapplecarrier_joint"]
        delta_positions = [
            float(q_target[0] - self.current_arm[0]),
            float(q_target[1] - self.current_arm[1]),
            float(q_target[2] - self.current_arm[2]),
            float(q_target[3] - self.current_arm[3]),
            float(goal_gc - self.current_yaw_joint),
        ]

        self._seq_id += 1
        req = RelativeJointMove.Request()
        req.sequence_id = self._seq_id
        req.joint_names = joint_names
        req.delta_positions = delta_positions
        vel = move.velocity if move.velocity is not None else self._traj_velocity
        min_dur = move.min_duration if move.min_duration is not None else self._traj_min_duration
        req.velocity = vel
        req.min_duration = max(min_dur,
                               abs(delta_positions[3]) / max(self._telescope_speed, 1e-3))
        self._traj_future = self.move_client.call_async(req)
        self._publish_ee_command(goal_xyz, marker_yaw)
        max_delta = max(abs(d) for d in delta_positions)
        self.get_logger().info(
            f"{move.phase.name}: requested relative move seq={req.sequence_id}, "
            f"max_delta={max_delta:.3f} rad, vel={vel:.3f}")
        return True

    def _send_grapple(self, direction) -> bool:
        if self.dry_run:
            self.get_logger().info(
                f"dry_run grapple {'CLOSE' if direction == GRIPPER_CLOSE else 'OPEN'}")
            self._grapple_future = None
            return True
        if not self.grapple_client.service_is_ready():
            if not hasattr(self, "_grapple_warn"):
                self.get_logger().warn(
                    "request_grapple_move service not available. Is the PLC node running?")
                self._grapple_warn = True
            return False
        self._seq_id += 1
        req = RequestGrappleMove.Request()
        req.grapple_move.sequence_id = self._seq_id
        req.grapple_move.is_executed = False
        req.grapple_move.gripper_move_to = int(direction)
        req.grapple_move.delta_time = int(self._grapple_dt_ms)
        req.grapple_move.delta_angle = int(self._grapple_da_deg)
        self._grapple_future = self.grapple_client.call_async(req)
        self.get_logger().info(
            f"Grapple {'CLOSE' if direction == GRIPPER_CLOSE else 'OPEN'} requested")
        return True

    def _check_traj_done(self):
        if not self._seen_executing and (
                self.plc_state == PLC_STATE_EXECUTE or self.plc_queue > 0):
            self._seen_executing = True
        plc_idle = (self._seen_executing
                    and self.plc_state in (PLC_STATE_HOLD, PLC_STATE_END)
                    and self.plc_queue == 0)
        converged = self._converged_to_goal()
        done = plc_idle and converged
        if self.dry_run:
            done = True
        if done:
            self._log_reach(self.moves[self.move_idx].phase)
            self._advance_move()
        elif self._waited_longer_than(self._traj_timeout):
            self._log_timeout(self.moves[self.move_idx], plc_idle)
            self._advance_move()

    def _converged_to_goal(self) -> bool:
        if self._goal_joints is None or not self._joints_received:
            return True
        arm_diff = np.abs(np.asarray(self.current_arm) - self._goal_joints[:4])
        gc_diff = abs(float(self.current_yaw_joint) - float(self._goal_joints[4]))
        rev_ok = bool((arm_diff[[0, 1, 2]] < self._arm_tol).all())
        tele_ok = bool(arm_diff[3] < self._telescope_tol)
        yaw_ok = bool(gc_diff < self._yaw_tol)
        return rev_ok and tele_ok and yaw_ok

    def _log_reach(self, phase):
        if self._goal_joints is None or not self._joints_received:
            return
        diff = np.asarray(self.current_arm) - self._goal_joints[:4]
        gc_diff = float(self.current_yaw_joint) - float(self._goal_joints[4])
        arm_err = float(np.max(np.abs(diff)))
        self.get_logger().info(
            f"{phase.name}: reached "
            f"(slew={diff[0]:+.3f} boom={diff[1]:+.3f} stick={diff[2]:+.3f} "
            f"tele={diff[3]:+.3f} yaw={gc_diff:+.3f}; arm_max={arm_err:.3f})")

    def _log_timeout(self, move, plc_idle):
        """On a phase timeout, log WHICH joint is over-tolerance + grapple-vs-goal position."""
        info = []
        if self._goal_joints is not None and self._joints_received:
            d = np.asarray(self.current_arm) - self._goal_joints[:4]
            gc = float(self.current_yaw_joint) - float(self._goal_joints[4])
            tols = [self._arm_tol, self._arm_tol, self._arm_tol, self._telescope_tol]
            names = ["slew", "boom", "stick", "tele"]
            over = [f"{names[i]}={d[i]:+.3f}(>{tols[i]})" for i in range(4) if abs(d[i]) >= tols[i]]
            if abs(gc) >= self._yaw_tol:
                over.append(f"yaw={gc:+.3f}(>{self._yaw_tol})")
            info.append("over-tol: " + (", ".join(over) if over else "none (PLC not idle)"))
        if move.position is not None and self.current_ee_pos is not None:
            gp = np.asarray(move.position, float); cp = np.asarray(self.current_ee_pos, float)
            info.append(f"grapple={cp.round(2).tolist()} goal={gp.round(2).tolist()} err={np.linalg.norm(gp - cp):.3f}m")
        self.get_logger().warn(
            f"{move.phase.name}: TIMEOUT (plc_idle={plc_idle}); {'; '.join(info)}; advancing anyway")

    def _check_grip_done(self):
        done = self.dry_run or (self._grapple_future is not None
                                and self._grapple_future.done())
        if done:
            self._advance_move()
        elif self._waited_longer_than(self._grip_timeout):
            self.get_logger().warn(
                f"{self.moves[self.move_idx].phase.name}: grapple timeout, advancing")
            self._advance_move()

    def _waited_longer_than(self, timeout_s) -> bool:
        if self._wait_start is None:
            return False
        elapsed = (self.get_clock().now() - self._wait_start).nanoseconds / 1e9
        return elapsed > timeout_s

    def _advance_move(self):
        self.move_idx += 1
        self._grapple_future = None
        if self.move_idx >= len(self.moves):
            self.cycle_count += 1
            self.get_logger().info(f"Cycle {self.cycle_count} complete")
            self.seq_state = SEQ_IDLE
        else:
            self.seq_state = SEQ_SEND

    # Publisher helpers

    def _publish_fsm_state(self, phase):
        msg = String()
        msg.data = f"{phase.name} (cycle {self.cycle_count})"
        self.pub_fsm_state.publish(msg)
        self._append_phase_log(phase.name)

    def _append_phase_log(self, phase_name):
        """Append a phase transition (name + current video_frame) to <run_dir>/phases.jsonl so the
        review video can overlay which FSM phase is active at each frame."""
        if not self._debug_save_dir:
            return
        try:
            rec = {"phase": phase_name,
                   "video_frame": (self._video_frames if self._record_video else None),
                   "t_wall": self.get_clock().now().nanoseconds * 1e-9}
            with open(os.path.join(self._debug_save_dir, "phases.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"phase log append failed: {e}")

    def _broadcast_marker_tf(self, child_frame, x, y, z, yaw):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._base_frame
        t.child_frame_id = child_frame
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = float(z)
        t.transform.rotation.w = math.cos(yaw / 2)
        t.transform.rotation.z = math.sin(yaw / 2)
        self._tf_broadcaster.sendTransform(t)

    def _write_marker_state(self, key, x, y, z):
        path = "/tmp/crane_marker_state.json"
        try:
            state = {}
            if os.path.exists(path):
                with open(path) as f:
                    state = json.load(f)
        except Exception:
            state = {}
        state[key] = [float(x), float(y), float(z)]
        fd, tmp = tempfile.mkstemp(prefix=".crane_marker_", dir="/tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def _publish_ee_command(self, pos, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_ee_cmd.publish(msg)
        self._broadcast_marker_tf("ee_command_marker", pos[0], pos[1], pos[2], yaw)
        self._write_marker_state("ee", pos[0], pos[1], pos[2])

    def _publish_input_cloud(self, obs):
        """Publish the cropped+FPS'd policy input cloud (base frame) for RViz debugging."""
        try:
            pts = obs.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
        except Exception:
            pts = np.asarray(obs, dtype=np.float32).reshape(-1, 3)
        pts = pts[np.abs(pts).sum(1) > 1e-6]   # drop zero-padding
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.height = 1
        msg.width = int(pts.shape[0])
        msg.fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate(("x", "y", "z"))]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * int(pts.shape[0])
        msg.is_dense = True
        msg.data = pts.tobytes()
        self.pub_policy_input.publish(msg)

    def _save_debug(self, obs, x, y, z, yaw):
        """Dump the cropped policy-input cloud + target to npz for offline Open3D viewing."""
        if not self._debug_save_dir:
            return
        try:
            os.makedirs(self._debug_save_dir, exist_ok=True)
            pts = obs.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
            pts = pts[np.abs(pts).sum(1) > 1e-6]
            fn = os.path.join(self._debug_save_dir,
                              f"policy_debug_{self.cycle_count + 1:03d}.npz")
            np.savez(fn, points=pts,
                     target=np.array([x, y, z, yaw], dtype=np.float32),
                     bounds_min=np.asarray(self._bounds_min, np.float32),
                     bounds_max=np.asarray(self._bounds_max, np.float32))
            self.get_logger().info(f"saved policy debug -> {fn} ({pts.shape[0]} pts)")
            self._append_decision_log(x, y, z, yaw, kind="gaze", npz=os.path.basename(fn))
            if self._debug_save_jpg:
                self._render_iso_jpg(pts, x, y, z, yaw, fn[:-4] + ".jpg")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"debug save failed: {e}")

    def _append_decision_log(self, x, y, z, yaw, kind="gaze", npz=None):
        """Append one decision record to <run_dir>/decisions.jsonl for exact video<->decision sync.
        video_frame is the count of basemast frames written so far, i.e. the mp4 frame index this
        decision becomes active on (null when not recording video). kind = "gaze" | "deposit"; npz =
        the panel dump to show from this frame -> the review video interleaves gaze + deposit panels."""
        try:
            rec = {
                "cycle": self.cycle_count + 1,
                "kind": kind,
                "npz": npz,
                "t_wall": self.get_clock().now().nanoseconds * 1e-9,
                "video_frame": (self._video_frames if self._record_video else None),
                "target": [float(x), float(y), float(z), float(yaw)],
            }
            with open(os.path.join(self._debug_save_dir, "decisions.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"decision log append failed: {e}")

    def _render_iso_jpg(self, pts, x, y, z, yaw, path):
        """Render the policy-input cloud + chosen target as an isometric-view JPG (matplotlib Agg)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
        except Exception as e:  # noqa: BLE001
            if not hasattr(self, "_mpl_warn"):
                self.get_logger().warn(f"iso-view JPG disabled (matplotlib unavailable: {e})")
                self._mpl_warn = True
            return
        try:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            if pts.shape[0]:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=pts[:, 2], cmap="viridis")
            ax.scatter([x], [y], [z], c="red", marker="*", s=320, edgecolor="k", zorder=5)
            ax.plot([x, x + 0.6 * math.cos(yaw)], [y, y + 0.6 * math.sin(yaw)], [z, z],
                    "r-", lw=2)
            ax.view_init(elev=28, azim=-60)   # isometric-ish
            try:
                ax.set_box_aspect((1, 1, 0.5))
            except Exception:  # noqa: BLE001
                pass
            ax.set_xlabel("base X"); ax.set_ylabel("base Y"); ax.set_zlabel("base Z")
            ax.set_title(f"cycle {self.cycle_count + 1}: target "
                         f"({x:.2f}, {y:.2f}, {z:.2f}) yaw {math.degrees(yaw):.0f} deg")
            try:
                fig.savefig(path, dpi=90, bbox_inches="tight")   # JPG if Pillow is present
            except ValueError:
                path = path[:-4] + ".png"                        # else fall back to PNG (always supported)
                fig.savefig(path, dpi=90, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"iso-view render failed: {e}")

    def _decode_rgb(self, msg):
        """Decode a sensor_msgs/Image (rgb8/bgr8/rgba8/bgra8) to an HxWx3 uint8 RGB array."""
        enc = msg.encoding
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("rgb8", "bgr8"):
                img = buf.reshape(msg.height, msg.width, 3)
                if enc == "bgr8":
                    img = img[:, :, ::-1]
            elif enc in ("rgba8", "bgra8"):
                img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
                if enc == "bgra8":
                    img = img[:, :, ::-1]
            else:
                if not hasattr(self, "_vid_enc_warn"):
                    self.get_logger().warn(f"unsupported video encoding {enc}; not recording")
                    self._vid_enc_warn = True
                return None
            return np.ascontiguousarray(img)
        except Exception:  # noqa: BLE001
            return None

    def _decode_compressed(self, msg):
        """Decode a sensor_msgs/CompressedImage (jpeg/png) to an HxWx3 uint8 RGB array.
        cv2/compressed_image_transport jpegs decode to correct RGB without a swap even when the
        format says bgr8; video_swap_rb forces a swap for the rare camera that stores it the other way."""
        try:
            import io
            data = bytes(msg.data)
            try:
                from PIL import Image as _PILImage
                img = np.asarray(_PILImage.open(io.BytesIO(data)).convert("RGB"))
            except Exception:
                import imageio.v2 as _iio
                img = np.asarray(_iio.imread(data))
                if img.ndim == 2:
                    img = np.repeat(img[:, :, None], 3, axis=2)
                img = img[:, :, :3]
            if self._video_swap_rb:
                img = img[:, :, ::-1]
            return np.ascontiguousarray(img)
        except Exception as e:  # noqa: BLE001
            if not hasattr(self, "_vid_dec_warn"):
                self.get_logger().warn(f"compressed frame decode failed ({e}); not recording")
                self._vid_dec_warn = True
            return None

    def _video_cb(self, msg):
        """Append basemast RGB frames to the run's mp4, rate-limited to video_fps."""
        if not self._record_video:
            return
        now = self.get_clock().now()
        if self._last_video_stamp is not None:
            dt = (now - self._last_video_stamp).nanoseconds * 1e-9
            if dt < 1.0 / max(self._video_fps, 1e-3):
                return
        frame = self._decode_compressed(msg) if self._video_compressed else self._decode_rgb(msg)
        if frame is None:
            return
        if self._video_writer is None:
            try:
                import imageio
                os.makedirs(self._debug_save_dir, exist_ok=True)
                path = os.path.join(self._debug_save_dir, "basemast_run.mp4")
                self._video_writer = imageio.get_writer(
                    path, fps=self._video_fps, macro_block_size=None)
                self.get_logger().info(f"video writer opened -> {path}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(
                    f"video recording unavailable ({e}); disabling. Record with "
                    f"'ros2 bag record -o {self._debug_save_dir}/run_bag {self._video_topic}' instead.")
                self._record_video = False
                return
        try:
            self._video_writer.append_data(frame)
            self._video_frames += 1
            self._last_video_stamp = now
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"video frame write failed: {e}; disabling recording")
            self._record_video = False

    def close_video(self):
        """Flush and close the video writer (call on shutdown)."""
        if self._video_writer is not None:
            try:
                self._video_writer.close()
                self.get_logger().info(
                    f"video saved: {self._video_frames} frames -> "
                    f"{self._debug_save_dir}/basemast_run.mp4")
            except Exception:  # noqa: BLE001
                pass
            self._video_writer = None

    def render_review_video(self):
        """At shutdown, stitch the run into <debug_save_dir>/review.mp4 via render_run_video.py
        (per-cycle policy panels over the basemast stream). Runs when record_video is on.
        Best-effort; logs and moves on."""
        if not (self._record_video and self._debug_save_dir):
            return
        script = self._render_video_script
        if not os.path.isfile(script):
            self.get_logger().warn(f"review_video: script not found at {script}; skipping.")
            return
        if not glob.glob(os.path.join(self._debug_save_dir, "policy_debug_*.npz")):
            self.get_logger().warn("review_video: no policy_debug_*.npz in run dir; skipping.")
            return
        self.get_logger().info(
            f"review_video: stitching review.mp4 from {self._debug_save_dir} ... (Ctrl-C to skip)")
        try:
            r = subprocess.run(["python3", script, self._debug_save_dir],
                               capture_output=True, text=True, timeout=1800)
            if r.returncode == 0:
                self.get_logger().info(f"review_video: {self._debug_save_dir}/review.mp4 written.")
            else:
                self.get_logger().warn(f"review_video: render_run_video failed:\n{r.stderr[-500:]}")
        except KeyboardInterrupt:
            self.get_logger().warn("review_video: interrupted; review.mp4 may be incomplete. Re-stitch "
                                   "later with render_run_video.py on the run dir.")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"review_video: {e}")

    def _publish_crop_box(self):
        """Publish the action-box crop region (base frame) as a translucent cube for RViz."""
        bmin = np.asarray(self._bounds_min, float)
        bmax = np.asarray(self._bounds_max, float)
        c = 0.5 * (bmin + bmax)
        size = bmax - bmin
        m = Marker()
        m.header.frame_id = self._base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "crop_box"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = float(c[0]); m.pose.position.y = float(c[1]); m.pose.position.z = float(c[2])
        m.pose.orientation.w = 1.0
        m.scale.x = float(size[0]); m.scale.y = float(size[1]); m.scale.z = float(size[2])
        m.color.r = 0.1; m.color.g = 0.8; m.color.b = 1.0; m.color.a = 0.2
        self.pub_crop_box.publish(m)

    def _publish_target(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_policy_target.publish(msg)
        self._broadcast_marker_tf("policy_target_marker", x, y, z, yaw)
        self._write_marker_state("policy", x, y, z)


def main():
    parser = argparse.ArgumentParser(description="Crane policy node")
    parser.add_argument("--policy_type", default="heuristic")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dry_run", action="store_true")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CranePolicyNode(
        policy_type_override=args.policy_type,
        checkpoint_override=args.checkpoint,
        dry_run_override=args.dry_run,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_video()
        node.render_review_video()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
