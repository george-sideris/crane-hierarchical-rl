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

    Optional external targeting (pile analyzer service from emt_acm_mh_planning;
    needs log_loading_interfaces built and the pile_analyzer_server running):
        ... -p target_source:=pile_analyzer
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
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField, CompressedImage
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

from fpi_crane_msgs.msg import PlcStatus
from fpi_crane_msgs.action import MeasureStability
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
SEQ_WAIT_STABILITY = "wait_stability"
SEQ_WAIT_ANALYSIS = "wait_analysis"   # pile-analyzer service call in flight

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


def _quat_rotate(qw, qx, qy, qz, v):
    """Rotate 3-vector v by quaternion (w,x,y,z); returns np.float64 (3,)."""
    q = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(q, v)
    return v + qw * t + np.cross(q, t)


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
        # Heuristic dig below the observed surface. 0.30 = real-tuned; 0.056 = the sim
        # expert's log-center convention (for sim-convention-on-real experiments).
        self.declare_parameter("heuristic_dig", 0.30)
        # scoring policies: support-gate strength. DEFAULT 0.0 = gate OFF (normal mode);
        # opt in with 0.25 (recommended for scoring_v1, which is noise-brittle ungated).
        # The candidate mask (action-box reachability) is separate and always on.
        self.declare_parameter("support_frac", 0.0)
        self.declare_parameter("checkpoint_path", checkpoint_override or "")
        # Where the grasp target comes from each cycle:
        #   "policy"        onboard point-cloud policy (default)
        #   "pile_analyzer" external log_pickup_analysis service (emt_acm_mh_planning
        #                   pile analyzer); any failure falls back to the onboard
        #                   policy for that cycle so the FSM never stalls
        self.declare_parameter("target_source", "policy")
        self.declare_parameter("analyzer_service", "/log_pickup_analysis")
        self.declare_parameter("analyzer_timeout", 15.0)
        # raw (uncompressed) rect RGB matching the analyzer server's camera_info_topic
        self.declare_parameter("analyzer_image_topic", "/zed_0/zed_node/rgb/color/rect/image")
        # attack_pose -> grasp-target reconciliation knobs
        self.declare_parameter("analyzer_z_offset", 0.0)
        self.declare_parameter("analyzer_yaw_offset", 0.0)
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("depth_range_min", 1.0)
        self.declare_parameter("depth_range_max", 10.0)
        self.declare_parameter("cossin", True)
        self.declare_parameter("device",
                               "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("tick_rate", 10.0)
        self.declare_parameter("dry_run", dry_run_override)
        self.declare_parameter("max_cycles", 50)
        # Resume a stopped run: start numbering at this cycle (files/records continue
        # as policy_debug_<start_cycle>... in the same debug_save_dir; pass the old
        # run dir explicitly instead of "auto"). start_bin picks the geometric bin
        # the first resumed deposit targets (-1 = start of the fill order).
        self.declare_parameter("start_cycle", 1)
        self.declare_parameter("start_bin", -1)
        # Crop box = the sim ACTION bounds (must match training: crane_rl_env_gaze.py action box).
        # 2026-06-28 calibrated rack: center (-4.364, 1.816), HALF_X=1.0 HALF_Y=3.5, Z [-1.30, 0.10].
        # (prev/stale old-rack: min [-5.0,-0.75,-1.373] max [-3.0,4.59,-0.373])
        self.declare_parameter("bounds_min", [-5.364, -1.684, -1.30])
        self.declare_parameter("bounds_max", [-3.364, 5.316, 0.10])
        # Widen the OBSERVATION crop by this much in x/y and BELOW in z, leaving the action box
        # (target decode) untouched. Must equal the --crop_margin the policy was trained with:
        # a policy trained WITHOUT structure in its cloud must not be shown structure at deploy,
        # and one trained WITH it must not have it cropped away. 0.0 = legacy tight crop.
        self.declare_parameter("crop_margin", 0.0)
        # The front-left rack pole clips the corner of the crop box (x > pole_x_gt AND y < pole_y_lt,
        # base_link). Sim training clouds never contain a pole there, and its ~20 points dominate the
        # PointNet max-pool and drag the target ~2.5 m off the pile top (verified offline 2026-07-13).
        # Always filtered out of the policy input; set pole_x_gt >= bounds_max x to disable.
        self.declare_parameter("pole_x_gt", -3.6)
        self.declare_parameter("pole_y_lt", -1.35)
        # front-RIGHT pole clipping the other box corner. After the 2026-07-20 zed_0 recal it drifts
        # around x=-3.6..-3.8, y=4.8..5.0 (recal + slew move it), and as a tall thin column it becomes
        # the cloud z-max and rails the target to the empty far-right corner. Box covers the whole
        # front-right corner (x>-3.9 AND y>4.3); the graspable interior pile sits at x<-3.9 so it's
        # untouched (verified: a good grasp at x=-4.38,y=4.44 stays put while the pole is stripped).
        self.declare_parameter("pole2_x_gt", -3.9)
        # 4.3 -> 4.6 (2026-07-29): the old zone started ~0.5 m below the pole and ate a visible
        # rectangle of real pile at the crane; 4.6 keeps ~0.2 m of slew-wander margin below the
        # pole (which moved with the rack, see rack_y_shift).
        self.declare_parameter("pole2_y_gt", 4.6)
        # BACK-left rack pole (far x side escapes the front filters' x> conditions; measured
        # 2026-07-29 at x[-5.36,-5.26], reaching the crop ceiling). Corner box: x < AND y <.
        self.declare_parameter("pole3_x_lt", -5.15)
        self.declare_parameter("pole3_y_lt", -1.45)
        # Physical rack drift along base y vs the training-era action box (current minus training).
        # Crop and pole zones follow the rack; the policy still operates in training-era coords.
        # Default -0.55: re-measured 2026-07-31 (supersedes the -0.45 cross-correlation estimate
        # of 2026-07-29, which left a ~0.1 m residual; the heuristic is translation-covariant so it
        # was unaffected, but LEARNED policies are not - a 0.45 m error moved their aim 1.2-1.6 m,
        # so 0.1 m is worth ~0.3 m of aim error). Re-measure whenever the rack is moved or the
        # camera is recalibrated; 0.0 = rack at the training-era position.
        self.declare_parameter("rack_y_shift", -0.4)
        # Platform bed floor: never EXECUTE a grasp z below bed_z + bed_margin, for EVERY
        # policy type (uniform envelope; replaces the heuristic-internal p5 clamp, which was
        # occlusion-biased and asymmetric). MEASURED 2026-07-30 from bare-rack clouds: the bed
        # surface sits at ~-1.29, i.e. the box bottom (-1.30) has NO built-in clearance.
        # Margin 0.0 (2026-07-31): bottom-layer log centers sit at bed+0.056 ~ -1.234, so any
        # margin above 0.066 puts the last layer out of reach for every controller. The 0.10
        # jam-safety margin did exactly that (floor -1.20, 3.4 cm above the centers) and the
        # heuristic missed every bottom-layer pick, so the margin is gone in sim and real alike
        # and bed_z is the hard floor. Raw pre-clamp z is logged per decision ("z_raw").
        self.declare_parameter("bed_z", -1.30)
        self.declare_parameter("bed_margin", 0.0)
        # radius-outlier removal: a lone isolated point wins the PointNet max-pool and rails the
        # target to it (front pole tip that drifts out of the boxes after a recal / with slew, or a
        # sensor speck). Drop any sampled point with < ror_min_neighbors within ror_radius (m).
        # Isolation-keyed so it leaves dense pile tops untouched. Set ror_radius <= 0 to disable.
        self.declare_parameter("ror_radius", 0.2)
        self.declare_parameter("ror_min_neighbors", 2)
        self.declare_parameter("hover_clear", 2.5)
        self.declare_parameter("approach_above", 0.6)
        self.declare_parameter("drop_position", [0.0, 3.0, 3.0])
        # Anti-droop carry: CARRY_HOME is one big slew+boom+stick move and the PLC interpolates it in
        # JOINT space, so the tip arcs and dips (droops) mid-swing between the two high endpoints.
        # Break the carry into carry_waypoints intermediate Cartesian poses along the hover->drop line,
        # all pinned at carry_height, so the joint-space interpolation between closely spaced waypoints
        # hugs a constant-height path. 0 = old single-move carry. Also shrinks each move so the PLC is
        # less likely to segment/abort it (the "fell short" case).
        self.declare_parameter("carry_waypoints", 1)
        # Transit height (base_link z) held across the carry waypoints. <=0 => auto = max(hover_z, drop_z)
        # so we never dip below either endpoint.
        self.declare_parameter("carry_height", -1.0)
        # Base-frame grapple yaw over the trailer: 0.0 = parallel to the bed. (Was pi/2, which only
        # looked parallel because the yaw goal ignored the slew travel of CARRY_HOME; that is now
        # compensated in _grapple_yaw_to_joint, so the true value applies.)
        self.declare_parameter("drop_yaw", 0.0)
        # Deposit scan: after CARRY_HOME (at the home x,y + yaw aligned), scan the basemast cloud for the
        # highest log in the trailer box and LOWER to that + offset before releasing (descend-only, capped
        # at the home z). Off by default -> old behavior (drop straight from drop_position). Bounds are in
        # base_link frame and match the sim --trailer_box_* viz.
        self.declare_parameter("deposit_scan", False)
        self.declare_parameter("deposit_offset", 0.8)         # m above the detected pile top
        self.declare_parameter("deposit_min_points", 1)        # min in-box points to trust the scan
        # Optional held-bundle exclusion: drop points above (grapple z - this clearance). 0 = off,
        # scan the fixed box only (up to trailer_bounds_max z). The bundle bottom and pile top
        # overlap in z, so this cannot cleanly separate them; a 2.0 default blanked real scans.
        self.declare_parameter("deposit_held_clearance", 0.0)
        # pile top = k-th highest in-box z (not raw max): a single dangling/noise point can't set it.
        self.declare_parameter("deposit_top_k", 5)
        # z floor -0.6: the trailer BED sits at z -0.4..0 in base_link (measured from the
        # 2026-07-15 bag), so a 0.0 floor saw nothing until the pile grew above base height
        # (empty trailer -> 0 pts and fallback). -0.6 sees bed + pile; ground (-1.29) stays out.
        self.declare_parameter("trailer_bounds_min", [-0.7, 0.2, -0.6])
        # z ceiling 1.5 (was 2.0) so the scan doesn't catch the held logs / trailer rim up near the grapple
        # at the home pose (that inflated the pile top and kept the deposit high).
        self.declare_parameter("trailer_bounds_max", [0.7, 5.302, 1.5])
        # If the scan finds nothing in the box, still lower to this fixed z (base_link) before releasing
        # instead of dropping from full home height. Clamped to <= home z (descend-only). Also the
        # blind-drop height for a bin the cam can't yet see (e.g. the near bin 0 while empty); once the
        # pile rises into view the per-bin adaptive z takes over.
        self.declare_parameter("deposit_fallback_z", 1.5)
        # bins mode, FIRST drop into a still-empty bin (no cached bundle-free height yet). Descend to
        # this fixed z (base_link) instead of scanning -- a live scan there would catch the HELD bundle
        # hanging into the box and read a bogus-high pile top. Set near the trailer bed so the first log
        # lands low; every later drop in that bin uses its cached post-release height + deposit_offset.
        self.declare_parameter("deposit_bin_empty_z", 0.8)
        # If true, run the scan + publish the deposit target/markers but SKIP the actual descend (watch
        # where it WOULD lower without moving the arm down). The rest of the cycle still runs.
        self.declare_parameter("deposit_dry_run", False)
        # Bin-fill deposition. "bins" (default) = split the trailer into compartments (between the
        # trailer poles, along base_link y) and fill them one at a time: each bundle drops
        # perpendicular (deposit_bin_yaw) in the middle of the current bin, with adaptive z from a
        # scan of THAT bin only; forces deposit_scan on. "pile" = single growing pile at
        # drop_position.
        self.declare_parameter("deposit_mode", "bins")
        # y-edges of the bins in base_link, ascending. N edges -> N-1 bins; bin i spans
        # [edge[i], edge[i+1]] with its center at the midpoint. Default = the measured trailer poles:
        # 4 bins, poles ~50 in (1.276 m) apart, tiling the trailer box (linspace(0.2, 5.302, 5)); the
        # 1.48 pole matches the wall seen in the deposit scans. Fill order is near-to-far (lowest y).
        self.declare_parameter("deposit_bin_edges", [0.2, 1.476, 2.751, 4.027, 5.302])
        # A bin is "full" once its scanned pile top reaches this z (base_link); then advance to the
        # next bin. When the LAST bin fills, deposition stops (no new pick cycles start). Set this near
        # the trailer pole/stake top so logs fill to the walls, not over them (empty-trailer structure
        # tops out ~0.35 base_link; 0.5 leaves a little headroom for now, tune to the poles).
        self.declare_parameter("deposit_bin_full_z", 0.7)
        # Grapple yaw over a bin (base_link). pi/2 lays the bundle ACROSS the bin (perpendicular to
        # the trailer length) so logs stack neatly. Overrides drop_yaw while in bins mode.
        self.declare_parameter("deposit_bin_yaw", math.pi / 2.0)
        # Inset (m) shrinking each bin's z-scan y-range away from its pole edges, so the boundary
        # poles are not sampled as the bin's pile top (same x/z limits as the pile deposition apply).
        self.declare_parameter("deposit_bin_scan_margin", 0.15)
        # Fill order: "far_to_near" (default) starts at the highest-y bin, "near_to_far" at the
        # lowest-y bin (bin 0, in front of the first divider). far_to_near avoids starting on the
        # camera-blind front bin (its first drops would fall back to deposit_fallback_z until logs
        # stack up). Either way, when the last bin in the sequence fills, deposition stops.
        self.declare_parameter("deposit_bin_fill_order", "far_to_near")
        # Skip the N bins nearest the crane (lowest y = lowest indices; bin 0 is the closest, in front
        # of the first divider). Those sit too close to load cleanly, so drop them from the fill
        # rotation entirely. Default 1 -> never fill bin 0. 0 -> fill every bin. The skipped bins are
        # still drawn in the overlay, they just never become the active target.
        self.declare_parameter("deposit_bin_skip_nearest", 1)
        # How the bins fill up:
        #   "sequential" (default) -- one bundle per bin, round-robin over the fill order, building the
        #      bins up in even layers (bin A, bin B, ... back to bin A). Each drop still gets its own
        #      adaptive z from a fresh scan of that bin. A bin whose scanned top reaches deposit_bin_full_z
        #      is skipped in the rotation; when every bin is full, deposition stops.
        #   "column" -- keep dropping into ONE bin until it reaches deposit_bin_full_z, then move to the
        #      next (the original behavior).
        self.declare_parameter("deposit_bin_fill_mode", "sequential")
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
        # Stitch review.mp4 at shutdown (opt-in). Off, the run just keeps its
        # basemast recording + jsonl logs and the review is rendered on demand:
        #     python3 calibration/render_run_video.py <run_dir> [--speed N]
        self.declare_parameter("stitch_review", False)
        # Playback speedup passed to the shutdown review stitch (--speed N: keep
        # every Nth frame). Segment-aware stitching covers ALL of a resumed run's
        # footage, which at 1x can take tens of minutes for a long multi-session
        # day; >1 makes shutdown fast at the cost of a faster-playing review.
        self.declare_parameter("review_video_speed", 1.0)
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
        # traj_timeout is only the LAST resort (PLC hangs mid-move / never reports idle). The normal
        # miss path is faster: once the PLC has executed and returned to idle but the arm is still
        # off-tolerance, the move is over -- it went as far as it will. Wait only traj_settle_s for
        # joint_states to settle, then RE-SEND the move (recomputed as the remaining delta from the
        # current joints) up to traj_max_resends times; only after that give up and advance.
        self.declare_parameter("traj_settle_s", 1.5)
        # How many times to re-send a move that the PLC executed but stopped short of goal tolerance
        # (e.g. the PLC segmented/aborted a large combined slew+lift). 0 = advance immediately on a
        # short-fall (old behavior). Each re-send is the remaining delta, so it converges toward goal.
        self.declare_parameter("traj_max_resends", 2)
        self.declare_parameter("telescope_speed", 0.04)   # m/s; lower => more time for the telescope
        self.declare_parameter("grip_timeout", 15.0)
        # CLOSE settle before lifting. The close is open-loop: the PLC keeps driving the jaws for
        # grapple_delta_time_ms after the command is sent, and the node never cancels it. Hold the
        # CLOSE phase this long for an initial bite, then start LIFT -- the grapple keeps closing
        # through the ascent (if the crane runs grapple + arm together). 2 s = short settle so it
        # lifts while still closing, instead of sitting out the full close.
        self.declare_parameter("grip_close_dwell_s", 2.0)
        self.declare_parameter("grip_open_dwell_s", 3.0)
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
        self.declare_parameter("measure_stability_enabled", False)
        # Spawn measure_stability_action_node ourselves when no server is discovered
        # at startup (child process, terminated on shutdown). Disable to manage the
        # server externally (e.g. custom IMU topic / frame params).
        self.declare_parameter("measure_stability_autostart", True)
        self.declare_parameter("measure_stability_action_name", "measure_stability")
        self.declare_parameter("measure_stability_timeout_sec", 10.0)
        self.declare_parameter("measure_stability_required_window_sec", 1.0)
        self.declare_parameter("measure_stability_required_dwell_sec", 0.5)
        # Cadence gates sized for the observed real feed (~10-15 Hz effective after
        # TF sync, not the nominal 200 Hz IMU rate): a 1 s window needs >= 6 samples
        # and tolerates gaps up to 0.6 s before the settle dwell resets.
        self.declare_parameter("measure_stability_min_samples", 6)
        self.declare_parameter("measure_stability_max_sample_gap_sec", 0.6)
        self.declare_parameter(
            "measure_stability_stability_variance_threshold_rad2", 1.0e-4)
        self.declare_parameter(
            "measure_stability_gravity_variance_threshold_rad2", 1.0e-5)
        self.declare_parameter(
            "measure_stability_grapple_variance_threshold_rad2", 1.0e-4)
        self.declare_parameter("measure_stability_reward_exponent", 4.0)
        self.declare_parameter("measure_stability_failure_policy", "advance")
        self.declare_parameter("measure_stability_goal_response_timeout_sec", 2.0)

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
        self._bounds_min = bounds_min   # training-era box: policy input coords + target decode
        self._bounds_max = bounds_max
        # rack_y_shift: physical rack drift along base y vs the training-era box (current minus
        # training; measured by cross-correlating a current gaze cloud against a July run).
        # The nets are NOT translation robust (verified offline 2026-07-29: a -0.45 m input shift
        # moves targets 1.2-1.6 m), so the box must never simply move with the rack. Instead the
        # CROP follows the physical rack (bounds + shift, pole zones + shift), the cropped points
        # are shifted BACK into training-era coords for the policy, and the decoded target is
        # shifted forward again for execution (see _policy_target).
        self._rack_y_shift = float(self.get_parameter("rack_y_shift").value)
        self._bed_z = float(self.get_parameter("bed_z").value)
        self._bed_margin = float(self.get_parameter("bed_margin").value)
        self._crop_margin = float(self.get_parameter("crop_margin").value)
        self._crop_min = bounds_min.copy()
        self._crop_max = bounds_max.copy()
        self._crop_min[1] += self._rack_y_shift
        self._crop_max[1] += self._rack_y_shift
        # observation-only widening (x/y both sides, z downward); action box stays bounds_min/max
        self._crop_min[0] -= self._crop_margin
        self._crop_max[0] += self._crop_margin
        self._crop_min[1] -= self._crop_margin
        self._crop_max[1] += self._crop_margin
        self._crop_min[2] -= self._crop_margin
        self._pole_x_gt = float(self.get_parameter("pole_x_gt").value)
        self._pole_y_lt = float(self.get_parameter("pole_y_lt").value) + self._rack_y_shift
        self._pole2_x_gt = float(self.get_parameter("pole2_x_gt").value)
        self._pole2_y_gt = float(self.get_parameter("pole2_y_gt").value) + self._rack_y_shift
        self._pole3_x_lt = float(self.get_parameter("pole3_x_lt").value)
        self._pole3_y_lt = float(self.get_parameter("pole3_y_lt").value) + self._rack_y_shift
        _ror_r = float(self.get_parameter("ror_radius").value)
        self._ror_radius = _ror_r if _ror_r > 0 else None   # <=0 disables
        self._ror_min_neighbors = int(self.get_parameter("ror_min_neighbors").value)
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
        self._deposit_held_clearance = float(self.get_parameter("deposit_held_clearance").value)
        self._deposit_top_k = max(1, int(self.get_parameter("deposit_top_k").value))
        self._cached_pile_top = None   # from the post-release scan; used by the NEXT cycle's deposit
        self._prescan_done = False     # pre-first-cycle trailer scan (home -> scan -> then gaze)
        self._prescan_active = False
        self._move_retries = 0         # re-sends of the current move after a PLC send failure
        self._fellshort_retries = 0    # re-sends after the PLC executed but stopped short of goal
        self._deposit_dry_run = bool(self.get_parameter("deposit_dry_run").value)
        self._trailer_bounds_min = np.array(self.get_parameter("trailer_bounds_min").value, dtype=np.float32)
        self._trailer_bounds_max = np.array(self.get_parameter("trailer_bounds_max").value, dtype=np.float32)
        self._deposit_fallback_z = float(self.get_parameter("deposit_fallback_z").value)
        self._deposit_bin_empty_z = float(self.get_parameter("deposit_bin_empty_z").value)
        self._cloud_latency_s = float(self.get_parameter("cloud_capture_latency_s").value)
        self._deposit_start_time = None
        self._deposit_waiting_fresh = False
        self._last_deposit_pts = None   # in-box trailer points from the latest scan (for the npz dump)
        # Bin-fill deposition state.
        self._deposit_mode = str(self.get_parameter("deposit_mode").value).lower()
        self._bin_edges = sorted(float(e) for e in self.get_parameter("deposit_bin_edges").value)
        self._deposit_bin_full_z = float(self.get_parameter("deposit_bin_full_z").value)
        self._deposit_bin_yaw = float(self.get_parameter("deposit_bin_yaw").value)
        self._deposit_bin_scan_margin = float(self.get_parameter("deposit_bin_scan_margin").value)
        self._deposit_bin_fill_order = str(self.get_parameter("deposit_bin_fill_order").value).lower()
        self._deposit_bin_fill_mode = str(self.get_parameter("deposit_bin_fill_mode").value).lower()
        nb = self._num_bins()
        # bins are indexed 0..nb-1 by ascending y; _bin_order is the fill sequence over those indices.
        if self._deposit_bin_fill_order.startswith("far"):
            base_order = list(range(nb - 1, -1, -1))
        else:
            base_order = list(range(nb))
        # Drop the N nearest (lowest-index) bins from the rotation -- too close to load cleanly.
        skip_n = max(0, int(self.get_parameter("deposit_bin_skip_nearest").value))
        self._bin_skip = set(range(min(skip_n, nb)))
        self._bin_order = [b for b in base_order if b not in self._bin_skip]
        if not self._bin_order and nb >= 1:
            self.get_logger().warn(
                f"deposit_bin_skip_nearest={skip_n} skips every bin; ignoring skip.")
            self._bin_skip = set()
            self._bin_order = base_order
        self._fill_ptr = 0                                        # position in _bin_order
        self._current_bin = self._bin_order[0] if self._bin_order else 0  # geometric bin being filled
        start_bin = int(self.get_parameter("start_bin").value)
        if start_bin >= 0:
            if start_bin in self._bin_order:
                self._fill_ptr = self._bin_order.index(start_bin)
                self._current_bin = start_bin
            else:
                self.get_logger().warn(
                    f"start_bin {start_bin} not in fill order {self._bin_order}; ignoring")
        self._bin_full = [False] * max(nb, 1)                     # per-bin: scanned top reached full_z
        self._bin_pile_top = [None] * max(nb, 1)                  # per-bin bundle-free post-release height
        self._bin_drops = [0] * max(nb, 1)                        # per-bin deposit count (this process)
        self._bins_done = False                                   # all bins full -> stop new cycles
        if self._deposit_mode == "bins":
            if nb < 1:
                self.get_logger().warn("deposit_mode=bins needs >=2 deposit_bin_edges; using pile mode.")
                self._deposit_mode = "pile"
            else:
                self._deposit_scan = True   # bin logic lives inside the deposit-scan machinery
                self.get_logger().info(
                    f"DEPOSIT bins mode: {nb} bins, edges={self._bin_edges}, "
                    f"full_z={self._deposit_bin_full_z:.2f}, yaw={self._deposit_bin_yaw:.3f} rad, "
                    f"fill_mode={self._deposit_bin_fill_mode}, order={self._deposit_bin_fill_order}, "
                    f"skip_nearest={sorted(self._bin_skip)} (fill sequence {self._bin_order}), "
                    f"first bin {self._current_bin} (center y={self._bin_center_y(self._current_bin):.2f}).")
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
                # No checkpoint (e.g. heuristic): store under the bind-mounted workspace so
                # runs are host-visible and survive container recreation. The container home
                # is unmounted writable-layer storage; use it only as a last resort.
                ws_logs = "/workspace/crane_testbed/logs"
                if os.path.isdir(ws_logs):
                    base = os.path.join(ws_logs, "crane_policy_debug")
                else:
                    base = os.path.join(os.path.expanduser("~"), "crane_policy_debug")
            self._debug_save_dir = os.path.join(base, f"run_{stamp}")
        if self._debug_save_dir:
            self.get_logger().info(f"Debug NPZ dumps -> {self._debug_save_dir}")
        # Resumed run (explicit debug_save_dir pointing at a previous session's dir): restore
        # the per-bin deposit state so mid-filled bins aren't treated as empty. Fresh "auto"
        # dirs have no bin_state.json, so this is a no-op for new runs.
        self._load_bin_state()
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
        self._traj_settle_s = float(self.get_parameter("traj_settle_s").value)
        self._traj_max_resends = int(self.get_parameter("traj_max_resends").value)
        self._telescope_speed = float(self.get_parameter("telescope_speed").value)
        self._grip_timeout = self.get_parameter("grip_timeout").value
        self._grip_close_dwell_s = float(self.get_parameter("grip_close_dwell_s").value)
        self._grip_open_dwell_s = float(self.get_parameter("grip_open_dwell_s").value)
        self._grapple_dt_ms = self.get_parameter("grapple_delta_time_ms").value
        self._grapple_da_deg = self.get_parameter("grapple_delta_angle_deg").value
        self._arm_tol = float(self.get_parameter("arm_tolerance").value)
        self._telescope_tol = float(self.get_parameter("telescope_tolerance").value)
        self._yaw_tol = float(self.get_parameter("yaw_tolerance").value)
        self._telescope_static = bool(self.get_parameter("telescope_static").value)
        self._telescope_static_value = float(self.get_parameter("telescope_static_value").value)
        self._measure_stability_enabled = bool(
            self.get_parameter("measure_stability_enabled").value)
        self._measure_stability_autostart = bool(
            self.get_parameter("measure_stability_autostart").value)
        self._measure_stability_action_name = self.get_parameter(
            "measure_stability_action_name").value
        self._measure_stability_timeout_s = float(
            self.get_parameter("measure_stability_timeout_sec").value)
        self._measure_stability_required_window_s = float(
            self.get_parameter("measure_stability_required_window_sec").value)
        self._measure_stability_required_dwell_s = float(
            self.get_parameter("measure_stability_required_dwell_sec").value)
        self._measure_stability_min_samples = int(
            self.get_parameter("measure_stability_min_samples").value)
        self._measure_stability_max_gap_s = float(
            self.get_parameter("measure_stability_max_sample_gap_sec").value)
        self._measure_stability_stability_var = float(
            self.get_parameter(
                "measure_stability_stability_variance_threshold_rad2").value)
        self._measure_stability_gravity_var = float(
            self.get_parameter(
                "measure_stability_gravity_variance_threshold_rad2").value)
        self._measure_stability_grapple_var = float(
            self.get_parameter(
                "measure_stability_grapple_variance_threshold_rad2").value)
        self._measure_stability_reward_exponent = float(
            self.get_parameter("measure_stability_reward_exponent").value)
        self._measure_stability_failure_policy = str(
            self.get_parameter("measure_stability_failure_policy").value).strip().lower()
        if self._measure_stability_failure_policy not in ("advance", "hold"):
            self.get_logger().warn(
                "measure_stability_failure_policy must be 'advance' or 'hold'; "
                "using 'advance'")
            self._measure_stability_failure_policy = "advance"
        self._measure_stability_goal_response_timeout_s = float(
            self.get_parameter("measure_stability_goal_response_timeout_sec").value)

        # Camera/base extrinsics (overridden by TF lookup once available)
        self.cam_pos = np.array([-1.0, 1.92, 1.577], dtype=np.float32)
        self.cam_quat = np.array([0.6124, 0.3536, 0.3536, 0.6124], dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Policy
        ckpt = checkpoint if checkpoint else None
        self.policy = load_policy(policy_type, ckpt, bounds_min, bounds_max,
                                  cossin, device,
                                  heuristic_dig=float(self.get_parameter("heuristic_dig").value),
                                  support_frac=float(self.get_parameter("support_frac").value))
        # The checkpoint is authoritative on input size. A mismatch with the num_points param is
        # always a config error, and left alone it would only surface at the first inference (or,
        # for architectures that tolerate it, as a silently mis-aimed policy). Adopt and shout.
        _ckpt_np = getattr(self.policy, "num_points", None)
        if _ckpt_np is not None and int(_ckpt_np) != int(self.num_points):
            self.get_logger().warn(
                f"num_points param {self.num_points} != checkpoint {_ckpt_np}; "
                f"using the checkpoint value (the cloud pipeline will FPS to {_ckpt_np})")
            self.num_points = int(_ckpt_np)
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
        self._carry_waypoints = max(0, int(self.get_parameter("carry_waypoints").value))
        self._carry_height = float(self.get_parameter("carry_height").value)
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
        self._cloud_stamp = None       # sensor stamp of latest_cloud = actual CAPTURE time
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
        self.cycle_count = max(0, int(self.get_parameter("start_cycle").value) - 1)
        self._seq_id = 0
        self._seen_executing = False
        self._idle_since = None        # clock time we first saw the PLC idle after executing this move
        self._wait_start = None
        self._grapple_future = None
        self._traj_future = None
        self._goal_joints = None
        self._stability_send_goal_future = None
        self._stability_goal_handle = None
        self._stability_result_future = None
        self._stability_goal_sent = False
        self._stability_result_handled = False
        self._stability_terminal_failure = False
        self._stability_measurement_start = None
        self._last_stability_result = None
        # Per-cycle consolidated record (grasp decision + deposit decision +
        # stability), flushed to episode.jsonl when the cycle completes.
        self._episode_record = {}

        # Subscribers. Primary input is the ZED registered cloud (matches what the ZED publishes);
        # depth-image path kept for setups that publish a depth image instead.
        from rclpy.qos import qos_profile_sensor_data
        # subscribe to the real ZED topic directly (was /zedx/points + a mandatory launch remap;
        # forgetting the remap silently hung every scan waiting for clouds that never came)
        self.declare_parameter("cloud_topic", "/zed_0/zed_node/point_cloud/cloud_registered")
        self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value,
                                 self._cloud_cb, qos_profile_sensor_data)
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
        self.stability_client = ActionClient(
            self,
            MeasureStability,
            self._measure_stability_action_name,
        )
        self._stability_server_proc = None
        if (self._measure_stability_enabled and self._measure_stability_autostart
                and not self.stability_client.wait_for_server(timeout_sec=1.5)):
            try:
                self._stability_server_proc = subprocess.Popen(
                    ["ros2", "run", "fpi_crane_rl_metrics",
                     "measure_stability_action_node"])
                self.get_logger().info(
                    "measure_stability server not found; autostarted "
                    f"measure_stability_action_node (pid {self._stability_server_proc.pid})")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f"could not autostart measure_stability_action_node: {exc}")

        # Optional external target source: the pile analyzer service. The srv type
        # lives in log_loading_interfaces (emt_acm_mh_planning); import lazily so the
        # node runs unchanged when that package is not built in this workspace.
        self._target_source = str(self.get_parameter("target_source").value).lower()
        self._analyzer_service = self.get_parameter("analyzer_service").value
        self._analyzer_timeout = float(self.get_parameter("analyzer_timeout").value)
        self._analyzer_image_topic = self.get_parameter("analyzer_image_topic").value
        self._analyzer_z_offset = float(self.get_parameter("analyzer_z_offset").value)
        self._analyzer_yaw_offset = float(self.get_parameter("analyzer_yaw_offset").value)
        self._analyzer_client = None
        self._analyzer_srv_type = None
        self._analyzer_future = None
        self._analyzer_obs = None        # policy-input cloud kept for fallback / debug dump
        self._latest_cloud_msg = None    # raw PointCloud2 (service request needs the ROS msg)
        self._latest_image_msg = None    # raw rect RGB Image for the service request
        if self._target_source == "pile_analyzer":
            try:
                from log_loading_interfaces.srv import LogPickupAnalysis
            except ImportError as e:
                self.get_logger().error(
                    f"target_source=pile_analyzer but log_loading_interfaces is not importable "
                    f"({e}); using the onboard policy instead. Build/source log_loading_interfaces "
                    f"in this workspace to enable the analyzer.")
                self._target_source = "policy"
            else:
                self._analyzer_srv_type = LogPickupAnalysis
                self._analyzer_client = self.create_client(
                    LogPickupAnalysis, self._analyzer_service)
                self.create_subscription(Image, self._analyzer_image_topic,
                                         self._analyzer_image_cb, qos_profile_sensor_data)
                self.get_logger().info(
                    f"Target source: pile_analyzer via {self._analyzer_service} "
                    f"(image topic {self._analyzer_image_topic}, "
                    f"timeout {self._analyzer_timeout:.1f}s, policy fallback armed)")

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
        self._latest_cloud_msg = msg   # raw msg kept for the pile-analyzer service request
        self._cloud_frame = msg.header.frame_id
        self._cloud_recv_time = self.get_clock().now()
        self._cloud_stamp = rclpy.time.Time.from_msg(msg.header.stamp)

    def _analyzer_image_cb(self, msg):
        self._latest_image_msg = msg

    def _cloud_captured_after(self, deadline) -> bool:
        """True if the latest cloud's CONTENT is from after `deadline` (rclpy Time).

        Receive-time + assumed cloud_capture_latency_s heuristic. (A sensor-stamp gate was tried
        2026-07-15 and reverted: the ZED stamps run on a different clock/offset than the node and
        the gate never opened, hanging the FSM at gaze.)"""
        if self.latest_cloud is None or self._cloud_recv_time is None:
            return False
        return self._cloud_recv_time >= deadline + Duration(seconds=self._cloud_latency_s)

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

    def _grapple_yaw_to_joint(self, target_grapple_yaw_b: float, slew_delta: float = 0.0) -> float:
        """Yaw-joint goal that puts the grapple at target_grapple_yaw_b (base frame) AFTER the arm
        move. The grapple's base-frame yaw is slew + yaw joint (+const), so a move that also turns
        the slew must take that travel out of the yaw joint: pass the commanded slew change as
        slew_delta. (Without it, CARRY_HOME arrived at home off-parallel by exactly the slew travel;
        the pi/2 drop_yaw default silently compensated for the typical rack->home slew.)"""
        def _wrap(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        rotation_diff = _wrap(target_grapple_yaw_b - self.current_ee_yaw) - slew_delta
        base = self.current_yaw_joint + rotation_diff
        # Pi-equivalent candidates, restricted to the URDF joint range (+-3.1241 with margin)
        # so a near-limit start rotates the long way instead of getting the move rejected.
        # The +-3.12 window spans almost 2*pi, so a feasible equivalent always exists.
        candidates = [base + k * math.pi for k in (-2, -1, 0, 1, 2)]
        feasible = [c for c in candidates if abs(c) <= 3.10]
        return min(feasible or candidates, key=lambda c: abs(c - self.current_yaw_joint))

    # Main sequencer tick

    def _tick(self):
        if self.cycle_count >= self.max_cycles:
            return
        self._update_extrinsics_from_tf()
        self._update_basegrapple_from_tf()
        if self._cloud_recv_time is None and not hasattr(self, "_no_cloud_warned"):
            if not hasattr(self, "_first_tick_time"):
                self._first_tick_time = self.get_clock().now()
            elif (self.get_clock().now() - self._first_tick_time) > Duration(seconds=10.0):
                self.get_logger().warn(
                    "No point cloud received yet on the cloud topic - every scan/capture will "
                    "wait forever. Is the ZED up / cloud_topic correct?")
                self._no_cloud_warned = True

        if self.seq_state == SEQ_IDLE:
            if self._deposit_mode == "bins" and self._bins_done:
                if not hasattr(self, "_bins_done_logged"):
                    self.get_logger().info("All trailer bins full; deposition complete, holding.")
                    self._bins_done_logged = True
                return
            if self._deposit_scan and not self._prescan_done:
                self._start_prescan()
            else:
                self._start_gaze()
        elif self.seq_state == SEQ_GAZE:
            self._check_gaze_done()
        elif self.seq_state == SEQ_SEND:
            self._send_current_move()
        elif self.seq_state == SEQ_WAIT_TRAJ:
            self._check_traj_done()
        elif self.seq_state == SEQ_WAIT_GRIP:
            self._check_grip_done()
        elif self.seq_state == SEQ_WAIT_STABILITY:
            self._check_stability_done()
        elif self.seq_state == SEQ_WAIT_ANALYSIS:
            self._check_analysis_done()

    def _start_prescan(self):
        """Before the FIRST cycle: drive to home over the trailer and scan it (no bundle held yet)
        so the first deposit already uses a cached, uncontaminated pile top instead of the
        bundle-contaminated live scan. Runs once; then every post-release scan keeps the cache
        fresh for the rest of the run."""
        if not self._joints_received or not self._plc_received:
            return
        self._prescan_active = True
        self.get_logger().info("PRESCAN: driving home to scan the trailer before the first cycle")
        m = String(); m.data = "PRESCAN (trailer)"; self.pub_fsm_state.publish(m)
        self._append_phase_log("PRESCAN")
        drop = list(self.fsm_cfg.drop_position)
        if self._deposit_mode == "bins" and not self._bins_done:
            drop[1] = self._bin_center_y(self._current_bin)
        self.moves = [
            Move(Phase.CLEAR, "traj", drop, None),
            Move(Phase.SETTLE, "cache_scan", None, None),
        ]
        self.move_idx = 0
        self.seq_state = SEQ_SEND

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
        self._cancel_stability_goal("cycle reset")
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
        # require the cloud's CAPTURE time (sensor stamp) to be post-settle, so the content is
        # genuinely from the held gaze viewpoint, not mid-slew
        if not self._cloud_captured_after(deadline):
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
            self._crop_min, self._crop_max, self.num_points,
            pole_x_gt=self._pole_x_gt, pole_y_lt=self._pole_y_lt,
            pole2_x_gt=self._pole2_x_gt, pole2_y_gt=self._pole2_y_gt,
            pole3_x_lt=self._pole3_x_lt, pole3_y_lt=self._pole3_y_lt,
            ror_radius=self._ror_radius, ror_min_neighbors=self._ror_min_neighbors,
        )
        self._publish_input_cloud(obs)   # DEBUG: what the policy actually sees
        self._publish_crop_box()         # DEBUG: where the crop region sits
        if self._target_source == "pile_analyzer" and self._start_pile_analysis(obs):
            return   # target arrives via SEQ_WAIT_ANALYSIS
        x, y, z, yaw = self._policy_target(obs)
        self._plan_from_target(x, y, z, yaw, obs, source="policy")

    def _policy_target(self, obs):
        """Onboard policy with the rack_y_shift shim: the net sees the cropped cloud shifted
        back into training-era coords and its target (decoded with the training-era bounds)
        is shifted forward again, so execution happens in the true physical frame."""
        s = self._rack_y_shift
        if s == 0.0:
            return self.policy.get_target(obs)
        o = obs.clone().reshape(-1, 3)
        o[:, 1] -= s
        x, y, z, yaw = self.policy.get_target(o.reshape(obs.shape))
        return x, y + s, z, yaw

    def _plan_from_target(self, x, y, z, yaw, obs, source="policy"):
        """Common tail of a capture: log/publish the chosen target and start the move sequence.
        Applies the platform bed floor (bed_z + bed_margin) to EVERY target source here, the
        single choke point before execution; the raw pre-clamp z is kept in the log line and
        decisions.jsonl so analysis can attribute misses to the decision vs the envelope."""
        z_raw = z
        floor = self._bed_z + self._bed_margin
        if z < floor:
            z = floor
            self.get_logger().warn(
                f"BED FLOOR: {source} z {z_raw:.2f} clamped to {floor:.2f} "
                f"(bed_z {self._bed_z:.2f} + margin {self._bed_margin:.2f})")
        # Live target-quality readout: observed points within 0.5 m of the commanded xy.
        # Healthy pile targets run ~50-100 of the FPS'd cloud; the 2026-08-03 gap/air failures
        # ran 0-20. Lets the operator see a bad decision BEFORE the crane commits to it.
        try:
            _pts = obs.detach().cpu().numpy().reshape(-1, 3)
            _pts = _pts[np.abs(_pts).sum(1) > 1e-6]
            support = int((np.linalg.norm(
                _pts[:, :2] - np.array([x, y], dtype=np.float32), axis=1) < 0.5).sum())
        except Exception:  # noqa: BLE001
            support = -1
        if 0 <= support < 20:
            self.get_logger().warn(
                f"LOW SUPPORT: only {support} observed points within 0.5 m of the target - "
                f"this looks like a gap/air/outlier decision")
        self.get_logger().info(
            f"Cycle {self.cycle_count + 1}: target=({x:.2f}, {y:.2f}, "
            f"{z:.2f}), yaw={math.degrees(yaw):.1f}deg [{source}] support={support}"
            + (f" [raw z {z_raw:.2f}]" if z != z_raw else ""))
        self._publish_target(x, y, z, yaw)
        self._save_debug(obs, x, y, z, yaw, z_raw=z_raw, support=support)
        self.moves = self._build_moves(x, y, z, yaw)
        self.move_idx = 0
        self.seq_state = SEQ_SEND

    def _start_pile_analysis(self, obs):
        """Kick off the async log_pickup_analysis call. Returns True when the request went
        out (result handled in _check_analysis_done); False lets the caller fall back to
        the onboard policy immediately."""
        if self._analyzer_client is None:
            return False
        if self._latest_image_msg is None:
            self.get_logger().warn(
                f"PILE_ANALYZER: no image on {self._analyzer_image_topic}; using policy")
            return False
        if not self._analyzer_client.service_is_ready():
            self.get_logger().warn(
                f"PILE_ANALYZER: service {self._analyzer_service} not available; using policy")
            return False
        req = self._analyzer_srv_type.Request()
        req.image = self._latest_image_msg
        req.map = self._latest_cloud_msg
        # Loading zone: the policy crop bounds as a base-frame rectangle (the server
        # transforms it via TF and only keeps xy, so z=0 is fine).
        x0, y0 = float(self._bounds_min[0]), float(self._bounds_min[1])
        x1, y1 = float(self._bounds_max[0]), float(self._bounds_max[1])
        req.vertices_loading_zone = [
            Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y0, z=0.0),
            Point(x=x1, y=y1, z=0.0), Point(x=x0, y=y1, z=0.0)]
        req.loading_zone_header.frame_id = self._base_frame
        req.loading_zone_header.stamp = self.get_clock().now().to_msg()
        self._analyzer_obs = obs
        self._analyzer_future = self._analyzer_client.call_async(req)
        self._wait_start = self.get_clock().now()
        self.seq_state = SEQ_WAIT_ANALYSIS
        m = String(); m.data = f"ANALYZE (cycle {self.cycle_count})"; self.pub_fsm_state.publish(m)
        self._append_phase_log("ANALYZE")
        self.get_logger().info(
            f"PILE_ANALYZER: requested analysis "
            f"(cloud frame {self._latest_cloud_msg.header.frame_id}, "
            f"image frame {self._latest_image_msg.header.frame_id})")
        return True

    def _check_analysis_done(self):
        fut = self._analyzer_future
        if fut is None:
            self._analysis_fallback("no request in flight")
            return
        if not fut.done():
            if self._waited_longer_than(self._analyzer_timeout):
                self._analyzer_future = None   # a late result is ignored
                self._analysis_fallback(f"timeout after {self._analyzer_timeout:.1f}s")
            return
        self._analyzer_future = None
        err = fut.exception()
        if err is not None:
            self._analysis_fallback(f"service call failed: {err}")
            return
        resp = fut.result()
        if not resp.success:
            self._analysis_fallback("analyzer returned success=False")
            return
        try:
            x, y, z, yaw = self._analysis_target(resp)
        except Exception as e:  # noqa: BLE001
            self._analysis_fallback(f"attack_pose unusable: {e}")
            return
        obs, self._analyzer_obs = self._analyzer_obs, None
        self._plan_from_target(x, y, z, yaw, obs, source="pile_analyzer")

    def _analysis_target(self, resp):
        """attack_pose -> (x, y, z, yaw) grasp target in our base frame."""
        p = resp.attack_pose.pose.pose
        pos = np.array([p.position.x, p.position.y, p.position.z], dtype=np.float64)
        yaw = _quat_to_yaw(p.orientation.w, p.orientation.x,
                           p.orientation.y, p.orientation.z)
        frame = resp.attack_pose.header.frame_id
        if frame and frame != self._base_frame:
            tf = self._tf_buffer.lookup_transform(
                self._base_frame, frame, rclpy.time.Time())
            tr, q = tf.transform.translation, tf.transform.rotation
            pos = _quat_rotate(q.w, q.x, q.y, q.z, pos) + np.array([tr.x, tr.y, tr.z])
            yaw += _quat_to_yaw(q.w, q.x, q.y, q.z)
        yaw = (yaw + self._analyzer_yaw_offset + math.pi) % (2.0 * math.pi) - math.pi
        return (float(pos[0]), float(pos[1]),
                float(pos[2]) + self._analyzer_z_offset, yaw)

    def _analysis_fallback(self, reason):
        """Analyzer path failed mid-cycle: plan with the onboard policy on the SAME capture."""
        self.get_logger().warn(f"PILE_ANALYZER: {reason}; falling back to onboard policy")
        obs, self._analyzer_obs = self._analyzer_obs, None
        if obs is None:
            self.get_logger().warn("PILE_ANALYZER: no cached capture; restarting cycle")
            self.seq_state = SEQ_IDLE
            return
        x, y, z, yaw = self._policy_target(obs)
        self._plan_from_target(x, y, z, yaw, obs, source="policy-fallback")

    def _carry_waypoints_between(self, hover, drop):
        """Intermediate constant-height transit poses from hover to drop (endpoints excluded).
        xy is linearly interpolated hover->drop; z is held at carry_height (or max of the two
        endpoints when carry_height <= 0) so the carry never dips below either end. Returns [] when
        carry_waypoints == 0."""
        n = self._carry_waypoints
        if n <= 0:
            return []
        carry_z = self._carry_height if self._carry_height > 0 else max(hover[2], drop[2])
        pts = []
        for i in range(1, n + 1):
            f = i / (n + 1)
            wp = [hover[0] + f * (drop[0] - hover[0]),
                  hover[1] + f * (drop[1] - hover[1]),
                  carry_z]
            # A waypoint is optional. Drop any that IK can't reach so we never enqueue a move that
            # would loop forever on repeated IK failure (the sender retries a False send). Seeded on
            # the current arm; a reachability gate, not the exact runtime solve.
            if self.ik.solve(wp, self.current_arm) is None:
                self.get_logger().warn(
                    f"CARRY_WAYPOINT: IK unreachable at {np.round(wp, 2).tolist()}; skipping it")
                continue
            pts.append(wp)
        return pts

    def _build_moves(self, x, y, z, yaw):
        cfg = self.fsm_cfg
        hover = [x, y, z + cfg.hover_clear]
        approach = [x, y, z + cfg.approach_above]
        drop = list(cfg.drop_position)
        dyaw = cfg.drop_yaw
        carry_yaw = dyaw          # yaw commanded during the transport slew
        align_at_bin = False      # bins: align perpendicular in-place AT the bin instead
        if self._deposit_mode == "bins" and not self._bins_done:
            # Deposit in the middle of the current bin, laid perpendicular to the trailer length.
            drop[1] = self._bin_center_y(self._current_bin)
            dyaw = self._deposit_bin_yaw
            # HOLD the grapple through the big rack->bin slew (carry_yaw=None): commanding the
            # perpendicular yaw *during* that ~90 deg slew makes the grapple rotate the whole way
            # over (world yaw = slew + carrier joint). Instead align perpendicular in-place at the
            # bin as a dedicated move, like the pick-side ALIGN_YAW.
            carry_yaw = None
            align_at_bin = True
            self.get_logger().info(
                f"BINS: cycle {self.cycle_count + 1} -> bin {self._current_bin} "
                f"(carry to xy=({drop[0]:.2f}, {drop[1]:.2f}), align to yaw={math.degrees(dyaw):.0f} deg at bin)")
        moves = [
            Move(Phase.HOVER_UP, "traj", hover, None),
            Move(Phase.ALIGN_YAW, "traj", hover, yaw,
                 velocity=self._yaw_velocity, min_duration=self._yaw_min_duration),
            Move(Phase.DESCEND, "traj", approach, yaw),
            Move(Phase.CLOSE, "grip", gripper=GRIPPER_CLOSE),
            Move(Phase.LIFT_HIGH, "traj", hover, yaw),
        ]
        # Anti-droop: transit from hover to drop through constant-height Cartesian waypoints so the
        # PLC's joint-space interpolation can't arc the tip down mid-swing. Each waypoint linearly
        # interpolates xy from hover to drop while z is held at carry_z; the final CARRY_HOME move
        # then closes the last leg to drop. Waypoints carry the same yaw as CARRY_HOME (carry_yaw:
        # held in bins mode, drop_yaw in pile mode) so we don't reintroduce mid-slew spin.
        for wp in self._carry_waypoints_between(hover, drop):
            moves.append(Move(Phase.CARRY_WAYPOINT, "traj", wp, carry_yaw))
        moves.append(Move(Phase.CARRY_HOME, "traj", drop, carry_yaw))
        if align_at_bin:
            # in-place perpendicular alignment at the bin (slow yaw move, no xy/z change)
            moves.append(Move(Phase.ALIGN_YAW, "traj", list(drop), dyaw,
                              velocity=self._yaw_velocity, min_duration=self._yaw_min_duration))
        if self._deposit_scan:
            # At home (yaw already aligned by CARRY_HOME in pile mode, or the bin ALIGN_YAW in bins
            # mode), scan the trailer for the pile top and descend
            # to it (+offset) before releasing. yaw=None => HOLD the current yaw (don't re-command it, so
            # the descend doesn't turn the grapple). The z here is a placeholder; _resolve_deposit fills it.
            moves.append(Move(Phase.LOWER_TO_DROP, "deposit", list(drop), None))
        moves.append(Move(Phase.OPEN, "grip", gripper=GRIPPER_OPEN))
        if self._deposit_scan:
            # Post-release: raise back to home over the trailer, then scan the pile WITHOUT a held
            # bundle in view and cache the top for the NEXT cycle's deposit (the pile only changes
            # when we deposit, so the cache stays exact until then).
            moves.append(Move(Phase.CLEAR, "traj", list(drop), None))
            moves.append(Move(Phase.SETTLE, "cache_scan", None, None))
        return moves

    def _send_current_move(self):
        move = self.moves[self.move_idx]
        self._publish_fsm_state(move.phase)
        if move.kind == "cache_scan":
            # Post-release trailer scan (empty grapple, back at home z): cache the pile top for the
            # next cycle's deposit. Same settle + freshness gating as a live deposit scan.
            if not self._cache_trailer_scan():
                return
            self._advance_move()
            return
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
                self._idle_since = None
                self._wait_start = self.get_clock().now()
                self.seq_state = SEQ_WAIT_TRAJ
        else:
            if self._send_grapple(move.gripper):
                self._wait_start = self.get_clock().now()
                self.seq_state = SEQ_WAIT_GRIP

    def _bin_state_path(self):
        return os.path.join(self._debug_save_dir, "bin_state.json") if self._debug_save_dir else None

    def _save_bin_state(self):
        """Persist per-bin deposit state so a resumed run (same debug_save_dir) doesn't treat
        mid-filled bins as empty (the empty-bin blind z would dig into the existing pile)."""
        path = self._bin_state_path()
        if not path or self._deposit_mode != "bins":
            return
        try:
            os.makedirs(self._debug_save_dir, exist_ok=True)
            state = {
                "bin_pile_top": self._bin_pile_top,
                "bin_drops": self._bin_drops,
                "bin_full": self._bin_full,
                "fill_ptr": self._fill_ptr,
                "current_bin": self._current_bin,
                "bins_done": self._bins_done,
            }
            with open(path, "w") as f:
                json.dump(state, f)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"bin state save failed: {exc}")

    def _load_bin_state(self):
        """Restore per-bin deposit state written by a previous session in this debug_save_dir."""
        path = self._bin_state_path()
        if not path or self._deposit_mode != "bins" or not os.path.isfile(path):
            return
        try:
            with open(path) as f:
                state = json.load(f)
            nb = len(self._bin_pile_top)
            self._bin_pile_top = (list(state.get("bin_pile_top", [])) + [None] * nb)[:nb]
            self._bin_drops = (list(state.get("bin_drops", [])) + [0] * nb)[:nb]
            self._bin_full = (list(state.get("bin_full", [])) + [False] * nb)[:nb]
            self._bins_done = bool(state.get("bins_done", False))
            if int(self.get_parameter("start_bin").value) < 0:
                # No explicit start_bin: continue where the previous session left off.
                self._fill_ptr = int(state.get("fill_ptr", self._fill_ptr))
                self._current_bin = int(state.get("current_bin", self._current_bin))
            self.get_logger().info(
                f"BIN STATE restored from {path}: tops={self._bin_pile_top} "
                f"drops={self._bin_drops} full={self._bin_full} current={self._current_bin}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"bin state load failed: {exc}")

    def _cache_trailer_scan(self) -> bool:
        """Post-release, back at home: settle + wait for a fresh cloud, scan the trailer (empty
        grapple, no bundle in view) and cache the pile top for the next cycle's deposit. Returns
        True when done (also on a failed scan: the cache clears and the next deposit live-scans)."""
        now = self.get_clock().now()
        if self._deposit_start_time is None:
            self._deposit_start_time = now
            self._deposit_waiting_fresh = False
            return False
        deadline = self._deposit_start_time + Duration(seconds=self._gaze_dwell_s)
        if now < deadline:
            return False
        if self._cloud_frame is None or not self._cloud_captured_after(deadline):
            if not self._deposit_waiting_fresh:
                self.get_logger().info("POST-RELEASE: waiting for a fresh trailer cloud to cache...")
                self._deposit_waiting_fresh = True
            return False
        pile_top = self._scan_trailer_top(y_range=self._bin_scan_range())
        if pile_top is None and now < deadline + Duration(seconds=8.0):
            # TF chain not ready yet (the ZED stamps TF on a lagging clock, so a fresh buffer
            # right after node start can't serve the lookup) or transient empty scan: retry.
            return False
        self._deposit_start_time = None
        self._deposit_waiting_fresh = False
        self._cached_pile_top = pile_top
        if pile_top is None:
            self.get_logger().warn("POST-RELEASE: trailer scan empty; next deposit will live-scan.")
        else:
            self.get_logger().info(f"POST-RELEASE: cached trailer pile top {pile_top:.2f} for next cycle")
        # bins mode: decide the next bin. In "column" mode we stay on the current bin until its scanned
        # top reaches full_z, then step to the next. In "sequential" mode we rotate to the next open bin
        # every cycle (even layers), skipping any bin already full. Either way a bin at/over full_z is
        # marked full and skipped; when all bins are full, deposition stops (handled in _tick).
        if self._deposit_mode == "bins":
            filled = self._current_bin
            # This scan is bundle-free (post-release). Cache it per-bin so the NEXT visit to this bin
            # uses its own true height instead of live-scanning the held bundle. None (empty scan) is
            # left as-is -> that bin still first-drops to the empty-bin z.
            if pile_top is not None:
                self._bin_pile_top[filled] = pile_top
            now_full = pile_top is not None and pile_top >= self._deposit_bin_full_z
            if now_full and not self._bin_full[filled]:
                self._bin_full[filled] = True
                self.get_logger().info(
                    f"BIN {filled} full (top {pile_top:.2f} >= {self._deposit_bin_full_z:.2f}).")
            rotate = (self._deposit_bin_fill_mode != "column") or now_full
            if rotate:
                self._cached_pile_top = None   # per-bin cache (_bin_pile_top) drives bins-mode z now
                self._advance_bin()
                if self._bins_done:
                    self.get_logger().info(
                        f"all {len(self._bin_order)} fillable bins filled -> deposition complete.")
                else:
                    self.get_logger().info(
                        f"BINS: -> next bin {self._current_bin} "
                        f"(center y={self._bin_center_y(self._current_bin):.2f}).")
            self._save_bin_state()
        return True

    def _advance_bin(self):
        """Step _current_bin to the next not-yet-full bin along the fill order, wrapping around.
        Iterates over _bin_order (which already excludes skipped bins). Sets _bins_done if every
        fillable bin is full."""
        norder = len(self._bin_order)
        if norder < 1:
            self._bins_done = True
            return
        for _ in range(norder):
            self._fill_ptr = (self._fill_ptr + 1) % norder
            cand = self._bin_order[self._fill_ptr]
            if not self._bin_full[cand]:
                self._current_bin = cand
                return
        self._bins_done = True

    def _resolve_deposit(self, move) -> bool:
        """Set move.position[2] = min(pile_top + offset, home_z). Prefers the pile top cached by the
        post-release scan of the PREVIOUS cycle (no bundle in view then, and the pile only changes
        when we deposit). Falls back to a live scan at home (settle + fresh cloud) on the first
        cycle or when the cache is empty. Returns True once the z is set."""
        home_z = float(self.fsm_cfg.drop_position[2])
        if self._deposit_mode == "bins" and not self._bins_done:
            # Bins mode NEVER live-scans at LOWER_TO_DROP: the grapple is still holding the bundle,
            # which hangs into the trailer box and reads as a bogus-high pile top (~the box ceiling).
            # Use this bin's cached bundle-free post-release height; first drop into an empty bin has
            # no cache -> descend to the fixed empty-bin z. Every later drop uses cache + offset.
            tx, ty = float(move.position[0]), float(move.position[1])
            cached = self._bin_pile_top[self._current_bin]
            if cached is not None:
                deposit_z = min(cached + self._deposit_offset, home_z)
                self.get_logger().info(
                    f"DEPOSIT: bin {self._current_bin} xy=({tx:.2f}, {ty:.2f}) cached pile_top="
                    f"{cached:.2f} + off {self._deposit_offset:.2f} -> descend to z={deposit_z:.2f} "
                    f"(home_z={home_z:.2f})")
            elif self._bin_drops[self._current_bin] > 0:
                # No cache but the bin already holds logs (post-release scan failed / node
                # restarted): the pile could be anywhere up to full_z, so blind-drop from
                # above the worst case instead of the empty-bin z (which digs into the pile).
                deposit_z = min(self._deposit_bin_full_z + self._deposit_offset, home_z)
                self.get_logger().warn(
                    f"DEPOSIT: bin {self._current_bin} xy=({tx:.2f}, {ty:.2f}) has "
                    f"{self._bin_drops[self._current_bin]} drops but no cached height -> "
                    f"safe blind z={deposit_z:.2f} (full_z + off, home_z={home_z:.2f})")
            else:
                deposit_z = min(self._deposit_bin_empty_z, home_z)
                self.get_logger().info(
                    f"DEPOSIT: bin {self._current_bin} xy=({tx:.2f}, {ty:.2f}) first drop (empty, no "
                    f"cache) -> descend to empty-bin z={deposit_z:.2f} (home_z={home_z:.2f})")
            if not self._deposit_dry_run:
                self._bin_drops[self._current_bin] += 1
                self._save_bin_state()
            self._publish_deposit_box()
            move.position = [tx, ty, float(deposit_z)]
            self._publish_deposit_target(tx, ty, deposit_z)
            self._save_deposit_debug(tx, ty, deposit_z)
            return True
        if self._cached_pile_top is not None:
            pile_top = self._cached_pile_top
            deposit_z = min(pile_top + self._deposit_offset, home_z)
            tx, ty = float(move.position[0]), float(move.position[1])
            bin_txt = f"bin {self._current_bin} " if self._deposit_mode == "bins" else ""
            self.get_logger().info(
                f"DEPOSIT: {bin_txt}xy=({tx:.2f}, {ty:.2f}) cached pile_top={pile_top:.2f} "
                f"+ off {self._deposit_offset:.2f} -> descend to z={deposit_z:.2f} (home_z={home_z:.2f})")
            self._publish_deposit_box()
            move.position = [tx, ty, float(deposit_z)]
            self._publish_deposit_target(tx, ty, deposit_z)
            self._save_deposit_debug(tx, ty, deposit_z)
            return True
        now = self.get_clock().now()
        if self._deposit_start_time is None:
            self._deposit_start_time = now
            self._deposit_waiting_fresh = False
            return False
        deadline = self._deposit_start_time + Duration(seconds=self._gaze_dwell_s)
        if now < deadline:
            return False
        # same freshness gate as gaze: cloud capture time (sensor stamp) must be post-settle.
        if self._cloud_frame is None or not self._cloud_captured_after(deadline):
            if not self._deposit_waiting_fresh:
                self.get_logger().info("DEPOSIT: at home, waiting for a fresh trailer cloud...")
                self._deposit_waiting_fresh = True
            return False
        self._publish_deposit_box()
        pile_top = self._scan_trailer_top(y_range=self._bin_scan_range())   # also publishes the in-box sampled points
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
            if self._deposit_mode == "bins" and not self._bins_done:
                # viz-only rescan of the ACTIVE bin: the cached-z deposit paths skip
                # the live scan, so _last_deposit_pts still held the previous
                # post-release scan (a different bin) and the dump showed the wrong
                # section of the trailer. The z decision above is untouched.
                self._scan_trailer_top(y_range=self._bin_scan_range())
            pts = self._last_deposit_pts if self._last_deposit_pts is not None else np.zeros((0, 3), np.float32)
            bins = self._deposit_mode == "bins"
            yaw = float(self._deposit_bin_yaw if (bins and not self._bins_done) else self.fsm_cfg.drop_yaw)
            # bins mode: save the compartment edges + active bin so the review overlay can draw them
            # (like the action box on the policy overlay). Empty edges -> no compartments drawn.
            bin_edges = np.asarray(self._bin_edges if bins else [], np.float32)
            fn = os.path.join(self._debug_save_dir, f"deposit_debug_{self.cycle_count + 1:03d}.npz")
            np.savez(fn, points=np.asarray(pts, np.float32),
                     target=np.array([x, y, z, yaw], dtype=np.float32),
                     bounds_min=np.asarray(self._trailer_bounds_min, np.float32),
                     bounds_max=np.asarray(self._trailer_bounds_max, np.float32),
                     bin_edges=bin_edges,
                     active_bin=np.array([self._current_bin if bins else -1], np.int32),
                     # bins skipped as too-close (never filled) -> overlay can gray them out.
                     skip_bins=np.array(sorted(self._bin_skip) if bins else [], np.int32),
                     # the fill max (base_link z) -> the review overlay caps the box/dividers here
                     # instead of the tall scan ceiling, so the visual matches the real max height.
                     max_z=np.array([self._deposit_bin_full_z], np.float32))
            self.get_logger().info(f"saved deposit debug -> {fn} ({np.asarray(pts).shape[0]} pts)")
            self._append_decision_log(x, y, z, yaw, kind="deposit", npz=os.path.basename(fn))
            if self._debug_save_jpg:
                self._render_deposit_jpg(np.asarray(pts, np.float32), x, y, z, yaw, fn[:-4] + ".jpg")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"deposit debug save failed: {e}")

    def _num_bins(self):
        return max(0, len(self._bin_edges) - 1)

    def _bin_y_range(self, i):
        """(ymin, ymax) of bin i in base_link."""
        return self._bin_edges[i], self._bin_edges[i + 1]

    def _bin_center_y(self, i):
        lo, hi = self._bin_y_range(i)
        return 0.5 * (lo + hi)

    def _bin_scan_range(self):
        """y-range to restrict the trailer z-scan to the CURRENT bin (bins mode), else None (full box).
        Inset by deposit_bin_scan_margin so the boundary poles are not sampled as the bin's pile top."""
        if self._deposit_mode != "bins" or self._bins_done:
            return None
        lo, hi = self._bin_y_range(self._current_bin)
        mgn = self._deposit_bin_scan_margin
        if hi - lo > 2.0 * mgn:
            lo, hi = lo + mgn, hi - mgn
        return lo, hi

    def _scan_trailer_top(self, held_clearance=None, y_range=None):
        """Transform the latest basemast cloud to base_link, crop to the trailer box, and return the
        top of the in-box points (the pile top), or None if too few points land in the box.
        held_clearance overrides deposit_held_clearance (e.g. smaller for an empty grapple)."""
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
        # z ceiling: fixed box top, AND below the held bundle (which hangs from the grapple into
        # the box; the pile never reaches that high, so grapple z - clearance separates them)
        clearance = self._deposit_held_clearance if held_clearance is None else held_clearance
        z_hi = float(mx[2])
        ee_z = float(self.current_ee_pos[2])
        if clearance > 0.0 and ee_z > float(mn[2]):  # skip if ee pose not yet known
            z_hi = min(z_hi, ee_z - clearance)
        # bins mode: restrict the scan to the current bin's y-slice of the trailer box.
        y_lo, y_hi = float(mn[1]), float(mx[1])
        if y_range is not None:
            y_lo = max(y_lo, float(y_range[0]))
            y_hi = min(y_hi, float(y_range[1]))
        m = ((pts_b[:, 0] >= mn[0]) & (pts_b[:, 0] <= mx[0])
             & (pts_b[:, 1] >= y_lo) & (pts_b[:, 1] <= y_hi)
             & (pts_b[:, 2] >= mn[2]) & (pts_b[:, 2] <= z_hi))
        n = int(m.sum())
        self._last_deposit_pts = pts_b[m].astype(np.float32)   # for the npz dump
        self._publish_deposit_scan(pts_b[m])   # show exactly what got sampled (base frame)
        if n < self._deposit_min_points:
            self.get_logger().warn(f"DEPOSIT: only {n} pts in trailer box below z={z_hi:.2f} "
                                   f"(< {self._deposit_min_points}).")
            return None
        # k-th highest z: robust to a few dangling/noise points setting the top
        z_in = np.sort(pts_b[m, 2])
        return float(z_in[-min(self._deposit_top_k, n)])

    def _publish_deposit_box(self):
        """Publish the trailer scan box (base frame) as a translucent orange cube for RViz."""
        bmin = np.asarray(self._trailer_bounds_min, float).copy()
        bmax = np.asarray(self._trailer_bounds_max, float).copy()
        yr = self._bin_scan_range()   # bins mode: draw only the active bin's slice
        if yr is not None:
            bmin[1], bmax[1] = float(yr[0]), float(yr[1])
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
        q_target = self.ik.solve(goal_xyz, self.current_arm)
        if q_target is None:
            self.get_logger().warn(
                f"{move.phase.name}: IK failed for goal {np.round(goal_xyz, 3).tolist()}")
            return False
        if move.yaw is None:
            goal_gc = self.current_yaw_joint
            marker_yaw = self.current_ee_yaw
        else:
            # the arm move also turns the slew; take that travel out of the yaw joint so the
            # grapple lands at move.yaw in the base frame AFTER the move
            slew_delta = float(q_target[0] - self.current_arm[0])
            goal_gc = self._grapple_yaw_to_joint(move.yaw, slew_delta=slew_delta)
            marker_yaw = move.yaw
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
        # Consume the mover's service response. Since the URDF joint-limit checking
        # landed in relative_joint_mover, a violating move is REJECTED via
        # success=False in this response; unread, that is indistinguishable from a
        # silent PLC failure and the node re-sends the same doomed request until
        # timeout. Log the reason and advance instead.
        if self._traj_future is not None and self._traj_future.done():
            try:
                resp = self._traj_future.result()
            except Exception as exc:  # noqa: BLE001
                resp = None
                self.get_logger().warn(f"relative_joint_move call failed: {exc}")
            self._traj_future = None
            if resp is not None and not resp.success:
                self.get_logger().warn(
                    f"{self.moves[self.move_idx].phase.name}: move REJECTED by "
                    f"relative_joint_mover: {resp.message}; advancing")
                self._log_timeout(self.moves[self.move_idx], True)
                self._advance_move()
                return
        # _seen_executing = the PLC actually REPORTED executing. plc_queue > 0 must NOT count:
        # it pulses during dispatch before EXECUTE, which let a converged-at-send move (tiny
        # ALIGN_YAW) pass as done 0.16 s after START; the next move's header then hit the PLC
        # mid-execution and its send failed.
        if not self._seen_executing and self.plc_state == PLC_STATE_EXECUTE:
            self._seen_executing = True
        plc_idle = (self.plc_state in (PLC_STATE_HOLD, PLC_STATE_END)
                    and self.plc_queue == 0)
        if not plc_idle:
            self._idle_since = None   # only time a CONTINUOUS idle span; a re-EXECUTE resets it
        converged = self._converged_to_goal()
        # moves too short for the 10 Hz tick to catch EXECUTE: accept idle+converged after a
        # 3 s grace instead of the queue heuristic
        done = converged and plc_idle and (self._seen_executing or self._waited_longer_than(3.0))
        if self.dry_run:
            done = True
        if done:
            self._log_reach(self.moves[self.move_idx].phase)
            if self.moves[self.move_idx].phase == Phase.LIFT_HIGH:
                self._start_stability_measurement()
            else:
                self._advance_move()
        elif not self._seen_executing and self._move_retries < 1 and self._waited_longer_than(5.0):
            # PLC never started executing: the plc_node's send to the PLC failed (seen live:
            # "Failed to send trajectory header message to PLC"). Re-send once after 5 s instead
            # of waiting out the full timeout and advancing with the arm in the wrong place.
            self._move_retries += 1
            self.get_logger().warn(
                f"{self.moves[self.move_idx].phase.name}: PLC never executed the move "
                f"(send failed?); re-sending once")
            self.seq_state = SEQ_SEND
        elif self._seen_executing and plc_idle:
            # PLC executed and is back at idle (HOLD/END, empty queue) but we're NOT converged:
            # the move is over -- it went as far as it will (fell short / PLC aborted). Don't sit
            # out the full traj_timeout. Hold traj_settle_s to let joint_states settle in case it
            # trickles into tolerance, then RE-SEND the remaining delta (up to traj_max_resends)
            # so it reaches the goal; only advance once the re-send budget is spent.
            if self._idle_since is None:
                self._idle_since = self.get_clock().now()
            elif (self.get_clock().now() - self._idle_since).nanoseconds / 1e9 > self._traj_settle_s:
                phase = self.moves[self.move_idx].phase.name
                if self._fellshort_retries < self._traj_max_resends:
                    self._fellshort_retries += 1
                    self.get_logger().warn(
                        f"{phase}: PLC finished but arm off-tolerance (fell short / aborted); "
                        f"re-sending remaining delta "
                        f"(attempt {self._fellshort_retries}/{self._traj_max_resends})")
                    self.seq_state = SEQ_SEND
                else:
                    self.get_logger().warn(
                        f"{phase}: still off-tolerance after {self._traj_max_resends} re-sends; "
                        f"advancing")
                    self._log_timeout(self.moves[self.move_idx], plc_idle)
                    self._advance_move()
        elif self._waited_longer_than(self._traj_timeout):
            self._log_timeout(self.moves[self.move_idx], plc_idle)
            self._advance_move()

    def _start_stability_measurement(self):
        if not self._measure_stability_enabled:
            self._advance_move()
            return
        if self._stability_goal_sent:
            return

        self._reset_stability_state(keep_last_result=True)
        self._stability_goal_sent = True
        self._stability_measurement_start = self.get_clock().now()
        self._wait_start = self._stability_measurement_start
        self.seq_state = SEQ_WAIT_STABILITY

        if not self.stability_client.server_is_ready():
            self._handle_stability_failure("measure_stability action server not available")
            return

        goal = MeasureStability.Goal()
        goal.measurement_start = self._stability_measurement_start.to_msg()
        goal.timeout_sec = self._measure_stability_timeout_s
        goal.required_window_sec = self._measure_stability_required_window_s
        goal.required_dwell_sec = self._measure_stability_required_dwell_s
        goal.min_samples = self._measure_stability_min_samples
        goal.max_sample_gap_sec = self._measure_stability_max_gap_s
        goal.stability_variance_threshold_rad2 = self._measure_stability_stability_var
        goal.gravity_variance_threshold_rad2 = self._measure_stability_gravity_var
        goal.grapple_variance_threshold_rad2 = self._measure_stability_grapple_var
        goal.reward_exponent = self._measure_stability_reward_exponent

        self._stability_send_goal_future = self.stability_client.send_goal_async(
            goal,
            feedback_callback=self._stability_feedback_cb,
        )
        self.get_logger().info(
            "LIFT_HIGH: reached; measuring stability before CARRY_HOME "
            f"(timeout={goal.timeout_sec:.1f}s, window={goal.required_window_sec:.2f}s, "
            f"dwell={goal.required_dwell_sec:.2f}s)")

    def _check_stability_done(self):
        if self._stability_result_handled or self._stability_terminal_failure:
            return

        if self._stability_send_goal_future is not None:
            if not self._stability_send_goal_future.done():
                if self._waited_longer_than(self._measure_stability_goal_response_timeout_s):
                    self._handle_stability_failure("measure_stability goal response timeout")
                return
            try:
                self._stability_goal_handle = self._stability_send_goal_future.result()
            except Exception as exc:  # noqa: BLE001
                self._handle_stability_failure(f"measure_stability send failed: {exc}")
                return
            self._stability_send_goal_future = None
            if not self._stability_goal_handle.accepted:
                self._handle_stability_failure("measure_stability goal rejected")
                return
            self._stability_result_future = self._stability_goal_handle.get_result_async()

        if self._stability_result_future is None:
            return
        if not self._stability_result_future.done():
            return

        try:
            wrapped = self._stability_result_future.result()
        except Exception as exc:  # noqa: BLE001
            self._handle_stability_failure(f"measure_stability result failed: {exc}")
            return

        result = wrapped.result
        if wrapped.status == GoalStatus.STATUS_SUCCEEDED and result.success:
            self._last_stability_result = result
            self._log_stability_success(result)
            self._stability_result_handled = True
            self._reset_stability_state(keep_last_result=True)
            self._advance_move()
            return

        status = self._goal_status_name(wrapped.status)
        reason = result.failure_reason or status
        self._handle_stability_failure(f"measure_stability {status}: {reason}",
                                       result=result)

    def _stability_feedback_cb(self, feedback_msg):
        feedback = feedback_msg.feedback
        if feedback.status and feedback.status != "candidate_settled":
            self.get_logger().debug(
                "MeasureStability feedback: "
                f"{feedback.status}, elapsed={feedback.elapsed_sec:.2f}s, "
                f"samples={feedback.window_sample_count}")

    def _log_stability_success(self, result):
        self.get_logger().info(
            "LIFT_HIGH stability measurement complete: "
            f"t_settle={result.t_settle_sec:.3f}s "
            f"theta_max={math.degrees(result.theta_max_rad):.3f}deg "
            f"theta_rms={math.degrees(result.theta_rms_rad):.3f}deg "
            f"theta_settle={math.degrees(result.theta_settle_rad):.3f}deg "
            f"theta_mean_vectors={math.degrees(result.theta_from_mean_vectors_rad):.3f}deg "
            f"samples={result.measurement_sample_count} "
            f"accepted_samples={result.accepted_window_sample_count} "
            f"reward={result.settle_reward:.4f}")
        if self._debug_save_dir:
            try:
                os.makedirs(self._debug_save_dir, exist_ok=True)
                rec = {
                    "cycle": self.cycle_count + 1,
                    "phase": "LIFT_HIGH",
                    "valid": True,
                    "t_wall": self.get_clock().now().nanoseconds * 1e-9,
                    "t_settle_sec": float(result.t_settle_sec),
                    "theta_max_rad": float(result.theta_max_rad),
                    "theta_rms_rad": float(result.theta_rms_rad),
                    "theta_settle_rad": float(result.theta_settle_rad),
                    "theta_from_mean_vectors_rad": float(result.theta_from_mean_vectors_rad),
                    "settle_dot_product": float(result.settle_dot_product),
                    "settle_reward": float(result.settle_reward),
                    "measurement_sample_count": int(result.measurement_sample_count),
                    "accepted_window_sample_count": int(result.accepted_window_sample_count),
                    # Full raw progression so the metric can be recomputed offline
                    "sample_offsets_sec": [round(float(t), 4) for t in result.sample_offsets_sec],
                    "sample_theta_rad": [round(float(th), 5) for th in result.sample_theta_rad],
                }
                png = self._render_stability_plot(result, settled=True)
                if png:
                    rec["png"] = png
                with open(os.path.join(self._debug_save_dir, "stability.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                self._episode_record["stability"] = {
                    k: v for k, v in rec.items()
                if k not in ("cycle", "phase", "sample_offsets_sec", "sample_theta_rad")}
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"stability log append failed: {exc}")

    def _handle_stability_failure(self, reason, result=None):
        if self._stability_result_handled or self._stability_terminal_failure:
            return
        # A timeout/abort result still carries a best-effort estimate over the
        # last window (settle_reward computed from unsettled samples); keep it
        # so the run has a number, flagged via valid=False in the log.
        best_effort = (result is not None
                       and math.isfinite(result.settle_reward))
        self._last_stability_result = result if best_effort else None
        self._stability_result_handled = True
        if best_effort:
            self.get_logger().warn(
                "LIFT_HIGH stability measurement did not settle: "
                f"{reason}; best-effort reward={result.settle_reward:.4f} "
                f"theta_settle={math.degrees(result.theta_settle_rad):.3f}deg "
                f"theta_max={math.degrees(result.theta_max_rad):.3f}deg "
                f"(over last {result.accepted_window_sample_count} samples); "
                f"failure_policy={self._measure_stability_failure_policy}")
        else:
            self.get_logger().warn(
                "LIFT_HIGH stability measurement invalid: "
                f"{reason}; failure_policy={self._measure_stability_failure_policy}")
        self._append_stability_failure_log(reason, result if best_effort else None)
        self._cancel_stability_goal(reason)
        self._reset_stability_state(keep_last_result=True)
        if self._measure_stability_failure_policy == "advance":
            self._advance_move()
        else:
            self._stability_terminal_failure = True
            self.seq_state = SEQ_WAIT_STABILITY

    def _append_stability_failure_log(self, reason, result=None):
        if not self._debug_save_dir:
            return
        try:
            os.makedirs(self._debug_save_dir, exist_ok=True)
            rec = {
                "cycle": self.cycle_count + 1,
                "phase": "LIFT_HIGH",
                "valid": False,
                "reason": reason,
                "t_wall": self.get_clock().now().nanoseconds * 1e-9,
            }
            if result is not None:
                rec.update({
                    "best_effort": True,
                    "theta_max_rad": float(result.theta_max_rad),
                    "theta_rms_rad": float(result.theta_rms_rad),
                    "theta_settle_rad": float(result.theta_settle_rad),
                    "theta_from_mean_vectors_rad": float(
                        result.theta_from_mean_vectors_rad),
                    "settle_dot_product": float(result.settle_dot_product),
                    "settle_reward": float(result.settle_reward),
                    "measurement_sample_count": int(
                        result.measurement_sample_count),
                    "window_sample_count": int(
                        result.accepted_window_sample_count),
                    "stability_variance_rad2": float(
                        result.final_stability_variance_rad2),
                    "gravity_variance_rad2": float(
                        result.final_gravity_variance_rad2),
                    "grapple_variance_rad2": float(
                        result.final_grapple_variance_rad2),
                    # Full raw progression so the metric can be recomputed offline
                    "sample_offsets_sec": [round(float(t), 4) for t in result.sample_offsets_sec],
                    "sample_theta_rad": [round(float(th), 5) for th in result.sample_theta_rad],
                })
                png = self._render_stability_plot(result, settled=False)
                if png:
                    rec["png"] = png
            with open(os.path.join(self._debug_save_dir, "stability.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            self._episode_record["stability"] = {
                k: v for k, v in rec.items()
                if k not in ("cycle", "phase", "sample_offsets_sec", "sample_theta_rad")}
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"stability failure log append failed: {exc}")

    def _cancel_stability_goal(self, reason):
        if self._stability_goal_handle is not None:
            try:
                self.get_logger().info(f"Cancelling MeasureStability goal: {reason}")
                self._stability_goal_handle.cancel_goal_async()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"MeasureStability cancel failed: {exc}")

    def _reset_stability_state(self, keep_last_result=False):
        last = self._last_stability_result if keep_last_result else None
        self._stability_send_goal_future = None
        self._stability_goal_handle = None
        self._stability_result_future = None
        self._stability_goal_sent = False
        self._stability_result_handled = False
        self._stability_terminal_failure = False
        self._stability_measurement_start = None
        self._last_stability_result = last

    @staticmethod
    def _goal_status_name(status):
        names = {
            GoalStatus.STATUS_UNKNOWN: "unknown",
            GoalStatus.STATUS_ACCEPTED: "accepted",
            GoalStatus.STATUS_EXECUTING: "executing",
            GoalStatus.STATUS_CANCELING: "canceling",
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "canceled",
            GoalStatus.STATUS_ABORTED: "aborted",
        }
        return names.get(status, f"status_{status}")

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
        # the service future resolves on command ACCEPT; hold the phase for the jaw motion too
        move = self.moves[self.move_idx]
        dwell = self._grip_close_dwell_s if move.gripper == GRIPPER_CLOSE else self._grip_open_dwell_s
        done = self.dry_run or (self._grapple_future is not None
                                and self._grapple_future.done()
                                and self._waited_longer_than(dwell))
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
        self._move_retries = 0
        self._fellshort_retries = 0
        if self.move_idx >= len(self.moves):
            if self._prescan_active:
                # pre-first-cycle trailer scan: not a grasp cycle, don't count it
                self._prescan_active = False
                self._prescan_done = True
                self._episode_record = {}
                self.get_logger().info("PRESCAN complete; starting cycle 1")
            else:
                self.cycle_count += 1
                self.get_logger().info(f"Cycle {self.cycle_count} complete")
                self._append_episode_log(self.cycle_count)
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
            os.makedirs(self._debug_save_dir, exist_ok=True)
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

    def _save_debug(self, obs, x, y, z, yaw, z_raw=None, support=None):
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
                     bounds_min=np.asarray(self._crop_min, np.float32),
                     bounds_max=np.asarray(self._crop_max, np.float32),
                     rack_y_shift=np.float32(self._rack_y_shift))
            self.get_logger().info(f"saved policy debug -> {fn} ({pts.shape[0]} pts)")
            self._append_decision_log(x, y, z, yaw, kind="gaze", npz=os.path.basename(fn), z_raw=z_raw, support=support)
            if self._debug_save_jpg:
                self._render_iso_jpg(pts, x, y, z, yaw, fn[:-4] + ".jpg")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"debug save failed: {e}")

    def _append_decision_log(self, x, y, z, yaw, kind="gaze", npz=None, z_raw=None, support=None):
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
            if z_raw is not None and z_raw != z:
                rec["z_raw"] = float(z_raw)   # pre-bed-floor command (clamped by the envelope)
            if support is not None:
                rec["support"] = int(support)  # observed points within 0.5 m of the target xy
            with open(os.path.join(self._debug_save_dir, "decisions.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            # "gaze" is the grasp-target decision made at the gaze pose
            ep = {k: rec[k] for k in ("target", "npz", "t_wall", "video_frame")}
            if kind == "deposit" and self._deposit_mode == "bins":
                ep["bin"] = int(self._current_bin)
            self._episode_record["grasp" if kind == "gaze" else kind] = ep
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"decision log append failed: {e}")

    def _append_episode_log(self, cycle, partial=False):
        """Flush the consolidated per-grasp record (grasp target + deposit + stability)
        to <run_dir>/episode.jsonl: one line per completed cycle, so a pile-clearing
        run reads as a sequential episode without joining the per-topic jsonl files."""
        if not self._debug_save_dir:
            self._episode_record = {}
            return
        if not self._episode_record:
            return
        try:
            os.makedirs(self._debug_save_dir, exist_ok=True)
            rec = {
                "cycle": cycle,
                "t_wall": self.get_clock().now().nanoseconds * 1e-9,
            }
            if partial:
                rec["partial"] = True
            rec.update(self._episode_record)
            with open(os.path.join(self._debug_save_dir, "episode.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"episode log append failed: {e}")
        finally:
            self._episode_record = {}

    def _render_stability_plot(self, result, settled):
        """Angle-vs-time PNG for one LIFT_HIGH stability measurement: grapple tilt
        theta with the accepted settle window shaded, plus the rolling std the
        detector gates on vs the collection threshold. Returns the basename."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            t = np.asarray(result.sample_offsets_sec, dtype=np.float64)
            th = np.degrees(np.asarray(result.sample_theta_rad, dtype=np.float64))
            if t.size < 2:
                return None
            meas0 = result.measurement_start.sec + result.measurement_start.nanosec * 1e-9
            w0 = (result.accepted_window_start.sec
                  + result.accepted_window_start.nanosec * 1e-9 - meas0)
            w1 = (result.accepted_window_end.sec
                  + result.accepted_window_end.nanosec * 1e-9 - meas0)
            win_s = self._measure_stability_required_window_s
            thr_deg = math.degrees(math.sqrt(self._measure_stability_stability_var))
            # rolling std of theta over the same window length the detector gates on
            # (the theta-variance gate; the two vector-scatter gates are not plotted)
            std = np.full(t.shape, np.nan)
            for i in range(t.size):
                j = int(np.searchsorted(t, t[i] - win_s))
                if i + 1 - j >= 2 and t[i] - t[j] >= win_s * 0.9:
                    std[i] = np.std(th[j:i + 1])

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6),
                                           gridspec_kw={"height_ratios": [2, 1]})
            ax1.plot(t, th, lw=0.8, color="tab:blue")
            if w1 > w0 > 0.0:
                ax1.axvspan(w0, w1, color="tab:green" if settled else "tab:orange",
                            alpha=0.25,
                            label="settle window" if settled else "last window (unsettled)")
            if math.isfinite(result.theta_settle_rad):
                ax1.axhline(math.degrees(result.theta_settle_rad), ls="--", lw=0.8,
                            color="tab:gray", label="theta_settle")
            if settled and math.isfinite(result.t_settle_sec):
                ax1.axvline(result.t_settle_sec, ls=":", lw=1.0, color="tab:green",
                            label="t_settle")
            status = "settled" if settled else "NOT settled"
            ax1.set_title(f"cycle {self.cycle_count + 1} LIFT_HIGH stability: {status}")
            ax1.set_ylabel("grapple tilt theta [deg]")
            ax1.legend(loc="upper right", fontsize=8, framealpha=0.6)
            # score box: the grasp's stability score front and center, color-coded
            # by verdict (legend is pinned to the opposite corner to avoid overlap)
            lines = []
            if math.isfinite(result.settle_reward):
                lines.append(f"score  {result.settle_reward:.4f}"
                             + ("" if settled else "  (best-effort)"))
            if math.isfinite(result.theta_settle_rad):
                lines.append(f"theta_settle  {math.degrees(result.theta_settle_rad):.2f} deg")
            if math.isfinite(result.theta_max_rad):
                lines.append(f"theta_max  {math.degrees(result.theta_max_rad):.2f} deg")
            if settled and math.isfinite(result.t_settle_sec):
                lines.append(f"t_settle  {result.t_settle_sec:.2f} s")
            if lines:
                ax1.text(0.02, 0.97, "\n".join(lines), transform=ax1.transAxes,
                         va="top", ha="left", fontsize=9, family="monospace",
                         bbox={"boxstyle": "round,pad=0.4", "alpha": 0.85,
                               "facecolor": "honeydew" if settled else "seashell",
                               "edgecolor": "tab:green" if settled else "tab:orange"})

            ax2.plot(t, std, lw=0.8, color="tab:purple")
            ax2.axhline(thr_deg, ls="--", lw=0.8, color="tab:red",
                        label=f"collect threshold ({thr_deg:.2f} deg std)")
            ax2.set_yscale("log")
            ax2.margins(y=0.2)
            ax2.set_ylabel(f"rolling std over {win_s:.1f}s [deg]")
            ax2.set_xlabel("time since hover-up [s]")
            ax2.legend(loc="best", fontsize=8, framealpha=0.6)
            fig.tight_layout()
            fn = f"stability_{self.cycle_count + 1:03d}.png"
            fig.savefig(os.path.join(self._debug_save_dir, fn), dpi=110)
            plt.close(fig)
            return fn
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"stability plot failed: {e}")
            return None

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

    def _render_deposit_jpg(self, pts, x, y, z, yaw, path):
        """Render the trailer scan cloud + deposit target as a top-view JPG with the trailer box and
        (bins mode) the compartment dividers + active bin, so a dry run leaves an openable image per
        cycle. Top-down (base y across, base x up) is the most legible view for the bin walk."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except Exception as e:  # noqa: BLE001
            if not hasattr(self, "_mpl_warn"):
                self.get_logger().warn(f"deposit JPG disabled (matplotlib unavailable: {e})")
                self._mpl_warn = True
            return
        try:
            bmin = np.asarray(self._trailer_bounds_min, float)
            bmax = np.asarray(self._trailer_bounds_max, float)
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            if pts.shape[0]:
                ax.scatter(pts[:, 1], pts[:, 0], s=3, c=pts[:, 2], cmap="viridis", alpha=0.7)
            # trailer box outline
            ax.add_patch(Rectangle((bmin[1], bmin[0]), bmax[1] - bmin[1], bmax[0] - bmin[0],
                                   fill=False, ec="0.4", lw=1.5))
            if self._deposit_mode == "bins":
                edges = np.asarray(self._bin_edges, float)
                mgn = self._deposit_bin_scan_margin
                for e in edges:
                    ax.axvline(e, color="k", lw=1.5, ls="--")          # poles
                for i in range(self._num_bins()):
                    y0, y1 = edges[i] + mgn, edges[i + 1] - mgn        # inset scan box
                    if i in self._bin_skip:
                        ec, lw, ls = "0.6", 1.5, ":"                    # skipped (too close): gray dotted
                    elif i == self._current_bin:
                        ec, lw, ls = "lime", 2, "-"                     # active target
                    else:
                        ec, lw, ls = "orange", 2, "-"                   # other fillable bins
                    ax.add_patch(Rectangle((y0, bmin[0]), y1 - y0, bmax[0] - bmin[0],
                                           fill=False, ec=ec, lw=lw, ls=ls))
            # deposit target + perpendicular bundle line (along base x)
            ax.plot(y, x, "r*", ms=18, mec="k")
            ax.plot([y, y + 0.6 * math.sin(yaw)], [x, x + 0.6 * math.cos(yaw)], "r-", lw=2)
            ax.set_xlim(bmin[1], bmax[1]); ax.set_ylim(bmin[0] - 0.15, bmax[0] + 0.15)
            ax.set_xlabel("base Y (trailer length)"); ax.set_ylabel("base X (width)")
            bin_txt = f" bin {self._current_bin}" if self._deposit_mode == "bins" else ""
            ax.set_title(f"deposit cycle {self.cycle_count + 1}{bin_txt}: "
                         f"target ({x:.2f}, {y:.2f}, {z:.2f}) yaw {math.degrees(yaw):.0f} deg")
            try:
                fig.savefig(path, dpi=90, bbox_inches="tight")
            except ValueError:
                path = path[:-4] + ".png"
                fig.savefig(path, dpi=90, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"deposit render failed: {e}")

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
                if os.path.exists(path):
                    # resumed run: keep the earlier segment instead of clobbering it
                    k = 1
                    while os.path.exists(os.path.join(
                            self._debug_save_dir, f"basemast_run_part{k}.mp4")):
                        k += 1
                    os.rename(path, os.path.join(
                        self._debug_save_dir, f"basemast_run_part{k}.mp4"))
                    self.get_logger().info(
                        f"existing basemast_run.mp4 kept as basemast_run_part{k}.mp4")
                # Start the ffmpeg encoder in its own session: terminal Ctrl-C sends
                # SIGINT to the whole foreground process group, and ffmpeg dying with
                # the node truncates the mp4 (no moov index -> unplayable) even when
                # python's close() reports success. Every truncated basemast so far
                # died this way. Detached, only the node gets the signal and close()
                # finalizes properly.
                _orig_popen = subprocess.Popen

                def _detached_popen(*a, **k):
                    k.setdefault("start_new_session", True)
                    return _orig_popen(*a, **k)

                subprocess.Popen = _detached_popen
                try:
                    self._video_writer = imageio.get_writer(
                        path, fps=self._video_fps, macro_block_size=None)
                finally:
                    subprocess.Popen = _orig_popen
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
        self._cancel_stability_goal("shutdown")
        if self._stability_server_proc is not None:
            try:
                self._stability_server_proc.terminate()
                self._stability_server_proc.wait(timeout=3.0)
            except Exception:  # noqa: BLE001
                try:
                    self._stability_server_proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._stability_server_proc = None
        # cut-short cycle: flush whatever was gathered, marked partial
        self._append_episode_log(self.cycle_count + 1, partial=True)
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
        if not bool(self.get_parameter("stitch_review").value):
            self.get_logger().info(
                "review_video: stitch_review is off; render on demand with "
                f"render_run_video.py {self._debug_save_dir}")
            return
        script = self._render_video_script
        if not os.path.isfile(script):
            self.get_logger().warn(f"review_video: script not found at {script}; skipping.")
            return
        if not glob.glob(os.path.join(self._debug_save_dir, "policy_debug_*.npz")):
            self.get_logger().warn("review_video: no policy_debug_*.npz in run dir; skipping.")
            return
        # resumed run: keep an earlier session's review instead of clobbering it
        # (same _partN scheme as basemast_run.mp4)
        review = os.path.join(self._debug_save_dir, "review.mp4")
        if os.path.exists(review):
            k = 1
            while os.path.exists(os.path.join(
                    self._debug_save_dir, f"review_part{k}.mp4")):
                k += 1
            os.rename(review, os.path.join(
                self._debug_save_dir, f"review_part{k}.mp4"))
            self.get_logger().info(
                f"review_video: existing review.mp4 kept as review_part{k}.mp4")
        self.get_logger().info(
            f"review_video: stitching review.mp4 from {self._debug_save_dir} ... (Ctrl-C to skip)")
        try:
            cmd = ["python3", script, self._debug_save_dir]
            speed = float(self.get_parameter("review_video_speed").value)
            if speed > 1.0:
                cmd += ["--speed", str(int(round(speed)))]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
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
        bmin = np.asarray(self._crop_min, float)
        bmax = np.asarray(self._crop_max, float)
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
