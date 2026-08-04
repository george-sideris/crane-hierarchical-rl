#!/usr/bin/env python3
"""ROS2 node: depth -> policy -> FSM -> EE commands.

Subscribes to /zedx/depth + camera info, runs the policy on a downsampled
point cloud, drives a 10-phase pick-and-place FSM, and forwards EE targets
to the JV controller via the /crane/set_target service.

────────────────────────────────────────────────────────────────────────
Sim test (three terminals, all inside the Docker container — source
env_setup.sh first in each terminal that uses ROS2):

    Terminal 1:  PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH \\
                 ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/sim_ros2_env.py
    Terminal 2:  /usr/bin/python3 /workspace/crane_testbed/scripts/ros2/jv_controller_node.py
    Terminal 3:  /usr/bin/python3 /workspace/crane_testbed/scripts/ros2/crane_policy_node.py \\
                     --policy_type bcrl --checkpoint /path/to/model.pt

────────────────────────────────────────────────────────────────────────
Real crane deployment

The policy node is hardware-agnostic — it just consumes ROS2 topics and
calls a service. To deploy on the real testbed, the integrator must
provide the equivalents of what sim_ros2_env.py publishes/subscribes:

    Topic                       Direction   Provider on real testbed
    -----                       ---------   -----------------------
    /zedx/depth                 PUB         ZED X driver (32FC1, meters)
    /zedx/camera_info           PUB         ZED X driver
    /tf_static                  PUB         crane URDF + ZED extrinsic
                                            (frames: world, crane_base,
                                            zedx_camera)
    /crane/joint_states         PUB         crane PLC/driver
    /crane/ee_pos_base          PUB         crane PLC/driver (basegrapple
                                            position in crane_base frame)
    /crane/grapple_yaw_base     PUB         crane PLC/driver (basegrapple
                                            yaw in crane_base frame, rad)
    /crane/action_bounds        PUB         static publisher with
                                            workspace bounds for the
                                            actual rack location
                                            (Float32MultiArray, 6 floats:
                                            min_x, min_y, min_z,
                                            max_x, max_y, max_z)
    /crane/joint_command        SUB         crane PLC/driver (consumes
                                            position + velocity targets)

Once those are running, the deployment side is identical to the sim
flow above — just skip Terminal 1 (sim_ros2_env.py) and run Terminals
2 and 3 against the real topics:

    Terminal A:  python3 jv_controller_node.py
    Terminal B:  python3 crane_policy_node.py \\
                     --policy_type bcrl --checkpoint /path/to/model.pt

Notes:
- The frame_id used by the policy node defaults to "crane_base" (see
  the --base_frame parameter); the URDF / static_transform_publisher
  must match.
- The deployed checkpoint expects 1024-point PCDs in the crane_base
  frame, depth-filtered to [1.0, 10.0] m. process_depth() in
  pointcloud_pipeline.py handles the conversion.
- Action bounds must match what the policy was trained against. For
  this checkpoint they are the rack-centered AABB in base frame:
  min ≈ (-5.0, -0.75, -1.373), max ≈ (-3.0, 4.59, -0.373). If your
  rack location differs, the bounds need to shift accordingly (see
  _compute_action_space_bounds in crane_rl_env_full.py for the
  formula) and the policy may need re-training.
"""

import argparse
import math
import numpy as np
import torch

import rclpy
from rclpy.node import Node
import csv
import time as pytime

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64, Float32MultiArray, String, Bool

from pointcloud_pipeline import process_depth
from fsm import CraneFSM, FSMConfig, Phase
from policy_loader import load_policy
from sim_interface.srv import SetTarget, CheckTargetReached


class CranePolicyNode(Node):
    def __init__(self, policy_type_override=None, checkpoint_override=None,
                 dry_run_override=False):
        super().__init__("crane_policy_node")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("policy_type", policy_type_override or "heuristic")
        self.declare_parameter("checkpoint_path", checkpoint_override or "")
        self.declare_parameter("num_points", 1024)
        self.declare_parameter("depth_range_min", 1.0)
        self.declare_parameter("depth_range_max", 10.0)
        self.declare_parameter("cossin", True)
        self.declare_parameter("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.declare_parameter("tick_rate", 10.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("max_cycles", 30)

        # Workspace bounds (base frame, meters)
        self.declare_parameter("bounds_min", [-5.0, -0.75, -1.373])
        self.declare_parameter("bounds_max", [-3.0, 4.59, -0.373])

        # Camera extrinsics (world frame) — overridden by TF lookup at runtime
        self.declare_parameter("cam_pos", [5.0, -1.0, 3.0])
        self.declare_parameter("cam_rpy", [0.0, 0.0, 0.0])

        # Crane base position (world frame) — overridden by TF lookup at runtime
        self.declare_parameter("base_pos", [0.0, 0.0, 0.0])
        self.declare_parameter("base_rpy", [0.0, 0.0, 0.0])

        # FSM config overrides
        self.declare_parameter("hover_clear", 2.5)
        self.declare_parameter("approach_above", 0.6)
        self.declare_parameter("ee_tolerance", 0.10)
        # Crane base frame: trailer is in +Y. Z=3.0 (RL env _home_rear is 2.0
        # but bumped +1 m for better trailer-pole clearance during deposition).
        self.declare_parameter("drop_position", [0.0, 2.2, 3.0])

        # Read parameters
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

        # Camera/base transforms — stored as quaternions (wxyz) for direct use with pipeline
        # Defaults match sim's printed values; can be overridden by TF or sim bridge
        self.cam_pos = np.array(self.get_parameter("cam_pos").value, dtype=np.float32)
        self.cam_quat = np.array([0.6124, 0.3536, 0.3536, 0.6124], dtype=np.float32)  # OpenGL convention from camera config

        self.base_pos = np.array([6.0, -2.92, 1.423], dtype=np.float32)  # sim's base_pos
        self.base_quat = np.array([1, 0, 0, 0], dtype=np.float32)  # sim's base has no rotation

        # ── Policy ────────────────────────────────────────────────────
        ckpt = checkpoint if checkpoint else None
        self.policy = load_policy(policy_type, ckpt, bounds_min, bounds_max, cossin, device)
        self.get_logger().info(f"Policy: {policy_type}" +
                               (f" from {checkpoint}" if checkpoint else ""))

        # ── FSM ───────────────────────────────────────────────────────
        fsm_cfg = FSMConfig(
            hover_clear=self.get_parameter("hover_clear").value,
            approach_above=self.get_parameter("approach_above").value,
            ee_tolerance=self.get_parameter("ee_tolerance").value,
            drop_position=self.get_parameter("drop_position").value,
        )
        self.fsm = CraneFSM(fsm_cfg)

        # ── State ─────────────────────────────────────────────────────
        self.intrinsics = None
        self.latest_depth = None
        self.current_ee_pos = [0.0, 0.0, 0.0]
        self.current_ee_yaw = 0.0       # basegrapple yaw in base frame (from sim)
        self.current_yaw_joint = 0.0    # yaw joint position (from /crane/joint_states)
        # Cache for grapple→joint yaw conversion. Recompute only when cmd.yaw
        # changes (e.g. on ALIGN_YAW or ALIGN_HOME_YAW entry); hold the cached
        # joint value through downstream phases so descent dynamics don't cause
        # the joint to chase basegrapple drift and visibly spin.
        self._last_cmd_yaw = None
        self._cached_joint_yaw = 0.0
        self.current_gripper = 0.0
        self.waiting_for_target = True
        self.cycle_count = 0

        # ── Input source setup ────────────────────────────────────────
        self._setup_ros2_source()

        # ── Publishers ────────────────────────────────────────────────
        self.pub_grasp_target = self.create_publisher(PoseStamped, "/crane/grasp_target", 10)
        self.pub_ee_cmd = self.create_publisher(PoseStamped, "/crane/ee_command", 10)
        self.pub_gripper = self.create_publisher(Float64, "/crane/gripper_command", 10)
        self.pub_fsm_state = self.create_publisher(String, "/crane/fsm_state", 10)
        self.pub_policy_target = self.create_publisher(PoseStamped, "/crane/policy_target", 10)

        # ── Service client to JV controller ───────────────────────────
        self.set_target_client = self.create_client(SetTarget, "/crane/set_target")
        self.check_target_client = self.create_client(CheckTargetReached, "/crane/is_target_reached")

        # Subscribe to cycle_complete from sequencer
        self.create_subscription(Bool, "/crane/cycle_complete", self._cycle_complete_cb, 10)

        # ── CSV logger ────────────────────────────────────────────────
        log_path = "/workspace/crane_testbed/scripts/ros2/policy_node_log.csv"
        self._log_file = open(log_path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            "time", "phase", "cycle",
            "ee_x", "ee_y", "ee_z",
            "target_x", "target_y", "target_z",
            "pos_error", "yaw", "yaw_target", "yaw_error",
            "gripper", "gripper_target",
        ])
        self._log_start = pytime.time()
        self.get_logger().info(f"Logging to {log_path}")

        # ── Timer ─────────────────────────────────────────────────────
        self.create_timer(1.0 / tick_rate, self._tick)

        self.get_logger().info(
            f"Started (tick={tick_rate}Hz, dry_run={self.dry_run})")

    # ── Input: ROS2 topics ────────────────────────────────────────────

    def _setup_ros2_source(self):
        """Subscribe to ROS2 topics from real crane sensors."""
        try:
            from cv_bridge import CvBridge
            self._cv_bridge = CvBridge()
        except ImportError:
            self._cv_bridge = None
            self.get_logger().warn("cv_bridge not available, will decode depth manually")

        self.create_subscription(Image, "/zedx/depth", self._depth_cb, 10)
        self.create_subscription(CameraInfo, "/zedx/camera_info", self._caminfo_cb, 10)
        # Use sim's ground-truth EE position (basegrapple in base frame)
        self.create_subscription(PointStamped, "/crane/ee_pos_base", self._ee_pos_base_cb, 10)
        # Sim publishes basegrapple yaw in base frame — used for FSM yaw checks AND
        # for converting policy yaw target → yaw-joint command (see _grapple_yaw_to_joint).
        self.create_subscription(Float64, "/crane/grapple_yaw_base", self._grapple_yaw_base_cb, 10)
        # joint_states provides current yaw joint position — needed by _grapple_yaw_to_joint
        from sensor_msgs.msg import JointState
        self.create_subscription(JointState, "/crane/joint_states", self._joint_states_cb, 10)
        # JV controller's FK estimate (legacy, kept for /crane/ee_state debugging only)
        self.create_subscription(PoseStamped, "/crane/ee_state", self._ee_state_cb, 10)
        self.create_subscription(Float64, "/crane/gripper_state", self._gripper_state_cb, 10)
        # Sim publishes the policy's training-time action bounds on connect
        # (TRANSIENT_LOCAL latched). Subscribe and override the hardcoded
        # defaults — without this the decoded x/y/z is wrong and the crane
        # never reaches the right grasp height.
        from rclpy.qos import QoSProfile, DurabilityPolicy
        bounds_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float32MultiArray, "/crane/action_bounds",
                                 self._action_bounds_cb, bounds_qos)

        # TF listener for camera and crane base transforms
        # Sim publishes correct ROS-convention camera quaternion via rclpy
        # Real crane: ZED X driver publishes TF natively
        try:
            from tf2_ros import Buffer, TransformListener
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            self._tf_available = True
            self.get_logger().info("TF listener started")
        except ImportError:
            self._tf_available = False
            self.get_logger().warn("tf2_ros not available, using parameter defaults")

        self.declare_parameter("camera_frame", "zedx_camera")
        self.declare_parameter("base_frame", "crane_base")
        self.declare_parameter("world_frame", "world")
        self._camera_frame = self.get_parameter("camera_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._world_frame = self.get_parameter("world_frame").value
        self._tf_initialized = False

        self.get_logger().info("ROS2 source: subscribing to /zedx/depth, /crane/ee_state, /tf_static")

    def _depth_cb(self, msg):
        if self._cv_bridge:
            self.latest_depth = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        else:
            self.latest_depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)

    def _caminfo_cb(self, msg):
        if self.intrinsics is None:
            K = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.intrinsics = K
            self.get_logger().info(f"Camera intrinsics: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")

    def _ee_pos_base_cb(self, msg):
        """Ground-truth basegrapple position from sim — used for FSM transitions."""
        self.current_ee_pos = [msg.point.x, msg.point.y, msg.point.z]

    def _action_bounds_cb(self, msg):
        """Override hardcoded action bounds with sim-published values so the
        policy's tanh outputs decode to the same x/y/z range it was trained on.
        Message: 6 floats = [min_x, min_y, min_z, max_x, max_y, max_z]."""
        if len(msg.data) != 6:
            self.get_logger().warn(f"Bad action_bounds length: {len(msg.data)}")
            return
        new_min = np.array(msg.data[0:3], dtype=np.float32)
        new_max = np.array(msg.data[3:6], dtype=np.float32)
        if hasattr(self.policy, "bounds_min"):
            old_min = self.policy.bounds_min.copy()
            old_max = self.policy.bounds_max.copy()
            self.policy.bounds_min = new_min
            self.policy.bounds_max = new_max
            self.get_logger().info(
                f"Action bounds updated from sim: min {old_min.round(3).tolist()} -> "
                f"{new_min.round(3).tolist()}, max {old_max.round(3).tolist()} -> "
                f"{new_max.round(3).tolist()}")

    def _grapple_yaw_base_cb(self, msg):
        """Ground-truth basegrapple yaw in base frame — used for FSM and conversion."""
        self.current_ee_yaw = float(msg.data)

    def _joint_states_cb(self, msg):
        """Track current yaw joint position for grapple→joint yaw conversion."""
        try:
            idx = list(msg.name).index("lowerpassive_to_basegrapple")
            self.current_yaw_joint = float(msg.position[idx])
        except (ValueError, IndexError):
            pass

    def _ee_state_cb(self, msg):
        """FK-based EE estimate from JV controller — kept for legacy logging only.
        Yaw is now sourced from /crane/grapple_yaw_base (sim ground truth)."""
        pass

    def _grapple_yaw_to_joint(self, target_grapple_yaw_b: float) -> float:
        """Convert desired basegrapple yaw (base frame) → yaw joint position.

        Ports crane_rl_env_full.py:_convert_grapple_yaw_to_joint_position.
        Required because the policy outputs basegrapple-frame yaw but the
        jv_controller expects a yaw-joint target — these are NOT the same
        when the lower passive link is rotated relative to the base.

        The returned joint position is NOT wrapped to [-π, π]; it is chosen
        as the candidate numerically closest to current_yaw_joint. This
        prevents the PD controller from taking the long way around if the
        joint has accumulated past ±π.
        """
        def _wrap(a):
            return (a + math.pi) % (2.0 * math.pi) - math.pi
        # Shortest rotation needed in basegrapple frame
        rotation_diff = _wrap(target_grapple_yaw_b - self.current_ee_yaw)
        base = self.current_yaw_joint + rotation_diff
        # Candidates: direct, ±180° (grapple has 180° symmetry).
        # Pick the one numerically closest to current joint position so the
        # PD command never wraps around (no 360° spins).
        candidates = [base, base + math.pi, base - math.pi]
        return min(candidates, key=lambda c: abs(c - self.current_yaw_joint))

    def _gripper_state_cb(self, msg):
        self.current_gripper = msg.data

    def _update_transforms_from_tf(self):
        """Look up camera and crane base transforms from TF.

        The sim publishes quat_w_ros (ROS camera convention) directly,
        so no convention conversion needed — just use the quaternion as-is.
        """
        if not self._tf_available or self._tf_initialized:
            return
        try:
            # Camera pose in world frame
            # lookup_transform(target, source) returns source expressed in target
            # We want camera expressed in world frame
            t_cam = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, rclpy.time.Time())
            p = t_cam.transform.translation
            q = t_cam.transform.rotation
            self.cam_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            self.cam_quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)

            # Crane base pose in world frame
            t_base = self._tf_buffer.lookup_transform(
                self._world_frame, self._base_frame, rclpy.time.Time())
            p2 = t_base.transform.translation
            q2 = t_base.transform.rotation
            self.base_pos = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
            self.base_quat = np.array([q2.w, q2.x, q2.y, q2.z], dtype=np.float32)

            self._tf_initialized = True
            self.get_logger().info(
                f"TF loaded: cam_pos={self.cam_pos}, base_pos={self.base_pos}")
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────

    def _tick(self):
        if self.fsm.cycle_count >= self.max_cycles:
            return

        # Update transforms from TF (once, on first availability)
        self._update_transforms_from_tf()

        if self.waiting_for_target:
            if self.latest_depth is None or self.intrinsics is None:
                return

            # Run point cloud pipeline (same math as Isaac Lab's sim pipeline)
            obs = process_depth(
                self.latest_depth, self.intrinsics,
                self.cam_pos, self.cam_quat,
                self.base_pos, self.base_quat,
                self.num_points, self.depth_range,
            )

            # Debug: check point cloud
            pc = obs.numpy().reshape(-1, 3)
            valid = np.any(pc != 0, axis=1)
            if valid.any():
                self.get_logger().info(
                    f"PCD: {valid.sum()}/{len(pc)} valid, "
                    f"x=[{pc[valid,0].min():.2f},{pc[valid,0].max():.2f}] "
                    f"y=[{pc[valid,1].min():.2f},{pc[valid,1].max():.2f}] "
                    f"z=[{pc[valid,2].min():.2f},{pc[valid,2].max():.2f}]")
                self.get_logger().info(
                    f"  depth shape={self.latest_depth.shape}, "
                    f"valid_depth={np.isfinite(self.latest_depth).sum()}/{self.latest_depth.size}, "
                    f"range=[{self.latest_depth[np.isfinite(self.latest_depth)].min():.2f},{self.latest_depth[np.isfinite(self.latest_depth)].max():.2f}]")
                self.get_logger().info(
                    f"  cam_pos={self.cam_pos}, cam_quat={self.cam_quat}")
                self.get_logger().info(
                    f"  base_pos={self.base_pos}, base_quat={self.base_quat}")
            else:
                self.get_logger().warn(f"PCD: 0 valid points! depth shape={self.latest_depth.shape}")

            # Run policy
            x, y, z, yaw = self.policy.get_target(obs)
            self.get_logger().info(
                f"Cycle {self.fsm.cycle_count + 1}: target=({x:.2f}, {y:.2f}, {z:.2f}), "
                f"yaw={math.degrees(yaw):.1f}deg")

            self._publish_target(x, y, z, yaw)
            self.fsm.set_target(x, y, z, yaw)
            self.waiting_for_target = False

        # FSM tick
        pre_phase = self.fsm.phase
        pre_dwell = self.fsm.dwell
        cmd = self.fsm.tick(self.current_ee_pos, self.current_ee_yaw, self.current_gripper)
        if pre_phase == Phase.HOVER_UP and self.fsm.phase_timer % 100 == 1:
            self.get_logger().info(
                f"[FSM DBG] dwell={self.fsm.dwell}, timer={self.fsm.phase_timer}, "
                f"ee={[round(v,2) for v in self.current_ee_pos]}, "
                f"tgt={[round(v,2) for v in self.fsm._hover_pos]}")
        # TEMP yaw debug — remove once yaw alignment is confirmed working
        if pre_phase != self.fsm.phase:
            self.get_logger().info(
                f"[YAW DBG] {pre_phase.name}->{self.fsm.phase.name} | "
                f"ee_yaw={math.degrees(self.current_ee_yaw):.1f}deg, "
                f"cmd.yaw={math.degrees(cmd.yaw):.1f}deg, "
                f"fsm.target_yaw={math.degrees(self.fsm.target_yaw):.1f}deg")

        # Send to JV controller via service.
        # cmd.yaw is in basegrapple-base frame; jv_controller expects yaw JOINT.
        # Convert once on phase entry (or when cmd.yaw changes), then hold the
        # cached value. Recomputing every tick makes the joint chase
        # basegrapple drift during descent and causes visible spinning.
        if not self.dry_run:
            phase_changed = (pre_phase != self.fsm.phase)
            yaw_changed = (self._last_cmd_yaw is None
                           or abs(cmd.yaw - self._last_cmd_yaw) > 1e-4)
            if phase_changed or yaw_changed:
                self._cached_joint_yaw = self._grapple_yaw_to_joint(cmd.yaw)
                self._last_cmd_yaw = cmd.yaw
            joint_yaw = self._cached_joint_yaw
            self._publish_ee_command(cmd.position, cmd.yaw)
            self._publish_gripper(cmd.gripper)
            self._send_target_to_jv(cmd.position, joint_yaw, cmd.gripper)

        # Log position error
        ee = self.current_ee_pos
        tgt = cmd.position  # this includes the ee_to_grapple offset
        # FSM targets are in grapple frame, so compare against grapple pos directly
        pos_err = math.sqrt((ee[0]-tgt[0])**2 + (ee[1]-tgt[1])**2 + (ee[2]-tgt[2])**2)
        yaw_err = abs(self.fsm._wrap_angle(self.current_ee_yaw - cmd.yaw))
        self._log_writer.writerow([
            f"{pytime.time() - self._log_start:.2f}",
            cmd.phase.name, self.fsm.cycle_count,
            f"{ee[0]:.4f}", f"{ee[1]:.4f}", f"{ee[2]:.4f}",
            f"{tgt[0]:.4f}", f"{tgt[1]:.4f}", f"{tgt[2]:.4f}",
            f"{pos_err:.4f}", f"{self.current_ee_yaw:.4f}", f"{cmd.yaw:.4f}", f"{yaw_err:.4f}",
            f"{self.current_gripper:.4f}", f"{cmd.gripper:.4f}",
        ])
        self._log_file.flush()

        state_msg = String()
        state_msg.data = f"{cmd.phase.name} (cycle {self.fsm.cycle_count})"
        self.pub_fsm_state.publish(state_msg)

        if cmd.cycle_complete:
            self.get_logger().info(f"Cycle {self.fsm.cycle_count} complete")
            self.waiting_for_target = True

    def _cycle_complete_cb(self, msg):
        """Handle cycle_complete from sequencer (if used)."""
        if msg.data:
            self.waiting_for_target = True

    def _send_target_to_jv(self, pos, yaw, gripper):
        """Send EE target to JV controller via SetTarget service.

        Only sends when the target changes significantly to avoid
        re-running IK 50 times/sec and causing jitter.
        """
        if not self.set_target_client.service_is_ready():
            if not hasattr(self, '_jv_warn_printed'):
                self.get_logger().warn("JV controller service /crane/set_target not available — is the JV controller running?")
                self._jv_warn_printed = True
            return

        # Only send if target changed (avoids re-running IK every tick)
        new_target = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
                      round(yaw, 3), round(gripper, 3))
        if hasattr(self, '_last_jv_target') and self._last_jv_target == new_target:
            return
        self._last_jv_target = new_target

        req = SetTarget.Request()
        req.target_xyz = [float(pos[0]), float(pos[1]), float(pos[2])]
        req.target_yaw = float(yaw)
        req.grapple_opening = float(gripper)
        self.set_target_client.call_async(req)
        self.get_logger().info(
            f"→ JV target: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
            f"yaw={math.degrees(yaw):.1f}°, grip={gripper:.2f}")

    # ── Publishing helpers ────────────────────────────────────────────

    def _publish_ee_command(self, pos, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "crane_base"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_ee_cmd.publish(msg)

    def _publish_gripper(self, opening):
        msg = Float64()
        msg.data = float(opening)
        self.pub_gripper.publish(msg)

    def _publish_target(self, x, y, z, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "crane_base"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.orientation.z = math.sin(yaw / 2)
        self.pub_policy_target.publish(msg)

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_matrix(quat_wxyz):
        """Convert wxyz quaternion to 3x3 rotation matrix.

        Uses the same formula as Isaac Lab's matrix_from_quat which handles
        non-unit quaternions via 2/(q.q) normalization.
        """
        q = np.array(quat_wxyz, dtype=np.float64)
        two_s = 2.0 / np.dot(q, q)
        w, x, y, z = q
        return np.array([
            [1 - two_s*(y*y + z*z), two_s*(x*y - z*w),     two_s*(x*z + y*w)],
            [two_s*(x*y + z*w),     1 - two_s*(x*x + z*z), two_s*(y*z - x*w)],
            [two_s*(x*z - y*w),     two_s*(y*z + x*w),     1 - two_s*(x*x + y*y)],
        ], dtype=np.float32)

    @staticmethod
    def _rpy_to_matrix(rpy):
        r, p, y = rpy
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr],
        ], dtype=np.float32)

    def destroy_node(self):
        if hasattr(self, '_log_file'):
            self._log_file.close()
        super().destroy_node()


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
    if args.dry_run:
        node.dry_run = True

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Completed {node.fsm.cycle_count} cycles")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
