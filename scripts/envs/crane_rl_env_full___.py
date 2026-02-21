#!/usr/bin/env python3
# crane_rl_env.py
#
# Direct-workflow RL environment for the crane with task-space control (Differential IK),
# rack + vectorized logs, and a heuristic expert.
#
#
# Copyright (c) 2022-2025

from __future__ import annotations
import os, math, argparse
from typing import List, Tuple, Optional, Sequence

import numpy as np
import torch

# Simple IK for better control stability
try:
    from ikpy.chain import Chain
    IK_AVAILABLE = True
except ImportError:
    print("Warning: ikpy not available - falling back to differential IK")
    IK_AVAILABLE = False
from gymnasium import spaces
from isaaclab.app import AppLauncher


class SimpleIKController:
    """One-shot IK controller with PD velocity control."""
    
    def __init__(self, urdf_path: str, num_envs: int, device: str, control_dt: float = 1/60.0,
                 gripper_kp: float = 12.0, gripper_max_vel: float = 15.0):
        self.num_envs = num_envs
        self.device = device
        self.control_dt = control_dt
        self._gripper_kp = gripper_kp
        self._gripper_max_vel = gripper_max_vel
        self.chain = None
        
        if IK_AVAILABLE:
            if os.path.exists(urdf_path):
                try:
                    # Active mask for 5 arm joints (base + 4 controlled + upperpassive)
                    active_mask = [False, True, True, True, True, True]  # Base + 5 active joints
                    self.chain = Chain.from_urdf_file(urdf_path, active_links_mask=active_mask, base_elements=["basemast"])
                    print(f"SimpleIK: Loaded {len([m for m in active_mask if m])} active joints from {urdf_path}")
                    print(f"SimpleIK: Chain links: {[link.name for link in self.chain.links]}")
                except Exception as e:
                    print(f"Warning: SimpleIK failed to load URDF {urdf_path} - {e}")
                    self.chain = None
            else:
                print(f"Warning: SimpleIK URDF file not found: {urdf_path}")
        else:
            print("Warning: SimpleIK ikpy not available")

        # Upperpassive to basegrapple offset (0.415m down along Z-axis)
        self.UPPERPASSIVE_TO_BASEGRAPPLE_OFFSET = np.array([0.0, 0.0, -0.415])
        
        # PD control parameters for arm joints
        # 4 arm joints: base, boom, stick, telescope
        self.KP_ARM = torch.tensor([4.0, 4.0, 4.0, 4.0], device=device, dtype=torch.float32)
        self.KD_ARM = torch.tensor([1.2, 1.2, 1.2, 1.2], device=device, dtype=torch.float32)
        self.MAX_VEL_ARM = torch.tensor([2.0, 2.0, 2.0, 2.0], device=device, dtype=torch.float32)
        
        # Gripper PD parameters (configurable for gentler closing)
        self.KP_GRIP = torch.tensor([gripper_kp, gripper_kp], device=device, dtype=torch.float32)
        self.KD_GRIP = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
        self.MAX_VEL_GRIP = torch.tensor([gripper_max_vel, gripper_max_vel], device=device, dtype=torch.float32)
        
        # PD control state per environment (4 arm joints + 2 grippers = 6 total)
        self.q_des_arm = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)
        self.prev_error_arm = torch.zeros(num_envs, 4, device=device, dtype=torch.float32)
        
        # Gripper desired positions and error history
        self.q_des_grip = torch.zeros(num_envs, 2, device=device, dtype=torch.float32)
        self.prev_error_grip = torch.zeros(num_envs, 2, device=device, dtype=torch.float32)
        
    def solve_ik_and_compute_velocities(self, target_pos_b: torch.Tensor, current_joints: torch.Tensor) -> torch.Tensor:
        """
        Solve IK for basegrapple target and compute PD velocity commands.
        Args:
            target_pos_b: [N, 3] target positions in base frame
            current_joints: [N, 4] current joint positions for controlled joints
        Returns:
            [N, 4] joint velocity commands for controlled joints
        """
        # First, solve IK to get desired joint positions
        if self.chain is not None:
            for i in range(self.num_envs):
                # Only debug first environment 
                debug_this_env = (i == 0) and hasattr(self, '_ik_debug_counter')
                if not hasattr(self, '_ik_debug_counter'):
                    self._ik_debug_counter = 0
                target_bg = target_pos_b[i].cpu().numpy()
                seed = current_joints[i].cpu().numpy()
                
                # Convert basegrapple target to upperpassive target
                target_up = target_bg - self.UPPERPASSIVE_TO_BASEGRAPPLE_OFFSET

                # IKPy expects [y, x, z] coordinate order
                T = np.eye(4)
                T[:3, 3] = [target_up[1], target_up[0], target_up[2]]
                
                try:
                    # Create full initial position for IK (6 links including dummy root)
                    full_initial = np.zeros(6)
                    full_initial[1:5] = seed  # Only first 4 controlled joints (skip dummy root and upperpassive)
                    # Let upperpassive (index 5) default to 0.0
                    
                    if debug_this_env and self._ik_debug_counter % 100 == 0:
                        print(f"IK DEBUG: target_bg={target_bg}, target_up={target_up}")
                        print(f"IK DEBUG: T matrix target: {T[:3, 3]}")
                        print(f"IK DEBUG: full_initial={full_initial}")
                        
                    # Solve IK
                    full_solution = self.chain.inverse_kinematics_frame(T, initial_position=full_initial)
                    
                    # Extract first 4 controlled joints (excluding dummy root and upperpassive)
                    # [basemast_to_mast, mast_to_mainboom, mainboom_to_stick, stick_to_telescope]
                    q = full_solution[1:5]  # Skip dummy root and upperpassive
                    
                    if debug_this_env and self._ik_debug_counter % 100 == 0:
                        print(f"IK DEBUG: full_solution={full_solution}")
                        print(f"IK DEBUG: extracted q={q}")
                        
                    # Basic validation
                    if not np.any(np.isnan(q)):
                        # Apply joint limits check
                        joint_limits = [
                            (-1.74533, 1.74533),    # basemast_to_mast
                            (-0.383972, 1.309),     # mast_to_mainboom  
                            (-3.08574, 0.035),      # mainboom_to_stick
                            (0.13, 1.8),            # stick_to_telescope
                        ]
                        
                        valid_solution = True
                        for j, (q_val, (q_min, q_max)) in enumerate(zip(q, joint_limits)):
                            if q_val < q_min or q_val > q_max:
                                valid_solution = False
                                if debug_this_env and self._ik_debug_counter % 100 == 0:
                                    print(f"IK DEBUG: Joint {j} limit violation: {q_val:.3f} not in [{q_min:.3f}, {q_max:.3f}]")
                                break
                        
                        if valid_solution:
                            self.q_des_arm[i] = torch.from_numpy(q).to(self.device)
                            if debug_this_env and self._ik_debug_counter % 100 == 0:
                                print(f"IK DEBUG: SUCCESS - set q_des_arm[{i}]={self.q_des_arm[i]}")
                        else:
                            if debug_this_env and self._ik_debug_counter % 100 == 0:
                                print(f"IK DEBUG: FAILED - solution violates joint limits")
                    else:
                        if debug_this_env and self._ik_debug_counter % 100 == 0:
                            print(f"IK DEBUG: FAILED - q contains NaN")
                        
                except Exception as e:
                    if debug_this_env and self._ik_debug_counter % 100 == 0:
                        print(f"IK DEBUG: EXCEPTION - {e}")
                        
                if debug_this_env:
                    self._ik_debug_counter += 1
        else:
            # No IK chain loaded - cannot update desired positions
            if not hasattr(self, '_no_chain_warned'):
                print("Warning: SimpleIK - No IK chain loaded, q_des will remain at defaults")
                self._no_chain_warned = True
        
        # Now compute PD velocity control for arm joints only
        return self._compute_arm_velocities(current_joints)
    
    def _compute_arm_velocities(self, current_arm_joints: torch.Tensor) -> torch.Tensor:
        """
        Compute PD velocity commands for arm joints using PD control.
        """
        # Compute arm joint errors
        err = self.q_des_arm - current_arm_joints
        
        # Compute derivative of error
        err_dot = (err - self.prev_error_arm) / self.control_dt
        
        # PD control: vel = KP * err + KD * err_dot
        vel = self.KP_ARM * err + self.KD_ARM * err_dot

        # Apply velocity limits
        vel = torch.clamp(vel, -self.MAX_VEL_ARM, self.MAX_VEL_ARM)

        # Store current error for next iteration
        self.prev_error_arm = err.clone()

        # Apply velocity scaling factor
        velocity_scale = getattr(self, 'velocity_scale', 1.0)
        vel *= velocity_scale
        
        return vel
    
    def set_gripper_targets(self, gripper_openings: torch.Tensor):
        """
        Set desired gripper positions for both grippers.
        Args:
            gripper_openings: [N] desired opening values for both grippers
        """
        # Both grippers get same target
        self.q_des_grip[:, 0] = gripper_openings  # Left gripper
        self.q_des_grip[:, 1] = gripper_openings  # Right gripper
    
    def compute_gripper_velocities(self, current_gripper_joints: torch.Tensor) -> torch.Tensor:
        """
        Compute PD velocity commands for gripper joints.
        Args:
            current_gripper_joints: [N, 2] current gripper positions [left, right]
        Returns:
            [N, 2] gripper velocity commands
        """
        # Compute gripper joint errors
        err = self.q_des_grip - current_gripper_joints
        
        # Compute derivative of error
        err_dot = (err - self.prev_error_grip) / self.control_dt
        
        # PD control with gripper-specific gains
        vel = self.KP_GRIP * err + self.KD_GRIP * err_dot
        
        # Apply gripper velocity limits
        vel = torch.clamp(vel, -self.MAX_VEL_GRIP, self.MAX_VEL_GRIP)
        
        # Store current error for next iteration
        self.prev_error_grip = err.clone()
        
        # Apply velocity scaling
        velocity_scale = getattr(self, 'velocity_scale', 1.0)
        vel *= velocity_scale
        
        return vel
    
    def set_velocity_scale(self, scale: float):
        """Set velocity scaling factor for smoother motion."""
        self.velocity_scale = max(0.1, min(2.0, scale))  # Clamp to reasonable bounds


# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Crane direct RL environment with task-space IK and heuristic baseline.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=12.0)

# Rack/crane placement (env-local)
parser.add_argument("--rack_x", type=float, default=2.0)
parser.add_argument("--rack_y", type=float, default=-1.0)
parser.add_argument("--crane_x", type=float, default=None)
parser.add_argument("--crane_y", type=float, default=None)
parser.add_argument("--crane_z", type=float, default=1.423)

# IK end-effector body (the jacobian/EEF for IK). We will still plan targets in BG frame.
parser.add_argument("--ee_body", type=str, default="upperpassive", help="IK end-effector body (recommend 'upperpassive').")
parser.add_argument("--ee_to_grapple_offset_z", type=float, default=0.415,
                    help="Z offset from basegrapple to upperpassive. Used to map BG target -> EE target.")

# IK controller type
parser.add_argument("--use_simple_ik", action="store_true", help="Use one-shot IKPy controller instead of differential IK")
parser.add_argument("--urdf_path", type=str, default="/workspace/crane_testbed/assets/urdf/fpiforwarder-upperpassive.urdf",
                    help="Path to URDF file for simple IK (truncated to upperpassive)")

# Logs grid (per env)
parser.add_argument("--num_logs", type=int, default=200)
parser.add_argument("--rows", type=int, default=20)
parser.add_argument("--layers", type=int, default=10)
parser.add_argument("--spacing_y", type=float, default=0.16)
parser.add_argument("--spacing_z", type=float, default=0.12)
parser.add_argument("--base_z", type=float, default=0.10)
parser.add_argument("--fixed_x", type=float, default=None)     # defaults to rack_x
parser.add_argument("--center_y", type=float, default=None)    # defaults to rack_y

# Jitter
parser.add_argument("--row_y_jitter", type=float, default=0.0)
parser.add_argument("--layer_y_offset", type=float, default=0.08)
parser.add_argument("--spawn_height", type=float, default=0.12)
parser.add_argument("--jitter_xy", type=float, default=0.0)
parser.add_argument("--jitter_height", type=float, default=0.0)
parser.add_argument("--jitter_ang", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mixed_log_patterns", action="store_true", help="Enable mixed log grid patterns across environments for training variety")

# Assets
parser.add_argument("--rack_usd", type=str, default="/workspace/crane_testbed/assets/scenes/short_rack.usdc")
parser.add_argument("--log_usd",  type=str, default="/workspace/crane_testbed/assets/scenes/testbed_log.usd")
parser.add_argument("--crane_usd", type=str, default="/workspace/crane_testbed/assets/scenes/crane.usd")

# Log physics & orientation
parser.add_argument("--log_mass",  type=float, default=15.0)
parser.add_argument("--log_roll",  type=float, default=0.0)
parser.add_argument("--log_pitch", type=float, default=0.0)
parser.add_argument("--log_yaw",   type=float, default=90.0)

# Note: This standalone script runs the heuristic baseline only.
# For RL training, use: python scripts/rsl_rl/train.py
# For RL inference, use: python scripts/rsl_rl/play.py --checkpoint <path>

# Heuristic & trailer
parser.add_argument("--heu_grip_open", type=float, default=0.0)
parser.add_argument("--heu_grip_closed", type=float, default=2.23)
parser.add_argument("--trailer_x_min", type=float, default=0.0)
parser.add_argument("--trailer_x_max", type=float, default=6.0)
parser.add_argument("--trailer_y_min", type=float, default=-2.0)
parser.add_argument("--trailer_y_max", type=float, default=2.0)
parser.add_argument("--drop_clearance", type=float, default=0.06)
parser.add_argument("--lane_spacing_y", type=float, default=0.40)
parser.add_argument("--lane_count", type=int, default=4)

# Settling
parser.add_argument("--settle_time", type=float, default=2.0, help="Seconds to settle logs before task starts")

# Gripper closing control (prevent penetration with slow, gentle closing)
parser.add_argument("--gripper_close_step", type=float, default=0.15,
                    help="Gripper closing step size in radians (smaller = slower, gentler)")
parser.add_argument("--gripper_close_delay", type=float, default=0.02,
                    help="Delay between gripper steps in seconds (larger = slower)")
parser.add_argument("--gripper_max_velocity", type=float, default=15.0,
                    help="Maximum gripper velocity in rad/s")
parser.add_argument("--gripper_kp", type=float, default=12.0,
                    help="Gripper PD proportional gain")

# Debug prints
parser.add_argument("--debug_every_steps", type=int, default=30, help="Telemetry print cadence.")
parser.add_argument("--debug_envs_max", type=int, default=4, help="Max envs to print per tick.")


# Solver
parser.add_argument("--solver", choices=["pgs","tgs"], default="pgs")
parser.add_argument("--enhanced_determinism", action="store_true")

# Visualization
parser.add_argument("--viz_markers", action="store_true", help="Enable debug VisualizationMarkers.")
parser.add_argument("--show_action_bounds", action="store_true", help="Show translucent box for action/rack bounds.")

parser.add_argument("--debug_logs", action="store_true",
                    help="Print periodic live log pose probes.")
parser.add_argument("--debug_reward_norm", action="store_true",
                    help="Enable reward normalization debug: prints + cylinder visualization.")

parser.add_argument("--use-target-selection-policy", action="store_true",
                    help="Train only the PH_HOVER_UP target-selection head")


# Rendering + AppLauncher args
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)

# Parse and maybe launch the app
if __name__ == "__main__":
    args_cli = parser.parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
else:
    args_cli = parser.parse_args(args=[])
    simulation_app = None

# Derived defaults
if args_cli.fixed_x is None:
    args_cli.fixed_x = args_cli.rack_x
if args_cli.center_y is None:
    args_cli.center_y = args_cli.rack_y
y_neg_end = args_cli.center_y - 0.5 * (args_cli.rows - 1) * args_cli.spacing_y
if args_cli.crane_x is None:
    args_cli.crane_x = 6.0
if args_cli.crane_y is None:
    args_cli.crane_y = y_neg_end

# --- Sanity checks / derived params ---
if args_cli.ee_to_grapple_offset_z < 0.0:
    print(f"[WARN] ee_to_grapple_offset_z is negative ({args_cli.ee_to_grapple_offset_z:.3f}). "
          f"Expected +0.415 for upperpassive above basegrapple. Using abs().")
    args_cli.ee_to_grapple_offset_z = abs(args_cli.ee_to_grapple_offset_z)


# ---------------- Imports (after app starts) ----------------
import omni.usd
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

from isaaclab.assets import RigidObject, RigidObjectCfg, ArticulationCfg, Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from pxr import UsdPhysics
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

# Optional PhysX schema / USD physics
try:
    from pxr import PhysxSchema
except Exception:
    PhysxSchema = None
try:
    from pxr import UsdPhysics
except Exception:
    UsdPhysics = None


# ===== Helpers =====
def euler_deg_to_quat(r: float, p: float, y: float) -> Tuple[float,float,float,float]:
    rx, ry, rz = math.radians(r), math.radians(p), math.radians(y)
    cx, cy, cz = math.cos(rx/2), math.cos(ry/2), math.cos(rz/2)
    sx, sy, sz = math.sin(rx/2), math.sin(ry/2), math.sin(rz/2)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    yq = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return (w, x, yq, z)

def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack((torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)), dim=-1)

def quat_multiply_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ), dim=-1)

# --- Small math helpers for composing base->world for markers ---
def _quat_wxyz_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ), dim=-1)

def _quat_rotate_vec_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q.unbind(-1); vx, vy, vz = v.unbind(-1)
    rw = - qx*vx - qy*vy - qz*vz
    rx = + qw*vx + qy*vz - qz*vy
    ry = + qw*vy + qz*vx - qx*vz
    rz = + qw*vz + qx*vy - qy*vx
    x = -rw*qx + rx*qw - ry*qz + rz*qy
    y = -rw*qy + ry*qw - rz*qx + rx*qz
    z = -rw*qz + rz*qw - rx*qy + ry*qx
    return torch.stack((x, y, z), dim=-1)

def _compose_base_local_to_world(base_pos_w: torch.Tensor, base_quat_w: torch.Tensor,
                                 local_pos_b: torch.Tensor,
                                 local_quat_b: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    if local_pos_b.ndim == 1: local_pos_b = local_pos_b.unsqueeze(0)
    if base_pos_w.ndim == 1:  base_quat_w = base_quat_w.unsqueeze(0); base_pos_w = base_pos_w.unsqueeze(0)
    wp = base_pos_w + _quat_rotate_vec_wxyz(base_quat_w, local_pos_b)
    if local_quat_b is None:
        wq = base_quat_w
    else:
        if local_quat_b.ndim == 1: local_quat_b = local_quat_b.unsqueeze(0)
        wq = _quat_wxyz_mul(base_quat_w, local_quat_b)
    return wp.squeeze(0), wq.squeeze(0)

def define_xform_idem(path: str, translation=None, quat_wxyz: Optional[Tuple[float,float,float,float]]=None):
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Xform.Define(stage, Sdf.Path(path)).GetPrim()
    xform = UsdGeom.Xformable(prim)
    if translation is not None:
        t_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t_op = op; break
        (t_op or xform.AddTranslateOp()).Set(Gf.Vec3d(float(translation[0]),
                                                      float(translation[1]),
                                                      float(translation[2])))
    if quat_wxyz is not None:
        o_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                o_op = op; break
        if o_op is None:
            o_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        qw, qx, qy, qz = quat_wxyz
        o_op.Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
    return prim

def force_opaque_materials_under(root_path: str):
    root = prim_utils.get_prim_at_path(root_path)
    if not root or not root.IsValid():
        return
    for p in Usd.PrimRange(root):
        if p.IsA(UsdGeom.Mesh):
            try:
                UsdGeom.Mesh(p).CreateDoubleSidedAttr(True)
            except Exception:
                pass
            try:
                pv_api = UsdGeom.PrimvarsAPI(p)
                if pv_api:
                    pv = pv_api.GetPrimvar("displayOpacity")
                    if pv and pv.IsDefined():
                        pv.Set(1.0)
                        pv.SetInterpolation(UsdGeom.Tokens.constant)
            except Exception:
                pass
        try:
            mb = UsdShade.MaterialBindingAPI(p)
            mat = None
            try:
                res = mb.ComputeBoundMaterial()
                mat = res[0] if isinstance(res, tuple) else res
            except Exception:
                mat = mb.GetDirectBinding().GetMaterial()
            if not mat:
                continue
            for child in mat.GetPrim().GetChildren():
                if not child.IsA(UsdShade.Shader):
                    continue
                sh = UsdShade.Shader(child)
                sid = (sh.GetIdAttr().Get() or "").lower()
                try:
                    if "usdpreviewsurface" in sid:
                        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
                        sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
                    sh.CreateInput("cutout_opacity", Sdf.ValueTypeNames.Float).Set(1.0)
                except Exception:
                    pass
                for inp in sh.GetInputs():
                    nm = inp.GetBaseName().lower()
                    try:
                        if ("opacity" in nm) or ("alpha" in nm):
                            inp.DisconnectSource()
                            try: inp.Set(1.0)
                            except Exception:
                                try: inp.Set(Gf.Vec3f(1.0,1.0,1.0))
                                except Exception: pass
                        if ("transmission" in nm) or ("refraction" in nm):
                            inp.DisconnectSource()
                            try: inp.Set(0.0)
                            except Exception:
                                try: inp.Set(Gf.Vec3f(0.0,0.0,0.0))
                                except Exception: pass
                        if ("thin_walled" in nm) or ("thinwalled" in nm):
                            try: inp.Set(False)
                            except Exception: pass
                    except Exception:
                        pass
        except Exception:
            pass

def plan_grid_yz(center_y_local: float, rows: int, layers: int, spacing_y: float, spacing_z: float,
                 base_z: float, cap: int, row_y_jitter: float, layer_y_offset: float, seed: Optional[int]):
    import random
    if row_y_jitter > 0.0 and seed is not None:
        random.seed(seed)
    layer_offs = [(layer_y_offset if (L % 2 == 1) else 0.0) for L in range(layers)]
    if layers > 0:
        mean_off = sum(layer_offs) / layers
        layer_offs = [o - mean_off for o in layer_offs]
    positions, placed = [], 0
    y0 = center_y_local - 0.5 * (rows - 1) * spacing_y
    for L in range(layers):
        if placed >= cap: break
        zc = base_z + L * spacing_z
        layer_offset = layer_offs[L]
        for i in range(rows):
            if placed >= cap: break
            yc = y0 + i * spacing_y + layer_offset
            if row_y_jitter > 0.0:
                yc += (random.random()*2 - 1) * row_y_jitter
            positions.append((yc, zc))
            placed += 1
    return positions

def plan_grid_yz_pattern_b(center_y_local: float, rows: int, layers: int, spacing_y: float, spacing_z: float,
                          base_z: float, cap: int, row_y_jitter: float, layer_y_offset: float, seed: Optional[int]):
    """Pattern B: Tower layout - 10 rows x 30 layers, shifted toward left side"""
    import random
    if row_y_jitter > 0.0 and seed is not None:
        random.seed(seed)
    
    # Pattern B: Very compact grid with much fewer rows, many more layers
    pattern_rows = 10  # Much fewer rows (1/3 of original)
    pattern_layers = 30  # Triple the original layers
    
    # Shift center toward left side (negative Y direction)
    left_shift = -1.0  # Move 1 meter to the left
    shifted_center = center_y_local + left_shift
    
    layer_offs = [(layer_y_offset if (L % 2 == 1) else 0.0) for L in range(pattern_layers)]
    if pattern_layers > 0:
        mean_off = sum(layer_offs) / pattern_layers
        layer_offs = [o - mean_off for o in layer_offs]
    
    positions, placed = [], 0
    y0 = shifted_center - 0.5 * (pattern_rows - 1) * spacing_y
    
    for L in range(pattern_layers):
        if placed >= cap: break
        zc = base_z + L * spacing_z
        layer_offset = layer_offs[L]
        
        for i in range(pattern_rows):
            if placed >= cap: break
            yc = y0 + i * spacing_y + layer_offset
            if row_y_jitter > 0.0:
                yc += (random.random()*2 - 1) * row_y_jitter
            positions.append((yc, zc))
            placed += 1
    
    return positions

def plan_grid_yz_pattern_c(center_y_local: float, rows: int, layers: int, spacing_y: float, spacing_z: float,
                          base_z: float, cap: int, row_y_jitter: float, layer_y_offset: float, seed: Optional[int]):
    """Pattern C: Tower layout - 10 rows x 30 layers, shifted toward right side"""
    import random
    if row_y_jitter > 0.0 and seed is not None:
        random.seed(seed)
    
    # Pattern C: Very compact grid with much fewer rows, many more layers
    pattern_rows = 10  # Much fewer rows (1/3 of original)
    pattern_layers = 30  # Triple the original layers
    
    # Shift center toward right side (positive Y direction)
    right_shift = 1.0  # Move 1 meter to the right
    shifted_center = center_y_local + right_shift
    
    layer_offs = [(layer_y_offset if (L % 2 == 1) else 0.0) for L in range(pattern_layers)]
    if pattern_layers > 0:
        mean_off = sum(layer_offs) / pattern_layers
        layer_offs = [o - mean_off for o in layer_offs]
    
    positions, placed = [], 0
    y0 = shifted_center - 0.5 * (pattern_rows - 1) * spacing_y
    
    for L in range(pattern_layers):
        if placed >= cap: break
        zc = base_z + L * spacing_z
        layer_offset = layer_offs[L]
        
        for i in range(pattern_rows):
            if placed >= cap: break
            yc = y0 + i * spacing_y + layer_offset
            if row_y_jitter > 0.0:
                yc += (random.random()*2 - 1) * row_y_jitter
            positions.append((yc, zc))
            placed += 1
    
    return positions

def plan_grid_yz_random(center_y_local: float, rows: int, layers: int, spacing_y: float, spacing_z: float,
                       base_z: float, cap: int, row_y_jitter: float, layer_y_offset: float, seed: Optional[int]):
    """Generate truly random log pile patterns that will settle into different profiles.

    Randomizes:
    - Grid dimensions (row count, layer count)
    - Spacing between logs (horizontal and vertical)
    - Center position shifts
    - Layer offsets for irregular stacking
    - Per-log jitter for organic pile shapes
    """
    import random
    if seed is not None:
        random.seed(seed)

    # Randomize grid dimensions (keep total ~200 logs)
    # More rows = wider pile, more layers = taller pile
    pattern_rows = random.randint(15, 25)  # More conservative range
    pattern_layers = random.randint(10, 20)  # More conservative range

    # Randomize spacing (affects density and pile shape)
    # Base spacing: y=0.16m, z=0.12m
    random_spacing_y = spacing_y * random.uniform(0.9, 1.15)  # 14-18cm horizontal
    random_spacing_z = spacing_z * random.uniform(0.9, 1.15)  # 11-14cm vertical

    # Randomize center shift (left/right bias) - keep within rack bounds
    center_shift = random.uniform(-0.6, 0.6)  # ±60cm shift (conservative)
    shifted_center = center_y_local + center_shift

    # Randomize layer offsets for irregular stacking
    layer_offs = []
    for L in range(pattern_layers):
        # Each layer can have different offset pattern
        if random.random() < 0.5:
            # Alternating offset
            offset = (layer_y_offset if (L % 2 == 1) else 0.0)
        else:
            # Random offset per layer (smaller range)
            offset = random.uniform(-0.05, 0.05)  # ±5cm instead of ±10cm
        layer_offs.append(offset)

    # Center the offsets
    if pattern_layers > 0:
        mean_off = sum(layer_offs) / pattern_layers
        layer_offs = [o - mean_off for o in layer_offs]

    positions, placed = [], 0
    y0 = shifted_center - 0.5 * (pattern_rows - 1) * random_spacing_y

    # Randomize base height variation (small to keep logs stable)
    base_z_variation = random.uniform(-0.02, 0.02)  # ±2cm base height

    # Keep generating layers until we have enough logs
    L = 0
    max_layers = 100  # Safety limit to prevent infinite loop

    while placed < cap and L < max_layers:
        zc = base_z + base_z_variation + L * random_spacing_z

        # Get layer offset (cycle through if we exceed original pattern_layers)
        layer_offset = layer_offs[L % len(layer_offs)] if layer_offs else 0.0

        for i in range(pattern_rows):
            if placed >= cap: break

            yc = y0 + i * random_spacing_y + layer_offset

            # Add moderate jitter for organic pile shapes (stay within rack bounds)
            yc += (random.random()*2 - 1) * 0.04  # ±4cm y jitter (reduced from 8cm)
            zc_jittered = zc + (random.random()*2 - 1) * 0.02  # ±2cm z jitter (reduced from 4cm)

            positions.append((yc, zc_jittered))
            placed += 1

        L += 1

    return positions

# ===== Crane config =====
CRANE_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Crane",
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.crane_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "basemast_to_mast": 0.0,
            "mast_to_mainboom": 1.0,
            "mainboom_to_stick": -0.9,
            "stick_to_telescope": 0.130,
            "telescope_to_upperpassive": 0.0,
            "upperpassive_to_lowerpassive": 0.0,
            "lowerpassive_to_basegrapple": 0.0,
            "basegrapple_to_gripperleft": 0.0,
            "basegrapple_to_gripperright": 0.0,
        },
        pos=(args_cli.crane_x, args_cli.crane_y, args_cli.crane_z),
    ),
    actuators={
        "crane_base": ImplicitActuatorCfg(
            joint_names_expr=["basemast_to_mast"], effort_limit_sim=800_000.0, velocity_limit_sim=1.0,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_boom": ImplicitActuatorCfg(
            joint_names_expr=["mast_to_mainboom"], effort_limit_sim=1_000_000.0, velocity_limit_sim=2.0,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_stick": ImplicitActuatorCfg(
            joint_names_expr=["mainboom_to_stick"], effort_limit_sim=600_000.0, velocity_limit_sim=2.0,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_telescope": ImplicitActuatorCfg(
            joint_names_expr=["stick_to_telescope"], effort_limit_sim=500_000.0, velocity_limit_sim=2.0,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "passive_chain": ImplicitActuatorCfg(
            joint_names_expr=["telescope_to_upperpassive", "upperpassive_to_lowerpassive"],
            effort_limit_sim=10000.0, velocity_limit_sim=2.0, stiffness=300.0, damping=500.0
        ),
        "grapple_rotate": ImplicitActuatorCfg(
            joint_names_expr=["lowerpassive_to_basegrapple"],
            effort_limit_sim=20000.0, velocity_limit_sim=5.0, stiffness=50_000.0, damping=20_000.0
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["basegrapple_to_gripper.*"],
            effort_limit_sim=7_500.0, velocity_limit_sim=25.0, stiffness=0.0, damping=500.0
        ),
    },
)

# ===== Scene config =====
@configclass
class CraneSceneCfg(InteractiveSceneCfg):
    """Scene configuration with crane articulation and static rack."""

    # Crane articulation
    crane = CRANE_CFG.replace(prim_path="{ENV_REGEX_NS}/Crane")

    # Rack as static asset (logs will be spawned manually for now to avoid timing issues)
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.rack_usd),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(args_cli.rack_x, args_cli.rack_y, 0.0))
    )

# ===== Env config =====
@configclass
class CraneDirectEnvCfgFull(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 600.0

    # Full action space: [x, y, z, yaw]
    # x: target position (front/back on rack)
    # y: target position (left/right along rack)
    # z: target height
    # yaw: grapple rotation to align with log
    action_space = 4

    # Top-N logs observation: (x, y, z, yaw) for top 32 logs sorted by height
    max_logs_obs: int = 32  # Number of logs to include in observation

    # Settling time (seconds) to let logs settle before training/task starts
    # This prevents target selection before logs have stopped moving
    settle_time: float = 2.0

    # Hierarchical RL mode: one step() = one complete pick-place cycle
    # When False: standard RL mode (one step = one physics timestep)
    use_hierarchical_rl: bool = False

    # Gripper closing speed control (prevent penetration when yaw misaligned)
    gripper_close_step: float = 0.10        # Radians per step - moderate speed
    gripper_close_delay_s: float = 0.02     # Seconds between steps
    gripper_max_velocity: float = 10.0      # Max velocity rad/s - moderate speed
    gripper_kp: float = 8.0                 # Proportional gain - lower force to prevent penetration

    # Top 32 logs observation: (x, y, z, yaw) per log = 128 values, normalized to [-1, 1]
    # Logs sorted by height (highest first), empty slots filled with -1
    observation_space = (max_logs_obs * 4)  # 32*4 = 128

    state_space = 0
    action_scale = 1.0
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=decimation)
    scene: CraneSceneCfg = CraneSceneCfg(
        num_envs=1024, env_spacing=args_cli.env_spacing, replicate_physics=True, clone_in_fabric=False
    )
    crane_cfg: ArticulationCfg = CRANE_CFG
    control_joint_names: List[str] = ["basemast_to_mast", "mast_to_mainboom", "mainboom_to_stick", "stick_to_telescope"]

    rack_x: float = args_cli.rack_x
    rack_y: float = args_cli.rack_y
    center_y_world: float = args_cli.center_y
    rows: int = args_cli.rows
    layers: int = args_cli.layers
    spacing_y: float = args_cli.spacing_y
    spacing_z: float = args_cli.spacing_z
    base_z: float = args_cli.base_z
    num_logs: int = args_cli.num_logs
    rack_usd: str = args_cli.rack_usd
    log_usd: str = args_cli.log_usd
    log_mass: float = args_cli.log_mass
    log_roll: float = args_cli.log_roll
    log_pitch: float = args_cli.log_pitch
    log_yaw: float = args_cli.log_yaw

    row_y_jitter: float = args_cli.row_y_jitter
    layer_y_offset: float = args_cli.layer_y_offset
    spawn_height: float = args_cli.spawn_height
    jitter_xy: float = args_cli.jitter_xy
    jitter_height: float = args_cli.jitter_height
    jitter_ang: float = args_cli.jitter_ang

    rew_alive = 0.1
    rew_pos_l2 = -0.25
    rew_vel_l1 = -0.01

    # Domain randomization (set in task configs - tasks.py)
    enable_domain_randomization: bool = False  # Default: no randomization

    # Reward configuration
    # reward_formula: "multiplicative" = efficiency × alignment, "additive" = efficiency + alignment
    reward_formula: str = "multiplicative"
    # normalize_reward: use efficiency (logs/available) instead of raw log count
    normalize_reward: bool = False
    max_graspable_logs: int = 20  # Physical grapple capacity cap for normalization
    failure_penalty: float = -3.0  # Penalty for 0-log grasps (-3.0 for multiplicative, -0.5 for additive)
    # use_stability_reward: penalize off-center grasps that cause grapple tilt
    # multiplicative: reward × stability, additive: + stability term
    use_stability_reward: bool = False
    debug_reward: bool = False  # Print debug info for reward computation

    # Camera configuration (ZED X style RGBD camera)
    enable_camera: bool = False  # Set True to enable camera sensor
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            # Position: user-configured camera view of rack
            # Rotation: quaternion from Isaac Sim (w, x, y, z) = (real, i, j, k)
            pos=(5.0, -1.0, 3.0),  # Camera position from GUI
            rot=(0.6124, 0.3536, 0.3536, 0.6124),  # Quaternion from Isaac Sim
            convention="opengl"  # Use OpenGL/USD native camera convention
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.12,  # ZED X focal length (mm) - wide angle
            focus_distance=5.0,
            horizontal_aperture=5.76,  # ZED X sensor width (mm)
            clipping_range=(0.3, 20.0)  # ZED X depth range
        ),
        width=640,  # Start with lower res for performance (can increase to 1280 or 1920)
        height=360,
    )


# ===== Env =====
class CraneDirectEnvFull(DirectRLEnv):
    """
    Full hierarchical RL environment for crane log grasping.

    Full action space:
    - 4D action space: [x, y, z, yaw] - Policy controls all dimensions
    - Per-log observation: 32 logs × 4 features (x, y, z, yaw) = 128 values

    The policy selects grasp targets and a heuristic FSM executes the pick-place cycle.

    - Task-space control via differential IK
    - Vectorized log spawning (Default 200 logs per environment), domain randomization through different spawn patterns on reset
    - Reward associated to grasp outcome (logs grasped x alignment)
    """
    cfg: CraneDirectEnvCfgFull

    # ---------- tuning ----------

    HOVER_CLEAR    = 2.5    # BG z above top log
    APPROACH_ABOVE = 0.6
    LIFT_CLEAR     = 0.80

    UP_TOL  = 0.10
    BG_TOL  = 0.25
    DWELL_N = 30

    # Gripper closing validation
    GRIPPER_TOL = 0.1           # Joint angle tolerance for gripper validation (radians)
    GRIPPER_STABILITY_TIME = 25 # Timesteps to stay within tolerance (~0.5s at 50Hz)
    GRIPPER_TIMEOUT = 100       # Timeout in timesteps (~2s at 50Hz)
    
    # General phase timeout for stuck recovery
    PHASE_TIMEOUT = 100         # Timeout in timesteps (~2s at 60Hz) for stuck recovery
    DESCEND_TIMEOUT = 300       # Longer timeout for descent phase (~5s at 60Hz)

    PH_HOVER_UP      = 0
    PH_ALIGN_YAW     = 1
    PH_DESCEND       = 2
    PH_CLOSE         = 3
    PH_LIFT_HIGH     = 4
    PH_CARRY_HOME      = 5
    PH_ALIGN_HOME_YAW  = 6
    PH_LOWER_TO_DROP   = 7
    PH_OPEN            = 8
    PH_SETTLE          = 9

    PHASE_NAMES = {
        PH_HOVER_UP:       "HOVER_UP",
        PH_ALIGN_YAW:      "ALIGN_YAW",
        PH_DESCEND:        "DESCEND",
        PH_CLOSE:          "CLOSE",
        PH_LIFT_HIGH:      "LIFT_HIGH",
        PH_CARRY_HOME:     "CARRY_HOME",
        PH_ALIGN_HOME_YAW: "ALIGN_HOME_YAW",
        PH_LOWER_TO_DROP:  "LOWER_TO_DROP",
        PH_OPEN:           "OPEN",
        PH_SETTLE:         "SETTLE",
    }

    # class-level safe defaults (in case of refactors)
    _viz_enabled = False
    _viz = None  # dict with 'ee', 'bg', 'log'
    _viz_proto_idx = {}

    def __init__(self, cfg: CraneDirectEnvCfgFull, render_mode: str | None = None, **kwargs):
        # ---- everything that _setup_scene might read must be set BEFORE super().__init__ ----
        # Debug render: controlled by --viz_markers flag OR debug_reward config
        cli_viz = args_cli.viz_markers if hasattr(args_cli, 'viz_markers') else False
        self._viz_enabled = cli_viz or cfg.debug_reward
        self._viz = None
        self._viz_proto_idx = {}


        if cfg.seed is None:
            cfg.seed = 0

        # Pre-initialize attributes before super().__init__() because _reset_idx may be called
        # during initialization before _setup_scene() sets these values
        init_device = cfg.sim.device if hasattr(cfg, 'sim') and hasattr(cfg.sim, 'device') else "cuda:0"
        qw, qx, qy, qz = euler_deg_to_quat(cfg.log_roll, cfg.log_pitch, cfg.log_yaw)
        self._log_quat = torch.tensor([qw, qx, qy, qz], device=init_device)
        self._logs_settled = False  # Track if logs have been settled

        # This will invoke _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)

        # ---- after scene exists ----
        self.crane: Articulation = self.scene.articulations["crane"]
        self._ctrl_joint_idx, _ = self.crane.find_joints(self.cfg.control_joint_names)
        self._q  = self.crane.data.joint_pos
        self._qd = self.crane.data.joint_vel
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Gym spaces
        act_dim = int(self.cfg.action_space)
        self.single_action_space = spaces.Box(
            low=-np.ones(act_dim, np.float32),
            high=np.ones(act_dim, np.float32),
            shape=(act_dim,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space
        obs_dim = int(self.cfg.observation_space)
        policy_box = spaces.Box(
            low=-np.inf*np.ones(obs_dim, np.float32),
            high=np.inf*np.ones(obs_dim, np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Dict({"policy": policy_box})
        self.observation_space = self.single_observation_space

        # Body IDs
        ee_body_name = args_cli.ee_body
        body_ids, _ = self.crane.find_bodies([ee_body_name])
        if len(body_ids) == 0:
            raise RuntimeError(f"EE body '{ee_body_name}' not found.")
        self._ee_body_id = int(body_ids[0])
        bg_ids, _ = self.crane.find_bodies(["basegrapple"])
        up_ids, _ = self.crane.find_bodies(["upperpassive"])
        if len(bg_ids) == 0 or len(up_ids) == 0:
            raise RuntimeError("Expected bodies 'basegrapple' and 'upperpassive' in the crane.")
        self._basegrapple_body_id = int(bg_ids[0])
        self._upperpassive_body_id = int(up_ids[0])

        # Jacobian index
        self._ee_jacobi_idx = self._ee_body_id - 1 if self.crane.is_fixed_base else self._ee_body_id

        # Initialize IK controller based on CLI flag
        self._use_simple_ik = args_cli.use_simple_ik
        if self._use_simple_ik:
            # Use actual physics timestep for PD control (from SimulationCfg)
            physics_dt = self.sim.get_physics_dt()
            self._simple_ik = SimpleIKController(
                args_cli.urdf_path, self.scene.num_envs, str(self.device), physics_dt,
                gripper_kp=self.cfg.gripper_kp,
                gripper_max_vel=self.cfg.gripper_max_velocity
            )
            self._diffik = None
            print(f"Using SimpleIK with PD velocity control (dt={physics_dt:.4f}s) - URDF: {args_cli.urdf_path}")
        else:
            # Differential IK - DLS with default parameters  
            dik_cfg = DifferentialIKControllerCfg(
                command_type="position", 
                use_relative_mode=False, 
                ik_method="dls"
            )
            self._diffik = DifferentialIKController(dik_cfg, num_envs=self.scene.num_envs, device=self.device)
            self._simple_ik = None
            print("Using DifferentialIK controller")

        # Command buffers
        self._ee_goal = torch.zeros(self.num_envs, 7, device=self.device)   # base-frame EE target (pos3, quat4)
        # Initialize quaternion part to identity quaternion
        self._ee_goal[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._joint_pos_des = torch.zeros(self.num_envs, len(self._ctrl_joint_idx), device=self.device)

        # Logs metadata (self._logs_obj already initialized before super().__init__())
        self._per_env_target: int = 0
        self._log_origins_world: torch.Tensor = torch.empty((0, 3), device=self.device)
        # Move _log_quat (initialized before super().__init__()) to the correct device
        self._log_quat = self._log_quat.to(self.device)
        self._bootstrap_done = False

        # Heuristic per-env state
        self._phase = torch.full((self.num_envs,), self.PH_HOVER_UP, dtype=torch.int64, device=self.device)
        self._has_log = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._yaw_targets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._pending_yaw_targets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)  # Store policy yaw for ALIGN_YAW
        self._dwell = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Hold timer for keeping logs lifted before evaluation (for visual feedback)
        self._lift_hold_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.LIFT_HOLD_DURATION = 60  # ~1 second at 60Hz to hold logs before despawn

        # Gripper closing validation state
        self._gripper_stability_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._gripper_timeout_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        
        
        # General phase timeout for stuck recovery
        self._phase_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Track which log is currently targeted
        self._current_target_log_id = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)  # -1 = no target

        # Per-pick frozen target log pos (BG frame) and quaternion (world frame) for viz
        self._target_log_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_log_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._dbg_target_bg    = torch.zeros(self.num_envs, 3, device=self.device)
        # Flag to track if target has been set for current pick cycle
        self._target_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Flag to track if heuristic has set first meaningful target
        self._heuristic_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Frozen target positions for HOVER and ALIGN phases (hover_xyz)
        self._frozen_target_bg = torch.zeros(self.num_envs, 3, device=self.device)
        self._frozen_target_upperpassive = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Flag to enable live log pose tracking (set to True after logs become accessible)
        self._use_live_log_poses = True  # Enable immediately - logs should be accessible after sim.reset()

        # Note: _logs_settled is initialized before super().__init__() to avoid AttributeError in _reset_idx

        # --- Target-selection learning buffers ---
        self.SELECT_RADIUS = 0.75  # [m] selection must land this close (xy) to count
        N = self.num_envs

        self._sel_mask     = torch.zeros(N, dtype=torch.bool, device=self.device)   # true only on the selection frame

        self._sel_xy       = torch.zeros(N, 2, device=self.device)
        self._sel_z        = torch.zeros(N, device=self.device)
        self._top_xy       = torch.zeros(N, 2, device=self.device)
        self._top_z        = torch.zeros(N, device=self.device)

        self._sel_is_valid = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Grippers
        g_ids, g_names = self.crane.find_joints(["basegrapple_to_gripperleft", "basegrapple_to_gripperright"])
        if len(g_ids) != 2:
            g_ids, g_names = self.crane.find_joints(["basegrapple_to_gripper.*"])
            if len(g_ids) != 2:
                raise RuntimeError(f"Could not find gripper joints. Found: ids={g_ids}, names={g_names}")
        self._grip_joint_ids = (int(g_ids[0]), int(g_ids[1]))
        self._grip_joint_names = (g_names[0], g_names[1])
        self._grip_open = float(args_cli.heu_grip_open)
        self._grip_closed = float(args_cli.heu_grip_closed)

        # gripper PD parameters (available regardless of SimpleIK flag)
        physics_dt = self.sim.get_physics_dt()
        self._gripper_control_dt = physics_dt
        # Use configurable parameters for gentler control (prevent penetration)
        self.KP_GRIP = torch.tensor(
            [self.cfg.gripper_kp, self.cfg.gripper_kp],
            device=self.device,
            dtype=torch.float32
        )
        self.KD_GRIP = torch.tensor([1.0, 1.0], device=self.device, dtype=torch.float32)
        self.MAX_VEL_GRIP = torch.tensor(
            [self.cfg.gripper_max_velocity, self.cfg.gripper_max_velocity],
            device=self.device,
            dtype=torch.float32
        )
        
        # discrete gripper stepping
        self.GRIP_OPEN = 0.05   # Gripper open position (radians)
        self.GRIP_MAX = 2.23    # Gripper maximum closed position (radians)

        # Use configurable parameters (prevent penetration)
        self.CLOSE_STEP = self.cfg.gripper_close_step
        self.CLOSE_DELAY_STEPS = int(self.cfg.gripper_close_delay_s / physics_dt)
        
        # Gripper PD control state per environment (2 grippers)
        self.q_des_grip = torch.full((self.num_envs, 2), self.GRIP_OPEN, device=self.device, dtype=torch.float32)
        self.prev_error_grip = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        
        # Per-environment gripper stepping state
        self._gripper_step_target = torch.full((self.num_envs,), self.GRIP_OPEN, device=self.device, dtype=torch.float32)
        self._gripper_step_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._gripper_stepping_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Trailer / drop
        self._trailer_x_min = float(args_cli.trailer_x_min)
        self._trailer_x_max = float(args_cli.trailer_x_max)
        self._trailer_y_min = float(args_cli.trailer_y_min)
        self._trailer_y_max = float(args_cli.trailer_y_max)
        self._drop_clearance = float(args_cli.drop_clearance)
        self._lane_spacing_y = float(args_cli.lane_spacing_y)
        self._lane_count = int(args_cli.lane_count)

        # Dual-stack system parameters for drop positioning
        # HOME_REAR and HOME_FRONT positions in base frame
        self._home_rear = torch.tensor([0.0, 2.2, 2.0], device=self.device, dtype=torch.float32)
        self._home_front = torch.tensor([0.0, 4.85, 2.5], device=self.device, dtype=torch.float32)
        self._stack_switch_threshold = 100  # Number of logs before switching to front stack
        self._drop_height_offset = 1.0  # Height above highest trailer log to drop new log
        
        # Stack state management (per environment)
        self._current_stack = ["REAR"] * self.num_envs  # Track which stack each env is using
        self._deposited_logs = [set() for _ in range(self.num_envs)]  # Track deposited logs per env
        self._failed_grasp_counts = [{} for _ in range(self.num_envs)]  # {log_id: fail_count} per env

        # Episode tracking for training
        self._cycle_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)  # Cycles per episode
        self._logs_at_cycle_start = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)  # Logs before current cycle

        # Reward component tracking for TensorBoard logging
        self._last_throughput_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._last_alignment_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_throughput_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_alignment_sum = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Grasp outcome tracking for hierarchical RL
        self._grasp_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._prev_logs_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._prev_grasp_alignment = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Episode-level metrics for TensorBoard (reset per episode)
        self._episode_successful_grasps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_failed_grasps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_total_logs_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_alignment_sum_weighted = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Sum of (logs * alignment)
        self._episode_stability_sum_weighted = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Sum of (logs * stability)
        self._prev_grasp_stability = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Last grasp stability

        # Logs available at target position (for normalized reward computation)
        self._logs_available_at_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Cylinder visualization position (world frame) for debug
        self._debug_cylinder_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._debug_cylinder_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Track logs knocked out of bounds (for penalty)
        self._logs_knocked_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._logs_out_of_bounds_penalty = 0.0  # No penalty - let policy explore freely
        self._initial_settling_complete = False  # Track if initial settling has finished

        # Cache drop goals to avoid repeated calculations and debug prints
        self._cached_drop_goals = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._drop_goal_calculated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Cache action space bounds for visualization
        # Will be computed from actual log positions after logs are created
        self._action_bounds_min = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._action_bounds_max = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._action_bounds_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Debug cadence
        self._dbg_every = max(1, int(args_cli.debug_every_steps))
        self._dbg_envs_max = max(1, int(args_cli.debug_envs_max))
        self._in_hierarchical_loop = False  # Flag to suppress debug prints during internal physics loop

    # ---------------- Hierarchical RL Step Override ----------------
    def step(self, action: torch.Tensor):
        """
        Step the environment.

        NO-DEPOSITION MODE: Always uses hierarchical stepping.
        ONE step() call = ONE complete pick-place cycle (no deposition)
            - Policy selects target at HOVER_UP
            - Heuristic FSM runs internally until LIFT_HIGH completes
            - Grasp is evaluated, logs despawned, crane returns to HOVER_UP
            - Returns (obs, reward, done) for the completed grasp
        """
        # No-deposition version always uses hierarchical mode
        # 1. Process action (target selection) at HOVER_UP
        action = action.to(self.device)

        # Suppress debug prints during internal loop to avoid terminal spam (set BEFORE _pre_physics_step)
        self._in_hierarchical_loop = True

        self._pre_physics_step(action)

        # 2. Run physics loop until all envs complete their grasp attempt
        # Each phase has PHASE_TIMEOUT, so cycles naturally complete or timeout
        # _cycle_complete_this_step prevents envs from starting new cycles within this step()
        MAX_STEPS = 1500  # Safety limit only - should never hit this

        self._cycle_complete_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for step_count in range(MAX_STEPS):
            # Step physics simulation
            # Only render during loop if GUI is open (for visual debugging)
            # Camera sensors are updated once at end of cycle by the wrapper, not every 4 steps
            should_render = self.sim.has_gui() and (step_count % 4 == 0)
            self.sim.step(render=should_render)

            # Update buffers
            self.scene.update(self.physics_dt)

            # Apply heuristic actions (IK controller)
            self._apply_action()

            # Update heuristic state machine (handles grasp evaluation and transition to HOVER_UP)
            self._heuristic_step()

            # Exit as soon as all envs have completed their grasp
            if self._cycle_complete_this_step.all():
                break

        # Safety: if we somehow hit max steps, give timeout penalty to incomplete envs
        if step_count >= MAX_STEPS - 1:
            for i in range(self.num_envs):
                if not self._cycle_complete_this_step[i]:
                    print(f"[env{i}] SAFETY TIMEOUT: phase={self.PHASE_NAMES[int(self._phase[i])]}, penalty=-2.0")
                    self._grasp_reward_buf[i] = -2.0
                    self._prev_logs_grasped[i] = 0.0
                    self._prev_grasp_alignment[i] = 0.0

        # Re-enable debug prints
        self._in_hierarchical_loop = False

        # 4. Compute observations, rewards, dones
        self.obs_buf = self._get_observations()
        self.reward_buf = self._grasp_reward_buf.clone()

        # Increment cycle count
        self._cycle_count += 1

        # Check termination: rack empty (success) OR 30 cycles (timeout)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self.num_envs):
            logs_remaining = self._count_logs_in_rack(i)
            if logs_remaining == 0:  # Rack empty = success
                terminated[i] = True
            elif self._cycle_count[i] >= 30:  # Timeout (reduced from 50 for no-deposition mode)
                terminated[i] = True

        truncated = torch.zeros_like(terminated)

        # Reset terminated envs
        reset_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        # Periodic garbage collection to prevent memory leaks during long training runs
        # Clear unused GPU memory every 100 steps to avoid CUDA OOM errors
        if not hasattr(self, '_gc_counter'):
            self._gc_counter = 0
        self._gc_counter += 1
        if self._gc_counter % 100 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # 5. Populate extras with episode metrics for logging
        self.extras.update(self._get_extras())

        # 6. Return only at decision points
        return self.obs_buf, self.reward_buf, terminated, truncated, self.extras


    # ---------------- Scene setup ----------------
    def _setup_scene(self):
        """Initialize simulation scene with ground, lighting, crane, and log rack."""
        # keep existing guard for viz fields
        if not hasattr(self, "_viz_enabled"):
            self._viz_enabled = False
            self._viz = None
            self._viz_proto_idx = {}

        # ground + light (crane and logs are now handled by scene config)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        dome = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        dome.func("/World/Light", dome)

        # build env_0 rack + Origins and then clone to envs
        self._make_env0_rack_and_origins()
        self.scene.clone_environments(copy_from_source=True)

        # Create logs RigidObject after rack structure exists (Isaac Lab standard timing pattern)
        self._create_logs_rigidobject()

        # capacity / positions cache
        per_env_capacity = self.cfg.rows * self.cfg.layers
        self._per_env_target = min(per_env_capacity, self.cfg.num_logs)
        self._rebuild_log_origins_world()

        # store default log orientation (kept for reset/init usage)
        qw, qx, qy, qz = euler_deg_to_quat(self.cfg.log_roll, self.cfg.log_pitch, self.cfg.log_yaw)
        self._log_quat = torch.tensor([qw, qx, qy, qz], device=self.device)

        # visuals/material hygiene
        try:
            force_opaque_materials_under("/World/envs")
            self.sim.render()
        except Exception:
            pass

        # optional: determinism flag
        if args_cli.enhanced_determinism and PhysxSchema is not None:
            try:
                scene_prim = prim_utils.get_prim_at_path("/World/physicsScene")
                api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
                if hasattr(api, "CreateEnableEnhancedDeterminismAttr"):
                    api.CreateEnableEnhancedDeterminismAttr(True)
            except Exception:
                pass

        # viz markers
        if self._viz_enabled and self._viz is None:
            self._viz = self._init_markers()

        # Camera setup (if enabled)
        self._camera = None
        if getattr(self.cfg, 'enable_camera', False):
            self._camera = TiledCamera(self.cfg.camera_cfg)
            self.scene.sensors["camera"] = self._camera
            print(f"[INFO]: Camera enabled - {self.cfg.camera_cfg.width}x{self.cfg.camera_cfg.height}, data_types={self.cfg.camera_cfg.data_types}")

        # done
        self._bootstrap_done = True
        print(f"[INFO]: Env ready. Running heuristic baseline.")
    
    def _calc_optimal_yaw_feedback(self, env_i: int) -> float:
        """Return target yaw (radians, in (-pi,pi]) for the grapple yaw joint, base-frame."""
        import math

        def wrap(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def quat_conj_wxyz(q):  # (w,x,y,z)
            return (q[0], -q[1], -q[2], -q[3])

        def quat_mul_wxyz(a, b):
            aw, ax, ay, az = a; bw, bx, by, bz = b
            return (
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            )

        def yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
            # yaw about Z (ZYX)
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # -- current yaw joint value  
        yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
        if len(yaw_joint_ids) == 0:
            return 0.0
        yaw_joint_id = int(yaw_joint_ids[0])
        current_yaw_joint = float(self.crane.data.joint_pos[env_i, yaw_joint_id].item())

        # -- base orientation (world->base as conjugate)
        base_quat_w = self.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)
        
        # Read actual basegrapple orientation
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]
        
        # Transform basegrapple to base frame
        q_bg_b = quat_mul_wxyz(q_base_inv, (float(bg_quat_w[0]), float(bg_quat_w[1]),
                                           float(bg_quat_w[2]), float(bg_quat_w[3])))
        current_bg_z_rotation = yaw_from_quat_wxyz(*q_bg_b)

        # -- get log quat in world (prefer stored target quat if valid)
        q_log_w = None
        try:
            # Use stored target quaternion if valid (regardless of _target_frozen)
            qf = self._target_log_quat_w[env_i]
            if torch.isfinite(qf).all() and qf.abs().sum().item() > 0.0:
                q_log_w = (float(qf[0]), float(qf[1]), float(qf[2]), float(qf[3]))
        except Exception:
            pass

        if q_log_w is None:
            try:
                pos_w, quat_w = self._get_logs_root_pose_w()
                if pos_w is not None and quat_w is not None and self._per_env_target > 0:
                    per_env = int(self._per_env_target)
                    s = env_i * per_env
                    e = min(s + per_env, self._logs_obj.num_instances)
                    if e > s:
                        k = int(torch.argmax(pos_w[s:e, 2]).item())
                        q = quat_w[s + k]
                        q_log_w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            except Exception:
                pass

        if q_log_w is None:
            # no info → keep current yaw joint value
            return current_yaw_joint

        # -- log yaw in base frame
        q_log_b = quat_mul_wxyz(q_base_inv, q_log_w)
        log_yaw_b = yaw_from_quat_wxyz(*q_log_b)

        # -- calculate rotation difference between log and basegrapple Z-rotations
        # No offset needed - grapple fingers should align with log's long axis directly
        rotation_diff = wrap(log_yaw_b - current_bg_z_rotation)

        # -- apply to yaw joint with 180° flip option
        target_yaw1 = current_yaw_joint + rotation_diff
        target_yaw2 = current_yaw_joint + rotation_diff + math.pi

        # Choose minimal rotation
        yaw1 = wrap(target_yaw1)
        yaw2 = wrap(target_yaw2)

        diff1 = abs(wrap(yaw1 - current_yaw_joint))
        diff2 = abs(wrap(yaw2 - current_yaw_joint))

        optimal_yaw = yaw1 if diff1 <= diff2 else yaw2


        return optimal_yaw

    def _calc_optimal_yaw_for_trailer_alignment(self, env_i: int, target_z_rotation: float = 0.0) -> float:
        """Calculate optimal yaw to align basegrapple with trailer/base frame orientation."""
        import math

        def wrap(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def quat_conj_wxyz(q):  # (w,x,y,z)
            return (q[0], -q[1], -q[2], -q[3])

        def quat_mul_wxyz(a, b):
            aw, ax, ay, az = a; bw, bx, by, bz = b
            return (
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            )

        def yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
            # yaw about Z (ZYX)
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # -- current yaw joint value  
        yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
        if len(yaw_joint_ids) == 0:
            return 0.0
        yaw_joint_id = int(yaw_joint_ids[0])
        current_yaw_joint = float(self.crane.data.joint_pos[env_i, yaw_joint_id].item())

        # -- base orientation (world->base as conjugate)
        base_quat_w = self.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)
        
        # Read actual basegrapple orientation
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]
        
        # Transform basegrapple to base frame
        q_bg_b = quat_mul_wxyz(q_base_inv, (float(bg_quat_w[0]), float(bg_quat_w[1]),
                                           float(bg_quat_w[2]), float(bg_quat_w[3])))
        current_bg_z_rotation = yaw_from_quat_wxyz(*q_bg_b)

        # -- calculate rotation difference between target (trailer) and basegrapple Z-rotations
        rotation_diff = wrap(target_z_rotation - current_bg_z_rotation)

        # -- apply to yaw joint with 180° flip option
        target_yaw1 = current_yaw_joint + rotation_diff
        target_yaw2 = current_yaw_joint + rotation_diff + math.pi

        # Choose minimal rotation
        yaw1 = wrap(target_yaw1)
        yaw2 = wrap(target_yaw2)

        diff1 = abs(wrap(yaw1 - current_yaw_joint))
        diff2 = abs(wrap(yaw2 - current_yaw_joint))

        optimal_yaw = yaw1 if diff1 <= diff2 else yaw2

        return optimal_yaw

    def _calc_optimal_yaw_for_rack_perpendicular(self, env_i: int) -> float:
        """Calculate optimal yaw to orient grapple perpendicular to rack.

        The rack has identity orientation in world frame (yaw=0).
        To be perpendicular to the rack (which extends along Y), we need
        to orient the grapple along X, which is yaw=π/2 in world frame.

        This transforms the target orientation to base frame and applies
        the same yaw joint calculation with 180° flip option.
        """
        import math

        def wrap(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def quat_conj_wxyz(q):  # (w,x,y,z)
            return (q[0], -q[1], -q[2], -q[3])

        def quat_mul_wxyz(a, b):
            aw, ax, ay, az = a; bw, bx, by, bz = b
            return (
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw,
            )

        def yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
            # yaw about Z (ZYX)
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # -- current yaw joint value
        yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
        if len(yaw_joint_ids) == 0:
            return 0.0
        yaw_joint_id = int(yaw_joint_ids[0])
        current_yaw_joint = float(self.crane.data.joint_pos[env_i, yaw_joint_id].item())

        # -- base orientation (world->base as conjugate)
        base_quat_w = self.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)

        # Read actual basegrapple orientation
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]

        # Transform basegrapple to base frame
        q_bg_b = quat_mul_wxyz(q_base_inv, (float(bg_quat_w[0]), float(bg_quat_w[1]),
                                           float(bg_quat_w[2]), float(bg_quat_w[3])))
        current_bg_z_rotation = yaw_from_quat_wxyz(*q_bg_b)

        # -- Rack orientation in world frame: identity quaternion (yaw=0)
        # To be PERPENDICULAR to rack (which extends along Y), we need yaw=π/2 in world
        rack_yaw_world = math.pi / 2.0  # 90 degrees - perpendicular to Y axis

        # Create quaternion for this target orientation in world frame
        # quat for pure Z rotation: (cos(θ/2), 0, 0, sin(θ/2))
        half_angle = rack_yaw_world / 2.0
        q_rack_perpendicular_w = (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))

        # Transform to base frame
        q_target_b = quat_mul_wxyz(q_base_inv, q_rack_perpendicular_w)
        target_yaw_b = yaw_from_quat_wxyz(*q_target_b)

        # -- calculate rotation difference between target and basegrapple Z-rotations
        rotation_diff = wrap(target_yaw_b - current_bg_z_rotation)

        # -- apply to yaw joint with 180° flip option
        target_yaw1 = current_yaw_joint + rotation_diff
        target_yaw2 = current_yaw_joint + rotation_diff + math.pi

        # Choose minimal rotation
        yaw1 = wrap(target_yaw1)
        yaw2 = wrap(target_yaw2)

        diff1 = abs(wrap(yaw1 - current_yaw_joint))
        diff2 = abs(wrap(yaw2 - current_yaw_joint))

        optimal_yaw = yaw1 if diff1 <= diff2 else yaw2

        return optimal_yaw

    def _find_yaw_joint_id(self) -> int | None:
        """Find the index of the yaw joint in the crane's joint list (not controlled joints)."""
        yaw_joint_names = ["lowerpassive_to_basegrapple"]
        for name in yaw_joint_names:
            joint_ids, _ = self.crane.find_joints([name])
            if len(joint_ids) > 0:
                # Return the actual joint ID in the crane, not the controlled joint index
                return joint_ids[0]
        return None
    
    def _build_target_selection_obs(self):
        """
        Top 32 logs observation for target selection.

        Returns (x, y, z, yaw) for top 32 logs sorted by height (highest first).
        Removed logs are excluded. Empty slots filled with sentinel (-1, -1, -1, -1).
        All values normalized to [-1, 1] based on action bounds.

        Returns [num_envs, 128] tensor (32 logs × 4 values each).
        """
        device = self.device
        N = self.num_envs
        max_logs = self.cfg.max_logs_obs  # 32

        # Output: [N, max_logs * 4] - each log has (x, y, z, yaw)
        obs_dim = max_logs * 4
        # Initialize with sentinel values (-1 = no log)
        all_obs = torch.full((N, obs_dim), -1.0, device=device, dtype=torch.float32)

        # Get log positions
        log_pos_w, log_quat_w = self._get_logs_root_pose_w()
        if log_pos_w is None or log_quat_w is None:
            return all_obs

        root_w = self.crane.data.root_pose_w
        root_pos_w = root_w[:, 0:3]
        root_quat_w = root_w[:, 3:7]

        for env_i in range(N):
            # Ensure action bounds are computed
            if not self._action_bounds_valid[env_i]:
                self._compute_action_space_bounds()

            min_b = self._action_bounds_min[env_i]
            max_b = self._action_bounds_max[env_i]
            x_min, x_max = min_b[0].item(), max_b[0].item()
            y_min, y_max = min_b[1].item(), max_b[1].item()
            z_min, z_max = min_b[2].item(), max_b[2].item()

            # Get available logs for this environment
            per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else 200
            start = env_i * 200
            end = start + per_env

            # Collect available (non-deposited) logs
            available_indices = []
            for log_idx in range(per_env):
                global_log_idx = start + log_idx
                if global_log_idx not in self._deposited_logs[env_i]:
                    available_indices.append(log_idx)

            if len(available_indices) == 0:
                continue

            # Get log positions and orientations
            available_indices_tensor = torch.tensor(available_indices, device=device, dtype=torch.long)
            env_log_pos_w = log_pos_w[start:end][available_indices_tensor]
            env_log_quat_w = log_quat_w[start:end][available_indices_tensor]
            num_logs = len(available_indices)

            # Transform to base frame
            root_pos_w_i = root_pos_w[env_i:env_i+1].expand(num_logs, -1)
            root_quat_w_i = root_quat_w[env_i:env_i+1].expand(num_logs, -1)
            log_pos_b, log_quat_b = subtract_frame_transforms(
                root_pos_w_i, root_quat_w_i, env_log_pos_w, env_log_quat_w
            )

            # Extract yaw from quaternions (rotation about Z axis)
            log_yaw = torch.atan2(
                2.0 * (log_quat_b[:, 0] * log_quat_b[:, 3] + log_quat_b[:, 1] * log_quat_b[:, 2]),
                1.0 - 2.0 * (log_quat_b[:, 2] ** 2 + log_quat_b[:, 3] ** 2)
            )

            # Sort by height (z) descending - highest logs first
            log_z = log_pos_b[:, 2]
            sorted_indices = torch.argsort(log_z, descending=True)

            # Take top max_logs
            num_to_use = min(num_logs, max_logs)
            top_indices = sorted_indices[:num_to_use]

            # Get (x, y, z, yaw) for top logs and normalize to [-1, 1]
            for i, idx in enumerate(top_indices):
                x = log_pos_b[idx, 0].item()
                y = log_pos_b[idx, 1].item()
                z = log_pos_b[idx, 2].item()
                yaw = log_yaw[idx].item()

                # Normalize positions to [-1, 1]
                x_norm = 2.0 * (x - x_min) / (x_max - x_min + 1e-6) - 1.0
                y_norm = 2.0 * (y - y_min) / (y_max - y_min + 1e-6) - 1.0
                z_norm = 2.0 * (z - z_min) / (z_max - z_min + 1e-6) - 1.0
                # Normalize yaw to [-1, 1] (yaw is in [-pi, pi])
                yaw_norm = yaw / 3.14159

                # Clamp to [-1, 1]
                x_norm = max(min(x_norm, 1.0), -1.0)
                y_norm = max(min(y_norm, 1.0), -1.0)
                z_norm = max(min(z_norm, 1.0), -1.0)
                yaw_norm = max(min(yaw_norm, 1.0), -1.0)

                # Store in observation: [x, y, z, yaw] per log
                obs_idx = i * 4
                all_obs[env_i, obs_idx] = x_norm
                all_obs[env_i, obs_idx + 1] = y_norm
                all_obs[env_i, obs_idx + 2] = z_norm
                all_obs[env_i, obs_idx + 3] = yaw_norm

        return all_obs

    def _compute_action_space_bounds(self) -> None:
        """Compute per-env action-space bounds (in crane base frame) aligned to the rack layout."""
        cfg = self.cfg
        N = self.num_envs

        # Lazily allocate buffers
        if not hasattr(self, "_action_bounds_min"):
            self._action_bounds_min   = torch.zeros((N, 3), device=self.device)
            self._action_bounds_max   = torch.zeros((N, 3), device=self.device)
            self._action_bounds_valid = torch.zeros((N,), dtype=torch.bool, device=self.device)

        # --- Rack/layout extents in WORLD ---
        y_half_span = 0.5 * (cfg.rows - 1) * cfg.spacing_y
        rack_center_y_w = cfg.center_y_world
        y_min_w = rack_center_y_w - y_half_span
        y_max_w = rack_center_y_w + y_half_span
        # Fixed Z span above rack base for the action-bounds box (rack-anchored, DR-invariant)
        # You can override by adding cfg.rack_z_span in your task config.
        rack_z_span = float(getattr(cfg, "rack_z_span", 0.6))  # meters above rack base

        # --- Tunable margins (base-frame box size) ---
        # Wider in X so the policy can choose end-grasps (increase if your logs are longer)
        margin_x_back  = 1   # was 0.25
        margin_x_front = 1   # was 0.50

        # Y overhang (keep some slack so you can hit edge logs / shifted patterns)
        margin_y = 1.15        # Increased from 1.00 to prevent edge logs from being despawned

        # Much shorter in Z: just enough above the stack for grasp points (hover is handled elsewhere)
        margin_z_top = 0.35    # was 1.50
        margin_z_bot = 0.05    # allow a tiny bit below rack base

        root_pose_w = self.crane.data.root_pose_w  # [N, 7]

        for env_i in range(N):
            base_pos_w  = root_pose_w[env_i, 0:3].unsqueeze(0)  # [1,3]
            base_quat_w = root_pose_w[env_i, 3:7].unsqueeze(0)  # [1,4]

            # Get environment origin offset for this env
            env_origin = self.scene.env_origins[env_i]

            # Rack anchor point in WORLD (at rack base height) - offset by env origin
            rack_anchor_w = torch.tensor(
                [[env_origin[0] + cfg.rack_x, env_origin[1] + rack_center_y_w, env_origin[2] + cfg.base_z]],
                device=self.device,
                dtype=torch.float32,
            )
            quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)

            # rack anchor expressed in BASE frame
            rack_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, rack_anchor_w, quat_w)  # [1,3]

            # Local offsets around the rack anchor (still in WORLD, but relative to rack_center_y_w / cfg.base_z)
            rack_y_min_local = y_min_w - rack_center_y_w
            rack_y_max_local = y_max_w - rack_center_y_w
            rack_z_top_local = rack_z_span  # fixed height above rack base (independent of pile pattern)

            # Build AABB in BASE frame (IMPORTANT: anchor Z at rack_pos_b[0,2])
            min_bounds = torch.tensor(
                [
                    rack_pos_b[0, 0] - margin_x_back,
                    rack_pos_b[0, 1] + rack_y_min_local - margin_y,
                    rack_pos_b[0, 2] - margin_z_bot,
                ],
                device=self.device,
            )

            max_bounds = torch.tensor(
                [
                    rack_pos_b[0, 0] + margin_x_front,
                    rack_pos_b[0, 1] + rack_y_max_local + margin_y,
                    rack_pos_b[0, 2] + rack_z_top_local + margin_z_top,
                ],
                device=self.device,
            )

            self._action_bounds_min[env_i] = min_bounds
            self._action_bounds_max[env_i] = max_bounds
            self._action_bounds_valid[env_i] = True

            # Only print bounds in debug mode (not during RL training)
            if not self._in_hierarchical_loop and args_cli.debug_every_steps > 0:
                print(
                    f"[env{env_i}] ACTION-BOUNDS (rack-based): "
                    f"x=[{min_bounds[0]:.2f},{max_bounds[0]:.2f}], "
                f"y=[{min_bounds[1]:.2f},{max_bounds[1]:.2f}], "
                f"z=[{min_bounds[2]:.2f},{max_bounds[2]:.2f}]"
            )


    def _set_movement_speed(self, phase: int, env_i: int, current_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """Apply command scaling for controlled movement phases."""
        # Define step sizes for different phases
        if phase in [self.PH_DESCEND, self.PH_LOWER_TO_DROP]:
            step_size = 0.15  # Slow descent to avoid disturbing pile
        elif phase == self.PH_LIFT_HIGH:
            step_size = 0.3  # Controlled lift
        elif phase == self.PH_CARRY_HOME:
            step_size = 0.5  # Faster carry
        else:
            step_size = 1.0  # Full speed for positioning phases

        # Interpolate toward target
        return current_pos + (target_pos - current_pos) * step_size

    def _init_markers(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        viz = {}

        # Target basegrapple (derived target)
        bg_target_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/BG_TARGET",
            markers={
                "bg_frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.45, 0.45, 0.45),
                ),
            },
        )
        viz["bg_target"] = VisualizationMarkers(bg_target_cfg)

        # Target log (red sphere)
        log_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/LOG",
            markers={
                "log_sphere": sim_utils.SphereCfg(
                    radius=0.15,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        viz["log"] = VisualizationMarkers(log_cfg)

        # Action space bounding box (translucent green box showing valid target region)
        bbox_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/ACTION_BOUNDS",
            markers={
                "bbox": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),  # Will be scaled per-env
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.8, 0.2),
                        opacity=0.15,
                    ),
                ),
            },
        )
        viz["action_bounds"] = VisualizationMarkers(bbox_cfg)

        # Grasp count prism (shows rack slice volume used for counting available logs)
        # Only visible when debug_reward is enabled
        prism_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/GRASP_PRISM",
            markers={
                "prism": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),  # Will be scaled per-env
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.5, 1.0),  # Blue
                        opacity=0.2,
                    ),
                ),
            },
        )
        viz["grasp_prism"] = VisualizationMarkers(prism_cfg)

        return viz

    # ---------------- Rack / logs authorship ----------------
    def _make_env0_rack_and_origins(self):
        stage = omni.usd.get_context().get_stage()
        env0 = "/World/envs/env_0"
        rack_path = f"{env0}/Rack"
        if prim_utils.get_prim_at_path(rack_path):
            stage.RemovePrim(Sdf.Path(rack_path))
        rcfg = sim_utils.UsdFileCfg(usd_path=self.cfg.rack_usd)
        rcfg.func(rack_path, rcfg, translation=(self.cfg.rack_x, self.cfg.rack_y, 0.0))
        logs_anchor = f"{rack_path}/LogsAnchor"
        if prim_utils.get_prim_at_path(logs_anchor):
            stage.RemovePrim(Sdf.Path(logs_anchor))
        define_xform_idem(logs_anchor, translation=(0.0, 0.0, 0.0))

        center_y_local = self.cfg.center_y_world - self.cfg.rack_y
        yz_local = plan_grid_yz(
            center_y_local=center_y_local,
            rows=self.cfg.rows,
            layers=self.cfg.layers,
            spacing_y=self.cfg.spacing_y,
            spacing_z=self.cfg.spacing_z,
            base_z=self.cfg.base_z,
            cap=min(self.cfg.rows * self.cfg.layers, self.cfg.num_logs),
            row_y_jitter=self.cfg.row_y_jitter,
            layer_y_offset=self.cfg.layer_y_offset,
            seed=self.cfg.seed,
        )
        qw, qx, qy, qz = euler_deg_to_quat(self.cfg.log_roll, self.cfg.log_pitch, self.cfg.log_yaw)
        for j, (y_local, z) in enumerate(yz_local):
            define_xform_idem(
                f"{logs_anchor}/Origin{j}",
                translation=(0.0, y_local, z),
                quat_wxyz=(qw, qx, qy, qz),
            )
        return yz_local

    def _create_logs_rigidobject(self):
        """Create logs RigidObject after rack structure exists (Isaac Lab pattern)."""
        # Create RigidObject configuration
        ro_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Rack/LogsAnchor/Origin.*/Log",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.cfg.log_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.0,
                    angular_damping=0.05,
                    max_depenetration_velocity=3.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.log_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.02,  # Larger offset = earlier detection (2cm safety margin)
                    rest_offset=0.0,
                ),
                # Semantic label for camera segmentation
                semantic_tags=[("class", "log")],
            ),
        )
        
        # Create RigidObject and add to scene
        logs_obj = RigidObject(cfg=ro_cfg)
        self.scene._rigid_objects["logs"] = logs_obj
        self._logs_obj = logs_obj

    # ---------------------------------------------

    def _rebuild_log_origins_world(self, randomize_patterns=False, env_ids=None):
        """Build log spawn positions.

        Args:
            randomize_patterns: If True, randomly select pattern for each env (domain randomization)
            env_ids: If specified, only rebuild patterns for these environments. If None, rebuild all.
        """
        per_env_cap = int(self.cfg.rows * self.cfg.layers)
        default_per_env_target = int(min(per_env_cap, self.cfg.num_logs))

        # Initialize per-env target tracking if needed
        if not hasattr(self, '_per_env_log_counts'):
            self._per_env_log_counts = torch.zeros(self.scene.num_envs, dtype=torch.int32, device=self.device)

        # Determine which environments to rebuild
        if env_ids is None:
            # First reset: build all environments
            envs_to_rebuild = range(self.scene.num_envs)
            rebuild_all = True
        else:
            # Subsequent resets: only rebuild specified environments
            # Convert tensor to list if necessary
            if isinstance(env_ids, torch.Tensor):
                envs_to_rebuild = env_ids.cpu().tolist()
            elif hasattr(env_ids, '__iter__'):
                envs_to_rebuild = list(env_ids)
            else:
                envs_to_rebuild = [env_ids]
            rebuild_all = False

        # If rebuilding all, create new tensor; otherwise update existing
        if rebuild_all:
            all_world = []

        for env_id in envs_to_rebuild:
            env_o = self.scene.env_origins[env_id]
            rack_world_x = env_o[0] + self.cfg.rack_x

            # Randomize log count for domain randomization (20-200 logs)
            if randomize_patterns:
                import random
                per_env_target = random.randint(20, 200)
                self._per_env_log_counts[env_id] = per_env_target
            else:
                per_env_target = default_per_env_target
                self._per_env_log_counts[env_id] = per_env_target

            # Select pattern function
            if randomize_patterns:
                # DOMAIN RANDOMIZATION: Truly random patterns
                pattern_func = plan_grid_yz_random
            elif args_cli.mixed_log_patterns:
                # Fixed patterns distributed across environments
                pattern_functions = [plan_grid_yz, plan_grid_yz_pattern_b, plan_grid_yz_pattern_c]
                pattern_names = ["Original (30×10)", "Tower-Left (10×30)", "Tower-Right (10×30)"]
                pattern_idx = env_id % 3
                pattern_func = pattern_functions[pattern_idx]
                if not hasattr(self, '_patterns_printed'):
                    print(f"Environment {env_id}: Using {pattern_names[pattern_idx]} log grid pattern")
            else:
                # Default: original pattern for all environments
                pattern_func = plan_grid_yz

            # Generate seed for this environment
            if randomize_patterns:
                import time
                # Time-based seed ensures different pattern on every reset
                pattern_seed = int(time.time() * 1000000) + env_id
            else:
                pattern_seed = self.cfg.seed + env_id if self.cfg.seed is not None else None

            # Generate grid positions for this environment
            yz_local_env = pattern_func(
                center_y_local=(self.cfg.center_y_world - self.cfg.rack_y),
                rows=self.cfg.rows,
                layers=self.cfg.layers,
                spacing_y=self.cfg.spacing_y,
                spacing_z=self.cfg.spacing_z,
                base_z=self.cfg.base_z,
                cap=per_env_target,
                row_y_jitter=self.cfg.row_y_jitter,  # Random function handles its own jitter
                layer_y_offset=self.cfg.layer_y_offset,
                seed=pattern_seed,
            )


            # Add logs for this environment
            # Always allocate 200 slots per env, but only populate per_env_target
            max_slots_per_env = 200

            if rebuild_all:
                # First reset: append to list (always 200 slots per env)
                for i in range(max_slots_per_env):
                    if i < per_env_target:
                        # Active log: use actual position
                        y_local, z = yz_local_env[i]
                        all_world.append([
                            rack_world_x,
                            env_o[1] + self.cfg.rack_y + y_local,
                            env_o[2] + z,
                        ])
                    else:
                        # Inactive log: spawn far away (despawned)
                        all_world.append([-1000.0, -1000.0, -1000.0])
            else:
                # Subsequent reset: update tensor slice for this environment
                start_idx = env_id * max_slots_per_env
                end_idx = start_idx + max_slots_per_env

                for i in range(max_slots_per_env):
                    if start_idx + i < self._log_origins_world.shape[0]:
                        if i < per_env_target:
                            # Active log: use actual position
                            y_local, z = yz_local_env[i]
                            self._log_origins_world[start_idx + i, 0] = rack_world_x
                            self._log_origins_world[start_idx + i, 1] = env_o[1] + self.cfg.rack_y + y_local
                            self._log_origins_world[start_idx + i, 2] = env_o[2] + z
                        else:
                            # Inactive log: move far away
                            self._log_origins_world[start_idx + i, 0] = -1000.0
                            self._log_origins_world[start_idx + i, 1] = -1000.0
                            self._log_origins_world[start_idx + i, 2] = -1000.0

        # Create tensor on first reset
        if rebuild_all:
            if len(all_world) > 0:
                self._log_origins_world = torch.tensor(all_world, device=self.device, dtype=torch.float32)
            else:
                self._log_origins_world = torch.empty((0, 3), device=self.device, dtype=torch.float32)

    # ---------------- control / stepping ----------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process policy actions before physics step.

        In hierarchical mode, stores target selection (x, y, z, yaw) for use by heuristic FSM.
        In non-hierarchical RL mode, updates EE goal incrementally.
        """
        # Keep the last action per env for target selection phase
        self._last_actions = actions  # shape [num_envs, 4]
        # Keep log buffers fresh + cadence probe
        if args_cli.debug_logs and self._logs_obj is not None:
            try:
                self._logs_obj.update(self.sim.get_physics_dt())
            except Exception:
                pass
            if (self.common_step_counter % self._dbg_every) == 0:
                self._probe_logs(tag=f"STEP@{int(self.common_step_counter)}")

        # In hierarchical mode, the step() override runs the FSM internally
        # So we skip calling _heuristic_step() here to avoid running it twice
        if self._in_hierarchical_loop:
            return

        # Both heuristic and rl modes run the heuristic FSM
        # (rl mode just changes who selects the target at HOVER_UP)
        # But don't start FSM until logs have settled
        if not self._logs_settled:
            return

        if self._log_origins_world.numel() == 0:
            self._rebuild_log_origins_world()
            if self._log_origins_world.numel() == 0:
                return
        self._heuristic_step()

    def _apply_action(self) -> None:
        """Compute and apply joint velocity commands using differential IK or SimpleIK controller."""
        # Don't apply any actions until logs have settled
        if not self._logs_settled:
            return

        jac_all = self.crane.root_physx_view.get_jacobians()
        jacobian = jac_all[:, self._ee_jacobi_idx, :, self._ctrl_joint_idx]

        ee_pose_w  = self.crane.data.body_pose_w[:, self._ee_body_id]
        root_pose_w = self.crane.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],  ee_pose_w[:, 3:7]
        )

        # Always set orientation (both modes use heuristic FSM)
        self._ee_goal[:, 3:7] = ee_quat_b

        joint_pos_ctrl = self.crane.data.joint_pos[:, self._ctrl_joint_idx]
        # Only command IK after heuristic has set meaningful targets
        if self._heuristic_ready.any():
            if self._use_simple_ik:
                # Use one-shot IKPy controller with PD velocity control                 # Convert EE targets back to basegrapple targets for IK solver
                bg_targets = self._ee_goal[:, 0:3].clone()
                bg_targets[:, 2] -= args_cli.ee_to_grapple_offset_z  # Remove offset to get basegrapple target
                
                # Compute velocity commands using PD control
                joint_velocities = self._simple_ik.solve_ik_and_compute_velocities(bg_targets, joint_pos_ctrl)

                # Apply arm velocity commands
                self.crane.set_joint_velocity_target(joint_velocities, joint_ids=self._ctrl_joint_idx)

                # Debug: Print velocity commands every few steps
                if hasattr(self, '_debug_step_counter'):
                    self._debug_step_counter += 1
                else:
                    self._debug_step_counter = 0
                    
                if self._debug_step_counter % 50 == 0:  # Every 50 steps (~1 second)
                    print(f"DEBUG: bg_target[0]={bg_targets[0].cpu().numpy()}")
                    print(f"DEBUG: current_joints[0]={joint_pos_ctrl[0].cpu().numpy()}")
                    print(f"DEBUG: q_des_arm[0]={self._simple_ik.q_des_arm[0].cpu().numpy()}")
                    print(f"DEBUG: joint_velocities[0]={joint_velocities[0].cpu().numpy()}")
                
                # Must call write_data_to_sim() to actually apply velocity commands
                self.crane.write_data_to_sim()
            else:
                # For differential IK, pass position command and current EE orientation separately
                self._diffik.set_command(self._ee_goal[:, 0:3], ee_quat=ee_quat_b)
                self._joint_pos_des = self._diffik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_ctrl)
                # Apply position targets (original approach)
                self.crane.set_joint_position_target(self._joint_pos_des, joint_ids=self._ctrl_joint_idx)
                # Position targets are applied automatically in Isaac Lab
                
        # Apply gripper PD velocity control (matches joint_velocity_telescope_node.py)
        left_id, right_id = self._grip_joint_ids
        
        # Get current gripper joint positions
        current_gripper_joints = self.crane.data.joint_pos[:, [left_id, right_id]]  # [N, 2]
        
        # Calculate position errors ( err = q_des - q)
        gripper_error = self.q_des_grip - current_gripper_joints  # [N, 2]
        
        # Calculate error derivatives ( err_dot = (err - prev_error) / dt)
        gripper_error_dot = (gripper_error - self.prev_error_grip) / self._gripper_control_dt
        
        # Apply PD control ( vel = KP * err + KD * err_dot)
        gripper_vel = self.KP_GRIP * gripper_error + self.KD_GRIP * gripper_error_dot
        
        # Apply velocity limits ( vel = np.clip(vel, -MAX_VEL, MAX_VEL))
        gripper_vel = torch.clamp(gripper_vel, -self.MAX_VEL_GRIP, self.MAX_VEL_GRIP)
        
        # Apply velocity commands ( publishes velocity, not position)
        self.crane.set_joint_velocity_target(gripper_vel, joint_ids=[left_id, right_id])
        
        # Store error for next iteration ( prev_error = err.copy())
        self.prev_error_grip = gripper_error.clone()
        
        # Apply gripper velocity commands to simulation
        self.crane.write_data_to_sim()
        
        # Debug gripper control every few steps  
        if hasattr(self, '_debug_step_counter') and self._debug_step_counter % 50 == 0:
            print(f"DEBUG: gripper_q_des[0]={self.q_des_grip[0].cpu().numpy()}")
            print(f"DEBUG: gripper_current[0]={current_gripper_joints[0].cpu().numpy()}")
            print(f"DEBUG: gripper_vel[0]={gripper_vel[0].cpu().numpy()}")
            print(f"DEBUG: gripper_error[0]={gripper_error[0].cpu().numpy()}")
            
        # else: keep current joint positions (no IK commands until heuristic is ready)

        # Apply direct yaw control for HOVER_UP (neutral=0), ALIGN_YAW, and ALIGN_HOME_YAW phases
        # HOVER_UP drives yaw to 0 first (like original ROS code), then ALIGN_YAW applies optimal yaw
        # This prevents yaw accumulation across cycles that causes multi-spin behavior
        if True:
            align_mask = (self._phase == self.PH_HOVER_UP) | (self._phase == self.PH_ALIGN_YAW) | (self._phase == self.PH_ALIGN_HOME_YAW)
            if align_mask.any():
                yaw_joint_id = self._find_yaw_joint_id()
                if yaw_joint_id is not None:
                    # Get env indices that are in ALIGN_YAW or ALIGN_HOME_YAW phase
                    align_env_ids = torch.where(align_mask)[0]
                    if len(align_env_ids) > 0:
                        # Apply yaw targets for envs in either yaw alignment phase
                        yaw_targets_for_align_envs = self._yaw_targets[align_env_ids].unsqueeze(-1)
                        self.crane.set_joint_position_target(
                            yaw_targets_for_align_envs,
                            joint_ids=[yaw_joint_id],
                            env_ids=align_env_ids
                        )

        self._draw_debug_markers()

    # ---------------- obs / rew / dones ----------------
    def _get_observations(self):
        """Build observation at decision point (HOVER_UP).

        For hierarchical RL, this is called by step() after cycle completes.
        For non-hierarchical mode, falls back to original behavior.
        """
        # Hierarchical RL mode: build target selection observation
        if getattr(self.cfg, "use_hierarchical_rl", False):
            # Build fresh observation for policy decision
            return {"policy": self._build_target_selection_obs()}

        # Fallback: joint positions and velocities for non-hierarchical mode
        q = self.crane.data.joint_pos[:, self._ctrl_joint_idx]
        qd= self.crane.data.joint_vel[:, self._ctrl_joint_idx]
        obs = torch.cat([q, qd], dim=-1)
        return {"policy": obs}


    def _get_rewards(self) -> torch.Tensor:
        """Return reward buffer.

        For hierarchical RL, rewards are computed in step() and stored in _grasp_reward_buf.
        For non-hierarchical mode, falls back to original behavior.
        """
        # Hierarchical RL mode: rewards computed in step() override
        if getattr(self.cfg, "use_hierarchical_rl", False):
            # Rewards already computed in step() override
            return self._grasp_reward_buf

        # Fallback: default motion reward for non-hierarchical mode
        q  = self._q[:, self._ctrl_joint_idx]
        qd = self._qd[:, self._ctrl_joint_idx]
        return compute_rewards(self.cfg.rew_alive, self.cfg.rew_pos_l2, self.cfg.rew_vel_l1, q, qd)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination conditions.

        In hierarchical mode: Episodes end when rack is empty (200 logs deposited) or 30 cycles completed.
        In non-hierarchical mode: Uses default timeout based on episode length.

        Returns:
            terminated: Boolean tensor indicating which environments finished successfully/failed
            time_out: Boolean tensor indicating which environments timed out
        """
        # When using heuristic mode OR target selection policy, use custom termination
        # Both modes run heuristic FSM
        if True:
            terminated = torch.zeros_like(self.episode_length_buf, dtype=torch.bool, device=self.device)

            # Terminate episode if:
            # 1. All logs deposited (200 logs)
            # 2. 30 cycles completed
            for i in range(self.num_envs):
                logs_deposited = len(self._deposited_logs[i])
                cycles = self._cycle_count[i].item()

                if logs_deposited >= 200 or cycles >= 30:
                    terminated[i] = True

            time_out = torch.zeros_like(terminated)
            return terminated, time_out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _get_extras(self) -> dict:
        """Return extra metrics for logging (used by RSL-RL).

        Metrics are logged to TensorBoard under "Episode/" prefix (or as-is if they contain "/").
        All metrics are averaged across environments.

        Key metrics for paper:
        - grasp_success_rate: % of grasp attempts that picked up >= 1 log
        - throughput: average logs per successful grasp
        - alignment: average alignment score (weighted by logs grasped)
        - episode_return: cumulative reward in episode
        - pile_clearing_pct: % of starting pile cleared
        """
        extras = {}

        # Hierarchical RL mode: compute episode statistics
        if getattr(self.cfg, "use_hierarchical_rl", False):
            # Basic counters
            cycles_per_env = self._cycle_count.float()
            successful = self._episode_successful_grasps.float()
            failed = self._episode_failed_grasps.float()
            total_grasps = successful + failed

            # Grasp success rate: % of grasps that picked up >= 1 log
            grasp_success_rate = torch.where(
                total_grasps > 0,
                successful / total_grasps * 100.0,
                torch.zeros_like(successful)
            ).mean().item()

            # Throughput: logs per successful grasp
            total_logs = self._episode_total_logs_grasped.float()
            throughput = torch.where(
                successful > 0,
                total_logs / successful,
                torch.zeros_like(total_logs)
            ).mean().item()

            # Alignment: weighted average (sum of logs*alignment / total logs)
            alignment = torch.where(
                total_logs > 0,
                self._episode_alignment_sum_weighted / total_logs,
                torch.zeros_like(total_logs)
            ).mean().item()

            # Stability: weighted average (sum of logs*stability / total logs)
            stability = torch.where(
                total_logs > 0,
                self._episode_stability_sum_weighted / total_logs,
                torch.ones_like(total_logs)  # Default to 1.0 (stable) if no logs
            ).mean().item()

            # Episode return (cumulative reward)
            episode_return = self._episode_return.mean().item()

            # Pile clearing percentage
            if hasattr(self, '_per_env_log_counts') and self._per_env_log_counts is not None:
                starting_logs = self._per_env_log_counts.float().clamp(min=1)
            else:
                starting_logs = torch.full((self.num_envs,), 200.0, device=self.device)
            pile_clearing_pct = (total_logs / starting_logs * 100.0).mean().item()

            extras["episode"] = {
                # Primary metrics for paper (will appear as Episode/X in TensorBoard)
                "grasp_success_rate": grasp_success_rate,
                "throughput": throughput,
                "alignment": alignment,
                "stability": stability,
                "episode_return": episode_return,
                "pile_clearing_pct": pile_clearing_pct,

                # Secondary metrics
                "logs_grasped": total_logs.mean().item(),
                "successful_grasps": successful.mean().item(),
                "failed_grasps": failed.mean().item(),
                "cycles": cycles_per_env.mean().item(),

                # Reward components (will appear as reward/X in TensorBoard due to "/")
                "reward/per_cycle": (self._episode_return / cycles_per_env.clamp(min=1)).mean().item(),
            }

        return extras

    # ---------------- Camera methods ----------------
    def get_camera_data(self) -> dict:
        """Get RGB and depth images from the camera.

        Returns:
            dict with keys:
                - 'rgb': (num_envs, H, W, 3) uint8 tensor
                - 'depth': (num_envs, H, W, 1) float32 tensor (meters)
                - 'intrinsics': (3, 3) camera intrinsic matrix
            Returns empty dict if camera not enabled.
        """
        if self._camera is None:
            return {}

        data = {}

        # Get RGB if available
        if "rgb" in self.cfg.camera_cfg.data_types:
            rgb = self._camera.data.output["rgb"]
            data["rgb"] = rgb  # (num_envs, H, W, 4) RGBA

        # Get depth if available
        if "depth" in self.cfg.camera_cfg.data_types:
            depth = self._camera.data.output["depth"]
            data["depth"] = depth  # (num_envs, H, W, 1)

        # Get semantic segmentation if available
        if "semantic_segmentation" in self.cfg.camera_cfg.data_types:
            sem_seg = self._camera.data.output["semantic_segmentation"]
            data["semantic_segmentation"] = sem_seg

        # Get intrinsics matrix
        data["intrinsics"] = self._camera.data.intrinsic_matrices[0]  # (3, 3)

        return data

    def get_pointcloud(self, env_idx: int = 0, max_points: int = None) -> torch.Tensor:
        """Get point cloud from depth image for a specific environment.

        Args:
            env_idx: Environment index to get point cloud for
            max_points: Maximum number of points to return (random subsample if exceeded)

        Returns:
            (N, 3) tensor of 3D points in camera frame, or empty tensor if camera not enabled
        """
        if self._camera is None:
            return torch.empty((0, 3), device=self.device)

        if "depth" not in self.cfg.camera_cfg.data_types:
            print("[WARN] Depth not enabled in camera data_types")
            return torch.empty((0, 3), device=self.device)

        # Get depth image for this environment
        depth = self._camera.data.output["depth"][env_idx]  # (H, W, 1)
        depth = depth.squeeze(-1)  # (H, W)

        # Get intrinsic matrix
        intrinsics = self._camera.data.intrinsic_matrices[env_idx]  # (3, 3)

        # Get camera pose for world-frame transformation (optional)
        # cam_pos = self._camera.data.pos_w[env_idx]  # (3,)
        # cam_quat = self._camera.data.quat_w_world[env_idx]  # (4,) wxyz

        # Create point cloud in camera frame
        points = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsics,
            depth=depth,
            keep_invalid=False,  # Remove invalid points (inf, nan)
            device=self.device
        )

        # Subsample if too many points
        if max_points is not None and points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0], device=self.device)[:max_points]
            points = points[indices]

        return points

    def get_pointcloud_world(self, env_idx: int = 0, max_points: int = None,
                              depth_range: tuple = (0.5, 10.0)) -> torch.Tensor:
        """Get point cloud in world frame for a specific environment.

        Args:
            env_idx: Environment index
            max_points: Maximum points to return
            depth_range: (min_depth, max_depth) in meters to filter points

        Returns:
            (N, 3) tensor of 3D points in world frame
        """
        if self._camera is None:
            return torch.empty((0, 3), device=self.device)

        if "depth" not in self.cfg.camera_cfg.data_types:
            return torch.empty((0, 3), device=self.device)

        depth = self._camera.data.output["depth"][env_idx].squeeze(-1)
        intrinsics = self._camera.data.intrinsic_matrices[env_idx]

        # Apply depth range filter
        min_depth, max_depth = depth_range
        depth_mask = (depth >= min_depth) & (depth <= max_depth) & (~torch.isinf(depth))
        filtered_depth = torch.where(depth_mask, depth, torch.tensor(float('inf'), device=self.device))

        # Get camera pose in world frame
        cam_pos = self._camera.data.pos_w[env_idx]  # (3,)
        cam_quat = self._camera.data.quat_w_world[env_idx]  # (4,) wxyz

        # Create point cloud in world frame
        points = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsics,
            depth=filtered_depth,
            keep_invalid=False,
            position=cam_pos,
            orientation=cam_quat,
            device=self.device
        )

        if max_points is not None and points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0], device=self.device)[:max_points]
            points = points[indices]

        return points

    def get_log_pointcloud_world(self, env_idx: int = 0, max_points: int = None,
                                   depth_range: tuple = (0.5, 10.0)) -> torch.Tensor:
        """Get point cloud of ONLY the logs in world frame.

        Uses semantic segmentation to filter camera depth to only include log pixels.

        Args:
            env_idx: Environment index
            max_points: Maximum points to return
            depth_range: (min_depth, max_depth) in meters to filter points

        Returns:
            (N, 3) tensor of 3D points in world frame, filtered to only include log surfaces
        """
        if self._camera is None:
            return torch.empty((0, 3), device=self.device)

        if "depth" not in self.cfg.camera_cfg.data_types:
            return torch.empty((0, 3), device=self.device)

        if "semantic_segmentation" not in self.cfg.camera_cfg.data_types:
            print("[WARN] semantic_segmentation not enabled - cannot filter to logs only")
            return self.get_pointcloud_world(env_idx, max_points)

        # Get camera data
        depth = self._camera.data.output["depth"][env_idx].squeeze(-1)  # (H, W)
        semantic_seg = self._camera.data.output["semantic_segmentation"][env_idx]  # (H, W, C)

        # Debug: print semantic segmentation shape
        print(f"[DEBUG] Semantic seg shape: {semantic_seg.shape}, dtype: {semantic_seg.dtype}")

        # Semantic segmentation returns class IDs - find the "log" class
        # The semantic value depends on how the class was registered
        # Typically the first channel contains the class ID
        if semantic_seg.dim() == 3:
            semantic_ids = semantic_seg[..., 0]  # Take first channel
        else:
            semantic_ids = semantic_seg

        # Find unique semantic IDs
        unique_ids = torch.unique(semantic_ids)
        print(f"[DEBUG] Unique semantic IDs: {unique_ids.tolist()}")

        # Create mask for log pixels with depth range filter
        # The "log" class should have a specific ID - we need to find it
        # For now, let's assume non-zero IDs that aren't background are logs
        min_depth, max_depth = depth_range
        log_mask = (semantic_ids > 0) & (~torch.isinf(depth)) & (depth >= min_depth) & (depth <= max_depth)

        num_log_pixels = log_mask.sum().item()
        print(f"[DEBUG] Log pixels found: {num_log_pixels}")

        if num_log_pixels == 0:
            return torch.empty((0, 3), device=self.device)

        # Apply mask to depth
        masked_depth = torch.where(log_mask, depth, torch.tensor(float('inf'), device=self.device))

        # Get camera pose
        intrinsics = self._camera.data.intrinsic_matrices[env_idx]
        cam_pos = self._camera.data.pos_w[env_idx]
        cam_quat = self._camera.data.quat_w_world[env_idx]

        # Create filtered point cloud
        points = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsics,
            depth=masked_depth,
            keep_invalid=False,
            position=cam_pos,
            orientation=cam_quat,
            device=self.device
        )

        if max_points is not None and points.shape[0] > max_points:
            indices = torch.randperm(points.shape[0], device=self.device)[:max_points]
            points = points[indices]

        return points

    # ---------------- reset / utils ----------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specified environments to initial state.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.

        Handles log pile settling on first reset and reinitializes grasp tracking buffers.
        """
        if env_ids is None:
            env_ids = self.crane._ALL_INDICES
        super()._reset_idx(env_ids)

        # Domain randomization: rebuild log origins for resetting environments with random patterns
        if self.cfg.enable_domain_randomization:
            if self._log_origins_world.numel() == 0:
                # First reset: build all environments
                self._rebuild_log_origins_world(randomize_patterns=True, env_ids=None)
            else:
                # Subsequent resets: only rebuild the environments that are resetting
                self._rebuild_log_origins_world(randomize_patterns=True, env_ids=env_ids)
        elif self._log_origins_world.numel() == 0:
            # First reset only: build with fixed patterns
            self._rebuild_log_origins_world(randomize_patterns=False, env_ids=None)

        # Run settling on first reset to let logs fall and stabilize
        # This prevents target selection before logs have stopped moving
        needs_settling = not self._logs_settled

        if self._logs_obj is not None:
            logs = self._logs_obj
            max_logs_per_env = 200  # Fixed slot allocation per environment

            if self._log_origins_world.numel() > 0:
                # Start with CURRENT state (so non-resetting envs keep their positions)
                current_pos = logs.data.root_pos_w.clone()
                current_quat = logs.data.root_quat_w.clone()
                current_lin_vel = logs.data.root_lin_vel_w.clone()
                current_ang_vel = logs.data.root_ang_vel_w.clone()

                # Only reset logs for the specified environments
                for env_id in (env_ids if env_ids is not None else range(self.num_envs)):
                    if isinstance(env_id, torch.Tensor):
                        env_id = int(env_id.item())
                    start_idx = env_id * max_logs_per_env
                    end_idx = start_idx + max_logs_per_env

                    # Reset to INITIAL positions and INITIAL orientations
                    current_pos[start_idx:end_idx, :3] = self._log_origins_world[start_idx:end_idx]
                    current_pos[start_idx:end_idx, 2] += float(self.cfg.spawn_height)
                    current_quat[start_idx:end_idx] = self._log_quat.unsqueeze(0).expand(max_logs_per_env, -1)
                    current_lin_vel[start_idx:end_idx] = 0.0
                    current_ang_vel[start_idx:end_idx] = 0.0

                # Write updated state to simulation
                root_pose = torch.cat([current_pos, current_quat], dim=-1)
                root_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)
                logs.write_root_pose_to_sim(root_pose)
                logs.write_root_velocity_to_sim(root_vel)

        c_root = self.crane.data.default_root_state.clone()
        c_root[:, :3] += self.scene.env_origins
        # Only reset cranes for specified environments (must index c_root by env_ids!)
        if env_ids is not None:
            self.crane.write_root_pose_to_sim(c_root[env_ids, :7], env_ids=env_ids)
            self.crane.write_root_velocity_to_sim(c_root[env_ids, 7:], env_ids=env_ids)
        else:
            self.crane.write_root_pose_to_sim(c_root[:, :7], env_ids=None)
            self.crane.write_root_velocity_to_sim(c_root[:, 7:], env_ids=None)

        # Set proper initial joint positions
        jpos = self.crane.data.default_joint_pos.clone()
        jvel = self.crane.data.default_joint_vel.clone()
        
        # Override with stable crane pose
        # These correspond to: [basemast_to_mast, mast_to_mainboom, mainboom_to_stick, stick_to_telescope]
        arm_home_positions = torch.tensor([0.0, 0.5, -1.5, 0.5], device=self.device)
        
        # Find arm joint indices and set positions
        arm_joint_names = ["basemast_to_mast", "mast_to_mainboom", "mainboom_to_stick", "stick_to_telescope"]
        for i, name in enumerate(arm_joint_names):
            joint_ids, _ = self.crane.find_joints([name])
            if len(joint_ids) > 0:
                jpos[:, joint_ids[0]] = arm_home_positions[i]
        
        self.crane.write_joint_state_to_sim(jpos[env_ids], jvel[env_ids], None, env_ids)
        
        # Reset dual-stack system state for specified environments
        for i in range(self.num_envs):
            if env_ids is None or i in env_ids:
                self._current_stack[i] = "REAR"  # Always start with rear stack
                self._deposited_logs[i].clear()  # Clear deposited logs tracking
                self._failed_grasp_counts[i].clear()  # Clear failed grasp tracking
                self._drop_goal_calculated[i] = False  # Reset drop goal cache

                # Reset episode tracking
                self._cycle_count[i] = 0
                self._logs_at_cycle_start[i] = 0

                # Reset reward component tracking
                self._last_throughput_reward[i] = 0.0
                self._last_alignment_reward[i] = 0.0
                self._episode_throughput_sum[i] = 0.0
                self._episode_alignment_sum[i] = 0.0

                # Reset grasp outcome tracking for hierarchical RL
                self._grasp_reward_buf[i] = 0.0
                self._prev_logs_grasped[i] = 0.0
                self._prev_grasp_alignment[i] = 0.0
                self._logs_knocked_off[i] = 0  # Reset out-of-bounds counter

                # Reset episode-level metrics for TensorBoard
                self._episode_successful_grasps[i] = 0
                self._episode_failed_grasps[i] = 0
                self._episode_return[i] = 0.0
                self._episode_total_logs_grasped[i] = 0
                self._episode_alignment_sum_weighted[i] = 0.0
                self._episode_stability_sum_weighted[i] = 0.0
                self._prev_grasp_stability[i] = 1.0

        # Initialize EE goal to home position over trailer to avoid commanding basemast (0,0,0)
        for i in range(self.num_envs):
            if env_ids is None or i in env_ids:
                home_bg = self._compute_drop_goal(i)
                home_upperpassive = self._bg_to_ee_target_pos(home_bg)
                self._ee_goal[i, 0:3] = home_upperpassive
                # Set neutral hanging orientation
                self._ee_goal[i, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        if self._diffik is not None:
            self._diffik.reset()
        # Simple IK doesn't need reset - it's stateless

        # Reset gripper state for specified environments
        for i in range(self.num_envs):
            if env_ids is None or i in env_ids:
                self._set_gripper(i, open_fraction=1.0)
        self._phase[:] = self.PH_HOVER_UP
        self._has_log[:] = False
        self._timer[:] = 0
        self._dwell[:] = 0
        self._lift_hold_timer[:] = 0
        self._phase_timer[:] = 0
        self._yaw_targets[:] = 0.0
        self._target_log_pos_b.zero_()
        self._target_log_quat_w.zero_()
        self._dbg_target_bg.zero_()
        self._target_frozen.zero_()
        self._frozen_target_bg.zero_()
        self._frozen_target_upperpassive.zero_()
        self._heuristic_ready.zero_()

        # --- Reset target-selection buffers (for just these envs) ---
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            # env_ids may be a list/tuple/tensor — normalize to LongTensor on device
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)

        self._sel_mask[ids] = False
        self._sel_xy[ids].zero_()
        self._sel_z[ids].zero_()
        self._top_xy[ids].zero_()
        self._top_z[ids].zero_()

        self._sel_is_valid[ids] = False
        
        # Reset log ID tracking
        self._current_target_log_id[:] = -1

        # Reset gripper state for specified environments
        for i in range(self.num_envs):
            if env_ids is None or i in env_ids:
                # Initialize gripper to open position
                # Reset gripper PD control state
                self.q_des_grip[i] = self._grip_open  # Set initial target to open
                self.prev_error_grip[i] = 0.0  # Clear error history
                
                # Reset gripper stepping state
                self._gripper_step_target[i] = self.GRIP_OPEN
                self._gripper_step_timer[i] = 0
                self._gripper_stepping_active[i] = False

        if self._logs_obj is not None:
        # bind/update views once so reads in the very next tick are valid
            try:
                self.scene.update(self.sim.get_physics_dt())
            except Exception:
                pass
            self._post_first_step_bootstrap()
        
        # bind/update views once so reads in the very next tick are valid
        try:
            self.scene.update(self.sim.get_physics_dt())
        except Exception:
            pass
        self._post_first_step_bootstrap()

        # DEBUG: show a one-time snapshot at reset
        self._probe_logs(tag="RESET", force=True)

        # Run settling on first reset to let logs fall and stabilize
        if needs_settling:
            self._settle_logs()


    def _set_gripper(self, env_id: int, open_fraction: float):
        """Set gripper target positions for a specific environment."""
        s = float(max(0.0, min(1.0, open_fraction)))
        
        # Convert open_fraction to actual joint position like ROS
        # Gripper open/close range: GRIP_OPEN = 0.05, GRIP_MAX = 2.23
        target_position = (1.0 - s) * self._grip_closed + s * self._grip_open
        
        # Set both grippers to same target ( q_des[i_L] = q_des[i_R] = target)
        self.q_des_grip[env_id, 0] = target_position  # Left gripper
        self.q_des_grip[env_id, 1] = target_position  # Right gripper
        
        # Disable stepping when setting target directly
        self._gripper_stepping_active[env_id] = False
        
    def _start_gripper_stepping(self, env_id: int):
        """Start discrete stepping gripper closure (like sequential_grasp.py)."""
        # Initialize stepping state ( goal = GRIP_OPEN)
        self._gripper_step_target[env_id] = self.GRIP_OPEN
        self._gripper_step_timer[env_id] = 0
        self._gripper_stepping_active[env_id] = True
        
        # Set initial target to open position
        self.q_des_grip[env_id, 0] = self.GRIP_OPEN  
        self.q_des_grip[env_id, 1] = self.GRIP_OPEN
        
    def _update_gripper_stepping(self, env_id: int):
        """Update discrete stepping for one environment."""
        if not self._gripper_stepping_active[env_id]:
            return
            
        # Increment step timer
        self._gripper_step_timer[env_id] += 1
        
        # Check if it's time for next step ( time.sleep(CLOSE_DELAY))
        if self._gripper_step_timer[env_id] >= self.CLOSE_DELAY_STEPS:
            # Take next step ( goal = min(goal + CLOSE_STEP, GRIP_MAX))
            current_target = self._gripper_step_target[env_id].item()
            next_target = min(current_target + self.CLOSE_STEP, self.GRIP_MAX)
            
            self._gripper_step_target[env_id] = next_target
            self._gripper_step_timer[env_id] = 0
            
            # Update PD control targets ( send_goal with new target)
            self.q_des_grip[env_id, 0] = next_target
            self.q_des_grip[env_id, 1] = next_target
            
            # Stop stepping when we reach maximum ( while goal < GRIP_MAX)
            if next_target >= self.GRIP_MAX:
                self._gripper_stepping_active[env_id] = False
    


    
    def _enable_gripper_shrink_wrap(self):
        """Find gripper collision meshes on the Crane prototype and enable high-quality convex decomposition."""
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

            stage = omni.usd.get_context().get_stage()
            num_envs = getattr(self.scene, "num_envs", getattr(self, "num_envs", 1))

            visited_prototypes = set()

            for env_id in range(num_envs):
                crane_path = f"/World/envs/env_{env_id}/Crane"
                crane_prim = stage.GetPrimAtPath(crane_path)
                if not crane_prim or not crane_prim.IsValid():
                    print(f"[GRIPPER] No Crane prim at {crane_path}")
                    continue

                # If this Crane is an instance, we need to modify its prototype instead
                if crane_prim.IsInstance():
                    proto = crane_prim.GetPrototype()
                    if not proto:
                        print(f"[GRIPPER] Crane at {crane_path} is instance but has no prototype?")
                        continue

                    proto_path = proto.GetPath().pathString
                    if proto_path in visited_prototypes:
                        print(f"[GRIPPER] Prototype {proto_path} already processed, skipping for env_{env_id}")
                        continue

                    visited_prototypes.add(proto_path)
                    search_root = proto
                    print(f"[GRIPPER] Scanning Crane prototype {proto_path} for gripper collision meshes…")
                else:
                    search_root = crane_prim
                    print(f"[GRIPPER] Scanning {crane_path} for gripper collision meshes…")

                meshes_found = 0

                for prim in Usd.PrimRange(search_root):
                    if not prim.IsA(UsdGeom.Mesh):
                        continue

                    path_str = prim.GetPath().pathString

                    # Narrow it down to collision meshes on the grippers
                    if "/collisions/" not in path_str:
                        continue
                    # Relax this if needed, but start with gripper-only:
                    if "gripper" not in path_str.lower():
                        continue

                    meshes_found += 1

                    # Treat it as a collider
                    coll_api = UsdPhysics.CollisionAPI.Apply(prim)
                    coll_api.CreateCollisionEnabledAttr(True)
                    coll_api.CreateApproximationAttr("convexDecomposition")

                    # Layer 2: Tighten contact offsets for earlier collision detection
                    physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    physx_coll.GetContactOffsetAttr().Set(0.005)  # Reduced from 0.01 to 0.005 (5mm)
                    physx_coll.GetRestOffsetAttr().Set(0.0)

                    # Note: CCD not supported with GPU physics, set in USD if using CPU

                    # High-quality convex decomposition
                    decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    decomp.GetMaxConvexHullsAttr().Set(128)        # more hulls to capture concavity
                    decomp.GetHullVertexLimitAttr().Set(64)
                    decomp.GetVoxelResolutionAttr().Set(1_000_000) # fine voxelization
                    decomp.GetErrorPercentageAttr().Set(1.0)       # tighter fit
                    decomp.GetMinThicknessAttr().Set(0.001)
                    decomp.GetShrinkWrapAttr().Set(True)

                    print(f"  [GRIPPER] convex decomp + shrink on {path_str}")

                if meshes_found == 0:
                    print(
                        f"[GRIPPER] WARNING: no gripper collision meshes found under "
                        f"{search_root.GetPath().pathString}"
                    )

        except Exception as e:
            print(f"Warning: Could not enable gripper shrink wrap: {e}")


    def _post_first_step_bootstrap(self):
        """Run once after the first sim step so PhysX views exist."""
        print("DEBUG: _post_first_step_bootstrap() called")
        if getattr(self, "_log_index_ready", False):
            print("DEBUG: Already bootstrapped, returning")
            return
        if self._logs_obj is None:
            print("DEBUG: _logs_obj is None, returning")
            return
        d = self._logs_obj.data
        # Guard: only proceed when views are populated
        if not (hasattr(d, "root_pos_w") and hasattr(d, "root_quat_w")):
            print("DEBUG: Views not populated yet, returning")
            return
        total = int(d.root_pos_w.shape[0])
        self._logs_total = total
        self._logs_per_env = max(1, total // self.scene.num_envs)
        self._log_index_ready = True
        print("DEBUG: Bootstrap complete")

    def _settle_logs(self):
        """
        Run settling phase to let logs fall and stabilize before training starts.
        This prevents target selection while logs are still in mid-air.
        """
        settle_time = getattr(self.cfg, "settle_time", 2.0)
        if settle_time <= 0.0:
            self._logs_settled = True
            return

        print(f"[INFO] Settling logs for {settle_time:.2f}s before training starts...")

        # Open grippers during settling
        for i in range(self.num_envs):
            self._set_gripper(i, 1.0)

        # Calculate number of settling steps
        dt = self.cfg.sim.dt * self.cfg.decimation
        settle_steps = int(settle_time / dt)

        # Freeze crane during settling by setting zero joint velocities
        zero_joint_vel = torch.zeros_like(self.crane.data.joint_vel)

        # Run settling loop with zero actions
        for step in range(settle_steps):
            # Set crane joint velocities to zero (freeze crane in place)
            self.crane.write_joint_velocity_to_sim(zero_joint_vel)

            # Step simulation without policy intervention
            self.sim.step()
            self.scene.update(self.cfg.sim.dt)

            # Update logs buffers periodically
            if self._logs_obj is not None and step % 10 == 0:
                try:
                    self._logs_obj.update(self.cfg.sim.dt)
                except Exception:
                    pass

        self._logs_settled = True

        # Update scene one more time after settling
        self.scene.update(self.cfg.sim.dt)
        if self._logs_obj is not None:
            try:
                self._logs_obj.update(self.cfg.sim.dt)
            except Exception:
                pass

        # Clean up logs that fell out of bounds during settling (no penalty)
        print("[INFO] Cleaning up logs that fell out of bounds during settling...")
        total_cleaned = 0
        for i in range(self.num_envs):
            cleaned = self._check_logs_out_of_bounds(i, apply_penalty=False)
            if cleaned > 0:
                total_cleaned += cleaned
        if total_cleaned > 0:
            print(f"[INFO] Removed {total_cleaned} logs that fell out during initial settling")

        self._initial_settling_complete = True

        print(f"[INFO] Logs settled after {settle_steps} steps")

        # Compute action space bounds from settled log positions
        # Compute action bounds once after settling when needed.
        if getattr(self.cfg, "use_hierarchical_rl", False) or getattr(args_cli, "show_action_bounds", False):
            self._compute_action_space_bounds()
    def _probe_logs(self, tag: str = "STEP", max_envs: int | None = None, force: bool = False):
            """Simple, readable snapshot of live log poses (world frame)."""
            if (not force) and (not args_cli.debug_logs):
                return
            if self._logs_obj is None:
                print(f"[LOGS {tag}] logs_obj=None")
                return
            # Keep the RigidObject buffers fresh so we read 'live' poses.
            try:
                self._logs_obj.update(self.sim.get_physics_dt())
            except Exception:
                pass

            d = self._logs_obj.data
            pos_w = getattr(d, "root_pos_w", None)
            if pos_w is None:
                # happens only before first step; fall back to defaults
                rs = d.default_root_state
                pos_w = rs[:, :3]

            total = int(pos_w.shape[0]) if pos_w is not None else 0
            if total == 0:
                print(f"[LOGS {tag}] total=0 (no instances yet)")
                return

            # crude per-env slice (works with your uniform cloning)
            per_env = max(1, total // max(1, self.num_envs))
            shown_envs = min(self.num_envs, max_envs or self._dbg_envs_max)
            print(f"[LOGS {tag}] total={total} ~per_env={per_env} num_envs={self.num_envs}")

            for env_i in range(shown_envs):
                s = env_i * per_env
                e = min(s + per_env, total)
                if e <= s:
                    print(f"  env{env_i}: slice empty"); 
                    continue
                z = pos_w[s:e, 2]
                zmin = float(z.min().item()); zmax = float(z.max().item())
                k_local = int(torch.argmax(z).item())
                k_abs = s + k_local
                p = pos_w[k_abs]
                print(f"  env{env_i}: [{s}:{e}) z=[{zmin:+.3f},{zmax:+.3f}] "
                    f"top_id={k_abs} p=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")

    def get_log_poses_world(self, env_id: int | None = None):
        """World poses of the logs from the live PhysX views."""
        d = self._logs_obj.data
        pos_w = getattr(d, "root_pos_w", None)
        quat_w = getattr(d, "root_quat_w", None)
        if pos_w is None or quat_w is None:
            # fallback to defaults if called too early
            rs = d.default_root_state
            pos_w = rs[:, :3]
            quat_w = rs[:, 3:7]

        if env_id is None or not getattr(self, "_log_index_ready", False):
            return pos_w, quat_w

        n = self._logs_per_env
        s = env_id * n
        e = s + n
        return pos_w[s:e], quat_w[s:e]

    # ---------------- frame utilities ----------------
    def _body_pose_b(self, env_i: int, body_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        body_pose_w = self.crane.data.body_pose_w[:, body_id]
        root_pose_w = self.crane.data.root_pose_w
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            body_pose_w[:, 0:3],  body_pose_w[:, 3:7]
        )
        return pos_b[env_i], quat_b[env_i]

    def _bg_to_ee_target_pos(self, bg_pos_b: torch.Tensor) -> torch.Tensor:
        t = bg_pos_b.clone()
        if args_cli.ee_body.lower() == "upperpassive":
            t[2] += float(args_cli.ee_to_grapple_offset_z)
        return t

    def _get_logs_root_pose_w(self):
        logs = self._logs_obj
        if logs is None:
            return None, None
        d = logs.data
        if hasattr(d, "root_state_w"):
            pos_w = d.root_state_w[:, 0:3]
            quat_w = d.root_state_w[:, 3:7]
        elif hasattr(d, "root_pos_w") and hasattr(d, "root_quat_w"):
            pos_w = d.root_pos_w
            quat_w = d.root_quat_w
        else:
            rs = d.default_root_state
            pos_w = rs[:, 0:3]
            quat_w = rs[:, 3:7]
        return pos_w, quat_w

    def _target_top_log_center_b(self, env_i: int) -> tuple[torch.Tensor, int]:
        """Find highest log in rack for heuristic baseline targeting.

        Args:
            env_i: Environment index

        Returns:
            log_pos_b: Target position in base frame [3]
            selected_log_id: Global log ID, or -1 if using fallback target
        """
        logs = self._logs_obj
        # Check if we should use live poses and if logs are accessible
        use_live = self._use_live_log_poses and logs is not None

        selected_log_id = -1  # -1 indicates fallback target (no specific log)

        if not use_live or logs.num_instances == 0 or not hasattr(self, '_per_env_log_counts'):
            # fallback: center of rack top
            env_o = self.scene.env_origins[env_i]
            pos_w = torch.tensor(
                [env_o[0] + self.cfg.rack_x,
                env_o[1] + self.cfg.rack_y,
                env_o[2] + (self.cfg.base_z + 0.30)],
                device=self.device, dtype=torch.float32
            )[None, :]
            quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
        else:
            # live physics poses
            pos_all_w, quat_all_w = self._get_logs_root_pose_w()
            if pos_all_w is None:
                env_o = self.scene.env_origins[env_i]
                pos_w = torch.tensor(
                    [env_o[0] + self.cfg.rack_x,
                    env_o[1] + self.cfg.rack_y,
                    env_o[2] + (self.cfg.base_z + 0.30)],
                    device=self.device, dtype=torch.float32
                )[None, :]
                quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
            else:
                per_env = int(self._per_env_log_counts[env_i])
                start = env_i * 200  # Each env has 200 slots
                end = start + per_env  # Only active logs
                if end <= start:
                    env_o = self.scene.env_origins[env_i]
                    pos_w = torch.tensor(
                        [env_o[0] + self.cfg.rack_x,
                        env_o[1] + self.cfg.rack_y,
                        env_o[2] + (self.cfg.base_z + 0.30)],
                        device=self.device, dtype=torch.float32
                    )[None, :]
                    quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
                else:
                    slice_pos = pos_all_w[start:end]
                    slice_quat = quat_all_w[start:end]
                    
                    # Filter out deposited logs (fail-count skipping disabled)
                    available_indices = []
                    available_log_ids = []
                    for log_idx in range(slice_pos.shape[0]):
                        global_log_idx = start + log_idx
                        if global_log_idx not in self._deposited_logs[env_i]:
                            available_indices.append(log_idx)
                            available_log_ids.append(global_log_idx)

                    
                    if len(available_indices) == 0:
                        # No available logs - use fallback
                        env_o = self.scene.env_origins[env_i]
                        pos_w = torch.tensor(
                            [env_o[0] + self.cfg.rack_x,
                            env_o[1] + self.cfg.rack_y,
                            env_o[2] + (self.cfg.base_z + 0.30)],
                            device=self.device, dtype=torch.float32
                        )[None, :]
                        quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
                    else:
                        # Pick highest available log
                        available_pos = slice_pos[available_indices]
                        available_quat = slice_quat[available_indices]
                        k = int(torch.argmax(available_pos[:, 2]).item())  # highest Z among available
                        pos_w = available_pos[k:k+1, :]
                        quat_w = available_quat[k:k+1, :]
                        selected_log_id = available_log_ids[k]  # Store the selected log ID

        # transform log pose from world -> crane base (BG frame logic uses base as reference)
        root_pose_w = self.crane.data.root_pose_w
        base_pos_w  = root_pose_w[env_i:env_i+1, 0:3]
        base_quat_w = root_pose_w[env_i:env_i+1, 3:7]
        log_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, pos_w, quat_w)

        # IMPORTANT: no artificial Z offset here (removed the +1.0m hack)
        # Return quat_w[0] for yaw alignment calculation
        return log_pos_b[0], selected_log_id, quat_w[0]

    # ---------------- heuristic expert ----------------
    def _heuristic_step(self):
        """Execute one step of the finite state machine for all environments.

        Implements a 10-phase pick-place cycle: HOVER_UP → ALIGN_YAW → DESCEND →
        CLOSE → LIFT_HIGH → CARRY_HOME → ALIGN_HOME_YAW → LOWER_TO_DROP → OPEN → SETTLE.

        The heuristic is used both as a baseline and as the executor in hierarchical RL mode.
        """
        # Force disable ALL debug prints during hierarchical internal loop to avoid terminal spam
        if self._in_hierarchical_loop:
            do_dbg = False
            max_envs = 0  # No debug envs
        else:
            do_dbg = (self.common_step_counter % self._dbg_every == 0)
            max_envs = min(self._dbg_envs_max, self.num_envs)


        for i in range(self.num_envs):
            phase = int(self._phase[i].item())
            name = self.PHASE_NAMES[phase]

            # Freeze target from policy action at start of pick cycle
            # Skip if this env already completed its grasp cycle this step() - wait for step() to return
            cycle_done = getattr(self, '_cycle_complete_this_step', None)
            if cycle_done is not None and cycle_done[i]:
                # Already evaluated this hierarchical step, don't start a new cycle
                continue  # Skip ALL processing for this env
            elif phase == self.PH_HOVER_UP and not self._target_frozen[i]:
                # Check if using policy (hierarchical RL) or heuristic mode
                use_policy = getattr(self.cfg, "use_hierarchical_rl", False)

                if use_policy:
                    # FULL ACTION SPACE: [x, y, z, yaw]
                    # Read last action (stored in _pre_physics_step)
                    a = self._last_actions[i] if hasattr(self, "_last_actions") else torch.zeros(self.cfg.action_space, device=self.device)

                    # Compute bounds if not already done
                    if not self._action_bounds_valid[i]:
                        self._compute_action_space_bounds()

                    # Use computed bounds
                    min_bounds = self._action_bounds_min[i]
                    max_bounds = self._action_bounds_max[i]

                    # Scale actions from [-1, 1] to bounded workspace
                    x_norm = torch.tanh(a[0])  # [-1, 1] -> X position
                    y_norm = torch.tanh(a[1])  # [-1, 1] -> Y position
                    z_norm = torch.tanh(a[2])  # [-1, 1] -> Z position
                    yaw_norm = torch.tanh(a[3])  # [-1, 1] -> Yaw angle

                    # Map positions to bounds: val = min + (norm + 1) / 2 * (max - min)
                    x_b = min_bounds[0] + (x_norm + 1.0) / 2.0 * (max_bounds[0] - min_bounds[0])
                    y_b = min_bounds[1] + (y_norm + 1.0) / 2.0 * (max_bounds[1] - min_bounds[1])
                    z_b = min_bounds[2] + (z_norm + 1.0) / 2.0 * (max_bounds[2] - min_bounds[2])
                    # Map yaw to [-pi, pi]
                    import math
                    yaw_target = yaw_norm * math.pi

                    # Cache target position and yaw for downstream phases
                    self._target_log_pos_b[i] = torch.stack([x_b, y_b, z_b])
                    # NOTE: Do NOT overwrite _target_log_quat_w here - it was set by Expert4D
                    # with the actual target log's quaternion. Overwriting with identity causes
                    # the grapple to always align with the rack instead of the specific log.
                    self._pending_yaw_targets[i] = yaw_target  # Store policy yaw for ALIGN_YAW phase
                    self._yaw_targets[i] = 0.0  # Drive yaw to neutral during HOVER_UP (like original ROS code)

                    # Find which log the policy selected (for reward computation)
                    if hasattr(self, '_logs_obj') and self._logs_obj is not None:
                        per_env = int(self._per_env_log_counts[i]) if hasattr(self, '_per_env_log_counts') else 16
                        start = i * 200  # Each env has 200 slots
                        end = start + per_env  # Only active logs

                        if end > start:
                            log_pos_w, _ = self._get_logs_root_pose_w()
                            if log_pos_w is not None:
                                # Get logs for this environment in world frame
                                all_env_log_pos_w = log_pos_w[start:end]

                                # Filter out deposited logs (fail-count skipping disabled)
                                available_local_indices = []
                                for log_idx in range(end - start):
                                    global_log_idx = start + log_idx
                                    if global_log_idx not in self._deposited_logs[i]:
                                        available_local_indices.append(log_idx)


                                if len(available_local_indices) == 0:
                                    # No available logs - skip selection and reset target
                                    self._sel_is_valid[i] = False
                                    self._target_frozen[i] = True
                                    self._current_target_log_id[i] = -1  # No valid target
                                    continue

                                available_indices_tensor = torch.tensor(available_local_indices, device=self.device, dtype=torch.long)
                                env_log_pos_w = all_env_log_pos_w[available_indices_tensor]

                                # Convert selected target to world frame
                                root_w = self.crane.data.root_pose_w[i]
                                target_w = root_w[:3] + self._target_log_pos_b[i]

                                # Find closest AVAILABLE log and highest AVAILABLE log
                                dists = torch.norm(env_log_pos_w - target_w.unsqueeze(0), dim=-1)
                                closest_available_idx = torch.argmin(dists).item()
                                highest_available_idx = torch.argmax(env_log_pos_w[:, 2]).item()  # Highest z among available

                                # Map back to local indices for reward computation
                                closest_local_idx = available_local_indices[closest_available_idx]
                                highest_local_idx = available_local_indices[highest_available_idx]

                                # Store for reward computation (using indices within available logs)
                                self._sel_xy[i] = self._target_log_pos_b[i][:2]
                                self._sel_z[i] = self._target_log_pos_b[i][2]
                                self._top_xy[i] = (env_log_pos_w[highest_available_idx] - root_w[:3])[:2]
                                self._top_z[i] = (env_log_pos_w[highest_available_idx] - root_w[:3])[2]
                                self._sel_is_valid[i] = True
                                self._sel_mask[i] = True  # Mark as selection frame

                                # NO SNAPPING - let policy learn XYZ control
                                # (target_log_pos_b already set from policy action at line 2878)
                                # Just track which log was closest for reward computation
                                self._current_target_log_id[i] = start + closest_local_idx

                else:
                    # Heuristic mode: pick the highest available log
                    log_pos_b, log_id, log_quat_w = self._target_top_log_center_b(i)
                    self._target_log_pos_b[i] = log_pos_b
                    self._current_target_log_id[i] = log_id
                    # Store the target log's quaternion for yaw alignment
                    self._target_log_quat_w[i] = log_quat_w

                # Calculate and freeze hover targets (shared by both modes)
                log_b = self._target_log_pos_b[i]
                hover_z = log_b[2] + self.HOVER_CLEAR
                self._frozen_target_bg[i] = torch.tensor(
                    [log_b[0].item(), log_b[1].item(), hover_z],
                    device=self.device, dtype=self._ee_goal.dtype
                )
                self._frozen_target_upperpassive[i] = torch.tensor(
                    [log_b[0].item(), log_b[1].item(), hover_z + float(args_cli.ee_to_grapple_offset_z)],
                    device=self.device, dtype=self._ee_goal.dtype
                )
                self._target_frozen[i] = True

            log_b = self._target_log_pos_b[i]

            cur_bg, _ = self._body_pose_b(i, self._basegrapple_body_id)
            cur_up, cur_up_quat = self._body_pose_b(i, self._upperpassive_body_id)

            if phase == self.PH_HOVER_UP:
                # Step 1: HOVER - send_goal(srv, hover_xyz, 0.0, GRIP_OPEN, "HOVER")
                # Use full target for fast hover movement
                # Use frozen targets (hover_xyz)
                target_bg = self._frozen_target_bg[i]
                target_upperpassive = self._frozen_target_upperpassive[i]

                # Keep gripper OPEN during hover
                self.q_des_grip[i, :] = self.GRIP_OPEN

                # Set full target immediately
                self._ee_goal[i, 0:3] = target_upperpassive
                self._dbg_target_bg[i] = target_bg
                # Mark heuristic as ready for this environment
                self._heuristic_ready[i] = True
                # Drive yaw to neutral (0) during HOVER - prevents yaw accumulation across cycles
                # This matches original ROS code: send_goal(srv, hover_xyz, 0.0, GRIP_OPEN, "HOVER")
                self._yaw_targets[i] = 0.0

                # Two-stage validation wait_position_only()
                # Stage 1: upperpassive precision
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()

                # Stage 2: basegrapple settling
                basegrapple_error = torch.norm(cur_bg - target_bg).item()

                if upperpassive_error < self.UP_TOL:
                    if basegrapple_error < self.BG_TOL:
                        self._dwell[i] = self._dwell[i] + 1
                    else:
                        self._dwell[i] = 0
                else:
                    self._dwell[i] = 0

                self._phase_timer[i] += 1  # Increment phase timeout timer

                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_ALIGN_YAW, f"{name}: position reached (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")
                elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking
                    self._transition(i, self.PH_ALIGN_YAW, f"{name}: timeout after {self.PHASE_TIMEOUT} steps (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress}")


            elif phase == self.PH_ALIGN_YAW:
                # Keep same BG position as HOVER (position-only hold)
                target_bg = self._frozen_target_bg[i]
                target_upperpassive = self._frozen_target_upperpassive[i]

                # Keep gripper OPEN during yaw alignment
                self.q_des_grip[i, :] = self.GRIP_OPEN

                # Hold position, don't let IK change yaw
                self._ee_goal[i, 0:3] = target_upperpassive
                self._ee_goal[i, 3:7] = cur_up_quat
                self._dbg_target_bg[i] = target_bg

                # Yaw target handling
                # In heuristic mode: recalculate optimal yaw for target log alignment
                # In policy mode (hierarchical RL): use the yaw from _pending_yaw_targets (set during HOVER_UP)
                if not getattr(self.cfg, "use_hierarchical_rl", False):
                    # Heuristic mode: recalculate yaw for optimal log alignment
                    # Now that arm is positioned above target (yaw near 0), calculate optimal alignment
                    self._yaw_targets[i] = self._calc_optimal_yaw_feedback(i)
                else:
                    # Policy mode: apply the policy's yaw prediction (stored during HOVER_UP)
                    self._yaw_targets[i] = self._pending_yaw_targets[i]

                # Two-stage validation (unchanged)
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()
                basegrapple_error = torch.norm(cur_bg - target_bg).item()

                if upperpassive_error < self.UP_TOL:
                    if basegrapple_error < self.BG_TOL:
                        self._dwell[i] = self._dwell[i] + 1
                    else:
                        self._dwell[i] = 0
                else:
                    self._dwell[i] = 0

                self._phase_timer[i] += 1  # Increment phase timeout timer

                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_DESCEND, f"{name}: yaw aligned (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")
                elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking
                    self._transition(i, self.PH_DESCEND, f"{name}: timeout after {self.PHASE_TIMEOUT} steps (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    # Show current yaw vs target
                    yaw_joint_id = self._find_yaw_joint_id()
                    if yaw_joint_id is not None:
                        current_yaw = self.crane.data.joint_pos[i, yaw_joint_id].item()
                        target_yaw = self._yaw_targets[i].item()
                        yaw_error = abs(current_yaw - target_yaw)
                        print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | yaw={current_yaw:+.3f} target={target_yaw:+.3f} err={yaw_error:.3f} | timeout={timeout_progress}")
                    else:
                        print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress}")


            elif phase == self.PH_DESCEND:
                # Step 3: DESCEND - controlled descent

                # Count available logs before descent (for normalized reward)
                # Only count once per cycle (when _logs_available_at_target is 0)
                if self.cfg.normalize_reward and self._logs_available_at_target[i] == 0:
                    self._logs_available_at_target[i] = self._count_logs_in_column(i)

                # Use command scaling for controlled descent (30% step size)
                approach_z = log_b[2] + self.APPROACH_ABOVE
                target_upperpassive = torch.tensor([log_b[0].item(), log_b[1].item(), approach_z + float(args_cli.ee_to_grapple_offset_z)],
                                                   device=self.device, dtype=self._ee_goal.dtype)
                target_bg = torch.tensor([log_b[0].item(), log_b[1].item(), approach_z],
                                         device=self.device, dtype=self._ee_goal.dtype)

                # Keep gripper OPEN during descent
                self.q_des_grip[i, :] = self.GRIP_OPEN

                # Apply command scaling for controlled descent
                current_upperpassive = cur_up
                scaled_target = self._set_movement_speed(phase, i, current_upperpassive, target_upperpassive)
                self._ee_goal[i, 0:3] = scaled_target
                self._dbg_target_bg[i] = target_bg

                # Two-stage validation
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()
                basegrapple_error = torch.norm(cur_bg - target_bg).item()

                if upperpassive_error < self.UP_TOL:
                    if basegrapple_error < self.BG_TOL:
                        self._dwell[i] = self._dwell[i] + 1
                    else:
                        self._dwell[i] = 0
                else:
                    self._dwell[i] = 0
                
                self._phase_timer[i] += 1  # Increment phase timeout timer

                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_CLOSE, f"{name}: position reached (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")
                elif self._phase_timer[i] >= self.DESCEND_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking (longer timeout for descent)
                    self._transition(i, self.PH_CLOSE, f"{name}: timeout after {self.DESCEND_TIMEOUT} steps (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.DESCEND_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress}")

            elif phase == self.PH_CLOSE:
                # Start discrete stepping on first entry to CLOSE phase
                if self._gripper_timeout_timer[i] == 0:
                    self._start_gripper_stepping(i)
                
                # Update discrete stepping ( goal = min(goal + CLOSE_STEP, GRIP_MAX))
                self._update_gripper_stepping(i)
                self._gripper_timeout_timer[i] += 1
                
                # Get current gripper joint positions
                left_id, right_id = self._grip_joint_ids
                cur_left = float(self.crane.data.joint_pos[i, left_id].item())
                cur_right = float(self.crane.data.joint_pos[i, right_id].item())
                # Use current stepping target for validation (not final target)
                target_value = float(self._gripper_step_target[i].item())
                
                # Calculate errors for both gripper joints
                left_error = abs(cur_left - target_value)
                right_error = abs(cur_right - target_value)
                max_error = max(left_error, right_error)
                
                # Only proceed when stepping is complete AND gripper is stable
                stepping_complete = not self._gripper_stepping_active[i] or target_value >= self.GRIP_MAX
                
                # Check if both grippers are within tolerance
                if max_error < self.GRIPPER_TOL and stepping_complete:
                    # Within tolerance and stepping complete - increment stability timer
                    self._gripper_stability_timer[i] += 1
                    
                    # Check if stable for required time
                    if self._gripper_stability_timer[i] >= self.GRIPPER_STABILITY_TIME:
                        # Gripper properly closed - proceed to lift
                        self._has_log[i] = True
                        self._gripper_timeout_timer[i] = 0
                        self._gripper_stability_timer[i] = 0
                        self._transition(i, self.PH_LIFT_HIGH, f"{name}: gripper closed and stable (target={target_value:.2f}, max_err={max_error:.3f})")
                else:
                    # Lost tolerance or stepping not complete - reset stability timer
                    self._gripper_stability_timer[i] = 0

                # Timeout check
                if self._gripper_timeout_timer[i] >= self.GRIPPER_TIMEOUT:
                    self._has_log[i] = True
                    self._gripper_timeout_timer[i] = 0
                    self._gripper_stability_timer[i] = 0
                    self._transition(i, self.PH_LIFT_HIGH, f"{name}: gripper timeout ({self.GRIPPER_TIMEOUT} steps), proceeding anyway")
                
                # Debug gripper state with stepping info
                if do_dbg and i < max_envs and self._gripper_timeout_timer[i] % 25 == 0:  # Every 0.5s at 50Hz
                    stability_progress = f"{self._gripper_stability_timer[i]}/{self.GRIPPER_STABILITY_TIME}"
                    timeout_progress = f"{self._gripper_timeout_timer[i]}/{self.GRIPPER_TIMEOUT}"
                    stepping_status = "ACTIVE" if self._gripper_stepping_active[i] else "COMPLETE"
                    print(f"[env{i}] PHASE={name} | gripper=[L:{cur_left:.3f},R:{cur_right:.3f}] | step_target={target_value:.3f} | "
                          f"max_err={max_error:.3f} (tol:{self.GRIPPER_TOL:.3f}) | stepping={stepping_status} | stable={stability_progress} | timeout={timeout_progress}")

            elif phase == self.PH_LIFT_HIGH:
                # Step 5: Lift (controlled lift with command scaling for stability)
                # Use 50% step size - moderate speed for stability with log

                # Lift back to original hover height
                # Use frozen hover target that was calculated during HOVER_UP phase
                target_bg = self._frozen_target_bg[i]  # This has hover height: log_z + HOVER_CLEAR (2.5m)
                target_upperpassive = self._frozen_target_upperpassive[i]

                # Apply command scaling for controlled lift
                current_upperpassive = cur_up
                scaled_target = self._set_movement_speed(phase, i, current_upperpassive, target_upperpassive)
                self._ee_goal[i, 0:3] = scaled_target
                self._dbg_target_bg[i] = target_bg

                # Check position errors for lifting (consistent with other phases)
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()
                basegrapple_error = torch.norm(cur_bg - target_bg).item()  # Full 3D distance like other phases

                self._phase_timer[i] += 1  # Increment phase timeout timer

                # Check if we're in hold mode (keeping logs lifted for visual feedback)
                if self._lift_hold_timer[i] > 0:
                    # In hold mode - keep position, decrement timer
                    self._lift_hold_timer[i] -= 1
                    if self._lift_hold_timer[i] == 0:
                        # Hold complete - now evaluate and transition
                        logs_grasped, alignment, stability = self._check_grasped_logs(i)
                        reward = self._compute_grasp_reward(logs_grasped, alignment, stability, i)
                        self._grasp_reward_buf[i] = reward
                        self._prev_logs_grasped[i] = float(logs_grasped)
                        self._prev_grasp_alignment[i] = alignment
                        self._prev_grasp_stability[i] = stability
                        self._cycle_complete_this_step[i] = True

                        # Update episode-level metrics for TensorBoard
                        self._episode_return[i] += reward
                        if logs_grasped > 0:
                            self._episode_successful_grasps[i] += 1
                            self._episode_total_logs_grasped[i] += logs_grasped
                            self._episode_alignment_sum_weighted[i] += logs_grasped * alignment
                            self._episode_stability_sum_weighted[i] += logs_grasped * stability
                        else:
                            self._episode_failed_grasps[i] += 1

                        # Reset logs available counter for next cycle
                        if self.cfg.normalize_reward:
                            self._logs_available_at_target[i] = 0

                        # Track failed grasps to avoid targeting same log repeatedly
                        target_log_id = int(self._current_target_log_id[i].item())
                        # if logs_grasped == 0 and target_log_id >= 0:
                        #     fail_count = self._failed_grasp_counts[i].get(target_log_id, 0) + 1
                        #     self._failed_grasp_counts[i][target_log_id] = fail_count
                        #     if fail_count >= 2:
                        #         print(f"[env{i}] LOG {target_log_id}: failed {fail_count}x, will skip in future")

                        cycle_num = self._cycle_count[i].item() + 1
                        logs_remaining = self._count_logs_in_rack(i)
                        # Subtract current grasp since despawn happens after this print
                        remaining_after_grasp = max(0, logs_remaining - logs_grasped)
                        print(f"[env{i}] CYCLE {cycle_num}/30 | GRASP: {logs_grasped} logs, align={alignment:.2f}, stab={stability:.2f}, rew={reward:.2f} | remaining={remaining_after_grasp}")

                        self._despawn_grasped_logs(i)
                        self._set_gripper(i, open_fraction=1.0)
                        self._target_frozen[i] = False
                        self._target_log_pos_b[i].zero_()
                        self._target_log_quat_w[i].zero_()
                        self._transition(i, self.PH_HOVER_UP, f"{name}: hold complete, returning to hover")
                else:
                    # Not in hold mode - check if we should enter it
                    self._dwell[i] = self._dwell[i] + 1 if (upperpassive_error < self.UP_TOL) else torch.tensor(0, device=self.device)

                    if self._dwell[i] >= self.DWELL_N:
                        # Dwell complete - enter hold mode
                        self._dwell[i] = 0
                        self._lift_hold_timer[i] = self.LIFT_HOLD_DURATION

                    elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                        # Timeout - enter hold mode (brief hold even on timeout)
                        self._lift_hold_timer[i] = self.LIFT_HOLD_DURATION

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.UP_TOL:0.2f}) | timeout={timeout_progress}")

            elif phase == self.PH_CARRY_HOME:
                # Step 1 of drop sequence:Move to specified HOME position
                # Determine current home position and stack region
                total_deposited = len(self._deposited_logs[i])
                if self._current_stack[i] == "REAR" and total_deposited >= self._stack_switch_threshold:
                    self._current_stack[i] = "FRONT"
                    if not self._in_hierarchical_loop:
                        print(f"[env{i}] STACK-SWITCH: Switching to FRONT stack after {total_deposited} total deposited logs")
                
                # Use configured home coordinates
                if self._current_stack[i] == "REAR":
                    home_position = self._home_rear  # [2.1, 0.0, 2.0]
                else:
                    home_position = self._home_front  # [4.7, 0.0, 2.5]
                
                # Move to home position
                # Keep current yaw during movement, align yaw in next phase
                target_upperpassive = self._bg_to_ee_target_pos(home_position)
                self._ee_goal[i, 0:3] = target_upperpassive
                self._dbg_target_bg[i] = home_position
                
                bg_err = torch.norm(cur_bg - home_position).item()
                self._dwell[i] = self._dwell[i] + 1 if (bg_err < self.BG_TOL) else torch.tensor(0, device=self.device)
                
                self._phase_timer[i] += 1  # Increment phase timeout timer
                
                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_ALIGN_HOME_YAW, f"{name}: reached home position")
                elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking
                    self._transition(i, self.PH_ALIGN_HOME_YAW, f"{name}: timeout after {self.PHASE_TIMEOUT} steps (bg_err={bg_err:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | bg_err={bg_err:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress} | stack={self._current_stack[i]}")

            elif phase == self.PH_ALIGN_HOME_YAW:
                # Align yaw to 0° for trailer alignment while holding home position
                if self._current_stack[i] == "REAR":
                    home_position = self._home_rear  # [2.1, 0.0, 2.0]
                else:
                    home_position = self._home_front  # [4.7, 0.0, 2.5]
                
                # Hold position, don't let IK change yaw
                target_upperpassive = self._bg_to_ee_target_pos(home_position)
                self._ee_goal[i, 0:3] = target_upperpassive
                self._ee_goal[i, 3:7] = cur_up_quat  # Hold current orientation
                self._dbg_target_bg[i] = home_position
                
                # Calculate yaw target to align basegrapple with trailer Y-axis (base frame Y-axis)
                # Use same feedback approach as log alignment, but target is base frame (0° Z-rotation)
                target_trailer_z_rotation = 0.0  # Base frame Y-axis alignment
                optimal_yaw = self._calc_optimal_yaw_for_trailer_alignment(i, target_trailer_z_rotation)
                self._yaw_targets[i] = optimal_yaw
                
                # Check position stability while yaw aligns
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()
                basegrapple_error = torch.norm(cur_bg - home_position).item()
                
                if upperpassive_error < self.UP_TOL:
                    if basegrapple_error < self.BG_TOL:
                        self._dwell[i] = self._dwell[i] + 1
                    else:
                        self._dwell[i] = 0
                else:
                    self._dwell[i] = 0

                self._phase_timer[i] += 1  # Increment phase timeout timer

                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_LOWER_TO_DROP, f"{name}: yaw aligned at home")
                elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking
                    self._transition(i, self.PH_LOWER_TO_DROP, f"{name}: timeout after {self.PHASE_TIMEOUT} steps (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress}")

            elif phase == self.PH_LOWER_TO_DROP:
                # Step 2 of drop sequence:Lower to dynamic drop height based on specific stack region
                # Use command scaling for controlled descent (30% step size)
                
                # Calculate drop goal only once per LOWER_TO_DROP phase and cache it
                if not self._drop_goal_calculated[i]:
                    drop_b = self._compute_drop_goal(i, verbose=False)  # Suppress verbose prints
                    self._cached_drop_goals[i] = drop_b
                    self._drop_goal_calculated[i] = True
                else:
                    drop_b = self._cached_drop_goals[i]
                
                # Apply command scaling for controlled descent
                target_upperpassive = self._bg_to_ee_target_pos(drop_b)
                current_upperpassive = cur_up
                scaled_target = self._set_movement_speed(phase, i, current_upperpassive, target_upperpassive)
                self._ee_goal[i, 0:3] = scaled_target
                self._dbg_target_bg[i] = drop_b
                # Check both upperpassive and basegrapple errors
                upperpassive_error = torch.norm(cur_up - target_upperpassive).item()
                basegrapple_error = torch.norm(cur_bg - drop_b).item()
                
                self._dwell[i] = self._dwell[i] + 1 if (upperpassive_error < self.UP_TOL and basegrapple_error < self.BG_TOL) else torch.tensor(0, device=self.device)
                self._phase_timer[i] += 1  # Increment phase timeout timer
                
                if self._dwell[i] >= self.DWELL_N:
                    self._dwell[i] = 0
                    self._transition(i, self.PH_OPEN, f"{name}: reached drop height")
                elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                    # Force transition on timeout to prevent infinite sticking
                    self._transition(i, self.PH_OPEN, f"{name}: timeout after {self.PHASE_TIMEOUT} steps (up_err={upperpassive_error:.3f}, bg_err={basegrapple_error:.3f})")

                if do_dbg and i < max_envs:
                    timeout_progress = f"{self._phase_timer[i]}/{self.PHASE_TIMEOUT}"
                    print(f"[env{i}] PHASE={name} | up_err={upperpassive_error:0.3f} (tol {self.UP_TOL:0.2f}) | bg_err={basegrapple_error:0.3f} (tol {self.BG_TOL:0.2f}) | timeout={timeout_progress}")

            elif phase == self.PH_OPEN:
                self._set_gripper(i, open_fraction=1.0)
                self._transition(i, self.PH_SETTLE, f"{name}: opened")

            elif phase == self.PH_SETTLE:
                self._timer[i] += 1

                # Check for deposited logs after gripper has been open for a few frames (logs have time to settle)
                if self._timer[i] == 10:  # Check once after logs have settled
                    newly_deposited = self._check_deposited_logs(i)
                    if newly_deposited:
                        self._deposited_logs[i].update(newly_deposited)
                        if not self._in_hierarchical_loop:
                            print(f"[env{i}] LOG-UPDATE: {len(newly_deposited)} newly deposited logs (total: {len(self._deposited_logs[i])})")

                if self._timer[i] > 15:
                    self._timer[i] = 0

                    # Increment cycle count (only in standalone mode, hierarchical mode handles this in step())
                    if not self._in_hierarchical_loop:
                        self._cycle_count[i] += 1

                    # Clear frozen target for next pick cycle
                    self._target_log_pos_b[i].zero_()
                    self._target_log_quat_w[i].zero_()
                    self._target_frozen[i] = False

                    self._transition(i, self.PH_HOVER_UP, f"{name}: done, looping")

    def _transition(self, env_i: int, new_phase: int, reason: str):
        """Transition FSM to a new phase and reset phase-specific state.

        Args:
            env_i: Environment index
            new_phase: Phase constant (e.g., PH_HOVER_UP, PH_DESCEND)
            reason: Debug string explaining why transition occurred
        """
        old = self.PHASE_NAMES[int(self._phase[env_i].item())]
        new = self.PHASE_NAMES[new_phase]
        self._phase[env_i] = new_phase
        self._dwell[env_i] = 0

        # Track cycle start when entering HOVER_UP (start of pick cycle)
        if new_phase == self.PH_HOVER_UP:
            self._logs_at_cycle_start[env_i] = len(self._deposited_logs[env_i])

        # Reset drop goal cache when entering LOWER_TO_DROP phase
        if new_phase == self.PH_LOWER_TO_DROP:
            self._drop_goal_calculated[env_i] = False

        # Reset gripper timers when entering CLOSE phase
        if new_phase == self.PH_CLOSE:
            self._gripper_timeout_timer[env_i] = 0
            self._gripper_stability_timer[env_i] = 0

        # Reset general phase timer on any transition
        self._phase_timer[env_i] = 0
        
        # Reset yaw print flag when leaving ALIGN_YAW phase
        if hasattr(self, f'_yaw_printed_{env_i}'):
            delattr(self, f'_yaw_printed_{env_i}')

        # Suppress phase transition prints during hierarchical internal loop
        if not self._in_hierarchical_loop:
            print(f"[env{env_i}:{old} → {new}] {reason}")


    def _find_highest_trailer_log(self, env_i: int, stack_region: str = "ALL", verbose: bool = False) -> float:
        """Find the Z coordinate of the highest deposited log in the specified stack region (in base frame coordinates)."""
        highest_z = 0  # Default minimum drop height (near trailer level)
        logs_found = 0
        
        try:
            # Use the deposited logs tracking
            if hasattr(self, '_logs_obj') and self._logs_obj is not None and len(self._deposited_logs[env_i]) > 0:
                # Get base frame transformation
                root_pose_w = self.crane.data.root_pose_w[env_i]
                base_pos_w = root_pose_w[0:3].unsqueeze(0)  # Add batch dimension
                base_quat_w = root_pose_w[3:7].unsqueeze(0)
                
                for global_log_idx in self._deposited_logs[env_i]:
                    if global_log_idx < self._logs_obj.num_instances:
                        # Get the position of this deposited log in world frame
                        log_pos_w = self._logs_obj.data.root_pos_w[global_log_idx].unsqueeze(0)  # Add batch dimension
                        log_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)  # Dummy orientation

                        # Transform to base frame (same as rest of system)
                        log_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, log_pos_w, log_quat_w)
                        x = log_pos_b[0, 0].item()  # X in base frame
                        y = log_pos_b[0, 1].item()  # Y in base frame
                        z = log_pos_b[0, 2].item()  # Z in base frame

                        # Filter by stack region (rear y=2.2, front y=4.85, midpoint ~3.5)
                        in_region = False
                        if stack_region == "ALL":
                            in_region = True
                        elif stack_region == "REAR":
                            in_region = (y < 3.5)  # Rear stack
                        elif stack_region == "FRONT":
                            in_region = (y >= 3.5)  # Front stack

                        if in_region:
                            highest_z = max(highest_z, z)
                            logs_found += 1
        except Exception as e:
            if verbose:
                print(f"[env{env_i}] Warning: Error finding highest deposited log: {e}")
        
        if verbose:
            print(f"[env{env_i}] TRAILER-HEIGHT[{stack_region}]: Found {logs_found} deposited logs, highest at z={highest_z:.2f}m (base frame)")
        return highest_z

    def _check_deposited_logs(self, env_i: int) -> set:
        """Check all logs and identify which ones are now in the trailer (check_deposited_logs)."""
        newly_deposited = set()
        
        try:
            if hasattr(self, '_logs_obj') and self._logs_obj is not None:
                # Get environment offset for this env
                per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else 200
                env_offset = env_i * 200  # Each env has 200 slots
                env_logs_end = env_offset + per_env  # Only active logs
                
                if env_offset < self._logs_obj.num_instances:
                    # Get crane base pose for coordinate transformation
                    root_pose_w = self.crane.data.root_pose_w[env_i]
                    base_pos_w = root_pose_w[0:3]
                    base_quat_w = root_pose_w[3:7]
                    
                    # Get positions of logs for this environment
                    log_positions = self._logs_obj.data.root_pos_w[env_offset:env_logs_end]
                    
                    for log_idx, log_pos in enumerate(log_positions):
                        global_log_idx = env_offset + log_idx
                        
                        # Skip already deposited logs
                        if global_log_idx in self._deposited_logs[env_i]:
                            continue
                            
                        # Transform log position from world to crane base coordinates
                        log_pos_w = log_pos.unsqueeze(0)  # Add batch dimension for subtract_frame_transforms
                        log_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0)  # Dummy orientation
                        log_pos_b, _ = subtract_frame_transforms(
                            base_pos_w.unsqueeze(0), base_quat_w.unsqueeze(0), 
                            log_pos_w, log_quat_w
                        )
                        
                        # Extract crane base coordinates
                        x_crane_base, y_crane_base, z_crane_base = log_pos_b[0, 0].item(), log_pos_b[0, 1].item(), log_pos_b[0, 2].item()

                        # Convert to trailer coordinate system
                        x_trailer, y_trailer = y_crane_base, x_crane_base  # Apply coordinate transform

                        # Check if log is in trailer boundaries (is_log_in_trailer)
                        if (self._trailer_x_min <= x_trailer <= self._trailer_x_max and 
                            self._trailer_y_min <= y_trailer <= self._trailer_y_max):
                            newly_deposited.add(global_log_idx)
                            
        except Exception as e:
            print(f"[env{env_i}] Warning: Error checking deposited logs: {e}")
        
        return newly_deposited

    def _compute_drop_goal(self, env_i: int, verbose: bool = False) -> torch.Tensor:
        """dual-stack drop positioning system."""
        # Stack switching logic based on total deposited logs
        total_deposited = len(self._deposited_logs[env_i])  # Use actual deposited set size
        if self._current_stack[env_i] == "REAR" and total_deposited >= self._stack_switch_threshold:
            self._current_stack[env_i] = "FRONT"
            if not self._in_hierarchical_loop:
                print(f"[env{env_i}] STACK-SWITCH: Switching to FRONT stack after {total_deposited} total deposited logs")
        
        # Determine current home position and stack region
        if self._current_stack[env_i] == "REAR":
            home_position = self._home_rear
            stack_region = "REAR"
        else:
            home_position = self._home_front  
            stack_region = "FRONT"
        
        # Calculate dynamic drop height based on existing logs in stack region
        trailer_height = self._find_highest_trailer_log(env_i, stack_region, verbose=verbose)
        drop_height = trailer_height + self._drop_height_offset
        
        # Use home position X,Y coordinates with dynamic Z height
        x, y = home_position[0].item(), home_position[1].item()
        z = drop_height
        
        if verbose:
            print(f"[env{env_i}] STACK-INFO: Using {self._current_stack[env_i]} stack, drop at ({x:.2f},{y:.2f},{z:.2f})")
            print(f"[env{env_i}] DROP-HEIGHT[{stack_region}]: stack_highest={trailer_height:.2f}m, drop_at={drop_height:.2f}m")
        
        return torch.tensor([x, y, z], device=self.device, dtype=self._ee_goal.dtype)

    def _compute_grapple_stability(self, env_i: int) -> float:
        """Compute grapple stability (how level it is after grasping).

        Measures the dot product between grapple's up vector and world up.
        A perfectly level grapple has stability = 1.0.
        A tilted grapple (due to off-center load) has stability < 1.0.

        Args:
            env_i: Environment index

        Returns:
            stability: 1.0 = perfectly level, 0.0 = horizontal (90° tilt)
        """
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]

        # Get grapple's local Z-axis (up direction) in world frame
        # Rotate [0, 0, 1] by basegrapple quaternion
        grapple_up = self._quat_rotate_vec_wxyz(bg_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device))

        # Dot product with world up [0, 0, 1]
        # 1.0 = perfectly level, 0.0 = tilted 90°
        stability = grapple_up[2].item()  # Just the Z component of grapple_up

        # Clamp to [0, 1] (negative means upside down, shouldn't happen)
        return max(0.0, stability)

    def _check_grasped_logs(self, env_i: int, proximity_radius: float = 1.5) -> tuple[int, float, float]:
        """Check how many logs are grasped (near basegrapple) after LIFT_HIGH.

        Args:
            env_i: Environment index
            proximity_radius: Distance threshold in meters (default 1.5m to capture ~20 logs in grapple)

        Returns:
            logs_grasped: Number of logs within proximity
            avg_alignment: Average orientation alignment score (0-1, where 1 is perfectly aligned)
            stability: Grapple stability score (1.0 = level, <1.0 = tilted due to off-center load)
        """
        if self._logs_obj is None:
            return 0, 0.0, 1.0

        # Get basegrapple pose in world frame
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_pos_w = bg_pose_w[0:3]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]

        # Extract basegrapple forward direction (assuming forward is +X in body frame)
        # Rotate [1, 0, 0] by basegrapple quaternion to get forward direction in world frame
        bg_forward = self._quat_rotate_vec_wxyz(bg_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device))

        # Get all log positions for this environment
        per_env = int(self._per_env_log_counts[env_i])  # Actual active log count
        start = env_i * 200  # Each env has 200 slots
        end = start + per_env  # Only check active logs

        logs_grasped = 0
        alignment_sum = 0.0

        for log_idx in range(start, end):
            # Skip already deposited logs
            if log_idx in self._deposited_logs[env_i]:
                continue

            log_pos_w = self._logs_obj.data.root_pos_w[log_idx]
            distance = torch.norm(log_pos_w - bg_pos_w).item()

            if distance < proximity_radius:
                logs_grasped += 1

                # Compute orientation alignment between log and basegrapple
                log_quat_w = self._logs_obj.data.root_quat_w[log_idx]  # [w, x, y, z]

                # Get log's forward direction (assuming log's length is along +X axis)
                log_forward = self._quat_rotate_vec_wxyz(log_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device))

                # Compute alignment as absolute value of dot product (since logs can point either direction)
                # 1.0 = perfectly aligned, 0.0 = perpendicular
                dot_product = torch.dot(bg_forward, log_forward).item()
                alignment = abs(dot_product)  # Use abs to handle logs pointing in opposite direction

                alignment_sum += alignment

        avg_alignment = alignment_sum / max(logs_grasped, 1)

        # Compute grapple stability (how level it is after lifting)
        stability = self._compute_grapple_stability(env_i)

        return logs_grasped, avg_alignment, stability

    def _count_logs_in_column(self, env_i: int, y_tolerance: float = 0.5) -> int:
        """Count available logs in a rack slice at the grapple's Y position.

        Uses a rectangular prism that is a segment of the rack bounds:
        - X: Full rack width (min_x to max_x)
        - Y: Narrow band around target Y position (±y_tolerance)
        - Z: Full rack height (min_z to max_z)

        This represents all logs that could potentially be grasped when the
        grapple descends at the target Y position across the rack width.

        Args:
            env_i: Environment index
            y_tolerance: Half-width of Y band around target (default 0.5m)

        Returns:
            Number of logs in the rack slice
        """
        if self._logs_obj is None:
            if self.cfg.debug_reward:
                print(f"[DEBUG] _count_logs_in_column env{env_i}: logs_obj is None, returning 0")
            return 0

        # Make sure action bounds are computed
        if not hasattr(self, '_action_bounds_valid') or not self._action_bounds_valid[env_i]:
            if self.cfg.debug_reward:
                print(f"[DEBUG] _count_logs_in_column env{env_i}: bounds not valid, returning 0")
            return 0

        # Get target position in body frame (this is where the grapple is going)
        target_pos_b = self._target_log_pos_b[env_i]
        target_y = target_pos_b[1].item()

        # Get rack bounds in body frame
        min_bounds = self._action_bounds_min[env_i]
        max_bounds = self._action_bounds_max[env_i]

        # Define the prism slice bounds
        slice_x_min = min_bounds[0].item()
        slice_x_max = max_bounds[0].item()
        slice_y_min = target_y - y_tolerance
        slice_y_max = target_y + y_tolerance
        slice_z_min = min_bounds[2].item()
        slice_z_max = max_bounds[2].item()

        # Get crane base pose for coordinate transform (world -> body)
        root_pose_w = self.crane.data.root_pose_w[env_i]
        base_pos_w = root_pose_w[0:3]
        base_quat_w = root_pose_w[3:7]

        # Inverse rotation
        base_quat_inv = base_quat_w.clone()
        base_quat_inv[1:4] = -base_quat_inv[1:4]

        per_env = int(self._per_env_log_counts[env_i])
        start = env_i * 200
        end = start + per_env

        count = 0
        for log_idx in range(start, end):
            if log_idx in self._deposited_logs[env_i]:
                continue

            # Get log position in world frame, transform to body frame
            log_pos_w = self._logs_obj.data.root_pos_w[log_idx]
            log_rel_w = log_pos_w - base_pos_w
            log_pos_b = self._quat_rotate_vec_wxyz(base_quat_inv, log_rel_w)

            # Check if within prism slice bounds
            x, y, z = log_pos_b[0].item(), log_pos_b[1].item(), log_pos_b[2].item()
            if (slice_x_min <= x <= slice_x_max and
                slice_y_min <= y <= slice_y_max and
                slice_z_min <= z <= slice_z_max):
                count += 1

        # Debug print
        if self.cfg.debug_reward:
            print(f"[DEBUG] Env {env_i}: Rack slice X=[{slice_x_min:.2f}, {slice_x_max:.2f}], "
                  f"Y=[{slice_y_min:.2f}, {slice_y_max:.2f}], Z=[{slice_z_min:.2f}, {slice_z_max:.2f}] "
                  f"-> {count} logs available")

        # Store for debug visualization
        if self.cfg.debug_reward:
            # Prism center in body frame
            prism_center_b = torch.tensor([
                (slice_x_min + slice_x_max) / 2.0,
                (slice_y_min + slice_y_max) / 2.0,
                (slice_z_min + slice_z_max) / 2.0
            ], device=self.device)
            # Convert to world frame
            prism_center_w = base_pos_w + self._quat_rotate_vec_wxyz(base_quat_w, prism_center_b)
            self._debug_cylinder_pos_w[env_i] = prism_center_w
            self._debug_cylinder_active[env_i] = True
            # Store prism dimensions for visualization
            if not hasattr(self, '_debug_prism_size'):
                self._debug_prism_size = torch.zeros((self.num_envs, 3), device=self.device)
            self._debug_prism_size[env_i, 0] = slice_x_max - slice_x_min
            self._debug_prism_size[env_i, 1] = slice_y_max - slice_y_min
            self._debug_prism_size[env_i, 2] = slice_z_max - slice_z_min

        return count

    def _count_logs_in_rack(self, env_i: int) -> int:
        """Count logs still in rack (not yet deposited or knocked off).

        Note: Knocked off logs are added to _deposited_logs, so they're included in the count.
        """
        per_env = int(self._per_env_log_counts[env_i])
        total_logs = per_env
        removed = len(self._deposited_logs[env_i])  # Includes both deposited and knocked off
        return total_logs - removed

    def _check_logs_out_of_bounds(self, env_i: int, apply_penalty: bool = True) -> int:
        """Check how many logs fell out of the rack bounds.

        Uses the same bounds as the action space visualization (green box).
        Bounds are in base frame - we convert log positions to base frame to check.

        Args:
            env_i: Environment index
            apply_penalty: If False, just clean up without counting (for initial settling)

        Returns:
            Number of logs that are now out of bounds (0 if apply_penalty=False)
        """
        if self._logs_obj is None:
            return 0

        # Make sure action bounds are computed
        if not hasattr(self, '_action_bounds_valid') or not self._action_bounds_valid[env_i]:
            return 0

        per_env = int(self._per_env_log_counts[env_i])  # Actual active log count
        start = env_i * 200  # Each env has 200 slots
        end = start + per_env  # Only check active logs

        # Get bounds in base frame (same as action space / visualization)
        min_bounds = self._action_bounds_min[env_i]  # [x_min, y_min, z_min]
        max_bounds = self._action_bounds_max[env_i]  # [x_max, y_max, z_max]

        # Get crane base pose for coordinate transform
        root_pose_w = self.crane.data.root_pose_w[env_i]
        base_pos_w = root_pose_w[0:3]
        base_quat_w = root_pose_w[3:7]  # [w, x, y, z]

        # Compute inverse rotation (conjugate for unit quaternion)
        base_quat_inv = base_quat_w.clone()
        base_quat_inv[1:4] = -base_quat_inv[1:4]  # Negate xyz components

        out_of_bounds_count = 0
        for log_idx in range(start, end):
            if log_idx >= self._logs_obj.num_instances:
                break
            if log_idx in self._deposited_logs[env_i]:
                continue  # Already deposited, don't check

            # Get log position in world frame
            log_pos_w = self._logs_obj.data.root_pos_w[log_idx]

            # Transform to base frame: log_pos_b = R_inv * (log_pos_w - base_pos_w)
            log_rel_w = log_pos_w - base_pos_w
            log_pos_b = self._quat_rotate_vec_wxyz(base_quat_inv, log_rel_w)

            # Check if outside action bounds (X, Y, and Z)
            out_of_bounds = (
                log_pos_b[0] < min_bounds[0] or log_pos_b[0] > max_bounds[0] or  # X bounds
                log_pos_b[1] < min_bounds[1] or log_pos_b[1] > max_bounds[1] or  # Y bounds
                log_pos_b[2] < min_bounds[2]  # Z min (below rack) - don't check z_max (logs can be lifted high)
            )

            if out_of_bounds:
                # Mark as deposited so it won't be checked again
                self._deposited_logs[env_i].add(log_idx)
                # Despawn by moving far away
                self._logs_obj.data.root_pos_w[log_idx, 0] = -1000.0
                self._logs_obj.data.root_pos_w[log_idx, 1] = -1000.0
                self._logs_obj.data.root_pos_w[log_idx, 2] = -1000.0

                # Zero velocities
                self._logs_obj.data.root_lin_vel_w[log_idx] = torch.zeros(3, device=self.device)
                self._logs_obj.data.root_ang_vel_w[log_idx] = torch.zeros(3, device=self.device)

                out_of_bounds_count += 1

        # Write changes to simulation if any logs were despawned
        if out_of_bounds_count > 0:
            root_pose = torch.cat([self._logs_obj.data.root_pos_w, self._logs_obj.data.root_quat_w], dim=-1)
            self._logs_obj.write_root_pose_to_sim(root_pose)

            root_vel = torch.cat([self._logs_obj.data.root_lin_vel_w, self._logs_obj.data.root_ang_vel_w], dim=-1)
            self._logs_obj.write_root_velocity_to_sim(root_vel)

        # Always return the count - the caller decides whether to penalize
        return out_of_bounds_count


    def _compute_grasp_reward(self, logs_grasped: int, alignment: float, stability: float, env_i: int) -> float:
        """Compute reward for a grasp attempt.

        Best-practice design goals:
          (1) Clear the pile: reward should scale with the *fraction of the pile removed*.
          (2) Quality: require good alignment and (optionally) stability.
          (3) Robust to domain randomization: keep reward magnitudes comparable when pile size varies.

        New formula (enable with cfg.reward_formula = "progress_gate"):

            r = S * (g / N0) * gate(a) * gate(s) + bonus_clear + penalty_empty

        where:
          - g is logs_grasped for this attempt,
          - N0 is the number of logs spawned at reset for env_i (self._per_env_log_counts[env_i]),
          - a is alignment in [0,1],
          - s is stability in [0,1] (or 1 if cfg.use_stability_reward is False),
          - gate(x) = eps + (1-eps)*clip(x,0,1) (soft multiplicative gating),
          - S is cfg.reward_progress_scale (default 100.0),
          - bonus_clear is cfg.reward_clear_bonus (default 0.0),
          - penalty_empty is cfg.empty_target_penalty (default 0.0; set negative to enable).

        We still track and remove logs knocked out of rack bounds, but do not penalize them by default.
        """
        # Always clean up logs that fell out of bounds and track how many fell.
        knocked_off = self._check_logs_out_of_bounds(env_i, apply_penalty=True)

        # Only attribute "knocked off" to the policy after initial settling.
        if getattr(self, "_initial_settling_complete", True):
            self._logs_knocked_off[env_i] = knocked_off
        else:
            knocked_off = 0

        # Episode log count (variable under DR).
        n0 = int(self._per_env_log_counts[env_i]) if hasattr(self, "_per_env_log_counts") else 200
        n0 = max(1, n0)

        raw_available = int(self._logs_available_at_target[env_i].item()) if hasattr(self, "_logs_available_at_target") else 0
        empty_target_penalty = float(getattr(self.cfg, "empty_target_penalty", 0.0))  # set negative to discourage empty picks

        # Failed grasp: just return failure penalty (+ optional empty-target penalty).
        if logs_grasped <= 0:
            total = float(self.cfg.failure_penalty)
            if empty_target_penalty != 0.0 and raw_available <= 0:
                total += empty_target_penalty
            return total

        # Clamp quality terms.
        a = max(0.0, min(float(alignment), 1.0))
        if getattr(self.cfg, "use_stability_reward", False):
            s = max(0.0, min(float(stability), 1.0))
        else:
            s = 1.0

        # Soft gates (eps prevents reward from collapsing to 0 early in training).
        eps_a = float(getattr(self.cfg, "reward_gate_eps_align", 0.10))
        eps_s = float(getattr(self.cfg, "reward_gate_eps_stab", 0.10))
        eps_a = max(0.0, min(eps_a, 1.0))
        eps_s = max(0.0, min(eps_s, 1.0))
        gate_a = eps_a + (1.0 - eps_a) * a
        gate_s = eps_s + (1.0 - eps_s) * s

        # === New default (recommended) ===
        if getattr(self.cfg, "reward_formula", "multiplicative") == "progress_gate":
            g_norm = float(logs_grasped) / float(n0)  # throughput as fraction of pile
            scale = float(getattr(self.cfg, "reward_progress_scale", 100.0))
            total_reward = scale * g_norm * gate_a * gate_s

            # Optional clear bonus: if removing these logs would empty the rack.
            clear_bonus = float(getattr(self.cfg, "reward_clear_bonus", 0.0))
            if clear_bonus != 0.0:
                remaining = self._count_logs_in_rack(env_i)
                if remaining <= logs_grasped:
                    total_reward += clear_bonus

            # Optional empty-target penalty (usually 0; enable if policy gets stuck picking empty middle).
            if empty_target_penalty != 0.0 and raw_available <= 0:
                total_reward += empty_target_penalty

            if getattr(self.cfg, 'debug_reward', False):
                print(
                    f"[DEBUG] Env {env_i}: progress_gate g={logs_grasped}/{n0} ({g_norm:.3f}), "
                    f"gate_a={gate_a:.3f} (a={a:.3f}), gate_s={gate_s:.3f} (s={s:.3f}), "
                    f"knocked_off={knocked_off} -> reward={total_reward:.3f}"
                )

            # NOTE: we do NOT penalize knocked_off by default in this formula.
            return float(total_reward)

        # === Original implementation (kept for backwards-compatibility) ===
        # Compute efficiency (normalized or raw)
        if self.cfg.normalize_reward:
            max_graspable = self.cfg.max_graspable_logs
            available = min(max(1, raw_available), max_graspable)
            efficiency = float(logs_grasped) / float(available)
            # Clamp efficiency to [0, 1] for additive formula
            efficiency = min(efficiency, 1.0)
        else:
            efficiency = float(logs_grasped)
            available = None  # Not used

        if self.cfg.reward_formula == "additive":
            if self.cfg.normalize_reward:
                if getattr(self.cfg, 'use_stability_reward', False):
                    total_reward = efficiency + alignment + stability
                else:
                    total_reward = efficiency + alignment
            else:
                if getattr(self.cfg, 'use_stability_reward', False):
                    total_reward = (efficiency / 20.0) + alignment + stability
                else:
                    total_reward = (efficiency / 20.0) + alignment
        else:
            stability_factor = stability if getattr(self.cfg, 'use_stability_reward', False) else 1.0
            if self.cfg.normalize_reward:
                base_reward = efficiency * 10.0 * alignment * stability_factor
                if alignment > 0.7:
                    alignment_bonus = efficiency * 10.0 * (alignment - 0.7) * 5.0 * stability_factor
                    total_reward = base_reward + alignment_bonus
                else:
                    total_reward = base_reward
            else:
                base_reward = efficiency * alignment * stability_factor
                if alignment > 0.7:
                    alignment_bonus = efficiency * (alignment - 0.7) * 5.0 * stability_factor
                    total_reward = base_reward + alignment_bonus
                else:
                    total_reward = base_reward

        # Out-of-bounds penalty term (legacy). Set self._logs_out_of_bounds_penalty=0.0 to disable.
        total_reward = total_reward + (knocked_off * self._logs_out_of_bounds_penalty)
        return float(total_reward)

    def _despawn_grasped_logs(self, env_i: int, proximity_radius: float = 1.5):
        """Despawn logs that were grasped (no-deposition mode).

        Instead of depositing logs, we teleport them far away and mark them as removed.
        This prevents deposition physics from disturbing the pile.

        Args:
            env_i: Environment index
            proximity_radius: Distance threshold to consider a log as grasped
        """
        if self._logs_obj is None:
            return

        # Get basegrapple pose
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_pos_w = bg_pose_w[0:3]

        # Get log range for this environment
        per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else 200
        start = env_i * 200  # Each env has 200 slots
        end = start + per_env  # Only active logs

        # Clone current state (don't modify the read-only simulation buffers directly!)
        current_pos = self._logs_obj.data.root_pos_w.clone()
        current_quat = self._logs_obj.data.root_quat_w.clone()
        current_lin_vel = self._logs_obj.data.root_lin_vel_w.clone()
        current_ang_vel = self._logs_obj.data.root_ang_vel_w.clone()

        despawned_count = 0
        logs_to_despawn = []

        for log_idx in range(start, end):
            # Skip already removed logs
            if log_idx in self._deposited_logs[env_i]:
                continue

            log_pos_w = current_pos[log_idx]
            distance = torch.norm(log_pos_w - bg_pos_w).item()

            if distance < proximity_radius:
                logs_to_despawn.append(log_idx)

        # Despawn all identified logs
        for log_idx in logs_to_despawn:
            # Teleport log far away (below ground, out of sight)
            current_pos[log_idx] = torch.tensor([0.0, 0.0, -1000.0], device=self.device, dtype=current_pos.dtype)
            # Keep orientation unchanged
            # Zero out velocities
            current_lin_vel[log_idx] = 0.0
            current_ang_vel[log_idx] = 0.0
            # Mark as deposited (removed from game)
            self._deposited_logs[env_i].add(log_idx)
            despawned_count += 1

        # Write updated poses and velocities to simulation
        if despawned_count > 0:
            root_pose = torch.cat([current_pos, current_quat], dim=-1)
            self._logs_obj.write_root_pose_to_sim(root_pose)

            root_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)
            self._logs_obj.write_root_velocity_to_sim(root_vel)

    @staticmethod
    def _quat_rotate_vec_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector by quaternion (w,x,y,z format).

        Args:
            q: Quaternion [w, x, y, z]
            v: Vector [x, y, z]

        Returns:
            Rotated vector
        """
        w = q[..., 0:1]
        qv = q[..., 1:4]
        uv = torch.cross(qv, v, dim=-1)
        uuv = torch.cross(qv, uv, dim=-1)
        return v + 2.0 * (w * uv + uuv)

    def _draw_debug_markers(self):
        if not self._viz_enabled or self._viz is None:
            return
        try:
            N = self.num_envs
            root_pose_w = self.crane.data.root_pose_w
            base_pos_w = root_pose_w[:, 0:3]
            base_quat_w = root_pose_w[:, 3:7]
            bg_target_b = self._dbg_target_bg.clone()
            if (bg_target_b.abs().sum(dim=1) == 0).any():
                # Fallback: use current basegrapple if no target set
                unknown = (bg_target_b.abs().sum(dim=1) == 0).nonzero(as_tuple=False).flatten()
                if unknown.numel() > 0:
                    bg_target_b[unknown] = torch.zeros_like(bg_target_b[unknown])
            bg_target_w = base_pos_w + self._quat_rotate_vec_wxyz(base_quat_w, bg_target_b)
            self._viz["bg_target"].visualize(bg_target_w, base_quat_w)

            log_b = self._target_log_pos_b.clone()
            if (log_b.abs().sum(dim=1) == 0).any():
                for idx in (log_b.abs().sum(dim=1) == 0).nonzero(as_tuple=False).flatten().tolist():
                    log_b[idx], _, _ = self._target_top_log_center_b(idx)  # Ignore log ID & quat in visualization
                    pass  # removed debug print
            
            log_pos_w = base_pos_w + self._quat_rotate_vec_wxyz(base_quat_w, log_b)

            # Visualize target as red sphere
            self._viz["log"].visualize(log_pos_w)            # Action space bounding box visualization
            # Only show if --show_action_bounds flag is set
            if getattr(args_cli, 'show_action_bounds', False):
                # Ensure bounds exist at least once (e.g., heuristic runs may not precompute them)
                if (not hasattr(self, '_action_bounds_valid')) or (not self._action_bounds_valid.any()):
                    self._compute_action_space_bounds()
                if hasattr(self, '_action_bounds_valid') and self._action_bounds_valid.any():
                    bbox_centers_b = (self._action_bounds_min + self._action_bounds_max) / 2.0
                    bbox_sizes_b = self._action_bounds_max - self._action_bounds_min
                    bbox_centers_w = base_pos_w + self._quat_rotate_vec_wxyz(base_quat_w, bbox_centers_b)
                    self._viz["action_bounds"].visualize(
                        translations=bbox_centers_w,
                        orientations=base_quat_w,
                        scales=bbox_sizes_b,
                    )

            # Grasp count prism visualization (rack slice)
            # Only show when debug_reward is enabled and prism is active
            if self.cfg.debug_reward and "grasp_prism" in self._viz:
                if hasattr(self, '_debug_cylinder_active') and self._debug_cylinder_active.any():
                    active_mask = self._debug_cylinder_active
                    prism_pos = self._debug_cylinder_pos_w.clone()
                    # For inactive envs, move prism far away (hide it)
                    prism_pos[~active_mask] = torch.tensor([0.0, 0.0, -100.0], device=self.device)
                    # Use base orientation for the prism
                    prism_quat = base_quat_w.clone()
                    # Scale based on stored prism dimensions
                    if hasattr(self, '_debug_prism_size'):
                        prism_scales = self._debug_prism_size.clone()
                        prism_scales[~active_mask] = 0.0  # Hide inactive
                    else:
                        prism_scales = torch.ones((N, 3), device=self.device)
                    self._viz["grasp_prism"].visualize(
                        translations=prism_pos,
                        orientations=prism_quat,
                        scales=prism_scales,
                    )
        except Exception as e:
            print(f"[VIZ] visualize failed: {e}")

@torch.jit.script
def compute_rewards(rew_alive: float, rew_pos_l2: float, rew_vel_l1: float, q: torch.Tensor, qd: torch.Tensor):
    r_alive = torch.full((q.shape[0],), rew_alive, device=q.device)
    r_pos   = rew_pos_l2 * torch.sum(q * q, dim=-1)
    r_vel   = rew_vel_l1 * torch.sum(torch.abs(qd), dim=-1)
    return r_alive + r_pos + r_vel

# ===== Main (smoke test / heuristic run) =====
def main():
    cfg = CraneDirectEnvCfgFull()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.crane_cfg = cfg.crane_cfg.replace(
        init_state=cfg.crane_cfg.init_state.replace(pos=(args_cli.crane_x, args_cli.crane_y, args_cli.crane_z))
    )
    use_tgs = (args_cli.solver == "tgs")
    cfg.sim.physx.solver_type = (1 if use_tgs else 0)
    cfg.sim.physx.enable_stabilization = True
    cfg.sim.physx.bounce_threshold_velocity = 0.2
    cfg.sim.physx.enable_ccd = True

    # High solver robustness to prevent penetration
    cfg.sim.physx.min_position_iteration_count = 192  # Increased from 150
    cfg.sim.physx.max_position_iteration_count = 255
    cfg.sim.physx.min_velocity_iteration_count = 65
    cfg.sim.physx.max_velocity_iteration_count = 255

    # Tighter friction thresholds for more accurate contact resolution
    cfg.sim.physx.friction_offset_threshold = 0.01       # From default 0.04
    cfg.sim.physx.friction_correlation_distance = 0.00625  # From default 0.025
    cfg.seed = 0 if args_cli.seed is None else max(0, int(args_cli.seed))

    # Set settling time from CLI args
    cfg.settle_time = args_cli.settle_time

    # Set gripper closing speed parameters from CLI args
    cfg.gripper_close_step = args_cli.gripper_close_step
    cfg.gripper_close_delay_s = args_cli.gripper_close_delay
    cfg.gripper_max_velocity = args_cli.gripper_max_velocity
    cfg.gripper_kp = args_cli.gripper_kp

    # Reward normalization debug mode
    if getattr(args_cli, 'debug_reward_norm', False):
        cfg.normalize_reward = True
        cfg.debug_reward = True
        print("[INFO]: Reward normalization debug mode ENABLED")
        print("[INFO]:   - Cylinder visualization shows logs counted for normalization")
        print("[INFO]:   - Debug prints show efficiency calculations")

    env = CraneDirectEnvFull(cfg)
    print("[INFO]: Completed setting up the environment...")

    # Trigger initial reset which will run settling
    # This calls _reset_idx() which spawns logs and runs _settle_logs()
    env.reset()
    print("[INFO]: Initial reset complete. Ready to start task.")

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # Heuristic FSM handles everything - zero actions
            actions = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

            obs, rew, terminated, truncated, info = env.step(actions)
            count += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()