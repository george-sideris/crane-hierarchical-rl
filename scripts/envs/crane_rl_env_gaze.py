#!/usr/bin/env python3
# crane_rl_env.py
#
# Direct-workflow RL environment for the crane with task-space control (Differential IK),
# rack + vectorized logs, and a heuristic expert.
#
#
# Copyright (c) 2022-2025

from __future__ import annotations
import os, math, json, argparse, collections, statistics
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


# Machine joint limits for [slew, boom, stick, telescope], in control_joint_names order.
# These are the crane's real limits, which are WIDER than what the sim assets ship:
#   crane.usd   slew +/- 100 deg,   telescope 0.13 .. 1.18 m
#   URDF        slew +/- 100 deg,   telescope 0.13 .. 1.8 m
#   machine     slew +/- 1.82 rad,  telescope 0.13 .. 1.8 m
#               (1.82 rad = 104.2 deg; encoder damage past it)
# The two assets were narrower than the machine in different places, so IK solved
# against one envelope while physics enforced another: a target outside the USD stop
# came back from IK as a valid solution and then silently railed short (0.30 m of rack
# unreachable at the near end from slew, 0.65 m of reach lost at the far end from
# telescope). Applied to the IK chain bounds below AND to the sim at startup (see
# CraneGazeEnv.__init__) rather than by editing crane.usd, which also ships to the FPI
# Omniverse repo. Mirrors ARM_JOINT_LIMITS in fpi_crane_rl/ik.py on the real side.
ARM_JOINT_LIMITS = [
    (-1.82, 1.82),          # basemast_to_mast
    (-0.383972, 1.309),     # mast_to_mainboom
    (-3.08574, 0.035),      # mainboom_to_stick
    (0.13, 1.8),            # stick_to_telescope
]


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
                    # The optimizer must search the machine's limits, not the URDF's, or it
                    # rails the joint at the URDF stop and returns that pose, which still
                    # passes the limit check in solve() (see ARM_JOINT_LIMITS).
                    for link_idx, bounds in zip(range(1, 1 + len(ARM_JOINT_LIMITS)), ARM_JOINT_LIMITS):
                        self.chain.links[link_idx].bounds = bounds
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
                        # Apply joint limits check (machine limits, see ARM_JOINT_LIMITS)
                        joint_limits = ARM_JOINT_LIMITS

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
parser.add_argument("--phase_timeout", type=int, default=250,
                    help="Steps a positioning phase may take before the FSM gives up (~60 Hz, so "
                         "250 = ~4 s). Was 100 (2 s), which cut off far-rack targets: those need a big "
                         "slew sweep plus near-full telescope extension, where DLS differential IK "
                         "converges slowly (the far corner sits at 6.88 m vs a 6.89 m envelope).")
parser.add_argument("--urdf_path", type=str, default="/workspace/crane_testbed/assets/urdf/fpiforwarder-upperpassive.urdf",
                    help="Path to URDF file for simple IK (truncated to upperpassive)")

# Logs grid (per env)
parser.add_argument("--num_logs", type=int, default=200)   # matches the real testbed inventory (200-log piles, e.g. the 2026-07-30 baseline)
parser.add_argument("--rows", type=int, default=40)        # spread along the 7.38 m rack length
parser.add_argument("--layers", type=int, default=8)
parser.add_argument("--spacing_y", type=float, default=0.16)
parser.add_argument("--spacing_z", type=float, default=0.12)
parser.add_argument("--base_z", type=float, default=0.10)
parser.add_argument("--fixed_x", type=float, default=None)     # defaults to rack_x
parser.add_argument("--center_y", type=float, default=None)    # defaults to rack_y

# Jitter
parser.add_argument("--row_y_jitter", type=float, default=0.0)
parser.add_argument("--layer_y_offset", type=float, default=0.08)
parser.add_argument("--spawn_height", type=float, default=0.12)
parser.add_argument("--profile_piles", action="store_true",
                    help="Spawn piles with randomized HEIGHT PROFILES (mound/ramp/two-mounds/flat) on a "
                         "hex-packed, support-constrained lattice that holds its shape through settling. "
                         "Use for BC collection so 'where the pile top is' becomes learnable "
                         "(flat piles cannot teach y-localization). Overrides the pattern choice.")
parser.add_argument("--force_pile_profile", type=str, default=None,
                    choices=["flat", "mound", "ramp_far", "ramp_near", "two_mounds", "jagged"],
                    help="Pin the --profile_piles height profile instead of drawing it at random "
                         "(normally two_mounds appears in only ~1 of 8 episodes). Used for the "
                         "qualitative double-mound diagnostic: with two_mounds the peaks are also "
                         "pinned wide apart (0.25/0.75, narrow) so the valley between them is "
                         "unambiguous. Leave unset for all normal collection, training and eval.")
parser.add_argument("--log_friction", type=float, default=0.0,
                    help="If > 0, apply a rigid-body material with this static friction (dynamic = 0.9x) "
                         "to the logs. Default log material (~0.5) lets mounds relax flat during settling; "
                         "real bark interlocks (~0.9). Raises the angle of repose so height profiles persist.")
parser.add_argument("--max_grasp_cycles", type=int, default=30,
                    help="Episode ends after this many grasp cycles (default 30 = full clearing). "
                         "For profile-pile BC collection set ~10: the expert harvests the mound within "
                         "a few cycles, so shorter episodes keep the mound-state fraction high.")
parser.add_argument("--log_ellipticity", type=float, default=0.0,
                    help="Cross-section OUT-OF-ROUNDNESS: each log variant gets its two transverse "
                         "axes scaled by (1+e) and (1-e) with e drawn up to this value, so logs are "
                         "elliptical rather than perfect cylinders. Perfect cylinders nest into a "
                         "near-optimal hex packing (sim grasps ~22 logs where the real grapple gets "
                         "~13); elliptical sections pack orientation-dependently and loosen the pile. "
                         "NOTE: the 6 variants carry DIFFERENT ellipticity, which is what breaks the "
                         "lattice - there is currently no per-log ROLL randomisation at spawn "
                         "(--jitter_ang is declared but unused), so the effect is partial. Try 0.15.")
parser.add_argument("--log_scale_mean", type=float, default=1.0,
                    help="NOMINAL log size multiplier applied to every log (diameter and length), "
                         "on top of the asset's d=0.113 m. Set this to match the measured mean log "
                         "at the testbed; --log_scale_jitter then adds symmetric spread around it. "
                         "Keeping mean and spread separate means a dataset change is attributable "
                         "to one or the other, not both at once.")
parser.add_argument("--log_scale_jitter", type=float, default=0.0,
                    help="If > 0, spawn logs as a random mix of 6 size variants: diameter scaled "
                         "down by up to this fraction (never up, so the hex lattice can't "
                         "interpenetrate), length varied +-half this. Mixed sizes interlock and "
                         "break perfect-cylinder packing. Try 0.12.")
parser.add_argument("--log_ang_damping", type=float, default=0.05,
                    help="Angular damping for the logs (default 0.05 = near-free rolling). Friction stops "
                         "SLIDING but perfect cylinders ROLL down any slope regardless of friction; real "
                         "logs have bark/taper = rolling resistance. Set ~2-4 so height profiles survive "
                         "settling (collection-time knob; emulates non-cylindrical logs).")
parser.add_argument("--jitter_xy", type=float, default=0.0)
parser.add_argument("--jitter_height", type=float, default=0.0)
parser.add_argument("--jitter_ang", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--mixed_log_patterns", action="store_true", help="Enable mixed log grid patterns across environments for training variety")

# Assets
parser.add_argument("--rack_usd", type=str, default="/workspace/crane_testbed/assets/scenes/rack.usd")  # GAZE: full calibrated rack
# ZED stereo noise model on the raw camera cloud (sim2real). Standard stereo error: axial sigma grows
# ~ depth^2. Coeff FITTED to the real ZED floor in bag 17_20 (~14mm@2.5m, 20mm@3.5m, 26mm@4.5m;
# real saturates ~25mm beyond 5m, the z^2 model overshoots there but the rack is 2-5m).
parser.add_argument("--zed_noise", action="store_true", help="Add ZED-X-like stereo noise to the raw camera cloud")
parser.add_argument("--zed_axial_coeff", type=float, default=0.0014, help="axial noise sigma = coeff * depth^2 (m); fitted to real ZED")
parser.add_argument("--zed_dropout", type=float, default=0.06, help="fraction of depth pixels randomly dropped (holes)")
parser.add_argument("--log_usd", type=str,
                    default="/workspace/crane_testbed/assets/scenes/testbed_log_taper.usd",
                    help="Log asset. Default is the TAPERED log (10.3 cm tip -> 12.4 cm butt "
                         "over 2.45 m, ~1 cm/m like real timber; same mean diameter, materials "
                         "and convex-hull collider as testbed_log.usd, which remains available "
                         "as the uniform-cylinder control).")
parser.add_argument("--crane_usd", type=str, default="/workspace/crane_testbed/assets/scenes/crane.usd")

# Log physics & orientation
parser.add_argument("--log_mass",  type=float, default=15.0,
                    help="Fixed per-log mass [kg]. Only used when --log_density is 0.")
parser.add_argument("--log_density", type=float, default=850.0,
                    help="Wood density [kg/m^3]; PhysX derives each log's mass from its ACTUAL mesh "
                         "volume, so size variants and the tapered profile are handled exactly (no "
                         "nominal-cylinder approximation). 850 = green softwood -> ~21 kg for a "
                         "2.45 m x 11.3 cm log. The most realistic run so far (2026-07-31 04:35) was "
                         "an accident of exactly this: mass_props silently failed and PhysX fell back "
                         "to density 1000, giving ~24.6 kg logs, bundles of 12-21 and lifelike tilt "
                         "spread - against 22-log bundles at the configured 15 kg. Set 0 to go back "
                         "to a fixed --log_mass.")
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
parser.add_argument("--stability_snapshot", action="store_true",
                    help="Legacy stability: single tilt sample at lift-hold end (default is windowed mean over the hold)")
# Passive pendulum joints (sim analog of hanger/bearingfork). The real joints are free pivots;
# spring stiffness shrinks the static tilt (inflates stability) and heavy damping kills the swing.
parser.add_argument("--passive_stiffness", type=float, default=0.0,
                    help="Passive chain joint stiffness (N*m/rad). 0 = free pivot like the real crane. Legacy value: 300.")
parser.add_argument("--passive_damping", type=float, default=100.0,
                    help="Passive chain joint damping (N*m*s/rad). Legacy value: 500 (overdamped for light grasps).")

# Gripper closing control (prevent penetration with slow, gentle closing)
parser.add_argument("--expert_target", type=str, default="highest", choices=["highest", "density"],
                    help="Expert target-selection rule. 'highest' = legacy greedy top point. "
                         "'density' = ORACLE: pick the log whose 1.5 m neighbourhood (the grasp "
                         "check's capture radius) contains the most logs, tie-broken by height. "
                         "Run both over the same seeds and compare cycles-to-clear: the gap is the "
                         "UPPER BOUND on what better target selection (e.g. RL) can win, measured "
                         "before spending a week training for it.")
parser.add_argument("--expert_dig", type=float, default=0.0,
                    help="If >0, the EXPERT executes grasps at (local surface - this) instead of the "
                         "top log's CENTRE (= surface - 0.056). The real baseline needs 0.25-0.30 to "
                         "get a bite the tongs can wrap; at the centre convention the tines only "
                         "graze the top layer. Changing it here moves BOTH the executed grasp and "
                         "the BC labels together (they derive from the same target), so the data "
                         "stays self-consistent - unlike --train_label_dig, which only relabels. "
                         "Clamped to the bed floor.")
parser.add_argument("--gripper_max_lead", type=float, default=1e9,
                    help="How far [rad] the jaw command may run ahead of the LAGGING jaw. DISABLED "
                         "by default (1e9 = open-loop stepping, as before): bounding the lead also "
                         "bounds the squeezing force, which starved the close so badly that the lift "
                         "outran it and bites fell out. Set e.g. 0.35 to re-enable the shared-linkage "
                         "model (keeps the two tongs at equal angles, gentler and slower close).")
parser.add_argument("--lift_step", type=float, default=0.3,
                    help="Interpolation fraction per control step for PH_LIFT_HIGH. NOTE: this is "
                         "NOT a speed knob. The arm is position-controlled, so lift FORCE comes from "
                         "the goal-vs-actual error; shrinking this shrinks the error and the arm can "
                         "no longer lift a loaded grapple (observed at 0.08: the lift stalls, the "
                         "grasp check then counts every log inside its 1.5 m proximity sphere, and "
                         "50+ logs get despawned). To slow the motion use --arm_velocity_scale, "
                         "which caps joint speed while leaving torque intact.")
parser.add_argument("--arm_velocity_scale", type=float, default=1.0,
                    help="Scales the velocity limits of slew/boom/stick/telescope. This is the "
                         "correct way to slow the crane down (speed is capped, torque is not), e.g. "
                         "0.4 to give the tongs more time to close on the way up.")
parser.add_argument("--gripper_close_duration_s", type=float, default=8.0,
                    help="How long the CLOSE command stays live, mirroring the real grapple's "
                         "delta_time exit (grapple_delta_time_ms=8000): the tongs keep squeezing at "
                         "fixed velocity for this long, INCLUDING through the lift, and stop on "
                         "whichever comes first - this timeout or full closure. Without it a bite "
                         "that stalls the jaws never finishes closing once the lift frees them.")
parser.add_argument("--gripper_effort", type=float, default=7500.0,
                    help="Grapple closing effort limit [N*m]. The real grapple has finite hydraulic "
                         "force and CHOKES on oversized bites (6 of 26 cycles in the 2026-07-30 "
                         "baseline, all on the full pile). At the default 7500 the sim jaws close "
                         "through anything, so sim bundles run 20-24 logs vs ~13 real and there is no "
                         "choking failure mode for RL to learn to avoid. Lower this until closure "
                         "starts failing around the real capacity (~20 logs): stalled jaws let the "
                         "bite fall away during LIFT_HIGH, so the grasp count drops on its own.")
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
parser.add_argument("--viz_markers", action="store_true", default=True, help="Enable debug VisualizationMarkers.")
parser.add_argument("--show_action_bounds", action="store_true", help="Show translucent box for action/rack bounds.")
parser.add_argument("--show_trailer_bounds", action="store_true", help="Show wireframe box for the trailer deposit scan bounds.")
# Trailer scan box, base_link frame. Tune live to sit inside the trailer (raise xmin to clear the left poles).
parser.add_argument("--trailer_box_xmin", type=float, default=-0.7)
parser.add_argument("--trailer_box_xmax", type=float, default=0.7)
parser.add_argument("--trailer_box_ymin", type=float, default=0.2)
parser.add_argument("--trailer_box_ymax", type=float, default=5.302)
parser.add_argument("--trailer_box_zmin", type=float, default=0.0)
parser.add_argument("--trailer_box_zmax", type=float, default=1.5)
parser.add_argument("--show_grasp_prism", action="store_true", help="Show wireframe rack-slice prism used by _count_logs_in_column() (debug).")
parser.add_argument("--grasp_prism_edge_thickness", type=float, default=None, help="Edge thickness (m) for grasp prism wireframe; defaults to action bounds thickness.")

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
    # Imported by a collection/training script: parse the REAL command line, ignoring args
    # that belong to the importing script. (This used to be parse_args(args=[]) which
    # silently reset EVERY env flag to its default -- --profile_piles, --spacing_y/z,
    # --rows, --max_grasp_cycles, --log_* etc. never took effect through train_bc_pointcloud.)
    args_cli, _unknown_cli = parser.parse_known_args()
    simulation_app = None

# GAZE env: crane sits at a FIXED world pose, and the rack is placed at the EXACT calibrated
# pose in the crane base frame (sim2real). crane_z=1.423 puts base z=-1.42 (rack bottom) at world z=0.
# Calibrated rack center in base frame (rack_calib.py, 2026-06-28 bag 17_20, both ZEDs floor-calibrated):
# (-4.364, 1.816), yaw -0.40 deg. (prev, pre-crane-move: -5.061, 2.527, 1.77)
RACK_BASE_X, RACK_BASE_Y, RACK_BASE_YAW_DEG = -4.364, 1.816, -0.40
if args_cli.crane_x is None:
    args_cli.crane_x = 6.0
if args_cli.crane_y is None:
    args_cli.crane_y = 0.0
# Derive the rack (and log-spawn center) from the crane + calibrated offset
args_cli.rack_x = args_cli.crane_x + RACK_BASE_X
args_cli.rack_y = args_cli.crane_y + RACK_BASE_Y
if args_cli.fixed_x is None:
    args_cli.fixed_x = args_cli.rack_x
if args_cli.center_y is None:
    args_cli.center_y = args_cli.rack_y
# --log_scale_jitter scales logs UP from nominal, so the lattice needs room for the FATTEST
# variant: inflate cell spacing
# by (1+jit) and shrink the row count by the same factor, so the pile footprint (rows*spacing)
# and hence the rack coverage stay put while the biggest logs stop interpenetrating at spawn.
_jit0 = float(getattr(args_cli, "log_scale_jitter", 0.0) or 0.0)
_mean0 = float(getattr(args_cli, "log_scale_mean", 1.0) or 1.0)
_ecc0 = float(getattr(args_cli, "log_ellipticity", 0.0) or 0.0)
if _jit0 > 0.0 or _mean0 != 1.0 or _ecc0 > 0.0:
    _f = _mean0 * (1.0 + _jit0) * (1.0 + _ecc0)   # widest possible transverse axis
    args_cli.spacing_y *= _f
    args_cli.spacing_z *= _f
    args_cli.rows = max(2, int(round(args_cli.rows / _f)))
    print(f"[LOGS] lattice for largest log (mean {_mean0:.2f} x (1+{_jit0:.2f})): spacing x{_f:.2f} -> y={args_cli.spacing_y:.3f} "
          f"z={args_cli.spacing_z:.3f}, rows -> {args_cli.rows} (footprint preserved)")

y_neg_end = args_cli.center_y - 0.5 * (args_cli.rows - 1) * args_cli.spacing_y

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
    # When cap < full grid, use fewer rows per layer but keep full rack span
    effective_rows = min(rows, cap)
    full_span = (rows - 1) * spacing_y
    effective_spacing = full_span / max(effective_rows - 1, 1) if effective_rows > 1 else spacing_y
    y0 = center_y_local - 0.5 * (effective_rows - 1) * effective_spacing
    for L in range(layers):
        if placed >= cap: break
        zc = base_z + L * spacing_z
        layer_offset = layer_offs[L]
        for i in range(effective_rows):
            if placed >= cap: break
            yc = y0 + i * effective_spacing + layer_offset
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
    """Generate random log pile with varied spacing, jitter, and a HEIGHT PROFILE.

    The height profile (flat / mound / ramps / two mounds) is carved at spawn by capping the
    layer count per column. (An older docstring referenced a post-settling carve via
    _apply_pile_profile(), but that function was never implemented, so DR piles came out as
    uniform flat slabs. A flat pile gives the expert's "highest log" no learnable location,
    so BC policies learned target-z but not where the top is in y.)
    """
    import math
    import random
    if seed is not None:
        random.seed(seed)

    # Randomize grid dimensions
    pattern_rows = random.randint(15, 25)
    pattern_layers_max = random.randint(10, 20)

    # When cap is small (curriculum), use fewer rows but keep full rack span
    effective_rows = min(pattern_rows, cap)
    full_span = (rows - 1) * spacing_y  # full rack width

    # Randomize spacing but ensure logs span the full rack width
    if effective_rows > 1:
        random_spacing_y = full_span / (effective_rows - 1) * random.uniform(0.95, 1.05)
    else:
        random_spacing_y = spacing_y * random.uniform(0.9, 1.15)
    random_spacing_z = spacing_z * random.uniform(0.9, 1.15)

    # Randomize center shift (disabled — keep logs centered on rack)
    max_shift = 0.0
    center_shift = random.uniform(-max_shift, max_shift)
    shifted_center = center_y_local + center_shift

    # --- Height profile: controls max layers per column ---
    # Randomize layer offsets for irregular stacking
    layer_offs = []
    for L in range(pattern_layers_max):
        if random.random() < 0.5:
            offset = (layer_y_offset if (L % 2 == 1) else 0.0)
        else:
            offset = random.uniform(-0.05, 0.05)
        layer_offs.append(offset)

    if pattern_layers_max > 0:
        mean_off = sum(layer_offs) / pattern_layers_max
        layer_offs = [o - mean_off for o in layer_offs]

    # --- Pick a height profile and carve per-column layer caps from it ---
    # Column budgets are scaled so they sum EXACTLY to cap (callers index positions[:cap]).
    profile = random.choice(["flat", "mound", "mound", "ramp_far", "ramp_near", "two_mounds"])
    peak_c = random.uniform(0.15, 0.85)     # mound center (fraction of rack length)
    # Wide mounds + high shoulders: narrow tall columns topple during settling and spill
    # over the rack (despawned), flattening the profile. Gentle wide mounds survive.
    p_width = random.uniform(0.20, 0.45)    # mound width (fraction)
    floor_f = random.uniform(0.30, 0.55)    # edge height as a fraction of the peak

    def _height_frac(c):
        if profile == "mound":
            return floor_f + (1.0 - floor_f) * math.exp(-0.5 * ((c - peak_c) / p_width) ** 2)
        if profile == "ramp_far":
            return floor_f + (1.0 - floor_f) * c
        if profile == "ramp_near":
            return floor_f + (1.0 - floor_f) * (1.0 - c)
        if profile == "two_mounds":
            g1 = math.exp(-0.5 * ((c - 0.28) / p_width) ** 2)
            g2 = math.exp(-0.5 * ((c - 0.72) / p_width) ** 2)
            return floor_f + (1.0 - floor_f) * max(g1, g2)
        return 1.0  # flat

    fr = [_height_frac(col / max(effective_rows - 1, 1)) for col in range(effective_rows)]
    tot = sum(fr)
    col_layers = [max(1, int(round(f / tot * cap))) for f in fr]
    # Physical clamp: columns taller than ~the rack topple and spill out (despawned),
    # which flattens the pile. Keep every column under ~2.0 m of spawn height.
    max_col = max(4, int(2.0 / max(random_spacing_z, 1e-3)))
    col_layers = [min(cl, max_col) for cl in col_layers]
    # Fix rounding/clamping so the column budgets sum exactly to cap
    # (callers index positions[:cap]). Prefer adding to the tallest unclamped columns.
    diff = cap - sum(col_layers)
    order = sorted(range(effective_rows), key=lambda c: -fr[c])
    guard = 0
    while diff != 0 and guard < 20 * cap:
        c = order[guard % effective_rows]
        if diff > 0 and col_layers[c] < max_col:
            col_layers[c] += 1
            diff -= 1
        elif diff < 0 and col_layers[c] > 1:
            col_layers[c] -= 1
            diff += 1
        guard += 1
    if diff > 0:  # every column at the clamp: relax it rather than under-fill
        for c in order:
            add = min(diff, 5)
            col_layers[c] += add
            diff -= add
            if diff == 0:
                break

    # Build positions layer by layer, respecting each column's height budget.
    positions, placed = [], 0
    y0 = shifted_center - 0.5 * (effective_rows - 1) * random_spacing_y
    base_z_variation = random.uniform(-0.02, 0.02)

    L = 0
    while placed < cap and L < 200:
        layer_offset = layer_offs[L % len(layer_offs)] if layer_offs else 0.0
        for col in range(effective_rows):
            if placed >= cap:
                break
            if L >= col_layers[col]:
                continue
            yc = y0 + col * random_spacing_y + layer_offset
            zc = base_z + base_z_variation + L * random_spacing_z
            yc += (random.random() * 2 - 1) * 0.04
            zc += (random.random() * 2 - 1) * 0.02
            positions.append((yc, zc))
            placed += 1
        L += 1

    return positions


def plan_grid_yz_hex_profile(center_y_local: float, rows: int, layers: int, spacing_y: float, spacing_z: float,
                             base_z: float, cap: int, row_y_jitter: float, layer_y_offset: float,
                             seed: Optional[int]):
    """Randomized height-profile pile that is STATICALLY STABLE by construction.

    plan_grid_yz_random drops a jittered square grid and lets physics settle - tall regions
    topple and every profile relaxes to a flat slab (settled cloud z-std ~0.05 vs ~0.125 on
    the real pile), so BC data never teaches WHERE the pile top is. This planner instead
    builds a hex-packed lattice (odd layers offset half a spacing, resting in the grooves of
    the layer below) whose per-column height follows a random profile (flat / mound / ramps /
    two mounds). A support constraint (|dh| <= 1 across adjacent half-columns) guarantees every
    log has two grooves under it, so the profile holds its shape instead of collapsing.
    Returns EXACTLY cap positions (callers index positions[:cap]).
    """
    import math
    import random
    if seed is not None:
        random.seed(seed)

    # Half-column lattice across the full rack span: sites at pitch spacing_y/2; a site (L, k)
    # is valid when k and L have the same parity (hex packing).
    full_span = (rows - 1) * spacing_y
    K = max(int(round(full_span / (spacing_y / 2.0))) + 1, 5)
    y0 = center_y_local - 0.5 * full_span

    # Random height profile (in layers) over the half-column grid.
    profile = random.choice(["flat", "mound", "mound", "ramp_far", "ramp_near",
                             "two_mounds", "jagged", "jagged"])
    peak_c = random.uniform(0.15, 0.85)
    # NARROW mounds on a wide floor: a wide bulge raises half the columns, so max-median
    # prominence stays ~0.1 even when the shape survives; the real pile is a LOCALIZED mound
    # (~0.34 max-median). Repose slope-limiting below keeps narrow shapes physical.
    p_width = random.uniform(0.06, 0.16)
    floor_f = random.uniform(0.05, 0.20)
    # two_mounds: random peak positions (were fixed 0.28/0.72)
    tm_c1 = random.uniform(0.12, 0.45)
    tm_c2 = random.uniform(0.55, 0.88)
    # Opt-in override for the QUALITATIVE double-mound diagnostic: pin the profile so the
    # geometry that broke the regression policy on the real rack is reproducible instead of
    # 1-in-8 random. Default None leaves normal collection/eval/training untouched.
    _forced = getattr(args_cli, "force_pile_profile", None)
    if _forced:
        profile = _forced
        print(f"[pile] profile FORCED -> {_forced}"
              + ("  (peaks pinned 0.25/0.75, width 0.09)" if _forced == "two_mounds" else ""))
        if _forced == "two_mounds":
            # With peaks at 0.25/0.75 and width 0.09, max(g1,g2) at the midpoint is
            # exp(-0.5*(0.25/0.09)^2) ~= 0.02 of peak, so the valley sits essentially at the
            # floor: a policy that averages the two valid modes aims into a hole.
            tm_c1, tm_c2, p_width = 0.25, 0.75, 0.09
    # jagged: 2-5 bumps at random centers/widths/heights -> multi-peaked uneven surface
    bumps = [(random.uniform(0.05, 0.95), random.uniform(0.04, 0.13), random.uniform(0.35, 1.0))
             for _ in range(random.randint(2, 5))]

    def _frac(c):
        if profile == "mound":
            return floor_f + (1.0 - floor_f) * math.exp(-0.5 * ((c - peak_c) / p_width) ** 2)
        if profile == "ramp_far":
            return floor_f + (1.0 - floor_f) * c
        if profile == "ramp_near":
            return floor_f + (1.0 - floor_f) * (1.0 - c)
        if profile == "two_mounds":
            g1 = math.exp(-0.5 * ((c - tm_c1) / p_width) ** 2)
            g2 = math.exp(-0.5 * ((c - tm_c2) / p_width) ** 2)
            return floor_f + (1.0 - floor_f) * max(g1, g2)
        if profile == "jagged":
            v = max(a * math.exp(-0.5 * ((c - y0) / w) ** 2) for y0, w, a in bumps)
            return floor_f + (1.0 - floor_f) * v
        return 1.0

    fr = [_frac(k / max(K - 1, 1)) for k in range(K)]

    # Build the profile in METERS with realistic amplitude and a REPOSE-ANGLE slope limit.
    # (The first version scaled heights to hit cap and only constrained |dh|<=1 per HALF-column,
    # which permits ~60 deg walls -> 2.6 m towers that always collapsed to ~0.11 m prominence.
    # The real pile is only ~0.35 m proud at <=~20 deg.)
    amp = random.uniform(0.25, 0.60)                       # peak height above the profile floor (m)
    pitch = spacing_y / 2.0
    max_slope = math.tan(math.radians(random.uniform(15.0, 22.0)))   # repose-limited
    prof = [amp * (f - floor_f) / max(1.0 - floor_f, 1e-6) for f in fr]   # 0..amp (m)
    # Fine-scale surface roughness on top of the macro shape (real piles are uneven
    # everywhere, not smooth between mounds). Slope-limiting below keeps it physical.
    prof = [max(0.0, p + random.uniform(-0.06, 0.06)) for p in prof]

    def _slope_limit(p):
        for k in range(1, K):
            p[k] = min(p[k], p[k - 1] + max_slope * pitch)
        for k in range(K - 2, -1, -1):
            p[k] = min(p[k], p[k + 1] + max_slope * pitch)
        return p

    prof = _slope_limit(prof)

    # Choose the flat floor F (m) so total sites ~= cap (each half-column hosts every other layer).
    def _heights(F):
        return [max(1, int(round((F + p) / spacing_z))) for p in prof]

    lo, hi = 0.05, 3.0
    for _ in range(30):
        F = 0.5 * (lo + hi)
        if sum(_heights(F)) / 2.0 < cap:
            lo = F
        else:
            hi = F
    h = _heights(hi)

    def _smooth(hh):
        # Support constraint (hex nesting needs both lower neighbors present).
        changed = True
        while changed:
            changed = False
            for k in range(K):
                cap_k = min(hh[k - 1] + 1 if k > 0 else 10 ** 6,
                            hh[k + 1] + 1 if k < K - 1 else 10 ** 6)
                if hh[k] > cap_k:
                    hh[k] = cap_k
                    changed = True
        return hh

    h = _smooth(h)

    # Fill bottom-up, parity-matched sites only; raise the whole profile if capacity runs short.
    positions = []
    guard = 0
    while len(positions) < cap and guard < 300:
        positions = []
        Lmax = max(h)
        for L in range(Lmax):
            for k in range(L % 2, K, 2):
                if L < h[k]:
                    yc = y0 + k * (spacing_y / 2.0) + (random.random() * 2 - 1) * 0.01
                    zc = base_z + L * spacing_z + (random.random() * 2 - 1) * 0.005
                    positions.append((yc, zc))
                if len(positions) >= cap:
                    break
            if len(positions) >= cap:
                break
        if len(positions) < cap:
            h = _smooth([v + 1 for v in h])
        guard += 1

    return positions[:cap]

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
            joint_names_expr=["basemast_to_mast"], effort_limit_sim=800_000.0,
            velocity_limit_sim=1.0 * args_cli.arm_velocity_scale,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_boom": ImplicitActuatorCfg(
            joint_names_expr=["mast_to_mainboom"], effort_limit_sim=1_000_000.0,
            velocity_limit_sim=2.0 * args_cli.arm_velocity_scale,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_stick": ImplicitActuatorCfg(
            joint_names_expr=["mainboom_to_stick"], effort_limit_sim=600_000.0,
            velocity_limit_sim=2.0 * args_cli.arm_velocity_scale,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "crane_telescope": ImplicitActuatorCfg(
            joint_names_expr=["stick_to_telescope"], effort_limit_sim=500_000.0,
            velocity_limit_sim=2.0 * args_cli.arm_velocity_scale,
            stiffness=1_000_000.0, damping=100_000.0
        ),
        "passive_chain": ImplicitActuatorCfg(
            joint_names_expr=["telescope_to_upperpassive", "upperpassive_to_lowerpassive"],
            effort_limit_sim=10000.0, velocity_limit_sim=2.0,
            # Free pivot + light damping by default (see --passive_stiffness/--passive_damping;
            # legacy spring-centered values were stiffness=300, damping=500)
            stiffness=args_cli.passive_stiffness, damping=args_cli.passive_damping
        ),
        "grapple_rotate": ImplicitActuatorCfg(
            joint_names_expr=["lowerpassive_to_basegrapple"],
            effort_limit_sim=20000.0, velocity_limit_sim=5.0, stiffness=50_000.0, damping=20_000.0
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["basegrapple_to_gripper.*"],
            effort_limit_sim=args_cli.gripper_effort, velocity_limit_sim=25.0,
            stiffness=0.0, damping=500.0
        ),
    },
)

# ===== Scene config =====
@configclass
class CraneSceneCfg(InteractiveSceneCfg):
    """Scene configuration with crane articulation and static rack."""

    # Crane articulation
    crane = CRANE_CFG.replace(prim_path="{ENV_REGEX_NS}/Crane")

    # Rack as static asset (logs spawned manually). GAZE: full rack.usd at the calibrated pose:
    # yaw -0.40 deg, height scaled to the real 2.54 m (rack.usd is 2.4 m -> z-scale 2.54/2.4).
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=args_cli.rack_usd, scale=(1.0, 1.0, 2.54 / 2.4)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(args_cli.rack_x, args_cli.rack_y, 0.0),
            rot=(0.999994, 0.0, 0.0, -0.003491),  # yaw -0.40 deg (wxyz)
        ),
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
        num_envs=1024, env_spacing=args_cli.env_spacing,
        # Multi-asset log variants (--log_scale_jitter) require replicate_physics=False:
        # with replication ON, PhysX parses env_0 and replicates it, so per-log size variants
        # silently never take effect (Isaac Lab sets /isaaclab/spawn/multi_assets to flag this).
        replicate_physics=(float(getattr(args_cli, "log_scale_jitter", 0.0) or 0.0) <= 0.0
                           and float(getattr(args_cli, "log_scale_mean", 1.0) or 1.0) == 1.0
                           and float(getattr(args_cli, "log_ellipticity", 0.0) or 0.0) <= 0.0),
        clone_in_fabric=False
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
    # ZED stereo noise model on the raw camera cloud (sim2real). See get_pointcloud_world.
    zed_noise: bool = args_cli.zed_noise
    zed_axial_coeff: float = args_cli.zed_axial_coeff   # axial sigma = coeff * depth^2 (m)
    zed_dropout: float = args_cli.zed_dropout           # fraction of depth pixels dropped (holes)
    log_usd: str = args_cli.log_usd
    log_mass: float = args_cli.log_mass
    log_density: float = args_cli.log_density
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
    heuristic_target_noise: float = 0.0  # Gaussian σ (meters) added to heuristic target position for robustness evals
    heuristic_yaw_noise_auto: bool = True  # Auto-derive yaw noise from position noise: σ_yaw = arctan(σ_pos / (log_length/2))
    # Reward configuration
    # reward_formula:
    #   - "multiplicative": (efficiency or throughput) × alignment × stability
    #   - "additive": efficiency + alignment (+ stability)
    reward_formula: str = "multiplicative"
    # normalize_reward: if True, use efficiency = logs_grasped / available (clamped); otherwise use throughput = logs_grasped
    normalize_reward: bool = True
    max_graspable_logs: int = 15  # Physical grapple capacity cap for normalization denominator
    normalized_efficiency_scale: float = 10.0  # Used when reward_formula="multiplicative" and normalize_reward=True
    failure_penalty: float = -1.0  # Penalty for 0-log grasps
    # Pay reward only for grasps the sim actually despawns. `logs_grasped` is counted by
    # proximity + per-log rise BEFORE the lift-height gate, but _despawn_grasped_logs is gated
    # ON that lift. Cycles that pass the first and fail the second remove NOTHING from the rack
    # yet were still paid (12-18% of cycles, measured 2026-08-08). That is a reward-hacking
    # channel: PPO can raise return by finding poses that trip the counter without completing a
    # lift, which is why training reward rose while deployed argmax clearing fell. Reported
    # metrics deliberately keep the ungated count so the reported-vs-TRUE accounting stays
    # comparable with rows recorded before this fix. Set False to reproduce the old behaviour.
    reward_requires_lift: bool = True
    use_alignment_reward: bool = True  # Multiply/add alignment term (ablate by setting False)
    # use_stability_reward: penalize off-center grasps that cause grapple tilt
    # multiplicative: reward × stability, additive: + stability term
    use_stability_reward: bool = True
    # Stability measurement. Default: mean grapple tilt sampled every step of the lift hold,
    # excluding a fixed dead time after the lift (swing transient), then sharpened ONCE with
    # stability_exponent: sigma = (mean cos theta)^k. Average-then-sharpen keeps the score
    # comparable to the real node's timer-paced windowed TF sampling regardless of sample rate.
    # stability_snapshot=True restores the legacy single-sample-at-hold-end value.
    stability_snapshot: bool = args_cli.stability_snapshot
    stability_exponent: int = 4
    debug_reward: bool = False  # Print debug info for reward computation

    # Platform bed floor: executed grasp z is clamped to box_bottom + this margin, mirroring
    # the crane_policy_node envelope (bed_z + bed_margin, uniform across all policy types on
    # the real crane). Measured 2026-07-30: the real bed sits at ~-1.29 (no built-in clearance
    # in the box bottom). 0.0 (2026-07-31): margins above 0.066 leave bottom-layer log centers
    # (bed+0.056) unreachable for every controller, and the previous 0.10 jam-safety value did
    # exactly that on the real crane (every last-layer pick missed), so the margin is removed
    # in sim and real alike. Matches the node default. > 0 re-enables the clamp.
    platform_bed_margin: float = 0.0
    # Also clamp the EXPERT's executed targets to the floor (not just the RL decode). True so
    # the clamped endgame behavior is observable; NOTE this changes the data-generating process,
    # so before the next BC collection run the clamped-vs-unclamped expert A/B (endgame grasp
    # success) and decide deliberately. False = expert unclamped (check-print only).
    platform_floor_expert: bool = True

    # End-of-episode clearing bonus: reward = clearing_pct * scale at termination.
    # 0.0 = disabled (default, no change to existing tasks).
    # Per-cycle time cost. Every grasp cycle costs the same on the real crane whether it lifts
    # 1 log or 18, so a greedy highest-point picker wastes the endgame on single-log grabs. With
    # a cost per cycle the objective becomes logs-per-cycle rather than logs-per-grasp, which is
    # what a fixed heuristic cannot optimise: the policy has a reason to prefer dense bites, and
    # to clear in an order that keeps the remainder consolidated instead of scattered.
    # 0.0 = off (legacy). Try ~1.0 (about one log's worth of reward).
    cycle_cost: float = 0.0
    clearing_bonus_scale: float = 0.0
    # Proportional clearing bonus: True = bonus proportional to % cleared, False = binary (100% only)
    proportional_clearing_bonus: bool = False
    clearing_bonus_threshold: float = 1.0  # Min clearing fraction for binary bonus (1.0 = 100%, 0.95 = 95%)
    # Tiered clearing bonuses: list of thresholds, each awards clearing_bonus_scale when crossed.
    # e.g. [0.5, 0.7, 0.9] with scale=50 gives +50 at 50%, +50 at 70%, +50 at 90% (up to +150 total).
    # None = disabled, use single threshold instead.
    clearing_bonus_thresholds: list[float] | None = None

    # Curriculum: gradually increase active logs. None = disabled (default).
    # List of (episode_threshold, num_active_logs) tuples, e.g.:
    # [(0, 15), (200, 50), (500, 100), (1000, 200)]
    curriculum_schedule: list[tuple[int, int]] | None = None

    # Camera configuration (ZED X style RGBD camera)
    # GAZE ENV: the policy camera is MOUNTED ON THE BASEMAST (the env's `mast` link, which
    # rotates with the slew joint) at the model-based-calibrated T_mast<-zed_0 extrinsic,
    # so the sim camera matches the real basemast ZED instead of the old free-floating cam.
    enable_camera: bool = False  # Set True to enable camera sensor
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Crane/mast/BasemastCam",
        offset=TiledCameraCfg.OffsetCfg(
            # calibrated T_mast<-zed_0 (optical), 2026-07-27 floor-constrained calib
            # (calib_zed_0_20260727_140751_floor_mount.npy, promoted calib_zed_0_mount.npy).
            # (prev 2026-07-02: pos=(-0.2013,0.1126,1.0917) rot=(0.5776,-0.8094,-0.0657,0.0833))
            # (prev 2026-06-28: pos=(-0.3835,0.0114,1.2877) rot=(0.5516,-0.8286,-0.0683,0.0665))
            # The camera RENDERS from the live slewed mast, so this original offset gives the correct
            # view of the rack. (The earlier "yaw offset" in the PCD comparison is NOT a camera-aim
            # bug - it's the PCD UNPROJECTION using a stale/un-slewed camera pose; fix that in
            # get_pointcloud_world, NOT here. Do not pre-bake the slew here or the render double-slews.)
            pos=(-0.4484, -0.0567, 1.2790),
            rot=(0.5513, -0.8300, -0.0751, 0.0393),  # quat wxyz
            convention="ros"  # ZED optical frame is ROS optical (X right, Y down, Z forward)
        ),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.2,  # ZED X wide-angle focal length (mm)
            focus_distance=5.0,
            # aperture set so fx = focal/aperture*width = 2.2/5.656*1920 = 746.9 px, matching the
            # real ZED's calibrated intrinsics (fx=fy=746.9, HFOV 104.8 deg) at 1920x1080.
            horizontal_aperture=5.656,
            clipping_range=(0.3, 20.0)  # ZED X depth range
        ),
        width=1920,   # MATCH the real ZED depth resolution (1920x1080) for matching cloud density.
        height=1080,  # heavy with many envs; drop to 1280x720 if collection is too slow/OOM.
    )

    # Video recording configuration
    record_video: bool = False          # Enable in-loop frame capture for video export
    video_capture_interval: int = 4     # Capture one frame every N physics steps
    # World-space overview camera: isometric view positioned to see all envs at once.
    # For num_envs=4 (2x2 grid, env_spacing=12), envs span ~±6m.
    # Tune pos for more/fewer envs; increase distance for larger grids.
    overview_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/OverviewCamera",
        offset=TiledCameraCfg.OffsetCfg(
            # Original overview angle (produced bcrl_overview.mp4)
            pos=(-3.0, -5.0, 10.0),
            rot=(0.864, 0.257, -0.131, -0.413),
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=40.0,
            horizontal_aperture=14.0,
            clipping_range=(0.5, 150.0)
        ),
        width=1280,
        height=720,
    )
    # Side-view camera: eye-level view of grapple during transport for stability comparison
    sideview_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/SideviewCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -4.0, 5.0),   # Side of crane, eye-level with grapple during transport
            rot=(0.924, 0.383, 0.0, 0.0),  # Looking slightly up toward grapple area
            convention="opengl"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=8.0,
            focus_distance=6.0,
            horizontal_aperture=10.0,
            clipping_range=(0.5, 50.0)
        ),
        width=640,
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
    PHASE_TIMEOUT = args_cli.phase_timeout   # timesteps (~60Hz) before a phase is declared stuck (--phase_timeout)
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
    # GAZE ENV: new cycle-entry phase. The crane drives to a fixed "gaze" pose so the
    # basemast-mounted camera views the rack (only slew moves the cam; the arm is posed
    # to clear the frame; grapple yaw = 0 since no target is selected yet). The policy's
    # target selection happens here from the basemast PCD. Reset and SETTLE both enter GAZE.
    PH_GAZE            = 10

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
        PH_GAZE:           "GAZE",
    }

    # --- Gaze-phase pose (matches the real calibration gaze bag for sim2real) ---
    # From the 2026-06-26 17_20_44 rack-calibration gaze bag (same bag rack_calib.py used).
    # (prev, old gaze bag: slew 0.9913, boom 0.6667, stick -0.7592, telescope 0.2795)
    # Real gaze slew (basemast_to_mast) that aims the basemast cam at the calibrated rack.
    GAZE_SLEW_RAD      = 1.1205
    # arm clearing config (boom, stick, telescope) so the arm/grapple don't occlude the rack.
    GAZE_ARM_BOOM_RAD      = 0.7871
    GAZE_ARM_STICK_RAD     = -0.9198
    GAZE_ARM_TELESCOPE_M   = 0.1300
    GAZE_DWELL_N       = 20     # steps the crane must hold the gaze pose before capture/select
    GAZE_TIMEOUT       = 200    # safety timeout for reaching the gaze pose
    GAZE_JOINT_TOL     = 0.05   # rad: max per-joint error to consider the gaze pose "reached"

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

        # Per-env LOG SLOT stride (set BEFORE super().__init__, which calls _setup_scene ->
        # _rebuild_log_origins_world that uses it). = spawn template size = min(rows*layers, num_logs).
        # All per-env log indexing uses this, not a hardcoded 200 (old short-rack value).
        self._logs_per_env = int(min(cfg.rows * cfg.layers, cfg.num_logs))

        # This will invoke _setup_scene()
        super().__init__(cfg, render_mode, **kwargs)

        # ---- after scene exists ----
        self.crane: Articulation = self.scene.articulations["crane"]
        self._ctrl_joint_idx, _ = self.crane.find_joints(self.cfg.control_joint_names)
        # crane.usd ships a narrower envelope than the machine (slew +/- 100 deg,
        # telescope 0.13..1.18); widen physics to the real limits so IK and physics
        # agree and the rack ends stay reachable. See ARM_JOINT_LIMITS.
        _arm_limits = torch.tensor(ARM_JOINT_LIMITS, device=self.device, dtype=torch.float32)
        self.crane.write_joint_position_limit_to_sim(
            _arm_limits.unsqueeze(0).expand(self.num_envs, -1, -1),
            joint_ids=self._ctrl_joint_idx)
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

        # Curriculum state (thresholds are in learning iterations)
        self._total_episodes_completed = 0  # for logging
        self._curriculum_step_count = 0  # total step() calls
        self._curriculum_num_steps_per_env = 4  # steps per iteration (matches rsl_rl_cfg)
        if self.cfg.curriculum_schedule is not None and len(self.cfg.curriculum_schedule) > 0:
            self._curriculum_active_logs = self.cfg.curriculum_schedule[0][1]
        else:
            self._curriculum_active_logs = self.cfg.num_logs
        self._curriculum_stage_idx = 0  # index into curriculum_schedule

        # Heuristic per-env state
        self._phase = torch.full((self.num_envs,), self.PH_GAZE, dtype=torch.int64, device=self.device)  # GAZE: cycle entry
        self._has_log = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._yaw_targets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._pending_yaw_targets = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)  # Store policy yaw for ALIGN_YAW
        self._dwell = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # Hold timer for keeping logs lifted before evaluation (for visual feedback)
        self._lift_hold_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.LIFT_HOLD_DURATION = 60  # ~1 second at 60Hz to hold logs before despawn
        # Windowed stability: mirror the real MeasureStability settle detector (rolling window
        # variance gate + dwell), same defaults as measure_stability_action_node:
        # required_window_sec=1.0, required_dwell_sec=0.5, stability_variance_threshold_rad2=1e-4.
        # Steps are ~60Hz FSM ticks. On timeout, report best-effort mean like the real node.
        self.STABILITY_WINDOW_N = 60    # rolling tilt window (~1.0s)
        self.STABILITY_DWELL_N = 30     # window must stay below variance threshold this long (~0.5s)
        self.STABILITY_VAR_THRESH = 1.0e-4  # rad^2
        self.STABILITY_HOLD_MAX = 240   # settle timeout during lift hold (~4s), then best-effort
        self._stab_theta_win = [[] for _ in range(self.num_envs)]  # rolling tilt samples (rad)
        self._stab_dwell = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._stab_settle_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        # Raw per-grasp tilt progressions (full hold, untrimmed) so the metric can be
        # recomputed offline under a different definition without re-running evals.
        # Eval scripts dump this into their metrics JSON as "stability_records".
        self._stab_theta_full = [[] for _ in range(self.num_envs)]
        self._stab_records = [[] for _ in range(self.num_envs)]

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
        # remaining physics steps in the live CLOSE window (real: grapple_delta_time_ms)
        self.MAX_LEAD = float(getattr(args_cli, "gripper_max_lead", 0.35))
        self.CLOSE_DURATION_STEPS = int(float(getattr(args_cli, "gripper_close_duration_s", 8.0)) / physics_dt)
        self._gripper_close_steps_left = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
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
        # Reward-vs-truth audit (see _audit_reward). Path is per-process so concurrent runs on
        # the multi-GPU boxes do not interleave into one file.
        self._reward_audit = []
        self._reward_audit_path = os.environ.get(
            "REWARD_AUDIT_PATH",
            f"/workspace/crane_testbed/logs/reward_audit/audit_{os.getpid()}.jsonl")
        self._prev_grasp_alignment = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Episode-level metrics for TensorBoard (reset per episode)
        self._episode_successful_grasps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_failed_grasps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._episode_total_logs_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._episode_alignment_sum_weighted = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Sum of (logs * alignment)
        self._episode_stability_sum_weighted = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Sum of (logs * stability)
        self._prev_grasp_stability = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Last grasp stability
        self._last_clearing_bonus = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)  # Last clearing bonus given
        self._completed_ep_returns = collections.deque(maxlen=100)  # Rolling buffer of completed-episode returns
        # True clearing fraction (1 - remaining/starting) of the most recently completed episode,
        # per env; -1 until that env finishes one. Recorded at termination before auto-reset so BC
        # data collection can report the env's authoritative clearing instead of summing per-cycle
        # grasp counts (which can exceed 100% via the proximity-radius grasp check).
        self._last_episode_clearing = torch.full((self.num_envs,), -1.0, device=self.device, dtype=torch.float32)

        # Logs available at target position (for normalized reward computation)
        self._logs_available_at_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Cylinder visualization position (world frame) for debug
        self._debug_cylinder_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._debug_cylinder_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Track logs knocked out of bounds (for penalty)
        self._logs_knocked_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._logs_out_of_bounds_penalty = 0.0  # Tracked for metrics but not penalized
        self._initial_settling_complete = False  # Track if initial settling has finished

        # Per-cycle knocked-off tracking via remaining-count delta
        # (more accurate than _logs_knocked_off which only catches OOB despawns)
        self._prev_logs_remaining = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._prev_cycle_knocked_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # AUTHORITATIVE post-cycle rack count, sampled AFTER the grasp despawn. Everything else
        # downstream (console prints, decisions.npz, pile_cleared_pct, cycles_to_95pct) is DERIVED
        # arithmetic - prev - grasped - knocked, with knocked itself a max(0, ...) clamp that
        # silently absorbs any disagreement. That derived value has been observed going NEGATIVE
        # and jumping UP mid-episode, neither of which a set-difference count can do, so it is not
        # trustworthy. This field is measured, not inferred; _accounting_mismatch counts the cycles
        # where the two disagree so the discrepancy is visible instead of averaged away.
        self._post_cycle_logs_in_rack = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._accounting_mismatch = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Snapshot of _logs_knocked_off at episode end (survives _reset_idx)
        self._final_episode_knocked_off = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # Snapshot of _cycle_count at episode end (survives _reset_idx)
        self._final_episode_cycle_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

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
        # Curriculum: advance stage based on learning iterations
        if self.cfg.curriculum_schedule is not None and self._logs_settled:
            self._curriculum_step_count += 1
            current_iter = self._curriculum_step_count // self._curriculum_num_steps_per_env
            schedule = self.cfg.curriculum_schedule
            while (self._curriculum_stage_idx < len(schedule) - 1
                   and current_iter >= schedule[self._curriculum_stage_idx + 1][0]):
                self._curriculum_stage_idx += 1
                new_active = schedule[self._curriculum_stage_idx][1]
                if new_active != self._curriculum_active_logs:
                    self._curriculum_active_logs = new_active
                    print(f"[Curriculum] Stage {self._curriculum_stage_idx}: "
                          f"{new_active} active logs at iteration {current_iter}")

        # No-deposition version always uses hierarchical mode
        # 1. Process action (target selection) at HOVER_UP
        action = action.to(self.device)

        # Snapshot remaining count before the cycle for knocked-off calculation
        for i in range(self.num_envs):
            self._prev_logs_remaining[i] = self._count_logs_in_rack(i)

        # Suppress debug prints during internal loop to avoid terminal spam (set BEFORE _pre_physics_step)
        self._in_hierarchical_loop = True

        self._pre_physics_step(action)

        # 2. Run physics loop until all envs complete their grasp attempt
        # Each phase has PHASE_TIMEOUT, so cycles naturally complete or timeout
        # _cycle_complete_this_step prevents envs from starting new cycles within this step()
        MAX_STEPS = 1500  # Safety limit only - should never hit this

        self._cycle_complete_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        _record = getattr(self.cfg, 'record_video', False)
        _capture_interval = getattr(self.cfg, 'video_capture_interval', 4)

        for step_count in range(MAX_STEPS):
            # Step physics simulation
            # Only render during loop if GUI is open (for visual debugging) or recording video
            should_render = (self.sim.has_gui() or _record) and (step_count % _capture_interval == 0)
            self.sim.step(render=should_render)

            # Update buffers
            self.scene.update(self.physics_dt)

            # Apply heuristic actions (IK controller)
            self._apply_action()

            # Update heuristic state machine (handles grasp evaluation and transition to HOVER_UP)
            self._heuristic_step()

            # Capture video frames — stream directly to disk via imageio writers
            if _record and should_render:
                # Policy view: tile all per-env camera frames into a grid
                if self._camera is not None:
                    self._camera.update(self.physics_dt)
                    rgb = self._camera.data.output.get("rgb")
                    if rgb is not None and rgb.shape[0] > 0:
                        frames_np = rgb[:, :, :, :3].cpu().numpy()  # (N, H, W, 3)
                        n = frames_np.shape[0]
                        if n == 1:
                            tiled = frames_np[0]
                        else:
                            cols = math.ceil(math.sqrt(n))
                            rows_grid = math.ceil(n / cols)
                            fH, fW = frames_np.shape[1], frames_np.shape[2]
                            tiled = np.zeros((rows_grid * fH, cols * fW, 3), dtype=frames_np.dtype)
                            for i, f in enumerate(frames_np):
                                r, c = i // cols, i % cols
                                tiled[r*fH:(r+1)*fH, c*fW:(c+1)*fW] = f
                        if self._video_writer is not None:
                            self._video_writer.append_data(tiled)
                            self._video_frame_count += 1
                # Overview: single world-space camera frame
                if self._overview_camera is not None:
                    self._overview_camera.update(self.physics_dt)
                    rgb_ov = self._overview_camera.data.output.get("rgb")
                    if rgb_ov is not None and rgb_ov.shape[0] > 0:
                        frame_ov = rgb_ov[0, :, :, :3].cpu().numpy()
                        if self._overview_writer is not None:
                            self._overview_writer.append_data(frame_ov)
                            self._overview_frame_count += 1
                # Sideview: eye-level camera for stability comparison
                if self._sideview_camera is not None:
                    self._sideview_camera.update(self.physics_dt)
                    rgb_sv = self._sideview_camera.data.output.get("rgb")
                    if rgb_sv is not None and rgb_sv.shape[0] > 0:
                        frame_sv = rgb_sv[0, :, :, :3].cpu().numpy()
                        if self._sideview_writer is not None:
                            self._sideview_writer.append_data(frame_sv)
                            self._sideview_frame_count += 1

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

        # Compute per-cycle knocked-off from remaining-count delta.
        # This captures ALL lost logs (OOB despawns + rolled off + any other
        # losses) and is more accurate than the OOB-only check inside
        # _heuristic_step.  Accumulate into _logs_knocked_off for episode totals.
        for i in range(self.num_envs):
            remaining_now = self._count_logs_in_rack(i)
            grasped = int(self._prev_logs_grasped[i].item())
            expected = int(self._prev_logs_remaining[i].item()) - grasped
            delta_knocked = max(0, expected - remaining_now)
            self._prev_cycle_knocked_off[i] = delta_knocked
            self._logs_knocked_off[i] += delta_knocked

            # Measured truth + a visible check against the derived value. Deliberately a WARNING,
            # not an assert: this runs inside long unattended evals and a raise would throw away
            # the whole row. derived == remaining_now whenever expected >= remaining_now, so any
            # mismatch means grasped/knocked over-accounts for what actually left the rack.
            self._post_cycle_logs_in_rack[i] = remaining_now
            derived = expected - delta_knocked
            if derived != remaining_now:
                self._accounting_mismatch[i] += 1
                if int(self._accounting_mismatch[i].item()) <= 5:
                    print(f"[ACCOUNTING] env{i} cycle {int(self._cycle_count[i].item())}: "
                          f"derived remaining {derived} != measured {remaining_now} "
                          f"(prev={int(self._prev_logs_remaining[i].item())} "
                          f"grasped={grasped} knocked={delta_knocked})")

        # Cycle boundary for the video overlay: the accounting above is now final for this
        # cycle, so every frame captured since the previous bound belongs to it.
        if _record:
            self._video_cycle_bounds.append(int(self._overview_frame_count))

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
            elif self._cycle_count[i] >= args_cli.max_grasp_cycles:  # Timeout (see --max_grasp_cycles)
                terminated[i] = True

        # End-of-episode clearing bonus
        clearing_bonus_scale = getattr(self.cfg, 'clearing_bonus_scale', 0.0)
        if clearing_bonus_scale > 0.0:
            proportional = getattr(self.cfg, 'proportional_clearing_bonus', False)
            for i in range(self.num_envs):
                if terminated[i]:
                    if proportional:
                        starting_logs = int(self._per_env_log_counts[i])
                        remaining = self._count_logs_in_rack(i)
                        clearing_pct = 1.0 - (remaining / max(starting_logs, 1))
                        bonus = clearing_pct * clearing_bonus_scale
                    else:
                        starting_logs = int(self._per_env_log_counts[i])
                        remaining = self._count_logs_in_rack(i)
                        clearing_pct = 1.0 - (remaining / max(starting_logs, 1))
                        tiered = getattr(self.cfg, 'clearing_bonus_thresholds', None)
                        if tiered is not None:
                            # Tiered: +scale for each threshold crossed
                            bonus = sum(clearing_bonus_scale for t in tiered if clearing_pct >= t)
                        else:
                            # Single threshold
                            threshold = getattr(self.cfg, 'clearing_bonus_threshold', 1.0)
                            bonus = clearing_bonus_scale if clearing_pct >= threshold else 0.0
                    self.reward_buf[i] += bonus
                    self._episode_return[i] += bonus
                    self._last_clearing_bonus[i] = bonus

        truncated = torch.zeros_like(terminated)

        # Reset terminated envs
        reset_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            # Cache completed-episode returns before reset zeroes them
            self._completed_ep_returns.extend(self._episode_return[reset_ids].tolist())
            # Record true clearing (1 - remaining/starting) BEFORE reset repopulates the rack.
            for i in reset_ids.tolist():
                starting = max(int(self._per_env_log_counts[i]), 1)
                remaining = self._count_logs_in_rack(i)
                self._last_episode_clearing[i] = min(1.0, max(0.0, 1.0 - remaining / starting))
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

        # ground + light
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
        if getattr(self.cfg, 'enable_camera', False) or getattr(self.cfg, 'record_video', False):
            self._camera = TiledCamera(self.cfg.camera_cfg)
            self.scene.sensors["camera"] = self._camera
            print(f"[INFO]: Camera enabled - {self.cfg.camera_cfg.width}x{self.cfg.camera_cfg.height}, data_types={self.cfg.camera_cfg.data_types}")

        # Overview/sideview cameras and streaming video writers (for video recording)
        self._overview_camera = None
        self._sideview_camera = None
        self._video_writer = None       # imageio writer for policy-view
        self._overview_writer = None    # imageio writer for overview
        self._sideview_writer = None    # imageio writer for sideview
        self._video_frame_count = 0
        self._overview_frame_count = 0
        self._sideview_frame_count = 0
        # Frame index at the END of each grasp cycle, so a post-hoc overlay can map any frame
        # back to the cycle it belongs to. Cycles have variable length (the FSM runs until the
        # grasp completes or times out), so this cannot be derived from fps alone.
        self._video_cycle_bounds = []
        if getattr(self.cfg, 'record_video', False):
            self._overview_camera = TiledCamera(self.cfg.overview_camera_cfg)
            self._sideview_camera = TiledCamera(self.cfg.sideview_camera_cfg)
            # NOTE: do NOT add to scene.sensors — world-space camera has 1 instance
            # but scene.reset() would pass multi-env ids causing index OOB.
            # We update it manually in the inner loop instead.
            print(f"[INFO]: Overview camera enabled for video recording - "
                  f"{self.cfg.overview_camera_cfg.width}x{self.cfg.overview_camera_cfg.height}")
            print(f"[INFO]: Sideview camera enabled for video recording - "
                  f"{self.cfg.sideview_camera_cfg.width}x{self.cfg.sideview_camera_cfg.height}")

        # done
        self._bootstrap_done = True
        print(f"[INFO]: Env ready. Running heuristic baseline.")
    
    def _normalize_yaw_for_symmetry(self, yaw: float) -> float:
        """Normalize yaw to canonical form accounting for log 180° rotational symmetry.

        Since logs are symmetric, yaw and yaw+π represent the same orientation.
        Normalize to range [-π/2, π/2] for consistency.

        Args:
            yaw: Input yaw in radians

        Returns:
            Normalized yaw in [-π/2, π/2]
        """
        import math

        # Wrap to [-π, π]
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi

        # Map to [-π/2, π/2] using 180° symmetry
        if yaw > math.pi / 2:
            yaw -= math.pi
        elif yaw < -math.pi / 2:
            yaw += math.pi

        return yaw

    def _get_target_grapple_yaw_b(self, env_i: int) -> float:
        """Get target basegrapple yaw orientation in basemast frame (state-independent).

        Returns the desired basegrapple yaw in basemast frame. For parallel grasps,
        this equals the log's yaw (basegrapple should align with log).
        Policy predicts this value, then it's converted to a yaw joint position
        at runtime based on current arm configuration.

        Note: Yaw is normalized to [-π/2, π/2] to account for log 180° symmetry.
        """
        import math

        def quat_conj_wxyz(q):
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
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # Get base orientation (world->base as conjugate)
        base_quat_w = self.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)

        # Get log quaternion (prefer stored target quat if valid)
        q_log_w = None
        try:
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
            # Fallback: Find closest log to current target position
            # This is useful during BC play to get ground truth for the location being targeted
            if hasattr(self, '_target_log_pos_b') and hasattr(self, '_logs_obj') and self._logs_obj is not None:
                try:
                    target_pos_b = self._target_log_pos_b[env_i]
                    root_w = self.crane.data.root_pose_w[env_i]
                    target_w = root_w[:3] + target_pos_b

                    # Find closest log
                    per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else self._logs_per_env
                    start = env_i * self._logs_per_env
                    end = start + per_env

                    pos_w, quat_w = self._get_logs_root_pose_w()
                    if pos_w is not None and quat_w is not None and end > start:
                        dists = torch.norm(pos_w[start:end] - target_w.unsqueeze(0), dim=-1)
                        closest_idx = start + torch.argmin(dists).item()
                        q_log_w = (float(quat_w[closest_idx][0]), float(quat_w[closest_idx][1]),
                                  float(quat_w[closest_idx][2]), float(quat_w[closest_idx][3]))
                except Exception:
                    pass

            if q_log_w is None:
                return 0.0  # No log info available

        # Transform log orientation to base frame
        # For parallel grasp: target basegrapple yaw = log yaw (align Y-axes)
        q_log_b = quat_mul_wxyz(q_base_inv, q_log_w)
        target_grapple_yaw_b = yaw_from_quat_wxyz(*q_log_b)

        # Normalize to canonical range [-π/2, π/2] for consistency
        target_grapple_yaw_b = self._normalize_yaw_for_symmetry(target_grapple_yaw_b)

        return target_grapple_yaw_b

    def _convert_grapple_yaw_to_joint_position(self, env_i: int, target_grapple_yaw_b: float) -> float:
        """Convert target basegrapple yaw in basemast frame to yaw joint position.

        Given a desired basegrapple orientation in basemast frame (state-independent),
        calculate the yaw joint position needed to achieve it based on
        current arm configuration.

        Args:
            env_i: Environment index
            target_grapple_yaw_b: Desired basegrapple yaw in basemast frame (radians)

        Returns:
            Yaw joint position (radians in [-π, π])
        """
        import math

        def wrap(a: float) -> float:
            return (a + math.pi) % (2.0 * math.pi) - math.pi

        def quat_conj_wxyz(q):
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
            return math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))

        # Get current yaw joint value
        yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
        if len(yaw_joint_ids) == 0:
            return 0.0
        yaw_joint_id = int(yaw_joint_ids[0])
        current_yaw_joint = float(self.crane.data.joint_pos[env_i, yaw_joint_id].item())

        # Get base orientation
        base_quat_w = self.crane.data.root_pose_w[env_i, 3:7]
        q_base_w = (float(base_quat_w[0]), float(base_quat_w[1]),
                    float(base_quat_w[2]), float(base_quat_w[3]))
        q_base_inv = quat_conj_wxyz(q_base_w)

        # Get current basegrapple orientation in base frame
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]
        q_bg_b = quat_mul_wxyz(q_base_inv, (float(bg_quat_w[0]), float(bg_quat_w[1]),
                                           float(bg_quat_w[2]), float(bg_quat_w[3])))
        current_bg_z_rotation = yaw_from_quat_wxyz(*q_bg_b)

        # Calculate rotation difference needed to reach target orientation
        rotation_diff = wrap(target_grapple_yaw_b - current_bg_z_rotation)

        # Apply to yaw joint with 180° flip option
        target_yaw1 = current_yaw_joint + rotation_diff
        target_yaw2 = current_yaw_joint + rotation_diff + math.pi

        # Choose minimal rotation
        yaw1 = wrap(target_yaw1)
        yaw2 = wrap(target_yaw2)
        diff1 = abs(wrap(yaw1 - current_yaw_joint))
        diff2 = abs(wrap(yaw2 - current_yaw_joint))

        return yaw1 if diff1 <= diff2 else yaw2

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
            per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else self._logs_per_env
            start = env_i * self._logs_per_env
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
        # Separate, WIDER despawn bounds = the full rack footprint (not the tighter action box),
        # so settled edge logs inside the rack are kept; only logs knocked OFF the rack despawn.
        if not hasattr(self, "_rack_bounds_min"):
            self._rack_bounds_min = torch.zeros((N, 3), device=self.device)
            self._rack_bounds_max = torch.zeros((N, 3), device=self.device)

        # GAZE: the rack sits at a FIXED calibrated pose in the crane base frame, so the action
        # bounds are a constant base-frame box around the rack footprint (no per-env world transform).
        # Rack center (base) = (-4.364, 1.816); footprint 2.16 (X/width) x 7.38 (Y/length);
        # bottom at base z ~ -1.42 (ground). Generous margins so settled/edge logs aren't culled.
        # Action bounds = INSIDE the rack (where the logs sit), a little shorter than the rack in
        # length and height — NOT the rack outer frame + margin. Rack half-dims are 1.08 (W) x
        # 3.69 (L), floor at base z ~ -1.42, top ~ +1.12.
        RACK_CX, RACK_CY = RACK_BASE_X, RACK_BASE_Y
        HALF_X = 1.00          # just inside the rack width (rack half = 1.08)
        HALF_Y = 3.50          # a little shorter than the rack length (rack half = 3.69)
        Z_MIN  = -1.30         # rack floor / lowest logs
        Z_MAX  =  0.10         # just above the settled pile (shorter than the rack height)
        _min_b = torch.tensor([RACK_CX - HALF_X, RACK_CY - HALF_Y, Z_MIN], device=self.device, dtype=torch.float32)
        _max_b = torch.tensor([RACK_CX + HALF_X, RACK_CY + HALF_Y, Z_MAX], device=self.device, dtype=torch.float32)
        # Despawn box = full rack footprint half-dims (1.08 W x 3.69 L). Floor aligned to the
        # GROUND (base z ~ -1.42), not the tighter action Z_MIN, so logs settled on the rack
        # floor are kept; only logs that fall THROUGH the ground despawn.
        RACK_HALF_X, RACK_HALF_Y = 1.08, 3.69
        RACK_Z_MIN = -1.42
        _rmin_b = torch.tensor([RACK_CX - RACK_HALF_X, RACK_CY - RACK_HALF_Y, RACK_Z_MIN], device=self.device, dtype=torch.float32)
        _rmax_b = torch.tensor([RACK_CX + RACK_HALF_X, RACK_CY + RACK_HALF_Y, Z_MAX], device=self.device, dtype=torch.float32)
        for env_i in range(N):
            self._action_bounds_min[env_i] = _min_b
            self._action_bounds_max[env_i] = _max_b
            self._rack_bounds_min[env_i] = _rmin_b
            self._rack_bounds_max[env_i] = _rmax_b
            self._action_bounds_valid[env_i] = True
        return

        # --- (legacy world-anchored computation below, unused in gaze env) ---
        # --- Fixed rack extents in WORLD ---
        # Anchored to the physical rack geometry (matches the original 20-row x 10-layer
        # grid: 0.5*(20-1)*0.16 = 1.52 m half-span, top at base_z + 9*0.12 + 0.12 = 1.30 m).
        # These are independent of the log spawn pattern so that action bounds stay
        # constant regardless of which pattern (A/B/C) is used.
        RACK_Y_HALF_SPAN = 1.52   # metres, fixed to rack Y extent
        RACK_Z_TOP       = 0.70   # metres above world origin (base_z + 0.60 m stack height)

        rack_center_y_w = cfg.center_y_world
        y_min_w = rack_center_y_w - RACK_Y_HALF_SPAN
        y_max_w = rack_center_y_w + RACK_Y_HALF_SPAN
        rack_z_span = RACK_Z_TOP - cfg.base_z  # height above rack base

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


    def _set_movement_speed(self, phase: int, env_i: int, current_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """Apply command scaling for controlled movement phases."""
        # Define step sizes for different phases
        if phase in [self.PH_DESCEND, self.PH_LOWER_TO_DROP]:
            step_size = 0.15  # Slow descent to avoid disturbing pile
        elif phase == self.PH_LIFT_HIGH:
            step_size = float(getattr(args_cli, "lift_step", 0.08))  # slow enough to close under load
        elif phase == self.PH_CARRY_HOME:
            step_size = 0.5  # Faster carry
        else:
            step_size = 1.0  # Full speed for positioning phases

        # Interpolate toward target
        return current_pos + (target_pos - current_pos) * step_size

    def _init_markers(self):
        # Remove stale debug marker prims from previous runs (prevents leftover boxes at origin).
        try:
            from pxr import Sdf
            stage = omni.usd.get_context().get_stage()
            debug_root = "/Visuals/CraneDebug"
            if prim_utils.get_prim_at_path(debug_root):
                try:
                    stage.RemovePrim(Sdf.Path(debug_root))
                except Exception:
                    stage.RemovePrim(debug_root)
        except Exception:
            pass
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
            prim_path="/Visuals/CraneDebug/ACTION_BOUNDS_FRAME",
            markers={
                # "Frame cube" implemented as 12 thin cuboids (edges) rendered via batched visualize() calls.
                "edge": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),  # Scaled per-edge
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        opacity=1.0,
                    ),
                ),
            },
        )
        viz["action_bounds"] = VisualizationMarkers(bbox_cfg)

        # Trailer deposit scan bounds (orange wireframe box). Shown only with --show_trailer_bounds.
        trailer_bbox_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/TRAILER_BOUNDS_FRAME",
            markers={
                "edge": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),  # Scaled per-edge (wireframe)
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.5, 0.0),  # Orange
                        opacity=1.0,
                    ),
                ),
            },
        )
        viz["trailer_bounds"] = VisualizationMarkers(trailer_bbox_cfg)

        # Grasp count prism (shows rack slice volume used for counting available logs)
        # Only visible when debug_reward is enabled
        prism_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CraneDebug/GRASP_PRISM",
            markers={
            "edge": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),  # Scaled per-edge (wireframe)
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.5, 1.0),  # Blue
                        opacity=1.0,
                    ),
                ),
            },
        )
        viz["grasp_prism"] = VisualizationMarkers(prism_cfg)

        # Hide grasp prism by default so it doesn't linger at the origin before first update.
        try:
            N = int(self.num_envs)
            hide_pos = torch.full((N * 12, 3), -1000.0, device=self.device, dtype=torch.float32)
            hide_quat = torch.zeros((N * 12, 4), device=self.device, dtype=torch.float32)
            hide_quat[:, 0] = 1.0
            hide_scale = torch.full((N * 12, 3), 1e-3, device=self.device, dtype=torch.float32)
            viz["grasp_prism"].visualize(translations=hide_pos, orientations=hide_quat, scales=hide_scale)
        except Exception:
            pass

        # Hide action_bounds by default so it doesn't linger at the origin before first update.
        try:
            N = int(self.num_envs)
            hide_pos = torch.full((N * 12, 3), -1000.0, device=self.device, dtype=torch.float32)
            hide_quat = torch.zeros((N * 12, 4), device=self.device, dtype=torch.float32)
            hide_quat[:, 0] = 1.0
            hide_scale = torch.full((N * 12, 3), 1e-3, device=self.device, dtype=torch.float32)
            viz["action_bounds"].visualize(translations=hide_pos, orientations=hide_quat, scales=hide_scale)
        except Exception:
            pass

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
        # Optional high-friction log material (see --log_friction): default (~0.5) lets
        # mounds relax flat during settling; bark-like friction preserves height profiles.
        _log_mat = None
        _fr = float(getattr(args_cli, "log_friction", 0.0) or 0.0)
        if _fr > 0.0:
            _log_mat = sim_utils.RigidBodyMaterialCfg(
                static_friction=_fr, dynamic_friction=0.9 * _fr)
            print(f"[LOGS] high-friction material: static={_fr}, dynamic={0.9*_fr:.2f}")

        def _log_usd_cfg(scale=None, mass=None):
            return sim_utils.UsdFileCfg(
                usd_path=self.cfg.log_usd,
                **({"scale": scale} if scale is not None else {}),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    linear_damping=0.0,
                    # rolling resistance stand-in (bark/taper); see --log_ang_damping
                    angular_damping=float(getattr(args_cli, "log_ang_damping", 0.05)),
                    max_depenetration_velocity=3.0,
                ),
                # Prefer DENSITY: PhysX computes mass from the actual mesh volume, so size variants
                # and the taper are exact. Falls back to an explicit mass (volume-scaled per variant)
                # when --log_density is 0.
                mass_props=(sim_utils.MassPropertiesCfg(density=self.cfg.log_density)
                            if self.cfg.log_density > 0.0 else
                            sim_utils.MassPropertiesCfg(mass=mass if mass is not None else self.cfg.log_mass)),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.02,  # Larger offset = earlier detection (2cm safety margin)
                    rest_offset=0.0,
                ),
                # Semantic label for camera segmentation
                semantic_tags=[("class", "log")],
            )

        # Per-log size variety (see --log_scale_jitter): each log instance randomly draws one
        # of several diameter variants, scaled UP ONLY from nominal d=0.113 (never below, so the
        # nominal stays the thin end). The hex lattice is tuned for the nominal diameter, so spacing is
        # inflated by (1+jit) at arg-parse time (and rows shrunk to keep the footprint) to give
        # the fattest variant room; without that, up-scaled logs interpenetrate at spawn. Mixed sizes
        # interlock and break up perfect-cylinder packing -> less rolling, more realistic piles.
        _jit = float(getattr(args_cli, "log_scale_jitter", 0.0) or 0.0)
        _mean = float(getattr(args_cli, "log_scale_mean", 1.0) or 1.0)
        _ecc = float(getattr(args_cli, "log_ellipticity", 0.0) or 0.0)
        print(f"[LOGS] asset: {self.cfg.log_usd}")
        if _jit > 0.0 or _mean != 1.0 or _ecc > 0.0:
            variants = []
            for i in range(6):
                r = _mean * (1.0 + _jit * (2.0 * i / 5.0 - 1.0))   # diameter: mean*(1 +- jit)
                l = _mean * (1.0 + _jit * ((i % 3) - 1.0))         # length:   mean*(1 +- jit)
                # out-of-round cross-section: transverse axes differ by +-e (e cycles across
                # variants), so the pile cannot settle into a perfect hex lattice
                e = _ecc * ((i % 3) / 2.0)
                sx, sz = r * (1.0 + e), r * (1.0 - e)
                m = self.cfg.log_mass * sx * sz * l          # volume-proportional mass
                variants.append(_log_usd_cfg(scale=(sx, l, sz), mass=m))  # length is local Y
            spawn_cfg = sim_utils.MultiAssetSpawnerCfg(assets_cfg=variants, random_choice=True)
            if self.cfg.log_density > 0.0:
                _vol = 3.14159 * (0.113 / 2) ** 2 * 2.45          # nominal log volume [m^3]
                _m0 = self.cfg.log_density * _vol * (_mean * (1 - _jit)) ** 3
                _m1 = self.cfg.log_density * _vol * (_mean * (1 + _jit)) ** 3
                print(f"[LOGS] mass from DENSITY {self.cfg.log_density:.0f} kg/m3 -> "
                      f"~{_m0:.1f}..{_m1:.1f} kg (exact per mesh volume)")
            else:
                _m0 = self.cfg.log_mass * (_mean * (1 - _jit)) ** 3
                _m1 = self.cfg.log_mass * (_mean * (1 + _jit)) ** 3
                print(f"[LOGS] mass follows volume: {_m0:.1f}..{_m1:.1f} kg (nominal {self.cfg.log_mass:.1f} kg)")
            print(f"[LOGS] size: nominal x{_mean:.2f} +- {100.0 * _jit:.0f}% -> diameter "
                  f"{0.113 * _mean * (1 - _jit):.3f}..{0.113 * _mean * (1 + _jit):.3f} m "
                  f"(6 variants, length likewise)"
                  + (f", ellipticity up to +-{100.0 * _ecc:.0f}% (out-of-round)" if _ecc > 0 else ""))
        else:
            spawn_cfg = _log_usd_cfg()

        ro_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Rack/LogsAnchor/Origin.*/Log",
            spawn=spawn_cfg,
        )
        
        # Create RigidObject and add to scene
        logs_obj = RigidObject(cfg=ro_cfg)
        self.scene._rigid_objects["logs"] = logs_obj
        self._logs_obj = logs_obj

        # High-friction logs (--log_friction): UsdFileCfg has no physics_material field in this
        # Isaac Lab version, so author the material once and bind it to the spawned log prims.
        # Friction is what makes an oversized bite INCOMPRESSIBLE (logs wedge against neighbours
        # instead of sliding apart), which is the mechanism behind real choking.
        if _log_mat is not None:
            try:
                _mat_path = "/World/Materials/LogMaterial"
                _log_mat.func(_mat_path, _log_mat)
                sim_utils.bind_physics_material(
                    "/World/envs/env_.*/Rack/LogsAnchor/Origin.*/Log", _mat_path)
                print(f"[LOGS] bound high-friction material at {_mat_path}")
            except Exception as e:  # noqa: BLE001
                print(f"[LOGS] WARN could not bind friction material: {e}")

        # Ground-truth verification of the multi-asset variants: read back the ACTUAL spawned
        # scales from the stage (a 2 cm diameter spread is hard to judge by eye in the viewport;
        # and with replicate_physics=True the variants silently never took effect).
        if _jit > 0.0:
            try:
                import isaacsim.core.utils.prims as _pu
                import omni.usd as _ou
                from pxr import UsdGeom as _ug
                from collections import Counter as _Ct
                _stage = _ou.get_context().get_stage()
                _paths = _pu.find_matching_prim_paths(
                    "/World/envs/env_.*/Rack/LogsAnchor/Origin.*/Log")
                _sc = []
                for _p in _paths[:400]:
                    for _op in _ug.Xformable(_stage.GetPrimAtPath(_p)).GetOrderedXformOps():
                        if _op.GetOpType() == _ug.XformOp.TypeScale:
                            _sc.append(tuple(round(float(v), 3) for v in _op.Get()))
                _c = _Ct(_sc)
                print(f"[LOGS] VERIFY: {len(_c)} distinct scale variants across {len(_sc)} "
                      f"sampled logs (expect ~6): {sorted(_c.keys())[:8]}")
                if len(_c) <= 1:
                    print("[LOGS] VERIFY: WARNING - all logs identical; variants NOT applied!")
            except Exception as _e:
                print(f"[LOGS] VERIFY skipped: {_e}")

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

            # Log count: curriculum > DR > default
            if self.cfg.curriculum_schedule is not None:
                fallback = self.cfg.curriculum_schedule[0][1] if self.cfg.curriculum_schedule else self.cfg.num_logs
                per_env_target = int(min(per_env_cap, getattr(self, '_curriculum_active_logs', fallback)))
            elif self.cfg.enable_domain_randomization:
                import random
                per_env_target = random.randint(20, self._logs_per_env)
            else:
                per_env_target = default_per_env_target
            self._per_env_log_counts[env_id] = per_env_target

            # Select pattern function
            if getattr(args_cli, "profile_piles", False):
                # Randomized height profiles on a stable hex lattice (BC y-localization data).
                pattern_func = plan_grid_yz_hex_profile
            elif randomize_patterns:
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
            # profile_piles needs a fresh seed per RESET like DR (else every episode
            # regenerates the same profile).
            if randomize_patterns or getattr(args_cli, "profile_piles", False):
                if self.cfg.seed is not None and self.cfg.seed > 0:
                    # Deterministic seed for reproducible evaluation
                    # getattr: _rebuild_log_origins_world runs once from _setup_scene, BEFORE
                    # __init__ reaches the line that creates _total_episodes_completed (only the
                    # seed>0 deterministic branch touches it, so seed-0 runs never hit this).
                    pattern_seed = self.cfg.seed + getattr(self, "_total_episodes_completed", 0) * self.num_envs + env_id
                else:
                    import time
                    # Time-based seed ensures different pattern on every reset (training)
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
                row_y_jitter=self.cfg.row_y_jitter,
                layer_y_offset=self.cfg.layer_y_offset,
                seed=pattern_seed,
            )
            # HARDENING: a planner shortfall must not kill a long collection run
            # (one overnight run died on `yz_local_env[i]` IndexError here). Pad by
            # jittered duplicates of existing positions and WARN with the culprit.
            if len(yz_local_env) < per_env_target:
                print(f"[WARN] pattern '{getattr(pattern_func, '__name__', '?')}' returned "
                      f"{len(yz_local_env)} < target {per_env_target} (env {env_id}, "
                      f"seed {pattern_seed}); padding to target")
                import random as _rnd
                base = list(yz_local_env) if len(yz_local_env) > 0 else [(0.0, self.cfg.base_z)]
                while len(base) < per_env_target:
                    y0, z0 = base[_rnd.randrange(len(base))]
                    base.append((y0 + _rnd.uniform(-0.06, 0.06), z0 + _rnd.uniform(0.0, 0.06)))
                yz_local_env = base


            # Add logs for this environment
            # Always allocate 200 slots per env, but only populate per_env_target
            max_slots_per_env = self._logs_per_env

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

        # --- GAZE drive: hold the fixed gaze pose via direct joint position targets (we have the
        #     exact gaze joint config, so no hand-tuning needed). Overrides IK for PH_GAZE envs.
        #     Only slew aims the basemast cam; boom/stick/telescope keep the arm out of frame;
        #     grapple yaw = 0 (no target picked yet).
        #     NOTE (verify in-sim): if the SimpleIK velocity actuators won't hold a position target,
        #     switch this to a velocity-toward-target command for gaze envs.
        gaze_mask = (self._phase == self.PH_GAZE)
        if gaze_mask.any():
            gaze_ids = gaze_mask.nonzero(as_tuple=False).squeeze(-1)
            gaze_q = torch.tensor(
                [self.GAZE_SLEW_RAD, self.GAZE_ARM_BOOM_RAD,
                 self.GAZE_ARM_STICK_RAD, self.GAZE_ARM_TELESCOPE_M],
                device=self.device, dtype=torch.float32,
            ).unsqueeze(0).expand(gaze_ids.shape[0], -1)
            self.crane.set_joint_position_target(gaze_q, joint_ids=self._ctrl_joint_idx, env_ids=gaze_ids)
            if not hasattr(self, "_grapple_yaw_joint_id"):
                _yj, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
                self._grapple_yaw_joint_id = int(_yj[0]) if len(_yj) > 0 else None
            if self._grapple_yaw_joint_id is not None:
                self.crane.set_joint_position_target(
                    torch.zeros(gaze_ids.shape[0], 1, device=self.device),
                    joint_ids=[self._grapple_yaw_joint_id], env_ids=gaze_ids,
                )
            self.crane.write_data_to_sim()

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

                if logs_deposited >= 200 or cycles >= args_cli.max_grasp_cycles:
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

            # Episode return (cumulative reward) — use completed-episode buffer
            # to avoid bias from mid-episode partial sums and just-reset zeros
            if len(self._completed_ep_returns) > 0:
                episode_return = statistics.mean(self._completed_ep_returns)
            else:
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

                # Logs knocked out of bounds (tracked separately; not penalized by default)
                "knocked_off_logs": self._logs_knocked_off.float().mean().item(),

                # Secondary metrics
                "logs_grasped": total_logs.mean().item(),
                "successful_grasps": successful.mean().item(),
                "failed_grasps": failed.mean().item(),
                "cycles": cycles_per_env.mean().item(),

                # Reward components (will appear as reward/X in TensorBoard due to "/")
                "reward/per_cycle": (self._episode_return / cycles_per_env.clamp(min=1)).mean().item(),
            }

            # Clearing bonus metrics (only when clearing bonus is enabled)
            if getattr(self.cfg, 'clearing_bonus_scale', 0.0) > 0.0:
                extras["episode"]["clearing_bonus"] = self._last_clearing_bonus.mean().item()

            # Curriculum metrics (only when curriculum is active)
            if self.cfg.curriculum_schedule is not None:
                extras["episode"]["curriculum/active_logs"] = float(self._curriculum_active_logs)
                extras["episode"]["curriculum/episodes_completed"] = float(self._total_episodes_completed)
                extras["episode"]["curriculum/stage"] = float(self._curriculum_stage_idx)

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

    def _gaze_camera_pose_w(self, env_idx: int):
        """True optical-frame camera pose in WORLD, from the LIVE mast link pose composed with the
        calibrated mount offset (cfg offset = T_mast<-zed_0_optical, ROS convention). Works around
        the stale camera-sensor pose, which reports the un-slewed mast and yaw-rotates the cloud.
        Returns (pos (3,), quat (4,) wxyz in ROS-optical convention)."""
        if not hasattr(self, "_mast_body_idx"):
            _bid, _ = self.crane.find_bodies(["mast"])
            self._mast_body_idx = int(_bid[0])
            off = self.cfg.camera_cfg.offset
            self._cam_off_pos = torch.tensor(off.pos, device=self.device, dtype=torch.float32)
            self._cam_off_rot = torch.tensor(off.rot, device=self.device, dtype=torch.float32)  # wxyz
        mast_pos = self.crane.data.body_pos_w[env_idx, self._mast_body_idx]
        mast_quat = self.crane.data.body_quat_w[env_idx, self._mast_body_idx]  # wxyz
        cam_pos = mast_pos + _quat_rotate_vec_wxyz(mast_quat, self._cam_off_pos)
        cam_quat = quat_multiply_wxyz(mast_quat, self._cam_off_rot)
        return cam_pos, cam_quat

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

    def _depth_for_cloud(self, env_idx: int, depth_range: tuple):
        """Range-filtered (+ optional ZED-noise) depth and intrinsics. Single source of the
        ZED noise model, shared by the optical/base/world cloud getters. Returns (None, None)
        if the camera/depth is unavailable."""
        if self._camera is None or "depth" not in self.cfg.camera_cfg.data_types:
            return None, None
        depth = self._camera.data.output["depth"][env_idx].squeeze(-1)
        intrinsics = self._camera.data.intrinsic_matrices[env_idx]
        inf = torch.tensor(float('inf'), device=self.device)
        min_depth, max_depth = depth_range
        depth_mask = (depth >= min_depth) & (depth <= max_depth) & (~torch.isinf(depth))
        filtered_depth = torch.where(depth_mask, depth, inf)
        # ZED stereo noise model (sim2real). Isaac renders perfect depth; a real ZED gets depth from
        # stereo disparity, so error grows ~ z^2 and pixels drop out (holes at edges / low texture).
        # Perturbing depth BEFORE unprojection puts the axial noise along the camera ray, like the
        # real sensor. coeff fitted to the real ZED floor (bag 17_20). Off by default.
        if getattr(self.cfg, "zed_noise", False):
            valid = torch.isfinite(filtered_depth)
            sigma = float(getattr(self.cfg, "zed_axial_coeff", 0.0014)) * filtered_depth ** 2
            filtered_depth = torch.where(valid, filtered_depth + torch.randn_like(filtered_depth) * sigma, filtered_depth)
            drop_p = float(getattr(self.cfg, "zed_dropout", 0.06))
            if drop_p > 0:
                drop = valid & (torch.rand_like(filtered_depth) < drop_p)
                filtered_depth = torch.where(drop, inf, filtered_depth)
        return filtered_depth, intrinsics

    @staticmethod
    def _R_from_quat_wxyz(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y]),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x]),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]),
        ])

    def get_pointcloud_base(self, env_idx: int = 0, max_points: int = None,
                             depth_range: tuple = (0.5, 10.0)) -> torch.Tensor:
        """Point cloud DIRECTLY in the crane base frame (optical -> base, no world pivot).

        Mirrors deployment exactly: the ZED cloud lives in the optical frame and tf gives
        base_link <- optical in one hop. Here we unproject to the optical frame, then apply
        T_base<-optical built from the LIVE optical pose (_gaze_camera_pose_w, slew-correct) and
        the crane root pose (world cancels). Gravity-aligned, crane-fixed, same frame the policy
        outputs its targets in -> the right observation frame for sim2real."""
        filtered_depth, intrinsics = self._depth_for_cloud(env_idx, depth_range)
        if filtered_depth is None:
            return torch.empty((0, 3), device=self.device)
        pts_opt = create_pointcloud_from_depth(
            intrinsic_matrix=intrinsics, depth=filtered_depth, keep_invalid=False, device=self.device)
        if pts_opt.shape[0] == 0:
            return pts_opt
        t_opt_w, q_opt_w = self._gaze_camera_pose_w(env_idx)        # optical pose in world (slew-correct)
        R_opt_w = self._R_from_quat_wxyz(q_opt_w)
        bp = self.crane.data.root_pos_w[env_idx]
        R_base_w = self._R_from_quat_wxyz(self.crane.data.root_quat_w[env_idx])
        R_bo = R_base_w.transpose(0, 1) @ R_opt_w                   # R_base<-optical
        t_bo = R_base_w.transpose(0, 1) @ (t_opt_w - bp)           # t_base<-optical
        pts_base = pts_opt @ R_bo.transpose(0, 1) + t_bo
        if max_points is not None and pts_base.shape[0] > max_points:
            idx = torch.randperm(pts_base.shape[0], device=self.device)[:max_points]
            pts_base = pts_base[idx]
        return pts_base

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
        filtered_depth, intrinsics = self._depth_for_cloud(env_idx, depth_range)
        if filtered_depth is None:
            return torch.empty((0, 3), device=self.device)

        # Camera pose for unprojection. NOTE: self._camera.data.pos_w/quat_w_ros are STALE for a
        # camera on the slewing mast — they report the un-slewed (slew-0) mast pose, while the depth
        # is rendered from the LIVE slewed mast. Using them yaw-rotates the whole cloud. So compute
        # the true optical pose from the LIVE mast link pose composed with the calibrated mount
        # offset (T_mast<-zed_0_optical, the cfg offset). quat_w_ros == this optical-frame quat.
        cam_pos, cam_quat = self._gaze_camera_pose_w(env_idx)

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


        # Semantic segmentation returns class IDs - find the "log" class
        # The semantic value depends on how the class was registered
        # Typically the first channel contains the class ID
        if semantic_seg.dim() == 3:
            semantic_ids = semantic_seg[..., 0]  # Take first channel
        else:
            semantic_ids = semantic_seg

        # Find unique semantic IDs
        unique_ids = torch.unique(semantic_ids)

        # Create mask for log pixels with depth range filter
        # The "log" class should have a specific ID - we need to find it
        # For now, let's assume non-zero IDs that aren't background are logs
        min_depth, max_depth = depth_range
        log_mask = (semantic_ids > 0) & (~torch.isinf(depth)) & (depth >= min_depth) & (depth <= max_depth)

        num_log_pixels = log_mask.sum().item()

        if num_log_pixels == 0:
            return torch.empty((0, 3), device=self.device)

        # Apply mask to depth
        masked_depth = torch.where(log_mask, depth, torch.tensor(float('inf'), device=self.device))

        # Camera pose from the LIVE mast (the sensor pose is stale / un-slewed — see
        # get_pointcloud_world). Required so the cloud isn't yaw-rotated.
        intrinsics = self._camera.data.intrinsic_matrices[env_idx]
        cam_pos, cam_quat = self._gaze_camera_pose_w(env_idx)

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

        # Curriculum is now advanced in step() based on learning iterations
        # Still count episodes for logging
        if self._logs_settled:
            self._total_episodes_completed += len(env_ids)

        # Always randomize pile arrangement (spacing, jitter, center shift) on reset.
        # Log count is only randomized when domain randomization is enabled.
        if self._log_origins_world.numel() == 0:
            self._rebuild_log_origins_world(randomize_patterns=True, env_ids=None)
        else:
            self._rebuild_log_origins_world(randomize_patterns=True, env_ids=env_ids)

        # Run settling on first reset to let logs fall and stabilize
        # This prevents target selection before logs have stopped moving
        needs_settling = not self._logs_settled

        if self._logs_obj is not None:
            logs = self._logs_obj
            max_logs_per_env = self._logs_per_env  # per-env log slot stride (was hardcoded 200)

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
                # NOTE: Do NOT reset _prev_logs_grasped, _prev_grasp_alignment,
                # _prev_grasp_stability, _prev_logs_remaining, or _prev_cycle_knocked_off
                # here. They are read by play scripts AFTER step() returns, and _reset_idx
                # runs INSIDE step(). They get overwritten by the next step()/cycle call.
                self._final_episode_knocked_off[i] = self._logs_knocked_off[i]
                self._final_episode_cycle_count[i] = self._cycle_count[i]
                self._logs_knocked_off[i] = 0  # Reset out-of-bounds counter

                # Reset episode-level metrics for TensorBoard
                # NOTE: _last_clearing_bonus is NOT reset here — it's read by _get_extras()
                # after step() returns, and only written when terminated[i] is True.
                self._episode_successful_grasps[i] = 0
                self._episode_failed_grasps[i] = 0
                self._episode_return[i] = 0.0
                self._episode_total_logs_grasped[i] = 0
                self._episode_alignment_sum_weighted[i] = 0.0
                self._episode_stability_sum_weighted[i] = 0.0

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
        self._phase[:] = self.PH_GAZE  # GAZE: reset enters the gaze phase
        self._has_log[:] = False
        self._timer[:] = 0
        self._dwell[:] = 0
        self._lift_hold_timer[:] = 0
        self._stab_theta_win = [[] for _ in range(self.num_envs)]
        self._stab_theta_full = [[] for _ in range(self.num_envs)]  # _stab_records kept (per-run log)
        self._stab_dwell[:] = 0
        self._stab_settle_steps[:] = 0
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
                self._gripper_close_steps_left[i] = 0   # cancel any live CLOSE window

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
        self._gripper_close_steps_left[env_id] = self.CLOSE_DURATION_STEPS
        
        # Set initial target to open position
        self.q_des_grip[env_id, 0] = self.GRIP_OPEN  
        self.q_des_grip[env_id, 1] = self.GRIP_OPEN
        
    def _update_gripper_stepping(self, env_id: int):
        """Update discrete stepping for one environment."""
        if not self._gripper_stepping_active[env_id]:
            return
            
        # The close window is a fixed DURATION at fixed velocity (real grapple semantics):
        # it runs through the lift and ends on whichever comes first, timeout or full closure.
        self._gripper_close_steps_left[env_id] -= 1
        if self._gripper_close_steps_left[env_id] <= 0:
            self._gripper_stepping_active[env_id] = False
            return

        # Increment step timer
        self._gripper_step_timer[env_id] += 1
        
        # Check if it's time for next step ( time.sleep(CLOSE_DELAY))
        if self._gripper_step_timer[env_id] >= self.CLOSE_DELAY_STEPS:
            # Step from the MEASURED position of the LAGGING jaw, not from the commanded value.
            # A real grapple's tongs share one hydraulic cylinder / linkage, so they cannot reach
            # different angles: whichever jaw is blocked by the bite holds the other back. Driving
            # the command off the commanded value instead let the free jaw run ahead (asymmetric
            # tongs) and let the command race away from stalled jaws, so when the lift later freed
            # them there was no command left to close into. Basing it on min(actual) gives both
            # behaviours for free: symmetry, and closing that resumes the moment the load frees.
            left_id, right_id = self._grip_joint_ids
            cur_min = min(float(self.crane.data.joint_pos[env_id, left_id].item()),
                          float(self.crane.data.joint_pos[env_id, right_id].item()))
            cmd = float(self._gripper_step_target[env_id].item())
            # advance at the commanded velocity, but never more than max_lead ahead of the
            # lagging jaw (shared-linkage compliance): fast when free, force-bounded when stalled
            next_target = min(cmd + self.CLOSE_STEP, cur_min + self.MAX_LEAD, self.GRIP_MAX)
            next_target = max(next_target, cmd)          # never retreat
            
            self._gripper_step_target[env_id] = next_target
            self._gripper_step_timer[env_id] = 0
            
            # Update PD control targets ( send_goal with new target)
            self.q_des_grip[env_id, 0] = next_target
            self.q_des_grip[env_id, 1] = next_target
            
            # Stop only when the JAWS themselves reach the limit; a stalled bite keeps the
            # command live so the squeeze continues as the lift frees the tongs (maintained
            # hydraulic pressure), instead of freezing at whatever the command had reached.
            if next_target >= self.GRIP_MAX and cur_min >= self.GRIP_MAX - self.CLOSE_STEP:
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
        total_cleaned = 0
        for i in range(self.num_envs):
            cleaned = self._check_logs_out_of_bounds(i, apply_penalty=False)
            if cleaned > 0:
                total_cleaned += cleaned

        self._initial_settling_complete = True

        # Compute action space bounds from settled log positions
        # Always compute bounds (needed for OOB checking in both heuristic and RL modes)
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
                start = env_i * self._logs_per_env  # Each env has 200 slots
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
                        available_pos = slice_pos[available_indices]
                        available_quat = slice_quat[available_indices]
                        if getattr(args_cli, "expert_target", "highest") == "density":
                            # ORACLE: most logs within the grasp check's 1.5 m capture radius
                            # (horizontal), tie-broken by height. Upper-bounds density-aware RL.
                            d = torch.cdist(available_pos[:, :2], available_pos[:, :2])
                            score = (d < 1.5).sum(1).float() + 0.01 * available_pos[:, 2]
                            k = int(torch.argmax(score).item())
                        else:
                            k = int(torch.argmax(available_pos[:, 2]).item())  # highest Z among available
                        pos_w = available_pos[k:k+1, :]
                        quat_w = available_quat[k:k+1, :]
                        selected_log_id = available_log_ids[k]  # Store the selected log ID

        # transform log pose from world -> crane base (BG frame logic uses base as reference)
        root_pose_w = self.crane.data.root_pose_w
        base_pos_w  = root_pose_w[env_i:env_i+1, 0:3]
        base_quat_w = root_pose_w[env_i:env_i+1, 3:7]
        log_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, pos_w, quat_w)

        # Expert dig convention: the selected log's CENTRE is (local surface - LOG_RADIUS), so
        # dropping z by (dig - LOG_RADIUS) makes the expert aim at (surface - dig). Applied here
        # so the executed grasp and the recorded BC label move together.
        _dig = float(getattr(args_cli, "expert_dig", 0.0) or 0.0)
        if _dig > 0.0:
            log_pos_b[0, 2] = log_pos_b[0, 2] - (_dig - 0.056)

        # DRY-RUN bed-floor monitor: the expert path is deliberately NOT clamped (its executed
        # depth defines sim grasp mechanics), but report every pick the platform envelope
        # would clamp so envelope pressure is visible when running the sim heuristic.
        _m = float(getattr(self.cfg, "platform_bed_margin", 0.0))
        if _m > 0.0 and hasattr(self, "_action_bounds_valid"):
            if not self._action_bounds_valid[env_i]:
                self._compute_action_space_bounds()
            _floor = float(self._action_bounds_min[env_i][2]) + _m
            if float(log_pos_b[0][2]) < _floor:
                # print once per grasp cycle (the picker re-evaluates every FSM tick)
                if not hasattr(self, "_floor_print_cycle"):
                    self._floor_print_cycle = {}
                _cyc = int(self._cycle_count[env_i].item()) if hasattr(self, "_cycle_count") else -1
                _say = self._floor_print_cycle.get(env_i) != _cyc
                self._floor_print_cycle[env_i] = _cyc
                if bool(getattr(self.cfg, "platform_floor_expert", True)):
                    if _say:
                        print(f"[BED FLOOR] env{env_i} cycle {_cyc + 1}: expert target z "
                              f"{float(log_pos_b[0][2]):.3f} CLAMPED to floor {_floor:.3f}")
                    log_pos_b[0, 2] = _floor
                elif _say:
                    print(f"[BED FLOOR check] env{env_i} cycle {_cyc + 1}: expert target z "
                          f"{float(log_pos_b[0][2]):.3f} below floor {_floor:.3f} (not clamped)")

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

            # Keep the CLOSE window ticking in EVERY phase, not just PH_CLOSE: the real grapple
            # holds its close command for grapple_delta_time_ms (~8 s) and carries on squeezing
            # through the lift, so a bite that stalls the tongs finishes closing once lifting
            # frees them. Ends on whichever comes first, the duration or full closure.
            if bool(self._gripper_stepping_active[i]) and phase != self.PH_CLOSE:
                if phase in (self.PH_LIFT_HIGH, self.PH_CARRY_HOME, self.PH_LOWER_TO_DROP):
                    self._update_gripper_stepping(i)      # still carrying: keep squeezing
                else:
                    # release/return/gaze: the bite is gone, drop the window so it cannot
                    # re-close the tongs against the open command on the next cycle
                    self._gripper_stepping_active[i] = False
                    self._gripper_close_steps_left[i] = 0

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
                    a = self._last_actions[i] if (hasattr(self, "_last_actions") and i < self._last_actions.shape[0]) else torch.zeros(self.cfg.action_space, device=self.device)

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

                    # Map positions to bounds: val = min + (norm + 1) / 2 * (max - min)
                    x_b = min_bounds[0] + (x_norm + 1.0) / 2.0 * (max_bounds[0] - min_bounds[0])
                    y_b = min_bounds[1] + (y_norm + 1.0) / 2.0 * (max_bounds[1] - min_bounds[1])
                    z_b = min_bounds[2] + (z_norm + 1.0) / 2.0 * (max_bounds[2] - min_bounds[2])

                    # Platform bed floor: never execute below box bottom + margin. Mirrors the
                    # crane_policy_node envelope (bed_z + bed_margin, applied to EVERY policy
                    # type on the real crane), so sim training/eval and deployment share one
                    # actuation envelope. The commanded z is the grasp CENTER; the closing
                    # tines sweep below it, hence the margin. 0 disables (counterfactual evals).
                    _bed_margin = float(getattr(self.cfg, "platform_bed_margin", 0.0))
                    if _bed_margin > 0.0:
                        z_b = torch.clamp(z_b, min=min_bounds[2] + _bed_margin)

                    # Decode yaw - supports both 4D and 5D action spaces
                    import math
                    if a.shape[0] == 5:
                        # 5D: [x, y, z, cos(2*yaw), sin(2*yaw)] - cos/sin encoding
                        yaw_cos = torch.tanh(a[3])  # cos(2*yaw) in [-1, 1]
                        yaw_sin = torch.tanh(a[4])  # sin(2*yaw) in [-1, 1]
                        yaw_target = torch.atan2(yaw_sin, yaw_cos) / 2.0  # Decode to yaw
                    else:
                        # 4D: [x, y, z, yaw] - direct encoding (legacy)
                        yaw_norm = torch.tanh(a[3])  # [-1, 1] -> Yaw angle
                        yaw_target = yaw_norm * (math.pi / 2)  # Map to [-π/2, π/2]

                    # Cache target position and yaw for downstream phases
                    self._target_log_pos_b[i] = torch.stack([x_b, y_b, z_b])
                    # NOTE: Do NOT overwrite _target_log_quat_w here - it was set by Expert4D
                    # with the actual target log's quaternion. Overwriting with identity causes
                    # the grapple to always align with the rack instead of the specific log.
                    self._pending_yaw_targets[i] = yaw_target  # Store policy yaw for ALIGN_YAW phase
                    # Keep current yaw during HOVER_UP (don't change it) - yaw alignment happens in ALIGN_YAW phase
                    yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
                    if len(yaw_joint_ids) > 0:
                        yaw_joint_id = int(yaw_joint_ids[0])
                        self._yaw_targets[i] = float(self.crane.data.joint_pos[i, yaw_joint_id].item())

                    # Find which log the policy selected (for reward computation)
                    if hasattr(self, '_logs_obj') and self._logs_obj is not None:
                        per_env = int(self._per_env_log_counts[i]) if hasattr(self, '_per_env_log_counts') else 16
                        start = i * self._logs_per_env  # per-env log slot stride
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
                    # Add observation noise to heuristic target position (for robustness evals)
                    noise_sigma = getattr(self.cfg, 'heuristic_target_noise', 0.0)
                    if noise_sigma > 0:
                        log_pos_b = log_pos_b + torch.randn(3, device=self.device) * noise_sigma
                    self._target_log_pos_b[i] = log_pos_b
                    self._current_target_log_id[i] = log_id
                    # Add orientation noise to heuristic target yaw (for robustness evals)
                    # Auto-derived from position noise: σ_yaw = arctan(σ_pos / (log_length/2))
                    yaw_noise_sigma = 0.0
                    if noise_sigma > 0 and getattr(self.cfg, 'heuristic_yaw_noise_auto', True):
                        import math
                        log_half_length = 1.5  # ~3m logs
                        yaw_noise_sigma = math.atan(noise_sigma / log_half_length)
                    if yaw_noise_sigma > 0:
                        import math
                        dyaw = float(torch.randn(1, device=self.device).item()) * yaw_noise_sigma
                        # Rotate quaternion around world Z by dyaw
                        hw = dyaw / 2.0
                        dq = torch.tensor([math.cos(hw), 0.0, 0.0, math.sin(hw)], device=self.device)
                        # quat_mul: q_new = dq * q_old (wxyz convention)
                        w1, x1, y1, z1 = dq[0], dq[1], dq[2], dq[3]
                        w2, x2, y2, z2 = log_quat_w[0], log_quat_w[1], log_quat_w[2], log_quat_w[3]
                        log_quat_w = torch.tensor([
                            w1*w2 - x1*x2 - y1*y2 - z1*z2,
                            w1*x2 + x1*w2 + y1*z2 - z1*y2,
                            w1*y2 - x1*z2 + y1*w2 + z1*x2,
                            w1*z2 + x1*y2 - y1*x2 + z1*w2,
                        ], device=self.device)
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

            if phase == self.PH_GAZE:
                # GAZE: crane holds the fixed gaze pose (joints driven in _apply_action) so the
                # basemast cam views the rack. Keep gripper OPEN. Dwell until the control joints
                # reach the gaze config, then start the pick cycle (HOVER_UP). Target selection is
                # still done at HOVER_UP for now; it will be retimed to gaze-settled in a later edit.
                self.q_des_grip[i, :] = self.GRIP_OPEN
                self._phase_timer[i] += 1
                gaze_q = torch.tensor(
                    [self.GAZE_SLEW_RAD, self.GAZE_ARM_BOOM_RAD,
                     self.GAZE_ARM_STICK_RAD, self.GAZE_ARM_TELESCOPE_M],
                    device=self.device, dtype=torch.float32,
                )
                q_now = self.crane.data.joint_pos[i, self._ctrl_joint_idx]
                joint_err = torch.max(torch.abs(q_now - gaze_q)).item()
                if joint_err < self.GAZE_JOINT_TOL:
                    self._dwell[i] = self._dwell[i] + 1
                else:
                    self._dwell[i] = 0
                if self._dwell[i] >= self.GAZE_DWELL_N or self._phase_timer[i] >= self.GAZE_TIMEOUT:
                    # DECISION POINT: end the step() here so the loop renders the basemast cam and
                    # the policy selects the target from the gaze-pose PCD. The crane is physically at
                    # the gaze pose now; we transition the phase to HOVER_UP so the *next* step()
                    # consumes the (gaze-PCD-based) action and runs the pick.
                    self._cycle_complete_this_step[i] = True
                    self._transition(i, self.PH_HOVER_UP, "gaze: settled — decision point (return)")

            elif phase == self.PH_HOVER_UP:
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
                # Keep current yaw during HOVER_UP (don't change it) - yaw alignment happens in ALIGN_YAW phase
                yaw_joint_ids, _ = self.crane.find_joints(["lowerpassive_to_basegrapple"])
                if len(yaw_joint_ids) > 0:
                    yaw_joint_id = int(yaw_joint_ids[0])
                    self._yaw_targets[i] = float(self.crane.data.joint_pos[i, yaw_joint_id].item())

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
                # In policy mode (hierarchical RL): convert policy's log_yaw_b to joint position
                if not getattr(self.cfg, "use_hierarchical_rl", False):
                    # Heuristic mode: recalculate yaw for optimal log alignment
                    # Now that arm is positioned above target, calculate optimal alignment
                    self._yaw_targets[i] = self._calc_optimal_yaw_feedback(i)
                else:
                    # Policy mode: convert policy's predicted basegrapple yaw to yaw joint position
                    # Policy predicts desired basegrapple orientation in basemast frame (state-independent)
                    # Now that we're hovering, convert it to joint position based on arm configuration
                    target_grapple_yaw_b = self._pending_yaw_targets[i]
                    self._yaw_targets[i] = self._convert_grapple_yaw_to_joint_position(i, target_grapple_yaw_b)

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
                        self._snapshot_lift_entry(i)
                        self._transition(i, self.PH_LIFT_HIGH, f"{name}: gripper closed and stable (target={target_value:.2f}, max_err={max_error:.3f})")
                else:
                    # Lost tolerance or stepping not complete - reset stability timer
                    self._gripper_stability_timer[i] = 0

                # Timeout check
                if self._gripper_timeout_timer[i] >= self.GRIPPER_TIMEOUT:
                    self._has_log[i] = True
                    self._gripper_timeout_timer[i] = 0
                    self._gripper_stability_timer[i] = 0
                    self._snapshot_lift_entry(i)
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
                    # In hold mode - keep position, decrement timer.
                    import math
                    self._lift_hold_timer[i] -= 1
                    if not self.cfg.stability_snapshot:
                        # Mirror the real settle detector: rolling tilt window; settled when the
                        # window variance stays under threshold for the dwell. Ends the hold early;
                        # the timeout (STABILITY_HOLD_MAX) falls through as best-effort.
                        win = self._stab_theta_win[i]
                        win.append(math.acos(min(1.0, self._grapple_tilt_cos(i))))
                        self._stab_theta_full[i].append(win[-1])  # untrimmed, for raw logging
                        if len(win) > self.STABILITY_WINDOW_N:
                            win.pop(0)
                        if len(win) >= self.STABILITY_WINDOW_N:
                            m = sum(win) / len(win)
                            var = sum((t - m) ** 2 for t in win) / len(win)
                            self._stab_dwell[i] = self._stab_dwell[i] + 1 if var < self.STABILITY_VAR_THRESH else 0
                            if self._stab_dwell[i] >= self.STABILITY_DWELL_N:
                                self._stab_settle_steps[i] = self.STABILITY_HOLD_MAX - int(self._lift_hold_timer[i])
                                self._lift_hold_timer[i] = 0  # settled - end the hold now
                        if self._lift_hold_timer[i] == 0 and self._stab_dwell[i] < self.STABILITY_DWELL_N:
                            self._stab_settle_steps[i] = self.STABILITY_HOLD_MAX  # timeout, best-effort
                    if self._lift_hold_timer[i] == 0:
                        # Hold complete - now evaluate and transition
                        # IMPORTANT: Check OOB first, then count grasped logs
                        # This prevents knocked-off logs from being counted toward pile clearing
                        # (OOB check only despawns logs outside X/Y bounds or below Z_min,
                        #  NOT logs lifted high, so grasped logs are safe)
                        knocked_off = self._check_logs_out_of_bounds(i, apply_penalty=False)
                        logs_grasped, alignment, stability = self._check_grasped_logs(i)
                        # GRASP-CHECK GATES (2026-08-05). Proximity-only counting despawns and
                        # rewards logs the grapple merely hovered near whenever the LIFT stalls
                        # low (observed: the lift_step-0.08 57-log mass despawn; reachable again
                        # via rigid stub fixtures jamming the tongs -> LIFT exits by timeout at
                        # pile height). Gate 1: the grapple must have actually gained height.
                        # Gate 2 lives inside _check_grasped_logs (per-log rise vs its pre-lift
                        # z). Failing the gates = a failed grasp: nothing counted, nothing
                        # despawned, -1 reward like any miss.
                        _bg_z = float(self.crane.data.body_pose_w[i, self._basegrapple_body_id][2])
                        _lift_ok = _bg_z >= float(self._lift_entry_bg_z[i]) + 0.8
                        if not _lift_ok and logs_grasped > 0:
                            print(f"[env{i}] GRASP GATE: lift stalled low "
                                  f"(bg z {_bg_z:.2f}, entry {float(self._lift_entry_bg_z[i]):.2f}) "
                                  f"-> {logs_grasped} proximity logs NOT counted/despawned")
                            logs_grasped, alignment = 0, 0.0
                        # Windowed stability (default): replace the instantaneous sample with the
                        # mean tilt over the hold window; sharpen once (average-then-sharpen).
                        tilt_deg = None
                        if not self.cfg.stability_snapshot and self._stab_theta_win[i]:
                            import math
                            # Mean tilt over the accepted (settled or best-effort) window,
                            # sharpened once - same as the real node's reward_exponent.
                            theta_mean = sum(self._stab_theta_win[i]) / len(self._stab_theta_win[i])
                            stability = math.cos(theta_mean) ** getattr(self.cfg, "stability_exponent", 4)
                            tilt_deg = math.degrees(theta_mean)
                            # Raw record (full untrimmed progression at FSM rate) so the metric
                            # can be recomputed under a different definition later.
                            self._stab_records[i].append({
                                "cycle": int(self._cycle_count[i].item()) + 1,
                                "settled": bool(self._stab_dwell[i] >= self.STABILITY_DWELL_N),
                                "settle_steps": int(self._stab_settle_steps[i].item()),
                                "theta_mean_rad": float(theta_mean),
                                "stability": float(stability),
                                "theta_rad": [round(t, 5) for t in self._stab_theta_full[i]],
                            })
                        # See cfg.reward_requires_lift: _lift_ok (computed above) gates the
                        # despawn below, so a cycle that fails it clears nothing. Pay on the
                        # gated count; keep `logs_grasped` ungated for the metrics/logging.
                        _rew_logs = logs_grasped
                        if getattr(self.cfg, "reward_requires_lift", True) and not _lift_ok:
                            _rew_logs = 0
                        reward = self._compute_grasp_reward(_rew_logs, alignment, stability, i, knocked_off)
                        self._grasp_reward_buf[i] = reward
                        # Guardrail data for the reward-vs-truth correlation check: a fine-tune
                        # whose reward does not track actual rack decrease is hacking, not
                        # learning. Consumed by check_reward_alignment.py.
                        self._audit_reward(reward, logs_grasped, _rew_logs, _lift_ok, i)
                        self._prev_logs_grasped[i] = float(logs_grasped)
                        self._prev_grasp_alignment[i] = alignment
                        self._prev_grasp_stability[i] = stability
                        # GAZE: do NOT end the step() at the grasp — the cycle now returns at the
                        # gaze pose (set in the PH_GAZE block) so the policy sees the gaze-pose PCD.
                        # (grasp reward above is still computed here.)

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

                        cycle_num = self._cycle_count[i].item()   # already incremented for this cycle
                        logs_remaining = self._count_logs_in_rack(i)
                        # Subtract current grasp since despawn happens after this print
                        remaining_after_grasp = max(0, logs_remaining - logs_grasped)
                        tilt_str = (f", tilt={tilt_deg:.1f}deg, settle={self._stab_settle_steps[i].item() / 60.0:.1f}s"
                                    if tilt_deg is not None else "")
                        print(f"[env{i}] CYCLE {cycle_num}/{args_cli.max_grasp_cycles} | GRASP: {logs_grasped} logs, align={alignment:.2f}, stab={stability:.2f}{tilt_str}, rew={reward:.2f} | remaining={remaining_after_grasp}")

                        if _lift_ok and logs_grasped > 0:
                            self._despawn_grasped_logs(i)
                        self._set_gripper(i, open_fraction=1.0)
                        self._target_frozen[i] = False
                        self._target_log_pos_b[i].zero_()
                        self._target_log_quat_w[i].zero_()
                        self._transition(i, self.PH_GAZE, f"{name}: hold complete, returning to gaze")
                else:
                    # Not in hold mode - check if we should enter it
                    self._dwell[i] = self._dwell[i] + 1 if (upperpassive_error < self.UP_TOL) else torch.tensor(0, device=self.device)

                    if self._dwell[i] >= self.DWELL_N:
                        # Dwell complete - enter hold mode
                        self._dwell[i] = 0
                        self._lift_hold_timer[i] = (self.LIFT_HOLD_DURATION if self.cfg.stability_snapshot
                                                    else self.STABILITY_HOLD_MAX)
                        self._stab_theta_win[i] = []
                        self._stab_theta_full[i] = []
                        self._stab_dwell[i] = 0

                    elif self._phase_timer[i] >= self.PHASE_TIMEOUT:
                        # Timeout - enter hold mode (brief hold even on timeout)
                        self._lift_hold_timer[i] = (self.LIFT_HOLD_DURATION if self.cfg.stability_snapshot
                                                    else self.STABILITY_HOLD_MAX)
                        self._stab_theta_win[i] = []
                        self._stab_theta_full[i] = []
                        self._stab_dwell[i] = 0

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

                    self._transition(i, self.PH_GAZE, f"{name}: done, looping to gaze")

    def _save_gaze_view(self, env_i: int):
        """Validation: render the basemast camera at the (settled) gaze pose and save
        RGB + depth + 3D point cloud to /workspace/crane_testbed/gaze_views/. Captured
        inside the FSM so it's the actual gaze-pose view (needs GUI/non-headless to render)."""
        try:
            import os
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            if self._camera is None:
                return
            self._camera.update(self.physics_dt)
            rgb = self._camera.data.output.get("rgb")
            depth = self._camera.data.output.get("depth")
            rgb_np = (rgb[env_i][:, :, :3].cpu().numpy().astype("uint8")
                      if rgb is not None and rgb.shape[0] > env_i else None)
            depth_np = (depth[env_i].squeeze(-1).cpu().numpy()
                        if depth is not None and depth.shape[0] > env_i else None)
            try:
                pc = self.get_pointcloud_world(env_i, max_points=8000).cpu().numpy()
            except Exception:
                pc = None
            if not hasattr(self, "_gaze_view_count"):
                self._gaze_view_count = {}
            n = self._gaze_view_count.get(env_i, 0)
            self._gaze_view_count[env_i] = n + 1
            slew = float(self.crane.data.joint_pos[env_i, self._ctrl_joint_idx[0]].item())
            out = "/workspace/crane_testbed/gaze_views"
            os.makedirs(out, exist_ok=True)
            fig = plt.figure(figsize=(18, 5))
            ax1 = fig.add_subplot(1, 3, 1)
            if rgb_np is not None:
                ax1.imshow(rgb_np)
            ax1.set_title("basemast RGB"); ax1.axis("off")
            ax2 = fig.add_subplot(1, 3, 2)
            if depth_np is not None:
                im = ax2.imshow(depth_np, cmap="turbo"); fig.colorbar(im, ax=ax2, fraction=0.046)
            ax2.set_title("basemast depth"); ax2.axis("off")
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            if pc is not None and len(pc):
                ax3.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, c=pc[:, 2], cmap="viridis")
            ax3.set_title("basemast PCD (world)")
            fig.suptitle(f"GAZE view env{env_i} #{n}  (slew={slew:.2f} rad)")
            fig.savefig(f"{out}/gaze_env{env_i}_{n:03d}.png", dpi=90, bbox_inches="tight")
            plt.close(fig)
            print(f"[gaze] saved view #{n} for env {env_i} (slew={slew:.2f}) -> {out}")
        except Exception as e:
            print(f"[gaze] _save_gaze_view failed: {e}")

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

        # Track cycle start when entering GAZE (start of pick cycle in the gaze env)
        if new_phase == self.PH_GAZE:
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
                per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else self._logs_per_env
                env_offset = env_i * self._logs_per_env  # Each env has 200 slots
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
        raw_stability = self._grapple_tilt_cos(env_i)

        # Sharpen so small tilts are penalized more strongly
        stability = raw_stability ** getattr(self.cfg, "stability_exponent", 4)
        return stability

    def _grapple_tilt_cos(self, env_i: int) -> float:
        """Cosine of the grapple tilt angle: dot(grapple up, world up), clamped to [0, 1]."""
        bg_pose_w = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id]
        bg_quat_w = bg_pose_w[3:7]  # [w, x, y, z]
        grapple_up = self._quat_rotate_vec_wxyz(bg_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device))
        # Z component of grapple 'up' in world frame; negative would mean upside down
        return max(0.0, grapple_up[2].item())

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

        # Extract basegrapple Y-axis in world frame (used for log length alignment)
        # Rotate [0, 1, 0] by basegrapple quaternion to get the axis in world frame
        bg_y_axis = self._quat_rotate_vec_wxyz(bg_quat_w, torch.tensor([0.0, 1.0, 0.0], device=self.device))

        # Get all log positions for this environment
        per_env = int(self._per_env_log_counts[env_i])  # Actual active log count
        start = env_i * self._logs_per_env  # Each env has 200 slots
        end = start + per_env  # Only check active logs

        logs_grasped = 0
        alignment_sum = 0.0

        for log_idx in range(start, end):
            # Skip already deposited logs
            if log_idx in self._deposited_logs[env_i]:
                continue

            log_pos_w = self._logs_obj.data.root_pos_w[log_idx]
            distance = torch.norm(log_pos_w - bg_pos_w).item()

            # rise gate: a truly grasped log MOVED UP with the grapple since lift entry;
            # a log that merely sits near a stalled/low grapple did not (>0.4 m of rise
            # cannot come from pile settling). Falls open when no snapshot exists.
            _rise_ok = True
            if hasattr(self, "_log_z_at_lift") and log_idx in self._log_z_at_lift.get(env_i, {}):
                _rise_ok = float(log_pos_w[2]) - self._log_z_at_lift[env_i][log_idx] > 0.4

            if distance < proximity_radius and _rise_ok:
                logs_grasped += 1

                # Compute orientation alignment between log and basegrapple
                log_quat_w = self._logs_obj.data.root_quat_w[log_idx]  # [w, x, y, z]

                # Get log's length axis in world frame (logs are modeled with length along +Y)
                log_length_axis = self._quat_rotate_vec_wxyz(log_quat_w, torch.tensor([0.0, 1.0, 0.0], device=self.device))

                # Alignment: grapple Y-axis parallel to log length axis
                # 1.0 = perfectly aligned, 0.0 = perpendicular
                dot_product = torch.dot(bg_y_axis, log_length_axis).item()
                # Logs can point either direction, so we use the even power
                # which maps [-1,1] → [0,1] with sharp dropoff on misalignment
                alignment = dot_product ** getattr(self.cfg, "alignment_exponent", 8)

                alignment_sum += alignment

        avg_alignment = alignment_sum / max(logs_grasped, 1)

        # Compute grapple stability (how level it is after lifting)
        stability = self._compute_grapple_stability(env_i)

        return logs_grasped, avg_alignment, stability

    def _snapshot_lift_entry(self, env_i: int):
        """Record grapple z and every active log's z at LIFT_HIGH entry (grasp-check gates)."""
        if not hasattr(self, "_lift_entry_bg_z"):
            self._lift_entry_bg_z = torch.zeros(self.num_envs, device=self.device)
            self._log_z_at_lift = {}
        self._lift_entry_bg_z[env_i] = self.crane.data.body_pose_w[env_i, self._basegrapple_body_id][2]
        per_env = int(self._per_env_log_counts[env_i])
        start = env_i * self._logs_per_env
        snap = {}
        if self._logs_obj is not None:
            for li in range(start, start + per_env):
                if li not in self._deposited_logs[env_i]:
                    snap[li] = float(self._logs_obj.data.root_pos_w[li][2])
        self._log_z_at_lift[env_i] = snap

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
        start = env_i * self._logs_per_env
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
        start = env_i * self._logs_per_env  # Each env has 200 slots
        end = start + per_env  # Only check active logs

        # Despawn against the FULL RACK footprint (wider than the action box), so settled edge
        # logs that sit inside the rack but outside the tighter action bounds are NOT culled.
        if hasattr(self, "_rack_bounds_min"):
            min_bounds = self._rack_bounds_min[env_i]  # [x_min, y_min, z_min]
            max_bounds = self._rack_bounds_max[env_i]  # [x_max, y_max, z_max]
        else:
            min_bounds = self._action_bounds_min[env_i]
            max_bounds = self._action_bounds_max[env_i]

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


    def _audit_reward(self, reward, logs_grasped, rew_logs, lift_ok, env_i):
        """Append one reward-vs-truth record, flushed periodically to jsonl.

        A fine-tune whose reward does not track actual rack decrease is hacking the objective,
        not learning the task - which is exactly what happened before cfg.reward_requires_lift
        (training reward rose while deployed argmax clearing fell). Recording both every cycle
        makes that detectable within a few hundred cycles instead of after a wasted run.
        Analysed by scripts/envs/check_reward_alignment.py.
        """
        rec = {"reward": float(reward), "logs_grasped": int(logs_grasped),
               "rew_logs": int(rew_logs), "lift_ok": bool(lift_ok),
               "rack_before_despawn": int(self._count_logs_in_rack(env_i)), "env": int(env_i)}
        self._reward_audit.append(rec)
        if len(self._reward_audit) >= 200:
            try:
                os.makedirs(os.path.dirname(self._reward_audit_path), exist_ok=True)
                with open(self._reward_audit_path, "a") as fh:
                    for r in self._reward_audit:
                        fh.write(json.dumps(r) + "\n")
            except Exception as e:                      # never let logging kill a long run
                print(f"[reward-audit] write failed ({e}); dropping buffer")
            self._reward_audit = []

    def _compute_grasp_reward(self, logs_grasped: int, alignment: float, stability: float, env_i: int, knocked_off: int = 0) -> float:
        """Compute reward for a grasp attempt.

        Reward is computed once per pick cycle (after LIFT_HIGH).

        Supported formulas (cfg.reward_formula):
          - "multiplicative":
              * normalize_reward=True : (logs_grasped / available) * scale * alignment * stability
              * normalize_reward=False: logs_grasped * alignment * stability
          - "additive":
              * normalize_reward=True : (logs_grasped / available) + alignment (+ stability)
              * normalize_reward=False: (logs_grasped / max_graspable_logs) + alignment (+ stability)

        Args:
            logs_grasped: Number of logs grasped
            alignment: Alignment score
            stability: Stability score
            env_i: Environment index
            knocked_off: Number of logs knocked out of bounds (already checked by caller)

        Notes:
          - Alignment and stability are already sharpened in their respective helper functions.
          - Out-of-bounds ("knocked off") logs are tracked separately; penalty weight defaults to 0.0.
          - Caller should check OOB BEFORE counting grasped logs to avoid double-counting.
        """
        # Only attribute "knocked off" to the policy after initial settling.
        # Note: _logs_knocked_off is now updated at the end of step() using the
        # more accurate remaining-count delta (_prev_cycle_knocked_off).
        # The OOB count here is still used for the per-cycle penalty.
        if not getattr(self, "_initial_settling_complete", True):
            knocked_off = 0

        knocked_off_penalty = float(getattr(self, "_logs_out_of_bounds_penalty", 0.0)) * float(knocked_off)

        raw_available = int(self._logs_available_at_target[env_i].item()) if hasattr(self, "_logs_available_at_target") else 0
        empty_target_penalty = float(getattr(self.cfg, "empty_target_penalty", 0.0))  # set negative to discourage empty picks

        # Failed grasp: return failure penalty (+ optional empty-target penalty) (+ knocked-off penalty).
        if logs_grasped <= 0:
            total = float(getattr(self.cfg, "failure_penalty", -1.0))
            if empty_target_penalty != 0.0 and raw_available <= 0:
                total += empty_target_penalty
            total += knocked_off_penalty
            # Per-cycle time cost: every cycle costs the same on the real crane regardless of
            # how many logs it lifts, so charging for it turns the objective into logs-per-CYCLE.
            return float(total) - float(getattr(self.cfg, "cycle_cost", 0.0) or 0.0)

        # Clamp quality terms.
        a = max(0.0, min(float(alignment), 1.0))
        if getattr(self.cfg, "use_stability_reward", False):
            s = max(0.0, min(float(stability), 1.0))
        else:
            s = 1.0

        # Alignment term/factor (allows ablation).
        use_align = bool(getattr(self.cfg, "use_alignment_reward", True))
        align_factor = a if use_align else 1.0
        align_term = a if use_align else 0.0

        # Efficiency / throughput.
        if getattr(self.cfg, "normalize_reward", False):
            denom = min(max(1, raw_available), int(getattr(self.cfg, "max_graspable_logs", 20)))
            efficiency = min(float(logs_grasped) / float(denom), 1.0)
        else:
            efficiency = float(logs_grasped)

        # Compute reward.
        formula = getattr(self.cfg, "reward_formula", "multiplicative")
        if formula == "additive":
            if getattr(self.cfg, "normalize_reward", False):
                total_reward = efficiency + align_term
                if getattr(self.cfg, "use_stability_reward", False):
                    total_reward += s
            else:
                max_g = float(getattr(self.cfg, "max_graspable_logs", 15))
                total_reward = (efficiency / max_g) + align_term
                if getattr(self.cfg, "use_stability_reward", False):
                    total_reward += s
        else:
            # multiplicative (default)
            stability_factor = s if getattr(self.cfg, "use_stability_reward", False) else 1.0
            if getattr(self.cfg, "normalize_reward", False):
                scale = float(getattr(self.cfg, "normalized_efficiency_scale", 10.0))
                total_reward = efficiency * scale * align_factor * stability_factor
            else:
                total_reward = efficiency * align_factor * stability_factor

        total_reward += knocked_off_penalty

        if getattr(self.cfg, "debug_reward", False):
            print(
                f"[DEBUG] Env {env_i}: logs={logs_grasped}, avail={raw_available}, eff={efficiency:.3f}, "
                f"a={a:.3f}, s={s:.3f}, knocked_off={knocked_off}, reward={total_reward:.3f}"
            )

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
        per_env = int(self._per_env_log_counts[env_i]) if hasattr(self, '_per_env_log_counts') else self._logs_per_env
        start = env_i * self._logs_per_env  # Each env has 200 slots
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

    def _frame_box_edges(self, mins, maxs, t):
        """12 thin-cuboid edges for a wireframe box. mins/maxs are [N,3] (base frame),
        t = edge thickness (m). Returns (centers_b [N,12,3], scales_b [N,12,3])."""
        N = mins.shape[0]
        x0, y0, z0 = mins[:, 0], mins[:, 1], mins[:, 2]
        x1, y1, z1 = maxs[:, 0], maxs[:, 1], maxs[:, 2]
        cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1); cz = 0.5 * (z0 + z1)
        Lx = (x1 - x0).clamp(min=1e-6); Ly = (y1 - y0).clamp(min=1e-6); Lz = (z1 - z0).clamp(min=1e-6)
        tt = torch.full((N,), t, device=mins.device, dtype=mins.dtype)
        centers_x = torch.stack([torch.stack([cx, y0, z0], 1), torch.stack([cx, y0, z1], 1),
                                 torch.stack([cx, y1, z0], 1), torch.stack([cx, y1, z1], 1)], 1)
        scales_x = torch.stack([torch.stack([Lx, tt, tt], 1)] * 4, 1)
        centers_y = torch.stack([torch.stack([x0, cy, z0], 1), torch.stack([x0, cy, z1], 1),
                                 torch.stack([x1, cy, z0], 1), torch.stack([x1, cy, z1], 1)], 1)
        scales_y = torch.stack([torch.stack([tt, Ly, tt], 1)] * 4, 1)
        centers_z = torch.stack([torch.stack([x0, y0, cz], 1), torch.stack([x0, y1, cz], 1),
                                 torch.stack([x1, y0, cz], 1), torch.stack([x1, y1, cz], 1)], 1)
        scales_z = torch.stack([torch.stack([tt, tt, Lz], 1)] * 4, 1)
        centers_b = torch.cat([centers_x, centers_y, centers_z], 1)  # [N,12,3]
        scales_b = torch.cat([scales_x, scales_y, scales_z], 1)      # [N,12,3]
        return centers_b, scales_b

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
            show_bg = bool(getattr(args_cli, 'debug_logs', False))
            if not show_bg:
                bg_target_w = torch.full_like(bg_target_w, -1000.0)
            self._viz["bg_target"].visualize(bg_target_w, base_quat_w)

            log_b = self._target_log_pos_b.clone()
            if (log_b.abs().sum(dim=1) == 0).any():
                for idx in (log_b.abs().sum(dim=1) == 0).nonzero(as_tuple=False).flatten().tolist():
                    log_b[idx], _, _ = self._target_top_log_center_b(idx)  # Ignore log ID & quat in visualization
                    pass  # removed debug print
            
            log_pos_w = base_pos_w + self._quat_rotate_vec_wxyz(base_quat_w, log_b)

            # Visualize target as red sphere (hide when showing action bounds or reward norm slice)
            if getattr(args_cli, 'show_action_bounds', False) or getattr(args_cli, 'debug_reward_norm', False):
                self._viz["log"].visualize(torch.full_like(log_pos_w, -1000.0))
            else:
                self._viz["log"].visualize(log_pos_w)

            # Action space bounding box visualization (frame cube)
            if getattr(args_cli, 'show_action_bounds', False):
                # Ensure bounds exist at least once (e.g., heuristic runs may not precompute them)
                if (not hasattr(self, '_action_bounds_valid')) or (not self._action_bounds_valid.any()):
                    self._compute_action_space_bounds()
                if hasattr(self, '_action_bounds_valid') and self._action_bounds_valid.any():
                    mins = self._action_bounds_min  # [N,3] in base frame
                    maxs = self._action_bounds_max  # [N,3] in base frame

                    x0, y0, z0 = mins[:, 0], mins[:, 1], mins[:, 2]
                    x1, y1, z1 = maxs[:, 0], maxs[:, 1], maxs[:, 2]
                    cx = 0.5 * (x0 + x1)
                    cy = 0.5 * (y0 + y1)
                    cz = 0.5 * (z0 + z1)

                    Lx = (x1 - x0).clamp(min=1e-6)
                    Ly = (y1 - y0).clamp(min=1e-6)
                    Lz = (z1 - z0).clamp(min=1e-6)

                    # Edge thickness (meters). Small but visible in RTX Real-Time.
                    t = float(getattr(args_cli, "action_bounds_edge_thickness", 0.01))
                    tt = torch.full((N,), t, device=self.device, dtype=mins.dtype)

                    # 4 edges along X (at y in {y0,y1}, z in {z0,z1})
                    centers_x = torch.stack(
                        [
                            torch.stack([cx, y0, z0], dim=1),
                            torch.stack([cx, y0, z1], dim=1),
                            torch.stack([cx, y1, z0], dim=1),
                            torch.stack([cx, y1, z1], dim=1),
                        ],
                        dim=1,
                    )  # [N,4,3]
                    scales_x = torch.stack([torch.stack([Lx, tt, tt], dim=1)] * 4, dim=1)  # [N,4,3]

                    # 4 edges along Y (at x in {x0,x1}, z in {z0,z1})
                    centers_y = torch.stack(
                        [
                            torch.stack([x0, cy, z0], dim=1),
                            torch.stack([x0, cy, z1], dim=1),
                            torch.stack([x1, cy, z0], dim=1),
                            torch.stack([x1, cy, z1], dim=1),
                        ],
                        dim=1,
                    )  # [N,4,3]
                    scales_y = torch.stack([torch.stack([tt, Ly, tt], dim=1)] * 4, dim=1)  # [N,4,3]

                    # 4 edges along Z (at x in {x0,x1}, y in {y0,y1})
                    centers_z = torch.stack(
                        [
                            torch.stack([x0, y0, cz], dim=1),
                            torch.stack([x0, y1, cz], dim=1),
                            torch.stack([x1, y0, cz], dim=1),
                            torch.stack([x1, y1, cz], dim=1),
                        ],
                        dim=1,
                    )  # [N,4,3]
                    scales_z = torch.stack([torch.stack([tt, tt, Lz], dim=1)] * 4, dim=1)  # [N,4,3]

                    centers_b = torch.cat([centers_x, centers_y, centers_z], dim=1)  # [N,12,3]
                    scales_b = torch.cat([scales_x, scales_y, scales_z], dim=1)      # [N,12,3]

                    centers_b = centers_b.reshape(N * 12, 3)
                    scales_b = scales_b.reshape(N * 12, 3)

                    base_pos_rep = base_pos_w.unsqueeze(1).expand(N, 12, 3).reshape(N * 12, 3)
                    base_quat_rep = base_quat_w.unsqueeze(1).expand(N, 12, 4).reshape(N * 12, 4)

                    centers_w = base_pos_rep + self._quat_rotate_vec_wxyz(base_quat_rep, centers_b)

                    self._viz["action_bounds"].visualize(
                        translations=centers_w,
                        orientations=base_quat_rep,
                        scales=scales_b,
                    )

            # Trailer deposit scan bounds (orange wireframe), base_link frame. Box for the pile-top scan;
            # all six faces are CLI-tunable (--trailer_box_{x,y,z}{min,max}) so it can be dialed in live.
            if getattr(args_cli, 'show_trailer_bounds', False):
                tmin = torch.tensor([args_cli.trailer_box_xmin, args_cli.trailer_box_ymin, args_cli.trailer_box_zmin],
                                    device=self.device, dtype=base_pos_w.dtype)
                tmax = torch.tensor([args_cli.trailer_box_xmax, args_cli.trailer_box_ymax, args_cli.trailer_box_zmax],
                                    device=self.device, dtype=base_pos_w.dtype)
                tmin = tmin.unsqueeze(0).expand(N, 3)
                tmax = tmax.unsqueeze(0).expand(N, 3)
                t = float(getattr(args_cli, "action_bounds_edge_thickness", 0.01))
                centers_tb, scales_tb = self._frame_box_edges(tmin, tmax, t)
                centers_tb = centers_tb.reshape(N * 12, 3)
                scales_tb = scales_tb.reshape(N * 12, 3)
                base_pos_rep = base_pos_w.unsqueeze(1).expand(N, 12, 3).reshape(N * 12, 3)
                base_quat_rep = base_quat_w.unsqueeze(1).expand(N, 12, 4).reshape(N * 12, 4)
                centers_tw = base_pos_rep + self._quat_rotate_vec_wxyz(base_quat_rep, centers_tb)
                self._viz["trailer_bounds"].visualize(
                    translations=centers_tw,
                    orientations=base_quat_rep,
                    scales=scales_tb,
                )
            # Grasp count prism visualization (rack slice) as wireframe (frame cube)
            # This is purely a debug visual for the rack-slice used in _count_logs_in_column().
            if "grasp_prism" in self._viz:
                # Show only when debug_reward is enabled (or when explicitly requested).
                show_prism = bool(getattr(self.cfg, "debug_reward", False) or getattr(args_cli, "show_grasp_prism", False))

                N = base_pos_w.shape[0]
                if not show_prism:
                    # Keep it hidden (move far away) so it doesn't linger at the origin.
                    hide_pos = torch.full((N * 12, 3), -1000.0, device=self.device, dtype=base_pos_w.dtype)
                    hide_quat = base_quat_w.unsqueeze(1).expand(N, 12, 4).reshape(N * 12, 4)
                    hide_scale = torch.full((N * 12, 3), 1e-3, device=self.device, dtype=base_pos_w.dtype)
                    self._viz["grasp_prism"].visualize(translations=hide_pos, orientations=hide_quat, scales=hide_scale)
                else:
                    active_mask = getattr(self, "_debug_cylinder_active", torch.zeros((N,), device=self.device, dtype=torch.bool))
                    has_size = hasattr(self, "_debug_prism_size")
                    if (not has_size) or (not active_mask.any()):
                        # No prism computed yet -> hide it.
                        hide_pos = torch.full((N * 12, 3), -1000.0, device=self.device, dtype=base_pos_w.dtype)
                        hide_quat = base_quat_w.unsqueeze(1).expand(N, 12, 4).reshape(N * 12, 4)
                        hide_scale = torch.full((N * 12, 3), 1e-3, device=self.device, dtype=base_pos_w.dtype)
                        self._viz["grasp_prism"].visualize(translations=hide_pos, orientations=hide_quat, scales=hide_scale)
                    else:
                        # Prism center (world) + size (base-aligned) -> build AABB in base frame
                        prism_center_w = self._debug_cylinder_pos_w.clone()  # [N,3]
                        prism_sizes_b = self._debug_prism_size.clone().clamp(min=1e-6)  # [N,3]

                        # Convert center to base frame
                        quat_id = torch.zeros((N, 4), device=self.device, dtype=base_pos_w.dtype)
                        quat_id[:, 0] = 1.0
                        prism_center_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, prism_center_w, quat_id)

                        half = 0.5 * prism_sizes_b
                        mins = prism_center_b - half
                        maxs = prism_center_b + half

                        x0, y0, z0 = mins[:, 0], mins[:, 1], mins[:, 2]
                        x1, y1, z1 = maxs[:, 0], maxs[:, 1], maxs[:, 2]
                        cx = 0.5 * (x0 + x1)
                        cy = 0.5 * (y0 + y1)
                        cz = 0.5 * (z0 + z1)

                        Lx = (x1 - x0).clamp(min=1e-6)
                        Ly = (y1 - y0).clamp(min=1e-6)
                        Lz = (z1 - z0).clamp(min=1e-6)

                        # Edge thickness (meters)
                        t = float(getattr(args_cli, "grasp_prism_edge_thickness", None) or getattr(args_cli, "action_bounds_edge_thickness", None) or 0.01)
                        tt = torch.full((N,), t, device=self.device, dtype=mins.dtype)

                        centers_x = torch.stack(
                            [
                                torch.stack([cx, y0, z0], dim=1),
                                torch.stack([cx, y0, z1], dim=1),
                                torch.stack([cx, y1, z0], dim=1),
                                torch.stack([cx, y1, z1], dim=1),
                            ],
                            dim=1,
                        )
                        scales_x = torch.stack([torch.stack([Lx, tt, tt], dim=1)] * 4, dim=1)

                        centers_y = torch.stack(
                            [
                                torch.stack([x0, cy, z0], dim=1),
                                torch.stack([x0, cy, z1], dim=1),
                                torch.stack([x1, cy, z0], dim=1),
                                torch.stack([x1, cy, z1], dim=1),
                            ],
                            dim=1,
                        )
                        scales_y = torch.stack([torch.stack([tt, Ly, tt], dim=1)] * 4, dim=1)

                        centers_z = torch.stack(
                            [
                                torch.stack([x0, y0, cz], dim=1),
                                torch.stack([x0, y1, cz], dim=1),
                                torch.stack([x1, y0, cz], dim=1),
                                torch.stack([x1, y1, cz], dim=1),
                            ],
                            dim=1,
                        )
                        scales_z = torch.stack([torch.stack([tt, tt, Lz], dim=1)] * 4, dim=1)

                        centers_b = torch.cat([centers_x, centers_y, centers_z], dim=1)  # [N,12,3]
                        scales_b = torch.cat([scales_x, scales_y, scales_z], dim=1)      # [N,12,3]

                        centers_b = centers_b.reshape(N * 12, 3)
                        scales_b = scales_b.reshape(N * 12, 3)

                        base_pos_rep = base_pos_w.unsqueeze(1).expand(N, 12, 3).reshape(N * 12, 3)
                        base_quat_rep = base_quat_w.unsqueeze(1).expand(N, 12, 4).reshape(N * 12, 4)
                        centers_w = base_pos_rep + self._quat_rotate_vec_wxyz(base_quat_rep, centers_b)

                        # Hide inactive envs so the prism doesn't linger at origin.
                        active_edges = active_mask.unsqueeze(1).expand(N, 12).reshape(N * 12)
                        centers_w[~active_edges] = -1000.0
                        scales_b[~active_edges] = 1e-3

                        self._viz["grasp_prism"].visualize(
                            translations=centers_w,
                            orientations=base_quat_rep,
                            scales=scales_b,
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