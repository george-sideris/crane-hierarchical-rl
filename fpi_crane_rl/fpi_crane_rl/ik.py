"""Crane inverse kinematics.

Solves [slew, boom, stick, telescope] for a Cartesian end-effector target.
By default all four arm joints are optimized (telescope active). Construct
CraneIK with telescope_active=False to mask the telescope inactive and hold it
fixed at its seed value (~0.13 m), for when the telescope encoder is unreliable
or the joint is mechanically locked.

Joint ordering matches the PLC's RobotLinks layout (0 slew, 1 boom, 2 stick,
3 telescope, 6 grapplecarrier). Grapplecarrier yaw is handled directly by the
caller; it does not pass through IK.
"""

import numpy as np

# IKPy needs deprecated NumPy aliases.
for _alias, _real in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

# NumPy 2.x stringifies scalars as 'np.float64(1.0)' instead of '1.0'.
# IKPy's older releases call float() on stringified values during URDF or
# transform handling and choke on the new repr. Restoring legacy print mode
# brings str() output back to '1.0'.
try:
    np.set_printoptions(legacy='1.25')
except (TypeError, ValueError):
    pass

from ikpy.chain import Chain


IDX_SLEW, IDX_BOOM, IDX_STICK, IDX_TELESCOPE = 0, 1, 2, 3
IDX_GRAPPLECARRIER = 6

# Chain link order: [base, slew, boom, stick, telescope, hanger]. The telescope
# (index 4) is optimized only when it is active; when it is held static (sensor
# unreliable / mechanically locked at ~0.13 m) it is masked inactive and stays
# fixed at the seed value, so only slew/boom/stick are optimized. Note that
# locking the telescope removes the arm's reach redundancy, so IK compensates
# with larger boom/stick angles. Hanger stays active so IK still tracks the
# passive geometry.
_TELESCOPE_CHAIN_IDX = 4


def _build_active_mask(telescope_active: bool):
    mask = [False, True, True, True, False, True]
    mask[_TELESCOPE_CHAIN_IDX] = bool(telescope_active)
    return mask


# Default mask: telescope active (all 4 arm joints optimized). Callers with an
# unreliable/locked telescope pass telescope_active=False to CraneIK.
ACTIVE_MASK = _build_active_mask(telescope_active=True)

# Offset from upperpassive link (IK target frame) to grapple base (EE).
EE_OFFSET_Z = -0.415

# Arm joint limits used as a post-IK validity check.
ARM_JOINT_LIMITS = [
    (-1.74533, 1.74533),   # slew
    (-0.383972, 1.309),    # boom
    (-3.08574, 0.035),     # stick
    (0.13, 1.8),           # telescope
]


class CraneIK:
    """Inverse kinematics for the 4-DOF arm (slew, boom, stick, telescope).

    TODO: consolidate the URDF used here with fpi_crane_description. This
    chain currently uses a truncated standalone URDF rooted at the
    `basemast` link with structural joint names; the rest of the stack
    uses the functional names declared by fpi_crane_description
    (slew_joint, boom_joint, etc.) Migrating requires retargeting
    `base_elements`, verifying the chain link order, and re-validating
    EE_OFFSET_Z against the new chain ending.
    """

    def __init__(self, urdf_path, telescope_active=True):
        self.telescope_active = bool(telescope_active)
        self.chain = Chain.from_urdf_file(
            urdf_path,
            active_links_mask=_build_active_mask(self.telescope_active),
            base_elements=["basemast"],
        )

    def solve(self, xyz, seed):
        """Solve for [slew, boom, stick, telescope] reaching xyz (base frame).
        Returns the 4-joint solution, or None if no in-limits solution is found.
        """
        seed = np.asarray(seed, dtype=float).copy()
        target_upperpassive = np.asarray(xyz, dtype=float) - np.array([0.0, 0.0, EE_OFFSET_Z])
        T = np.eye(4)
        T[:3, 3] = target_upperpassive

        full_initial = np.zeros(6)
        full_initial[1] = seed[0]  # slew
        full_initial[2] = seed[1]  # boom
        full_initial[3] = seed[2]  # stick
        full_initial[4] = seed[3]  # telescope

        best_solution = None
        best_error = float("inf")
        for iteration in range(3):
            initial_pos = (full_initial.copy() if iteration == 0
                           else full_initial + np.random.normal(0, 0.05, 6))
            for i, link in enumerate(self.chain.links):
                if i == 0 or i >= len(initial_pos):
                    continue
                if link.bounds is not None:
                    lo, hi = link.bounds
                    if np.isfinite(lo) and np.isfinite(hi):
                        initial_pos[i] = np.clip(initial_pos[i], lo + 0.001, hi - 0.001)
            try:
                # Pass a plain Python list so IKPy never sees NumPy scalars.
                full = self.chain.inverse_kinematics_frame(
                    T, initial_position=[float(v) for v in initial_pos])
            except Exception:
                continue
            q = np.array([full[1], full[2], full[3], full[4]])
            if np.any(np.isnan(q)):
                continue
            full_joints_test = np.zeros(6)
            full_joints_test[1] = q[0]
            full_joints_test[2] = q[1]
            full_joints_test[3] = q[2]
            full_joints_test[4] = q[3]
            full_joints_test[5] = full[5]
            fk_test = self.chain.forward_kinematics(
                [float(v) for v in full_joints_test])[:3, 3]
            error = np.linalg.norm(fk_test - target_upperpassive)
            if error < best_error:
                best_error = error
                best_solution = q.copy()
            if error < 0.001:
                break

        if best_solution is None:
            return None
        for q_val, (q_min, q_max) in zip(best_solution, ARM_JOINT_LIMITS):
            if q_val < q_min or q_val > q_max:
                return None
        return best_solution
