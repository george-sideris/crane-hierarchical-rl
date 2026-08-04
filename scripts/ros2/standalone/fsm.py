"""Standalone FSM for crane pick-and-place cycle.

Pure control logic with no physics or simulation dependency. Computes EE
targets and gripper commands for each phase.
"""

import math
from dataclasses import dataclass, field
from enum import IntEnum


class Phase(IntEnum):
    HOVER_UP = 0
    ALIGN_YAW = 1
    DESCEND = 2
    CLOSE = 3
    LIFT_HIGH = 4
    CARRY_HOME = 5
    ALIGN_HOME_YAW = 6
    LOWER_TO_DROP = 7
    OPEN = 8
    SETTLE = 9
    CLEAR = 10  # lift back up over the trailer before the next cycle


PHASE_NAMES = {p: p.name for p in Phase}


@dataclass
class FSMConfig:
    """FSM tuning parameters (defaults match simulation)."""
    hover_clear: float = 2.5           # meters above target for hover
    approach_above: float = 0.6        # meters above target for approach
    grip_open: float = 0.05            # gripper open position (rad)
    grip_closed: float = 2.23          # gripper closed position (rad)
    gripper_step: float = 0.15         # discrete gripper step (rad)
    ee_tolerance: float = 0.05         # EE position tolerance (m)
    bg_tolerance: float = 0.25         # basegrapple tolerance (m)
    yaw_tolerance: float = 0.10        # yaw tolerance (rad)
    dwell_count: int = 10              # frames within tolerance before transition (1s at 10Hz)
    phase_timeout: int = 100           # general phase timeout (10s at 10Hz)
    descend_timeout: int = 100         # descent phase timeout (10s at 10Hz)
    gripper_timeout: int = 60          # gripper close timeout (6s at 10Hz)
    settle_frames: int = 15            # settle wait after open (frames)
    # Home position in crane base frame. Z bumped to 3.0 to give about
    # 1 m clearance over the trailer surface, keeping the EE away from
    # trailer poles during deposition.
    drop_position: list = field(default_factory=lambda: [0.0, 2.2, 3.0])
    drop_yaw: float = 0.0             # yaw at drop position (rad)
    ee_to_grapple_offset_z: float = 0.415  # vertical offset from grapple to EE


@dataclass
class EECommand:
    """Output command for one FSM tick."""
    position: list      # [x, y, z] target EE position in base frame
    yaw: float          # target yaw (rad)
    gripper: float      # gripper opening (0.05=open, 2.23=closed)
    phase: Phase        # current FSM phase
    cycle_complete: bool = False  # true when full cycle finished (ready for new target)


class CraneFSM:
    """10-phase pick-and-place FSM.

    Usage:
        fsm = CraneFSM()
        fsm.set_target(x, y, z, yaw)  # from policy
        while not done:
            cmd = fsm.tick(current_ee_pos, current_ee_yaw, current_gripper)
            publish(cmd)
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or FSMConfig()
        self.phase = Phase.HOVER_UP
        self.dwell = 0
        self.phase_timer = 0
        self.gripper_target = self.cfg.grip_open
        self.gripper_stepping = False
        self.gripper_step_count = 0
        self.gripper_stability_count = 0
        self.cycle_count = 0

        # Target set by policy
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0

        # Cached positions
        self._hover_pos = [0.0, 0.0, 0.0]
        self._approach_pos = [0.0, 0.0, 0.0]

    def set_target(self, x: float, y: float, z: float, yaw: float):
        """Set the grasp target from policy output (base frame coordinates)."""
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw

        # Precompute phase positions
        self._hover_pos = [x, y, z + self.cfg.hover_clear]
        self._approach_pos = [x, y, z + self.cfg.approach_above]

        # Reset FSM to start of cycle
        self.phase = Phase.HOVER_UP
        self.dwell = 0
        self.phase_timer = 0
        self.gripper_target = self.cfg.grip_open
        self.gripper_stepping = False

    def tick(self, ee_pos: list, ee_yaw: float, gripper_pos: float) -> EECommand:
        """Advance FSM by one tick given current state feedback.

        Args:
            ee_pos: [x, y, z] current EE position in base frame.
            ee_yaw: current yaw angle (rad).
            gripper_pos: current gripper opening (rad).

        Returns:
            EECommand with target position, yaw, gripper, and phase info.
        """
        self.phase_timer += 1
        cycle_complete = False

        if self.phase == Phase.HOVER_UP:
            target_pos = self._hover_pos
            target_yaw = ee_yaw  # hold current yaw during hover
            grip = self.cfg.grip_open
            if self._check_position(ee_pos, target_pos) and self._check_tolerance(ee_pos, target_pos, self.cfg.bg_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self._advance(Phase.ALIGN_YAW)

        elif self.phase == Phase.ALIGN_YAW:
            target_pos = self._hover_pos
            target_yaw = self.target_yaw
            grip = self.cfg.grip_open
            yaw_ok = abs(self._wrap_angle(ee_yaw - target_yaw)) < self.cfg.yaw_tolerance
            pos_ok = self._check_tolerance(ee_pos, target_pos, self.cfg.bg_tolerance)
            if yaw_ok and pos_ok:
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self._advance(Phase.DESCEND)

        elif self.phase == Phase.DESCEND:
            target_pos = self._approach_pos
            target_yaw = self.target_yaw
            grip = self.cfg.grip_open
            if self._check_tolerance(ee_pos, target_pos, self.cfg.ee_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.descend_timeout:
                self._advance(Phase.CLOSE)

        elif self.phase == Phase.CLOSE:
            target_pos = self._approach_pos  # hold position
            target_yaw = self.target_yaw
            # Command grip_closed in one shot, let PD drive it there.
            self.gripper_target = self.cfg.grip_closed
            grip = self.gripper_target
            if abs(gripper_pos - self.gripper_target) < 0.1:
                self.gripper_stability_count += 1
            else:
                self.gripper_stability_count = 0
            if self.gripper_stability_count >= 25 or self.phase_timer >= self.cfg.gripper_timeout:
                self._advance(Phase.LIFT_HIGH)

        elif self.phase == Phase.LIFT_HIGH:
            target_pos = self._hover_pos
            target_yaw = self.target_yaw
            grip = self.gripper_target  # keep closed
            if self._check_tolerance(ee_pos, target_pos, self.cfg.ee_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self._advance(Phase.CARRY_HOME)

        elif self.phase == Phase.CARRY_HOME:
            # Carry at the fixed drop altitude rather than the variable
            # hover altitude, so the EE arrives at the drop pose ready
            # to open.
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]
            target_yaw = self.cfg.drop_yaw
            grip = self.gripper_target
            if self._check_tolerance(ee_pos, target_pos, self.cfg.bg_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self._advance(Phase.ALIGN_HOME_YAW)

        elif self.phase == Phase.ALIGN_HOME_YAW:
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]  # at home altitude
            target_yaw = self.cfg.drop_yaw
            grip = self.gripper_target
            yaw_ok = abs(self._wrap_angle(ee_yaw - target_yaw)) < self.cfg.yaw_tolerance
            pos_ok = self._check_tolerance(ee_pos, target_pos, self.cfg.bg_tolerance)
            if yaw_ok and pos_ok:
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                # Temporary: skip LOWER_TO_DROP and open at the home
                # altitude. Proper deposit Z should be computed from the
                # highest log Z observed in the stick camera once the
                # stick camera is wired in.
                self._advance(Phase.OPEN)

        elif self.phase == Phase.LOWER_TO_DROP:
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]
            target_yaw = self.cfg.drop_yaw
            grip = self.gripper_target
            if self._check_tolerance(ee_pos, target_pos, self.cfg.ee_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self._advance(Phase.OPEN)

        elif self.phase == Phase.OPEN:
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]
            target_yaw = self.cfg.drop_yaw
            grip = self.cfg.grip_open
            self.gripper_target = self.cfg.grip_open
            # Immediate transition
            self._advance(Phase.SETTLE)

        elif self.phase == Phase.SETTLE:
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]
            target_yaw = self.cfg.drop_yaw
            grip = self.cfg.grip_open
            if self.phase_timer >= self.cfg.settle_frames:
                self._advance(Phase.CLEAR)

        elif self.phase == Phase.CLEAR:
            # Stay at the drop altitude. The next cycle's HOVER_UP will
            # reposition to the new log's hover above.
            drop = self.cfg.drop_position
            target_pos = [drop[0], drop[1], drop[2]]
            target_yaw = self.cfg.drop_yaw
            grip = self.cfg.grip_open
            if self._check_tolerance(ee_pos, target_pos, self.cfg.bg_tolerance):
                self.dwell += 1
            else:
                self.dwell = 0
            if self.dwell >= self.cfg.dwell_count or self.phase_timer >= self.cfg.phase_timeout:
                self.cycle_count += 1
                cycle_complete = True
                # Reset to HOVER_UP. Caller should call set_target() with new policy output.
                self.phase = Phase.HOVER_UP
                self.dwell = 0
                self.phase_timer = 0

        # cmd_pos is in basegrapple frame. jv_controller's ik_xyz already
        # adds the upperpassive offset internally; adding it here would
        # double-count and put the crane about 0.415 m too high.
        cmd_pos = list(target_pos)

        return EECommand(
            position=cmd_pos,
            yaw=target_yaw,
            gripper=grip,
            phase=self.phase,
            cycle_complete=cycle_complete,
        )

    def _advance(self, next_phase: Phase):
        self.phase = next_phase
        self.dwell = 0
        self.phase_timer = 0

    def _check_position(self, current: list, target: list) -> bool:
        return self._check_tolerance(current, target, self.cfg.ee_tolerance)

    @staticmethod
    def _check_tolerance(current: list, target: list, tol: float) -> bool:
        dx = current[0] - target[0]
        dy = current[1] - target[1]
        dz = current[2] - target[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz) < tol

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
