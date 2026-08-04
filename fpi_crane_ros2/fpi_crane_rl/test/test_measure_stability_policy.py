import sys
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

from fpi_crane_msgs.action import MeasureStability

sys.modules.setdefault("fpi_crane_rl.ik", SimpleNamespace(CraneIK=object))

from fpi_crane_rl.crane_policy_node import (  # noqa: E402
    SEQ_SEND,
    SEQ_WAIT_STABILITY,
    CranePolicyNode,
    Move,
)
from fpi_crane_rl.fsm import Phase  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now_value = FakeTime(1_000_000_000)

    def now(self):
        return self.now_value


class FakeTime:
    def __init__(self, ns):
        self.nanoseconds = ns

    def to_msg(self):
        msg = Time()
        msg.sec = self.nanoseconds // 1_000_000_000
        msg.nanosec = self.nanoseconds % 1_000_000_000
        return msg

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


class DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self):
        return True

    def result(self):
        return self._result


class PendingFuture:
    def done(self):
        return False


class FakeGoalHandle:
    def __init__(self, accepted=True, result=None, status=GoalStatus.STATUS_SUCCEEDED):
        self.accepted = accepted
        self.cancel_count = 0
        if result is None:
            result = MeasureStability.Result()
            result.success = True
        self._wrapped = SimpleNamespace(status=status, result=result)

    def get_result_async(self):
        return DoneFuture(self._wrapped)

    def cancel_goal_async(self):
        self.cancel_count += 1
        return DoneFuture(None)


class FakeActionClient:
    def __init__(self, future):
        self.future = future
        self.goals = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        return self.future


def make_node():
    node = CranePolicyNode.__new__(CranePolicyNode)
    node._measure_stability_enabled = True
    node._measure_stability_timeout_s = 20.0
    node._measure_stability_required_window_s = 1.0
    node._measure_stability_required_dwell_s = 0.5
    node._measure_stability_min_samples = 20
    node._measure_stability_max_gap_s = 0.15
    node._measure_stability_stability_var = 1.0e-4
    node._measure_stability_gravity_var = 1.0e-5
    node._measure_stability_grapple_var = 1.0e-4
    node._measure_stability_reward_exponent = 4.0
    node._measure_stability_goal_response_timeout_s = 2.0
    node._measure_stability_failure_policy = "advance"
    node._stability_send_goal_future = None
    node._stability_goal_handle = None
    node._stability_result_future = None
    node._stability_goal_sent = False
    node._stability_result_handled = False
    node._stability_terminal_failure = False
    node._stability_measurement_start = None
    node._last_stability_result = None
    node._debug_save_dir = ""
    node.moves = [
        Move(Phase.LIFT_HIGH, "traj", [1, 2, 3], 0.0),
        Move(Phase.CARRY_HOME, "traj", [0, 3, 3], 1.57),
    ]
    node.move_idx = 0
    node.seq_state = "wait_traj"
    node._wait_start = FakeTime(0)
    node.cycle_count = 0
    clock = FakeClock()
    node.get_clock = lambda: clock
    node.get_logger = lambda: SimpleNamespace(
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    node._waited_longer_than = lambda timeout: False
    node._log_reach = lambda phase: None
    node._converged_to_goal = lambda: True
    node._append_stability_failure_log = lambda reason: None
    node._log_stability_success = lambda result: None
    node.advance_count = 0

    def advance():
        node.advance_count += 1
        node.move_idx += 1
        node.seq_state = SEQ_SEND

    node._advance_move = advance
    return node


def test_exactly_one_action_goal_sent_after_lift_high_reached():
    node = make_node()
    node.stability_client = FakeActionClient(PendingFuture())
    node._start_stability_measurement()
    node._start_stability_measurement()
    assert node.seq_state == SEQ_WAIT_STABILITY
    assert len(node.stability_client.goals) == 1
    goal = node.stability_client.goals[0]
    assert goal.timeout_sec == 20.0
    assert goal.required_window_sec == 1.0


def test_repeated_timer_ticks_do_not_send_duplicate_goals():
    node = make_node()
    node.stability_client = FakeActionClient(PendingFuture())
    node._start_stability_measurement()
    node._check_stability_done()
    node._check_stability_done()
    assert len(node.stability_client.goals) == 1


def test_advance_not_called_and_carry_home_not_sent_while_active():
    node = make_node()
    node.stability_client = FakeActionClient(PendingFuture())
    node._start_stability_measurement()
    node._check_stability_done()
    assert node.advance_count == 0
    assert node.move_idx == 0
    assert node.moves[node.move_idx].phase == Phase.LIFT_HIGH


def test_successful_action_completion_advances_exactly_once():
    result = MeasureStability.Result()
    result.success = True
    goal_handle = FakeGoalHandle(result=result)
    node = make_node()
    node.stability_client = FakeActionClient(DoneFuture(goal_handle))
    node._start_stability_measurement()
    node._check_stability_done()
    node._check_stability_done()
    assert node.advance_count == 1
    assert node.move_idx == 1


def test_reset_or_shutdown_cancels_active_goal():
    node = make_node()
    goal_handle = FakeGoalHandle()
    node._stability_goal_handle = goal_handle
    node._cancel_stability_goal("test")
    assert goal_handle.cancel_count == 1


def test_failure_policy_advance_advances_with_invalid_measurement():
    node = make_node()
    node._measure_stability_failure_policy = "advance"
    node._handle_stability_failure("rejected")
    assert node.advance_count == 1


def test_failure_policy_hold_remains_safely_held():
    node = make_node()
    node._measure_stability_failure_policy = "hold"
    node._handle_stability_failure("aborted")
    assert node.advance_count == 0
    assert node.seq_state == SEQ_WAIT_STABILITY
    assert node._stability_terminal_failure
