"""Stuck-task fail-safe (Settings > General > Task Timeout, see
core.runner_constants.TASK_TIMEOUT_*): core.runner._wait_for_match_result's
stall check, MacroRunner.start()'s clamping, and _run_task's handling of a
"stalled" result (relaunch + resume repeat count via the existing recovery
loop -- see test_mid_task_recovery_resumes_repeat_count_instead_of_restarting
in test_runner_recovery.py for the pattern this mirrors).
"""
import threading
import time
from unittest.mock import Mock

from core import runner as runner_module
from core import runner_constants
from core.runner import MacroRunner


def _runner():
    return MacroRunner(Mock(), Mock(), Mock())


# --------------------------------------------------------------------------
# Defaults / start() clamping
# --------------------------------------------------------------------------
def test_task_last_progress_at_defaults_to_none():
    """None (not 0.0) outside a real task run -- see the check in
    _wait_for_match_result, which must never fire against an uninitialized
    baseline (e.g. Settings > Debug's Test Pre Start/Battle, which never
    goes through _run_task)."""
    runner = _runner()
    assert runner._task_last_progress_at is None
    assert runner._task_timeout_seconds == runner_constants.TASK_TIMEOUT_DEFAULT_MINUTES * 60.0


def _started_runner(monkeypatch, **start_kwargs):
    """runner.start() spawns a REAL background thread running _run -- stub
    that out to a no-op so these clamping tests don't launch one against a
    fake hwnd (join it immediately after so nothing outlives the test)."""
    runner = _runner()
    monkeypatch.setattr(runner, "_run", lambda *_a, **_k: None)
    runner.start(lambda: 123, lambda: [], **start_kwargs)
    runner._thread.join(timeout=5)
    return runner


def test_start_clamps_task_timeout_minutes_below_the_minimum(monkeypatch):
    runner = _started_runner(monkeypatch, task_timeout_minutes=1)
    assert runner._task_timeout_seconds == runner_constants.TASK_TIMEOUT_MIN_MINUTES * 60.0


def test_start_clamps_task_timeout_minutes_above_the_maximum(monkeypatch):
    runner = _started_runner(monkeypatch, task_timeout_minutes=99999)
    assert runner._task_timeout_seconds == runner_constants.TASK_TIMEOUT_MAX_MINUTES * 60.0


def test_start_falls_back_to_default_on_bad_task_timeout_value(monkeypatch):
    runner = _started_runner(monkeypatch, task_timeout_minutes="not a number")
    assert runner._task_timeout_seconds == runner_constants.TASK_TIMEOUT_DEFAULT_MINUTES * 60.0


def test_start_stores_a_valid_task_timeout_minutes(monkeypatch):
    runner = _started_runner(monkeypatch, task_timeout_minutes=90)
    assert runner._task_timeout_seconds == 90 * 60.0


# --------------------------------------------------------------------------
# _wait_for_match_result: the stall check itself
# --------------------------------------------------------------------------
def test_returns_stalled_once_the_timeout_elapses(monkeypatch):
    runner = _runner()
    runner._task_timeout_seconds = 60.0
    runner._task_last_progress_at = time.monotonic() - 61.0

    result = runner._wait_for_match_result(123, threading.Event(), task={})

    assert result == "stalled"


def test_does_not_stall_before_the_timeout_elapses(monkeypatch):
    runner = _runner()
    runner._task_timeout_seconds = 3600.0
    runner._task_last_progress_at = time.monotonic()
    # Immediately stop the run so the loop exits after one poll instead of
    # actually watching for victory/defeat.
    stop_event = threading.Event()
    stop_event.set()

    result = runner._wait_for_match_result(123, stop_event, task={})

    assert result is None


def test_never_stalls_when_progress_was_never_tracked(monkeypatch):
    """The Settings > Debug battle test (start_debug_test) never goes
    through _run_task, so _task_last_progress_at stays None -- the stall
    check must be a no-op rather than comparing against an unset baseline."""
    runner = _runner()
    assert runner._task_last_progress_at is None
    stop_event = threading.Event()
    stop_event.set()

    result = runner._wait_for_match_result(123, stop_event, task={})

    assert result is None


# --------------------------------------------------------------------------
# _run_task: handling a "stalled" result from _play_one_match
# --------------------------------------------------------------------------
def test_stalled_result_relaunches_and_resumes_repeat_count(monkeypatch):
    """Mirrors test_mid_task_recovery_resumes_repeat_count_instead_of_
    restarting in test_runner_recovery.py: a stall on one repeat must not
    reset the whole task's repeat count back to 1 on the resumed attempt."""
    runner = _runner()
    runner._stop_event = threading.Event()
    monkeypatch.setattr(runner, "_checkpoint", lambda _stop: False)
    monkeypatch.setattr(runner, "_run_task_setup", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_challenge_has_ready_stage", lambda: False)
    monkeypatch.setattr(runner, "_crafting_wants_in", lambda *_a, **_k: False)
    monkeypatch.setattr(runner, "_fuel_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_auto_shop_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_wait_teleport_in", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_recover_to_lobby", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_handle_match_result", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_current_hwnd", 123)
    monkeypatch.setattr(runner_module.wm, "is_window", lambda _hwnd: True)
    rejoin = Mock(return_value=True)
    monkeypatch.setattr(runner, "_attempt_rejoin", rejoin)

    calls = {"n": 0}

    def play_one_match(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 3:
            return "stalled"
        return "win"

    monkeypatch.setattr(runner, "_play_one_match", play_one_match)
    task = {"mode": "story", "map": "King's Tomb", "stage": "1", "repeat": 5, "play_mode": "solo"}

    completed = runner._run_task(123, runner._stop_event, task, 1, 1, {}, 3, 8, {}, {})

    assert completed is True
    rejoin.assert_called_once_with(123, runner._stop_event)
    # 5 configured repeats: 1, 2, 3(stalls), resume at 3, 4, 5 -- 6 total
    # _play_one_match calls. A bug that restarted the count from 1 would
    # land on 8 instead (2 successes + 1 stall, then 5 more from scratch).
    assert calls["n"] == 6


def test_stalled_result_resets_the_progress_clock_before_retrying(monkeypatch):
    runner = _runner()
    runner._stop_event = threading.Event()
    monkeypatch.setattr(runner, "_checkpoint", lambda _stop: False)
    monkeypatch.setattr(runner, "_run_task_setup", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_challenge_has_ready_stage", lambda: False)
    monkeypatch.setattr(runner, "_crafting_wants_in", lambda *_a, **_k: False)
    monkeypatch.setattr(runner, "_fuel_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_auto_shop_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_recover_to_lobby", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_handle_match_result", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_current_hwnd", 123)
    monkeypatch.setattr(runner_module.wm, "is_window", lambda _hwnd: True)
    monkeypatch.setattr(runner, "_attempt_rejoin", Mock(return_value=True))

    calls = {"n": 0}

    def play_one_match(*_a, **_k):
        calls["n"] += 1
        return "stalled" if calls["n"] == 1 else "win"

    monkeypatch.setattr(runner, "_play_one_match", play_one_match)
    runner._task_last_progress_at = 0.0  # stale on purpose
    task = {"mode": "story", "map": "King's Tomb", "stage": "1", "repeat": 1, "play_mode": "solo"}

    runner._run_task(123, runner._stop_event, task, 1, 1, {}, 3, 8, {}, {})

    assert runner._task_last_progress_at > 0.0


def test_stalled_result_sends_a_distinct_event_webhook(monkeypatch):
    runner = _runner()
    runner._stop_event = threading.Event()
    monkeypatch.setattr(runner, "_checkpoint", lambda _stop: False)
    monkeypatch.setattr(runner, "_run_task_setup", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_challenge_has_ready_stage", lambda: False)
    monkeypatch.setattr(runner, "_crafting_wants_in", lambda *_a, **_k: False)
    monkeypatch.setattr(runner, "_fuel_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_auto_shop_wants_in", lambda: False)
    monkeypatch.setattr(runner, "_recover_to_lobby", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_handle_match_result", lambda *_a, **_k: True)
    monkeypatch.setattr(runner, "_current_hwnd", 123)
    monkeypatch.setattr(runner_module.wm, "is_window", lambda _hwnd: True)
    monkeypatch.setattr(runner, "_attempt_rejoin", Mock(return_value=True))
    webhook_events = Mock()
    monkeypatch.setattr(runner, "_send_event_webhook", webhook_events)

    calls = {"n": 0}

    def play_one_match(*_a, **_k):
        calls["n"] += 1
        return "stalled" if calls["n"] == 1 else "win"

    monkeypatch.setattr(runner, "_play_one_match", play_one_match)
    task = {"mode": "story", "map": "King's Tomb", "stage": "1", "repeat": 1, "play_mode": "solo"}

    runner._run_task(123, runner._stop_event, task, 1, 1, {}, 3, 8, {}, {})

    webhook_events.assert_called_once()
    args = webhook_events.call_args.args
    assert args[2] == "Task Stalled -- Restarting Roblox"


def test_stop_event_during_stall_returns_false_without_relaunching(monkeypatch):
    runner = _runner()
    stop_event = threading.Event()
    monkeypatch.setattr(runner, "_checkpoint", lambda _stop: False)
    monkeypatch.setattr(runner, "_run_task_setup", lambda *_a, **_k: True)
    rejoin = Mock()
    monkeypatch.setattr(runner, "_attempt_rejoin", rejoin)

    def play_one_match(*_a, **_k):
        stop_event.set()
        return "stalled"

    monkeypatch.setattr(runner, "_play_one_match", play_one_match)
    task = {"mode": "story", "map": "King's Tomb", "stage": "1", "repeat": 1, "play_mode": "solo"}

    completed = runner._run_task(123, stop_event, task, 1, 1, {}, 3, 8, {}, {})

    assert completed is False
    rejoin.assert_not_called()
