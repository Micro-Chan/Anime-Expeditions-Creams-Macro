"""Stat Farm (core.runner_statfarm): progress-state helpers, the round-robin
scheduler, worthiness hover-check, the reroll flow's per-unit roll loop, the
_wait_for_match_result stall-check gating it adds, the shared
_click_restart_via_settings helper it factors out of the End Run block, and
the task-level orchestrator.
"""
import threading
from unittest.mock import MagicMock, Mock, call

import pytest

from core import constants
from core import runner_statfarm as sf
from core.runner import MacroRunner


def _runner():
    return MacroRunner(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture(autouse=True)
def _isolated_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "APP_DIR", str(tmp_path))


# --------------------------------------------------------------------------
# Progress-state helpers
# --------------------------------------------------------------------------
def test_reset_writes_an_empty_file_even_after_prior_progress():
    sf.mark_done(sf.load_statfarm_progress(), 3, 2)  # no-op on a throwaway dict
    data = {"3": [True, False, False, False, False]}
    sf.save_statfarm_progress(data)
    assert sf.load_statfarm_progress() == data

    sf.reset_statfarm_progress()

    assert sf.load_statfarm_progress() == {}


def test_load_returns_empty_dict_when_no_file_exists_yet():
    assert sf.load_statfarm_progress() == {}


def test_mark_done_creates_the_loadout_entry_on_first_use():
    data = {}
    sf.mark_done(data, 5, 2)
    assert data == {"5": [False, False, True, False, False]}


def test_done_count_and_is_fully_done():
    data = {"1": [True, True, False, False, False]}
    assert sf.done_count(data, 1) == 2
    assert sf.done_count(data, 2) == 0  # never touched -- treated as 0, not an error
    assert sf.is_fully_done(data, 1) is False

    for i in range(5):
        sf.mark_done(data, 1, i)
    assert sf.done_count(data, 1) == 5
    assert sf.is_fully_done(data, 1) is True


def test_not_done_positions_skips_marked_ones():
    data = {"1": [True, False, True, False, False]}
    assert sf.not_done_positions(data, 1) == [1, 3, 4]
    assert sf.not_done_positions(data, 9) == [0, 1, 2, 3, 4]  # untouched loadout -- all 5 open


# --------------------------------------------------------------------------
# _stat_farm_pick_next_loadout: round-robin scheduling
# --------------------------------------------------------------------------
def test_picks_the_loadout_with_the_fewest_filter_hits():
    runner = _runner()
    progress = {"1": [True, True, False, False, False], "2": [False] * 5}
    assert runner._stat_farm_pick_next_loadout([1, 2], progress) == 2


def test_ties_break_by_loadout_number_ascending():
    runner = _runner()
    progress = {"3": [False] * 5, "5": [False] * 5}
    assert runner._stat_farm_pick_next_loadout([5, 3], progress) == 3


def test_a_fully_done_loadout_is_skipped_in_favor_of_one_behind():
    runner = _runner()
    progress = {"1": [True] * 5, "2": [True, False, False, False, False]}
    assert runner._stat_farm_pick_next_loadout([1, 2], progress) == 2


def test_returns_none_once_every_selected_loadout_is_fully_done():
    runner = _runner()
    progress = {"1": [True] * 5, "2": [True] * 5}
    assert runner._stat_farm_pick_next_loadout([1, 2], progress) is None


# --------------------------------------------------------------------------
# _stat_farm_check_worthiness
# --------------------------------------------------------------------------
def test_worthiness_check_true_when_every_position_shows_400(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(sf.vision, "find_image", lambda *_a, **_k: {"score": 0.99})
    assert runner._stat_farm_check_worthiness(123, threading.Event(), [0, 1, 2]) is True


def test_worthiness_check_double_clicks_each_hovered_position(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(sf.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    monkeypatch.setattr(sf.vision, "find_image", lambda *_a, **_k: {"score": 0.99})

    runner._stat_farm_check_worthiness(123, threading.Event(), [0, 2])

    assert runner._mouse.double_click.call_args_list == [
        call(*sf.STATFARM_HOTBAR_HOVER[0]),
        call(*sf.STATFARM_HOTBAR_HOVER[2]),
    ]


def test_worthiness_check_false_on_the_first_miss(monkeypatch):
    runner = _runner()
    calls = []

    def find_image(_hwnd, name):
        calls.append(name)
        return None if len(calls) == 2 else {"score": 0.9}

    monkeypatch.setattr(sf.vision, "find_image", find_image)
    assert runner._stat_farm_check_worthiness(123, threading.Event(), [0, 1, 2]) is False
    assert len(calls) == 2  # stopped checking after the miss, didn't check position 2


def test_worthiness_check_skips_already_done_positions(monkeypatch):
    runner = _runner()
    checked = []
    monkeypatch.setattr(sf.vision, "find_image", lambda *_a, **_k: checked.append(1) or {"score": 0.9})
    runner._stat_farm_check_worthiness(123, threading.Event(), [2])  # only position 2 passed in
    assert len(checked) == 1


def test_worthiness_check_stops_on_a_checkpoint_stop(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(sf.vision, "find_image", lambda *_a, **_k: pytest.fail("should not search after stop"))
    assert runner._stat_farm_check_worthiness(123, stop, [0]) is False


# --------------------------------------------------------------------------
# _stat_farm_check_or_restart
# --------------------------------------------------------------------------
def _prep_check_or_restart(runner, monkeypatch, wave, interval=10, target=150):
    monkeypatch.setattr(sf.time, "time", lambda: 100.0)
    monkeypatch.setattr(sf.vision, "capture_window_region_bgr", lambda *_a, **_k: object())
    monkeypatch.setattr(sf.wave_module, "read_wave", lambda *_a, **_k: (wave, None))
    runner._statfarm_wave_state = {}
    runner._statfarm_check_interval = interval
    runner._statfarm_wave_target = target
    runner._statfarm_active_loadout = 1


def test_not_a_check_interval_wave_takes_no_action(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=7)
    runner._stat_farm_check_worthiness = Mock()
    assert runner._stat_farm_check_or_restart(123, threading.Event()) is None
    runner._stat_farm_check_worthiness.assert_not_called()


def test_worthy_at_a_boundary_leaves_to_the_lobby(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=10)
    runner._stat_farm_check_worthiness = Mock(return_value=True)
    leave = Mock(return_value=True)
    runner._leave_infinite_at_wave_limit = leave

    assert runner._stat_farm_check_or_restart(123, threading.Event()) == "statfarm_worthy"
    leave.assert_called_once()


def test_not_worthy_below_wave_target_keeps_playing(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=10, target=150)
    runner._stat_farm_check_worthiness = Mock(return_value=False)
    runner._click_restart_via_settings = Mock()

    assert runner._stat_farm_check_or_restart(123, threading.Event()) is None
    runner._click_restart_via_settings.assert_not_called()


def test_not_worthy_at_wave_target_restarts_in_place(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=150, target=150)
    runner._stat_farm_check_worthiness = Mock(return_value=False)
    restart = Mock(return_value=True)
    runner._click_restart_via_settings = restart

    assert runner._stat_farm_check_or_restart(123, threading.Event()) == "statfarm_restarted"
    restart.assert_called_once()


def test_failed_restart_click_returns_none_not_restarted(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=150, target=150)
    runner._stat_farm_check_worthiness = Mock(return_value=False)
    runner._click_restart_via_settings = Mock(return_value=False)

    assert runner._stat_farm_check_or_restart(123, threading.Event()) is None


def test_same_wave_number_only_acts_once(monkeypatch):
    runner = _runner()
    _prep_check_or_restart(runner, monkeypatch, wave=10)
    runner._stat_farm_check_worthiness = Mock(return_value=False)
    runner._click_restart_via_settings = Mock(return_value=False)

    runner._stat_farm_check_or_restart(123, threading.Event())
    runner._statfarm_wave_state["next_check"] = 0.0  # allow another poll through the throttle
    runner._stat_farm_check_or_restart(123, threading.Event())

    assert runner._stat_farm_check_worthiness.call_count == 1


# --------------------------------------------------------------------------
# _click_restart_via_settings (shared helper, extracted from the End Run block)
# --------------------------------------------------------------------------
def test_click_restart_via_settings_full_sequence(monkeypatch):
    runner = _runner()
    import core.runner as runner_module
    monkeypatch.setattr(runner_module.time, "sleep", lambda *_a, **_k: None)
    clicks = []
    runner._click_found_image = lambda hwnd, name, timeout, stop: clicks.append(name) or {"score": 0.9}

    assert runner._click_restart_via_settings(123, threading.Event(), "Test") is True
    assert clicks == ["nav_settings", "restart_btn", "restart_btn"]


def test_click_restart_via_settings_stops_on_first_missing_image(monkeypatch):
    runner = _runner()
    import core.runner as runner_module
    monkeypatch.setattr(runner_module.time, "sleep", lambda *_a, **_k: None)
    runner._click_found_image = lambda *_a, **_k: None

    assert runner._click_restart_via_settings(123, threading.Event(), "Test") is False


def test_end_run_block_still_sets_battle_end_run_requested_after_refactor(monkeypatch):
    """Regression: _run_end_run_tick must keep its existing win-recording
    side effect unchanged after the 3-click sequence moved into the shared
    _click_restart_via_settings helper."""
    runner = _runner()
    runner._click_restart_via_settings = Mock(return_value=True)
    runner._battle_end_run_requested = False

    runner._run_end_run_tick(123, threading.Event(), {}, 1)

    assert runner._battle_end_run_requested is True


def test_end_run_block_does_not_set_flag_when_restart_click_fails():
    runner = _runner()
    runner._click_restart_via_settings = Mock(return_value=False)
    runner._battle_end_run_requested = False

    runner._run_end_run_tick(123, threading.Event(), {}, 1)

    assert runner._battle_end_run_requested is False


# --------------------------------------------------------------------------
# _wait_for_match_result gating: _check_infinite_wave_limit must be skipped
# while a Stat Farm grind is active, and untouched otherwise.
# --------------------------------------------------------------------------
def test_check_infinite_wave_limit_is_skipped_while_statfarm_active(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    task = {"mode": "story", "stage": "Infinite", "infinite_wave_limit": 20}
    runner._checkpoint = Mock(side_effect=[False, True])  # one poll, then stop
    runner._statfarm_active = True
    runner._stat_farm_check_or_restart = Mock(return_value=None)
    check_limit = Mock(return_value=None)
    runner._check_infinite_wave_limit = check_limit
    runner._run_battle_blocks_tick = Mock()
    runner._tick_loop_phases = Mock()

    runner._wait_for_match_result(123, stop, task=task)

    check_limit.assert_not_called()
    runner._stat_farm_check_or_restart.assert_called_once()


def test_check_infinite_wave_limit_still_runs_when_statfarm_inactive(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    task = {"mode": "story", "stage": "Infinite", "infinite_wave_limit": 20}
    runner._checkpoint = Mock(side_effect=[False, True])
    assert runner._statfarm_active is False
    check_limit = Mock(return_value=None)
    runner._check_infinite_wave_limit = check_limit
    runner._run_battle_blocks_tick = Mock()
    runner._tick_loop_phases = Mock()

    runner._wait_for_match_result(123, stop, task=task)

    check_limit.assert_called_once()


# --------------------------------------------------------------------------
# _apply_team_loadout is a no-op while Stat Farm is driving the loadout
# --------------------------------------------------------------------------
def test_apply_team_loadout_is_a_noop_while_statfarm_active():
    runner = _runner()
    runner._statfarm_active = True
    panel = Mock()
    runner._apply_team_loadout_panel = panel

    assert runner._apply_team_loadout(123, threading.Event(), {"macro": "Farm", "team": "3"}) is True
    panel.assert_not_called()


# --------------------------------------------------------------------------
# _stat_farm_reroll_one_unit: filter-hit vs. exhausted-4-rolls
# --------------------------------------------------------------------------
def test_reroll_stops_and_marks_done_on_a_filter_hit(monkeypatch):
    runner = _runner()
    progress = {}
    monkeypatch.setattr(sf.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    reroll_clicks = []
    runner._click_found_image = lambda hwnd, name, timeout, stop: reroll_clicks.append(name) or {"score": 0.9}
    # Filter hits on the 2nd roll.
    hits = iter([None, {"score": 0.9}])
    monkeypatch.setattr(sf.vision, "wait_for_image", lambda *_a, **_k: next(hits))

    hit = runner._stat_farm_reroll_one_unit(123, threading.Event(), 1, 2, progress)

    assert hit is True
    assert reroll_clicks.count("reroll_btn") == 2
    assert progress == {"1": [False, False, True, False, False]}


def test_reroll_exhausts_all_attempts_without_a_hit(monkeypatch):
    runner = _runner()
    progress = {}
    monkeypatch.setattr(sf.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    runner._click_found_image = lambda hwnd, name, timeout, stop: {"score": 0.9}
    monkeypatch.setattr(sf.vision, "wait_for_image", lambda *_a, **_k: None)

    hit = runner._stat_farm_reroll_one_unit(123, threading.Event(), 1, 0, progress)

    assert hit is False
    assert progress == {}  # never marked done


def test_reroll_stops_immediately_if_reroll_btn_is_never_found(monkeypatch):
    runner = _runner()
    progress = {}
    monkeypatch.setattr(sf.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    runner._click_found_image = lambda *_a, **_k: None

    hit = runner._stat_farm_reroll_one_unit(123, threading.Event(), 1, 0, progress)

    assert hit is False


# --------------------------------------------------------------------------
# _run_stat_farm_task: end-to-end orchestration (heavily mocked)
# --------------------------------------------------------------------------
def _stub_orchestrator(runner, monkeypatch):
    runner._run_task_setup = Mock(return_value=True)
    runner._apply_team_loadout_explicit = Mock(return_value=True)
    runner._wait_teleport_in = Mock(return_value=True)
    runner._stat_farm_run_reroll_flow = Mock()
    runner._checkpoint = Mock(return_value=False)


def test_no_loadouts_selected_finishes_immediately_without_entering():
    runner = _runner()
    runner._run_task_setup = Mock()
    task = {"map": "School Grounds", "stat_farm_loadouts": []}

    completed = runner._run_stat_farm_task(123, threading.Event(), task, 1, 1, {}, None, None, {}, {})

    assert completed is True
    runner._run_task_setup.assert_not_called()


def test_orchestrator_round_robins_and_finishes_once_all_loadouts_done(monkeypatch):
    runner = _runner()
    _stub_orchestrator(runner, monkeypatch)
    task = {"map": "School Grounds", "stat_farm_loadouts": [1, 2],
            "stat_farm_wave_target": 150, "stat_farm_check_interval": 10}

    # Each loadout needs 2 "worthy" Infinite passes (reroll flow marks 2-3
    # more units done per visit via the mocked side effect below) before
    # being fully done. Track calls to prove the round-robin alternates
    # rather than draining loadout 1 all the way before touching loadout 2.
    play_calls = {"n": 0}
    applied_loadouts = []

    def fake_apply(_hwnd, _stop, loadout_num, *_a, **_k):
        applied_loadouts.append(loadout_num)
        return True
    runner._apply_team_loadout_explicit = fake_apply

    def fake_play(*_a, **_k):
        play_calls["n"] += 1
        return "statfarm_worthy"
    runner._play_one_match = fake_play

    progress_state = {"1": [False] * 5, "2": [False] * 5}

    def fake_reroll(_hwnd, _stop, loadout_num):
        # Mark 3 of the active loadout's units done per reroll visit.
        done = progress_state[str(loadout_num)]
        marked = 0
        for i in range(5):
            if not done[i] and marked < 3:
                done[i] = True
                marked += 1
        sf.save_statfarm_progress(progress_state)
    runner._stat_farm_run_reroll_flow = fake_reroll
    sf.save_statfarm_progress(progress_state)

    completed = runner._run_stat_farm_task(123, threading.Event(), task, 1, 1, {}, None, None, {}, {})

    assert completed is True
    assert sf.load_statfarm_progress() == {"1": [True] * 5, "2": [True] * 5}
    # Alternated rather than draining loadout 1 first: 1, 2, 1, 2 (3+2, 3+2).
    assert applied_loadouts == [1, 2, 1, 2]


def test_orchestrator_restarts_in_place_without_reapplying_loadout(monkeypatch):
    runner = _runner()
    _stub_orchestrator(runner, monkeypatch)
    task = {"map": "School Grounds", "stat_farm_loadouts": [1]}

    apply_calls = []
    runner._apply_team_loadout_explicit = lambda *a, **k: apply_calls.append(a) or True
    results = iter(["statfarm_restarted", "statfarm_worthy"])
    runner._play_one_match = lambda *_a, **_k: next(results)

    def fake_reroll(_hwnd, _stop, loadout_num):
        sf.save_statfarm_progress({"1": [True] * 5})
    runner._stat_farm_run_reroll_flow = fake_reroll

    completed = runner._run_stat_farm_task(123, threading.Event(), task, 1, 1, {}, None, None, {}, {})

    assert completed is True
    assert len(apply_calls) == 1  # never re-applied after the in-place restart
    runner._wait_teleport_in.assert_called_once()


def test_stop_event_during_the_grind_returns_false():
    runner = _runner()
    stop = threading.Event()
    runner._run_task_setup = Mock(return_value=True)
    runner._apply_team_loadout_explicit = Mock(return_value=True)

    def fake_play(*_a, **_k):
        stop.set()
        return None
    runner._play_one_match = fake_play
    task = {"map": "School Grounds", "stat_farm_loadouts": [1]}

    completed = runner._run_stat_farm_task(123, stop, task, 1, 1, {}, None, None, {}, {})

    assert completed is False
