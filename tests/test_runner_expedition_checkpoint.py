"""At Checkpoint's idle/holding/released handoff with the Expedition
checkpoint search (core.runner_expedition._check_expedition_wave_result /
_check_expedition_checkpoint_by_color) -- see core.detect's at_checkpoint
flatten case and core.runner_blocks' at_checkpoint/_at_checkpoint_release
dispatch for the block-engine side of this same handoff.
"""
import threading
from unittest.mock import MagicMock

from core import runner_expedition as re_mod
from core import vision
from core.runner import MacroRunner


def _checkpoint_runner(**overrides):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._log = lambda *a, **k: None
    runner._debug_save = lambda *a, **k: None
    runner._find_start_game_button = lambda hwnd: (None, None)
    runner._dismiss_reward_card_if_found = lambda hwnd: False
    runner._expedition_color_buttons = False
    runner._expedition_extract_count = 0
    runner._expedition_extract_accept_at = 2  # first sighting declines -- simplest branch to verify
    runner._interruptible_sleep = lambda seconds, stop: None
    runner._click_and_verify_gone = lambda *a, **k: True
    runner._expedition_checkpoint_state = "idle"
    runner._expedition_checkpoint_seen = False
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


def _match(score=0.9):
    return {"score": score, "x": 0, "y": 0, "w": 10, "h": 10, "cx": 5, "cy": 5}


def test_checkpoint_pauses_when_a_live_at_checkpoint_block_exists(monkeypatch):
    runner = _checkpoint_runner(_expedition_checkpoint_seen=True)
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))

    def fake_find_image(hwnd, name, **kwargs):
        if name == "exp_extract":
            return _match()
        raise vision.TemplateNotFound(name)
    monkeypatch.setattr(re_mod.vision, "find_image", fake_find_image)

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    assert result is None
    assert runner._expedition_checkpoint_state == "holding"
    # Paused before touching any of the real accept/decline/click logic.
    assert runner._expedition_extract_count == 0


def test_checkpoint_short_circuits_entirely_while_holding(monkeypatch):
    """Once holding, the search doesn't even re-run detection -- proven by
    making the checkpoint-specific searches raise, which would fail the test
    if either were called -- the shared "defeat" check still legitimately
    runs every tick regardless of holding (see the next test), so only
    exp_extract/exp_continue are forbidden here."""
    runner = _checkpoint_runner(_expedition_checkpoint_seen=True, _expedition_checkpoint_state="holding")
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))

    def fake_find_image(hwnd, name, **kwargs):
        if name in ("exp_extract", "exp_continue"):
            raise AssertionError(f"should not re-detect {name!r} while holding")
        raise vision.TemplateNotFound(name)
    monkeypatch.setattr(re_mod.vision, "find_image", fake_find_image)

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    assert result is None
    assert runner._expedition_checkpoint_state == "holding"  # untouched -- only At Checkpoint's release changes it


def test_checkpoint_defeat_still_detected_while_holding(monkeypatch):
    """Gap 5: holding only skips the extract/continue click, not the shared
    checks (start-game re-check/reward-card/defeat) above it."""
    runner = _checkpoint_runner(_expedition_checkpoint_seen=True, _expedition_checkpoint_state="holding")
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))

    def fake_find_image(hwnd, name, **kwargs):
        if name == "defeat":
            return _match()
        raise AssertionError(f"should not check {name!r} -- defeat should be found first")
    monkeypatch.setattr(re_mod.vision, "find_image", fake_find_image)

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    assert result == "loss"


def test_checkpoint_acts_once_released_and_resets_to_idle(monkeypatch):
    runner = _checkpoint_runner(_expedition_checkpoint_seen=True, _expedition_checkpoint_state="released")
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))

    def fake_find_image(hwnd, name, **kwargs):
        if name == "exp_extract":
            return _match()
        raise vision.TemplateNotFound(name)
    monkeypatch.setattr(re_mod.vision, "find_image", fake_find_image)

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    # accept_at=2, first sighting -> decline branch, not yet a win.
    assert result is None
    assert runner._expedition_extract_count == 1
    assert runner._expedition_checkpoint_state == "idle"


def test_checkpoint_with_no_live_at_checkpoint_block_behaves_unchanged(monkeypatch):
    """self._expedition_checkpoint_seen never set (no At Checkpoint block in
    the routine at all) -- the pause/release machinery must be a complete
    no-op, identical to before At Checkpoint existed."""
    runner = _checkpoint_runner(_expedition_checkpoint_seen=False)
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))

    def fake_find_image(hwnd, name, **kwargs):
        if name == "exp_extract":
            return _match()
        raise vision.TemplateNotFound(name)
    monkeypatch.setattr(re_mod.vision, "find_image", fake_find_image)

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    assert result is None
    assert runner._expedition_extract_count == 1  # acted on immediately, never paused
    assert runner._expedition_checkpoint_state == "idle"


def test_checkpoint_by_color_only_pauses_on_extract_offer_not_plain_continue(monkeypatch):
    """Scope agreed with the user: only Extract/Continue checkpoint sightings
    matter (camera alignment + placement allowed) -- a plain wave-Continue
    (not offering Extract) must never pause."""
    runner = _checkpoint_runner(_expedition_checkpoint_seen=True, _expedition_color_buttons=True)
    monkeypatch.setattr(re_mod.wm, "get_window_rect_screen", lambda hwnd: (0, 0, 1152, 756))
    # Only stubs out the wait -- time.time() stays real so the deadline loop
    # below still behaves correctly. find_color_run always returning a match
    # (even for the follow-up-Continue search) makes that loop exit after
    # exactly one pass, so this never actually needs to wait out the clock.
    monkeypatch.setattr(re_mod.time, "sleep", lambda *a, **k: None)

    # A plain wave-Continue: centered (not past the mirror margin), so
    # offers_extract reads False in _check_expedition_checkpoint_by_color.
    center_match = {"cx": 576, "cy": 700}
    monkeypatch.setattr(re_mod.vision, "find_color_run", lambda *a, **k: dict(center_match))

    result = runner._check_expedition_wave_result(hwnd=0, stop_event=threading.Event())

    assert result is None
    assert runner._expedition_checkpoint_state == "idle"  # never paused
