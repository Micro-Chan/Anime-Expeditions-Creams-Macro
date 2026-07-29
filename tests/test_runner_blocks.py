"""Tests for core/runner_blocks.py (BlockOps mixin)."""
from unittest.mock import MagicMock, patch
import pytest

from core import keys
from core.runner_blocks import BlockOps


class DummyRunner(BlockOps):
    def __init__(self):
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)

    def _strip_auto_upgrade_for_expedition(self, blocks, task):
        return blocks


def test_load_battle_blocks_empty_macro():
    runner = DummyRunner()
    task = {}
    result = runner._load_battle_blocks(task)
    assert result == []


@patch("core.templates.load_template")
def test_load_battle_blocks_dict_format(mock_load):
    mock_load.return_value = {
        "blocks": {
            "battle": [{"type": "upgrade_unit", "slot": 1}]
        }
    }
    runner = DummyRunner()
    task = {"macro": "test_macro"}
    result = runner._load_battle_blocks(task)
    assert len(result) == 1
    assert result[0]["type"] == "upgrade_unit"


@patch("core.templates.load_template")
def test_load_battle_blocks_legacy_flat_list(mock_load):
    mock_load.return_value = {
        "blocks": [{"type": "place_unit"}]
    }
    runner = DummyRunner()
    task = {"macro": "old_macro"}
    result = runner._load_battle_blocks(task)
    assert result == []
    assert any("old format" in log for log in runner.logs)


@patch("core.templates.load_template")
def test_load_battle_blocks_legacy_three_phase(mock_load):
    mock_load.return_value = {
        "blocks": {
            "during": [{"type": "wait", "ms": 1000}],
            "after": [{"type": "walk", "path": "path1"}]
        }
    }
    runner = DummyRunner()
    task = {"macro": "legacy_macro"}
    result = runner._load_battle_blocks(task)
    assert len(result) == 2
    assert any("legacy during/after" in log for log in runner.logs)


def test_walk_block_replays_with_phase_label(monkeypatch):
    """The Walk block replays a recorded path and labels its log by phase --
    so the same block works in Pre Start (multiple allowed) and Battle."""
    from core import runner_blocks

    runner = DummyRunner()
    runner._keyboard = MagicMock()
    runner._set_status = lambda **k: None

    replayed = {}
    monkeypatch.setattr(runner_blocks.walk_paths, "load_path",
                        lambda name: {"events": [("w", "down", 0.0)]})
    monkeypatch.setattr(runner_blocks.walk_paths, "replay_events",
                        lambda events, kb, stop, sprint=False: replayed.setdefault("hit", True))

    import threading
    block = {"type": "walk", "params": {"path": "MyPath"}}
    runner._run_walk_block_tick(threading.Event(), block, 2, phase_label="Pre Start")

    assert replayed.get("hit") is True
    assert any("Pre Start block #2 (Walk)" in m for m in runner.logs)


def test_walk_block_no_path_is_skipped(monkeypatch):
    from core import runner_blocks
    import threading

    runner = DummyRunner()
    runner._keyboard = MagicMock()
    runner._set_status = lambda **k: None
    called = {"replay": False}
    monkeypatch.setattr(runner_blocks.walk_paths, "replay_events",
                        lambda *a, **k: called.__setitem__("replay", True))

    runner._run_walk_block_tick(threading.Event(), {"type": "walk", "params": {"path": ""}}, 1,
                                phase_label="Pre Start")

    assert called["replay"] is False
    assert any("no path selected" in m for m in runner.logs)

# ---------------------------------------------------------------------------
# Placement: the scan box must stay inside the game window
# ---------------------------------------------------------------------------
# Centering a PLACE_SEARCH_BOX_SIZE box on a saved spot within half a box of an
# edge used to capture pixels from outside the game entirely. On Windows the
# docked game sits inside this app's own frame, so those pixels are the macro's
# control panel -- near-white in the Light theme, and therefore accepted as a
# valid placement tile.

class _ScanRunner(BlockOps):
    def __init__(self):
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)


def _record_capture(white_at=None):
    """capture_region stand-in: records the region asked for, and optionally
    paints one white pixel at a given WINDOW coordinate."""
    import numpy as np
    seen = {}

    def capture_region(x, y, w, h):
        seen.clear()
        seen.update(x=x, y=y, w=w, h=h)
        patch = np.zeros((h, w, 3), np.uint8)
        if white_at is not None:
            px, py = white_at[0] - x, white_at[1] - y
            if 0 <= px < w and 0 <= py < h:
                patch[py, px] = (255, 255, 255)
        return patch

    return capture_region, seen


@pytest.mark.parametrize("spot", [
    (5, 400), (400, 3), (1149, 400), (400, 753), (2, 2), (1150, 754), (576, 378),
])
def test_scan_box_never_reads_outside_the_window(spot, monkeypatch):
    from core.config import FIXED_WIN_H, FIXED_WIN_W
    capture, seen = _record_capture()
    monkeypatch.setattr("core.ocr.capture_region", capture)

    _ScanRunner()._scan_place_search_box(0, 0, *spot)

    assert seen["x"] >= 0 and seen["y"] >= 0, f"captured off the top/left: {seen}"
    assert seen["x"] + seen["w"] <= FIXED_WIN_W, f"captured past the right edge: {seen}"
    assert seen["y"] + seen["h"] <= FIXED_WIN_H, f"captured past the bottom edge: {seen}"


def test_scan_box_does_not_accept_a_white_pixel_outside_the_window(monkeypatch):
    # x=5 used to capture x=-14..24; a white pixel at x=-6 is the macro's own
    # panel, and was returned as a placement tile at offset (-11, 0).
    capture, _ = _record_capture(white_at=(-6, 400))
    monkeypatch.setattr("core.ocr.capture_region", capture)
    assert _ScanRunner()._scan_place_search_box(0, 0, 5, 400) is None


@pytest.mark.parametrize("spot,white,expected", [
    ((576, 378), (580, 378), (4, 0)),    # no clamping needed
    ((576, 378), (576, 373), (0, -5)),
    ((5, 400), (9, 400), (4, 0)),        # box shifted right by the clamp
    ((4, 6), (7, 9), (3, 3)),            # shifted on both axes
])
def test_scan_box_offset_is_measured_from_the_requested_spot(spot, white, expected, monkeypatch):
    """Clamping moves the box, so the offset has to be relative to the spot the
    caller asked about -- not the middle of whatever region got captured. Get
    this wrong and every placement near an edge lands somewhere else."""
    capture, _ = _record_capture(white_at=white)
    monkeypatch.setattr("core.ocr.capture_region", capture)
    assert _ScanRunner()._scan_place_search_box(0, 0, *spot) == expected


# ---------------------------------------------------------------------------
# Placement: a broken quick-place chain must not leave Shift held
# ---------------------------------------------------------------------------
# Consecutive Place Unit blocks with the same hotkey hold Left Shift so the unit
# stays selected and later placements can skip re-pressing the hotkey. If the
# next member never runs -- marked "Once" on a repeat, or missing its position --
# nothing released Shift, and the FOLLOWING Place Unit block (a different unit)
# took the "Shift is down, same unit still selected" path and placed the
# previous unit on its tile.

class _ShiftRunner(BlockOps):
    def __init__(self):
        self.logs = []
        self.keyboard_events = []
        self._quick_place_shift_down = True      # a chain is in progress
        self._battle_block_index = 0
        self._battle_block_state = {}
        self._last_unit_ordinal = 0
        self._keyboard = MagicMock()
        self._keyboard.key_up = lambda vk: self.keyboard_events.append(("up", vk))

    def _log(self, msg):
        self.logs.append(msg)

    def _set_status(self, **kw):
        pass


def test_once_skipped_battle_block_releases_the_quick_place_shift():
    runner = _ShiftRunner()
    blocks = [{"type": "place_unit", "once": True, "hotkey": "1",
               "params": {"name": "Archer", "x": 100, "y": 100}}]

    runner._run_battle_blocks_tick(1, MagicMock(is_set=lambda: False), blocks, first_repeat=False)

    assert runner._quick_place_shift_down is False, "Shift left held after the chain broke"
    assert ("up", keys.VK_SHIFT) in runner.keyboard_events


def test_place_unit_with_no_position_releases_the_quick_place_shift():
    runner = _ShiftRunner()
    block = {"type": "place_unit", "hotkey": "1", "params": {"name": "Archer"}}   # no x/y

    runner._run_place_unit_block(1, MagicMock(is_set=lambda: False), 0, 0, block,
                                 index=1, macro_name="m", next_is_same_unit=False)

    assert runner._quick_place_shift_down is False, "Shift left held after a no-position skip"
    assert ("up", keys.VK_SHIFT) in runner.keyboard_events


def test_place_unit_with_no_position_keeps_shift_when_the_chain_continues():
    """The guard is conditional on purpose: if the NEXT block is the same unit,
    the chain is still alive and Shift must stay down, exactly as the other
    early returns in this function already do."""
    runner = _ShiftRunner()
    block = {"type": "place_unit", "hotkey": "1", "params": {"name": "Archer"}}

    runner._run_place_unit_block(1, MagicMock(is_set=lambda: False), 0, 0, block,
                                 index=1, macro_name="m", next_is_same_unit=True)

    assert runner._quick_place_shift_down is True


def test_place_unit_ignore_highlight_records_position_even_if_verify_never_confirms():
    """Ignore Highlight means the saved x/y IS the unit's position by
    definition -- Click Unit (and Sell/Upgrade/Target Priority/Auto Upgrade,
    which all resolve a unit the same way) must still be able to find it even
    when unit_exist verification never confirms the placement (no asset
    added, a stale/occupied tile, whatever) -- the click still landed there
    regardless of what verify saw afterward."""
    from core.runner_blocks import BlockOps
    from unittest.mock import MagicMock
    from core import vision

    class DummyRunner(BlockOps):
        def __init__(self):
            self._placed_unit_positions = {}
            self._quick_place_shift_down = False
            self._mouse = MagicMock()
            self._keyboard = MagicMock()
            self.logs = []
        def _log(self, msg):
            self.logs.append(msg)
        def _set_status(self, **kw):
            pass
        def _checkpoint(self, stop_event):
            return False
        def _release_quick_place_shift(self):
            pass

    runner = DummyRunner()
    block = {"type": "place_unit", "hotkey": "1", "ignoreHighlight": True,
             "params": {"name": "Archer", "x": 100, "y": 200}}
    stop_event = MagicMock(is_set=lambda: False)

    # Patch the two lookups on the real vision module (not a stand-in
    # module), so `except vision.TemplateNotFound` inside the runner still
    # matches the exception actually raised.
    from core import runner_blocks
    original_find_image, original_wait_for_image = vision.find_image, vision.wait_for_image
    vision.find_image = MagicMock(side_effect=vision.TemplateNotFound("max_placement_reached"))
    vision.wait_for_image = MagicMock(side_effect=vision.TemplateNotFound("unit_exist"))
    try:
        runner._run_place_unit_block(1, stop_event, 0, 0, block, index=1, macro_name="m",
                                      unit_ordinal=5, next_is_same_unit=False, verify=True)
    finally:
        vision.find_image = original_find_image
        vision.wait_for_image = original_wait_for_image

    assert runner._placed_unit_positions.get(5) == (100, 200)


def test_run_target_priority_tick():
    from core.runner_blocks import BlockOps
    from unittest.mock import MagicMock

    class DummyRunner(BlockOps):
        def __init__(self):
            self._placed_unit_positions = {1: (100, 200)}
            self._coords = {"unit_info_reset_x": 10, "unit_info_reset_y": 10}
            self._mouse = MagicMock()
            self._keyboard = MagicMock()
            self.logs = []
        def _log(self, msg):
            self.logs.append(msg)
        def _set_status(self, **kw):
            pass
        def _checkpoint(self, stop_event):
            return False

    runner = DummyRunner()
    block = {"type": "target_priority", "params": {"index": 1, "priority": "Boss"}}
    stop_event = MagicMock(is_set=lambda: False)

    with MagicMock():
        from core import runner_blocks
        original_wm = runner_blocks.wm
        runner_blocks.wm = MagicMock(get_window_rect_screen=lambda hwnd: (0, 0, 800, 600))
        try:
            done = runner._run_target_priority_tick(123, stop_event, block, 1)
        finally:
            runner_blocks.wm = original_wm

    assert done is True
    assert any("pressing R to set target priority to Boss" in log for log in runner.logs)
    assert runner._keyboard.tap.called


def test_run_click_unit_tick():
    from core.runner_blocks import BlockOps
    from unittest.mock import MagicMock

    class DummyRunner(BlockOps):
        def __init__(self):
            self._placed_unit_positions = {1: (100, 200)}
            self._coords = {"unit_info_reset_x": 10, "unit_info_reset_y": 10}
            self._mouse = MagicMock()
            self._keyboard = MagicMock()
            self.logs = []
        def _log(self, msg):
            self.logs.append(msg)
        def _set_status(self, **kw):
            pass
        def _checkpoint(self, stop_event):
            return False

    runner = DummyRunner()
    block = {"type": "click_unit", "params": {"index": 1}}
    stop_event = MagicMock(is_set=lambda: False)

    from core import runner_blocks
    original_wm = runner_blocks.wm
    runner_blocks.wm = MagicMock(get_window_rect_screen=lambda hwnd: (0, 0, 800, 600))
    try:
        done = runner._run_click_unit_tick(123, stop_event, block, 1)
    finally:
        runner_blocks.wm = original_wm

    assert done is True
    assert any("clicked unit at (100, 200)" in log for log in runner.logs)
    runner._mouse.click.assert_any_call(100, 200)


def test_run_click_unit_tick_no_unit_selected():
    from core.runner_blocks import BlockOps
    from unittest.mock import MagicMock

    class DummyRunner(BlockOps):
        def __init__(self):
            self._placed_unit_positions = {}
            self._mouse = MagicMock()
            self.logs = []
        def _log(self, msg):
            self.logs.append(msg)

    runner = DummyRunner()
    block = {"type": "click_unit", "params": {"index": ""}}
    stop_event = MagicMock(is_set=lambda: False)

    done = runner._run_click_unit_tick(123, stop_event, block, 1)

    assert done is True
    assert not runner._mouse.click.called


def _use_key_runner():
    from core.runner_blocks import BlockOps
    from unittest.mock import MagicMock

    class DummyRunner(BlockOps):
        def __init__(self):
            self._coords = {"unit_info_reset_x": 10, "unit_info_reset_y": 10}
            self._mouse = MagicMock()
            self._keyboard = MagicMock()
            self.logs = []
        def _log(self, msg):
            self.logs.append(msg)
        def _set_status(self, **kw):
            pass
        def _checkpoint(self, stop_event):
            return False

    return DummyRunner()


def test_run_auto_upgrade_use_key_presses_hotkey_n_times_for_priority():
    """Use Key mode: priority 5 means 5 presses of the configured hotkey,
    not clicks around the priority_upgrade icon."""
    from core import keys

    runner = _use_key_runner()
    block = {"params": {"priority": "5"}, "hotkey": "t"}
    stop_event = MagicMock(is_set=lambda: False)

    done = runner._run_auto_upgrade_use_key(stop_event, block, "label", 0, 0, "t")

    assert done is True
    assert runner._keyboard.tap.call_count == 5
    assert runner._keyboard.tap.call_args_list[0].args[0] == keys.key_name_to_vk("t")


def test_run_auto_upgrade_use_key_none_priority_presses_nothing():
    runner = _use_key_runner()
    block = {"params": {"priority": "None"}, "hotkey": "t"}
    stop_event = MagicMock(is_set=lambda: False)

    done = runner._run_auto_upgrade_use_key(stop_event, block, "label", 0, 0, "t")

    assert done is True
    assert not runner._keyboard.tap.called


def test_run_auto_upgrade_unit_tick_ignores_use_key_without_a_hotkey():
    """"If no hotkey is set it will default to USE KEY toggle off behaviour,
    even if USE KEY is on" -- so with useKey True but no hotkey, the normal
    click-the-priority-icon path must still run (proven here by it reaching
    the vision search that path is the only one that calls)."""
    from unittest.mock import MagicMock
    from core import runner_blocks, vision

    runner = _use_key_runner()
    runner._placed_unit_positions = {1: (100, 200)}
    block = {"params": {"index": 1, "priority": "3"}, "useKey": True, "hotkey": ""}
    stop_event = MagicMock(is_set=lambda: False)

    original_wm = runner_blocks.wm
    original_find_image_any = vision.find_image_any
    runner_blocks.wm = MagicMock(get_window_rect_screen=lambda hwnd: (0, 0, 800, 600))
    vision.find_image_any = MagicMock(side_effect=vision.TemplateNotFound("priority_upgrade"))
    try:
        done = runner._run_auto_upgrade_unit_tick(123, stop_event, block, 1)
    finally:
        runner_blocks.wm = original_wm
        vision.find_image_any = original_find_image_any

    assert done is True
    assert not runner._keyboard.tap.called
    assert any("priority_upgrade" in log for log in runner.logs)
