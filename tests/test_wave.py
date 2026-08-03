from unittest.mock import MagicMock

import numpy as np
import pytest

from core import runner_blocks
from core import wave
from core.runner import MacroRunner

# The new (widened) WAVE_REGION's dimensions -- real-shaped zero arrays so
# core.wave's real crop/clamp logic runs against genuine array bounds
# instead of a stand-in shape that happens to always fit.
_REGION_SHAPE = (61, 144, 3)


def _mock_char_hits(hits_by_name):
    """Stands in for wave._find_char_hits: looks up canned hits by the
    template name being searched (wave_0..wave_9, wave_slash), same idea as
    the old OCR-era tests feeding canned `readings` per call, just keyed by
    name instead of call order since read_wave now searches every character
    name once per read rather than sweeping masks/psm modes."""
    def _find_char_hits(_haystack, name, _char):
        return hits_by_name.get(name, [])
    return _find_char_hits


def _fixed_icon(cx=120, cy=30):
    return lambda _gray: {"cx": cx, "cy": cy, "x": cx - 5, "y": cy - 5, "w": 10, "h": 10, "score": 0.99}


# --------------------------------------------------------------------------
# read_wave: no icon match at all
# --------------------------------------------------------------------------
def test_no_icon_match_returns_none_none_and_logs_which_stage_failed(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", lambda _gray: None)
    logged = []
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8), log=logged.append) == (None, None)
    assert len(logged) == 1
    assert "wave_icon" in logged[0]


def test_empty_region_returns_none_none():
    assert wave.read_wave(None) == (None, None)
    assert wave.read_wave(np.zeros((0, 0, 3), dtype=np.uint8)) == (None, None)


# --------------------------------------------------------------------------
# read_wave: reconstructing current/max from digit + slash hits
# --------------------------------------------------------------------------
def test_finite_wave_parses_current_and_max_from_digit_and_slash_hits(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    # "4 / 15" laid out left to right by cx.
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_4": [{"char": "4", "cx": 5, "score": 0.95}],
        "wave_slash": [{"char": "/", "cx": 15, "score": 0.95}],
        "wave_1": [{"char": "1", "cx": 25, "score": 0.95}],
        "wave_5": [{"char": "5", "cx": 35, "score": 0.95}],
    }))
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8)) == (4, 15)


def test_current_only_infinite_mode_has_no_slash_hit(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    # "42 wave" -- no slash template ever hit, so this is Infinite mode.
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_4": [{"char": "4", "cx": 5, "score": 0.95}],
        "wave_2": [{"char": "2", "cx": 15, "score": 0.95}],
    }))
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8)) == (42, None)


def test_no_char_hits_at_all_returns_none_none_and_logs_which_stage_failed(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({}))
    logged = []
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8), log=logged.append) == (None, None)
    assert len(logged) == 1
    assert "no digit/slash template matched" in logged[0]


def test_current_above_maximum_is_rejected(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    # "24 / 15" -- a finite stage cannot be beyond its own maximum.
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_2": [{"char": "2", "cx": 5, "score": 0.95}],
        "wave_4": [{"char": "4", "cx": 15, "score": 0.95}],
        "wave_slash": [{"char": "/", "cx": 25, "score": 0.95}],
        "wave_1": [{"char": "1", "cx": 35, "score": 0.95}],
        "wave_5": [{"char": "5", "cx": 45, "score": 0.95}],
    }))
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8)) == (None, None)


def test_zero_maximum_is_rejected(monkeypatch):
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    # "3 / 0" -- not a real wave counter reading.
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_3": [{"char": "3", "cx": 5, "score": 0.95}],
        "wave_slash": [{"char": "/", "cx": 15, "score": 0.95}],
        "wave_0": [{"char": "0", "cx": 25, "score": 0.95}],
    }))
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8)) == (None, None)


def test_missing_max_digits_logs_the_raw_detected_sequence(monkeypatch):
    """A live-reported failure mode: a slash matches but nothing on its
    right scores above threshold -- read_wave can't recover a max digit
    count that was never detected, but it should report what it DID see
    (via the optional `log` callable) instead of a bare "couldn't read"."""
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_4": [{"char": "4", "cx": 5, "score": 0.95}],
        "wave_slash": [{"char": "/", "cx": 15, "score": 0.95}],
        # nothing scored for the max side at all
    }))
    logged = []
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8), log=logged.append) == (None, None)
    assert len(logged) == 1
    assert '"4/"' in logged[0]


def test_multiple_slash_hits_only_split_on_the_first(monkeypatch):
    """A stray second slash-shaped hit (e.g. noise past the real digits)
    must not corrupt the maximum -- everything after the first slash is
    still read as one digit string, non-digit characters make it unparsable
    and the whole reading is rejected rather than silently truncated."""
    monkeypatch.setattr(wave, "_find_icon", _fixed_icon())
    monkeypatch.setattr(wave, "_find_char_hits", _mock_char_hits({
        "wave_4": [{"char": "4", "cx": 5, "score": 0.95}],
        "wave_slash": [
            {"char": "/", "cx": 15, "score": 0.95},
            {"char": "/", "cx": 35, "score": 0.90},
        ],
        "wave_1": [{"char": "1", "cx": 25, "score": 0.95}],
    }))
    assert wave.read_wave(np.zeros(_REGION_SHAPE, dtype=np.uint8)) == (None, None)


# --------------------------------------------------------------------------
# _dedupe_hits: cross-character non-max suppression
# --------------------------------------------------------------------------
def test_dedupe_hits_keeps_highest_scoring_candidate_at_same_position():
    hits = [
        {"char": "3", "cx": 50, "score": 0.91},
        # Within WAVE_DEDUP_DISTANCE_PX of the "3" above -- an ambiguous
        # font shape read as two different digits by two templates. Only
        # the higher-scoring interpretation should survive.
        {"char": "8", "cx": 51, "score": 0.97},
        # Far enough away to be a genuinely different glyph -- keeps its
        # own slot regardless of score.
        {"char": "1", "cx": 80, "score": 0.90},
    ]
    kept = sorted(wave._dedupe_hits(hits), key=lambda h: h["cx"])
    assert kept == [
        {"char": "8", "cx": 51, "score": 0.97},
        {"char": "1", "cx": 80, "score": 0.90},
    ]


def test_dedupe_hits_distance_boundary_is_inclusive():
    hits = [
        {"char": "1", "cx": 10, "score": 0.99},
        # Exactly WAVE_DEDUP_DISTANCE_PX away, not hardcoded, so this stays
        # correct if that constant is ever retuned again.
        {"char": "7", "cx": 10 + wave.WAVE_DEDUP_DISTANCE_PX, "score": 0.80},
    ]
    assert wave._dedupe_hits(hits) == [{"char": "1", "cx": 10, "score": 0.99}]


def test_dedupe_hits_just_past_the_boundary_keeps_both():
    hits = [
        {"char": "1", "cx": 10, "score": 0.99},
        {"char": "7", "cx": 10 + wave.WAVE_DEDUP_DISTANCE_PX + 1, "score": 0.80},
    ]
    kept = sorted(wave._dedupe_hits(hits), key=lambda h: h["cx"])
    assert kept == hits


# --------------------------------------------------------------------------
# _digit_search_crop: clamps to the captured region's own bounds
# --------------------------------------------------------------------------
def test_digit_search_crop_clamps_near_left_edge():
    gray = np.zeros((61, 144), dtype=np.uint8)
    # icon_cx=10: a naive cx-100 would go negative -- must clamp to x=0
    # rather than wrapping or raising.
    crop = wave._digit_search_crop(gray, icon_cx=10, icon_cy=30)
    assert crop is not None
    assert crop.shape == (10, 10)  # y: 25..35, x: 0..10


def test_digit_search_crop_clamps_near_top_edge():
    gray = np.zeros((61, 144), dtype=np.uint8)
    crop = wave._digit_search_crop(gray, icon_cx=120, icon_cy=1)
    assert crop is not None
    assert crop.shape[0] == 6  # y0 clamps to 0, y1 = 1 + 5 = 6


def test_digit_search_crop_none_when_icon_is_outside_the_region():
    gray = np.zeros((61, 144), dtype=np.uint8)
    crop = wave._digit_search_crop(gray, icon_cx=0, icon_cy=200)
    assert crop is None


# --------------------------------------------------------------------------
# Wait for Wave block: unchanged behavior, mocks read_wave wholesale so
# these don't care how the reading was actually produced.
# --------------------------------------------------------------------------
def test_wait_for_wave_requires_two_target_readings_before_later_blocks(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_state = {}
    block = {"params": {"wave": 10}}
    readings = iter([(14, 15), (4, 15), (10, 15), (10, 15)])

    monkeypatch.setattr(runner_blocks.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner_blocks.wm,
        "get_window_rect_screen",
        lambda _hwnd: (0, 0, 1152, 756),
    )
    monkeypatch.setattr(
        runner_blocks.vision,
        "capture_window_region_bgr",
        lambda _hwnd, _region: np.zeros((61, 104, 3)),
    )
    monkeypatch.setattr("core.wave.read_wave", lambda _image, log=None: next(readings))

    # A plausible one-frame jump above the target is not trusted.
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    # Returning below the target clears that pending confirmation.
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    # The real target also waits for one confirmation...
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    # ...and only the consecutive second read unlocks the next block.
    assert runner._run_wait_wave_tick(123, block, 1) is True


def test_wait_for_wave_captures_the_roblox_window_not_the_screen(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_state = {}
    block = {"params": {"wave": 10}}
    captured = []

    monkeypatch.setattr(
        runner_blocks.vision,
        "capture_window_region_bgr",
        lambda hwnd, region: captured.append((hwnd, region)) or np.zeros((61, 104, 3)),
    )
    monkeypatch.setattr(
        "core.ocr.capture_region",
        lambda *_args: pytest.fail("wave read must not capture the physical screen"),
    )
    monkeypatch.setattr(runner_blocks.time, "time", lambda: 100.0)
    monkeypatch.setattr("core.wave.read_wave", lambda _image, log=None: (10, 15))

    runner._run_wait_wave_tick(123, block, 1)

    assert captured == [(123, runner_blocks.WAVE_REGION)]


def test_wait_for_wave_supports_current_only_unlimited_counter(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_state = {}
    block = {"params": {"wave": 10}}
    readings = iter([(9, None), (10, None), (10, None)])

    monkeypatch.setattr(runner_blocks.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner_blocks.wm,
        "get_window_rect_screen",
        lambda _hwnd: (0, 0, 1152, 756),
    )
    monkeypatch.setattr(
        runner_blocks.vision,
        "capture_window_region_bgr",
        lambda _hwnd, _region: np.zeros((61, 104, 3)),
    )
    monkeypatch.setattr("core.wave.read_wave", lambda _image, log=None: next(readings))

    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    assert runner._run_wait_wave_tick(123, block, 1) is True
    assert all(
        "/None" not in logged.args[0]
        for logged in runner._log.call_args_list
    )


def test_wait_for_wave_rejects_inconsistent_impossible_unlimited_reads(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_state = {}
    block = {"params": {"wave": 45}}
    readings = iter([(1414, None), (46, None), (46, None)])

    monkeypatch.setattr(runner_blocks.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner_blocks.wm,
        "get_window_rect_screen",
        lambda _hwnd: (0, 0, 1152, 756),
    )
    monkeypatch.setattr(
        runner_blocks.vision,
        "capture_window_region_bgr",
        lambda _hwnd, _region: np.zeros((61, 104, 3)),
    )
    monkeypatch.setattr("core.wave.read_wave", lambda _image, log=None: next(readings))

    # The impossible first read cannot combine with the real wave 46.
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    assert runner._run_wait_wave_tick(123, block, 1) is False
    runner._battle_block_state["next_check"] = 0.0
    # Only the second identical 46 reading satisfies the confirmation.
    assert runner._run_wait_wave_tick(123, block, 1) is True
