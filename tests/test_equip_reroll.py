"""core.equip_reroll.read_reroll_count and the Detect Equipment Rerolls
block's dispatch (core.runner_blocks._run_detect_reroll_tick).
"""
from unittest.mock import MagicMock

import numpy as np

from core import equip_reroll
from core.runner import MacroRunner


def _runner():
    return MacroRunner(MagicMock(), MagicMock(), MagicMock())


# --------------------------------------------------------------------------
# read_reroll_count
# --------------------------------------------------------------------------
def test_returns_none_when_icon_not_found(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: None)
    assert equip_reroll.read_reroll_count(123) is None


def test_returns_none_when_icon_template_missing(monkeypatch):
    def boom(*a, **k):
        raise equip_reroll.vision.TemplateNotFound("expeditions_equip_reroll")
    monkeypatch.setattr(equip_reroll.vision, "find_image", boom)
    assert equip_reroll.read_reroll_count(123) is None


def _mock_digit_hits(hits_by_digit):
    """Stands in for equip_reroll._find_digit_hits: looks up canned hits by
    the digit being searched, same idea test_wave.py's _mock_char_hits uses
    for core.wave._find_char_hits."""
    def _find_digit_hits(_haystack, digit):
        return hits_by_digit.get(digit, [])
    return _find_digit_hits


def test_reads_a_single_digit_count(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: np.zeros((40, 50), dtype=np.uint8))
    monkeypatch.setattr(equip_reroll, "_find_digit_hits", _mock_digit_hits({
        3: [{"char": "3", "cx": 5, "score": 0.95}],
    }))

    assert equip_reroll.read_reroll_count(123) == 3


def test_reads_a_multi_digit_count_left_to_right(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: np.zeros((40, 50), dtype=np.uint8))
    # "12" laid out left to right by cx, fed out of order to prove the sort.
    monkeypatch.setattr(equip_reroll, "_find_digit_hits", _mock_digit_hits({
        2: [{"char": "2", "cx": 20, "score": 0.9}],
        1: [{"char": "1", "cx": 5, "score": 0.9}],
    }))

    assert equip_reroll.read_reroll_count(123) == 12


def test_no_digit_hits_returns_none(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: np.zeros((40, 50), dtype=np.uint8))
    monkeypatch.setattr(equip_reroll, "_find_digit_hits", _mock_digit_hits({}))

    assert equip_reroll.read_reroll_count(123) is None


# --------------------------------------------------------------------------
# _dedupe_hits: cross-digit non-max suppression
# --------------------------------------------------------------------------
def test_dedupe_hits_keeps_highest_scoring_candidate_at_same_position():
    hits = [
        {"char": "3", "cx": 50, "score": 0.91},
        # Within REROLL_DEDUP_DISTANCE_PX of the "3" above -- an ambiguous
        # font shape read as two different digits by two templates. Only
        # the higher-scoring interpretation should survive.
        {"char": "8", "cx": 51, "score": 0.97},
        # Far enough away to be a genuinely different glyph -- keeps its
        # own slot regardless of score.
        {"char": "1", "cx": 80, "score": 0.90},
    ]
    kept = sorted(equip_reroll._dedupe_hits(hits), key=lambda h: h["cx"])
    assert kept == [
        {"char": "8", "cx": 51, "score": 0.97},
        {"char": "1", "cx": 80, "score": 0.90},
    ]


def test_dedupe_hits_distance_boundary_is_inclusive():
    hits = [
        {"char": "1", "cx": 10, "score": 0.99},
        # Exactly REROLL_DEDUP_DISTANCE_PX away, not hardcoded, so this stays
        # correct if that constant is ever retuned again.
        {"char": "7", "cx": 10 + equip_reroll.REROLL_DEDUP_DISTANCE_PX, "score": 0.80},
    ]
    assert equip_reroll._dedupe_hits(hits) == [{"char": "1", "cx": 10, "score": 0.99}]


def test_dedupe_hits_just_past_the_boundary_keeps_both():
    hits = [
        {"char": "1", "cx": 10, "score": 0.99},
        {"char": "7", "cx": 10 + equip_reroll.REROLL_DEDUP_DISTANCE_PX + 1, "score": 0.80},
    ]
    kept = sorted(equip_reroll._dedupe_hits(hits), key=lambda h: h["cx"])
    assert kept == hits


# --------------------------------------------------------------------------
# _find_digit_hits
# --------------------------------------------------------------------------
def test_find_digit_hits_returns_empty_when_template_missing(monkeypatch):
    """Not every equip_reroll<N> folder needs to exist for this to work --
    a missing one is just never a candidate, same fail-safe spirit as
    everywhere else a reference image can be absent."""
    def boom(name, template_dir):
        raise equip_reroll.vision.TemplateNotFound(name)
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", boom)

    assert equip_reroll._find_digit_hits(np.zeros((10, 10), dtype=np.uint8), 5) == []


def test_find_digit_hits_matches_a_real_template_via_matchtemplate(monkeypatch):
    """Exercises the actual cv2.matchTemplate call (not mocked away), so a
    change to the search itself -- not just the surrounding plumbing -- gets
    caught. A flat/constant template would degenerate to a NaN correlation
    (no variance to normalize against), so this uses a genuinely varied
    2x2 patch embedded in the haystack at a known offset.

    Deliberately not asserting there's exactly one hit -- the correlation
    plateau right around the true position routinely clears threshold at
    more than one offset (that's exactly what _dedupe_hits exists to clean
    up, tested separately above); this only checks the true position is
    found, tagged correctly, and scores as a near-perfect match."""
    haystack = np.zeros((10, 12), dtype=np.uint8)
    template = np.array([[10, 200], [50, 90]], dtype=np.uint8)
    haystack[3:5, 4:6] = template  # top-left corner at (x=4, y=3)

    def fake_load(name, template_dir):
        assert name == "equip_reroll7"
        return [(template, None)]
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", fake_load)

    hits = equip_reroll._find_digit_hits(haystack, 7)

    true_hit = next((h for h in hits if h["cx"] == 5), None)  # x=4 + tw//2=1
    assert true_hit is not None
    assert true_hit["char"] == "7"
    assert true_hit["score"] > 0.99


def test_region_is_computed_relative_to_the_icon_center(monkeypatch):
    captured = []
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})

    def fake_capture(hwnd, region):
        captured.append(region)
        return np.zeros((1, 1), dtype=np.uint8)
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", fake_capture)

    def boom(*a, **k):
        raise equip_reroll.vision.TemplateNotFound("x")
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", boom)

    equip_reroll.read_reroll_count(123)

    assert captured == [(466, 60, 34, 25)]


def test_region_clamps_near_the_top_left_edge(monkeypatch):
    captured = []
    # cx=20/cy=25 clamps BOTH x0 and y0 to 0 (cx-34 and cy-40 both go
    # negative) while still leaving a non-empty box: x1=20 > 0, and
    # y1=cy-15=10 > 0.
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 20, "cy": 25, "score": 0.99})

    def fake_capture(hwnd, region):
        captured.append(region)
        return np.zeros((1, 1), dtype=np.uint8)
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", fake_capture)

    def boom(*a, **k):
        raise equip_reroll.vision.TemplateNotFound("x")
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", boom)

    equip_reroll.read_reroll_count(123)

    assert captured == [(0, 0, 20, 10)]


def test_returns_none_when_the_box_collapses_at_the_corner(monkeypatch):
    """icon at (0, 0) clamps both edges to the icon's own position -- an
    empty box, never worth a capture call at all."""
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 0, "cy": 0, "score": 0.99})

    def boom(*a, **k):
        raise AssertionError("should not capture when the box collapses to nothing")
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", boom)

    assert equip_reroll.read_reroll_count(123) is None


def test_returns_none_when_capture_fails(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: None)
    assert equip_reroll.read_reroll_count(123) is None


# --------------------------------------------------------------------------
# Detect Equipment Rerolls block dispatch (core.runner_blocks)
# --------------------------------------------------------------------------
def test_tick_sets_boolean_true_when_count_meets_minimum(monkeypatch):
    runner = _runner()
    runner._macro_booleans = {}
    logs = []
    runner._log = logs.append
    monkeypatch.setattr("core.equip_reroll.read_reroll_count", lambda hwnd: 4)

    runner._run_detect_reroll_tick(123, {"boolName": "worth_it", "params": {"minRerolls": 3}}, 1)

    assert runner._macro_booleans == {"worth_it": True}
    assert any("detected 4 reroll(s)" in m for m in logs)


def test_tick_sets_boolean_false_when_count_below_minimum(monkeypatch):
    runner = _runner()
    runner._macro_booleans = {}
    runner._log = lambda *a, **k: None
    monkeypatch.setattr("core.equip_reroll.read_reroll_count", lambda hwnd: 2)

    runner._run_detect_reroll_tick(123, {"boolName": "worth_it", "params": {"minRerolls": 3}}, 1)

    assert runner._macro_booleans == {"worth_it": False}


def test_tick_sets_boolean_true_when_count_equals_minimum(monkeypatch):
    runner = _runner()
    runner._macro_booleans = {}
    runner._log = lambda *a, **k: None
    monkeypatch.setattr("core.equip_reroll.read_reroll_count", lambda hwnd: 3)

    runner._run_detect_reroll_tick(123, {"boolName": "worth_it", "params": {"minRerolls": 3}}, 1)

    assert runner._macro_booleans == {"worth_it": True}


def test_tick_sets_boolean_false_when_icon_not_found(monkeypatch):
    runner = _runner()
    runner._macro_booleans = {}
    logs = []
    runner._log = logs.append
    monkeypatch.setattr("core.equip_reroll.read_reroll_count", lambda hwnd: None)

    runner._run_detect_reroll_tick(123, {"boolName": "worth_it", "params": {"minRerolls": 3}}, 1)

    assert runner._macro_booleans == {"worth_it": False}
    assert any("not found" in m for m in logs)


def test_tick_no_bool_name_skips_and_never_reads(monkeypatch):
    runner = _runner()
    runner._macro_booleans = {}
    runner._log = lambda *a, **k: None

    def boom(hwnd):
        raise AssertionError("should not read when no variable is selected")
    monkeypatch.setattr("core.equip_reroll.read_reroll_count", boom)

    runner._run_detect_reroll_tick(123, {"boolName": "", "params": {"minRerolls": 3}}, 1)

    assert runner._macro_booleans == {}
