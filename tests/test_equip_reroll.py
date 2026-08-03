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


def test_picks_the_highest_scoring_value(monkeypatch):
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: np.zeros((40, 50), dtype=np.uint8))

    # Each value's "template" is a 1x1 array holding its own value, so the
    # mocked best_match_in_gray can tell candidates apart by content alone
    # -- no real image data needed to prove which one wins.
    def fake_load(name, template_dir):
        value = int(name.replace("equip_reroll", ""))
        return [(np.array([[value]], dtype=np.uint8), None)]
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", fake_load)

    scores = {1: 0.5, 2: 0.6, 3: 0.95, 4: 0.7, 5: 0.4, 6: 0.3}

    def fake_best_match(haystack, template_gray, mask=None):
        return {"score": scores[int(template_gray[0, 0])]}
    monkeypatch.setattr(equip_reroll.vision, "best_match_in_gray", fake_best_match)

    assert equip_reroll.read_reroll_count(123) == 3


def test_missing_individual_value_templates_are_skipped(monkeypatch):
    """Not every equip_reroll<N> folder needs to exist for this to work --
    a missing one is just never a candidate, same fail-safe spirit as
    everywhere else a reference image can be absent."""
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 500, "cy": 100, "score": 0.99})
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", lambda *a, **k: np.zeros((40, 50), dtype=np.uint8))

    def fake_load(name, template_dir):
        if name == "equip_reroll5":
            return [(np.array([[5]], dtype=np.uint8), None)]
        raise equip_reroll.vision.TemplateNotFound(name)
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", fake_load)
    monkeypatch.setattr(equip_reroll.vision, "best_match_in_gray", lambda h, t, mask=None: {"score": 0.5})

    assert equip_reroll.read_reroll_count(123) == 5


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

    assert captured == [(450, 60, 50, 40)]


def test_region_clamps_near_the_top_left_edge(monkeypatch):
    captured = []
    monkeypatch.setattr(equip_reroll.vision, "find_image", lambda *a, **k: {"cx": 10, "cy": 5, "score": 0.99})

    def fake_capture(hwnd, region):
        captured.append(region)
        return np.zeros((1, 1), dtype=np.uint8)
    monkeypatch.setattr(equip_reroll.vision, "capture_game_gray", fake_capture)

    def boom(*a, **k):
        raise equip_reroll.vision.TemplateNotFound("x")
    monkeypatch.setattr(equip_reroll.vision, "load_template_grays", boom)

    equip_reroll.read_reroll_count(123)

    assert captured == [(0, 0, 10, 5)]


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
