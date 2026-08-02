import threading
from unittest.mock import MagicMock

import numpy as np

from core import detect
from core import runner_blocks as rb
from core.runner import MacroRunner


# --------------------------------------------------------------------------
# flatten
# --------------------------------------------------------------------------
def test_flatten_without_detect_stamps_ordinals_and_passes_blocks_through():
    blocks = [
        {"type": "place_unit", "params": {}},
        {"type": "wait_ms"},
        {"type": "place_unit", "params": {}},
    ]
    flat, nxt = detect.flatten(blocks, 1)
    assert [b["type"] for b in flat] == ["place_unit", "wait_ms", "place_unit"]
    assert [b.get("_ordinal") for b in flat] == [1, None, 2]
    assert nxt == 3


def test_flatten_does_not_mutate_the_source_blocks():
    src = [{"type": "place_unit", "params": {}}]
    detect.flatten(src, 5)
    assert "_ordinal" not in src[0]  # flatten stamps a copy, never the saved dict


def test_flatten_then_else_offsets_route_both_branches():
    blocks = [
        {"type": "detect", "image": "a",
         "then": [{"type": "place_unit", "params": {}}, {"type": "wait_ms"}],
         "else": [{"type": "place_unit", "params": {}}]},
        {"type": "place_unit", "params": {}},
    ]
    flat, nxt = detect.flatten(blocks, 1)
    types = [b["type"] for b in flat]
    assert types == ["detect", "place_unit", "wait_ms", "_jump", "place_unit", "place_unit"]
    # ordinals stamped by static position (detect itself takes no number):
    # then's unit is #1, else's is #2, the trailing unit is #3.
    assert [b.get("_ordinal") for b in flat] == [None, 1, None, None, 2, 3]
    assert nxt == 4
    detect_block, jump = flat[0], flat[3]
    # FALSE from the detect (index 0) lands on the first else block (index 4)
    assert 0 + detect_block["_else_offset"] == 4
    # After the then branch runs, the _jump (index 3) skips the else block (index 4) -> index 5
    assert 3 + jump["_offset"] == 5
    assert 0 + detect_block["_end_offset"] == 5


def test_loop_settings_normalize_limits_and_interval():
    assert detect.loop_settings({"loop": True, "loopAttempts": "3", "loopIntervalMs": "250"}) == (True, 3, 0.25)
    assert detect.loop_settings({"loop": True, "loopAttempts": 0, "loopIntervalMs": 1}) == (True, 0, 0.1)
    assert detect.loop_settings({"loop": False}) == (False, 0, 1.0)


def test_flatten_empty_then_still_jumps_correctly():
    flat, _ = detect.flatten([{"type": "detect", "image": "a", "then": [], "else": [{"type": "wait_ms"}]}], 1)
    assert [b["type"] for b in flat] == ["detect", "_jump", "wait_ms"]
    assert 0 + flat[0]["_else_offset"] == 2   # false -> first else block
    assert 1 + flat[1]["_offset"] == 3        # true path: jump past else


def test_flatten_nested_detect_ordinals():
    blocks = [{
        "type": "detect", "image": "a",
        "then": [{
            "type": "detect", "image": "b",
            "then": [{"type": "place_unit", "params": {}}],
            "else": [{"type": "place_unit", "params": {}}],
        }],
        "else": [{"type": "place_unit", "params": {}}],
    }]
    flat, nxt = detect.flatten(blocks, 1)
    # three place_units total, numbered in static then-before-else order
    ordinals = [b.get("_ordinal") for b in flat if b["type"] == "place_unit"]
    assert ordinals == [1, 2, 3]
    assert nxt == 4


def test_flatten_battle_continues_prestart_numbering():
    prestart = [{"type": "place_unit", "params": {}}, {"type": "place_unit", "params": {}}]
    _, start = detect.flatten(prestart, 1)
    assert start == 3
    battle, _ = detect.flatten([{"type": "place_unit", "params": {}}], start)
    assert battle[0]["_ordinal"] == 3


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def _patch_vision(monkeypatch, present, matches_by_name=None):
    """present: set of names that have a reference image. matches_by_name:
    name -> match dict (or None). find_image returns the match or None; a name
    not in `present` raises TemplateNotFound."""
    matches_by_name = matches_by_name or {}
    monkeypatch.setattr(detect.vision, "detect_template_dir", lambda name: "ui")

    def find_image(hwnd, name, region=None, threshold=None, template_dir=None):
        if name not in present:
            raise detect.vision.TemplateNotFound(name)
        return matches_by_name.get(name)
    monkeypatch.setattr(detect.vision, "find_image", find_image)

    def find_image_all(hwnd, name, region=None, threshold=None, template_dir=None, max_results=50):
        if name not in present:
            raise detect.vision.TemplateNotFound(name)
        m = matches_by_name.get(name)
        return [m] if m else []
    monkeypatch.setattr(detect.vision, "find_image_all", find_image_all)


def test_evaluate_single_found_and_not_found(monkeypatch):
    hit = {"cx": 100, "cy": 200, "score": 0.97, "x": 90, "y": 190, "w": 20, "h": 20}
    _patch_vision(monkeypatch, present={"boss", "empty"}, matches_by_name={"boss": hit, "empty": None})
    runner = MagicMock()
    found, matches = detect.evaluate(runner, 1, {"mode": "single", "image": "boss"})
    assert found is True and matches == [hit]
    found, matches = detect.evaluate(runner, 1, {"mode": "single", "image": "empty"})
    assert found is False and matches == []


def test_evaluate_multi_and_or(monkeypatch):
    a = {"cx": 1, "cy": 1, "score": 0.9}
    _patch_vision(monkeypatch, present={"a", "b"}, matches_by_name={"a": a, "b": None})
    runner = MagicMock()
    assert detect.evaluate(runner, 1, {"mode": "multi", "images": ["a", "b"], "logic": "and"})[0] is False
    assert detect.evaluate(runner, 1, {"mode": "multi", "images": ["a", "b"], "logic": "or"})[0] is True
    assert detect.evaluate(runner, 1, {"mode": "multi", "images": ["a"], "logic": "and"})[0] is True


def test_evaluate_show_all_returns_locations(monkeypatch):
    hit = {"cx": 5, "cy": 6, "score": 0.95}
    _patch_vision(monkeypatch, present={"a"}, matches_by_name={"a": hit})
    found, matches = detect.evaluate(MagicMock(), 1, {"mode": "single", "image": "a", "showAll": True})
    assert found is True and matches == [hit]


def test_evaluate_missing_image_is_not_found_and_warns(monkeypatch):
    _patch_vision(monkeypatch, present=set())
    logs = []
    runner = MagicMock()
    runner._log = logs.append
    found, matches = detect.evaluate(runner, 1, {"mode": "single", "image": "ghost"})
    assert found is False and matches == []
    assert any("no reference image" in m for m in logs)


def test_diagnose_frame_uses_one_full_frame_and_offsets_region_matches(monkeypatch):
    local_best = {"x": 2, "y": 3, "w": 8, "h": 6, "cx": 6, "cy": 6, "score": 0.84}
    monkeypatch.setattr(detect.vision, "detect_template_dir", lambda _name: "detect")
    monkeypatch.setattr(
        detect.vision,
        "find_in_gray_multiscale_diagnostic",
        lambda *_args, **_kwargs: {"match": None, "best": local_best},
    )
    frame = np.zeros((40, 50, 3), dtype=np.uint8)

    report = detect.diagnose_frame(
        frame,
        {"mode": "single", "image": "Defense", "region": {"x": 10, "y": 8, "w": 20, "h": 15},
         "threshold": 0.90},
    )

    assert report["found"] is False
    detail = report["details"][0]
    assert detail["name"] == "Defense"
    assert detail["score"] == 0.84
    assert detail["best_match"]["x"] == 12
    assert detail["best_match"]["y"] == 11
    assert detail["best_match"]["cx"] == 16
    assert detail["best_match"]["cy"] == 14


def test_diagnose_frame_preserves_multi_image_and_or_logic(monkeypatch):
    hit = {"x": 1, "y": 2, "w": 4, "h": 5, "cx": 3, "cy": 4, "score": 0.95}
    monkeypatch.setattr(detect.vision, "detect_template_dir", lambda _name: "detect")

    def diagnose(_frame, name, **_kwargs):
        match = hit if name == "Defense" else None
        return {"match": match, "best": match}

    monkeypatch.setattr(detect.vision, "find_in_gray_multiscale_diagnostic", diagnose)
    frame = np.zeros((20, 30, 3), dtype=np.uint8)

    found_or = detect.diagnose_frame(
        frame, {"mode": "multi", "images": ["Defense", "Elite"], "logic": "or"})
    found_and = detect.diagnose_frame(
        frame, {"mode": "multi", "images": ["Defense", "Elite"], "logic": "and"})

    assert found_or["found"] is True
    assert found_and["found"] is False
    assert [detail["name"] for detail in found_and["details"]] == ["Defense", "Elite"]


def test_render_diagnostic_draws_region_and_best_candidate():
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    report = {
        "found": False,
        "region": (5, 6, 20, 15),
        "details": [{
            "name": "Defense", "matched": False, "score": 0.84,
            "threshold": 0.90, "best_match": {
                "x": 8, "y": 9, "w": 6, "h": 5, "cx": 11, "cy": 11,
                "score": 0.84,
            }, "match": None, "matches": [],
        }],
    }

    rendered = detect.render_diagnostic(frame, report)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_render_diagnostic_uses_red_for_accepted_matches():
    frame = np.zeros((50, 40, 3), dtype=np.uint8)
    rendered = detect.render_diagnostic(frame, {
        "found": True,
        "region": None,
        "details": [{
            "name": "nav_play", "matched": True,
            "match": {"x": 8, "y": 32, "w": 6, "h": 5, "cx": 11, "cy": 34, "score": 0.95},
            "matches": [],
        }],
    })

    red = (rendered[:, :, 2] > 200) & (rendered[:, :, 1] < 50) & (rendered[:, :, 0] < 50)
    assert red.any()


# --------------------------------------------------------------------------
# raw condition expression -- allowlist
# --------------------------------------------------------------------------
class _FakeCtx:
    def __init__(self, present, counts=None):
        self.present = set(present)
        self.counts = counts or {}

    def find(self, name):
        return name in self.present

    def count(self, name):
        return self.counts.get(name, 0)


def test_eval_expression_allows_boolean_and_compare():
    ctx = _FakeCtx({"a"}, counts={"c": 3})
    assert detect._eval_expr("find('a') and not find('b')", ctx) is True
    assert detect._eval_expr("find('b') or find('a')", ctx) is True
    assert detect._eval_expr("count('c') >= 2", ctx) is True
    assert detect._eval_expr("count('c') > 5", ctx) is False


def test_eval_expression_blocks_dangerous_input_and_fails_safe():
    ctx = _FakeCtx(set())
    for bad in [
        "__import__('os').system('echo hi')",
        "find.__class__",
        "open('x')",
        "[find('a') for _ in range(3)]",
        "find('a'); find('b')",
        "1 if find('a') else 0",
    ]:
        logs = []
        assert detect._eval_expr(bad, ctx, log=logs.append) is False
        assert logs, f"expected a warning log for blocked expr: {bad}"


def test_eval_expression_empty_is_false():
    assert detect._eval_expr("", _FakeCtx(set())) is False


# --------------------------------------------------------------------------
# runner tick: detect routes the flat index into the taken branch
# --------------------------------------------------------------------------
def _drive_battle(runner, flat):
    stop = threading.Event()
    for _ in range(50):
        if runner._battle_block_index >= len(flat):
            break
        runner._run_battle_blocks_tick(0, stop, flat, True, "m")


def test_battle_tick_runs_then_branch_when_found(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    runner._run_wait_ms_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    monkeypatch.setattr(rb.detect, "evaluate", lambda r, h, b: (True, []))

    flat, _ = rb.detect.flatten([
        {"type": "detect", "image": "x",
         "then": [{"type": "send_key", "_tag": "then"}],
         "else": [{"type": "send_key", "_tag": "else"}]},
        {"type": "wait_ms", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["then", "after"]


def test_battle_tick_runs_else_branch_when_not_found(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    runner._run_wait_ms_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    monkeypatch.setattr(rb.detect, "evaluate", lambda r, h, b: (False, []))

    flat, _ = rb.detect.flatten([
        {"type": "detect", "image": "x",
         "then": [{"type": "send_key", "_tag": "then"}],
         "else": [{"type": "send_key", "_tag": "else"}]},
        {"type": "wait_ms", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["else", "after"]


# --------------------------------------------------------------------------
# If / Set Boolean: same branching shape as Detect, but on a named boolean
# variable (self._macro_booleans) instead of an image search.
# --------------------------------------------------------------------------
def test_flatten_if_then_else_offsets_route_both_branches():
    """If blocks flatten through the exact same jump machinery as Detect --
    core.detect._flatten_into treats btype in ("detect", "if") identically."""
    blocks = [
        {"type": "if", "boolName": "ready",
         "then": [{"type": "place_unit", "params": {}}, {"type": "wait_ms"}],
         "else": [{"type": "place_unit", "params": {}}]},
        {"type": "place_unit", "params": {}},
    ]
    flat, nxt = detect.flatten(blocks, 1)
    types = [b["type"] for b in flat]
    assert types == ["if", "place_unit", "wait_ms", "_jump", "place_unit", "place_unit"]
    assert [b.get("_ordinal") for b in flat] == [None, 1, None, None, 2, 3]
    assert nxt == 4
    if_block, jump = flat[0], flat[3]
    assert 0 + if_block["_else_offset"] == 4
    assert 3 + jump["_offset"] == 5


def test_evaluate_if_reads_macro_booleans():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._macro_booleans = {"ready": True, "done": False}
    assert runner._evaluate_if({"boolName": "ready"}) is True
    assert runner._evaluate_if({"boolName": "done"}) is False


def test_evaluate_if_no_name_or_unset_variable_is_false():
    """No variable picked, or one no Set Boolean block has run yet, both
    read as False -- fail-safe, same spirit as Detect's missing-image path."""
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._macro_booleans = {}
    assert runner._evaluate_if({"boolName": ""}) is False
    assert runner._evaluate_if({"boolName": "never_set"}) is False


def test_run_set_boolean_tick_creates_and_overwrites():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._macro_booleans = {}
    logs = []
    runner._log = logs.append

    runner._run_set_boolean_tick({"params": {"name": "ready", "value": "True"}}, 1)
    assert runner._macro_booleans == {"ready": True}

    runner._run_set_boolean_tick({"params": {"name": "ready", "value": "False"}}, 2)
    assert runner._macro_booleans == {"ready": False}
    assert any('"ready" = True' in m for m in logs)
    assert any('"ready" = False' in m for m in logs)


def test_run_set_boolean_tick_no_name_skips():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._macro_booleans = {}
    logs = []
    runner._log = logs.append

    runner._run_set_boolean_tick({"params": {"name": "", "value": "True"}}, 1)
    assert runner._macro_booleans == {}
    assert any("no variable name set" in m for m in logs)


def test_battle_tick_runs_if_then_branch_when_bool_true():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._macro_booleans = {"ready": True}
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    runner._run_wait_ms_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "if", "boolName": "ready",
         "then": [{"type": "send_key", "_tag": "then"}],
         "else": [{"type": "send_key", "_tag": "else"}]},
        {"type": "wait_ms", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["then", "after"]


def test_battle_tick_runs_if_else_branch_when_bool_false():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._macro_booleans = {"ready": False}
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    runner._run_wait_ms_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "if", "boolName": "ready",
         "then": [{"type": "send_key", "_tag": "then"}],
         "else": [{"type": "send_key", "_tag": "else"}]},
        {"type": "wait_ms", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["else", "after"]


def test_set_boolean_then_if_end_to_end():
    """Set Boolean followed by an If reading the same name, both dispatched
    through the real _run_battle_blocks_tick -- the two blocks actually
    cooperating through self._macro_booleans, not just each tested alone."""
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._macro_booleans = {}
    runner._log = lambda *a, **k: None
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "set_boolean", "params": {"name": "ready", "value": "True"}},
        {"type": "if", "boolName": "ready",
         "then": [{"type": "send_key", "_tag": "then"}],
         "else": [{"type": "send_key", "_tag": "else"}]},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["then"]
    assert runner._macro_booleans == {"ready": True}


# --------------------------------------------------------------------------
# Detect self-loop: retry the SAME Detect until it's found (or a search-limit
# gives up into Else), instead of resolving Then/Else on the very first look.
# --------------------------------------------------------------------------
def test_looped_detect_retries_then_runs_then_once(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    outcomes = iter([(False, []), (True, [])])
    monkeypatch.setattr(rb.detect, "evaluate", lambda r, h, b: next(outcomes))
    clock = iter([100.0, 100.2])
    monkeypatch.setattr(rb.time, "time", lambda: next(clock))

    flat, _ = rb.detect.flatten([{
        "type": "detect", "image": "x", "loop": True,
        "loopAttempts": 3, "loopIntervalMs": 100,
        "then": [{"type": "send_key", "_tag": "then"}],
        "else": [{"type": "send_key", "_tag": "else"}],
    }])
    completed = set()
    runner._run_battle_blocks_tick(0, threading.Event(), flat, True, "m", persistent_detects=completed)
    assert runner._battle_block_index == 0
    runner._run_battle_blocks_tick(0, threading.Event(), flat, True, "m", persistent_detects=completed)
    _drive_battle(runner, flat)
    assert recorded == ["then"]
    assert completed == {0}

    # Loop A/B starts the flat list over, but the completed Detect skips its
    # whole construct instead of firing Then again while the image remains.
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._run_battle_blocks_tick(0, threading.Event(), flat, True, "m", persistent_detects=completed)
    assert runner._battle_block_index == len(flat)
    assert recorded == ["then"]


def test_looped_detect_uses_else_once_after_search_limit(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))
    monkeypatch.setattr(rb.detect, "evaluate", lambda r, h, b: (False, []))
    clock = iter([200.0, 200.2])
    monkeypatch.setattr(rb.time, "time", lambda: next(clock))

    flat, _ = rb.detect.flatten([{
        "type": "detect", "image": "x", "loop": True,
        "loopAttempts": 2, "loopIntervalMs": 100,
        "then": [{"type": "send_key", "_tag": "then"}],
        "else": [{"type": "send_key", "_tag": "else"}],
    }])
    completed = set()
    runner._run_battle_blocks_tick(0, threading.Event(), flat, True, "m", persistent_detects=completed)
    runner._run_battle_blocks_tick(0, threading.Event(), flat, True, "m", persistent_detects=completed)
    _drive_battle(runner, flat)
    assert recorded == ["else"]
    assert completed == {0}


def test_prestart_looped_detect_waits_until_found(monkeypatch):
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._log = lambda *a, **k: None
    runner._log_detect_outcome = lambda *a, **k: None
    runner._checkpoint = lambda _stop: False
    sleeps = []
    runner._interruptible_sleep = lambda seconds, _stop: sleeps.append(seconds)
    outcomes = iter([(False, []), (True, [])])
    monkeypatch.setattr(rb.detect, "evaluate", lambda r, h, b: next(outcomes))

    result = runner._run_prestart_detect(
        0, threading.Event(),
        {"type": "detect", "loop": True, "loopAttempts": 0, "loopIntervalMs": 100},
        1,
    )
    assert result is True
    assert sleeps == [0.1]


# --------------------------------------------------------------------------
# Repeat While: a real loop (not a one-shot branch like Detect/If) over the
# same named-boolean condition If reads -- checked only at the top of each
# pass, backward _jump instead of Detect/If's forward-only then/jump/else.
# --------------------------------------------------------------------------
def test_flatten_repeat_while_offsets_skip_or_loop_back():
    blocks = [
        {"type": "repeat_while", "boolName": "go",
         "then": [{"type": "place_unit", "params": {}}, {"type": "wait_ms"}],
         "else": []},
        {"type": "place_unit", "params": {}},
    ]
    flat, nxt = detect.flatten(blocks, 1)
    types = [b["type"] for b in flat]
    assert types == ["repeat_while", "place_unit", "wait_ms", "_jump", "place_unit"]
    assert [b.get("_ordinal") for b in flat] == [None, 1, None, None, 2]
    assert nxt == 3
    rw_block, loop_back = flat[0], flat[3]
    # False -> skip the whole construct (body + loop-back jump), landing on
    # the place_unit right after it.
    assert 0 + rw_block["_else_offset"] == 4
    # After the body finishes, jump BACKWARD to the repeat_while itself to
    # re-check the condition -- not forward past it like detect/if's own
    # trailing _jump does.
    assert 3 + loop_back["_offset"] == 0


def test_battle_tick_repeat_while_skips_body_when_false_from_start():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._macro_booleans = {"go": False}
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "repeat_while", "boolName": "go",
         "then": [{"type": "send_key", "_tag": "body"}], "else": []},
        {"type": "send_key", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["after"]


def test_battle_tick_repeat_while_runs_body_once_then_exits():
    """The condition is re-checked at the top of the very next pass, so a
    body that flips its own variable to False stops the loop after exactly
    one pass instead of running again."""
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._macro_booleans = {"go": True}
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "repeat_while", "boolName": "go", "then": [
            {"type": "send_key", "_tag": "body"},
            {"type": "set_boolean", "params": {"name": "go", "value": "False"}},
        ], "else": []},
        {"type": "send_key", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["body", "after"]
    assert runner._macro_booleans == {"go": False}


def test_battle_tick_repeat_while_loops_multiple_passes_then_exits():
    """Proves the backward jump actually re-enters the body more than once
    -- not just a single conditional skip -- by flipping the condition from
    OUTSIDE the loop body (a real Set Boolean can't count) once the body has
    run three times."""
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._macro_booleans = {"go": True}
    recorded = []

    def fake_send_key(stop, block, num, phase_label="Battle"):
        recorded.append(block.get("_tag"))
        if recorded.count("body") >= 3:
            runner._macro_booleans["go"] = False

    runner._run_send_key_tick = fake_send_key

    flat, _ = rb.detect.flatten([
        {"type": "repeat_while", "boolName": "go",
         "then": [{"type": "send_key", "_tag": "body"}], "else": []},
        {"type": "send_key", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["body", "body", "body", "after"]


# --------------------------------------------------------------------------
# At Checkpoint: structurally an "if" (body in "then", "else" unused), but
# the condition is an internal engine flag (self._expedition_checkpoint_state)
# rather than a user boolean, and the "then" branch falls through to a
# release marker instead of a plain _jump -- see core.runner_expedition's
# idle/holding/released handoff.
# --------------------------------------------------------------------------
def test_flatten_at_checkpoint_offsets_skip_or_fall_through():
    blocks = [
        {"type": "at_checkpoint",
         "then": [{"type": "place_unit", "params": {}}, {"type": "wait_ms"}],
         "else": []},
        {"type": "place_unit", "params": {}},
    ]
    flat, nxt = detect.flatten(blocks, 1)
    types = [b["type"] for b in flat]
    assert types == ["at_checkpoint", "place_unit", "wait_ms", "_at_checkpoint_release", "place_unit"]
    assert [b.get("_ordinal") for b in flat] == [None, 1, None, None, 2]
    assert nxt == 3
    ctrl = flat[0]
    # False -> skip straight past the release marker too, landing on the
    # place_unit right after the whole construct -- same end_index the True
    # path reaches after the release marker advances past itself.
    assert 0 + ctrl["_else_offset"] == 4


def test_battle_tick_at_checkpoint_skips_body_when_idle():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._expedition_checkpoint_state = "idle"
    runner._expedition_checkpoint_seen = False
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "at_checkpoint", "then": [{"type": "send_key", "_tag": "body"}], "else": []},
        {"type": "send_key", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["after"]
    # Reached at all (even though it took the False branch) proves this
    # instance is alive -- the checkpoint search reads this before ever
    # pausing (see runner_expedition).
    assert runner._expedition_checkpoint_seen is True


def test_battle_tick_at_checkpoint_runs_body_and_releases_when_holding():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._battle_block_index = 0
    runner._battle_block_state = {}
    runner._log = lambda *a, **k: None
    runner._expedition_checkpoint_state = "holding"
    runner._expedition_checkpoint_seen = False
    recorded = []
    runner._run_send_key_tick = lambda stop, block, num, phase_label="Battle": recorded.append(block.get("_tag"))

    flat, _ = rb.detect.flatten([
        {"type": "at_checkpoint", "then": [{"type": "send_key", "_tag": "body"}], "else": []},
        {"type": "send_key", "_tag": "after"},
    ])
    _drive_battle(runner, flat)
    assert recorded == ["body", "after"]
    assert runner._expedition_checkpoint_seen is True
    # The release marker hands the checkpoint search "released", not back to
    # "idle" directly -- see runner_expedition's handoff for why (acting on
    # the very next check, not re-pausing on the same still-visible sighting).
    assert runner._expedition_checkpoint_state == "released"
