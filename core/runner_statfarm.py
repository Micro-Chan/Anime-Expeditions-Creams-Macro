"""Stat Farm (Task Queue mode): grinds worthiness on an Infinite map, then
spends it rerolling fodder units' (Team Loadout slots 2-6 of 6) stats until
an in-game filter hit shows a "confirm" screen, one unit at a time,
round-robining across several selected Team Loadouts so progress stays even
across all of them instead of maxing one out before starting the next.

Split out as its own mixin -- same pattern as core.runner_shop's ShopOps,
core.runner_crafting's CraftingOps, core.runner_expedition's ExpeditionOps --
because its state machine (grind-to-worthy -> reroll -> round-robin to the
next loadout, no fixed repeat count) is fundamentally different from every
other Task Queue mode's single repeat loop (see core.runner._run_task).
Methods here run with MacroRunner's full self: shared state and helpers
(_log, _checkpoint, _click_found_image, _apply_team_loadout_explicit,
_click_restart_via_settings, ...) resolve normally.

Progress (which grid positions have already hit the player's in-game
filter, per loadout) is tracked in a small scratch JSON file, NOT
settings.json -- by design it is wiped at the start of every task run (see
reset_statfarm_progress, called once by _run_stat_farm_task) and untouched
by Pause/Resume, so a Stop/restart always starts the round-robin fresh
rather than trying to remember progress across sessions.
"""
import json
import os
import threading
import time

from . import constants
from . import vision
from . import wave as wave_module
from .runner_constants import *  # noqa: F401,F403 -- the shared constants namespace

STATFARM_PROGRESS_FILE = "statfarm_progress.json"
STATFARM_FODDER_SLOTS = 5  # grid positions 0-4, one per fodder unit (loadout slots 2-6)


def _statfarm_progress_path() -> str:
    return os.path.join(constants.APP_DIR, STATFARM_PROGRESS_FILE)


def reset_statfarm_progress() -> None:
    """Wipes all Stat Farm progress. Called once, at the very start of a
    Stat Farm task run (see StatFarmOps._run_stat_farm_task) -- the only
    place this ever happens; a Pause/Resume mid-run never touches it."""
    save_statfarm_progress({})


def load_statfarm_progress() -> dict:
    try:
        with open(_statfarm_progress_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_statfarm_progress(data: dict) -> None:
    try:
        with open(_statfarm_progress_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def mark_done(data: dict, loadout_num, position: int) -> None:
    key = str(loadout_num)
    done = data.setdefault(key, [False] * STATFARM_FODDER_SLOTS)
    if 0 <= position < len(done):
        done[position] = True


def done_count(data: dict, loadout_num) -> int:
    return sum(1 for x in data.get(str(loadout_num), []) if x)


def is_fully_done(data: dict, loadout_num) -> bool:
    done = data.get(str(loadout_num), [False] * STATFARM_FODDER_SLOTS)
    return len(done) >= STATFARM_FODDER_SLOTS and all(done)


def not_done_positions(data: dict, loadout_num) -> list:
    done = data.get(str(loadout_num), [False] * STATFARM_FODDER_SLOTS)
    return [i for i in range(STATFARM_FODDER_SLOTS) if i >= len(done) or not done[i]]


class StatFarmOps:
    def _stat_farm_pick_next_loadout(self, selected_loadouts, progress: dict):
        """The selected loadout with the fewest filter hits so far (ties
        broken by loadout number, ascending) -- always farms whichever is
        currently behind rather than maxing one out before starting the
        next. None once every selected loadout is fully done (5/5 fodder
        units filter-hit) -- the task's completion signal."""
        candidates = [n for n in selected_loadouts if not is_fully_done(progress, n)]
        if not candidates:
            return None
        best = min(done_count(progress, n) for n in candidates)
        return min(n for n in candidates if done_count(progress, n) == best)

    def _stat_farm_check_worthiness(self, hwnd, stop_event: threading.Event, positions: list) -> bool:
        """Hovers each not-yet-done fodder unit's in-battle hotbar position,
        double-clicks it, then checks for worth_check_400. True only if
        every one of them is currently showing 400% worthiness."""
        for i in positions:
            if self._checkpoint(stop_event):
                return False
            x, y = vision.ref_to_screen(hwnd, *STATFARM_HOTBAR_HOVER[i])
            self._mouse.move_to(x, y)
            time.sleep(STATFARM_HOVER_SETTLE)
            self._mouse.double_click(x, y)
            time.sleep(STATFARM_HOVER_SETTLE)
            try:
                match = vision.find_image(hwnd, "worth_check_400")
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] Stat Farm: {exc}")
                return False
            if match is None:
                return False
        return True

    def _stat_farm_check_or_restart(self, hwnd, stop_event: threading.Event):
        """Polled every _wait_for_match_result tick while self._statfarm_active
        (see there). Every self._statfarm_check_interval waves: checks
        worthiness on the active loadout's not-yet-done fodder units.

        - All of them at 400% -> leaves to the lobby for real (Leave Stage ->
          Return to Lobby) and returns "statfarm_worthy".
        - Not all done AND the wave target's been reached -> restarts the
          SAME Infinite run in place (same trick the End Run Setup block
          uses: Settings -> Restart -> Restart-confirm) and returns
          "statfarm_restarted" -- staying on the same loadout, no lobby trip.
        - Otherwise (not a check-interval wave, or not done and not yet at
          the wave target) -> None, keep playing.
        """
        state = self._statfarm_wave_state
        now = time.time()
        if now < state.get("next_check", 0.0):
            return None
        state["next_check"] = now + WAIT_WAVE_POLL_INTERVAL

        try:
            image = vision.capture_window_region_bgr(hwnd, WAVE_REGION)
            if image is None:
                raise RuntimeError("window capture returned no image")
            current, _maximum = wave_module.read_wave(image, log=self._log)
        except Exception as exc:
            if not state.get("read_error_logged"):
                self._log(f"[Macro] Stat Farm couldn't read the wave counter ({exc}) -- "
                           f"retrying every {WAIT_WAVE_POLL_INTERVAL:.0f}s.")
                state["read_error_logged"] = True
            return None
        if current is None:
            return None
        state.pop("read_error_logged", None)

        interval = max(1, int(self._statfarm_check_interval))
        if current == state.get("last_checked_wave") or current % interval != 0:
            return None
        state["last_checked_wave"] = current

        progress = load_statfarm_progress()
        positions = not_done_positions(progress, self._statfarm_active_loadout)
        self._set_status(action=f"Stat Farm wave {current}: checking worthiness...")
        if self._stat_farm_check_worthiness(hwnd, stop_event, positions):
            self._log(f"[Macro] Stat Farm: all fodder units at 400% worthiness (wave {current}) -- "
                       f"leaving to reroll.")
            return "statfarm_worthy" if self._leave_infinite_at_wave_limit(hwnd, stop_event, current) else None

        if current >= self._statfarm_wave_target:
            self._log(f"[Macro] Stat Farm: wave {current} reached without full worthiness -- "
                       f"restarting in place.")
            if self._click_restart_via_settings(hwnd, stop_event, "Stat Farm"):
                return "statfarm_restarted"
            return None
        return None

    def _stat_farm_reroll_one_unit(self, hwnd, stop_event: threading.Event, loadout_num, position: int,
                                     progress: dict) -> bool:
        """Opens unit-selection, picks grid `position`, then rolls up to
        STATFARM_REROLL_MAX_ATTEMPTS times, stopping the instant stat_filter
        appears (the player's in-game filter hit). Returns whether the
        filter was hit this visit; marks progress and saves it either way
        the caller needs to know (a hit; an exhaustion just means "try again
        next pass")."""
        open_x, open_y = vision.ref_to_screen(hwnd, *STATFARM_UNIT_SELECT_OPEN)
        self._mouse.click(open_x, open_y)
        time.sleep(STATFARM_CLICK_SETTLE)
        if self._checkpoint(stop_event):
            return False
        grid_x, grid_y = vision.ref_to_screen(hwnd, *STATFARM_UNIT_SELECT_GRID[position])
        self._mouse.click(grid_x, grid_y)
        time.sleep(STATFARM_CLICK_SETTLE)
        confirm_x, confirm_y = vision.ref_to_screen(hwnd, *STATFARM_UNIT_SELECT_CONFIRM)
        self._mouse.click(confirm_x, confirm_y)

        for attempt in range(1, STATFARM_REROLL_MAX_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            if not self._click_found_image(hwnd, "reroll_btn", NAV_CLICK_TIMEOUT, stop_event):
                self._log(f"[Macro] Stat Farm: couldn't find the reroll button -- "
                           f"stopping loadout {loadout_num} position {position} for this pass.")
                return False
            try:
                filter_match = vision.wait_for_image(
                    hwnd, "stat_filter", timeout=STATFARM_FILTER_CHECK_TIMEOUT, stop_event=stop_event)
            except vision.TemplateNotFound:
                filter_match = None
            if filter_match is not None:
                back_x, back_y = vision.ref_to_screen(hwnd, *STATFARM_FILTER_BACK)
                self._mouse.click(back_x, back_y)
                mark_done(progress, loadout_num, position)
                save_statfarm_progress(progress)
                self._log(f"[Macro] Stat Farm: loadout {loadout_num} position {position} hit the "
                           f"filter on roll {attempt}/{STATFARM_REROLL_MAX_ATTEMPTS}.")
                return True
        self._log(f"[Macro] Stat Farm: loadout {loadout_num} position {position} used all "
                   f"{STATFARM_REROLL_MAX_ATTEMPTS} rolls without hitting the filter this pass.")
        return False

    def _stat_farm_run_reroll_flow(self, hwnd, stop_event: threading.Event, loadout_num) -> None:
        """The confirmed reroll UI flow: nav_ui -> nav_stat -> E ->
        upgrade_statreroll -> reroll one not-yet-done fodder unit at a time
        -> exit back to the lobby."""
        label = "Stat Farm reroll"
        if not self._click_found_image(hwnd, "nav_ui", NAV_CLICK_TIMEOUT, stop_event):
            self._log(f"[Macro] {label}: couldn't open the UI menu -- skipping this loadout's reroll pass.")
            return
        time.sleep(STATFARM_NAV_WAIT)
        if self._checkpoint(stop_event):
            return
        if not self._click_found_image(hwnd, "nav_stat", NAV_CLICK_TIMEOUT, stop_event):
            self._log(f"[Macro] {label}: couldn't open the Stat screen -- skipping.")
            return
        time.sleep(STATFARM_NAV_WAIT)
        if self._checkpoint(stop_event):
            return
        self._keyboard.tap(ord("E"))
        time.sleep(STATFARM_NAV_WAIT)
        if self._checkpoint(stop_event):
            return
        if not self._click_found_image(hwnd, "upgrade_statreroll", NAV_CLICK_TIMEOUT, stop_event):
            self._log(f"[Macro] {label}: couldn't open the reroll UI -- skipping.")
            return
        time.sleep(STATFARM_NAV_WAIT)
        try:
            opened = vision.wait_for_image(
                hwnd, "upgrade_statreroll_isopen", timeout=NAV_CLICK_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {label}: {exc}")
            return
        if opened is None:
            self._log(f"[Macro] {label}: reroll UI never confirmed open -- skipping.")
            return

        progress = load_statfarm_progress()
        for position in not_done_positions(progress, loadout_num):
            if self._checkpoint(stop_event):
                return
            self._stat_farm_reroll_one_unit(hwnd, stop_event, loadout_num, position, progress)

        exit_x, exit_y = vision.ref_to_screen(hwnd, *STATFARM_EXIT_TO_LOBBY)
        self._mouse.click(exit_x, exit_y)
        self._log(f"[Macro] {label}: exiting to the lobby.")
        try:
            vision.wait_for_image(hwnd, "nav_play", timeout=TEAM_PANEL_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound:
            pass

    def _run_stat_farm_task(self, hwnd, stop_event: threading.Event, task: dict, task_index: int,
                              task_count: int, coords: dict, scroll_power: int, scroll_nudges: int,
                              default_walk_paths: dict, webhook: dict) -> bool:
        """Stat Farm's task-level orchestrator -- same return contract as
        _run_task (False only on a real stop). Unlike every other Task
        Queue mode, this never "gives up": per the user, it either finishes
        (every selected loadout's 5 fodder units filter-hit) or keeps
        grinding indefinitely, since Z-tier rolls are rare."""
        reset_statfarm_progress()
        map_name = task.get("map") or "School Grounds"
        selected_loadouts = [n for n in (task.get("stat_farm_loadouts") or []) if isinstance(n, int)]
        if not selected_loadouts:
            self._log(f"[Macro] Stat Farm task {task_index}/{task_count}: no loadouts selected -- nothing to farm.")
            return True
        self._statfarm_check_interval = max(1, int(task.get("stat_farm_check_interval")
                                                     or STATFARM_DEFAULT_CHECK_INTERVAL))
        self._statfarm_wave_target = max(1, int(task.get("stat_farm_wave_target")
                                                  or STATFARM_DEFAULT_WAVE_TARGET))

        sub_task = dict(task)
        sub_task["mode"] = "story"
        sub_task["stage"] = "Infinite"
        sub_task["map"] = map_name
        # Infinite is always Hard in-game regardless of what's set (same as
        # every other Story > Infinite task, see SPECIAL_STAGES_NO_DIFFICULTY
        # in core.runner) -- kept explicit here rather than left to whatever
        # `task` happens to carry, since Stat Farm's own Task Builder field
        # shows a locked "Hard" chip to match.
        sub_task["difficulty"] = "Hard"

        self._set_status(current_task=f"{task_index} / {task_count}", map=map_name,
                          action="Stat Farm: entering the stage...", mode="stat_farm", stage="Infinite",
                          play_mode=task.get("play_mode") or "solo", macro=task.get("macro") or "-")
        if not self._run_task_setup(hwnd, stop_event, sub_task, "story", map_name, coords,
                                      scroll_power, scroll_nudges, webhook):
            if stop_event.is_set():
                return False
            self._log(f"[Macro] Stat Farm task {task_index}/{task_count}: couldn't enter \"{map_name}\".")
            return True

        first_entry = True
        while True:
            if self._checkpoint(stop_event):
                return False
            progress = load_statfarm_progress()
            loadout_num = self._stat_farm_pick_next_loadout(selected_loadouts, progress)
            if loadout_num is None:
                self._log(f'[Macro] Stat Farm task {task_index}/{task_count}: every selected loadout '
                           f'has a filter hit on all {STATFARM_FODDER_SLOTS} fodder units -- done.')
                return True

            self._statfarm_active = True
            self._statfarm_active_loadout = loadout_num
            if not self._apply_team_loadout_explicit(hwnd, stop_event, loadout_num):
                self._statfarm_active = False
                if stop_event.is_set():
                    return False
                self._log(f"[Macro] Stat Farm: couldn't apply Team Loadout {loadout_num} -- stopping.")
                return True

            result = None
            pass_first_repeat = first_entry
            while True:
                self._statfarm_wave_state = {}
                result = self._play_one_match(hwnd, stop_event, sub_task, default_walk_paths,
                                                first_repeat=pass_first_repeat, webhook=webhook)
                pass_first_repeat = False
                first_entry = False
                if result == "statfarm_restarted":
                    if not self._wait_teleport_in(hwnd, stop_event, webhook, sub_task):
                        self._statfarm_active = False
                        if stop_event.is_set():
                            return False
                        self._log("[Macro] Stat Farm: didn't re-teleport in after restarting -- stopping.")
                        return True
                    continue
                break
            self._statfarm_active = False

            if result != "statfarm_worthy":
                if stop_event.is_set():
                    return False
                self._log(f"[Macro] Stat Farm task {task_index}/{task_count}: loadout {loadout_num}'s "
                           f"Infinite pass didn't resolve cleanly -- stopping.")
                return True

            self._stat_farm_run_reroll_flow(hwnd, stop_event, loadout_num)
            if self._checkpoint(stop_event):
                return False
