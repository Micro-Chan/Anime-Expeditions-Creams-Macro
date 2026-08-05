"""Detect Equipment Rerolls (Pre Start/Battle/Loop block): reads the
equipment-reroll count shown next to the expeditions_equip_reroll icon
before a run is committed to, so a task can skip a low-value roll instead
of spending 7-9 minutes on it.

Template matching, same digit-search approach as core.wave: the assets moved
from one fixed-count glyph per value (equip_reroll1..6) to individual digit
templates (equip_reroll0..9), so the count is now read as a genuine number
rather than classified as one of a fixed set of known counts -- every digit
template is searched for ALL of its on-screen hits in the box (not just the
single best), hits from different digit templates landing on the same glyph
are collapsed to the higher-scoring one, and what's left is read left-to-
right by x position, exactly like core.wave's current/max digit
reconstruction.
"""
import cv2
import numpy as np

from . import vision

REROLL_DIGITS = range(10)
# Bounding box for the count, relative to expeditions_equip_reroll's match
# center: (cx - 50, cy - 40) is the top-left, the icon's own center (cx, cy)
# is the bottom-right.
REROLL_BOX_LEFT = 34
REROLL_BOX_UP = 40
# Two digit-template hits (any digit, not just the same one -- two DIFFERENT
# digit templates can both clear threshold at nearly the same position on a
# font where two digits are visually close) whose centers land within this
# many pixels are the same glyph seen twice; only the higher-scoring one
# survives. 
REROLL_DEDUP_DISTANCE_PX = 3


def read_reroll_count(hwnd):
    """The on-screen reroll count next to expeditions_equip_reroll right
    now, read digit-by-digit (see module docstring), or None if the icon
    itself isn't found (including no reference image saved for it yet --
    same fail-safe spirit as Detect's missing-image handling), the computed
    box came out empty, or no digit template matched anything inside it."""
    threshold = vision._effective_threshold("expeditions_equip_reroll", vision.DEFAULT_THRESHOLD)
    try:
        icon = vision.find_image(hwnd, "expeditions_equip_reroll", threshold=threshold)
    except vision.TemplateNotFound:
        return None
    if icon is None:
        return None

    x0 = max(0, icon["cx"] - REROLL_BOX_LEFT)
    y0 = max(0, icon["cy"] - REROLL_BOX_UP)
    x1, y1 = icon["cx"], icon["cy"]-15
    if x1 <= x0 or y1 <= y0:
        return None
    region = (x0, y0, x1 - x0, y1 - y0)

    haystack = vision.capture_game_gray(hwnd, region)
    if haystack is None or haystack.size == 0:
        return None

    hits = []
    for digit in REROLL_DIGITS:
        hits.extend(_find_digit_hits(haystack, digit))
    if not hits:
        return None

    kept = _dedupe_hits(hits)
    kept.sort(key=lambda hit: hit["cx"])
    raw = "".join(hit["char"] for hit in kept)
    return int(raw) if raw.isdigit() else None


def _find_digit_hits(haystack_gray: np.ndarray, digit: int) -> list:
    """Every location ANY of equip_reroll<digit>'s reference variants match
    at/above its own (per-name-tunable, see Settings > General > Image
    Manager) threshold, tagged with the digit it represents. Deliberately
    NOT deduplicated here -- unlike vision.find_all_in_gray's single-name
    suppression, two DIFFERENT digit templates can both score above
    threshold at nearly the same position, so suppression has to run across
    every digit together (see _dedupe_hits) rather than per name."""
    hits = []
    hh, hw = haystack_gray.shape[:2]
    name = f"equip_reroll{digit}"
    try:
        templates = vision.load_template_grays(name, vision.UI_ASSETS_DIR)
    except vision.TemplateNotFound:
        return hits
    threshold = vision._effective_threshold(name, vision.DEFAULT_THRESHOLD)
    for template_gray, mask in templates:
        th, tw = template_gray.shape[:2]
        if th > hh or tw > hw:
            continue
        if mask is not None:
            result = cv2.matchTemplate(haystack_gray, template_gray, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            result = cv2.matchTemplate(haystack_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        result[~np.isfinite(result)] = -1
        ys, xs = np.where(result >= threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            hits.append({"char": str(digit), "cx": x + tw // 2, "score": float(result[y, x])})
    return hits


def _dedupe_hits(hits: list) -> list:
    """Highest-score-first greedy non-max suppression across EVERY digit
    candidate together, not per-digit (see _find_digit_hits) -- keeps the
    best-scoring interpretation of each on-screen glyph position and drops
    the rest."""
    ordered = sorted(hits, key=lambda hit: hit["score"], reverse=True)
    kept = []
    for hit in ordered:
        if any(abs(hit["cx"] - k["cx"]) <= REROLL_DEDUP_DISTANCE_PX for k in kept):
            continue
        kept.append(hit)
    return kept
