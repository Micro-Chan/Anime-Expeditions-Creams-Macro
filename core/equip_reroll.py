"""Detect Equipment Rerolls (Pre Start/Battle/Loop block): reads the 1-6
equipment-reroll count shown next to the expeditions_equip_reroll icon
before a run is committed to, so a task can skip a low-value roll instead
of spending 7-9 minutes on it.

Template matching, same reasoning as core.wave -- but simpler: the count is
always exactly one glyph from a fixed 1..6 vocabulary, never a
multi-character sequence, so this is forced-choice classification among 6
known templates (equip_reroll1..6) rather than something to reconstruct
digit-by-digit. No anchor-relative multi-hit search, no dedup -- just
"which one of these 6 scores highest in the fixed box next to the icon",
so unlike core.wave's digit hits, a candidate's raw score is trusted as-is
with no per-name threshold floor: expeditions_equip_reroll already being
found is what proves a real count is showing somewhere in that box, so
picking the single best of 6 known shapes is safe even when every score is
mediocre (a genuinely blank/miscaptured box would still favor whichever
template happens to correlate best, but that's the same trade-off any
forced 1-of-N classification makes).
"""
from . import vision

REROLL_VALUES = range(1, 7)
# Bounding box for the count, relative to expeditions_equip_reroll's match
# center: (cx - 50, cy - 40) is the top-left, the icon's own center (cx, cy)
# is the bottom-right.
REROLL_BOX_LEFT = 50
REROLL_BOX_UP = 40


def read_reroll_count(hwnd):
    """The best-matching reroll count (1-6) next to expeditions_equip_reroll
    on screen right now, or None if the icon itself isn't found (including
    no reference image saved for it yet -- same fail-safe spirit as
    Detect's missing-image handling) or the computed box came out empty."""
    threshold = vision._effective_threshold("expeditions_equip_reroll", vision.DEFAULT_THRESHOLD)
    try:
        icon = vision.find_image(hwnd, "expeditions_equip_reroll", threshold=threshold)
    except vision.TemplateNotFound:
        return None
    if icon is None:
        return None

    x0 = max(0, icon["cx"] - REROLL_BOX_LEFT)
    y0 = max(0, icon["cy"] - REROLL_BOX_UP)
    x1, y1 = icon["cx"], icon["cy"]
    if x1 <= x0 or y1 <= y0:
        return None
    region = (x0, y0, x1 - x0, y1 - y0)

    haystack = vision.capture_game_gray(hwnd, region)
    if haystack is None or haystack.size == 0:
        return None

    best_value, best_score = None, -1.0
    for value in REROLL_VALUES:
        name = f"equip_reroll{value}"
        try:
            templates = vision.load_template_grays(name, vision.UI_ASSETS_DIR)
        except vision.TemplateNotFound:
            continue
        for template_gray, mask in templates:
            match = vision.best_match_in_gray(haystack, template_gray, mask)
            if match is not None and match["score"] > best_score:
                best_score = match["score"]
                best_value = value
    return best_value
