"""Reads the "Wait for Wave" badge (Battle > Wait for Wave block): a small
"<icon> <current> / <max> wave" readout (e.g. "0 / 15 wave") in the top HUD.

Template matching, not OCR: the badge is a fixed-font, fixed-scale game UI
element, not arbitrary text, so this correlates known reference crops
(Assets/ui/wave_0 .. wave_9, wave_slash, wave_icon) against the live capture
instead of running a general-purpose text classifier at it. That sidesteps
the whole class of misread this module used to fight (O/0, S/5 confusion, a
sharpened "5" turning into "8", "0/15" fusing into "015", the HUD icon
hallucinated as an extra leading digit) -- a digit template simply can't be
confused for a different digit's shape the way an OCR engine's classifier
can.

wave_icon is actually a crop of the WORD "wave", not the icon graphic --
named that historically, kept for compatibility with the asset folder
already on disk. It's the anchor because it's the one element in the badge
whose glyph is visually unique on screen (the icon graphic itself recurs
elsewhere in the HUD and would false-positive-match there). The digits and
slash are searched in a window measured LEFT of wave_icon's match center,
not a fixed absolute box, so this keeps working even if the badge's overall
on-screen position drifts a little between setups -- only its position
RELATIVE to the word "wave" needs to stay constant.
"""
import cv2
import numpy as np

from . import vision

# How far left of wave_icon's center (in pixels) to search for the digits --
# generous on purpose: empty space just never matches a digit template, so
# erring wide is free, while erring narrow silently clips leading digits of
# a large wave count.
WAVE_DIGIT_SEARCH_WIDTH = 100
# Half-height of that search band, i.e. the digits are assumed to sit within
# +-this many pixels of wave_icon's own vertical center.
WAVE_DIGIT_SEARCH_HALF_HEIGHT = 5
# Two candidate hits (any characters, not just the same digit -- see
# _dedupe_hits) whose centers are this close together are the same glyph
# read by two different templates; only the higher-scoring one survives.
WAVE_DEDUP_DISTANCE_PX = 2


def read_wave(region_bgr):
    """Returns (current, maximum) ints, maximum None for an Infinite-mode
    "<current> wave" badge (no slash), or (None, None) if wave_icon couldn't
    be found at all or nothing parsed into a sane reading."""
    if region_bgr is None or getattr(region_bgr, "size", 0) == 0:
        return None, None
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)

    icon = _find_icon(gray)
    if icon is None:
        return None, None

    crop = _digit_search_crop(gray, icon["cx"], icon["cy"])
    if crop is None:
        return None, None

    hits = []
    for digit in range(10):
        name = f"wave_{digit}"
        hits.extend(_find_char_hits(crop, name, str(digit)))
    hits.extend(_find_char_hits(crop, "wave_slash", "/"))
    if not hits:
        return None, None

    kept = _dedupe_hits(hits)
    kept.sort(key=lambda hit: hit["cx"])

    slash_at = next((i for i, hit in enumerate(kept) if hit["char"] == "/"), None)
    if slash_at is None:
        # No slash at all -- Infinite mode's plain "<current> wave" badge.
        current = _parse_digits("".join(hit["char"] for hit in kept))
        return (current, None) if current is not None else (None, None)

    current = _parse_digits("".join(hit["char"] for hit in kept[:slash_at]))
    maximum = _parse_digits("".join(hit["char"] for hit in kept[slash_at + 1:]))
    if current is None or maximum is None:
        return None, None
    # A finite stage cannot be beyond its own maximum -- same sanity guard
    # the old OCR-based reader used, kept here as cheap insurance even
    # though template matching shouldn't produce the leading-digit
    # hallucination that originally motivated it.
    if maximum <= 0 or current > maximum:
        return None, None
    return current, maximum


def _find_icon(gray: np.ndarray) -> dict:
    """wave_icon's match (see the module docstring for why it's the anchor
    instead of the icon graphic), or None if it isn't on screen / has no
    reference image saved yet."""
    threshold = vision._effective_threshold("wave_icon", vision.DEFAULT_THRESHOLD)
    try:
        return vision.find_in_gray_multiscale(gray, "wave_icon", vision.UI_ASSETS_DIR, threshold)
    except vision.TemplateNotFound:
        return None


def _digit_search_crop(gray: np.ndarray, icon_cx: int, icon_cy: int):
    """The sub-image to search for digits/slash in: WAVE_DIGIT_SEARCH_WIDTH
    px immediately left of the icon's center, WAVE_DIGIT_SEARCH_HALF_HEIGHT
    px above/below it -- clamped to the captured region's own bounds so an
    icon match near the left/top/bottom edge doesn't wrap or index
    negative. None if the clamped box is empty."""
    h, w = gray.shape[:2]
    x0 = max(0, icon_cx - WAVE_DIGIT_SEARCH_WIDTH)
    x1 = max(x0, min(w, icon_cx))
    y0 = max(0, icon_cy - WAVE_DIGIT_SEARCH_HALF_HEIGHT)
    y1 = min(h, icon_cy + WAVE_DIGIT_SEARCH_HALF_HEIGHT)
    crop = gray[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def _find_char_hits(haystack_gray: np.ndarray, name: str, char: str) -> list:
    """Every location ANY of `name`'s reference variants match at/above its
    (per-name-tunable, see Settings > General > Image Manager) threshold,
    tagged with the character it represents. Deliberately NOT deduplicated
    here -- unlike vision.find_all_in_gray's single-name suppression, two
    DIFFERENT characters' templates can both score above threshold at
    nearly the same position (a font that's visually close between two
    digits), so suppression has to run across every character together
    (see _dedupe_hits) rather than per name."""
    hits = []
    hh, hw = haystack_gray.shape[:2]
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
            hits.append({"char": char, "cx": x + tw // 2, "score": float(result[y, x])})
    return hits


def _dedupe_hits(hits: list) -> list:
    """Highest-score-first greedy non-max suppression across EVERY
    digit/slash candidate together, not per-character (see _find_char_hits)
    -- keeps the best-scoring interpretation of each on-screen glyph
    position and drops the rest."""
    ordered = sorted(hits, key=lambda hit: hit["score"], reverse=True)
    kept = []
    for hit in ordered:
        if any(abs(hit["cx"] - k["cx"]) <= WAVE_DEDUP_DISTANCE_PX for k in kept):
            continue
        kept.append(hit)
    return kept


def _parse_digits(digits: str):
    return int(digits) if digits.isdigit() else None


def save_region_preview(region_bgr, path: str) -> None:
    """Saves exactly what read_wave would see -- no annotations -- so a bad
    calibration (wrong region entirely) is visible at a glance, same
    convention as core.game_stats.save_region_preview."""
    cv2.imwrite(path, region_bgr)
