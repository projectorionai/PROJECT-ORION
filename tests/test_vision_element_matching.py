"""
Tests for VisionAgent's UIA element matching (_find_element_sync,
_element_score, nearby_candidates) and the OCR click fallback
(find_element_via_ocr) — Mark XXI, Tracks D1/D2/D5.

_detect_elements_sync is monkeypatched to return synthetic UIA-shaped
elements, so no real pywinauto/accessibility-tree dependency is needed in
CI; these tests exercise the real matching/ranking/candidate logic, which
had zero coverage before this pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.vision import VisionAgent


class _Signal:
    def emit(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


def _el(name: str, x=0, y=0, w=40, h=20) -> dict:
    return {"role": "Button", "name": name, "rect": (x, y, w, h),
            "center": (x + w // 2, y + h // 2), "enabled": True}


def _vision(elements: list[dict]) -> VisionAgent:
    v = VisionAgent(_StubBus(), grabber=None, file_intel=None)
    v._detect_elements_sync = lambda kinds, max_elements=400: list(elements)
    return v


# ── _find_element_sync (Track D1: fuzzy/normalised matching) ───────────────

def test_finds_an_exact_match():
    v = _vision([_el("Save"), _el("Cancel")])
    result = v._find_element_sync("Save", "all")
    assert result["name"] == "Save"


def test_finds_a_control_with_a_mnemonic_ampersand():
    v = _vision([_el("&Save"), _el("&Cancel")])
    result = v._find_element_sync("Save", "all")
    assert result["name"] == "&Save"


def test_finds_a_control_with_a_trailing_ellipsis():
    v = _vision([_el("Save As…"), _el("Save All")])
    result = v._find_element_sync("Save As", "all")
    assert result["name"] == "Save As…"


def test_prefers_exact_match_over_substring_match():
    v = _vision([_el("Save As"), _el("Save")])
    result = v._find_element_sync("Save", "all")
    assert result["name"] == "Save"


def test_prefers_startswith_over_bare_substring():
    v = _vision([_el("Undo Save"), _el("Save As")])
    result = v._find_element_sync("Save", "all")
    assert result["name"] == "Save As"   # starts with "save"


def test_falls_back_to_token_overlap():
    v = _vision([_el("Export Report To PDF")])
    result = v._find_element_sync("PDF Report", "all")
    assert result is not None
    assert result["name"] == "Export Report To PDF"


def test_returns_none_when_nothing_matches_at_all():
    v = _vision([_el("Cancel"), _el("Close")])
    assert v._find_element_sync("Save", "all") is None


def test_returns_none_for_a_blank_query():
    v = _vision([_el("Save")])
    assert v._find_element_sync("   ", "all") is None


def test_returns_none_when_no_elements_detected():
    v = _vision([])
    assert v._find_element_sync("Save", "all") is None


# ── nearby_candidates / _element_score (Track D5) ───────────────────────────

def test_nearby_candidates_ranks_the_closest_matches_first():
    v = _vision([_el("Save All"), _el("Save As"), _el("Cancel")])
    names = v._nearby_candidates_sync("Save", "all", limit=5)
    assert names[:2] == ["Save All", "Save As"] or names[:2] == ["Save As", "Save All"]
    assert "Cancel" in names


def test_nearby_candidates_respects_the_limit():
    v = _vision([_el(f"Item {i}") for i in range(10)])
    names = v._nearby_candidates_sync("Item", "all", limit=3)
    assert len(names) == 3


def test_nearby_candidates_deduplicates_repeated_names():
    v = _vision([_el("Save"), _el("Save"), _el("Cancel")])
    names = v._nearby_candidates_sync("Save", "all", limit=5)
    assert names.count("Save") == 1


def test_nearby_candidates_with_no_elements_is_empty():
    v = _vision([])
    assert v._nearby_candidates_sync("Save", "all") == []


def test_nearby_candidates_with_blank_query_is_empty():
    v = _vision([_el("Save")])
    assert v._nearby_candidates_sync("", "all") == []


# ── find_element_via_ocr (Track D2) ─────────────────────────────────────────

class _StubOcrEngine:
    def __init__(self, box=None, available=True):
        self._box = box
        self.available = available
        self.queries = []

    def find_text_box(self, image, query):
        self.queries.append(query)
        return self._box


class _StubGrabber:
    def __init__(self, bounds=(0, 0, 1920, 1080)):
        self._bounds = bounds
        self.captured_monitor = "unset"

    def monitor_bounds(self, monitor=None):
        return self._bounds

    def capture_image(self, max_side=0, monitor=None):
        self.captured_monitor = monitor
        return object()


async def _find_via_ocr(v, query, monitor=None):
    return await v.find_element_via_ocr(query, monitor=monitor)


def test_find_element_via_ocr_returns_none_without_an_engine():
    import asyncio
    v = VisionAgent(_StubBus(), grabber=_StubGrabber(), file_intel=None)
    assert asyncio.run(_find_via_ocr(v, "Save")) is None


def test_find_element_via_ocr_returns_none_when_engine_unavailable():
    import asyncio
    v = VisionAgent(_StubBus(), grabber=_StubGrabber(), file_intel=None)
    v.attach_ocr_engine(_StubOcrEngine(box=(10, 10, 30, 20), available=False))
    assert asyncio.run(_find_via_ocr(v, "Save")) is None


def test_find_element_via_ocr_translates_to_virtual_desktop_coordinates():
    import asyncio
    grabber = _StubGrabber(bounds=(1920, 0, 1920, 1080))   # second monitor, offset right
    v = VisionAgent(_StubBus(), grabber=grabber, file_intel=None)
    v.attach_ocr_engine(_StubOcrEngine(box=(10, 20, 40, 15)))
    element = asyncio.run(_find_via_ocr(v, "Save"))
    assert element is not None
    assert element["rect"] == (1930, 20, 40, 15)          # 10 + 1920 offset
    assert element["center"] == (1930 + 20, 20 + 7)
    assert element["role"] == "OCR"


def test_find_element_via_ocr_returns_none_when_ocr_finds_nothing():
    import asyncio
    v = VisionAgent(_StubBus(), grabber=_StubGrabber(), file_intel=None)
    v.attach_ocr_engine(_StubOcrEngine(box=None))
    assert asyncio.run(_find_via_ocr(v, "Ghost")) is None


def test_find_element_via_ocr_never_raises_on_a_broken_grabber():
    import asyncio

    class _BrokenGrabber:
        def monitor_bounds(self, monitor=None):
            raise RuntimeError("no display")

    v = VisionAgent(_StubBus(), grabber=_BrokenGrabber(), file_intel=None)
    v.attach_ocr_engine(_StubOcrEngine(box=(1, 1, 1, 1)))
    assert asyncio.run(_find_via_ocr(v, "Save")) is None
