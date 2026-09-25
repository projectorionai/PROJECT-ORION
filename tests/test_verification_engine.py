"""
Deterministic tests for the VisualVerificationEngine (Improvement Pass,
Priority 1.1): the act → capture → verify → retry loop and the self-correcting
UIA re-location path, with the screen capture and pixel diff fully mocked so
no test touches the real display.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.verification import VerificationResult, VisualVerificationEngine


@pytest.fixture(autouse=True)
def _no_real_sleeps(monkeypatch):
    # click_text's bidirectional scroll-and-retry loop (Track D3) makes
    # several genuine asyncio.sleep(0.35) settle-pauses per failing
    # attempt — real behaviour worth keeping in production, but these
    # tests assert on logic/call-order, not wall-clock timing, so there's
    # nothing to gain from actually waiting.
    async def _instant(_seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", _instant)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubGrabber:
    def __init__(self):
        self.captures = 0

    def capture_array(self, region=None):
        self.captures += 1
        return object()          # opaque frame; the diff is mocked anyway


class _StubVision:
    """find_element returns queued elements (None = not found this attempt).
    find_element_via_ocr/nearby_candidates default to "nothing" (OCR
    unavailable) unless a test overrides them — matching the real
    VisionAgent's behaviour when no OcrEngine is attached."""

    def __init__(self, elements=(), ocr_elements=(), candidates=()):
        self.grabber = _StubGrabber()
        self._elements = list(elements)
        self._ocr_elements = list(ocr_elements)
        self._candidates = list(candidates)
        self.queries = []
        self.ocr_queries = []

    async def find_element(self, query, kinds="all"):
        self.queries.append((query, kinds))
        if not self._elements:
            return None
        return self._elements.pop(0)

    async def find_element_via_ocr(self, query, monitor=None):
        self.ocr_queries.append(query)
        if not self._ocr_elements:
            return None
        return self._ocr_elements.pop(0)

    async def nearby_candidates(self, query, kinds="all", limit=5):
        return list(self._candidates)[:limit]


class _StubControl:
    def __init__(self):
        self.last_action = {"region": (0, 0, 100, 100)}
        self.clicks = []
        self.scrolls = []

    def click(self, x, y):
        self.clicks.append((x, y))
        return ToolResult(f"clicked ({x},{y})")

    def scroll(self, amount):
        self.scrolls.append(amount)
        return ToolResult("scrolled")


class _StubDisplay:
    def clamp_to_desktop(self, x, y):
        return (max(0, int(x)), max(0, int(y)))


def _engine(vision=None, control=None, ratios=()):
    """Build an engine whose pixel diff yields the queued ratios (then 0.0)."""
    eng = VisualVerificationEngine(
        _StubBus(), control or _StubControl(), vision or _StubVision(), _StubDisplay(),
    )
    queue = list(ratios)
    eng._change_ratio = lambda before, after: (queue.pop(0) if queue else 0.0)
    return eng


# ── verify_action: the act → capture → diff loop ─────────────────────────────

def test_verify_action_confirms_visible_change():
    eng = _engine(ratios=[0.5])
    calls = []

    def action():
        calls.append(1)
        return ToolResult("pressed the button")

    result = asyncio.run(eng.verify_action(action, settle_s=0.0))
    assert isinstance(result, VerificationResult)
    assert result.ok and result.attempts == 1 and result.change_ratio == 0.5
    assert "verified" in result.detail
    assert calls == [1]                       # no needless retry on success


def test_verify_action_retries_then_reports_unverified():
    eng = _engine(ratios=[0.0, 0.0])
    calls = []

    def action():
        calls.append(1)
        return ToolResult("pressed")

    result = asyncio.run(eng.verify_action(action, settle_s=0.0, attempts=2))
    assert not result.ok
    assert result.attempts == 2 and len(calls) == 2
    assert "unverified" in result.detail


def test_verify_action_succeeds_on_retry():
    eng = _engine(ratios=[0.0, 0.9])
    result = asyncio.run(
        eng.verify_action(lambda: ToolResult("pressed"), settle_s=0.0, attempts=3))
    assert result.ok and result.attempts == 2 and result.change_ratio == 0.9


def test_non_repeatable_action_is_never_run_twice():
    """Typing/clicking: an unchanged screen is re-measured, never re-done —
    re-running typed the whole text a second time."""
    eng = _engine(ratios=[0.0, 0.0, 0.0])
    calls = []

    def action():
        calls.append(1)
        return ToolResult("typed")

    result = asyncio.run(eng.verify_action(action, settle_s=0.0, attempts=3,
                                           repeatable=False))
    assert not result.ok and calls == [1]
    assert "Not repeated" in result.detail


def test_non_repeatable_action_catches_a_slow_screen_without_repeating():
    eng = _engine(ratios=[0.0, 0.4])
    calls = []

    def action():
        calls.append(1)
        return ToolResult("clicked")

    result = asyncio.run(eng.verify_action(action, settle_s=0.0, repeatable=False))
    assert result.ok and calls == [1] and result.change_ratio == 0.4


def test_verify_action_short_circuits_on_failed_action():
    eng = _engine(ratios=[0.9])
    result = asyncio.run(
        eng.verify_action(lambda: ToolResult("backend offline", ok=False), settle_s=0.0))
    assert not result.ok and result.detail == "backend offline"
    assert result.attempts == 1               # no retry when the action itself failed


def test_verify_action_threshold_respected():
    # A change below min_change is not verification.
    eng = _engine(ratios=[0.001, 0.001])
    result = asyncio.run(
        eng.verify_action(lambda: ToolResult("ok"), settle_s=0.0,
                          min_change=0.002, attempts=2))
    assert not result.ok


# ── click_element: self-correcting coordinates ────────────────────────────────

def _element(name, cx, cy):
    return {"name": name, "role": "button", "center": (cx, cy),
            "rect": (cx - 20, cy - 10, 40, 20)}


def test_click_element_clicks_centre_and_verifies():
    vision = _StubVision([_element("Save", 100, 200)])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.5])
    result = asyncio.run(eng.click_element("Save"))
    assert result.ok and control.clicks == [(100, 200)]
    assert "verified" in result.text


def test_click_element_relocates_moved_element_on_retry():
    # The element moves between attempts; the retry must click the NEW
    # coordinates from the live UIA tree, not the stale first position.
    vision = _StubVision([_element("Save", 100, 200), _element("Save", 300, 400)])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.0, 0.9])
    result = asyncio.run(eng.click_element("Save", attempts=2))
    assert result.ok
    assert control.clicks == [(100, 200), (300, 400)]


def test_click_element_missing_element_fails_informatively():
    eng = _engine(_StubVision([]), ratios=[0.9])
    result = asyncio.run(eng.click_element("Ghost Button"))
    assert not result.ok and "Could not find" in result.text


def test_click_element_gives_up_after_bounded_attempts():
    vision = _StubVision([_element("Noop", 10, 10)] * 3)
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.0, 0.0, 0.0])
    result = asyncio.run(eng.click_element("Noop", attempts=3))
    assert not result.ok and len(control.clicks) == 3
    assert "could not visually confirm" in result.text


def test_click_element_propagates_control_failure():
    vision = _StubVision([_element("Save", 100, 200)])
    control = _StubControl()
    control.click = lambda x, y: ToolResult("autonomy disabled", ok=False)
    eng = _engine(vision, control, ratios=[0.9])
    result = asyncio.run(eng.click_element("Save"))
    assert not result.ok and result.text == "autonomy disabled"


# ── click_text: scroll-and-retry navigation (Track D3: both directions) ──────

def test_click_text_scrolls_down_first_when_not_found():
    vision = _StubVision([])           # never found by UIA or OCR
    control = _StubControl()
    eng = _engine(vision, control, ratios=[])
    result = asyncio.run(eng.click_text("Below The Fold", attempts=1))
    assert not result.ok
    assert control.scrolls
    assert control.scrolls[0] < 0      # first phase scrolls down


def test_click_text_also_searches_upward_after_scrolling_down_fails():
    vision = _StubVision([])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[])
    asyncio.run(eng.click_text("Above The Fold", attempts=1))
    # the old version only ever scrolled down; an upward phase must exist too.
    assert any(s > 0 for s in control.scrolls)


def test_click_text_returns_to_origin_before_searching_upward():
    vision = _StubVision([])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[])
    asyncio.run(eng.click_text("Ghost Section", attempts=1))
    # 5 down-steps of -4 (net -20) must be followed by a single +20 step
    # back to the origin before the upward search begins.
    assert sum(control.scrolls[:5]) == -20
    assert control.scrolls[5] == 20


def test_click_text_restores_scroll_position_on_total_failure():
    vision = _StubVision([])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[])
    asyncio.run(eng.click_text("Ghost Section", attempts=1))
    assert sum(control.scrolls) == 0   # net displacement restored to zero


def test_click_text_stops_scrolling_once_found_below_the_fold():
    # _locate_and_click tries kinds button/link/menu/all in order per
    # attempt: the pre-scroll attempt must exhaust all 4 as None, then
    # after exactly one scroll(-4) the next attempt's "button" check finds
    # it (queued twice: once for the probe, once for click_element's own
    # internal re-locate of the same kind).
    found = _element("Deep Link", 50, 60)
    vision = _StubVision([None, None, None, None, found, found])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.5])
    result = asyncio.run(eng.click_text("Deep Link", attempts=1))
    assert result.ok
    assert control.scrolls == [-4]


# ── click_via_ocr / OCR fallback (Mark XXI, Track D2) ────────────────────────

def test_locate_and_click_falls_back_to_ocr_when_uia_finds_nothing():
    vision = _StubVision(elements=[], ocr_elements=[_element("Weird Icon", 15, 15)])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.5])
    result = asyncio.run(eng.click_text("Weird Icon", attempts=1))
    assert result.ok
    assert vision.ocr_queries == ["Weird Icon"]
    assert control.clicks == [(15, 15)]


def test_ocr_fallback_is_not_tried_when_uia_already_found_it():
    # _locate_and_click's probe and click_element's own internal re-locate
    # both call find_element for the winning kind, so the stub needs two
    # copies queued for a single successful click via that path.
    saved = _element("Save", 10, 10)
    vision = _StubVision(elements=[saved, saved], ocr_elements=[_element("Save", 99, 99)])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.5])
    result = asyncio.run(eng.click_text("Save", attempts=1))
    assert result.ok
    assert vision.ocr_queries == []            # OCR never consulted
    assert control.clicks == [(10, 10)]         # the UIA coordinates were used


def test_click_via_ocr_relocates_on_retry_like_click_element():
    vision = _StubVision(ocr_elements=[_element("Icon", 10, 10), _element("Icon", 40, 40)])
    control = _StubControl()
    eng = _engine(vision, control, ratios=[0.0, 0.9])
    result = asyncio.run(eng.click_via_ocr("Icon", attempts=2))
    assert result.ok
    assert control.clicks == [(10, 10), (40, 40)]


def test_click_via_ocr_fails_informatively_when_ocr_also_finds_nothing():
    vision = _StubVision(elements=[], ocr_elements=[])
    eng = _engine(vision, _StubControl())
    result = asyncio.run(eng.click_via_ocr("Ghost", attempts=1))
    assert not result.ok
    assert "Ghost" in result.text


# ── informative failure messages (Mark XXI, Track D5) ─────────────────────────

def test_not_found_message_lists_nearby_candidates():
    vision = _StubVision(elements=[], ocr_elements=[],
                         candidates=["Save As...", "Save All", "Save Copy"])
    eng = _engine(vision, _StubControl())
    result = asyncio.run(eng.click_text("Save", attempts=1))
    assert not result.ok
    assert "Save As" in result.text
    assert "nearest matches" in result.text.lower()


def test_not_found_message_has_no_candidate_clause_when_none_exist():
    vision = _StubVision(elements=[], ocr_elements=[], candidates=[])
    eng = _engine(vision, _StubControl())
    result = asyncio.run(eng.click_text("Ghost", attempts=1))
    assert not result.ok
    assert "nearest matches" not in result.text.lower()


def test_not_found_message_never_raises_when_control_has_no_foreground_title(monkeypatch):
    vision = _StubVision(elements=[], ocr_elements=[])
    control = _StubControl()   # has no _foreground_title at all
    eng = _engine(vision, control)
    result = asyncio.run(eng.click_text("Ghost", attempts=1))
    assert not result.ok       # must not raise despite the missing method


# ── navigation trace (Mark XXI, Track D7) ─────────────────────────────────────

def test_successful_click_is_recorded_to_the_trace():
    vision = _StubVision([_element("Save", 100, 200)])
    eng = _engine(vision, _StubControl(), ratios=[0.5])
    asyncio.run(eng.click_element("Save"))
    recent = eng.trace.recent(5)
    assert len(recent) == 1
    assert recent[0]["query"] == "Save"
    assert recent[0]["strategy"] == "uia"
    assert recent[0]["found"] is True
    assert recent[0]["verified"] is True


def test_failed_click_is_recorded_with_candidates():
    vision = _StubVision(elements=[], ocr_elements=[], candidates=["Save As"])
    eng = _engine(vision, _StubControl())
    asyncio.run(eng.click_text("Save", attempts=1))
    failures = eng.trace.failures(5)
    assert failures
    assert failures[-1]["query"] == "Save"
    assert failures[-1]["strategy"] == "ocr"


def test_an_engine_with_no_trace_argument_still_records_to_its_own():
    vision = _StubVision([_element("Save", 100, 200)])
    eng = VisualVerificationEngine(_StubBus(), _StubControl(), vision, _StubDisplay())
    eng._change_ratio = lambda before, after: 0.5
    asyncio.run(eng.click_element("Save"))
    assert len(eng.trace.recent(5)) == 1


# ── real pixel diff (only when cv2/numpy are installed) ───────────────────────

def test_change_ratio_with_real_frames():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    eng = VisualVerificationEngine(_StubBus(), _StubControl(), _StubVision(), _StubDisplay())
    base = np.zeros((50, 50, 3), dtype=np.uint8)
    changed = base.copy()
    changed[:25, :, :] = 255           # half the frame flips
    assert eng._change_ratio(base, base) == 0.0
    assert eng._change_ratio(base, changed) == pytest.approx(0.5, abs=0.05)


def test_change_ratio_handles_none_frames():
    eng = VisualVerificationEngine(_StubBus(), _StubControl(), _StubVision(), _StubDisplay())
    assert eng._change_ratio(None, None) == 0.0
