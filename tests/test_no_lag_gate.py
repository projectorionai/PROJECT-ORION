"""
The no-lag gate (Mark XXVI) — "ORION must not stutter", enforced automatically.

Every previous lag fix in this project was reactive: the user felt a stutter, a
human profiled by hand, a cause was found. Nothing stopped the next one. This
file is the standing check.

It runs ORION's real per-turn work inside a running ``StallDetector`` and fails
if the event loop is ever blocked past the threshold. That is the same watchdog
that ships in the product, pointed at the code under test, so it detects exactly
what a user would feel — and names the function, so a failure here is a bug
report rather than a mystery.

**Why the thresholds are what they are.** The gate asserts on a per-operation
budget AND on total loop blockage, both deliberately loose — roughly an order of
magnitude above the measured values. A tight budget on a machine already running
five thousand other tests is a flake generator, and a flaky performance gate is
worse than none: it gets disabled. Loose bounds still catch every regression
this project has actually had, because those were 10x-100x, not 20%.

Historical regressions this gate would have caught, all of which shipped:

    psutil.cpu_percent(interval=0.2)        200 ms, every resource check
    SQLite default journal + full fsync     2.81 ms per write, hundreds of them
    tool pre-filter rebuilding its index    4.34 ms, every single turn
    the OCR probe reconstructing engines    759 ms, every diagnostics poll
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.latency import LatencyTracker, StallDetector  # noqa: E402

#: Loose on purpose — see the module docstring. A real regression is 10x, not 20%.
STALL_THRESHOLD_MS = 300.0

#: Per-operation ceilings. Measured values are in the comments; the assertion
#: sits far above them so only a genuine regression trips it.
BUDGETS_MS = {
    "reflex match": 5.0,             # measured ~0.003
    "tool pre-filter": 30.0,         # measured 0.093 (was 4.34)
    "flashcard due query": 40.0,     # measured 0.56 on a 2k deck
    "focus cycle": 30.0,             # measured 0.10 (four commits)
    "capability matrix": 60.0,       # measured 3.9 warm (was 759)
    "learned reflex match": 5.0,     # measured 0.002
}


def _assert_no_stalls(detector: StallDetector, what: str) -> None:
    if not detector.stalls:
        return
    worst = max(detector.stalls, key=lambda s: s.ms)
    pytest.fail(
        "%s blocked the event loop for %.0f ms — a user would feel this as a "
        "stutter in the face and the voice.\n  culprit: %s\n%s"
        % (what, worst.ms, worst.culprit(), detector.report()))


async def _yielding(iterations: int, work) -> None:
    """Run `work` repeatedly, yielding between iterations like a real turn."""
    for _ in range(iterations):
        work()
        await asyncio.sleep(0)


# ── the per-turn path ────────────────────────────────────────────────────────

async def test_the_per_turn_path_never_blocks_the_loop():
    """Reflex matching plus tool pre-filtering happen on every single turn,
    before ORION can begin answering."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.reflex import match_reflex
    from orion_core.tool_resolver import ResolverState, reset_index, resolve

    reset_index()
    state = ResolverState()
    phrases = ["start a focus block", "what's due", "why were you so slow",
               "tell me something interesting about the brain", "list my plugins"]

    detector = StallDetector(LatencyTracker(), threshold_ms=STALL_THRESHOLD_MS)
    detector.start()
    try:
        resolve("warm", state, TOOL_DECLARATIONS)          # build the index once
        await _yielding(60, lambda: [match_reflex(p) for p in phrases])
        await _yielding(60, lambda: resolve(phrases[0], state, TOOL_DECLARATIONS))
        _assert_no_stalls(detector, "the per-turn path")
    finally:
        detector.stop()


async def test_reflex_matching_is_inside_budget():
    from orion_core.reflex import match_reflex
    match_reflex("warm")
    start = time.perf_counter()
    for _ in range(500):
        match_reflex("tell me something interesting about the brain")
    per_ms = (time.perf_counter() - start) * 1000 / 500
    assert per_ms < BUDGETS_MS["reflex match"], (
        "reflex matching costs %.3f ms" % per_ms)


async def test_tool_pre_filtering_is_inside_budget():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    from orion_core.tool_resolver import ResolverState, reset_index, resolve
    reset_index()
    state = ResolverState()
    resolve("warm", state, TOOL_DECLARATIONS)
    start = time.perf_counter()
    for _ in range(200):
        resolve("start a focus block on revision", state, TOOL_DECLARATIONS)
    per_ms = (time.perf_counter() - start) * 1000 / 200
    assert per_ms < BUDGETS_MS["tool pre-filter"], (
        "the tool pre-filter costs %.2f ms a turn over %d tools"
        % (per_ms, len(TOOL_DECLARATIONS)))


# ── database work on the loop ────────────────────────────────────────────────

async def test_a_burst_of_database_writes_never_blocks_the_loop(tmp_path):
    """The fsync-per-write regression: 200 commits cost 562 ms before the WAL
    policy, and every millisecond landed on the thread drawing the face."""
    from orion_core.focus import FocusEngine, FocusStore

    engine = FocusEngine(FocusStore(tmp_path / "focus.db"))
    detector = StallDetector(LatencyTracker(), threshold_ms=STALL_THRESHOLD_MS)
    detector.start()
    try:
        start = time.perf_counter()
        for i in range(40):
            engine.start(label="cycle %d" % i, minutes=25)
            engine.complete()
            await asyncio.sleep(0)
        elapsed_ms = (time.perf_counter() - start) * 1000 / 40
        _assert_no_stalls(detector, "a burst of focus writes")
        assert elapsed_ms < BUDGETS_MS["focus cycle"], (
            "a focus cycle costs %.2f ms — is the SQLite policy still applied?"
            % elapsed_ms)
    finally:
        detector.stop()
        engine.store.close()


async def test_reading_due_flashcards_is_inside_budget(tmp_path):
    """This runs on every review; it was a full table scan filtered in Python."""
    from orion_core.study import StudyEngine, StudyStore

    engine = StudyEngine(StudyStore(tmp_path / "study.db"))
    for i in range(600):
        engine.add("deck", "front %d" % i, "back %d" % i)

    detector = StallDetector(LatencyTracker(), threshold_ms=STALL_THRESHOLD_MS)
    detector.start()
    try:
        engine.due()
        start = time.perf_counter()
        for _ in range(30):
            engine.due()
            await asyncio.sleep(0)
        per_ms = (time.perf_counter() - start) * 1000 / 30
        _assert_no_stalls(detector, "reading due flashcards")
        assert per_ms < BUDGETS_MS["flashcard due query"], (
            "due() costs %.2f ms on a 600-card deck — has it gone back to "
            "scanning in Python?" % per_ms)
    finally:
        detector.stop()
        engine.store.close()


# ── polled diagnostics ───────────────────────────────────────────────────────

async def test_the_capability_matrix_stays_cheap_to_poll():
    """Diagnostics and the proactive loop poll this; it cost 759 ms a call when
    it rebuilt an OCR engine each time."""
    from orion_core.capability_health import capability_matrix

    capability_matrix(deep=False)                       # warm the probe cache
    detector = StallDetector(LatencyTracker(), threshold_ms=STALL_THRESHOLD_MS)
    detector.start()
    try:
        start = time.perf_counter()
        for _ in range(10):
            capability_matrix(deep=False)
            await asyncio.sleep(0)
        per_ms = (time.perf_counter() - start) * 1000 / 10
        _assert_no_stalls(detector, "polling the capability matrix")
        assert per_ms < BUDGETS_MS["capability matrix"], (
            "the capability matrix costs %.1f ms a poll" % per_ms)
    finally:
        detector.stop()


# ── learned reflexes ─────────────────────────────────────────────────────────

async def test_learned_reflex_lookup_never_blocks_the_loop(tmp_path):
    from orion_core.reflex_learning import ReflexLearner

    learner = ReflexLearner(tmp_path / "learn.db")
    for i in range(150):
        phrase = "saved phrase number %d" % i
        for _ in range(3):
            learner.observe(phrase, "web_search", {})
        learner.promote(phrase)

    detector = StallDetector(LatencyTracker(), threshold_ms=STALL_THRESHOLD_MS)
    detector.start()
    try:
        learner.match("warm")
        start = time.perf_counter()
        for _ in range(300):
            learner.match("saved phrase number 75")
            learner.match("something that was never learned at all")
        per_ms = (time.perf_counter() - start) * 1000 / 600
        _assert_no_stalls(detector, "learned reflex lookup")
        assert per_ms < BUDGETS_MS["learned reflex match"], (
            "a learned lookup costs %.3f ms" % per_ms)
    finally:
        detector.stop()
        learner.close()


# ── the face, which paints on this very thread ───────────────────────────────

def test_the_face_paint_stays_inside_a_frame(qapp_or_skip):
    """Not run under the detector: rendering to a QImage in a test is a tight
    synchronous loop by design, so 'the loop was blocked' is meaningless here.
    The budget is the honest measure."""
    from PyQt6.QtGui import QColor, QImage

    from orion_core.gui.face import HologramFace

    face = HologramFace()
    face.resize(600, 600)
    face.isVisible = lambda: True
    face._blink_in = 10_000
    face.set_state("SPEAKING")
    for _ in range(20):
        face._tick()
    img = QImage(600, 600, QImage.Format.Format_RGB32)
    face.render(img)                                   # warm

    start = time.perf_counter()
    for _ in range(12):
        img.fill(QColor("#050608"))
        face.render(img)
    per_frame = (time.perf_counter() - start) * 1000 / 12
    # 4.6 ms measured at 600px after the batching pass (was 7.93).
    assert per_frame < 25.0, (
        "the face costs %.1f ms a frame — at 30 fps that is %.0f%% of the "
        "budget, on the same thread as the audio callback"
        % (per_frame, per_frame / 33.3 * 100))


@pytest.fixture(scope="module")
def qapp_or_skip():
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── the gate's own integrity ─────────────────────────────────────────────────

async def test_the_gate_would_actually_fail_on_a_real_stall():
    """A performance gate that cannot fail is decoration. Prove it fires."""
    detector = StallDetector(LatencyTracker(), threshold_ms=100.0)
    detector.start()
    try:
        time.sleep(0.35)                       # a deliberate block
        await asyncio.sleep(0.2)
        assert detector.stalls, "the gate cannot detect a stall — it is decorative"
        # pytest.fail raises Failed, which derives from BaseException.
        with pytest.raises(BaseException):
            _assert_no_stalls(detector, "the deliberate block")
    finally:
        detector.stop()


def test_every_budget_has_a_measured_value_recorded():
    """Each budget must be justified by a measurement, not a guess."""
    src = Path(__file__).read_text(encoding="utf-8")
    block = src[src.index("BUDGETS_MS = {"):src.index("def _assert_no_stalls")]
    for name in BUDGETS_MS:
        line = next(l for l in block.splitlines() if repr(name)[1:-1] in l)
        assert "measured" in line, "budget %r has no recorded measurement" % name
