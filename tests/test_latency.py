"""
Latency observability (Mark XXVI) — ORION detecting his own lag.

The value of this module is entirely in whether it catches a REAL block and
names the REAL culprit, so that is what these tests do: they block the event
loop on purpose, from a named function, and assert the watchdog both notices and
points at that function. A monitoring tool that reports "something was slow" is
worth nothing.

The duration test matters too. An earlier version reported the threshold value
(160 ms) for a 450 ms freeze — a detection, but a misleading number, and the
number is the point.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.latency import (  # noqa: E402
    DEFAULT_THRESHOLD_MS,
    LatencyTracker,
    Span,
    Stall,
    StallDetector,
    _short_frame,
    percentile,
    report,
)


# ── percentiles ───────────────────────────────────────────────────────────────

def test_percentile_of_an_empty_list_is_zero():
    assert percentile([], 0.95) == 0.0


def test_percentile_picks_the_nearest_rank():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) in (50.0, 51.0)
    assert percentile(values, 0.95) in (95.0, 96.0)
    assert percentile(values, 1.0) == 100.0
    assert percentile(values, 0.0) == 1.0


def test_percentile_of_one_value():
    assert percentile([7.0], 0.95) == 7.0


# ── the tracker ───────────────────────────────────────────────────────────────

def test_a_span_records_elapsed_time():
    tracker = LatencyTracker()
    with tracker.span("work"):
        time.sleep(0.02)
    spans = tracker.spans("work")
    assert len(spans) == 1
    assert 10.0 < spans[0].ms < 500.0


def test_a_span_is_recorded_even_when_the_block_raises():
    """A slow failure is still a slow turn; hiding it flatters the numbers."""
    tracker = LatencyTracker()
    with pytest.raises(ValueError):
        with tracker.span("doomed"):
            raise ValueError("boom")
    assert len(tracker.spans("doomed")) == 1


def test_the_ring_is_bounded():
    tracker = LatencyTracker(capacity=10)
    for i in range(50):
        tracker.record("x", float(i))
    assert len(tracker.spans()) == 10
    # the OLDEST are discarded, so the newest value survives
    assert max(s.ms for s in tracker.spans()) == 49.0


def test_summary_reports_percentiles_per_name():
    tracker = LatencyTracker()
    for value in range(1, 101):
        tracker.record("db", float(value))
    for value in (5.0, 5.0):
        tracker.record("net", value)
    summary = {s.name: s for s in tracker.summary()}
    assert summary["db"].count == 100
    assert summary["db"].worst == 100.0
    assert summary["db"].p95 >= summary["db"].p50
    assert summary["net"].count == 2


def test_summary_is_ordered_by_p95_worst_first():
    tracker = LatencyTracker()
    tracker.record("fast", 1.0)
    tracker.record("slow", 900.0)
    tracker.record("medium", 50.0)
    assert [s.name for s in tracker.summary()] == ["slow", "medium", "fast"]


def test_slowest_returns_the_worst_spans():
    tracker = LatencyTracker()
    for value in (1.0, 99.0, 50.0):
        tracker.record("x", value)
    assert [s.ms for s in tracker.slowest(2)] == [99.0, 50.0]


def test_an_empty_tracker_reports_honestly():
    assert "No timings recorded" in LatencyTracker().report()


def test_clear_empties_the_ring():
    tracker = LatencyTracker()
    tracker.record("x", 1.0)
    tracker.clear()
    assert tracker.spans() == []


def test_the_tracker_is_thread_safe():
    import threading
    tracker = LatencyTracker(capacity=5000)

    def hammer():
        for _ in range(200):
            tracker.record("concurrent", 1.0)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(tracker.spans("concurrent")) == 1600


def test_span_line_renders_detail():
    assert "why" in Span("n", 1.0, time.time(), "why").line()


# ── frame shortening ──────────────────────────────────────────────────────────

def test_a_traceback_line_becomes_function_at_file_line():
    raw = '  File "C:\\\\Users\\\\x\\\\orion_core\\\\audio.py", line 42, in read_frames\n'
    assert _short_frame(raw) == "read_frames() at audio.py:42"


def test_an_unparseable_frame_degrades_gracefully():
    assert _short_frame("not a traceback line") == "not a traceback line"


# ── stalls ────────────────────────────────────────────────────────────────────

def test_a_stall_with_no_frames_says_unknown():
    assert Stall(ms=100.0, at=time.time()).culprit() == "unknown"


def test_a_stall_skips_plumbing_frames_and_names_real_code():
    stall = Stall(ms=300.0, at=time.time(), frames=[
        '  File "/x/orion_core/app.py", line 10, in run_application\n',
        '  File "/x/orion_core/audio.py", line 42, in read_frames\n',
        '  File "/x/asyncio/base_events.py", line 5, in run_forever\n',
        '  File "/x/orion_core/latency.py", line 1, in _capture\n',
    ])
    assert stall.culprit() == "read_frames() at audio.py:42"


def test_a_stall_of_only_plumbing_still_returns_something():
    stall = Stall(ms=300.0, at=time.time(), frames=[
        '  File "/x/asyncio/base_events.py", line 5, in run_forever\n'])
    assert "run_forever" in stall.culprit()


def test_record_stall_feeds_the_tracker_and_the_callback():
    tracker = LatencyTracker()
    seen: list[Stall] = []
    detector = StallDetector(tracker, on_stall=seen.append)
    detector.record_stall(400.0, ['  File "/x/a.py", line 1, in slow_thing\n'])
    assert len(seen) == 1
    assert tracker.spans("event-loop stall")[0].ms == 400.0
    assert "slow_thing" in tracker.spans("event-loop stall")[0].detail


def test_a_broken_stall_callback_does_not_propagate():
    """The watchdog must survive a reporter that raises, or a logging bug
    silently disables lag detection."""
    def boom(_stall):
        raise RuntimeError("reporter is broken")
    detector = StallDetector(LatencyTracker(), on_stall=boom)
    detector.record_stall(400.0, [])          # must not raise
    assert len(detector.stalls) == 1


def test_the_stall_ring_is_bounded():
    detector = StallDetector(LatencyTracker(), capacity=5)
    for _ in range(20):
        detector.record_stall(300.0, [])
    assert len(detector.stalls) == 5


def test_a_clean_run_reports_no_stalls():
    assert "No event-loop stalls" in StallDetector(LatencyTracker()).report()


def test_the_report_names_the_worst_offenders():
    detector = StallDetector(LatencyTracker())
    detector.record_stall(120.0, ['  File "/x/a.py", line 1, in mild\n'])
    detector.record_stall(900.0, ['  File "/x/b.py", line 2, in terrible\n'])
    text = detector.report()
    assert "terrible() at b.py:2" in text
    assert text.index("terrible") < text.index("mild"), "worst must come first"


def test_the_default_threshold_ignores_ordinary_jitter():
    assert 100.0 <= DEFAULT_THRESHOLD_MS <= 500.0


# ── the real thing: catch an actual blocked loop ─────────────────────────────

def the_deliberately_blocking_function():
    """Stands in for psutil.cpu_percent(interval=0.2) — a real past ORION bug."""
    time.sleep(0.45)


async def test_the_watchdog_catches_a_real_block_and_names_the_function():
    tracker = LatencyTracker()
    detector = StallDetector(tracker, threshold_ms=150.0)
    detector.start()
    try:
        await asyncio.sleep(0.25)
        assert not detector.stalls, "a healthy loop must not be reported as stalled"

        the_deliberately_blocking_function()
        await asyncio.sleep(0.25)

        assert detector.stalls, "a 450 ms block on the loop went undetected"
        stall = detector.stalls[-1]
        assert any("the_deliberately_blocking_function" in f for f in stall.frames), (
            "the watchdog detected a stall but could not name it: " + stall.culprit())
        # duration must reflect the WHOLE block, not the moment of detection
        assert stall.ms > 350.0, (
            "reported %.0f ms for a 450 ms block — it is reporting the "
            "threshold, not the stall" % stall.ms)
    finally:
        detector.stop()


async def test_a_responsive_loop_produces_no_stalls():
    detector = StallDetector(LatencyTracker(), threshold_ms=150.0)
    detector.start()
    try:
        for _ in range(12):
            await asyncio.sleep(0.03)          # yields constantly
        assert list(detector.stalls) == []
    finally:
        detector.stop()


async def test_awaiting_a_long_sleep_is_not_a_stall():
    """An awaited sleep hands control back — the loop is free. Flagging that
    would make the detector cry wolf on every idle moment."""
    detector = StallDetector(LatencyTracker(), threshold_ms=150.0)
    detector.start()
    try:
        await asyncio.sleep(0.5)
        assert list(detector.stalls) == []
    finally:
        detector.stop()


async def test_start_is_idempotent_and_stop_is_clean():
    detector = StallDetector(LatencyTracker())
    detector.start()
    first = detector._thread
    detector.start()
    assert detector._thread is first
    assert detector.running
    detector.stop()
    assert not detector.running


def test_stop_without_start_is_safe():
    StallDetector(LatencyTracker()).stop()


# ── wiring ────────────────────────────────────────────────────────────────────

def test_the_module_level_report_combines_both_views():
    text = report()
    assert "stall" in text.lower()


def test_the_diagnostics_tool_routes_latency():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "dispatch_files.py"
           ).read_text(encoding="utf-8")
    assert '"latency"' in src and "latency_report" in src


def test_the_schema_advertises_the_latency_action():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "diagnostics")
    assert "latency" in tool["description"]


def test_the_app_starts_the_watchdog_and_can_disable_it():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py"
           ).read_text(encoding="utf-8")
    assert "_stall_detector.start()" in src
    assert "ORION_STALL_WATCH" in src, "there must be an escape hatch"


async def test_the_watchdog_costs_almost_nothing_when_idle():
    """It is on by default, so it has to be free. The heartbeat is a float
    store and the watch thread sleeps; neither should be measurable."""
    detector = StallDetector(LatencyTracker(), threshold_ms=1000.0)
    detector.start()
    try:
        start = time.perf_counter()
        for _ in range(200):
            await asyncio.sleep(0)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, "200 loop turns took %.2f s with the watchdog on" % elapsed
    finally:
        detector.stop()
