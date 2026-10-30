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
    LOG_THRESHOLD_MS,
    REENTRIES,
    LatencyTracker,
    Span,
    Stall,
    StallDetector,
    StallReporter,
    _short_frame,
    install_reentry_watch,
    percentile,
    reentry_culprit,
    reentry_line,
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


_LATENCY_QT = None


def test_heartbeat_survives_a_nested_qt_loop_inside_a_task(monkeypatch):
    """A modal Qt pump must not destroy the observer's next heartbeat."""
    import qasync
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication

    global _LATENCY_QT
    _LATENCY_QT = QApplication.instance() or QApplication([])
    loop = qasync.QEventLoop(_LATENCY_QT)
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    detector = StallDetector(LatencyTracker())
    ticks = []
    original_beat = detector.beat

    def observe_beat():
        original_beat()
        ticks.append(time.monotonic())

    monkeypatch.setattr(detector, "beat", observe_beat)

    async def flow():
        detector.start(loop)
        try:
            await asyncio.sleep(0.12)
            assert ticks, "the observer never started"
            nested = QEventLoop()
            QTimer.singleShot(180, nested.quit)
            nested.exec()  # still inside this task's step: the reported fault
            resumed_at = time.monotonic()
            await asyncio.sleep(0.16)
            assert not errors, [str(e.get("exception")) for e in errors]
            assert any(tick > resumed_at for tick in ticks), (
                "heartbeat died during nested Qt dispatch")
        finally:
            detector.stop()
        stopped_ticks = len(ticks)
        await asyncio.sleep(0.12)
        assert len(ticks) == stopped_ticks, "stop left an active heartbeat"

    with loop:
        loop.run_until_complete(flow())


async def test_stopped_heartbeat_does_not_leak_into_a_restart(monkeypatch):
    detector = StallDetector(LatencyTracker())
    ticks = []
    original_beat = detector.beat

    def observe_beat():
        original_beat()
        ticks.append(time.monotonic())

    monkeypatch.setattr(detector, "beat", observe_beat)
    detector.start()
    try:
        await asyncio.sleep(0.13)
        assert ticks
        detector.stop()
        stopped_ticks = len(ticks)
        await asyncio.sleep(0.13)
        assert len(ticks) == stopped_ticks
        detector.start()
        await asyncio.sleep(0.13)
        assert len(ticks) > stopped_ticks, "restart lost the heartbeat"
    finally:
        detector.stop()
    final_ticks = len(ticks)
    await asyncio.sleep(0.13)
    assert len(ticks) == final_ticks


def test_late_callback_from_a_stopped_run_cannot_rearm_after_restart(monkeypatch):
    """Exercise delivery already in flight when timer cancellation arrives."""
    class Timer:
        def __init__(self, callback, args):
            self.callback, self.args = callback, args
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def deliver(self):
            self.callback(*self.args)

    class Loop:
        def __init__(self):
            self.timers = []

        def call_later(self, delay, callback, *args):
            timer = Timer(callback, args)
            self.timers.append(timer)
            return timer

    loop = Loop()
    detector = StallDetector(LatencyTracker())
    ticks = []
    monkeypatch.setattr(detector, "beat", lambda: ticks.append(1))
    detector.start(loop)
    old_timer = loop.timers[-1]
    detector.stop()
    assert old_timer.cancelled
    old_timer.deliver()
    assert not ticks and len(loop.timers) == 1
    detector.start(loop)
    try:
        current_timer = loop.timers[-1]
        old_timer.deliver()
        assert not ticks and len(loop.timers) == 2
        current_timer.deliver()
        assert ticks == [1] and len(loop.timers) == 3
    finally:
        detector.stop()
    assert loop.timers[-1].cancelled


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


# ── Mark XXXV W1.1: every block over 100 ms, reported with its final length ──

def a_named_blocking_call(seconds: float) -> None:
    time.sleep(seconds)


async def test_a_stall_is_reported_once_when_it_ends_with_its_final_length():
    """The PERF: line used to be written at DETECTION, so it carried the
    threshold-ish number (the 8-10 Oct logs: lots of '2xx ms'). Now it comes
    once, when the loop gets its turn back, with the whole length."""
    seen: list[tuple[float, float]] = []
    detector = StallDetector(LatencyTracker(), threshold_ms=LOG_THRESHOLD_MS,
                             on_stall=lambda s: seen.append((s.ms, time.monotonic())))
    detector.start()
    try:
        await asyncio.sleep(0.15)
        assert not seen, "a healthy loop reported a stall"
        a_named_blocking_call(0.40)
        ended = time.monotonic()
        await asyncio.sleep(0.2)
        assert len(seen) == 1, seen
        ms, reported_at = seen[0]
        assert reported_at >= ended - 0.005, "reported before the block ended"
        assert 330.0 <= ms <= 480.0, ms
        assert "a_named_blocking_call" in detector.stalls[-1].culprit()
    finally:
        detector.stop()


async def test_a_short_block_over_100_ms_is_caught_at_the_log_threshold():
    seen: list[Stall] = []
    detector = StallDetector(LatencyTracker(), threshold_ms=LOG_THRESHOLD_MS,
                             on_stall=seen.append)
    detector.start()
    try:
        await asyncio.sleep(0.15)
        a_named_blocking_call(0.17)
        await asyncio.sleep(0.15)
        assert len(seen) == 1 and 110.0 <= seen[0].ms <= 230.0, [s.ms for s in seen]
    finally:
        detector.stop()


async def test_a_long_freeze_is_reported_while_it_lasts_from_the_watchdog():
    """A freeze that never ends must still leave its culprit on disk: the
    progress report comes from the watchdog thread, mid-block."""
    import threading
    progress: list[tuple[Stall, str]] = []
    finished: list[Stall] = []
    detector = StallDetector(
        LatencyTracker(), threshold_ms=LOG_THRESHOLD_MS, on_stall=finished.append,
        on_stall_progress=lambda s: progress.append((s, threading.current_thread().name)))
    detector.start()
    try:
        await asyncio.sleep(0.12)
        a_named_blocking_call(1.25)
        await asyncio.sleep(0.15)
        assert len(progress) == 1, [p[0].ms for p in progress]
        stall, thread_name = progress[0]
        assert thread_name == "orion-stall-watchdog"
        assert 1000.0 <= stall.ms < 1250.0
        assert "a_named_blocking_call" in stall.culprit()
        assert len(finished) == 1 and finished[0].ms >= 1150.0
    finally:
        detector.stop()


def test_a_frame_from_a_test_file_called_latency_is_not_plumbing():
    stall = Stall(ms=300.0, at=0.0, frames=[
        '  File "C:/orion/orion_core/latency.py", line 9, in _watch\n',
        '  File "C:/orion/tests/test_latency.py", line 5, in slow_probe\n',
    ])
    assert stall.culprit().startswith("slow_probe()")


def test_a_stack_of_only_loop_machinery_is_named_as_native_code():
    stall = Stall(ms=300.0, at=0.0, frames=[
        '  File "C:/py/Lib/site-packages/qasync/__init__.py", line 3, in run_forever\n'])
    assert stall.culprit().startswith("Qt/native code (no Python frame)")
    # app.py's loop runner is plumbing too (owner's 12 Oct log named it).
    runner = Stall(ms=300.0, at=0.0, frames=[
        '  File "C:/o/orion_core/app.py", line 2705, in _run_until_torn_down\n',
        '  File "C:/py/Lib/site-packages/qasync/__init__.py", line 3, in run_forever\n'])
    assert runner.culprit().startswith("Qt/native code (no Python frame)")


def test_the_reporter_sends_long_stalls_to_the_window_and_short_ones_to_the_file():
    window: list[str] = []
    file: list[str] = []
    reporter = StallReporter(window.append, file.append)
    frames = ['  File "C:/x/orion_core/audio.py", line 42, in open_stream\n']
    reporter.finished(Stall(ms=412.0, at=0.0, frames=frames))
    reporter.finished(Stall(ms=130.0, at=0.0, frames=frames))
    assert window == ["PERF: event loop blocked 412 ms by open_stream() at audio.py:42"]
    assert file == ["PERF: event loop blocked 130 ms by open_stream() at audio.py:42"]
    reporter.ongoing(Stall(ms=3000.0, at=0.0, frames=frames))
    assert file[-1] == "PERF: event loop STILL blocked after 3.0 s by open_stream() at audio.py:42"


def test_the_reporter_rate_limits_short_blocks_and_counts_the_rest():
    now = [0.0]
    file: list[str] = []
    reporter = StallReporter(None, file.append, per_minute=3, clock=lambda: now[0])
    for _ in range(7):
        reporter.finished(Stall(ms=120.0, at=0.0, frames=[]))
    assert len(file) == 3
    now[0] = 61.0
    reporter.finished(Stall(ms=120.0, at=0.0, frames=[]))
    assert file[-1].endswith("(+4 shorter block(s) not listed)")
    # a long stall is never rate-limited away
    window: list[str] = []
    reporter.window_log = window.append
    for _ in range(10):
        reporter.finished(Stall(ms=900.0, at=0.0, frames=[]))
    assert len(window) == 10


def test_a_broken_log_destination_does_not_reach_the_watchdog():
    def boom(_line):
        raise RuntimeError("wrapped C/C++ object of type OrionBus has been deleted")
    reporter = StallReporter(boom, boom)
    reporter.finished(Stall(ms=900.0, at=0.0, frames=[]))
    reporter.finished(Stall(ms=120.0, at=0.0, frames=[]))
    reporter.ongoing(Stall(ms=5000.0, at=0.0, frames=[]))


# ── "Cannot enter into task": every one logged, with the call that pumped ────

_REENTRY_MESSAGE = (
    "Cannot enter into task <Task pending name='orion-live-receive' "
    "coro=<LiveWorker._receive_loop() running at C:\\o\\orion_core\\live_worker.py:900>> "
    "while another task <Task pending name='tool:camera' coro=<Dispatcher.dispatch() "
    "running at C:\\o\\orion_core\\dispatcher.py:400>> is being executed.")


def test_the_reentry_line_names_both_tasks_and_the_pumping_call():
    stack = [
        '  File "C:/py/Lib/site-packages/qasync/__init__.py", line 3, in run_forever\n',
        '  File "C:/py/Lib/asyncio/events.py", line 89, in _run\n',
        '  File "C:/o/orion_core/dispatcher.py", line 400, in dispatch\n',
        '  File "C:/o/orion_core/camera_devices.py", line 72, in open_camera\n',
        '  File "C:/py/Lib/site-packages/qasync/__init__.py", line 9, in timerEvent\n',
        '  File "C:/py/Lib/asyncio/events.py", line 89, in _run\n',
        '  File "C:/py/Lib/asyncio/base_events.py", line 5, in call_exception_handler\n',
    ]
    assert reentry_culprit(stack) == "open_camera() at camera_devices.py:72"
    line = reentry_line(_REENTRY_MESSAGE, stack)
    assert line.startswith("LOOP: Cannot enter into task 'orion-live-receive' "
                           "(LiveWorker._receive_loop)")
    assert "'tool:camera' (Dispatcher.dispatch) pumped the Windows message queue" in line
    assert "open_camera() at camera_devices.py:72" in line


def test_a_real_nested_pump_is_logged_with_its_culprit_and_handed_on(monkeypatch):
    """The 8 Oct mechanism, for real: a task pumps Qt messages inside its own
    step, qasync tries to step a ticking task nested inside it. The watch must
    log it naming the pumping function, then pass the fault on unchanged."""
    import qasync
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication

    global _LATENCY_QT
    _LATENCY_QT = QApplication.instance() or QApplication([sys.argv[0]])
    loop = qasync.QEventLoop(_LATENCY_QT)
    handed_on: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: handed_on.append(context))
    logged: list[str] = []
    REENTRIES.clear()
    assert install_reentry_watch(loop, logged.append)

    def a_modal_dialog_inside_a_task():
        nested = QEventLoop()
        QTimer.singleShot(150, nested.quit)
        nested.exec()

    ticks = [0]
    after_pump: list[int] = []

    async def ticker():
        while True:
            await asyncio.sleep(0.02)
            ticks[0] += 1

    async def pumper():
        await asyncio.sleep(0.05)
        a_modal_dialog_inside_a_task()

    async def flow():
        tick = asyncio.ensure_future(ticker())
        tick.set_name("probe-ticker")
        pump = asyncio.ensure_future(pumper())
        pump.set_name("probe-pumper")
        await pump
        after_pump.append(ticks[0])
        await asyncio.sleep(0.3)
        after_pump.append(ticks[0])
        tick.cancel()
        await asyncio.wait({tick}, timeout=1.0)     # a dead task would never finish

    with loop:
        loop.run_until_complete(flow())
    assert logged, "the nested pump was not logged"
    assert "Cannot enter into task 'probe-ticker' (ticker)" in logged[0], logged[0]
    assert "is stepped again once that step ends" in logged[0], logged[0]
    # 28 Oct 2026: the refused task carries on - before, it never ran again
    # (orion-send-realtime starved by a forget dialog, Live deaf to the owner).
    assert after_pump[1] > after_pump[0], "the starved ticker never resumed"
    assert "'probe-pumper' (pumper)" in logged[0]
    assert "a_modal_dialog_inside_a_task()" in logged[0], logged[0]
    assert handed_on and "cannot enter into task" in str(
        handed_on[0].get("exception")).lower(), "the previous handler must still see it"
    assert REENTRIES and "re-entry" in report()


def test_the_reentry_watch_chains_on_a_plain_asyncio_loop_too():
    loop = asyncio.new_event_loop()
    try:
        handed_on: list[dict] = []
        loop.set_exception_handler(lambda _loop, context: handed_on.append(context))
        logged: list[str] = []
        assert install_reentry_watch(loop, logged.append)
        loop.call_exception_handler({"message": "x", "exception": RuntimeError(_REENTRY_MESSAGE)})
        loop.call_exception_handler({"message": "y", "exception": ValueError("other")})
        assert len(logged) == 1 and len(handed_on) == 2
    finally:
        loop.close()


def test_the_app_wires_the_100_ms_log_and_the_reentry_watch():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py"
           ).read_text(encoding="utf-8")
    assert "ORION_STALL_LOG_MS" in src and "StallReporter(bus.log.emit, _session_log.write)" in src
    assert "on_stall_progress = _stall_reporter.ongoing" in src
    repair = src.index("selfrepair.install(loop)")
    watch = src.index("_install_reentry_watch(loop, bus.log.emit)")
    assert repair < watch, "the watch must be installed after selfrepair, or it is replaced"


# ── who ran through a stall (13 Oct 2026) ─────────────────────────────────────
#
# The owner's 12 Oct log said "event loop blocked 1184 ms by Qt/native code (no
# Python frame)": true of the loop's stack, but the loop was only WAITING -
# the meaning model's ONNX session build held Python's lock on another thread.

_RUNNER = ['  File "C:/x/orion_core/app.py", line 2705, in _run_until_torn_down']


def test_a_stall_names_the_thread_that_ran_through_it():
    from orion_core.latency import perf_line
    stall = Stall(ms=1184.0, at=0.0, frames=_RUNNER,
                  others=[("orion-semantic-warm", 1120.0, "load() at semantic.py:189")],
                  loop_cpu_ms=16.0)
    assert perf_line(stall) == (
        "PERF: event loop blocked 1184 ms by Qt/native code (no Python frame) under "
        "_run_until_torn_down() at app.py:2705 - meanwhile thread 'orion-semantic-warm' "
        "ran 1120 ms (now at load() at semantic.py:189); the loop's thread only waited "
        "(16 ms of work)")
    ticking = Stall(ms=300.0, at=0.0, frames=_RUNNER, others=[("tick", 20.0, "")],
                    loop_cpu_ms=290.0)
    assert " - meanwhile" not in perf_line(ticking)          # ticked over: not blamed
    both = Stall(ms=300.0, at=0.0, frames=_RUNNER, others=[("worker", 250.0, "")],
                 loop_cpu_ms=280.0)
    assert perf_line(both).endswith(" - meanwhile thread 'worker' ran 250 ms")  # loop worked
    held = Stall(ms=400.0, at=0.0, held=True, others=[("gil-hog", 390.0, "")])
    assert held.culprit().startswith("another thread holding Python's lock")
    assert Stall(ms=400.0, at=0.0, held=True).culprit().startswith("every Python thread paused")


def test_the_program_entry_above_the_loop_runner_is_never_blamed():
    """17 Oct: native blocks were blamed on 'main() at app.py:3000' - the
    caller of the whole loop - once the runner itself counted as plumbing."""
    frames = ['  File "C:/x/orion.py", line 88, in <module>',
              '  File "C:/x/orion_core/app.py", line 3003, in main\n'
              '    restart_requested = bool(_run_until_torn_down(loop,',
              '  File "C:/x/orion_core/app.py", line 2840, in _run_until_torn_down',
              '  File "C:/x/qasync/__init__.py", line 400, in run_forever']
    assert Stall(ms=247.0, at=0.0, frames=frames).culprit() == (
        "Qt/native code (no Python frame) under _run_until_torn_down() at app.py:2840")
    deeper = frames + ['  File "C:/x/orion_core/audio_devices.py", line 67, in _devices']
    assert Stall(ms=325.0, at=0.0, frames=deeper).culprit() == "_devices() at audio_devices.py:67"


def test_the_thread_clock_counts_a_busy_threads_cpu():
    import threading
    from orion_core.latency import _ThreadClock
    clock = _ThreadClock()
    try:
        before = clock.snapshot()
        if not before:
            pytest.skip("no per-thread CPU clock on this platform")
        me = next(key for key in before if key[0] == threading.get_ident())
        end = time.perf_counter() + 0.2
        while time.perf_counter() < end:
            pass
        assert clock.snapshot()[me] - before[me] >= 100.0       # 200 ms spun (15.6 ms ticks)
    finally:
        clock.close()


def test_an_old_cpu_note_blames_nobody():
    """A worker busy for ten seconds BEFORE a stall did not cause it."""
    import threading
    from orion_core.latency import _ThreadClock
    detector = StallDetector(threshold_ms=100.0)
    clock = _ThreadClock()
    try:
        fresh = (time.monotonic(), clock.snapshot())
        if not fresh[1]:
            pytest.skip("no per-thread CPU clock on this platform")
        spun = threading.Event()

        def spin() -> None:
            end = time.perf_counter() + 0.2
            while time.perf_counter() < end:
                pass
            spun.set()
            time.sleep(0.5)                      # still alive when attributed
        threading.Thread(target=spin, name="spinner", daemon=True).start()
        assert spun.wait(5)
        old = Stall(ms=150.0, at=0.0)
        detector._attribute(old, clock, (time.monotonic() - 10.0, fresh[1]))
        assert old.others == [] and old.loop_cpu_ms is None
        new = Stall(ms=150.0, at=0.0)
        detector._attribute(new, clock, fresh)
        assert new.others and new.others[0][0] == "spinner"
    finally:
        clock.close()


def test_the_loop_notes_a_late_beat_itself():
    """The loop's own measure: the one record of a stall the watchdog froze through."""
    detector = StallDetector(threshold_ms=100.0)
    loop = asyncio.new_event_loop()
    try:
        generation = detector._beat_generation = object()
        detector._beat = time.monotonic() - 0.5
        detector._heartbeat(loop, generation)
        assert detector._missed is not None and detector._missed[1] >= 400.0
    finally:
        detector.stop()
        loop.close()


def test_a_thread_holding_pythons_lock_is_named():
    """A worker in one long native call that keeps Python's lock (a regular
    expression that backtracks, here) freezes the loop and the watchdog alike.
    The stall must be reported and name that thread - whichever of the two
    threads gets Python back first."""
    import re
    import threading
    from orion_core.latency import perf_line
    pattern = re.compile(r"(x+x+)+y")
    size = 16
    while size < 32:                       # ~0.25-0.5 s in one call, on this machine
        started = time.perf_counter()
        pattern.match("x" * size)
        if time.perf_counter() - started >= 0.25:
            break
        size += 1
    lines: list[str] = []
    detector = StallDetector(threshold_ms=100.0,
                             on_stall=lambda stall: lines.append(perf_line(stall)))
    finished = threading.Event()

    def hog() -> None:
        time.sleep(0.6)                    # the watchdog notes healthy CPU first
        pattern.match("x" * size)
        finished.set()
        time.sleep(0.6)                    # still alive, so still known by name

    async def main() -> None:
        detector.start()
        threading.Thread(target=hog, name="gil-hog", daemon=True).start()
        while not finished.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)
        detector.stop()

    asyncio.run(main())
    assert any("thread 'gil-hog' ran" in line for line in lines), lines


# ── the meaning model starts at a quiet moment (13 Oct 2026) ─────────────────
#
# Building its ONNX session holds Python's lock in one native call: 105-114 ms
# from source; 1184 ms in the installed app on 12 Oct, in the same second the
# microphone reported "input overflow" while the owner was answering ORION.

class _FakeTime:
    def __init__(self) -> None:
        self.now = 100.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_the_conversation_is_quiet_only_after_a_still_moment():
    from orion_core import semantic
    fake = _FakeTime()
    quiet = semantic.ConversationQuiet(clock=fake.clock)
    assert not quiet.quiet()                              # just launched
    fake.sleep(semantic.QUIET_S)
    assert quiet.quiet()
    for busy in ("PROCESSING", "SPEAKING", "CONNECTING"):
        quiet.note_state(busy)
        fake.sleep(600)
        assert not quiet.quiet(), busy                    # mid-turn, however long
    quiet.note_state("LISTENING")
    fake.sleep(semantic.QUIET_S - 1)
    assert not quiet.quiet()                              # the user may be answering
    fake.sleep(1)
    assert quiet.quiet()
    for silent in ("STANDBY", "PAUSED"):
        quiet.note_state(silent)
        assert quiet.quiet(), silent                      # nothing heard or said


def test_the_meaning_model_waits_for_a_quiet_moment(monkeypatch):
    from orion_core import semantic
    started: list[str] = []
    monkeypatch.setenv("ORION_SEMANTIC", "1")
    monkeypatch.setattr(semantic, "_warm_now",
                        lambda on_ready, log, allow_download, when="": started.append(when))

    def warm(quiet) -> str:
        fake = _FakeTime()
        semantic.warm_when_quiet(lambda: quiet(fake), clock=fake.clock,
                                 sleep=fake.sleep).join(5)
        return started.pop()

    def broken(_time) -> bool:
        raise RuntimeError("bus gone")
    assert warm(lambda t: t.now >= 140.0) == "started at a quiet moment, 40 s after launch"
    assert warm(lambda t: True) == "started at a quiet moment, 10 s after launch"  # never sooner
    assert warm(lambda t: False) == (f"started at {semantic.WARM_LATEST_S:.0f} s "
                                     "with no quiet moment")      # a busy session still gets it
    assert warm(broken) == "started at a quiet moment, 10 s after launch"   # recall not lost
    monkeypatch.setenv("ORION_SEMANTIC", "0")
    assert semantic.warm_when_quiet(lambda: True) is None


def test_orion_starts_the_meaning_model_at_a_quiet_moment():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py"
              ).read_text(encoding="utf-8")
    block = source[source.index("_semantic.ConversationQuiet()"):][:600]
    assert "bus.state.connect(_sem_conversation.note_state)" in block
    assert "_semantic.warm_when_quiet(" in block and "foreground_active()" in block
    assert "Timer(\n        25.0" not in source                # the blind timer is gone
    # The intent trainer shares that quiet test (it trained through speech on 16 Oct).
    trainer = source[source.index("_intent_brain.set_quiet_probe("):][:200]
    assert "_sem_conversation.quiet()" in trainer and "foreground_active()" in trainer
