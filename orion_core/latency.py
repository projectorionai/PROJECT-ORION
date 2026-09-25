"""
Latency observability — ORION measuring his own responsiveness.

Every lag bug in this project has been the same shape: something ran on the
qasync loop that should not have. ``psutil.cpu_percent(interval=0.2)`` froze it
for 200 ms. A full-fsync SQLite commit cost 2.81 ms and there were hundreds. The
voxel loop spent 441 ms per second in QPainter. Each was found by a human
profiling by hand, after the user noticed a stutter.

That is the wrong way round. This module makes the stall detect ITSELF, and —
crucially — names the code that caused it *while it is still running*.

Two pieces:

**StallDetector** — a watchdog on a plain thread. The event loop touches a
heartbeat every ``BEAT_S``; the watchdog checks it and, when the heartbeat goes
stale beyond ``threshold_ms``, grabs the loop thread's stack with
``sys._current_frames()``. That timing is the whole point: the blocking call has
not returned yet, so the stack IS the culprit. A monitor that sampled after the
fact would only ever show you the code that ran next.

The watchdog deliberately lives on a thread rather than a task, because a task
cannot observe a loop that is blocked — it would be queued behind the very thing
it is trying to measure.

**LatencyTracker** — a bounded ring of named spans with percentile summaries.
p95 rather than mean: a mean hides exactly the occasional 300 ms turn that a
user actually feels, and lag is a tail-latency problem, not an average one.

Both are cheap enough to leave on: a heartbeat is a float store, a span is an
append to a deque. Neither allocates unboundedly — the ring discards oldest.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

#: How often the event loop marks itself alive. Small enough to catch a stall
#: quickly, large enough that the timer itself is not the load.
BEAT_S = 0.05

#: How often the watchdog thread looks. Finer than BEAT_S so detection latency
#: is dominated by the threshold, not the sampling.
WATCH_S = 0.02

#: Below this, a "stall" is just normal scheduling jitter on a desktop OS
#: (Qt painting a complex frame, a GC pause). Above it, something is wrong.
DEFAULT_THRESHOLD_MS = 250.0

#: Frames from these modules are plumbing, never the answer to "what blocked?".
_NOISE = ("asyncio/", "asyncio\\", "qasync", "selectors.py", "threading.py",
          "latency.py")


# ── spans ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Span:
    """One measured operation."""
    name: str
    ms: float
    at: float
    detail: str = ""

    def line(self) -> str:
        tail = f"  {self.detail}" if self.detail else ""
        return f"{self.name:<28} {self.ms:8.2f} ms{tail}"


@dataclass
class Summary:
    """Percentile view of one span name. p95, because lag is a tail problem."""
    name: str
    count: int
    p50: float
    p95: float
    worst: float
    total: float

    def line(self) -> str:
        return (f"  {self.name:<26} n={self.count:<5} p50 {self.p50:7.2f}  "
                f"p95 {self.p95:7.2f}  max {self.worst:7.2f} ms")


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No numpy: this runs in the monitoring path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


class LatencyTracker:
    """A bounded ring of spans, summarisable by name.

    Thread-safe: spans are recorded from the loop, from worker threads and from
    the watchdog, and a torn read of the deque would be a monitoring tool
    creating its own bug.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self._spans: deque[Span] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, name: str, ms: float, detail: str = "") -> Span:
        span = Span(name=name, ms=float(ms), at=time.time(), detail=detail)
        with self._lock:
            self._spans.append(span)
        return span

    @contextmanager
    def span(self, name: str, detail: str = "") -> Iterator[None]:
        """Time a block. Records even if the block raises — a slow failure is
        still a slow turn, and hiding it would flatter the numbers."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0, detail)

    def spans(self, name: str | None = None) -> list[Span]:
        with self._lock:
            found = list(self._spans)
        return [s for s in found if name is None or s.name == name]

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()

    def summary(self, name: str | None = None) -> list[Summary]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for span in self.spans(name):
            grouped[span.name].append(span.ms)
        out = [
            Summary(name=key, count=len(v), p50=percentile(v, 0.50),
                    p95=percentile(v, 0.95), worst=max(v), total=sum(v))
            for key, v in grouped.items()
        ]
        return sorted(out, key=lambda s: -s.p95)

    def slowest(self, limit: int = 10) -> list[Span]:
        return sorted(self.spans(), key=lambda s: -s.ms)[:limit]

    def report(self, limit: int = 12) -> str:
        rows = self.summary()[:limit]
        if not rows:
            return "No timings recorded yet."
        head = f"Latency — {len(self.spans())} spans, slowest by p95:"
        return head + "\n" + "\n".join(r.line() for r in rows)


# ── stall detection ───────────────────────────────────────────────────────────

def _short_frame(frame: str) -> str:
    """'File "C:/.../audio.py", line 42, in read' -> 'read() at audio.py:42'."""
    import re
    match = re.search(r'File "([^"]+)", line (\d+), in (\S+)', frame)
    if not match:
        return frame.strip()[:90]
    path, line, func = match.groups()
    return f"{func}() at {path.replace(chr(92), '/').rsplit('/', 1)[-1]}:{line}"


@dataclass
class Stall:
    """A period where the event loop did not get a turn."""
    ms: float
    at: float
    frames: list[str] = field(default_factory=list)

    def culprit(self) -> str:
        """The most likely offending frame — the deepest non-plumbing one.

        Rendered as ``function() at file.py:line`` rather than a raw traceback
        line: an absolute Windows path wraps three times in a log pane and
        buries the one thing the reader needs, which is the function name.
        """
        for frame in reversed(self.frames):
            if not any(noise in frame for noise in _NOISE):
                return _short_frame(frame)
        return _short_frame(self.frames[-1]) if self.frames else "unknown"

    def line(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.at))
        return f"  {stamp}  {self.ms:7.0f} ms blocked   {self.culprit()}"


class StallDetector:
    """Watches the event loop from a thread and records what blocked it.

    Usage::

        detector = StallDetector(tracker)
        detector.start(loop)        # from inside the running loop
        ...
        detector.stop()
    """

    def __init__(self, tracker: LatencyTracker | None = None, *,
                 threshold_ms: float = DEFAULT_THRESHOLD_MS,
                 on_stall: Callable[[Stall], None] | None = None,
                 capacity: int = 200) -> None:
        self.tracker = tracker
        self.threshold_ms = float(threshold_ms)
        self.on_stall = on_stall
        self.stalls: deque[Stall] = deque(maxlen=capacity)
        self._beat = time.monotonic()
        self._loop_thread_id: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._beat_task: Any = None
        self._lock = threading.Lock()

    # -- the loop side ---------------------------------------------------------

    def beat(self) -> None:
        """Mark the loop alive. A float store; safe to call very often."""
        self._beat = time.monotonic()

    async def _heartbeat(self) -> None:
        self._loop_thread_id = threading.get_ident()
        while not self._stop.is_set():
            self.beat()
            await asyncio.sleep(BEAT_S)

    # -- the watchdog side -----------------------------------------------------

    def _capture(self) -> list[str]:
        """The loop thread's stack, right now, while it is still stuck."""
        thread_id = self._loop_thread_id
        if thread_id is None:
            return []
        frame = sys._current_frames().get(thread_id)
        if frame is None:
            return []
        try:
            return [line.rstrip() for line in
                    traceback.format_stack(frame)][-14:]
        except Exception:
            return []

    def _watch(self) -> None:
        """Detect a stall once, then keep extending it until the loop recovers.

        Reporting the duration at the moment the threshold is crossed would say
        "160 ms" for a 450 ms freeze — technically a detection, but a misleading
        number, and the number is the whole point. So the stall is recorded when
        first seen (the stack must be captured WHILE it is stuck) and its
        duration is revised upward on each tick until the heartbeat moves again.
        """
        reported_for: float = 0.0
        active: Stall | None = None
        while not self._stop.is_set():
            time.sleep(WATCH_S)
            beat = self._beat
            late_ms = (time.monotonic() - beat) * 1000.0

            if late_ms < self.threshold_ms:
                active = None                 # loop is healthy again
                continue
            if beat == reported_for:
                if active is not None:        # same stall, still going
                    with self._lock:
                        active.ms = late_ms
                continue
            reported_for = beat
            active = self.record_stall(late_ms, self._capture())

    def record_stall(self, ms: float, frames: list[str] | None = None) -> Stall:
        """Record a stall. Public so it can be driven deterministically in tests
        rather than by trying to make a real thread hang on a timer."""
        stall = Stall(ms=float(ms), at=time.time(), frames=list(frames or []))
        with self._lock:
            self.stalls.append(stall)
        if self.tracker is not None:
            self.tracker.record("event-loop stall", stall.ms, stall.culprit())
        if self.on_stall is not None:
            try:
                self.on_stall(stall)
            except Exception:
                pass              # a broken reporter must not kill the watchdog
        return stall

    # -- lifecycle -------------------------------------------------------------

    def start(self, loop: Any = None) -> None:
        """Begin watching. Call from inside the loop being watched."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._beat = time.monotonic()
        # Identify the watched thread HERE, not inside the heartbeat coroutine.
        # start() is documented as being called from the loop being watched, and
        # a task may not get its first turn before something blocks — in which
        # case _capture() had no thread id and every stall reported "unknown",
        # losing the one piece of information the watchdog exists to provide.
        self._loop_thread_id = threading.get_ident()
        loop = loop or _running_loop()
        if loop is not None:
            self._beat_task = loop.create_task(self._heartbeat())
        self._thread = threading.Thread(target=self._watch,
                                        name="orion-stall-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        task = self._beat_task
        if task is not None and not task.done():
            task.cancel()
        self._beat_task = None
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def report(self, limit: int = 10) -> str:
        with self._lock:
            found = list(self.stalls)
        if not found:
            return (f"No event-loop stalls over {self.threshold_ms:.0f} ms — "
                    "ORION stayed responsive.")
        worst = sorted(found, key=lambda s: -s.ms)[:limit]
        head = (f"{len(found)} event-loop stall(s) over {self.threshold_ms:.0f} ms "
                "— something blocked the thread that draws the face:")
        return head + "\n" + "\n".join(s.line() for s in worst)


def _running_loop() -> Any:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


# ── process-wide handles ──────────────────────────────────────────────────────

TRACKER = LatencyTracker()
DETECTOR = StallDetector(TRACKER)


def span(name: str, detail: str = ""):
    """Time a block against the process-wide tracker."""
    return TRACKER.span(name, detail)


def record(name: str, ms: float, detail: str = "") -> Span:
    return TRACKER.record(name, ms, detail)


def report() -> str:
    """Everything ORION knows about his own responsiveness."""
    return DETECTOR.report() + "\n\n" + TRACKER.report()


__all__ = [
    "BEAT_S", "DEFAULT_THRESHOLD_MS", "DETECTOR", "TRACKER", "LatencyTracker",
    "Span", "Stall", "StallDetector", "Summary", "percentile", "record",
    "report", "span",
]
