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
import re
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

#: In a real run every block of the loop longer than this is written to
#: session.log as a PERF: line naming the ORION frame (Mark XXXV W1.1). The
#: window log still only shows blocks of DEFAULT_THRESHOLD_MS and over.
LOG_THRESHOLD_MS = 100.0

#: A freeze still going at these points is written to the file straight from
#: the watchdog thread: the thread that would deliver a log signal is the one
#: that is stuck, and a freeze that never ends must still leave its culprit.
PROGRESS_MS = (1000.0, 3000.0, 10000.0, 30000.0)
PROGRESS_EVERY_MS = 60000.0

#: While the loop is healthy the watchdog notes every Python thread's CPU time
#: this often, so a stall can be laid at the door of the thread that ran
#: through it (13 Oct 2026). A worker holding Python's lock in one long native
#: call freezes the loop AND this watchdog, and the loop's stack then shows
#: only the loop runner: "Qt/native code (no Python frame)" really meant
#: "waiting for Python". (The meaning model's ONNX session build held it
#: 105-114 ms from source and 1184 ms in the installed app on 12 Oct.)
CPU_SAMPLE_S = 0.25
#: A note older than the stall by more than this says nothing about the stall
#: (a worker busy for ten seconds before it is not what blocked the loop).
CPU_NOTE_SLACK_S = 1.0

#: Frames from these modules are plumbing, never the answer to "what blocked?".
#: (This module by its exact name: "latency.py" alone also matched tests and
#: any other file that merely ends that way.)
#: app.py's loop runner. Everything outside it on the loop thread's stack is
#: the program starting the loop, never a culprit.
_RUNNER = "in _run_until_torn_down"

_NOISE = ("asyncio/", "asyncio\\", "qasync", "selectors.py", "threading.py",
          '/latency.py"', '\\latency.py"',
          # app.py's loop runner: a block "in" it is Qt or native work the
          # loop itself ran (painting, layout, a Windows call). The owner's
          # 12 Oct log blamed "_run_until_torn_down() at app.py:2705".
          "in _run_until_torn_down")


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

def _is_orion_frame(frame: str) -> bool:
    """A frame in ORION's own code (frozen or from source)."""
    return "orion_core" in frame or "orion.py" in frame


def _short_frame(frame: str) -> str:
    """'File "C:/.../audio.py", line 42, in read' -> 'read() at audio.py:42'."""
    match = re.search(r'File "([^"]+)", line (\d+), in (\S+)', frame)
    if not match:
        return frame.strip()[:90]
    path, line, func = match.groups()
    return f"{func}() at {path.replace(chr(92), '/').rsplit('/', 1)[-1]}:{line}"


def _blame(frames: list[str]) -> str:
    """The deepest non-plumbing frame of a stack, short; '' if none."""
    deepest = next((f for f in reversed(frames)
                    if not any(noise in f for noise in _NOISE)), None)
    return _short_frame(deepest) if deepest is not None else ""


def _stack_lines(frame: Any, limit: int = 14) -> list[str]:
    """A thread's stack as traceback-style 'File ..., line N, in f' lines,
    without reading source files (cheap enough for every thread at once)."""
    lines: list[str] = []
    while frame is not None and len(lines) < 60:
        code = frame.f_code
        lines.append(f'  File "{code.co_filename}", line {frame.f_lineno}, in {code.co_name}')
        frame = frame.f_back
    lines.reverse()
    return lines[-limit:]


class _ThreadClock:
    """CPU time so far (ms) of each live Python thread, keyed (ident, native id).

    Windows: GetThreadTimes on handles kept open between calls; elsewhere
    each thread's CPU clock where Python offers one, else nothing at all.
    """

    def __init__(self) -> None:
        self._handles: dict[tuple[int, int], Any] = {}
        self._kernel32: Any = None

    def snapshot(self) -> dict[tuple[int, int], float]:
        threads = [(t.ident, t.native_id) for t in threading.enumerate()
                   if t.ident is not None and t.native_id is not None]
        try:
            if sys.platform == "win32":
                return self._windows(threads)
            cpu_clock = getattr(time, "pthread_getcpuclockid", None)
            if cpu_clock is None:
                return {}
            out: dict[tuple[int, int], float] = {}
            for key in threads:
                try:
                    out[key] = time.clock_gettime(cpu_clock(key[0])) * 1000.0
                except (OSError, OverflowError, ValueError):
                    pass
            return out
        except Exception:
            return {}                 # a diagnostic must never cost the watchdog

    def _windows(self, threads: list[tuple[int, int]]) -> dict[tuple[int, int], float]:
        import ctypes
        from ctypes import wintypes
        k32 = self._kernel32
        if k32 is None:
            # Its own kernel32 instance: argtypes set here touch no other module.
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenThread.restype = wintypes.HANDLE
            k32.GetThreadTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
            k32.GetThreadTimes.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._kernel32 = k32
        live = set(threads)
        for key in [k for k in self._handles if k not in live]:
            k32.CloseHandle(self._handles.pop(key))
        times = [wintypes.FILETIME() for _ in range(4)]
        refs = [ctypes.byref(item) for item in times]
        out: dict[tuple[int, int], float] = {}
        for key in threads:
            handle = self._handles.get(key)
            if handle is None:
                handle = k32.OpenThread(0x0800, False, key[1])  # THREAD_QUERY_LIMITED_INFORMATION
                if not handle:
                    continue
                self._handles[key] = handle
            if k32.GetThreadTimes(handle, *refs):
                kernel, user = times[2], times[3]
                out[key] = (((kernel.dwHighDateTime << 32) | kernel.dwLowDateTime)
                            + ((user.dwHighDateTime << 32) | user.dwLowDateTime)) / 10_000.0
        return out

    def close(self) -> None:
        k32 = self._kernel32
        for handle in self._handles.values():
            try:
                k32.CloseHandle(handle)
            except Exception:
                pass
        self._handles.clear()


@dataclass
class Stall:
    """A period where the event loop did not get a turn."""
    ms: float
    at: float
    frames: list[str] = field(default_factory=list)
    #: Python's lock was held elsewhere: every Python thread waited, the
    #: watchdog too, so no stack of the loop could be taken while it lasted.
    held: bool = False
    #: Who ran while the loop could not: (thread, CPU ms, where it is now),
    #: busiest first; and the loop thread's own CPU over the same stretch.
    others: list[tuple[str, float, str]] = field(default_factory=list)
    loop_cpu_ms: float | None = None

    def meanwhile(self) -> str:
        """The thread that ran through the stall, when one clearly did, and
        whether the loop's own thread was working or only waiting."""
        if not self.others:
            return ""
        name, cpu, where = self.others[0]
        if cpu < max(40.0, 0.4 * self.ms):
            return ""
        text = f"thread '{name}' ran {cpu:.0f} ms" + (f" (now at {where})" if where else "")
        if self.loop_cpu_ms is not None and self.loop_cpu_ms < 0.25 * self.ms:
            text += f"; the loop's thread only waited ({self.loop_cpu_ms:.0f} ms of work)"
        return text

    def culprit(self) -> str:
        """The most likely offending frame — the deepest non-plumbing one.

        Rendered as ``function() at file.py:line`` rather than a raw traceback
        line: an absolute Windows path wraps three times in a log pane and
        buries the one thing the reader needs, which is the function name.
        """
        if self.held and not self.frames:
            if self.meanwhile():
                return "another thread holding Python's lock (every Python thread waited)"
            return ("every Python thread paused (Python's lock held in native code, "
                    "or the whole process stalled)")
        # Frames OUTSIDE the loop runner (main(), the program's entry) only
        # started the loop; just what the loop ran can have blocked it. The
        # owner's 17 Oct log blamed "main() at app.py:3000" - the caller of
        # the whole loop - once the runner itself was counted as plumbing.
        runner = max((i for i, f in enumerate(self.frames) if _RUNNER in f), default=-1)
        inner = self.frames[runner + 1:]
        deepest = next((f for f in reversed(inner)
                        if not any(noise in f for noise in _NOISE)), None)
        if deepest is None:
            if not self.frames:
                return "unknown"
            # Only loop machinery on the Python stack: the time went in native
            # code the loop called itself - Qt painting or layout, a WebEngine
            # call, a Windows message - not in any Python function.
            anchor = self.frames[runner] if runner >= 0 else self.frames[-1]
            return f"Qt/native code (no Python frame) under {_short_frame(anchor)}"
        culprit = _short_frame(deepest)
        if not _is_orion_frame(deepest):
            # "run() at onnxruntime_inference_collection.py:320" said what
            # stalled but not which feature asked for it; the nearest ORION
            # frame does (owner's log, 7 Oct 2026).
            ours = next((f for f in reversed(inner) if _is_orion_frame(f)
                         and not any(noise in f for noise in _NOISE)), None)
            if ours is not None:
                culprit += f" from {_short_frame(ours)}"
        return culprit

    def line(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.at))
        meanwhile = self.meanwhile()
        return (f"  {stamp}  {self.ms:7.0f} ms blocked   {self.culprit()}"
                + (f" - meanwhile {meanwhile}" if meanwhile else ""))


def perf_line(stall: Stall) -> str:
    """The log line for a finished stall."""
    meanwhile = stall.meanwhile()
    return (f"PERF: event loop blocked {stall.ms:.0f} ms by {stall.culprit()}"
            + (f" - meanwhile {meanwhile}" if meanwhile else ""))


class StallReporter:
    """Where a stall's PERF: line goes in a real run (Mark XXXV W1.1).

    A block of ``window_ms`` or more goes to *window_log* (the bus log, which
    session_log also copies to disk), as before. A shorter one, down to the
    detector's threshold, goes to *file_log* only - session.log - at most
    *per_minute* lines a minute, so a busy start-up cannot bury the rest of
    the log; the count of any left out rides on the next line written.

    Both callbacks run on the watchdog thread: both destinations must be
    thread-safe (a Qt signal emit and session_log.write are).
    """

    def __init__(self, window_log: Callable[[str], Any] | None,
                 file_log: Callable[[str], Any] | None, *,
                 window_ms: float = DEFAULT_THRESHOLD_MS, per_minute: int = 30,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.window_log = window_log
        self.file_log = file_log
        self.window_ms = float(window_ms)
        self.per_minute = max(1, int(per_minute))
        self._clock = clock
        self._sent: deque[float] = deque()
        self._dropped = 0

    @staticmethod
    def _send(target: Callable[[str], Any] | None, line: str) -> None:
        if target is None:
            return
        try:
            target(line)
        except Exception:
            pass                    # a closed bus at shutdown: never the watchdog's problem

    def finished(self, stall: Stall) -> None:
        line = perf_line(stall)
        if stall.ms >= self.window_ms:
            self._send(self.window_log, line)
            return
        now = self._clock()
        while self._sent and now - self._sent[0] >= 60.0:
            self._sent.popleft()
        if len(self._sent) >= self.per_minute:
            self._dropped += 1
            return
        self._sent.append(now)
        if self._dropped:
            line += f"  (+{self._dropped} shorter block(s) not listed)"
            self._dropped = 0
        self._send(self.file_log, line)

    def ongoing(self, stall: Stall) -> None:
        meanwhile = stall.meanwhile()
        self._send(self.file_log, f"PERF: event loop STILL blocked after "
                                  f"{stall.ms / 1000.0:.1f} s by {stall.culprit()}"
                                  + (f" - meanwhile {meanwhile}" if meanwhile else ""))


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
                 on_stall_progress: Callable[[Stall], None] | None = None,
                 capacity: int = 200) -> None:
        self.tracker = tracker
        self.threshold_ms = float(threshold_ms)
        # on_stall: once per stall, when it ENDS, with its final length.
        # on_stall_progress: during a long one (PROGRESS_MS), with a fresh
        # stack. Both are called on the watchdog thread.
        self.on_stall = on_stall
        self.on_stall_progress = on_stall_progress
        self.stalls: deque[Stall] = deque(maxlen=capacity)
        self._beat = time.monotonic()
        self._loop_thread_id: int | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._beat_handle: asyncio.Handle | None = None
        self._beat_generation: object | None = None
        self._lock = threading.Lock()
        #: (beat it followed, lateness ms) of a late beat, as the loop itself
        #: measured it - the only record of a stall the watchdog froze through.
        self._missed: tuple[float, float] | None = None

    # -- the loop side ---------------------------------------------------------

    def beat(self) -> None:
        """Mark the loop alive. A float store; safe to call very often."""
        self._beat = time.monotonic()

    def _heartbeat(self, loop: Any, generation: object) -> None:
        # A nested Qt loop can dispatch this while an outer coroutine is still
        # executing. Resuming a Task then raises "Cannot enter into task" and
        # loses its scheduled step (owner report, 8 October 2026). A plain
        # callback only updates the timestamp; it needs no task re-entry.
        if self._stop.is_set() or generation is not self._beat_generation:
            return
        self._beat_handle = None
        previous = self._beat
        self.beat()
        late_ms = self._late_ms(previous, self._beat)
        if late_ms >= self.threshold_ms:
            self._missed = (previous, late_ms)
        if not self._stop.is_set() and generation is self._beat_generation:
            self._beat_handle = loop.call_later(
                BEAT_S, self._heartbeat, loop, generation)

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

    @staticmethod
    def _late_ms(beat: float, now: float) -> float:
        """How late the heartbeat is: the time since the last beat beyond the
        BEAT_S it was scheduled for. A healthy loop reads about zero."""
        return (now - beat - BEAT_S) * 1000.0

    def _watch(self) -> None:
        """Detect a stall once, follow it, and report it when it ends.

        Reporting the duration at the moment the threshold is crossed would say
        "160 ms" for a 450 ms freeze — technically a detection, but a misleading
        number, and the number is the whole point. So the stack is captured when
        the stall is first seen (WHILE it is stuck - that stack is the culprit),
        the duration is revised on each tick, and the stall is reported once the
        heartbeat moves again, with its exact final lateness. A long freeze is
        also reported while it lasts (PROGRESS_MS): one that never ends would
        otherwise never be reported at all.

        Each report also says who used the CPU through it (per-thread CPU
        since the last healthy moment), and whether the loop's own thread
        worked or only waited. And when Python's lock was held elsewhere this
        thread froze too: if the loop then got its turn back first, the stale
        beat was never seen here, so the lateness the loop measured itself is
        reported instead, provided this thread was frozen through it as well.
        """
        active: Stall | None = None
        active_beat = 0.0
        next_progress = 0
        clock = _ThreadClock()
        baseline: tuple[float, dict[tuple[int, int], float]] | None = None
        reported_beat: float | None = None
        frozen_at, frozen_ms = 0.0, 0.0
        woke = time.monotonic()
        try:
            while not self._stop.is_set():
                time.sleep(WATCH_S)
                beat = self._beat
                now = time.monotonic()
                overslept = (now - woke - WATCH_S) * 1000.0     # this thread, frozen?
                woke = now
                if overslept >= self.threshold_ms / 2:
                    frozen_at, frozen_ms = now, overslept
                if active is not None:
                    if beat != active_beat:       # the loop got its turn back
                        final = max(self._late_ms(active_beat, beat), active.ms)
                        with self._lock:
                            active.ms = final
                        self._attribute(active, clock, baseline)
                        self._finished(active)
                        reported_beat, active, baseline = active_beat, None, None
                    else:                         # same stall, still going
                        late_ms = self._late_ms(active_beat, now)
                        with self._lock:
                            active.ms = late_ms
                        if late_ms >= self._progress_point(next_progress):
                            next_progress += 1
                            self._progress(active, clock, baseline)
                        continue
                missed, self._missed = self._missed, None
                if (missed is not None and missed[0] != reported_beat
                        and missed[1] <= 60000.0                    # not a sleeping PC
                        and now - frozen_at < 1.0 and frozen_ms >= missed[1] / 2):
                    stall = Stall(ms=missed[1], at=time.time() - missed[1] / 1000.0,
                                  held=True)
                    self._attribute(stall, clock, baseline)
                    with self._lock:
                        self.stalls.append(stall)
                    self._finished(stall)
                    baseline = None
                late_ms = self._late_ms(beat, now)
                if late_ms < self.threshold_ms:
                    # Noted while the loop is well short of a stall - also when
                    # it runs a little late for minutes on end, or the note
                    # would go stale exactly when it is needed.
                    if late_ms < self.threshold_ms / 2 and (
                            baseline is None or now - baseline[0] >= CPU_SAMPLE_S):
                        baseline = (now, clock.snapshot())
                    continue
                active = Stall(ms=late_ms, at=time.time() - late_ms / 1000.0,
                               frames=self._capture())
                active_beat = beat
                next_progress = 0
                with self._lock:
                    self.stalls.append(active)
            if active is not None:                # stopped mid-stall: still say so
                self._finished(active)
        finally:
            clock.close()

    def _attribute(self, stall: Stall, clock: _ThreadClock,
                   baseline: tuple[float, dict[tuple[int, int], float]] | None) -> None:
        """Fill in who used the CPU since the last healthy moment."""
        if baseline is None or not baseline[1]:
            return
        if time.monotonic() - baseline[0] > stall.ms / 1000.0 + CPU_NOTE_SLACK_S:
            return                # too old a note to say who ran through THIS stall
        try:
            before = baseline[1]
            after = clock.snapshot()
            names = {t.ident: t.name for t in threading.enumerate()}
            frames = sys._current_frames()
            me = threading.get_ident()
            rows: list[tuple[float, int]] = []
            for key, total in after.items():
                ident = key[0]
                if ident == me:
                    continue
                used = total - before.get(key, 0.0)       # born since: all its time
                if ident == self._loop_thread_id:
                    stall.loop_cpu_ms = used
                elif used > 0:
                    rows.append((used, ident))
            rows.sort(reverse=True)
            stall.others = [(names.get(ident, f"thread {ident}"), used,
                             _blame(_stack_lines(frames.get(ident))))
                            for used, ident in rows[:3]]
        except Exception:
            pass                  # a diagnostic must never kill the watchdog

    @staticmethod
    def _progress_point(index: int) -> float:
        if index < len(PROGRESS_MS):
            return PROGRESS_MS[index]
        return PROGRESS_MS[-1] + PROGRESS_EVERY_MS * (index - len(PROGRESS_MS) + 1)

    def _progress(self, active: Stall, clock: _ThreadClock | None = None,
                  baseline: tuple[float, dict[tuple[int, int], float]] | None = None) -> None:
        callback = self.on_stall_progress
        if callback is None:
            return
        snapshot = Stall(ms=active.ms, at=active.at,
                         frames=self._capture() or list(active.frames))
        if clock is not None:
            self._attribute(snapshot, clock, baseline)
        try:
            callback(snapshot)
        except Exception:
            pass                  # a broken reporter must not kill the watchdog

    def _finished(self, stall: Stall) -> None:
        if self.tracker is not None:
            self.tracker.record("event-loop stall", stall.ms, stall.culprit())
        if self.on_stall is not None:
            try:
                self.on_stall(stall)
            except Exception:
                pass              # a broken reporter must not kill the watchdog

    def record_stall(self, ms: float, frames: list[str] | None = None) -> Stall:
        """Record a finished stall. Public so it can be driven deterministically
        in tests rather than by trying to make a real thread hang on a timer."""
        stall = Stall(ms=float(ms), at=time.time(), frames=list(frames or []))
        with self._lock:
            self.stalls.append(stall)
        self._finished(stall)
        return stall

    # -- lifecycle -------------------------------------------------------------

    def start(self, loop: Any = None) -> None:
        """Begin watching. Call from inside the loop being watched."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._beat = time.monotonic()
        self._missed = None
        # Identify the watched thread HERE, not inside the heartbeat callback.
        # start() is documented as being called from the loop being watched, and
        # a callback may not get its first turn before something blocks — in which
        # case _capture() had no thread id and every stall reported "unknown",
        # losing the one piece of information the watchdog exists to provide.
        self._loop_thread_id = threading.get_ident()
        loop = loop or _running_loop()
        self._beat_generation = generation = object()
        if loop is not None:
            self._beat_handle = loop.call_later(
                BEAT_S, self._heartbeat, loop, generation)
        self._thread = threading.Thread(target=self._watch,
                                        name="orion-stall-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # A callback already dispatched by Qt must not re-arm a stopped or
        # restarted detector, even if cancellation arrived too late.
        self._beat_generation = None
        handle, self._beat_handle = self._beat_handle, None
        if handle is not None:
            handle.cancel()
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


# ── nested message pumps ──────────────────────────────────────────────────────
#
# "Cannot enter into task" is not noise. It is what happens when ORION code
# running inside an asyncio task pumps the Windows message queue - a modal
# QDialog.exec(), a camera open, ShellExecute, a PortAudio reopen, a COM
# call: qasync then tries to step another task nested inside the first, Python
# refuses, and that other task never resumes (8 Oct 2026: the Live receive loop, dead for
# ten minutes after the camera opened). selfrepair reports only the first one
# per session, as its own scheduling noise; this logs every one, naming the
# task that was pumping and the line that did it.
#
# 28 Oct 2026: logging was not enough - a forget dialog starved
# orion-send-realtime, Gemini Live heard nothing more, and ORION answered what
# the small local recogniser misheard. The refused step is now offered to the
# task again every REENTRY_RETRY_S until the pumping step has ended, so the
# task carries on instead of dying with the session still open.

_NESTED_DISPATCH = re.compile(r'asyncio[\\/]events\.py", line \d+, in _run\b')
_TASK_REPR = re.compile(r"<Task\s+\w+\s+name='([^']*)'(?:\s+coro=<([\w.<>]+?)\()?")

#: The most recent re-entry lines, newest last, for the latency report.
REENTRIES: deque[str] = deque(maxlen=50)
#: How soon a refused task step is offered again (seconds).
REENTRY_RETRY_S = 0.05


def _task_label(match: tuple[str, str]) -> str:
    name, coro = match
    if "<locals>" in coro:                # outer.<locals>.inner -> inner
        coro = coro.rsplit(".", 1)[-1]
    return f"'{name}' ({coro})" if coro else f"'{name}'"


def _installed_handler(loop: Any) -> Any:
    """The exception handler *loop* will actually call.

    qasync keeps it in a private attribute of its own and does not override
    ``get_exception_handler()``, which on ORION's loop therefore always says
    None - chaining to that would silently cut selfrepair out.
    """
    try:
        for key, value in vars(loop).items():
            if key.endswith("__exception_handler"):
                return value
    except TypeError:
        pass
    return loop.get_exception_handler()


def reentry_culprit(stack: list[str]) -> str:
    """The call that pumped, from the stack at the moment of the fault.

    The fault is raised while the pumping task is still inside its own step,
    so the stack runs: loop machinery, that task's frames down to the pumping
    call, then the nested dispatch (asyncio's Handle._run) that tried to step
    the victim. The culprit is the deepest real frame before that dispatch.
    """
    cut = len(stack)
    for index in range(len(stack) - 1, -1, -1):
        if _NESTED_DISPATCH.search(stack[index]):
            cut = index
            break
    return Stall(ms=0.0, at=0.0, frames=list(stack[:cut])).culprit()


def reentry_line(message: str, stack: list[str]) -> str:
    """One log line for a "Cannot enter into task" fault."""
    found = _TASK_REPR.findall(message)
    victim = _task_label(found[0]) if found else "a task"
    pumper = _task_label(found[1]) if len(found) > 1 else "another task"
    return (f"LOOP: Cannot enter into task {victim} - {pumper} pumped the Windows "
            f"message queue inside its own step at {reentry_culprit(stack)}; "
            f"{victim} is stepped again once that step ends.")


def retry_refused_step(loop: Any, context: dict[str, Any]) -> bool:
    """Offer the refused step to its task again until the loop lets it in.

    *context* is the loop's exception context for the fault; its handle is the
    task's own step (or wake-up) that Python would not enter. Nothing else will
    ever schedule that task again, so this is the only way it resumes.
    """
    handle = context.get("handle")
    callback = getattr(handle, "_callback", None)
    task = getattr(callback, "__self__", None)
    if not isinstance(task, asyncio.Task):
        return False
    args = tuple(getattr(handle, "_args", None) or ())
    step_context = getattr(handle, "_context", None)

    def again() -> None:
        if task.done():
            return
        try:
            callback(*args)
        except RuntimeError as exc:
            if "cannot enter into task" not in str(exc).lower():
                raise
            loop.call_later(REENTRY_RETRY_S, again, context=step_context)

    loop.call_later(REENTRY_RETRY_S, again, context=step_context)
    return True


def install_reentry_watch(loop: Any, log: Callable[[str], Any]) -> bool:
    """Log every "Cannot enter into task" on *loop*, offer the refused task
    its step again (retry_refused_step), then hand the fault on to the
    handler installed before this one (selfrepair's) unchanged.

    Install after selfrepair: ``set_exception_handler`` replaces, it does not
    chain, so whichever is installed last is the one that runs.
    """
    try:
        previous = _installed_handler(loop)
    except Exception:
        return False

    def handler(the_loop: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if exc is not None and "cannot enter into task" in str(exc).lower():
            try:
                line = reentry_line(str(exc), traceback.format_stack())
                REENTRIES.append(line)
                log(line)
            except Exception:
                pass
            try:
                retry_refused_step(the_loop, context)
            except Exception:
                pass
        if previous is not None:
            previous(the_loop, context)
        else:
            the_loop.default_exception_handler(context)

    try:
        loop.set_exception_handler(handler)
    except Exception:
        return False
    return True


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
    text = DETECTOR.report() + "\n\n" + TRACKER.report()
    reentries = list(REENTRIES)
    if reentries:
        text += (f"\n\n{len(reentries)} task re-entry fault(s) - a Windows message "
                 "pump inside a task, newest last:\n" + "\n".join(
                     "  " + line for line in reentries[-5:]))
    return text


__all__ = [
    "BEAT_S", "DEFAULT_THRESHOLD_MS", "DETECTOR", "LOG_THRESHOLD_MS",
    "PROGRESS_MS", "REENTRIES", "TRACKER", "LatencyTracker", "Span", "Stall",
    "StallDetector", "StallReporter", "Summary", "install_reentry_watch",
    "percentile", "perf_line", "record", "reentry_culprit", "reentry_line",
    "retry_refused_step",
    "report", "span",
]
