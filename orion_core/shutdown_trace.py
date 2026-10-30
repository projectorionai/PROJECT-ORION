"""
Where shutdown spends its time — and where it was when the watchdog fired.

Start-up has had a phase-by-phase budget for a long time (startup_budget.py);
shutdown had nothing. So when quitting ran past the 25-second watchdog, the
process simply vanished: ``os._exit`` skips every handler, the activity log
only ever lived in the GUI, and nothing on disk said which of ~30 teardown
steps had been holding it. A smoke test on 2026-09-23 measured a quit taking
30.5 s — past its own watchdog — with no way to tell why.

Each teardown step now runs inside ``TRACE.phase(label)``. A clean exit writes
the per-step timings; the watchdog writes the same report, marked FORCED and
naming the step it interrupted, immediately before it kills the process. Both
go to ``config/diagnostics/last_shutdown.txt``, overwritten each time.

Naming the step was not enough. On 8 and 9 Oct 2026 the installed app was
FORCED "Still inside: remote gateway" although that build's gateway stop was
already capped at 6 s, and a reproduction with every kind of phone stream
open never took longer than 4 s. "Still inside" cannot tell a step that is
WAITING (an await that never ends) from a step during which the event loop
itself was BLOCKED by synchronous code somewhere else. So teardown now also
records, and the report says:

* every awaited step that hit its budget, as "gave up after X s", with where
  it was suspended (its await chain, innermost first);
* every synchronous step that ran over its budget (it cannot be interrupted);
* every stretch the event loop could not run, from a teardown heartbeat;
* and, when FORCED, how long ago the loop last ran and the main thread's
  stack at that moment.

``budget()`` also gives the whole teardown one deadline inside the watchdog,
so several slow steps cannot add up past it either.

Writing the report never raises: it is diagnostics on the way out, and must
not be the thing that stops ORION leaving.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

_ASYNCIO_DIR = Path(asyncio.__file__).resolve().parent


def _where(frame: Any) -> str:
    code = frame.f_code
    return f"{Path(code.co_filename).name}:{frame.f_lineno} in {code.co_name}()"


def waiting_at(task: Any, depth: int = 3) -> str:
    """Where an awaited step is suspended: the innermost frames of its await
    chain outside asyncio itself, innermost first, e.g.
    ``web_server.py:71 in shutdown() <- web_runner.py:330 in cleanup()``.
    Follows into a Task it awaits. Empty when it cannot be told; never raises
    (the watchdog calls it from another thread)."""
    try:
        frames = []
        coro = task.get_coro() if hasattr(task, "get_coro") else task
        for _ in range(100):
            if coro is None:
                break
            if isinstance(coro, asyncio.Future):
                get_coro = getattr(coro, "get_coro", None)
                coro = get_coro() if get_coro is not None else None
                continue
            frame = (getattr(coro, "cr_frame", None) or getattr(coro, "gi_frame", None)
                     or getattr(coro, "ag_frame", None))
            if frame is not None:
                frames.append(frame)
            coro = (getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
                    or getattr(coro, "ag_await", None))
        own = [f for f in frames
               if Path(f.f_code.co_filename).resolve().parent != _ASYNCIO_DIR]
        chosen = (own or frames)[-max(1, depth):]
        return " <- ".join(_where(f) for f in reversed(chosen))
    except Exception:
        return ""


def main_thread_at(depth: int = 4) -> str:
    """The main thread's innermost frames right now, innermost first. ORION's
    event loop runs on the main thread, so while it is blocked this is the
    code blocking it. Never raises."""
    try:
        frame = sys._current_frames().get(threading.main_thread().ident)
        if frame is None:
            return ""
        stack = traceback.extract_stack(frame)[-max(1, depth):]
        return " <- ".join(f"{Path(s.filename).name}:{s.lineno} in {s.name}()"
                           for s in reversed(stack))
    except Exception:
        return ""


class ShutdownTrace:
    #: Teardown has this long from the moment shutdown was requested. What is
    #: left of the 25 s watchdog covers the closes after the last awaited
    #: step, the report and the interpreter's own exit.
    TEARDOWN_BUDGET_S = 20.0
    #: However late it is, an awaited step still gets this long, so one that
    #: has already finished can say so instead of being given up on unseen.
    MIN_STEP_S = 0.1
    #: An event-loop stall worth naming in the report.
    STALL_S = 0.25

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.started: float | None = None
        self._stack: list[str] = []
        self.phases: list[tuple[str, float]] = []
        #: Threads the process was still waiting for once teardown returned.
        self.lingering: list[str] = []
        #: Awaited steps abandoned at their budget: (step, budget s, detail).
        self.gave_up: list[tuple[str, float, str]] = []
        #: Synchronous steps over their budget: (step, seconds, budget s).
        self.slow: list[tuple[str, float, float]] = []
        #: Stretches the event loop could not run: (step(s), seconds).
        self.stalls: list[tuple[str, float]] = []
        #: The step's task being awaited right now, for the watchdog's report.
        self.awaiting: Any = None
        #: When the teardown heartbeat last ran (the loop was alive then).
        self.last_beat: float | None = None

    def begin(self) -> None:
        """Mark the moment shutdown was requested (idempotent)."""
        with self._lock:
            if self.started is None:
                self.started = self._clock()

    @contextmanager
    def phase(self, label: str) -> Iterator[None]:
        """Time one teardown step; ``current`` names it while it runs."""
        with self._lock:
            self._stack.append(label)
            path = " > ".join(self._stack)
        began = self._clock()
        try:
            yield
        finally:
            with self._lock:
                self.phases.append((path, self._clock() - began))
                if self._stack and self._stack[-1] == label:
                    self._stack.pop()

    @property
    def current(self) -> str | None:
        """The innermost step running now, as its full path ("worker > mic")."""
        with self._lock:
            return " > ".join(self._stack) or None

    def elapsed(self) -> float:
        return 0.0 if self.started is None else self._clock() - self.started

    # ── budgets ──────────────────────────────────────────────────────────────

    def budget(self, own: float) -> float:
        """A step's own budget, cut to what is left of the teardown deadline
        (never below ``MIN_STEP_S``). Before ``begin()`` it is the step's own."""
        own = max(0.0, float(own))
        if self.started is None:
            return own
        left = self.started + self.TEARDOWN_BUDGET_S - self._clock()
        return max(self.MIN_STEP_S, min(own, left))

    def note_gave_up(self, label: str, budget: float, where: str = "", *,
                     still_running: bool = False, detail: str | None = None) -> None:
        """Record a step abandoned at its budget. *detail* replaces the
        default "cancelled/still running; waiting at ..." text, for a step
        that bounds itself and reports what it left behind (a drain)."""
        if detail is None:
            detail = "still running" if still_running else "cancelled"
            if where:
                detail += f"; waiting at {where}"
        with self._lock:
            self.gave_up.append((label, float(budget), detail))

    def note_slow(self, label: str, seconds: float, budget: float) -> bool:
        """Record a synchronous step that took longer than *budget*; True if
        it did. It cannot be interrupted, only named."""
        if seconds <= budget:
            return False
        with self._lock:
            self.slow.append((label, float(seconds), float(budget)))
        return True

    @staticmethod
    def waiting_at(task: Any) -> str:
        return waiting_at(task)

    async def watch_loop(self, period: float = 0.1) -> None:
        """Teardown heartbeat: note every time the event loop could not run
        for ``STALL_S`` or more, with the step(s) it happened in. Runs until
        cancelled."""
        while True:
            before = self._clock()
            self.last_beat = before
            label = self.current
            await asyncio.sleep(period)
            now = self._clock()
            self.last_beat = now
            late = now - before - period
            if late >= self.STALL_S:
                after = self.current
                where = label or "between steps"
                if after != label:
                    where = f"{where} .. {after or 'between steps'}"
                with self._lock:
                    self.stalls.append((where, late))

    # ── the report ───────────────────────────────────────────────────────────

    def _forced_lines(self) -> list[str]:
        lines = []
        task = self.awaiting
        if task is not None:
            where = waiting_at(task)
            if where:
                lines.append(f"  awaiting: {where}")
        if self.last_beat is not None:
            idle = self._clock() - self.last_beat
            if idle >= 1.0:
                lines.append(f"  event loop: BLOCKED - it last ran {idle:.1f} s ago")
            else:
                lines.append(f"  event loop: running (last ran {idle:.1f} s ago) - "
                             "the step is waiting, not blocking it")
        main = main_thread_at()
        if main:
            lines.append(f"  main thread: {main}")
        return lines

    def report(self, *, forced: bool = False) -> str:
        current = self.current
        with self._lock:
            phases = list(self.phases)
            gave_up = list(self.gave_up)
            slow = list(self.slow)
            stalls = list(self.stalls)
        total = self.elapsed()
        first = (("FORCED by the shutdown watchdog" if forced else "Clean shutdown")
                 + f" after {total:.1f} s")
        if gave_up and not forced:
            first += f"; {len(gave_up)} step(s) gave up at their budget"
        lines = [first + "."]
        if forced:
            lines.append("Still inside: " + (current or "no traced step — between steps "
                                             "or in untraced teardown"))
            lines.extend(self._forced_lines())
        spent = sum(seconds for _label, seconds in phases)
        lines.append(f"{len(phases)} traced step(s), {spent:.1f} s between them. Slowest first:")
        budgets = {label: budget for label, budget, _detail in gave_up}
        for label, seconds in sorted(phases, key=lambda item: item[1], reverse=True)[:15]:
            mark = f"  (gave up after {budgets[label]:.1f} s)" if label in budgets else ""
            lines.append(f"  {seconds:6.2f} s  {label}{mark}")
        if gave_up:
            lines.append("Gave up at their budget (cancelled and left behind):")
            lines.extend(f"  {label}: gave up after {budget:.1f} s; {detail}"
                         for label, budget, detail in gave_up)
        if slow:
            lines.append("Over budget (synchronous, could not be interrupted):")
            lines.extend(f"  {label}: {seconds:.2f} s, budget {budget:.1f} s"
                         for label, seconds, budget in slow)
        if stalls:
            lines.append("Event loop blocked during teardown:")
            lines.extend(f"  {seconds:6.2f} s  {where}"
                         for where, seconds in sorted(stalls, key=lambda s: s[1], reverse=True)[:8])
        if self.lingering:
            lines.append("After teardown the process still had to wait for:")
            lines.extend(f"  {entry}" for entry in self.lingering)
        return "\n".join(lines) + "\n"

    def write(self, path: Path, *, forced: bool = False) -> bool:
        """Persist the report. Never raises."""
        try:
            from .atomic_io import atomic_write_text
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, time.strftime("%Y-%m-%d %H:%M:%S  ")
                              + self.report(forced=forced))
            return True
        except Exception:
            return False


def lingering_threads() -> list[str]:
    """Non-daemon threads the interpreter must wait for before the process can
    end, each with the line it is on.

    Teardown finishing is not the process finishing: Python joins every
    non-daemon thread first, including an executor worker still in the middle
    of a job. One measured quit on 2026-09-23 finished teardown in 4.0 s and
    the process at 10.0 s, and nothing said what the other six were spent on.

    An IDLE executor worker is left out — it is parked in its own _worker loop
    and is woken and joined at exit in no time.
    """
    frames = sys._current_frames()
    found = []
    for thread in threading.enumerate():
        if thread.daemon or thread is threading.main_thread():
            continue
        frame = frames.get(thread.ident)
        where = ""
        if frame is not None:
            code = frame.f_code
            if code.co_name == "_worker" and code.co_filename.replace("\\", "/").endswith(
                    "concurrent/futures/thread.py"):
                continue
            where = f"  at {Path(code.co_filename).name}:{frame.f_lineno} in {code.co_name}()"
        found.append(thread.name + where)
    return found


#: The one trace for this process's shutdown.
TRACE = ShutdownTrace()

__all__ = ["ShutdownTrace", "TRACE", "lingering_threads", "main_thread_at", "waiting_at"]
