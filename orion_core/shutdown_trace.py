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

Writing the report never raises: it is diagnostics on the way out, and must
not be the thing that stops ORION leaving.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


class ShutdownTrace:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self.started: float | None = None
        self._stack: list[str] = []
        self.phases: list[tuple[str, float]] = []
        #: Threads the process was still waiting for once teardown returned.
        self.lingering: list[str] = []

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

    def report(self, *, forced: bool = False) -> str:
        current = self.current
        with self._lock:
            phases = list(self.phases)
        total = self.elapsed()
        lines = [
            ("FORCED by the shutdown watchdog" if forced else "Clean shutdown")
            + f" after {total:.1f} s.",
        ]
        if forced:
            lines.append("Still inside: " + (current or "no traced step — between steps "
                                             "or in untraced teardown"))
        spent = sum(seconds for _label, seconds in phases)
        lines.append(f"{len(phases)} traced step(s), {spent:.1f} s between them. Slowest first:")
        for label, seconds in sorted(phases, key=lambda item: item[1], reverse=True)[:15]:
            lines.append(f"  {seconds:6.2f} s  {label}")
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

__all__ = ["ShutdownTrace", "TRACE", "lingering_threads"]
