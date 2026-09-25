"""
Startup-time budget (Mark X.12 §5.1; enforcement added in the Mark XX
architectural-audit pass).

Instruments the composition root so a slow launch is *diagnosable* and a
regression — a new heavy step that blocks first paint — is visible rather than
merely felt. Callers ``mark()`` each major phase; ``report()`` renders the
per-phase deltas, the running total, and flags any phase over a slow threshold
or a total over the target.

``target_seconds`` is caller-supplied, not a hardcoded constant here: the real
desktop app (``app.py``) constructs this with ``target_seconds=5.0``, which is
the honest number for a boot sequence that brings up audio, vision, a live
model connection and every JARVIS subsystem — not the 2s figure an earlier
draft of this docstring claimed for a "headless" mode that was never actually
wired to a smaller target anywhere in the codebase. ``over_budget()`` is no
longer purely advisory: ``app.py`` checks it after the last phase mark and
logs a WARN plus bumps a telemetry counter when it trips, so a regression
shows up in both the live log and the metrics history, not only in a report
nobody reads unless they go looking. The GUI/audio startup can't run headless
in CI, so the report itself is still logged at runtime rather than asserted —
but the utility is fully unit-tested with an injected clock. A real clock is
the default; tests pass a deterministic one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Phase:
    name: str
    at: float       # seconds since the budget started, when this mark was taken
    delta: float    # seconds since the previous mark (or since start for the first)


class StartupBudget:
    def __init__(self, *, target_seconds: float = 5.0,
                 slow_phase_seconds: float = 0.2,
                 clock: Callable[[], float] = time.perf_counter,
                 started_at: float | None = None) -> None:
        self._clock = clock
        # A launcher can pass its perf_counter timestamp so imports and Qt
        # setup count towards the user's wait, rather than disappearing from
        # the report before the application coroutine begins.
        self._t0 = clock() if started_at is None else float(started_at)
        self._last = self._t0
        self.target_seconds = float(target_seconds)
        self.slow_phase_seconds = float(slow_phase_seconds)
        self.phases: list[Phase] = []

    def mark(self, name: str) -> float:
        """Record the end of a phase; returns its duration in seconds."""
        now = self._clock()
        delta = now - self._last
        self.phases.append(Phase(name=name, at=now - self._t0, delta=delta))
        self._last = now
        return delta

    @property
    def total(self) -> float:
        """Elapsed time — up to the last mark if any, else up to now."""
        if self.phases:
            return self.phases[-1].at
        return self._clock() - self._t0

    def over_budget(self) -> bool:
        return self.total > self.target_seconds

    def slow_phases(self) -> list[Phase]:
        return [p for p in self.phases if p.delta > self.slow_phase_seconds]

    def breach_summary(self) -> str:
        """A short, actionable message for when over_budget() is True — the
        slowest phases first, so a regression is diagnosable at a glance
        without reading the full phase-by-phase report."""
        slow = sorted(self.slow_phases(), key=lambda p: p.delta, reverse=True)[:3]
        culprits = ", ".join(f"{p.name} (+{p.delta * 1000:.0f}ms)" for p in slow)
        return (
            f"startup budget exceeded: {self.total:.2f}s > {self.target_seconds:.0f}s target"
            + (f" — slowest: {culprits}" if culprits else " — no single phase stands out")
        )

    def report(self) -> str:
        lines = ["STARTUP BUDGET:"]
        for p in self.phases:
            flag = "   <- slow" if p.delta > self.slow_phase_seconds else ""
            lines.append(
                f"  {p.at:6.2f}s  (+{p.delta * 1000:6.0f} ms)  {p.name}{flag}")
        verdict = "OVER" if self.over_budget() else "within"
        lines.append(
            f"  total {self.total:.2f}s ({verdict} {self.target_seconds:.0f}s target)")
        return "\n".join(lines)
