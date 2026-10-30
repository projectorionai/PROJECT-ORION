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

Warm and cold starts (Mark XXXX.3). Six boots on 27 Oct: warm starts took
3.37-3.48 s (and 4.38 s), while the first start after Windows came up took
8.6 s - 2.8 s of it before the first line of ORION ran, Windows reading
Python and Qt off the disk - and the first after an install 5.1 s (Defender
scanning the new files as they loaded). Those are the disk's, not ORION's,
so a warm start is held to WARM_TARGET_S and a cold one (the launcher phase
over COLD_LAUNCH_S) to COLD_TARGET_S, and the report says which it was.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

#: Readiness is not one moment. A painted face does not mean the microphone is
#: open, and an open microphone does not mean a model will answer. Each of
#: these is recorded the first time it becomes true, separately from the boot
#: phases, because the later ones arrive after the boot report is written.
READINESS: tuple[str, ...] = (
    "first_paint",        # the Core Window is on screen
    "command_ready",      # typed/remote commands are dispatched
    "microphone_ready",   # audio capture is running
    "provider_ready",     # a model has accepted a session or answered
)


@dataclass(frozen=True)
class Phase:
    name: str
    at: float       # seconds since the budget started, when this mark was taken
    delta: float    # seconds since the previous mark (or since start for the first)


#: A launcher + imports + Qt phase this long means Windows was reading
#: ORION from disk, not from its cache (a warm one takes 0.17-0.47 s).
COLD_LAUNCH_S = 1.0
WARM_TARGET_S = 4.0
COLD_TARGET_S = 9.0


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
        self.milestones: dict[str, float] = {}
        self._milestone_lock = threading.Lock()
        self._on_milestone: Callable[[str, float], Any] | None = None

    def on_milestone(self, callback: Callable[[str, float], Any] | None) -> None:
        """Called (name, seconds since start) when a milestone is first reached."""
        self._on_milestone = callback

    def milestone(self, name: str) -> float | None:
        """Record that ``name`` is ready. Only the first report counts: a
        reconnect later in the session is not a slower boot.

        Safe from any thread; the callback runs on the caller's thread and may
        not raise into it. Returns the seconds since start, or None if the
        milestone had already been recorded.
        """
        with self._milestone_lock:
            if name in self.milestones:
                return None
            at = self._clock() - self._t0
            self.milestones[name] = at
        callback = self._on_milestone
        if callback is not None:
            try:
                callback(name, at)
            except Exception:
                pass
        return at

    def readiness_line(self) -> str:
        """``first_paint 0.65s, command_ready 3.12s, microphone_ready -``."""
        known = list(READINESS) + [n for n in self.milestones if n not in READINESS]
        return ", ".join(
            f"{name} {self.milestones[name]:.2f}s" if name in self.milestones
            else f"{name} -" for name in known)

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

    def cold(self) -> bool:
        """Was this a cold start (the first phase spent reading from disk)?"""
        return bool(self.phases) and self.phases[0].delta > COLD_LAUNCH_S

    def apply_start_kind(self) -> str:
        """Hold a warm start to WARM_TARGET_S and a cold one to COLD_TARGET_S.
        Returns "cold" or "warm"."""
        kind = "cold" if self.cold() else "warm"
        self.target_seconds = COLD_TARGET_S if kind == "cold" else WARM_TARGET_S
        return kind

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
        kind = ""
        if self.cold():
            kind = (f"; cold start - {self.phases[0].delta:.1f}s before ORION's first line, "
                    "Windows reading him from disk")
        lines.append(
            f"  total {self.total:.2f}s ({verdict} {self.target_seconds:.0f}s target{kind})")
        lines.append(f"  readiness: {self.readiness_line()}")
        return "\n".join(lines)


# ── the boot in progress ──────────────────────────────────────────────────────
#
# The microphone and the model become ready deep inside the audio engine and
# the Live worker, long after run_application has finished building things.
# They report through this module rather than being handed the budget.

_current: StartupBudget | None = None


def set_current(budget: StartupBudget | None) -> None:
    global _current
    _current = budget


def current() -> StartupBudget | None:
    return _current


def milestone(name: str) -> None:
    """Record a readiness milestone on the running boot, if there is one."""
    budget = _current
    if budget is not None:
        try:
            budget.milestone(name)
        except Exception:
            pass
