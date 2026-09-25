"""
Resource Governor — ORION shedding its own optional work when memory runs out.

``resource_monitor`` already classified pressure, but only when someone asked
(the ``resource_status`` tool), and its list of sheddable work was read by
nothing. On the machine this was written for — 16 GB, measured at 98% used
with 430 MB free — every optional thing kept running while Chromium ran out of
room and started discarding the face's surfaces (the "black stutter").

This is the part that ACTS. It samples physical memory every few seconds and
moves between states with hysteresis, so a momentary spike changes nothing:

    NORMAL  ->  CONSTRAINED (>= 85% used)  ->  PRESSURED (>= 92%)  ->  CRITICAL (>= 96%)

A level is entered after it holds for two samples and left after three samples
clearly below it; leaving PRESSURED or worse passes through RECOVERY (30 s)
before anything shed is brought back, so it cannot flap.

Components register what they can give up at a level (and how to restore it).
Background loops ask ``optional_work_allowed()`` before optional work. Voice,
the face, and whatever the user is doing right now are never shed — the point
is graceful degradation of the extras, not a quieter ORION.

Thresholds: ORION_MEM_THRESHOLDS="85,92,96" (percent used). ORION_GOVERNOR=0
turns acting off (it still reports).
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable


class Pressure(IntEnum):
    NORMAL = 0
    CONSTRAINED = 1
    PRESSURED = 2
    CRITICAL = 3


@dataclass
class Shedder:
    name: str
    level: Pressure
    shed: Callable[[], Any]
    restore: Callable[[], Any] | None = None
    active: bool = False


def _thresholds() -> tuple[float, float, float]:
    raw = os.getenv("ORION_MEM_THRESHOLDS", "").strip()
    try:
        values = tuple(float(v) for v in raw.split(",")) if raw else ()
    except ValueError:
        values = ()
    if len(values) == 3 and values[0] < values[1] < values[2] <= 100:
        return values  # type: ignore[return-value]
    return (85.0, 92.0, 96.0)


def _read_memory() -> tuple[float, float] | None:
    """(percent used, MB available) or None."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return float(vm.percent), float(vm.available) / 2**20
    except Exception:
        return None


class ResourceGovernor:
    SAMPLE_S = 5.0
    ENTER_SAMPLES = 2
    LEAVE_SAMPLES = 3
    HYSTERESIS = 3.0            # percentage points below a threshold to count as "clear"
    RECOVERY_S = 30.0

    def __init__(self, bus: Any = None, *, reader: Callable[[], Any] = _read_memory,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.bus = bus
        self.reader = reader
        self.clock = clock
        self.thresholds = _thresholds()
        self.level = Pressure.NORMAL
        self.recovering_until = 0.0
        self.last_reading: tuple[float, float] | None = None
        self.shedders: list[Shedder] = []
        self._above = 0
        self._below = 0
        self._candidate = Pressure.NORMAL
        self._stop = asyncio.Event()
        self.acting = os.getenv("ORION_GOVERNOR", "1").strip().lower() not in {
            "0", "false", "no", "off"}

    # ── registration ────────────────────────────────────────────────────────

    def register(self, name: str, level: Pressure, shed: Callable[[], Any],
                 restore: Callable[[], Any] | None = None) -> None:
        self.shedders.append(Shedder(name, Pressure(level), shed, restore))

    # ── queries ─────────────────────────────────────────────────────────────

    @property
    def recovering(self) -> bool:
        return self.clock() < self.recovering_until

    def effective(self) -> Pressure:
        """The level optional work should respect (RECOVERY keeps PRESSURED)."""
        if self.recovering:
            return max(self.level, Pressure.PRESSURED)
        return self.level

    def optional_work_allowed(self) -> bool:
        return not self.acting or self.effective() < Pressure.PRESSURED

    def state_name(self) -> str:
        return "recovery" if self.recovering and self.level < Pressure.PRESSURED \
            else self.level.name.lower()

    def status(self) -> dict[str, Any]:
        used, free = self.last_reading or (None, None)
        return {
            "state": self.state_name(),
            "memory_used_percent": used,
            "memory_free_mb": None if free is None else round(free),
            "thresholds": list(self.thresholds),
            "shedding": [s.name for s in self.shedders if s.active],
            "acting": self.acting,
        }

    # ── the state machine ───────────────────────────────────────────────────

    def _target(self, used: float) -> Pressure:
        constrained, pressured, critical = self.thresholds
        if used >= critical:
            return Pressure.CRITICAL
        if used >= pressured:
            return Pressure.PRESSURED
        if used >= constrained:
            return Pressure.CONSTRAINED
        return Pressure.NORMAL

    def observe(self, used: float, free_mb: float = 0.0) -> Pressure:
        """Feed one reading; returns the (possibly new) level. Pure apart from
        the shedders it runs, so the whole policy is testable with numbers."""
        self.last_reading = (float(used), float(free_mb))
        target = self._target(used)
        previous = self.level
        if target > self.level:
            self._below = 0
            self._above = self._above + 1 if target >= self._candidate else 1
            self._candidate = target
            if self._above >= self.ENTER_SAMPLES:
                self.level, self._above = target, 0
        elif target < self.level:
            self._above = 0
            clear = used < self.thresholds[int(self.level) - 1] - self.HYSTERESIS
            self._below = self._below + 1 if clear else 0
            if self._below >= self.LEAVE_SAMPLES:
                self.level, self._below = target, 0
                if previous >= Pressure.PRESSURED > target:
                    self.recovering_until = self.clock() + self.RECOVERY_S
        else:
            self._above = self._below = 0
        if self.level != previous:
            self._announce(previous)
        self._apply()
        return self.level

    def _apply(self) -> None:
        if not self.acting:
            return
        level = self.effective()
        for shedder in self.shedders:
            should = level >= shedder.level
            if should and not shedder.active:
                shedder.active = True
                self._call(shedder.shed, f"shed {shedder.name}")
            elif not should and shedder.active:
                shedder.active = False
                if shedder.restore is not None:
                    self._call(shedder.restore, f"restore {shedder.name}")

    def _call(self, fn: Callable[[], Any], what: str) -> None:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    result.close()
        except Exception as exc:
            self._log(f"RESOURCE: could not {what} - {exc}")

    def _announce(self, previous: Pressure) -> None:
        used, free = self.last_reading or (0.0, 0.0)
        if self.level > previous:
            shed = [s.name for s in self.shedders if s.level <= self.level]
            self._log(
                f"RESOURCE: memory {used:.0f}% used ({free / 1024:.1f} GB free) — "
                f"{self.level.name}. "
                + (f"Easing off: {', '.join(shed)}. " if shed and self.acting else "")
                + "Voice, the face and the current task are untouched.")
        else:
            self._log(f"RESOURCE: memory {used:.0f}% used — back to {self.level.name}"
                      + (" (recovering before restoring anything)" if self.recovering else "")
                      + ".")

    def _log(self, text: str) -> None:
        if self.bus is not None:
            try:
                self.bus.log.emit(text)
            except Exception:
                pass

    # ── the loop ────────────────────────────────────────────────────────────

    def tick(self) -> None:
        reading = self.reader()
        if reading is not None:
            self.observe(*reading)
        elif self.recovering is False:
            self._apply()

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.SAMPLE_S)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()


def release_python_memory() -> None:
    """A full collection — cheap relative to what it can return under CRITICAL."""
    gc.collect()


#: The process-wide governor (set by app.py); None in tests and tools.
GOVERNOR: ResourceGovernor | None = None


def optional_work_allowed() -> bool:
    """For background loops: False while memory is pressured or recovering."""
    return GOVERNOR is None or GOVERNOR.optional_work_allowed()


__all__ = ["GOVERNOR", "Pressure", "ResourceGovernor", "Shedder",
           "optional_work_allowed", "release_python_memory"]
