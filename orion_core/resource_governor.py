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

CPU load and GPU memory are sampled on the same tick, each through a gauge
with its own hysteresis (busy after two samples at or above the upper
threshold, clear after three below the lower one: ORION_CPU_BUSY="85,70",
ORION_GPU_BUSY="85,75"). They do not shed anything themselves; admission
reads them to defer optional work. A gauge with no reading is None — unknown —
and GPU telemetry is only read once something else has initialised it, so the
loop never pays for loading NVML or launching nvidia-smi.
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


def _load_thresholds(name: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = os.getenv(name, "").strip()
    try:
        values = tuple(float(v) for v in raw.split(",")) if raw else ()
    except ValueError:
        values = ()
    if len(values) == 2 and 0 < values[1] < values[0] <= 100:
        return values  # type: ignore[return-value]
    return default


class _CpuSampler:
    """Whole-machine CPU percent from ``cpu_times`` deltas.

    Not ``psutil.cpu_percent(interval=None)``: that keeps ONE module-wide
    baseline, so every other caller (the telemetry panel samples it each
    second) would silently shorten this reading's window.
    """

    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None

    def __call__(self) -> float | None:
        try:
            import psutil
            times = psutil.cpu_times()
        except Exception:
            return None
        idle = float(getattr(times, "idle", 0.0)) + float(getattr(times, "iowait", 0.0))
        total = float(sum(times))
        previous, self._last = self._last, (idle, total)
        if previous is None or total <= previous[1]:
            return None
        busy = 1.0 - (idle - previous[0]) / (total - previous[1])
        return max(0.0, min(100.0, busy * 100.0))


def _read_gpu_memory() -> float | None:
    """GPU memory percent in use, or None when unknown.

    Only reads telemetry something else already initialised (the core
    window's sampler): the first NVML load, or an nvidia-smi launch, must never
    land on the event loop just because the governor ticked.
    """
    try:
        from . import gpu_stats
        if not gpu_stats._state.get("init"):
            return None
        reading = gpu_stats.latest()
    except Exception:
        return None
    if not reading or not reading.get("available"):
        return None
    try:
        return float(reading.get("mem_percent"))
    except (TypeError, ValueError):
        return None


class LoadGauge:
    """A busy flag with hysteresis — True, False, or None while unknown."""

    ENTER_SAMPLES = 2
    LEAVE_SAMPLES = 3

    def __init__(self, enter: float, leave: float) -> None:
        self.enter = float(enter)
        self.leave = float(leave)
        self.state: bool | None = None
        self.value: float | None = None
        self._above = 0
        self._below = 0

    def observe(self, value: float | None) -> bool | None:
        if value is None:
            # No reading is not the same as an idle device: say unknown.
            self.value, self.state, self._above, self._below = None, None, 0, 0
            return None
        self.value = float(value)
        if self.state is None:
            self.state = False
        if not self.state:
            self._above = self._above + 1 if value >= self.enter else 0
            if self._above >= self.ENTER_SAMPLES:
                self.state, self._above = True, 0
        else:
            self._below = self._below + 1 if value < self.leave else 0
            if self._below >= self.LEAVE_SAMPLES:
                self.state, self._below = False, 0
        return self.state


class ResourceGovernor:
    SAMPLE_S = 5.0
    ENTER_SAMPLES = 2
    LEAVE_SAMPLES = 3
    HYSTERESIS = 3.0            # percentage points below a threshold to count as "clear"
    RECOVERY_S = 30.0

    def __init__(self, bus: Any = None, *, reader: Callable[[], Any] = _read_memory,
                 clock: Callable[[], float] = time.monotonic,
                 cpu_reader: Callable[[], float | None] | None = None,
                 gpu_reader: Callable[[], float | None] | None = _read_gpu_memory) -> None:
        self.bus = bus
        self.reader = reader
        self.clock = clock
        self.cpu_reader = cpu_reader if cpu_reader is not None else _CpuSampler()
        self.gpu_reader = gpu_reader
        self.cpu = LoadGauge(*_load_thresholds("ORION_CPU_BUSY", (85.0, 70.0)))
        self.gpu = LoadGauge(*_load_thresholds("ORION_GPU_BUSY", (85.0, 75.0)))
        self._listeners: list[Callable[["ResourceGovernor"], Any]] = []
        self.thresholds = _thresholds()
        self.level = Pressure.NORMAL
        self.recovering_until = 0.0
        self.last_reading: tuple[float, float] | None = None
        self.memory_sample_fresh = False
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

    def add_listener(self, listener: Callable[["ResourceGovernor"], Any]) -> None:
        """Called with the governor after every sample (admission re-plans)."""
        self._listeners.append(listener)

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

    def cpu_busy(self) -> bool | None:
        return self.cpu.state

    def gpu_busy(self) -> bool | None:
        return self.gpu.state

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
            "cpu_percent": None if self.cpu.value is None else round(self.cpu.value, 1),
            "cpu_busy": self.cpu.state,
            "gpu_memory_percent": None if self.gpu.value is None else round(self.gpu.value, 1),
            "gpu_busy": self.gpu.state,
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
                    asyncio.get_running_loop()
                except RuntimeError:
                    result.close()
                else:
                    # Owned and observed: a bare create_task could be
                    # collected mid-flight, and its failure went unreported.
                    from . import background
                    background.spawn(result, name=f"resource: {what}")
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

    def observe_load(self, cpu_percent: float | None = None,
                     gpu_memory_percent: float | None = None) -> None:
        """Feed one CPU / GPU-memory reading (None = unknown)."""
        self.cpu.observe(cpu_percent)
        self.gpu.observe(gpu_memory_percent)

    def _read(self, reader: Callable[[], float | None] | None) -> float | None:
        if reader is None:
            return None
        try:
            value = reader()
            return None if value is None else float(value)
        except Exception:
            return None

    def tick(self) -> None:
        self.memory_sample_fresh = False
        reading = self.reader()
        self.memory_sample_fresh = reading is not None
        if reading is not None:
            self.observe(*reading)
        elif self.recovering is False:
            self._apply()
        self.observe_load(self._read(self.cpu_reader), self._read(self.gpu_reader))
        for listener in list(self._listeners):
            try:
                listener(self)
            except Exception as exc:
                self._log(f"RESOURCE: listener failed - {exc}")

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


__all__ = ["GOVERNOR", "LoadGauge", "Pressure", "ResourceGovernor", "Shedder",
           "optional_work_allowed", "release_python_memory"]
