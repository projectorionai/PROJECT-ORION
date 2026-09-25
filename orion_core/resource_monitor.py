"""
Non-destructive resource-pressure monitor (Section 8).

When another application starts consuming heavy CPU, RAM, disk or GPU, ORION
must NOT abandon or interrupt its active task.  This subsystem watches host and
own-process load and produces graduated, non-destructive recommendations:

* Momentary spikes are ignored — a level must hold above its threshold for a
  configurable duration window before it engages.
* System-wide load and ORION's own per-process load are tracked separately.
* Physical memory (RAM), swap and ORION's working set are DISTINCT metrics and
  are only ever described as separate when their values are actually present —
  the module never conflates "RAM" and "memory".
* It NEVER terminates, suspends, throttles, closes or modifies another process.
  There is deliberately no method to do so; it only recommends.
* High-resource applications are identified only from information already
  available to the current user, and command-line arguments are never exposed.

The evaluator is pure and deterministic (readings are injected), so behaviour is
tested with simulated load.  A psutil-backed reader supplies real readings at
runtime and degrades gracefully when psutil or a metric is unavailable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

try:
    from .constants import (
        RESOURCE_CRITICAL_CPU,
        RESOURCE_CRITICAL_HOLD_S,
        RESOURCE_CRITICAL_MEM,
        RESOURCE_ELEVATED_CPU,
        RESOURCE_ELEVATED_HOLD_S,
        RESOURCE_ELEVATED_MEM,
    )
except Exception:  # pragma: no cover
    RESOURCE_ELEVATED_CPU, RESOURCE_CRITICAL_CPU = 85.0, 95.0
    RESOURCE_ELEVATED_MEM, RESOURCE_CRITICAL_MEM = 85.0, 95.0
    RESOURCE_ELEVATED_HOLD_S, RESOURCE_CRITICAL_HOLD_S = 20.0, 10.0


class PressureLevel(str, Enum):
    NOMINAL = "nominal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    INSTABILITY_RISK = "instability_risk"


class RecommendedAction(str, Enum):
    CONTINUE = "continue"                          # nominal — nothing to do
    CONTINUE_AND_NOTE = "continue_and_note"        # elevated but tolerable
    CHECKPOINT_AND_REDUCE = "checkpoint_and_reduce"  # critical — keep task, shed load
    CHECKPOINT_AND_ASK = "checkpoint_and_ask"      # OS at risk — request direction


# Background activities that may be shed under pressure without harming the task.
SHEDDABLE_ACTIVITY = (
    "optional file indexing",
    "self-improvement ticks",
    "provider health probes",
    "avatar animations",
)


@dataclass
class ResourceThresholds:
    elevated_cpu: float = RESOURCE_ELEVATED_CPU
    critical_cpu: float = RESOURCE_CRITICAL_CPU
    elevated_mem: float = RESOURCE_ELEVATED_MEM
    critical_mem: float = RESOURCE_CRITICAL_MEM
    elevated_hold_s: float = RESOURCE_ELEVATED_HOLD_S
    critical_hold_s: float = RESOURCE_CRITICAL_HOLD_S
    # Physical memory near-full AND swap heavily used → genuine thrash/OOM risk.
    instability_swap: float = 90.0


@dataclass
class ResourceReading:
    """One sample.  ``None`` means 'not reported by this host' — never guessed."""

    cpu_percent: float                     # system-wide CPU utilisation
    mem_percent: float                     # physical memory (RAM) used %
    swap_percent: float | None = None      # swap/page-file used %
    disk_busy_percent: float | None = None
    gpu_percent: float | None = None
    gpu_mem_percent: float | None = None
    self_cpu_percent: float | None = None  # ORION's own process CPU
    self_rss_mb: float | None = None       # ORION's working set (resident set)
    timestamp: float = 0.0


@dataclass
class ProcessInfo:
    """A high-resource process, sanitised — no command-line arguments."""

    pid: int
    name: str
    cpu_percent: float = 0.0
    mem_percent: float = 0.0

    def redacted(self) -> dict[str, Any]:
        return {"pid": self.pid, "name": self.name,
                "cpu_percent": round(self.cpu_percent, 1),
                "mem_percent": round(self.mem_percent, 1)}


@dataclass
class PressureAssessment:
    level: PressureLevel
    action: RecommendedAction
    engaged: bool                          # has the level held past its window?
    affected_resources: list[str] = field(default_factory=list)
    offender: dict[str, Any] | None = None
    shed: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def preserve_task(self) -> bool:
        """The active task is preserved for every level except when the OS is at
        immediate risk and the user must decide."""
        return self.action is not RecommendedAction.CHECKPOINT_AND_ASK

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "action": self.action.value,
            "engaged": self.engaged,
            "affected_resources": list(self.affected_resources),
            "offender": self.offender,
            "shed": list(self.shed),
            "preserve_task": self.preserve_task,
            "message": self.message,
        }


def identify_offenders(processes: list[ProcessInfo], resource: str,
                       limit: int = 1) -> list[dict[str, Any]]:
    """Top processes by ``resource`` ('cpu' or 'mem'), redacted.  Uses only the
    name/pid/utilisation already available to the current user."""
    key = (lambda p: p.cpu_percent) if resource == "cpu" else (lambda p: p.mem_percent)
    ranked = sorted(processes, key=key, reverse=True)
    return [p.redacted() for p in ranked[:max(1, limit)]]


class ResourceForecaster:
    """Predictive resource modelling (improvement #29).

    Keeps a sliding window of readings per metric and fits a least-squares
    trend line; when that trend crosses the critical threshold within the
    warning horizon, a preventative notice fires BEFORE any pressure level
    engages — anticipation instead of reaction.  Pure and deterministic
    (readings injected), like the monitor it feeds."""

    WINDOW = 90                 # readings kept per metric
    MIN_POINTS = 12             # below this a trend is noise, not a signal
    HORIZON_S = 45 * 60.0       # warn when exhaustion is inside this window
    REWARN_S = 30 * 60.0        # per-metric quiet period between warnings

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._points: dict[str, deque[tuple[float, float]]] = {}
        self._last_warned: dict[str, float] = {}

    def observe(self, metric: str, value: float | None, ts: float | None = None) -> None:
        if value is None:
            return
        bucket = self._points.setdefault(metric, deque(maxlen=self.WINDOW))
        bucket.append((ts if ts is not None else self._clock(), float(value)))

    def seconds_until(self, metric: str, threshold: float) -> float | None:
        """Projected seconds until the metric's trend crosses ``threshold``;
        None when there is no meaningful upward trend (or too few points)."""
        points = self._points.get(metric)
        if points is None or len(points) < self.MIN_POINTS:
            return None
        t0 = points[0][0]
        xs = [t - t0 for t, _ in points]
        ys = [v for _, v in points]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        variance = sum((x - mean_x) ** 2 for x in xs)
        if variance <= 0:
            return None
        slope = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys)) / variance      # %-points per second
        if slope <= 1e-6:
            return None                                       # flat or improving
        current = ys[-1]
        if current >= threshold:
            return 0.0
        return (threshold - current) / slope

    def warnings(self, thresholds: dict[str, float]) -> list[dict[str, Any]]:
        """Preventative warnings for every metric whose projected breach falls
        inside the horizon — rate-limited per metric so a slow climb warns
        once, not every sample."""
        now = self._clock()
        out: list[dict[str, Any]] = []
        for metric, threshold in thresholds.items():
            eta = self.seconds_until(metric, threshold)
            if eta is None or eta > self.HORIZON_S:
                continue
            if now - self._last_warned.get(metric, float("-inf")) < self.REWARN_S:
                continue
            self._last_warned[metric] = now
            out.append({"metric": metric, "eta_minutes": max(0, round(eta / 60.0)),
                        "threshold": threshold})
        return out


class ResourceMonitor:
    """Evaluates readings into non-destructive pressure recommendations."""

    def __init__(self, thresholds: ResourceThresholds | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.thresholds = thresholds or ResourceThresholds()
        self._clock = clock or time.monotonic
        self._elevated_since: float | None = None
        self._critical_since: float | None = None
        self._instability_since: float | None = None
        # Events recorded to notify the user AFTER the active task finishes.
        self._pending_notifications: list[dict[str, Any]] = []
        self._last_engaged = PressureLevel.NOMINAL
        # Improvement #29: trend-based early warning ahead of any breach.
        self.forecaster = ResourceForecaster(clock=self._clock)

    # ── evaluation ────────────────────────────────────────────────────────────

    def _breached(self, reading: ResourceReading) -> tuple[bool, bool, bool, list[str]]:
        t = self.thresholds
        affected: list[str] = []
        cpu_elev = reading.cpu_percent >= t.elevated_cpu
        cpu_crit = reading.cpu_percent >= t.critical_cpu
        mem_elev = reading.mem_percent >= t.elevated_mem
        mem_crit = reading.mem_percent >= t.critical_mem
        if reading.cpu_percent >= t.elevated_cpu:
            affected.append("CPU")
        if reading.mem_percent >= t.elevated_mem:
            affected.append("physical memory (RAM)")
        swap = reading.swap_percent
        if swap is not None and swap >= t.instability_swap:
            affected.append("swap")
        elevated = cpu_elev or mem_elev
        critical = cpu_crit or mem_crit
        instability = mem_crit and swap is not None and swap >= t.instability_swap
        return elevated, critical, instability, affected

    def observe(self, reading: ResourceReading,
                processes: list[ProcessInfo] | None = None) -> PressureAssessment:
        now = reading.timestamp or self._clock()
        elevated, critical, instability, affected = self._breached(reading)

        # Predictive early warning (improvement #29): feed the trend model and
        # queue a preventative note when exhaustion is projected — this rides
        # the existing after-task notification path and never alters levels.
        self.forecaster.observe("cpu", reading.cpu_percent, ts=now)
        self.forecaster.observe("ram", reading.mem_percent, ts=now)
        self.forecaster.observe("swap", reading.swap_percent, ts=now)
        for warning in self.forecaster.warnings({
            "cpu": self.thresholds.critical_cpu,
            "ram": self.thresholds.critical_mem,
            "swap": self.thresholds.instability_swap,
        }):
            self._pending_notifications.append({
                "level": "predictive",
                "affected": [warning["metric"]],
                "offender": None,
                "at": now,
                "message": (
                    f"Heads-up: {warning['metric'].upper()} is trending towards its "
                    f"critical threshold ({warning['threshold']:.0f}%) in roughly "
                    f"{warning['eta_minutes']} minute(s). Consider closing heavy "
                    "applications or letting me shed background work pre-emptively."),
            })

        # Track how long each level has held continuously.
        self._elevated_since = now if elevated and self._elevated_since is None else \
            (None if not elevated else self._elevated_since)
        self._critical_since = now if critical and self._critical_since is None else \
            (None if not critical else self._critical_since)
        self._instability_since = now if instability and self._instability_since is None else \
            (None if not instability else self._instability_since)

        t = self.thresholds
        held_elevated = (self._elevated_since is not None
                         and now - self._elevated_since >= t.elevated_hold_s)
        held_critical = (self._critical_since is not None
                         and now - self._critical_since >= t.critical_hold_s)
        held_instability = (self._instability_since is not None
                            and now - self._instability_since >= t.critical_hold_s)

        # Highest engaged level wins.
        if held_instability:
            return self._make(PressureLevel.INSTABILITY_RISK, True, affected, reading, processes)
        if held_critical:
            return self._make(PressureLevel.CRITICAL, True, affected, reading, processes)
        if held_elevated:
            return self._make(PressureLevel.ELEVATED, True, affected, reading, processes)
        # A breach is happening but has not yet held long enough — a momentary
        # spike.  Report the raw level but mark it NOT engaged: the task runs on.
        raw = (PressureLevel.INSTABILITY_RISK if instability else
               PressureLevel.CRITICAL if critical else
               PressureLevel.ELEVATED if elevated else PressureLevel.NOMINAL)
        return self._make(raw, False, affected, reading, processes)

    def _make(self, level: PressureLevel, engaged: bool, affected: list[str],
              reading: ResourceReading, processes: list[ProcessInfo] | None) -> PressureAssessment:
        offender = None
        if processes and affected:
            resource = "cpu" if "CPU" in affected else "mem"
            top = identify_offenders(processes, resource, limit=1)
            offender = top[0] if top else None

        if not engaged or level is PressureLevel.NOMINAL:
            action = RecommendedAction.CONTINUE
            msg = "Resource use is within normal bounds; continuing the task."
        elif level is PressureLevel.ELEVATED:
            action = RecommendedAction.CONTINUE_AND_NOTE
            who = f" ({offender['name']})" if offender else ""
            msg = (f"Resource pressure is elevated on {', '.join(affected) or 'the system'}{who}. "
                   "Continuing the current task; I'll note it and mention it once the task finishes.")
            self._record_notification(level, affected, offender)
        elif level is PressureLevel.CRITICAL:
            action = RecommendedAction.CHECKPOINT_AND_REDUCE
            who = f" ({offender['name']})" if offender else ""
            msg = (f"Resource pressure is critical on {', '.join(affected) or 'the system'}{who}. "
                   "Checkpointing task state and shedding non-essential background work; "
                   "the active task keeps running.")
        else:  # INSTABILITY_RISK
            action = RecommendedAction.CHECKPOINT_AND_ASK
            msg = ("The operating system is at immediate risk of instability "
                   "(physical memory near full with heavy swap use). I've checkpointed "
                   "task state — how would you like me to proceed?")

        shed = list(SHEDDABLE_ACTIVITY) if action in (
            RecommendedAction.CHECKPOINT_AND_REDUCE, RecommendedAction.CHECKPOINT_AND_ASK) else []
        self._last_engaged = level if engaged else self._last_engaged
        return PressureAssessment(level, action, engaged, affected, offender, shed, msg)

    # ── after-task notifications ──────────────────────────────────────────────

    def _record_notification(self, level: PressureLevel, affected: list[str],
                             offender: dict[str, Any] | None) -> None:
        self._pending_notifications.append({
            "level": level.value,
            "affected": list(affected),
            "offender": offender,
            "at": self._clock(),
        })

    def drain_notifications(self) -> list[dict[str, Any]]:
        """Return and clear queued 'notify after the task' events."""
        items = list(self._pending_notifications)
        self._pending_notifications.clear()
        return items


class SystemResourceReader:
    """psutil-backed real reading source; every metric degrades to ``None`` when
    unavailable rather than fabricating a value."""

    def __init__(self) -> None:
        self._psutil = None
        self._proc = None
        try:
            import psutil  # type: ignore
            self._psutil = psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(None)          # prime the first delta
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None

    def available(self) -> bool:
        return self._psutil is not None

    def read(self) -> ResourceReading | None:
        ps = self._psutil
        if ps is None:
            return None
        try:
            vm = ps.virtual_memory()
            try:
                swap = ps.swap_memory().percent
            except Exception:
                swap = None
            self_cpu = self_rss = None
            if self._proc is not None:
                try:
                    self_cpu = self._proc.cpu_percent(None)
                    self_rss = self._proc.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
            return ResourceReading(
                cpu_percent=float(ps.cpu_percent(None)),
                mem_percent=float(vm.percent),
                swap_percent=None if swap is None else float(swap),
                self_cpu_percent=self_cpu,
                self_rss_mb=self_rss,
                timestamp=time.monotonic(),
            )
        except Exception:
            return None

    def top_processes(self, limit: int = 5) -> list[ProcessInfo]:
        """High-resource processes visible to the current user, sanitised.
        Never reads command-line arguments or unrelated process detail."""
        ps = self._psutil
        if ps is None:
            return []
        out: list[ProcessInfo] = []
        for proc in ps.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                out.append(ProcessInfo(
                    pid=int(info.get("pid") or 0),
                    name=str(info.get("name") or "?"),
                    cpu_percent=float(info.get("cpu_percent") or 0.0),
                    mem_percent=float(info.get("memory_percent") or 0.0),
                ))
            except Exception:
                continue
        out.sort(key=lambda p: p.cpu_percent + p.mem_percent, reverse=True)
        return out[:max(1, limit)]
