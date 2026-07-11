"""
Telemetry core (Phase 12) — structured logging, metrics and health.

The single observability seam for Mark IX.  Three co-operating pieces:

    StructuredLogger  — level + component + message + fields, emitted both to a
                        ring buffer (for the Command Centre) and, in a
                        human-readable form, onto ``bus.log`` so the existing
                        console keeps working unchanged.

    MetricsRegistry   — thread-safe counters, gauges and timers.  Everything
                        that matters (audio queue depth, playback latency,
                        tool durations, agent health) is published here and
                        sampled by the dashboard at its own cadence, so hot
                        paths never touch a widget.

    HealthRegistry    — components register a name and heartbeat/status; the
                        Command Centre renders green/amber/red without any
                        component importing the GUI.

Design rules: never blocks (locks are held only for O(1) dict ops), never
raises into a caller (telemetry must not break a live turn), and holds no
reference to Qt beyond the bus signal it emits on.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Deque

from .bus import OrionBus
from .utils import utc_stamp


class Level(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogRecord:
    at: str
    level: str
    component: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)

    def console_line(self) -> str:
        tail = ""
        if self.fields:
            tail = " " + " ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"{self.component}: {self.message}{tail}"


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

class MetricsRegistry:
    """Thread-safe counters, gauges and rolling timers."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, Deque[float]] = {}

    def incr(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, millis: float) -> None:
        with self._lock:
            bucket = self._timers.get(name)
            if bucket is None:
                bucket = deque(maxlen=256)
                self._timers[name] = bucket
            bucket.append(float(millis))

    def timer(self, name: str) -> "_Timer":
        return _Timer(self, name)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timer_stats: dict[str, dict[str, float]] = {}
            for name, bucket in self._timers.items():
                if not bucket:
                    continue
                values = sorted(bucket)
                count = len(values)
                timer_stats[name] = {
                    "count": float(count),
                    "avg_ms": round(sum(values) / count, 2),
                    "p50_ms": round(values[count // 2], 2),
                    "p95_ms": round(values[min(count - 1, int(count * 0.95))], 2),
                    "max_ms": round(values[-1], 2),
                }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": timer_stats,
            }


class _Timer:
    """Context manager: ``with metrics.timer('tool.dispatch'):``."""

    __slots__ = ("_registry", "_name", "_start")

    def __init__(self, registry: MetricsRegistry, name: str) -> None:
        self._registry = registry
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._registry.observe(self._name, (time.perf_counter() - self._start) * 1000.0)


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH
# ──────────────────────────────────────────────────────────────────────────────

class Health(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class ComponentHealth:
    name: str
    status: str = Health.UNKNOWN.value
    detail: str = ""
    last_beat: float = 0.0
    # Event-driven services (exporter, geo, emotion, identity…) only "beat" when
    # used, so staleness is not a fault — mark them idle-OK.  Only true periodic
    # workers should be flagged when they miss a heartbeat: they set ttl > 0.
    ttl: float = 0.0

    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.last_beat) if self.last_beat else -1.0


class HealthRegistry:
    """Component → status.  Only heartbeat-monitored components go stale."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._components: dict[str, ComponentHealth] = {}

    def register(self, name: str) -> None:
        with self._lock:
            self._components.setdefault(name, ComponentHealth(name=name))

    def beat(self, name: str, status: str = Health.OK.value, detail: str = "",
             ttl: float = 0.0) -> None:
        """Record a heartbeat.  ``ttl`` is how long (seconds) this beat stays
        fresh before the component is considered stale; 0 (the default) means
        the component is event-driven and never auto-degrades on idleness."""
        with self._lock:
            comp = self._components.get(name)
            if comp is None:
                comp = ComponentHealth(name=name)
                self._components[name] = comp
            comp.status = status
            comp.detail = detail
            comp.last_beat = time.monotonic()
            if ttl > 0.0:
                comp.ttl = ttl

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for comp in self._components.values():
                status = comp.status
                age = comp.age_s()
                # Only a periodic worker (ttl set) that overran its interval by
                # a wide margin is degraded — idle on-demand services stay OK.
                if status == Health.OK.value and comp.ttl > 0.0 and age > comp.ttl * 2.5:
                    status = Health.DEGRADED.value
                rows.append({
                    "name": comp.name,
                    "status": status,
                    "detail": comp.detail,
                    "age_s": round(age, 1) if age >= 0 else None,
                })
            return sorted(rows, key=lambda r: r["name"])


# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED LOGGER
# ──────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    """Ring-buffered structured logging that also mirrors to ``bus.log``."""

    def __init__(self, bus: OrionBus, capacity: int = 1500) -> None:
        self.bus = bus
        self._lock = RLock()
        self._records: Deque[LogRecord] = deque(maxlen=capacity)

    def log(self, level: Level, component: str, message: str, **fields: Any) -> None:
        record = LogRecord(
            at=utc_stamp(), level=level.value, component=component,
            message=str(message), fields=fields,
        )
        with self._lock:
            self._records.append(record)
        try:
            # Mirror to the legacy console; keep the original "COMP: msg" shape.
            self.bus.log.emit(record.console_line())
        except RuntimeError:
            pass  # Qt shutting down

    def debug(self, component: str, message: str, **f: Any) -> None:
        self.log(Level.DEBUG, component, message, **f)

    def info(self, component: str, message: str, **f: Any) -> None:
        self.log(Level.INFO, component, message, **f)

    def warn(self, component: str, message: str, **f: Any) -> None:
        self.log(Level.WARN, component, message, **f)

    def error(self, component: str, message: str, **f: Any) -> None:
        self.log(Level.ERROR, component, message, **f)

    def recent(self, limit: int = 200, min_level: Level | None = None) -> list[dict[str, Any]]:
        order = {l.value: i for i, l in enumerate(Level)}
        floor = order.get(min_level.value, 0) if min_level else 0
        with self._lock:
            records = list(self._records)
        out = [
            {"at": r.at, "level": r.level, "component": r.component,
             "message": r.message, "fields": r.fields}
            for r in records
            if order.get(r.level, 0) >= floor
        ]
        return out[-limit:]


# ──────────────────────────────────────────────────────────────────────────────
# FACADE
# ──────────────────────────────────────────────────────────────────────────────

class Telemetry:
    """Single object handed to every subsystem: log, metrics, health together."""

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self.log = StructuredLogger(bus)
        self.metrics = MetricsRegistry()
        self.health = HealthRegistry()

    def snapshot(self) -> dict[str, Any]:
        """Full observability picture for the Command Centre."""
        return {
            "at": utc_stamp(),
            "metrics": self.metrics.snapshot(),
            "health": self.health.snapshot(),
            "logs": self.log.recent(limit=120),
        }
