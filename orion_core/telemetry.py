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

import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Deque, Optional

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

    # Per-tool counters follow ``tool.<name>.calls`` / ``tool.<name>.failures``;
    # this pattern separates them from the aggregate ``tool.calls`` counter so
    # both the usage audit and the history sampler can tell them apart.
    _PER_TOOL_RE = re.compile(r"^tool\.(?P<tool>[^.]+)\.(?P<kind>calls|failures)$")

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self.log = StructuredLogger(bus)
        self.metrics = MetricsRegistry()
        self.health = HealthRegistry()
        # Optional rolling on-disk history (Priority 3.1); enabled by the app.
        self.history: Optional[Any] = None

    def snapshot(self) -> dict[str, Any]:
        """Full observability picture for the Command Centre."""
        return {
            "at": utc_stamp(),
            "metrics": self.metrics.snapshot(),
            "health": self.health.snapshot(),
            "logs": self.log.recent(limit=120),
        }

    # ── per-tool usage audit (Priority 3.4) ────────────────────────────────────

    def record_tool_call(self, name: str, ok: bool) -> None:
        """One dispatcher invocation → aggregate + per-tool counters."""
        self.metrics.incr("tool.calls")
        self.metrics.incr(f"tool.{name}.calls")
        if not ok:
            self.metrics.incr("tool.failures")
            self.metrics.incr(f"tool.{name}.failures")

    def tool_usage(self, top: int | None = None) -> list[dict[str, Any]]:
        """Which tools actually get invoked, busiest first — the feature-audit
        view. Derived from the per-tool counters (Priority 3.4)."""
        counters = self.metrics.snapshot()["counters"]
        usage: dict[str, dict[str, int]] = {}
        for key, value in counters.items():
            m = self._PER_TOOL_RE.match(key)
            if not m:
                continue
            row = usage.setdefault(m["tool"], {"tool": m["tool"], "calls": 0, "failures": 0})
            row[m["kind"]] = int(value)
        ranked = sorted(usage.values(), key=lambda r: r["calls"], reverse=True)
        return ranked[:top] if top else ranked

    # ── rolling history (Priority 3.1) ─────────────────────────────────────────

    def enable_history(self, db_path: "Path | str", **kwargs: Any) -> Optional[Any]:
        """Attach a MetricsHistory store so trends persist over days/weeks."""
        try:
            from .metrics_history import MetricsHistory
            self.history = MetricsHistory(Path(db_path), **kwargs)
            self.health.register("metrics_history")
            self.health.beat("metrics_history", Health.OK.value, "rolling store attached")
        except Exception as exc:                        # degrade gracefully
            self.history = None
            try:
                self.log.warn("TELEMETRY", f"metrics history unavailable - {exc}")
            except Exception:
                pass
        return self.history

    def _history_series(self) -> dict[str, float]:
        """Curated numeric series worth trending: aggregate counters, all
        gauges and timer p95s. Per-tool counters are excluded here (they are
        persisted separately as cumulative usage)."""
        snap = self.metrics.snapshot()
        series: dict[str, float] = {}
        for name, value in snap["counters"].items():
            if self._PER_TOOL_RE.match(name):
                continue                                # captured via tool_usage
            series[f"c.{name}"] = float(value)
        for name, value in snap["gauges"].items():
            series[f"g.{name}"] = float(value)
        for name, stats in snap["timers"].items():
            series[f"t.{name}.p95"] = float(stats.get("p95_ms", 0.0))
        return series

    def tick_history(self) -> bool:
        """Sample current metrics into the rolling store (throttled internally).
        Safe to call from any periodic tick; a no-op when history is disabled."""
        if self.history is None:
            return False
        try:
            wrote = self.history.sample(self._history_series())
            if wrote:
                usage = {r["tool"]: r for r in self.tool_usage()}
                if usage:
                    self.history.record_tool_usage(usage)
            return wrote
        except Exception:
            return False
