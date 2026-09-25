"""
telemetry_bind.py — put ORION's real measurements onto swarm nodes.

The audit's most frustrating finding: every number the brief asks each node to
display was ALREADY being collected, and thrown away. The dispatcher has
recorded ``tool.<name>.ms`` on every call since Mark X, and
``record_tool_call`` maintains ``tool.<name>.calls`` / ``.failures``;
MetricsRegistry.snapshot() already computes p50/p95/max/avg per bucket, and
HealthRegistry.snapshot() already carries status, detail and heartbeat age.
The old compose_snapshot() read none of it, so nodes showed no latency, no
throughput and no error rate.

The other reason this module exists is cost. MetricsRegistry.snapshot() sorts
every rolling bucket to compute percentiles — 1.32 ms across 120 tools,
measured. Calling it per node would multiply that by the node count on every
refresh. TelemetryIndex takes both snapshots exactly ONCE and turns them into
O(1) lookups, so binding a 1,500-node graph costs the same two snapshot calls
as binding a 50-node one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .model import EMPTY_TELEMETRY, Activity, Health, NodeTelemetry, SwarmNode

# An error rate above this reads as a genuine fault rather than noise, and
# flips the node's activity to ERROR so it is visible without inspection.
_ERROR_RATE_ALARM = 0.25
# Below this many calls, an error rate is statistically meaningless — one
# failure out of two calls is not a degraded subsystem.
_MIN_CALLS_FOR_ALARM = 4


def _health_from(status: str) -> Health:
    try:
        return Health(str(status or "").strip().upper())
    except ValueError:
        return Health.UNKNOWN


class TelemetryIndex:
    """A single point-in-time read of ORION's metrics, indexed for O(1) reuse.

    Construct one per refresh, then bind as many nodes as needed against it.
    Every accessor is total — an unknown name returns empty telemetry rather
    than raising, because a subsystem that has never been called is a normal
    state, not an error.
    """

    __slots__ = ("_counters", "_gauges", "_timers", "_health", "_ok")

    def __init__(self, telemetry: Any | None = None) -> None:
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, dict[str, float]] = {}
        self._health: dict[str, dict[str, Any]] = {}
        # False when telemetry was absent or unreadable; callers surface this
        # as a degraded source rather than silently showing zeroes as truth.
        self._ok = False
        if telemetry is None:
            return
        try:
            metrics = telemetry.metrics.snapshot()
            self._counters = dict(metrics.get("counters") or {})
            self._gauges = dict(metrics.get("gauges") or {})
            self._timers = dict(metrics.get("timers") or {})
            self._ok = True
        except Exception:
            pass
        try:
            self._health = {
                str(row.get("name")): row
                for row in (telemetry.health.snapshot() or [])
            }
            self._ok = True
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._ok

    def counter(self, name: str, default: float = 0.0) -> float:
        return float(self._counters.get(name, default))

    def gauge(self, name: str) -> float | None:
        value = self._gauges.get(name)
        return None if value is None else float(value)

    def health_of(self, name: str) -> Health:
        row = self._health.get(str(name or ""))
        return _health_from(row.get("status")) if row else Health.UNKNOWN

    def heartbeat_age(self, name: str) -> float | None:
        row = self._health.get(str(name or ""))
        if not row:
            return None
        age = row.get("age_s")
        return None if age is None else float(age)

    def health_detail(self, name: str) -> str:
        row = self._health.get(str(name or ""))
        return str(row.get("detail") or "") if row else ""

    def for_tool(self, tool: str) -> NodeTelemetry:
        """Telemetry for a dispatcher tool, from the counters and timer bucket
        the dispatcher already writes on every call."""
        tool = str(tool or "").strip()
        if not tool:
            return EMPTY_TELEMETRY
        timings = self._timers.get(f"tool.{tool}.ms") or {}
        return NodeTelemetry(
            calls=int(self.counter(f"tool.{tool}.calls")),
            failures=int(self.counter(f"tool.{tool}.failures")),
            latency_p50_ms=timings.get("p50_ms"),
            latency_p95_ms=timings.get("p95_ms"),
            latency_max_ms=timings.get("max_ms"),
            latency_avg_ms=timings.get("avg_ms"),
        )

    def for_component(self, component: str) -> NodeTelemetry:
        """Telemetry for a registered subsystem — heartbeat age and whatever
        the component reported as its current detail string."""
        return NodeTelemetry(
            heartbeat_age_s=self.heartbeat_age(component),
            current_action=self.health_detail(component),
        )


def activity_for(telemetry: NodeTelemetry, health: Health) -> Activity:
    """Derive what a node is visibly DOING from its measurements.

    Deliberately conservative: a node only reads as ERROR on a meaningful
    sample (see _MIN_CALLS_FOR_ALARM), so a single early failure never paints
    a healthy subsystem red — the old graph had no activity concept at all,
    and over-alarming would be a worse replacement than none.
    """
    if health is Health.DOWN:
        return Activity.OFFLINE
    if (telemetry.calls >= _MIN_CALLS_FOR_ALARM
            and telemetry.error_rate >= _ERROR_RATE_ALARM):
        return Activity.ERROR
    if telemetry.current_action:
        return Activity.BUSY
    if telemetry.queue_depth:
        return Activity.ACTIVE
    if telemetry.is_quiet:
        return Activity.IDLE
    return Activity.ACTIVE


def bind_node(node: SwarmNode, telemetry: NodeTelemetry,
              health: Health | None = None) -> SwarmNode:
    """Return *node* carrying *telemetry*, with activity (and optionally
    health) derived from it. Immutable — produces a new node."""
    resolved_health = health if health is not None else node.health
    from dataclasses import replace
    return replace(
        node,
        telemetry=telemetry,
        health=resolved_health,
        activity=activity_for(telemetry, resolved_health),
    )
