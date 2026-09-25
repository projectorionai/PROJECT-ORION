"""
inspect.py — what the inspector shows about a node (Mark XXII, Phase 4).

The brief lists a long set of fields a selected node should display: name,
description, subsystem, health, CPU, latency, token usage, connected nodes,
queued requests, capabilities, dependencies, live telemetry, recent actions.
This module turns a SwarmNode plus its graph context into exactly that, as
ordered (label, value) rows.

It lives here rather than in the Qt widget for two reasons. It is pure, so
every formatting rule — how an absent measurement renders, how an error rate
is worded, how neighbours are grouped — is unit-testable without a GUI. And
the same rows can then feed the GL inspector, the WebEngine fallback list and
the phone/remote surface without three implementations drifting apart.

One formatting rule is load-bearing: a measurement that does not exist renders
as an em dash, never as 0. "0 ms latency" and "this node has never reported
latency" are completely different facts, and a monitoring surface that
conflates them is actively misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .model import Activity, Health, NodeKind, SwarmEdge, SwarmNode, SwarmSnapshot

ABSENT = "—"

_KIND_LABELS: dict[NodeKind, str] = {
    NodeKind.CORE: "Core",
    NodeKind.AGENT: "Specialist agent",
    NodeKind.SUBSYSTEM: "Subsystem",
    NodeKind.MCP_SERVER: "MCP server",
    NodeKind.MCP_TOOL: "MCP tool",
    NodeKind.WORKFLOW: "Workflow",
    NodeKind.MEMORY_TIER: "Memory tier",
    NodeKind.MODEL: "Model",
    NodeKind.TOOL: "Tool",
    NodeKind.PAGE: "Command Deck page",
}

_ACTIVITY_LABELS: dict[Activity, str] = {
    Activity.IDLE: "Idle",
    Activity.ACTIVE: "Active",
    Activity.BUSY: "Busy",
    Activity.THINKING: "Thinking",
    Activity.ERROR: "Failing",
    Activity.OFFLINE: "Offline",
}


def _ms(value: float | None) -> str:
    if value is None:
        return ABSENT
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.1f} ms"


def _count(value: int | None) -> str:
    return ABSENT if value is None else f"{value:,}"


def _seconds(value: float | None) -> str:
    if value is None or value < 0:
        return ABSENT
    if value < 60:
        return f"{value:.0f}s ago"
    if value < 3600:
        return f"{value / 60:.0f}m ago"
    return f"{value / 3600:.1f}h ago"


@dataclass(frozen=True, slots=True)
class InspectorRow:
    label: str
    value: str
    # "alarm" when this row is the reason a node needs attention, so the UI
    # can emphasise it without re-deriving the judgement.
    severity: str = "normal"


@dataclass(frozen=True, slots=True)
class InspectorReport:
    title: str
    subtitle: str
    rows: tuple[InspectorRow, ...]
    neighbours: tuple[str, ...]
    action_label: str
    action_target: str | None

    def as_text(self) -> str:
        """Plain-text rendering — used by the list fallback and by anything
        that wants to log or speak the report."""
        lines = [f"{self.title} · {self.subtitle}"]
        lines += [f"{row.label}: {row.value}" for row in self.rows]
        if self.neighbours:
            lines.append("Connected: " + ", ".join(self.neighbours))
        return "\n".join(lines)


def _neighbours(node_id: str, edges: Iterable[SwarmEdge]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        other = None
        if edge.source == node_id:
            other = edge.target
        elif edge.target == node_id:
            other = edge.source
        if other and other not in seen:
            seen.add(other)
            out.append(other)
    return tuple(out)


def _action_for(node: SwarmNode, can_open_agent: bool) -> tuple[str, str | None]:
    # A cluster anchor is the swarm's doorway into the Command Deck: opening
    # it navigates to that zone. This is what makes the graph a navigation
    # surface rather than a picture of one — every zone is reachable by
    # clicking the thing that represents it.
    if node.is_cluster:
        return (f"Open {node.cluster_zone.title()} zone", node.cluster_zone)
    if node.kind is NodeKind.PAGE:
        return (f"Open {node.label}", node.label)
    if node.kind is NodeKind.AGENT and can_open_agent:
        return ("Open workspace", node.id.split(":", 1)[-1])
    if node.kind is NodeKind.WORKFLOW:
        return ("Run workflow", node.label)
    if node.kind is NodeKind.MCP_SERVER:
        return ("Reconnect server", node.label)
    return ("No action available", None)


def build_report(
    node: SwarmNode,
    snapshot: SwarmSnapshot | None = None,
    can_open_agent: bool = False,
) -> InspectorReport:
    """Everything known about *node*, ready to render."""
    telemetry = node.telemetry
    rows: list[InspectorRow] = []

    health_severity = "alarm" if node.health in (Health.DEGRADED, Health.DOWN) else "normal"
    rows.append(InspectorRow("Health", node.health.value, health_severity))
    rows.append(InspectorRow(
        "Activity", _ACTIVITY_LABELS.get(node.activity, node.activity.value),
        "alarm" if node.activity is Activity.ERROR else "normal"))
    rows.append(InspectorRow("Cluster", node.cluster))

    if node.detail:
        rows.append(InspectorRow("Role", node.detail))
    if telemetry.current_action:
        rows.append(InspectorRow("Current action", telemetry.current_action))

    rows.append(InspectorRow("Requests", _count(telemetry.calls)))
    if telemetry.calls:
        rate = telemetry.error_rate
        rows.append(InspectorRow(
            "Failures", f"{telemetry.failures:,}  ({rate * 100:.0f}%)",
            "alarm" if rate >= 0.25 else "normal"))
    else:
        # Distinguish "never called" from "called and never failed".
        rows.append(InspectorRow("Failures", ABSENT))

    rows.append(InspectorRow("Latency p50", _ms(telemetry.latency_p50_ms)))
    rows.append(InspectorRow("Latency p95", _ms(telemetry.latency_p95_ms)))
    rows.append(InspectorRow("Latency max", _ms(telemetry.latency_max_ms)))
    rows.append(InspectorRow("Queue", _count(telemetry.queue_depth)))
    rows.append(InspectorRow("Tokens", _count(telemetry.tokens)))
    rows.append(InspectorRow("Last heartbeat", _seconds(telemetry.heartbeat_age_s)))

    if node.dependencies:
        rows.append(InspectorRow("Depends on", ", ".join(node.dependencies)))
    if node.capabilities:
        shown = ", ".join(node.capabilities[:8])
        if len(node.capabilities) > 8:
            shown += f"  (+{len(node.capabilities) - 8} more)"
        rows.append(InspectorRow("Capabilities", shown))

    neighbours = _neighbours(node.id, snapshot.edges) if snapshot else ()
    action_label, action_target = _action_for(node, can_open_agent)

    return InspectorReport(
        title=node.label,
        subtitle=_KIND_LABELS.get(node.kind, node.kind.value),
        rows=tuple(rows),
        neighbours=neighbours,
        action_label=action_label,
        action_target=action_target,
    )


def report_from_dict(payload: dict[str, Any],
                     can_open_agent: bool = False) -> InspectorReport:
    """Build a report from the older dict node shape.

    The WebEngine path still emits dicts through its console bridge, so this
    keeps both surfaces on one formatting implementation instead of letting
    the legacy path grow its own."""
    from .model import NodeTelemetry

    raw_telemetry = payload.get("telemetry") or {}
    telemetry = NodeTelemetry(
        calls=int(raw_telemetry.get("calls") or payload.get("calls") or 0),
        failures=int(raw_telemetry.get("failures") or 0),
        latency_p50_ms=raw_telemetry.get("latency_p50_ms"),
        latency_p95_ms=raw_telemetry.get("latency_p95_ms"),
        latency_max_ms=raw_telemetry.get("latency_max_ms"),
        queue_depth=raw_telemetry.get("queue_depth"),
        tokens=raw_telemetry.get("tokens"),
        heartbeat_age_s=raw_telemetry.get("heartbeat_age_s"),
        current_action=str(raw_telemetry.get("current_action") or ""),
    )
    try:
        kind = NodeKind(str(payload.get("kind") or "subsystem"))
    except ValueError:
        kind = NodeKind.SUBSYSTEM
    try:
        health = Health(str(payload.get("health") or "UNKNOWN").upper())
    except ValueError:
        health = Health.UNKNOWN
    try:
        activity = Activity(str(payload.get("activity") or "idle"))
    except ValueError:
        activity = Activity.IDLE

    node = SwarmNode(
        id=str(payload.get("id") or ""), kind=kind,
        label=str(payload.get("label") or payload.get("id") or "?"),
        cluster=str(payload.get("cluster") or "SYSTEM"),
        health=health, activity=activity,
        detail=str(payload.get("detail") or payload.get("state") or ""),
        telemetry=telemetry,
        capabilities=tuple(payload.get("capabilities") or []),
        dependencies=tuple(payload.get("dependencies") or []),
    )
    return build_report(node, None, can_open_agent)
