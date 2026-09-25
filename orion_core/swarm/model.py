"""
model.py — the typed swarm graph (Mark XXII, Phase 0).

Pure data. No Qt, no I/O, no behaviour beyond derivation from own fields, so
every renderer (native GL, WebEngine, remote, headless) consumes exactly the
same objects and the model stays testable without a display.

The split that matters is between VISUAL identity and TELEMETRY payload:

    SwarmNode        what the node IS and how it should be drawn
    NodeTelemetry    live numbers that change constantly

They are separated because they have completely different update economics.
Health flipping OK -> DOWN must repaint the node (a GPU buffer write); p95
latency drifting 12.1 -> 12.4 ms must not — it only matters if that node is
currently selected in the inspector. diff.py relies on this split to classify
change, which is what stops the renderer doing work proportional to the size
of the graph rather than to what actually moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    """What a node fundamentally is. str-valued so it serialises directly."""

    CORE = "core"
    AGENT = "agent"
    SUBSYSTEM = "subsystem"
    MCP_SERVER = "mcp_server"
    MCP_TOOL = "mcp_tool"
    WORKFLOW = "workflow"
    MEMORY_TIER = "memory_tier"
    MODEL = "model"
    TOOL = "tool"
    # A Command Deck page. The swarm is meant to contain ALL of ORION, not
    # just his agents, so every surface the deck offers is a node in the
    # graph and opening it is how you get there — the deck's tab bar becomes
    # a view of the swarm rather than a separate navigation system.
    PAGE = "page"


class Health(str, Enum):
    """Whether the thing is working. Mirrors Telemetry.health's vocabulary so
    no translation table is needed at the boundary."""

    OK = "OK"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class Activity(str, Enum):
    """What the thing is doing right now — orthogonal to Health. A node can be
    perfectly healthy and idle, or busy and degraded; collapsing these into one
    field is what made the old graph unable to show ORION actually working."""

    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    THINKING = "thinking"
    ERROR = "error"
    OFFLINE = "offline"


# Health values severe enough that the node should read as a problem at a
# glance, without the reader having to compare colours.
_ALARMING = frozenset({Health.DEGRADED, Health.DOWN})


@dataclass(frozen=True, slots=True)
class NodeTelemetry:
    """Live measurements for one node.

    Every field is optional-by-default rather than required, because ORION's
    subsystems genuinely differ in what they can report: a memory tier has a
    row count and no latency, an MCP server has latency and no token usage.
    Absent is represented as None and rendered as "—", never as a misleading 0.
    """

    calls: int = 0
    failures: int = 0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_max_ms: float | None = None
    latency_avg_ms: float | None = None
    queue_depth: int | None = None
    tokens: int | None = None
    uptime_s: float | None = None
    heartbeat_age_s: float | None = None
    current_action: str = ""

    @property
    def error_rate(self) -> float:
        """Failures as a fraction of calls, 0.0 when nothing has been called.

        Derived rather than stored so it can never disagree with the counters
        it comes from."""
        if self.calls <= 0:
            return 0.0
        return min(1.0, self.failures / self.calls)

    @property
    def is_quiet(self) -> bool:
        """True when this node has done nothing measurable — used to render
        never-invoked subsystems as dim rather than as healthy-and-idle."""
        return self.calls == 0 and not self.current_action

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "error_rate": round(self.error_rate, 4),
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_avg_ms": self.latency_avg_ms,
            "queue_depth": self.queue_depth,
            "tokens": self.tokens,
            "uptime_s": self.uptime_s,
            "heartbeat_age_s": self.heartbeat_age_s,
            "current_action": self.current_action,
        }


EMPTY_TELEMETRY = NodeTelemetry()


@dataclass(frozen=True, slots=True)
class SwarmNode:
    """One real thing inside ORION.

    `id` is a stable namespaced key ("agent:coding", "mcp:github") — stable
    across refreshes so the renderer can keep a node's position, selection and
    GPU instance slot instead of everything jumping whenever the graph is
    recomposed.
    """

    id: str
    kind: NodeKind
    label: str
    cluster: str
    health: Health = Health.UNKNOWN
    activity: Activity = Activity.IDLE
    detail: str = ""
    parent: str | None = None
    telemetry: NodeTelemetry = EMPTY_TELEMETRY
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()

    @property
    def visual_key(self) -> tuple[Any, ...]:
        """Everything that changes how this node is DRAWN.

        diff.py compares this, not the whole node, to decide whether a GPU
        buffer write is required — so telemetry churn (the common case, every
        tick) never triggers one."""
        return (self.id, self.kind, self.label, self.cluster,
                self.health, self.activity, self.parent)

    @property
    def needs_attention(self) -> bool:
        return self.health in _ALARMING or self.activity is Activity.ERROR

    @property
    def is_cluster(self) -> bool:
        """True for the twelve synthesised zone anchors.

        Identified by id prefix rather than a dedicated NodeKind: a cluster IS
        a subsystem grouping, and giving it its own kind would force every
        kind-based branch (palette, layout radius, inspector) to learn about a
        node that has no runtime object behind it. The prefix is assigned in
        exactly one place (sources.cluster_id), so it is not a loose
        convention."""
        return self.id.startswith("cluster:")

    @property
    def cluster_zone(self) -> str:
        """The zone this node's id names, for a cluster anchor — the Command
        Deck zone the swarm navigates to when it is opened."""
        return self.id.split(":", 1)[1] if self.is_cluster else ""

    def with_telemetry(self, telemetry: NodeTelemetry) -> "SwarmNode":
        """A copy carrying new measurements — the model is immutable, so
        binding live data produces a new node rather than mutating a shared
        one out from under whatever is currently rendering it."""
        return replace(self, telemetry=telemetry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "label": self.label,
            "cluster": self.cluster,
            "health": self.health.value,
            "activity": self.activity.value,
            "detail": self.detail,
            "parent": self.parent,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "telemetry": self.telemetry.as_dict(),
        }


class EdgeKind(str, Enum):
    """What a connection MEANS. The old graph had one anonymous edge type, so
    'core owns this agent' and 'this module depends on that one' were drawn
    identically and neither could show traffic."""

    MEMBERSHIP = "membership"      # cluster/core owns this node
    DEPENDENCY = "dependency"      # module A needs module B
    DELEGATION = "delegation"      # work handed to an agent
    TOOL_CALL = "tool_call"        # dispatcher invoked a tool
    MCP_REQUEST = "mcp_request"    # request to an MCP server
    MEMORY_WRITE = "memory_write"  # something persisted
    DATA_STREAM = "data_stream"    # continuous flow (audio, vision)


@dataclass(frozen=True, slots=True)
class SwarmEdge:
    """A connection between two nodes.

    `traffic` is a 0..1 intensity the renderer maps to packet density, so an
    edge can exist structurally while carrying nothing — which is the normal
    resting state and must look different from an edge under load."""

    source: str
    target: str
    kind: EdgeKind = EdgeKind.MEMBERSHIP
    traffic: float = 0.0

    @property
    def key(self) -> tuple[str, str, EdgeKind]:
        return (self.source, self.target, self.kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "traffic": round(self.traffic, 4),
        }


@dataclass(frozen=True, slots=True)
class SwarmSnapshot:
    """One complete observation of ORION at an instant."""

    nodes: tuple[SwarmNode, ...] = ()
    edges: tuple[SwarmEdge, ...] = ()
    core_state: str = "STANDBY"
    captured_at: float = 0.0
    # Sources that raised while composing. Carried rather than swallowed so
    # the UI can say "MCP data unavailable" instead of silently drawing a
    # graph that is quietly missing a whole subsystem.
    degraded_sources: tuple[str, ...] = field(default=())

    def node_ids(self) -> frozenset[str]:
        return frozenset(n.id for n in self.nodes)

    def by_id(self) -> dict[str, SwarmNode]:
        return {n.id: n for n in self.nodes}

    def in_cluster(self, cluster: str) -> tuple[SwarmNode, ...]:
        return tuple(n for n in self.nodes if n.cluster == cluster)

    def of_kind(self, kind: NodeKind) -> tuple[SwarmNode, ...]:
        return tuple(n for n in self.nodes if n.kind is kind)

    def needing_attention(self) -> tuple[SwarmNode, ...]:
        return tuple(n for n in self.nodes if n.needs_attention)

    def as_dict(self) -> dict[str, Any]:
        return {
            "core_state": self.core_state,
            "captured_at": self.captured_at,
            "degraded_sources": list(self.degraded_sources),
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
        }
