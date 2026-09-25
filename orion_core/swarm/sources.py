"""
sources.py — compose a live SwarmSnapshot from ORION's real subsystems.

The successor to gui/swarm_view.py's compose_snapshot(). That function was
measured at 0.067 ms and is genuinely fast, so its shape is deliberately kept:
one pure function, every backend optional, each source wrapped independently
so a single dead subsystem can never blank the whole graph. What changes is
what it PRODUCES — typed nodes carrying real telemetry, arranged into the
twelve clusters ORION is actually organised into, instead of six anonymous
kinds on one flat ring.

Two behaviours are new and deliberate:

  * Cluster nodes are synthesised. ORION has no runtime object called
    "AUTOMATION", but the zone is real (it is a Command Deck tab), and giving
    it a node is what lets members orbit their cluster instead of all 36
    subsystems orbiting core in one undifferentiated ring.

  * Failures are reported, not swallowed. The old version caught every source
    exception and moved on silently, so a crashed MCP host looked identical to
    having no MCP servers configured. Each failure is now recorded in
    `degraded_sources` so the UI can say which part of the picture is missing.
"""

from __future__ import annotations

import time
from typing import Any

from .clusters import (
    AUTOMATION,
    CLUSTER_ORDER,
    COMMUNICATION,
    INTELLIGENCE,
    MEMORY,
    cluster_for_module,
    cluster_of_agent,
    normalise_cluster,
)
from .model import (
    Activity,
    EdgeKind,
    Health,
    NodeKind,
    NodeTelemetry,
    SwarmEdge,
    SwarmNode,
    SwarmSnapshot,
)
from .telemetry_bind import TelemetryIndex, activity_for

CORE_ID = "core"


def cluster_id(cluster: str) -> str:
    return f"cluster:{cluster}"


class _Composer:
    """Accumulates nodes/edges while keeping per-source failure isolated.

    A class rather than a pile of locals purely so each _add_* step can fail
    independently and record why, without threading three accumulators through
    every function signature.
    """

    def __init__(self, index: TelemetryIndex) -> None:
        self.index = index
        self.nodes: list[SwarmNode] = []
        self.edges: list[SwarmEdge] = []
        self.degraded: list[str] = []
        self._seen: set[str] = set()

    def add_node(self, node: SwarmNode) -> bool:
        """Nodes are keyed by id; a duplicate is dropped rather than allowed to
        create two GPU instances that fight over one identity."""
        if node.id in self._seen:
            return False
        self._seen.add(node.id)
        self.nodes.append(node)
        return True

    def add_edge(self, source: str, target: str, kind: EdgeKind,
                 traffic: float = 0.0) -> None:
        self.edges.append(SwarmEdge(source=source, target=target, kind=kind,
                                    traffic=traffic))

    def run(self, name: str, fn: Any) -> None:
        """Run one source. A raising source degrades that source only."""
        try:
            fn()
        except Exception:
            self.degraded.append(name)


def compose(
    core_state: str = "STANDBY",
    agents: Any | None = None,
    registries: Any | None = None,
    telemetry: Any | None = None,
    mcp_host: Any | None = None,
    workflow_engine: Any | None = None,
    memory: Any | None = None,
    dispatcher: Any | None = None,
    deck_pages: dict[str, str] | None = None,
) -> SwarmSnapshot:
    """One complete observation of ORION, as a typed snapshot.

    *deck_pages* maps Command Deck page name -> zone name. Passed as plain
    data rather than the deck object itself so this module stays free of Qt
    and the GUI, which is what lets the same composer feed the native
    renderer, the remote surface and a headless diagnostic."""
    index = TelemetryIndex(telemetry)
    c = _Composer(index)

    # ── core ────────────────────────────────────────────────────────────────
    state = str(core_state or "STANDBY")
    c.add_node(SwarmNode(
        id=CORE_ID, kind=NodeKind.CORE, label="ORION", cluster=INTELLIGENCE,
        health=Health.OK,
        activity=Activity.THINKING if state.upper() in {
            "THINKING", "PROCESSING"} else Activity.IDLE,
        detail=state,
        telemetry=NodeTelemetry(
            calls=int(index.counter("tool.calls")),
            failures=int(index.counter("tool.failures")),
            current_action=state,
        ),
    ))

    # ── clusters (synthesised; see module docstring) ────────────────────────
    for cluster in CLUSTER_ORDER:
        c.add_node(SwarmNode(
            id=cluster_id(cluster), kind=NodeKind.SUBSYSTEM, label=cluster.title(),
            cluster=cluster, health=Health.OK, activity=Activity.IDLE,
            detail=f"{cluster.title()} cluster", parent=CORE_ID,
        ))
        c.add_edge(CORE_ID, cluster_id(cluster), EdgeKind.MEMBERSHIP)

    # ── specialist agents ───────────────────────────────────────────────────
    def _agents() -> None:
        if agents is None:
            return
        for row in agents.describe():
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            cluster = cluster_of_agent(name)
            calls = int(row.get("calls") or 0)
            action = str(row.get("last_active") or "")
            tel = NodeTelemetry(calls=calls, current_action=action)
            c.add_node(SwarmNode(
                id=f"agent:{name}", kind=NodeKind.AGENT,
                label=str(row.get("title") or name), cluster=cluster,
                health=Health.OK, activity=activity_for(tel, Health.OK),
                detail=str(row.get("focus") or ""), parent=cluster_id(cluster),
                telemetry=tel,
            ))
            c.add_edge(cluster_id(cluster), f"agent:{name}", EdgeKind.DELEGATION)

    # ── registered subsystems ───────────────────────────────────────────────
    def _modules() -> None:
        if registries is None:
            return
        described = registries.modules.all_described(telemetry)
        known = {f"module:{n}" for n in described}
        for name, record in described.items():
            cluster = cluster_for_module(name)
            health = Health.UNKNOWN
            try:
                health = Health(str(record.get("health") or "UNKNOWN").upper())
            except ValueError:
                pass
            tel = index.for_component(name)
            c.add_node(SwarmNode(
                id=f"module:{name}", kind=NodeKind.SUBSYSTEM, label=name,
                cluster=cluster, health=health,
                activity=activity_for(tel, health),
                detail=str(record.get("role") or ""), parent=cluster_id(cluster),
                telemetry=tel,
                dependencies=tuple(str(d) for d in (record.get("dependencies") or [])),
            ))
            c.add_edge(cluster_id(cluster), f"module:{name}", EdgeKind.MEMBERSHIP)
        # Dependency edges only between modules that both exist, so a
        # not-yet-mapped dependency never produces a dangling edge.
        for name, record in described.items():
            for dep in record.get("dependencies") or []:
                if f"module:{dep}" in known:
                    c.add_edge(f"module:{name}", f"module:{dep}", EdgeKind.DEPENDENCY)

    # ── MCP servers and their tools ─────────────────────────────────────────
    def _mcp() -> None:
        if mcp_host is None:
            return
        catalogue = mcp_host.catalogue() or {}
        try:
            health_map = mcp_host.health_snapshot() or {}
        except Exception:
            health_map = {}
        for server, info in catalogue.items():
            tools = list((info or {}).get("tools") or [])
            try:
                health = Health(str(health_map.get(server) or "UNKNOWN").upper())
            except ValueError:
                health = Health.UNKNOWN
            server_node = f"mcp:{server}"
            c.add_node(SwarmNode(
                id=server_node, kind=NodeKind.MCP_SERVER, label=str(server),
                cluster=COMMUNICATION, health=health,
                activity=Activity.OFFLINE if health is Health.DOWN else Activity.IDLE,
                detail=str((info or {}).get("description") or f"{len(tools)} tool(s)"),
                parent=cluster_id(COMMUNICATION),
                capabilities=tuple(str(t.get("name") or "") for t in tools if t),
            ))
            c.add_edge(cluster_id(COMMUNICATION), server_node, EdgeKind.MEMBERSHIP)
            # Each MCP tool is dispatchable as mcp__<server>__<tool> (Track E1),
            # so its telemetry is real, per-tool, and already being recorded.
            for tool in tools:
                tool_name = str((tool or {}).get("name") or "").strip()
                if not tool_name:
                    continue
                tel = index.for_tool(f"mcp__{server}__{tool_name}")
                c.add_node(SwarmNode(
                    id=f"mcp_tool:{server}:{tool_name}", kind=NodeKind.MCP_TOOL,
                    label=tool_name, cluster=COMMUNICATION, health=health,
                    activity=activity_for(tel, health),
                    detail=str((tool or {}).get("description") or ""),
                    parent=server_node, telemetry=tel,
                ))
                c.add_edge(server_node, f"mcp_tool:{server}:{tool_name}",
                           EdgeKind.MCP_REQUEST)

    # ── workflows ───────────────────────────────────────────────────────────
    def _workflows() -> None:
        if workflow_engine is None:
            return
        for name, definition in (workflow_engine.definitions or {}).items():
            steps = list((definition or {}).get("steps") or [])
            c.add_node(SwarmNode(
                id=f"workflow:{name}", kind=NodeKind.WORKFLOW, label=str(name),
                cluster=AUTOMATION, health=Health.OK, activity=Activity.IDLE,
                detail=str((definition or {}).get("description")
                           or f"{len(steps)} step(s)"),
                parent=cluster_id(AUTOMATION),
                capabilities=tuple(str((s or {}).get("tool") or "") for s in steps),
            ))
            c.add_edge(cluster_id(AUTOMATION), f"workflow:{name}", EdgeKind.MEMBERSHIP)

    # ── memory tiers ────────────────────────────────────────────────────────
    def _memory() -> None:
        if memory is None:
            return
        from .memory_field import describe_tiers, detail_for

        # Memory is placed by RECENCY, not listed alphabetically: working
        # memory sits closest to ORION and archived storage recedes into deep
        # space, so the shape of the field is the shape of his recall.
        for tier in describe_tiers(memory.tiers_snapshot()):
            c.add_node(SwarmNode(
                id=tier.node_id, kind=NodeKind.MEMORY_TIER, label=tier.name,
                cluster=MEMORY, health=Health.OK,
                # Live tiers read as active; cold storage stays quiet until a
                # recall actually reaches it.
                activity=(Activity.ACTIVE if tier.rows and tier.is_live
                          else Activity.IDLE),
                detail=detail_for(tier), parent=cluster_id(MEMORY),
                telemetry=NodeTelemetry(queue_depth=tier.rows),
            ))
            c.add_edge(cluster_id(MEMORY), tier.node_id, EdgeKind.MEMORY_WRITE)

    # ── Command Deck pages ──────────────────────────────────────────────────
    # The deck migrated INTO the swarm: every page ORION offers is a node in
    # its zone's cluster, so the graph contains all of him rather than only
    # his agents, and opening a page node is how you navigate there. The
    # deck's tab bar becomes one view of this graph instead of a parallel
    # navigation system that has to be kept in sync by hand.
    def _pages() -> None:
        if not deck_pages:
            return
        for page, zone in deck_pages.items():
            page_name = str(page or "").strip()
            if not page_name:
                continue
            cluster = normalise_cluster(zone)
            node_id = f"page:{page_name}"
            c.add_node(SwarmNode(
                id=node_id, kind=NodeKind.PAGE, label=page_name,
                cluster=cluster, health=Health.OK, activity=Activity.IDLE,
                detail=f"{cluster.title()} deck page",
                parent=cluster_id(cluster),
            ))
            c.add_edge(cluster_id(cluster), node_id, EdgeKind.MEMBERSHIP)

    c.run("pages", _pages)
    c.run("agents", _agents)
    c.run("modules", _modules)
    c.run("mcp", _mcp)
    c.run("workflows", _workflows)
    c.run("memory", _memory)
    if telemetry is not None and not index.available:
        c.degraded.append("telemetry")

    return SwarmSnapshot(
        nodes=tuple(c.nodes),
        edges=tuple(c.edges),
        core_state=state,
        captured_at=time.time(),
        degraded_sources=tuple(c.degraded),
    )
