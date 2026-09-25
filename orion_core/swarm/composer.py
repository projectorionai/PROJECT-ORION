"""
composer.py — incremental snapshot composition (Mark XXII, Phase 2).

Phase 0 replaced a thin 1,568-node graph with a rich 1,700-node one carrying
real telemetry, twelve clusters and MCP tools as first-class nodes. That was
the right trade — it eliminated 170 KB of bridge traffic and a full scene
rebuild per tick — but it made compose() itself slower, from 2.2 ms to 9.0 ms
at the 1,000-agent target, because it allocates ~3,600 frozen dataclasses
every time it runs. Profiling showed no single hotspot left: the cost IS the
allocation.

So the fix is to stop allocating. This composer separates the two things
compose() does, because they change at completely different rates:

    STRUCTURE   which nodes and edges exist          changes rarely
    TELEMETRY   the numbers hanging off them         changes constantly

Structure is cached behind a cheap signature and only rebuilt when the set of
agents, modules, MCP tools, workflows or memory tiers actually changes.
Telemetry is rebound every tick — but a node whose measurements are unchanged
is REUSED rather than reallocated, so the steady state allocates roughly
nothing.

This is a pure optimisation and produces snapshots identical to sources.compose;
test_swarm_composer.py asserts that equivalence directly rather than trusting it.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from .model import Health, NodeKind, SwarmNode, SwarmSnapshot
from .sources import compose as compose_full
from .telemetry_bind import TelemetryIndex, activity_for


def _structure_signature(
    agents: Any | None, registries: Any | None, mcp_host: Any | None,
    workflow_engine: Any | None, memory: Any | None,
) -> tuple:
    """A cheap fingerprint of WHICH things exist, ignoring their measurements.

    Deliberately reads only names/keys, never telemetry: if this included
    anything that changes per tick the cache would never hit. Each source is
    isolated so a raising backend invalidates only its own part — and, by
    producing a distinct marker, correctly forces a rebuild rather than
    silently reusing a structure that no longer reflects reality.
    """
    parts: list[Any] = []

    def _try(fn: Any, label: str) -> None:
        try:
            parts.append(fn())
        except Exception:
            parts.append((label, "ERROR"))

    if agents is not None:
        _try(lambda: ("agents", tuple(sorted(
            str(r.get("name") or "") for r in agents.describe()))), "agents")
    if registries is not None:
        _try(lambda: ("modules", tuple(sorted(
            registries.modules.all_described(None).keys()))), "modules")
    if mcp_host is not None:
        _try(lambda: ("mcp", tuple(sorted(
            (str(server), tuple(sorted(
                str((t or {}).get("name") or "")
                for t in ((info or {}).get("tools") or []))))
            for server, info in (mcp_host.catalogue() or {}).items()))), "mcp")
    if workflow_engine is not None:
        _try(lambda: ("workflows", tuple(sorted(
            str(k) for k in (workflow_engine.definitions or {})))), "workflows")
    if memory is not None:
        _try(lambda: ("memory", tuple(sorted(
            str(k) for k in (memory.tiers_snapshot() or {})))), "memory")
    return tuple(parts)


class SnapshotComposer:
    """Stateful, cache-backed equivalent of sources.compose.

    One instance per swarm surface. Not thread-safe by design — it is owned by
    whichever component polls it, and sharing one across threads would race on
    the cache for no benefit, since composing is already sub-millisecond once
    warm.
    """

    __slots__ = ("_signature", "_nodes", "_edges", "_rebuilds", "_reuses",
                 "_node_allocations")

    def __init__(self) -> None:
        self._signature: tuple | None = None
        self._nodes: tuple[SwarmNode, ...] = ()
        self._edges: tuple = ()
        self._rebuilds = 0
        self._reuses = 0
        self._node_allocations = 0

    @property
    def structure_rebuilds(self) -> int:
        return self._rebuilds

    @property
    def structure_reuses(self) -> int:
        return self._reuses

    @property
    def node_allocations(self) -> int:
        """Nodes reallocated since construction — the number this class exists
        to keep near zero in the steady state."""
        return self._node_allocations

    def compose(
        self,
        core_state: str = "STANDBY",
        agents: Any | None = None,
        registries: Any | None = None,
        telemetry: Any | None = None,
        mcp_host: Any | None = None,
        workflow_engine: Any | None = None,
        memory: Any | None = None,
        deck_pages: dict[str, str] | None = None,
    ) -> SwarmSnapshot:
        signature = _structure_signature(
            agents, registries, mcp_host, workflow_engine, memory)
        # Deck pages are part of the graph's SHAPE, so a change to them must
        # invalidate the cached structure like any other source.
        signature = signature + (("pages", tuple(sorted(
            (deck_pages or {}).items()))),)

        if signature != self._signature or not self._nodes:
            # Structure changed (or first run): fall back to the full,
            # authoritative composer. One implementation of the graph's shape,
            # never two that could drift.
            snapshot = compose_full(
                core_state=core_state, agents=agents, registries=registries,
                telemetry=telemetry, mcp_host=mcp_host,
                workflow_engine=workflow_engine, memory=memory,
                deck_pages=deck_pages)
            self._signature = signature
            self._nodes = snapshot.nodes
            self._edges = snapshot.edges
            self._rebuilds += 1
            self._node_allocations += len(snapshot.nodes)
            return snapshot

        # Structure is unchanged — reuse every node object whose measurements
        # also held steady, and rebind only those that moved.
        self._reuses += 1
        index = TelemetryIndex(telemetry)
        degraded: list[str] = []
        if telemetry is not None and not index.available:
            degraded.append("telemetry")

        health_map: dict[str, Health] = {}
        if registries is not None:
            try:
                for name, record in registries.modules.all_described(telemetry).items():
                    try:
                        health_map[name] = Health(
                            str(record.get("health") or "UNKNOWN").upper())
                    except ValueError:
                        health_map[name] = Health.UNKNOWN
            except Exception:
                degraded.append("modules")

        rebound: list[SwarmNode] = []
        for node in self._nodes:
            updated = self._rebind(node, index, health_map, core_state)
            if updated is not node:
                self._node_allocations += 1
            rebound.append(updated)

        self._nodes = tuple(rebound)
        return SwarmSnapshot(
            nodes=self._nodes,
            edges=self._edges,
            core_state=str(core_state or "STANDBY"),
            captured_at=time.time(),
            degraded_sources=tuple(degraded),
        )

    @staticmethod
    def _rebind(node: SwarmNode, index: TelemetryIndex,
                health_map: dict[str, Health], core_state: str) -> SwarmNode:
        """Return *node* updated for current measurements, or the SAME object
        when nothing changed — identity is the signal the caller uses to count
        (and avoid) allocations."""
        if node.kind is NodeKind.CORE:
            telemetry = replace(
                node.telemetry,
                calls=int(index.counter("tool.calls")),
                failures=int(index.counter("tool.failures")),
                current_action=str(core_state or ""),
            )
            if telemetry == node.telemetry and node.detail == core_state:
                return node
            return replace(node, telemetry=telemetry, detail=str(core_state or ""))

        if node.kind is NodeKind.SUBSYSTEM and node.id.startswith("module:"):
            name = node.id.split(":", 1)[1]
            health = health_map.get(name, node.health)
            telemetry = index.for_component(name)
            activity = activity_for(telemetry, health)
            if (telemetry == node.telemetry and health is node.health
                    and activity is node.activity):
                return node
            return replace(node, telemetry=telemetry, health=health,
                           activity=activity)

        if node.kind is NodeKind.MCP_TOOL:
            # id shape: mcp_tool:<server>:<tool>  ->  tool mcp__<server>__<tool>
            parts = node.id.split(":", 2)
            if len(parts) == 3:
                telemetry = index.for_tool(f"mcp__{parts[1]}__{parts[2]}")
                activity = activity_for(telemetry, node.health)
                if telemetry == node.telemetry and activity is node.activity:
                    return node
                return replace(node, telemetry=telemetry, activity=activity)

        # Cluster anchors, agents, workflows and memory tiers carry no
        # per-tick metric source of their own, so they are returned untouched.
        return node

    def invalidate(self) -> None:
        """Force a full rebuild on the next compose — used when a caller knows
        the structure changed in a way the signature cannot see."""
        self._signature = None
        self._nodes = ()
        self._edges = ()

    def stats(self) -> dict[str, Any]:
        total = self._rebuilds + self._reuses
        return {
            "composes": total,
            "structure_rebuilds": self._rebuilds,
            "structure_reuses": self._reuses,
            "node_allocations": self._node_allocations,
            "cached_nodes": len(self._nodes),
            "reuse_ratio": round(self._reuses / total, 4) if total else 0.0,
        }
