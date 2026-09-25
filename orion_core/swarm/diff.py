"""
diff.py — what actually changed between two swarm snapshots (Mark XXII, Phase 0).

This module exists to fix the single worst defect the audit found: the old
view pushed the ENTIRE graph over the bridge every two seconds and the JS side
responded by disposing and reallocating every geometry and material. Cost
scaled with how much ORION contains rather than with how much of it moved —
7 KB and a full scene rebuild at today's size, 196 KB and the same rebuild at
the 1,000-agent target.

The classification is the whole point, and it rests on model.py's split
between visual identity and telemetry payload:

    added / removed        the graph's shape changed  -> allocate/free instances
    visual_changed         how a node LOOKS changed   -> write its GPU instance
    telemetry_changed      only its numbers changed   -> refresh inspector only

That third bucket is the common case on almost every tick — latency and call
counts drift constantly while health and membership are stable — and it is
precisely the case that must NOT touch the GPU. Separating it is what lets the
renderer idle at zero upload cost while ORION is merely running normally.

Pure functions over immutable snapshots: no Qt, no I/O, no shared state, so
this is exhaustively testable and safe to call from a worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import SwarmEdge, SwarmNode, SwarmSnapshot

# Traffic is a 0..1 float that jitters continuously. Re-uploading an edge
# because its intensity moved by 0.001 would reintroduce exactly the per-tick
# churn this module exists to remove, so changes below this threshold are
# treated as no change at all.
_TRAFFIC_EPSILON = 0.02


@dataclass(frozen=True, slots=True)
class SwarmDelta:
    """The minimum work required to bring a renderer up to date.

    Every collection is ordered deterministically (by node/edge id) so a delta
    is reproducible and diffable in tests, and so GPU instance slots are
    assigned in a stable order rather than whatever order a dict happened to
    iterate in.
    """

    added: tuple[SwarmNode, ...] = ()
    removed: tuple[str, ...] = ()
    visual_changed: tuple[SwarmNode, ...] = ()
    telemetry_changed: tuple[SwarmNode, ...] = ()
    edges_added: tuple[SwarmEdge, ...] = ()
    edges_removed: tuple[tuple[str, str, str], ...] = ()
    edges_traffic_changed: tuple[SwarmEdge, ...] = ()
    core_state_changed: bool = False
    degraded_sources_changed: bool = False

    @property
    def is_empty(self) -> bool:
        """True when nothing needs doing at all — the expected steady state,
        and the cheapest possible frame."""
        return not (
            self.added or self.removed or self.visual_changed
            or self.telemetry_changed or self.edges_added or self.edges_removed
            or self.edges_traffic_changed or self.core_state_changed
            or self.degraded_sources_changed
        )

    @property
    def needs_gpu_upload(self) -> bool:
        """True only when the SHAPE or APPEARANCE of the graph changed.

        Telemetry-only churn deliberately does not qualify: the renderer can
        skip its buffer write entirely and the inspector still updates."""
        return bool(self.added or self.removed or self.visual_changed
                    or self.edges_added or self.edges_removed)

    @property
    def needs_layout(self) -> bool:
        """True only when nodes appeared or disappeared. Recolouring a node
        never moves it, so a health flip must not trigger a re-solve."""
        return bool(self.added or self.removed)

    def summary(self) -> dict[str, Any]:
        """Compact counts for logging and the diagnostics panel."""
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "visual_changed": len(self.visual_changed),
            "telemetry_changed": len(self.telemetry_changed),
            "edges_added": len(self.edges_added),
            "edges_removed": len(self.edges_removed),
            "edges_traffic_changed": len(self.edges_traffic_changed),
            "core_state_changed": self.core_state_changed,
            "gpu_upload": self.needs_gpu_upload,
            "relayout": self.needs_layout,
        }


EMPTY_DELTA = SwarmDelta()


def _edge_map(snapshot: SwarmSnapshot) -> dict[tuple[str, str, str], SwarmEdge]:
    # kind is stored as its str value so the key survives JSON round-tripping
    # (the remote/phone surface sends deltas over the wire).
    return {(e.source, e.target, e.kind.value): e for e in snapshot.edges}


def diff_snapshots(
    previous: SwarmSnapshot | None, current: SwarmSnapshot
) -> SwarmDelta:
    """Classify every change between *previous* and *current*.

    A None *previous* (first ever frame) yields every node as `added`, which
    is correct: the renderer genuinely must allocate all of them once.
    """
    if current is None:  # defensive: never let a bad caller crash the render loop
        return EMPTY_DELTA
    if previous is None:
        return SwarmDelta(
            added=tuple(sorted(current.nodes, key=lambda n: n.id)),
            edges_added=tuple(sorted(current.edges, key=lambda e: e.key)),
            core_state_changed=True,
            degraded_sources_changed=bool(current.degraded_sources),
        )

    old_nodes = previous.by_id()
    new_nodes = current.by_id()

    added = tuple(sorted(
        (n for nid, n in new_nodes.items() if nid not in old_nodes),
        key=lambda n: n.id))
    removed = tuple(sorted(nid for nid in old_nodes if nid not in new_nodes))

    visual_changed: list[SwarmNode] = []
    telemetry_changed: list[SwarmNode] = []
    for nid, node in new_nodes.items():
        old = old_nodes.get(nid)
        if old is None:
            continue  # already counted as added
        if old.visual_key != node.visual_key:
            # Appearance changed. Any telemetry drift rides along with the
            # same upload, so it is deliberately NOT also reported as a
            # telemetry change — that would make callers do the work twice.
            visual_changed.append(node)
        elif old.telemetry != node.telemetry or old.detail != node.detail:
            telemetry_changed.append(node)

    old_edges = _edge_map(previous)
    new_edges = _edge_map(current)

    edges_added = tuple(sorted(
        (e for key, e in new_edges.items() if key not in old_edges),
        key=lambda e: e.key))
    edges_removed = tuple(sorted(key for key in old_edges if key not in new_edges))

    traffic_changed: list[SwarmEdge] = []
    for key, edge in new_edges.items():
        old_edge = old_edges.get(key)
        if old_edge is None:
            continue
        if abs(old_edge.traffic - edge.traffic) >= _TRAFFIC_EPSILON:
            traffic_changed.append(edge)

    return SwarmDelta(
        added=added,
        removed=removed,
        visual_changed=tuple(sorted(visual_changed, key=lambda n: n.id)),
        telemetry_changed=tuple(sorted(telemetry_changed, key=lambda n: n.id)),
        edges_added=edges_added,
        edges_removed=edges_removed,
        edges_traffic_changed=tuple(sorted(traffic_changed, key=lambda e: e.key)),
        core_state_changed=previous.core_state != current.core_state,
        degraded_sources_changed=(
            previous.degraded_sources != current.degraded_sources),
    )
