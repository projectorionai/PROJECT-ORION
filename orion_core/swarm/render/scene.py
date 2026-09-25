"""
scene.py — where the diff finally pays off (Mark XXII, Phase 1).

Owns the bridge from model to GPU state: snapshot in, minimal instance-buffer
mutation out. Still pure — numpy and the swarm model only, no Qt and no GL —
so the entire update path is unit-testable without a graphics context.

The update rule exploits a property the analytic layout gives for free. In
layout.py a node's position derives ONLY from its own id and its cluster's
fixed ring angle — never from sibling count or list index. So inserting the
fifty-first agent cannot move the other fifty, and a structural change means
"place the new nodes", not "re-solve and re-upload the graph". That is why
`added` and `visual_changed` are the only buckets that ever touch the buffer:

    added              compute position, write instance
    visual_changed     appearance (or cluster) changed, rewrite instance
    removed            retire the slot back to the pool
    telemetry_changed  DO NOTHING — inspector-only, zero GPU cost

That last line is the entire performance thesis of the redesign, and
test_swarm_scene.py asserts it directly.
"""

from __future__ import annotations

from typing import Any

from ..diff import SwarmDelta, diff_snapshots
from ..model import NodeKind, SwarmEdge, SwarmNode, SwarmSnapshot
from .edges import EdgeBuffer
from .instances import InstanceBuffer
from .layout import LayoutResult, LayoutSolver
from .palette import appearance_for


def _edge_key(edge: SwarmEdge) -> tuple[str, str, str]:
    return (edge.source, edge.target, edge.kind.value)


class SwarmScene:
    """Model-to-GPU-state controller."""

    __slots__ = ("buffer", "edges", "solver", "_layout", "_snapshot",
                 "_selected", "_last_delta", "_uploads", "_skipped_uploads",
                 "_clock", "_last_ring_solve", "cognition")

    def __init__(self, buffer: InstanceBuffer | None = None,
                 solver: LayoutSolver | None = None,
                 edges: EdgeBuffer | None = None,
                 clock: Any | None = None,
                 cognition: Any | None = None) -> None:
        self.buffer = buffer if buffer is not None else InstanceBuffer()
        self.edges = edges if edges is not None else EdgeBuffer()
        self.solver = solver if solver is not None else LayoutSolver()
        self._layout: LayoutResult | None = None
        self._snapshot: SwarmSnapshot | None = None
        self._selected: str | None = None
        self._last_delta: SwarmDelta | None = None
        # Counters, surfaced through stats() so the claim "telemetry churn
        # costs no upload" is observable in the running app, not just in tests.
        self._uploads = 0
        self._skipped_uploads = 0
        import time as _time
        from ..cognition_state import CognitionState

        self._clock = clock if clock is not None else _time.monotonic
        self._last_ring_solve = -1e9      # diagnostic timestamp of last solve
        # Live cognition drives brightness/speed per node. Constructed here
        # rather than injected everywhere so the scene always has one, and
        # shared with the view so tool dispatch can engage nodes on it.
        self.cognition = cognition if cognition is not None else CognitionState()

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def snapshot(self) -> SwarmSnapshot | None:
        return self._snapshot

    @property
    def last_delta(self) -> SwarmDelta | None:
        return self._last_delta

    @property
    def selected(self) -> str | None:
        return self._selected

    def node(self, node_id: str) -> SwarmNode | None:
        if self._snapshot is None:
            return None
        return self._snapshot.by_id().get(node_id)

    def selected_node(self) -> SwarmNode | None:
        return None if self._selected is None else self.node(self._selected)

    def position_of(self, node_id: str) -> tuple[float, float, float] | None:
        if self._layout is None:
            return None
        return self._layout.positions.get(node_id)

    # ── the update path ──────────────────────────────────────────────────────

    def apply(self, snapshot: SwarmSnapshot) -> SwarmDelta:
        """Fold a new observation into GPU state, touching as little as
        possible. Returns the delta so callers can react (log it, refresh an
        inspector, trigger a pulse) without recomputing it."""
        previous = self._snapshot
        delta = diff_snapshots(previous, snapshot)
        self._last_delta = delta
        self._snapshot = snapshot

        if delta.is_empty:
            self._skipped_uploads += 1
            return delta

        # Positions normally only need re-solving when the node SET changed —
        # a health flip or a latency tick never moves anything.
        #
        # The exception is a node that changed CLUSTER or PARENT. Those are
        # visual changes (both are in visual_key) but not additions, so
        # needs_layout is False and the cached layout would still hold the
        # node's old coordinates — it would recolour in place and never
        # actually move to its new cluster. Detected by comparing against the
        # previous snapshot rather than by re-solving on every visual change,
        # because health flips are frequent and re-parenting is rare.
        # Stable positions make the swarm navigable: a telemetry update must
        # not move every subsystem or rewrite every GPU instance.  A layout is
        # only needed for a changed topology (or the first snapshot).
        relayout = delta.needs_layout or self._layout is None
        if not relayout and delta.visual_changed and previous is not None:
            was = previous.by_id()
            for node in delta.visual_changed:
                old = was.get(node.id)
                if old is not None and (old.cluster != node.cluster
                                        or old.parent != node.parent):
                    relayout = True
                    break
        if relayout:
            self._layout = self.solver.solve(snapshot.nodes)
            self._last_ring_solve = self._clock()
            # A structural relayout is rare. When it happens, update the whole
            # coherent topology so nodes and edge endpoints always agree.
            for node in snapshot.nodes:
                self._write(node)
            for edge in snapshot.edges:
                self._write_edge(edge)

        # Traffic changes are cheap and frequent: two floats per edge, no
        # positions touched, no node work at all. Handled before the
        # needs_gpu_upload gate because a pure traffic frame must still
        # animate even though no node changed appearance.
        for edge in delta.edges_traffic_changed:
            self.edges.set_traffic(_edge_key(edge), edge.traffic)

        if not delta.needs_gpu_upload:
            # Telemetry-only frame: the inspector will read the new snapshot,
            # and the node buffer is left entirely alone.
            self._skipped_uploads += 1
            return delta

        for node_id in delta.removed:
            self.buffer.remove(node_id)

        for node in delta.added:
            self._write(node)
        for node in delta.visual_changed:
            self._write(node)

        for key in delta.edges_removed:
            self.edges.remove(key)
        for edge in delta.edges_added:
            self._write_edge(edge)
        # A relayout moves node positions, so every surviving edge's endpoints
        # are now stale and must be rewritten — not just the new ones.
        if relayout:
            for edge in snapshot.edges:
                self._write_edge(edge)

        self._uploads += 1
        return delta

    def _write_edge(self, edge: SwarmEdge) -> None:
        layout = self._layout
        if layout is None:
            return
        source = layout.positions.get(edge.source)
        target = layout.positions.get(edge.target)
        if source is None or target is None:
            # An edge to a node that is not in the graph would otherwise be
            # drawn to the origin, producing a spray of lines into the centre.
            return
        self.edges.upsert(_edge_key(edge), source, target, edge.traffic)

    def _write(self, node: SwarmNode) -> None:
        # The core node is not drawn as a sphere: ORION's own face is overlaid
        # at the centre of the graph and IS the core. Drawing a sphere there
        # too would sit a featureless ball on top of his face. It keeps its
        # instance slot (radius 0 = degenerate, invisible) so edges still
        # terminate at the exact centre and appear to radiate from him.
        if node.kind is NodeKind.CORE:
            self.buffer.upsert(node.id, position=(0.0, 0.0, 0.0),
                               colour=(0.0, 0.0, 0.0), emissive=0.0,
                               pulse_hz=0.0, radius=0.0, selected=False)
            return
        layout = self._layout
        position = (layout.positions.get(node.id, (0.0, 0.0, 0.0))
                    if layout else (0.0, 0.0, 0.0))
        radius = layout.radii.get(node.id, 0.7) if layout else 0.7
        look = appearance_for(node)
        # Cognition modulates the drawn result: engaged nodes brighten and
        # pulse faster, unrelated ones dim while something else works. This
        # is where "watch ORION think" stops being a metaphor — the values
        # come from real dispatched work, not an animation curve.
        think = self.cognition.for_node(node.id)
        emissive = min(2.5, look.emissive * think.brightness)
        pulse = look.pulse_hz * think.speed
        if think.activation > 0.05 and pulse <= 0.0:
            # An engaged node pulses even if its resting state was steady,
            # so involvement is visible rather than merely brighter.
            pulse = 0.8 * think.speed
        self.buffer.upsert(
            node.id,
            position=position,
            colour=look.colour,
            emissive=emissive,
            pulse_hz=pulse,
            radius=radius * look.radius_scale,
            selected=(node.id == self._selected),
        )

    # ── selection ────────────────────────────────────────────────────────────

    def select(self, node_id: str | None) -> SwarmNode | None:
        """Change the inspector's subject. Dirties at most two slots — never
        the graph — so selecting is free regardless of how large it is."""
        if node_id is not None and self.buffer.slot_of(node_id) is None:
            node_id = None
        self._selected = node_id
        self.buffer.set_selected(node_id)
        return self.selected_node()

    def select_by_slot(self, slot: int) -> SwarmNode | None:
        """Resolve a picking hit (a slot index from the colour-ID buffer)
        back to a node and select it."""
        return self.select(self.buffer.id_at(slot))

    # ── diagnostics ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        total = self._uploads + self._skipped_uploads
        edge_stats = self.edges.stats()
        return {
            **self.buffer.stats(),
            "frames": total,
            "uploads": self._uploads,
            "skipped_uploads": self._skipped_uploads,
            "upload_ratio": round(self._uploads / total, 4) if total else 0.0,
            "selected": self._selected,
            "nodes": 0 if self._snapshot is None else len(self._snapshot.nodes),
            "edges": edge_stats["live"],
            "edge_vertices": edge_stats["vertices"],
        }

    def reset(self) -> None:
        """Full teardown — used on GL context loss, where the driver-side
        buffer is gone and everything must be re-uploaded from scratch."""
        self.buffer.reset()
        self.edges.reset()
        self._layout = None
        self._snapshot = None
        self._last_delta = None
        self._selected = None
