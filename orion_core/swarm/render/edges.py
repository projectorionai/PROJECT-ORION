"""
edges.py — the edge vertex buffer (Mark XXII, Phase 3).

Edges are GL_LINES: two vertices per edge, one buffer, one draw call. They are
kept in a separate buffer from nodes rather than folded in, for a reason that
matters beyond tidiness — picking maps a node instance slot to a node id, so
letting line vertices share that indexing would silently corrupt selection.

Each vertex carries a `t` parameter (0.0 at the source end, 1.0 at the target)
so the shader can animate a packet travelling along the edge purely from
`t`, `traffic` and a time uniform. Nothing about the animation is computed on
the CPU and nothing is re-uploaded to make it move: a busy edge and an idle
one differ by one float, and the packet travels for free on the GPU.

`seed` decorrelates the packets. Without it every edge in the graph would
pulse in lockstep, which reads as a screensaver rather than as traffic.

Same slot-pool and dirty-range discipline as InstanceBuffer, for the same
reason: traffic changes constantly and must never force a full re-upload.

numpy only — no Qt, no GL.
"""

from __future__ import annotations

import zlib

import numpy as np

# Per-vertex floats:
#   0..2  position xyz
#   3..5  colour rgb
#   6     t along the edge (0.0 at source, 1.0 at target)
#   7     traffic 0..1
#   8     seed (decorrelates packet phase between edges)
EDGE_FLOATS = 9
VERTS_PER_EDGE = 2

_OFF_POS = 0
_OFF_COLOUR = 3
_OFF_T = 6
_OFF_TRAFFIC = 7
_OFF_SEED = 8

_INITIAL_EDGES = 512

EdgeKey = tuple[str, str, str]

# Edge colour by kind. Deliberately dim and cool: edges are context, and at
# ~1,900 of them they must never compete with the nodes for attention. The
# packet highlight in the shader is what carries the eye, not the line.
# Warm, low-luminance filaments in ORION's own crimson register — the cool
# blues these started as belonged to a different colour world from the shell
# they are drawn inside. Kept dim on purpose: at ~1,900 edges they are the
# connective tissue, and the travelling packet is what should catch the eye.
_KIND_COLOURS: dict[str, tuple[float, float, float]] = {
    "membership":   (0.34, 0.20, 0.22),
    "dependency":   (0.38, 0.24, 0.24),
    "delegation":   (0.52, 0.28, 0.30),
    "tool_call":    (0.48, 0.26, 0.26),
    "mcp_request":  (0.46, 0.28, 0.44),
    "memory_write": (0.44, 0.26, 0.32),
    "data_stream":  (0.40, 0.30, 0.28),
}
_FALLBACK_COLOUR = (0.32, 0.20, 0.22)


def edge_colour(kind: str) -> tuple[float, float, float]:
    return _KIND_COLOURS.get(str(kind), _FALLBACK_COLOUR)


def _seed_for(key: EdgeKey) -> float:
    """A stable 0..1 phase offset per edge.

    crc32 rather than hash() for the same reason layout.py uses it: Python's
    str hashing is randomised per process, so packets would re-phase on every
    restart."""
    raw = zlib.crc32("|".join(key).encode("utf-8")) & 0xFFFF
    return raw / 65535.0


class EdgeBuffer:
    """Slot-pooled line-vertex buffer with dirty tracking."""

    __slots__ = ("_data", "_slot_of", "_key_at", "_free", "_high_water",
                 "_dirty_lo", "_dirty_hi", "_dirty_count")

    def __init__(self, capacity: int = _INITIAL_EDGES) -> None:
        capacity = max(1, int(capacity))
        self._data = np.zeros((capacity * VERTS_PER_EDGE, EDGE_FLOATS),
                              dtype=np.float32)
        self._slot_of: dict[EdgeKey, int] = {}
        self._key_at: dict[int, EdgeKey] = {}
        self._free: list[int] = []
        self._high_water = 0
        self._dirty_lo: int | None = None
        self._dirty_hi: int | None = None
        self._dirty_count = 0

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return int(self._data.shape[0]) // VERTS_PER_EDGE

    @property
    def vertex_count(self) -> int:
        """Vertices to draw — the high-water mark in vertices, since retired
        slots are collapsed to a zero-length line and cost nothing."""
        return self._high_water * VERTS_PER_EDGE

    @property
    def live_count(self) -> int:
        return len(self._slot_of)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def slot_of(self, key: EdgeKey) -> int | None:
        return self._slot_of.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._slot_of

    # ── dirty tracking ───────────────────────────────────────────────────────

    def _touch(self, slot: int) -> None:
        self._dirty_count += 1
        if self._dirty_lo is None or slot < self._dirty_lo:
            self._dirty_lo = slot
        if self._dirty_hi is None or slot > self._dirty_hi:
            self._dirty_hi = slot

    @property
    def is_dirty(self) -> bool:
        return self._dirty_lo is not None

    @property
    def dirty_slot_count(self) -> int:
        return self._dirty_count

    def dirty_vertex_range(self) -> tuple[int, int] | None:
        """Half-open [start, end) VERTEX span needing upload."""
        if self._dirty_lo is None or self._dirty_hi is None:
            return None
        return (self._dirty_lo * VERTS_PER_EDGE,
                (self._dirty_hi + 1) * VERTS_PER_EDGE)

    def clear_dirty(self) -> None:
        self._dirty_lo = None
        self._dirty_hi = None
        self._dirty_count = 0

    def mark_all_dirty(self) -> None:
        if self._high_water:
            self._dirty_lo = 0
            self._dirty_hi = self._high_water - 1
            self._dirty_count = self._high_water

    # ── capacity ─────────────────────────────────────────────────────────────

    def _ensure_capacity(self, edges_needed: int) -> None:
        if edges_needed <= self.capacity:
            return
        new_capacity = self.capacity
        while new_capacity < edges_needed:
            new_capacity *= 2
        grown = np.zeros((new_capacity * VERTS_PER_EDGE, EDGE_FLOATS),
                         dtype=np.float32)
        keep = self._high_water * VERTS_PER_EDGE
        grown[:keep] = self._data[:keep]
        self._data = grown

    def _acquire_slot(self) -> int:
        if self._free:
            return self._free.pop()
        slot = self._high_water
        self._ensure_capacity(slot + 1)
        self._high_water = slot + 1
        return slot

    # ── mutation ─────────────────────────────────────────────────────────────

    def upsert(self, key: EdgeKey,
               source_pos: tuple[float, float, float],
               target_pos: tuple[float, float, float],
               traffic: float = 0.0) -> int:
        """Insert or update one edge's two vertices."""
        slot = self._slot_of.get(key)
        if slot is None:
            slot = self._acquire_slot()
            self._slot_of[key] = slot
            self._key_at[slot] = key
        colour = edge_colour(key[2])
        seed = _seed_for(key)
        base = slot * VERTS_PER_EDGE
        for offset, position, t in ((0, source_pos, 0.0), (1, target_pos, 1.0)):
            row = self._data[base + offset]
            row[_OFF_POS:_OFF_POS + 3] = position
            row[_OFF_COLOUR:_OFF_COLOUR + 3] = colour
            row[_OFF_T] = t
            row[_OFF_TRAFFIC] = traffic
            row[_OFF_SEED] = seed
        self._touch(slot)
        return slot

    def set_traffic(self, key: EdgeKey, traffic: float) -> bool:
        """Update ONLY the traffic value — the common case, every tick.

        Writes two floats instead of eighteen and never touches positions, so
        a graph where only activity changed uploads almost nothing."""
        slot = self._slot_of.get(key)
        if slot is None:
            return False
        base = slot * VERTS_PER_EDGE
        self._data[base][_OFF_TRAFFIC] = traffic
        self._data[base + 1][_OFF_TRAFFIC] = traffic
        self._touch(slot)
        return True

    def remove(self, key: EdgeKey) -> bool:
        slot = self._slot_of.pop(key, None)
        if slot is None:
            return False
        self._key_at.pop(slot, None)
        base = slot * VERTS_PER_EDGE
        # Zeroed => both endpoints coincide => a degenerate, invisible line.
        self._data[base:base + VERTS_PER_EDGE] = 0.0
        self._free.append(slot)
        self._touch(slot)
        return True

    # ── reads ────────────────────────────────────────────────────────────────

    def view(self) -> np.ndarray:
        return self._data[:self._high_water * VERTS_PER_EDGE]

    def traffic_of(self, key: EdgeKey) -> float | None:
        slot = self._slot_of.get(key)
        if slot is None:
            return None
        return float(self._data[slot * VERTS_PER_EDGE][_OFF_TRAFFIC])

    def endpoints_of(self, key: EdgeKey):
        slot = self._slot_of.get(key)
        if slot is None:
            return None
        base = slot * VERTS_PER_EDGE
        return (tuple(float(v) for v in self._data[base][_OFF_POS:_OFF_POS + 3]),
                tuple(float(v) for v in self._data[base + 1][_OFF_POS:_OFF_POS + 3]))

    def reset(self) -> None:
        self._data[:self._high_water * VERTS_PER_EDGE] = 0.0
        self._slot_of.clear()
        self._key_at.clear()
        self._free.clear()
        self._high_water = 0
        self.clear_dirty()

    def stats(self) -> dict[str, int]:
        span = self.dirty_vertex_range()
        return {
            "capacity": self.capacity,
            "live": self.live_count,
            "free": self.free_slots,
            "vertices": self.vertex_count,
            "dirty_slots": self.dirty_slot_count,
            "dirty_vertices": 0 if span is None else span[1] - span[0],
        }
