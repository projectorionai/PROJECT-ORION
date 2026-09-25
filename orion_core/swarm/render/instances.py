"""
instances.py — the GPU instance buffer, its slot pool and dirty tracking
(Mark XXII, Phase 1).

This is where the audit's headline defect is actually paid off. The old
renderer disposed and reallocated every geometry and material on every tick;
this buffer allocates once, reuses slots from a free pool, and uploads only
the span that changed. In the steady state — which is almost every frame — it
uploads nothing at all.

Layout is one contiguous float32 array of shape (capacity, INSTANCE_FLOATS),
which is exactly what glBufferSubData wants: no per-frame packing, no Python
object walk, no serialisation. The renderer hands the GPU a memory view.

Three behaviours are worth knowing:

SLOT STABILITY. A node keeps its slot for as long as it exists, so its GPU
instance identity is stable and picking by slot index stays valid between
frames.

SLOT REUSE. Removing a node returns its slot to a free list rather than
compacting the array. Compaction would move every subsequent node to a new
slot and dirty the entire buffer — turning one removal into a full upload,
which is the exact cost this class exists to avoid.

RETIRED SLOTS ARE INVISIBLE, NOT ABSENT. A freed slot has its radius zeroed
so the GPU draws it as degenerate. The draw count stays at the high-water
mark, which keeps the draw call a single glDrawArraysInstanced regardless of
churn.

numpy only — no Qt, no GL. Fully unit-testable without a graphics context.
"""

from __future__ import annotations

import numpy as np

# Per-instance floats, in the exact order the vertex shader reads them:
#   0..2  position xyz
#   3..5  colour rgb
#   6     emissive intensity
#   7     pulse frequency (Hz; 0 = steady)
#   8     radius (0 = retired slot, drawn degenerate)
#   9     selected flag (1.0 when this node is the inspector's subject)
INSTANCE_FLOATS = 10

_OFF_POS = 0
_OFF_COLOUR = 3
_OFF_EMISSIVE = 6
_OFF_PULSE = 7
_OFF_RADIUS = 8
_OFF_SELECTED = 9

_INITIAL_CAPACITY = 256


class InstanceBuffer:
    """A growable, slot-pooled array of per-instance render data."""

    __slots__ = ("_data", "_slot_of", "_id_at", "_free", "_high_water",
                 "_dirty_lo", "_dirty_hi", "_dirty_count")

    def __init__(self, capacity: int = _INITIAL_CAPACITY) -> None:
        capacity = max(1, int(capacity))
        self._data = np.zeros((capacity, INSTANCE_FLOATS), dtype=np.float32)
        self._slot_of: dict[str, int] = {}
        self._id_at: dict[int, str] = {}
        self._free: list[int] = []
        self._high_water = 0
        self._dirty_lo: int | None = None
        self._dirty_hi: int | None = None
        self._dirty_count = 0

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return int(self._data.shape[0])

    @property
    def draw_count(self) -> int:
        """Instances to draw: the high-water mark, not the live node count.
        Retired slots below it are degenerate and cost the GPU nothing."""
        return self._high_water

    @property
    def live_count(self) -> int:
        return len(self._slot_of)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def slot_of(self, node_id: str) -> int | None:
        return self._slot_of.get(node_id)

    def id_at(self, slot: int) -> str | None:
        """Reverse lookup — how a picking hit (which yields a slot index) is
        resolved back to the node the user actually clicked."""
        return self._id_at.get(int(slot))

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._slot_of

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

    def dirty_range(self) -> tuple[int, int] | None:
        """Half-open [start, end) slot span needing upload, or None.

        A range rather than a set of slots because glBufferSubData takes one
        contiguous region; uploading a slightly wider span in one call beats
        issuing many small calls. `dirty_slot_count` lets a caller notice when
        the span has become so sparse that a different strategy is warranted."""
        if self._dirty_lo is None or self._dirty_hi is None:
            return None
        return (self._dirty_lo, self._dirty_hi + 1)

    @property
    def dirty_slot_count(self) -> int:
        return self._dirty_count

    def clear_dirty(self) -> None:
        self._dirty_lo = None
        self._dirty_hi = None
        self._dirty_count = 0

    def mark_all_dirty(self) -> None:
        """Force a full re-upload — used after a GL context loss, where the
        driver's copy of the buffer is gone and CPU-side dirty state no longer
        describes reality."""
        if self._high_water:
            self._dirty_lo = 0
            self._dirty_hi = self._high_water - 1
            self._dirty_count = self._high_water

    # ── capacity ─────────────────────────────────────────────────────────────

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        # Amortised doubling: growth is O(1) per insert overall, so adding a
        # thousand agents does not trigger a thousand reallocations.
        new_capacity = self.capacity
        while new_capacity < needed:
            new_capacity *= 2
        grown = np.zeros((new_capacity, INSTANCE_FLOATS), dtype=np.float32)
        grown[:self._high_water] = self._data[:self._high_water]
        self._data = grown

    def _acquire_slot(self) -> int:
        if self._free:
            # LIFO reuse keeps recently freed (cache-warm) slots in play and
            # keeps the high-water mark from creeping.
            return self._free.pop()
        slot = self._high_water
        self._ensure_capacity(slot + 1)
        self._high_water = slot + 1
        return slot

    # ── mutation ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        node_id: str,
        position: tuple[float, float, float],
        colour: tuple[float, float, float],
        emissive: float,
        pulse_hz: float,
        radius: float,
        selected: bool = False,
    ) -> int:
        """Insert or update one instance. Returns its slot.

        Writes are unconditional rather than compared-then-written: at ten
        floats a memcmp costs about as much as the store, and callers already
        filter by SwarmDelta, so anything reaching here is known to have
        changed."""
        slot = self._slot_of.get(node_id)
        if slot is None:
            slot = self._acquire_slot()
            self._slot_of[node_id] = slot
            self._id_at[slot] = node_id
        row = self._data[slot]
        row[_OFF_POS:_OFF_POS + 3] = position
        row[_OFF_COLOUR:_OFF_COLOUR + 3] = colour
        row[_OFF_EMISSIVE] = emissive
        row[_OFF_PULSE] = pulse_hz
        row[_OFF_RADIUS] = radius
        row[_OFF_SELECTED] = 1.0 if selected else 0.0
        self._touch(slot)
        return slot

    def remove(self, node_id: str) -> bool:
        """Retire a node's slot. Returns False if it was never present."""
        slot = self._slot_of.pop(node_id, None)
        if slot is None:
            return False
        self._id_at.pop(slot, None)
        # Zero the whole row, not just the radius: a stale colour left behind
        # would resurface the moment the slot is reused and written partially.
        self._data[slot] = 0.0
        self._free.append(slot)
        self._touch(slot)
        return True

    def set_selected(self, node_id: str | None) -> None:
        """Move the selection highlight. Only the two affected slots are
        dirtied — selecting a node never re-uploads the graph."""
        for slot in self._selected_slots():
            self._data[slot][_OFF_SELECTED] = 0.0
            self._touch(slot)
        if node_id is None:
            return
        slot = self._slot_of.get(node_id)
        if slot is not None:
            self._data[slot][_OFF_SELECTED] = 1.0
            self._touch(slot)

    def _selected_slots(self) -> list[int]:
        if not self._high_water:
            return []
        column = self._data[:self._high_water, _OFF_SELECTED]
        return [int(i) for i in np.nonzero(column)[0]]

    # ── reads ────────────────────────────────────────────────────────────────

    def view(self) -> np.ndarray:
        """The live region of the buffer — a VIEW, not a copy, so uploading
        costs no allocation."""
        return self._data[:self._high_water]

    def dirty_view(self) -> np.ndarray | None:
        span = self.dirty_range()
        if span is None:
            return None
        return self._data[span[0]:span[1]]

    def row(self, node_id: str) -> np.ndarray | None:
        slot = self._slot_of.get(node_id)
        return None if slot is None else self._data[slot]

    def position_of(self, node_id: str) -> tuple[float, float, float] | None:
        row = self.row(node_id)
        if row is None:
            return None
        return (float(row[0]), float(row[1]), float(row[2]))

    def reset(self) -> None:
        """Drop everything. Capacity is retained deliberately — the array is
        about to be refilled, and releasing it would only force the same
        allocation again moments later."""
        self._data[:self._high_water] = 0.0
        self._slot_of.clear()
        self._id_at.clear()
        self._free.clear()
        self._high_water = 0
        self.clear_dirty()

    def stats(self) -> dict[str, int]:
        """Diagnostics — surfaced in the system-health panel so buffer
        behaviour is observable rather than a matter of faith."""
        span = self.dirty_range()
        return {
            "capacity": self.capacity,
            "draw_count": self.draw_count,
            "live": self.live_count,
            "free": self.free_slots,
            "dirty_slots": self.dirty_slot_count,
            "dirty_span": 0 if span is None else span[1] - span[0],
        }
