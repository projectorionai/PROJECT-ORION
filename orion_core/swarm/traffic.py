"""
traffic.py — edges that mean something (Mark XXII, Phase 3).

The brief is specific: edges should represent actual communication — memory
writes, task delegation, prompt routing, MCP calls, tool execution — and
traffic intensity should reflect real workload. The old graph drew one
anonymous edge type at a fixed opacity, so "core owns this agent" and "this
subsystem is hammering that one right now" looked identical.

This ledger turns real dispatch events into a 0..1 intensity per edge.

Decay is exponential with a half-life, not a sliding window of timestamps.
A window means storing every event and re-scanning it on each sample —
unbounded memory and O(events) work. A decayed level is one float per edge,
updated in O(1), and it produces the behaviour that actually matters visually:
a burst flares immediately and fades smoothly rather than dropping off a cliff
the moment an event ages out of the window.

Decay is applied LAZILY, computed from the elapsed time at read, so an idle
ledger costs nothing at all — there is no timer here and no background sweep.

Pure Python: no Qt, no I/O, monotonic clock injectable for deterministic tests.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Iterable

from .clusters import AUTOMATION, COMMUNICATION, MEMORY, SYSTEM, cluster_of_agent
from .model import EdgeKind, SwarmEdge

# Seconds for an idle edge's intensity to halve. ~2.5 s reads as "recent
# activity" to a human watching: fast enough that a finished burst visibly
# subsides, slow enough that a steady trickle of calls stays lit.
DEFAULT_HALF_LIFE_S = 2.5

# One event's contribution. Below 1.0 so a single call is a flicker rather
# than a full-brightness alarm, and repeated calls visibly accumulate.
DEFAULT_BOOST = 0.34

# Levels below this are treated as zero and dropped, so edges that stopped
# carrying traffic release their slot instead of accumulating forever.
_PRUNE_FLOOR = 0.01

EdgeKey = tuple[str, str, str]


class TrafficLedger:
    """Decayed activity level per edge, driven by real events."""

    __slots__ = ("_levels", "_stamps", "_half_life", "_boost", "_clock",
                 "_events", "_max_edges")

    def __init__(self, half_life_s: float = DEFAULT_HALF_LIFE_S,
                 boost: float = DEFAULT_BOOST,
                 clock: Callable[[], float] | None = None,
                 max_edges: int = 4096) -> None:
        self._levels: dict[EdgeKey, float] = {}
        self._stamps: dict[EdgeKey, float] = {}
        self._half_life = max(1e-3, float(half_life_s))
        self._boost = max(0.0, min(1.0, float(boost)))
        self._clock = clock if clock is not None else time.monotonic
        self._events = 0
        # Hard ceiling so a pathological event storm cannot grow this without
        # bound; the quietest edges are evicted first.
        self._max_edges = max(16, int(max_edges))

    # ── recording ────────────────────────────────────────────────────────────

    def record(self, source: str, target: str,
               kind: EdgeKind = EdgeKind.TOOL_CALL, weight: float = 1.0) -> None:
        """Note one real communication event."""
        if not source or not target:
            return
        key: EdgeKey = (str(source), str(target), kind.value)
        now = self._clock()
        level = self._decayed(key, now) + self._boost * max(0.0, float(weight))
        self._levels[key] = min(1.0, level)
        self._stamps[key] = now
        self._events += 1
        if len(self._levels) > self._max_edges:
            self._evict(now)

    def _decayed(self, key: EdgeKey, now: float) -> float:
        """Current level for *key*, decayed from when it was last touched."""
        level = self._levels.get(key)
        if level is None:
            return 0.0
        elapsed = max(0.0, now - self._stamps.get(key, now))
        if elapsed == 0.0:
            return level
        return level * math.pow(0.5, elapsed / self._half_life)

    def _evict(self, now: float) -> None:
        ranked = sorted(self._levels, key=lambda k: self._decayed(k, now))
        for key in ranked[:len(ranked) // 4 or 1]:
            self._levels.pop(key, None)
            self._stamps.pop(key, None)

    # ── reading ──────────────────────────────────────────────────────────────

    def intensity(self, source: str, target: str,
                  kind: EdgeKind = EdgeKind.TOOL_CALL) -> float:
        return self._decayed((str(source), str(target), kind.value), self._clock())

    def prune(self) -> int:
        """Drop edges that have decayed to nothing. Returns how many went.

        Optional — reading is already correct without it — but it keeps the
        dicts small during long sessions."""
        now = self._clock()
        dead = [k for k in self._levels if self._decayed(k, now) < _PRUNE_FLOOR]
        for key in dead:
            self._levels.pop(key, None)
            self._stamps.pop(key, None)
        return len(dead)

    def apply(self, edges: Iterable[SwarmEdge]) -> tuple[SwarmEdge, ...]:
        """Return *edges* with `traffic` set from measured activity.

        An edge with no recorded traffic keeps 0.0 rather than being dropped:
        structural edges must stay visible at rest, or the graph would appear
        to dissolve whenever ORION goes quiet. Unchanged edges are returned as
        the SAME object so the diff engine sees no change and skips the upload."""
        now = self._clock()
        out: list[SwarmEdge] = []
        for edge in edges:
            level = self._decayed((edge.source, edge.target, edge.kind.value), now)
            if abs(level - edge.traffic) < 1e-6:
                out.append(edge)
            else:
                out.append(SwarmEdge(source=edge.source, target=edge.target,
                                     kind=edge.kind, traffic=round(level, 4)))
        return tuple(out)

    def active_edges(self, floor: float = 0.05) -> dict[EdgeKey, float]:
        now = self._clock()
        return {k: round(v, 4) for k in self._levels
                if (v := self._decayed(k, now)) >= floor}

    def busiest(self, limit: int = 5) -> list[tuple[EdgeKey, float]]:
        """The most active edges right now — used by the inspector and the
        diagnostics panel to name what ORION is currently doing."""
        active = self.active_edges(floor=0.0)
        return sorted(active.items(), key=lambda kv: kv[1], reverse=True)[:limit]

    def stats(self) -> dict[str, Any]:
        return {
            "events": self._events,
            "tracked_edges": len(self._levels),
            "active_edges": len(self.active_edges()),
            "half_life_s": self._half_life,
        }

    def reset(self) -> None:
        self._levels.clear()
        self._stamps.clear()


def edge_for_tool_call(tool: str, args: dict[str, Any] | None = None
                       ) -> tuple[str, str, EdgeKind]:
    """Map a dispatched tool call onto the edge it represents.

    Returns an edge that ACTUALLY EXISTS in the graph sources.compose builds —
    source, target AND kind must all match, because TrafficLedger.apply only
    lights edges it can find by that exact triple. An earlier version routed
    agent work to ``core -> agent:<name>``, which looks reasonable but is not
    the topology: agents hang off their CLUSTER, so every recorded event was
    silently discarded and no edge ever lit up.

    The structural edge is the communication path, so traffic is carried on it
    rather than by synthesising a parallel transient edge — one line between
    two nodes, sometimes busy, is easier to read than two overlapping ones.
    """
    args = args or {}
    name = str(tool or "")

    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) >= 3 and parts[1]:
            # mcp:<server> -> mcp_tool:<server>:<tool>, as built by _mcp()
            return (f"mcp:{parts[1]}", f"mcp_tool:{parts[1]}:{parts[2]}",
                    EdgeKind.MCP_REQUEST)
        if len(parts) >= 2 and parts[1]:
            return (f"cluster:{COMMUNICATION}", f"mcp:{parts[1]}",
                    EdgeKind.MEMBERSHIP)

    if name in ("agent_dispatch", "reason") and args.get("agent"):
        agent = str(args["agent"])
        # cluster:<zone> -> agent:<name>, as built by _agents()
        return (f"cluster:{cluster_of_agent(agent)}", f"agent:{agent}",
                EdgeKind.DELEGATION)

    if name == "workflow" and args.get("name"):
        return (f"cluster:{AUTOMATION}", f"workflow:{args['name']}",
                EdgeKind.MEMBERSHIP)

    if name in ("memory", "remember", "recall"):
        return ("core", f"cluster:{MEMORY}", EdgeKind.MEMBERSHIP)

    return ("core", f"cluster:{SYSTEM}", EdgeKind.MEMBERSHIP)
