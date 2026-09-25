"""
bridges.py — commands you can watch travel (Mark XXIII).

cognition_state already lights the nodes a command touched. That answers
"which parts were involved" but not "what is happening right now", because a
glow appears everywhere along the route at once: a request still waiting on a
slow MCP server looks identical to one that finished instantly.

A bridge is the missing half. It is a single command IN FLIGHT — a packet of
energy that leaves ORION, crosses the deck page, the cluster, the subsystem,
the agent, the MCP server, arrives, waits there for as long as the real work
actually takes, and then comes back. The return leg is the point: it is the
only thing on screen that distinguishes "done" from "still going", and it
arrives exactly when the tool really resolved.

Three rules keep it a readout rather than an animation:

  * The pulse HOLDS at its destination while the work runs. A command that
    takes eight seconds shows a packet sitting on that node for eight
    seconds. Nothing else in the interface shows latency as duration.

  * A failure does not return. The packet stops where it failed and flares,
    so the point of failure is a position on screen, not a log line.

  * Nothing is invented. A bridge only exists because a tool was really
    dispatched, and it only returns because that tool really resolved.

Positions are expressed as (from_node, to_node, t) rather than coordinates:
this module never learns where anything is. The renderer already knows every
node's position and interpolates — which is what lets bridges keep tracking
their nodes as the cognition rings turn, and lets the whole model be tested
without a scene.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

# How long a packet takes to cross one hop. Fixed per SEGMENT rather than per
# route so a long route genuinely takes longer to traverse — the distance a
# command travels through ORION is real information, and normalising it away
# would make a six-hop MCP call look as direct as a local lookup.
SEGMENT_S = 0.26

# The return leg runs faster: the answer coming back should feel like a
# result arriving, not a second identical journey.
RETURN_SPEEDUP = 1.7

# A failed packet flares for this long at the point of failure, then clears.
FLARE_S = 0.9

# A bridge whose tool never resolves would otherwise hold forever. Real work
# can be slow, so this is generous — it exists to stop a leak, not to lie
# about a timeout.
MAX_HOLD_S = 90.0

# Concurrency cap. ORION dispatches tools in parallel, so several bridges at
# once is normal; hundreds would be unreadable as well as expensive.
MAX_LIVE_BRIDGES = 24


class Phase(str, Enum):
    OUTBOUND = "outbound"     # travelling towards the work
    HOLDING = "holding"       # arrived; the real work is running
    RETURNING = "returning"   # carrying the result back to ORION
    FAILED = "failed"         # stopped where it broke
    DONE = "done"


@dataclass(frozen=True, slots=True)
class Pulse:
    """One packet's position this frame, as a point along an edge."""

    bridge_id: int
    source: str               # node it is travelling from
    target: str               # node it is travelling to
    t: float                  # 0..1 along that segment
    phase: Phase
    intensity: float          # 0..1 — brightness/size
    label: str                # what this command is, for the renderer to show

    @property
    def is_return(self) -> bool:
        return self.phase is Phase.RETURNING


@dataclass(slots=True)
class Bridge:
    """One command travelling through ORION."""

    id: int
    route: tuple[str, ...]
    label: str
    started_at: float
    resolved_at: float | None = None
    ok: bool = True
    _cleared_at: float | None = field(default=None, repr=False)

    @property
    def hops(self) -> int:
        return max(0, len(self.route) - 1)

    @property
    def outbound_s(self) -> float:
        return self.hops * SEGMENT_S

    def phase_at(self, now: float) -> Phase:
        elapsed = now - self.started_at
        if self.resolved_at is not None and not self.ok:
            # A failure stops the packet wherever it had reached, then flares.
            if now - self.resolved_at > FLARE_S:
                return Phase.DONE
            return Phase.FAILED
        if elapsed < self.outbound_s:
            return Phase.OUTBOUND
        if self.resolved_at is None:
            # Still working. Held — unless it has been silent long enough that
            # holding it forever would just be a leak.
            return Phase.HOLDING if elapsed < MAX_HOLD_S else Phase.DONE
        start = max(self.resolved_at, self.started_at + self.outbound_s)
        if now < start:
            return Phase.HOLDING
        if now - start < self.outbound_s / RETURN_SPEEDUP:
            return Phase.RETURNING
        return Phase.DONE


def _segment(route: Sequence[str], progress: float) -> tuple[str, str, float]:
    """Which hop a packet is on, and how far along it."""
    hops = max(1, len(route) - 1)
    clamped = min(max(0.0, progress), 1.0)
    exact = clamped * hops
    index = min(int(exact), hops - 1)
    return (route[index], route[index + 1], exact - index)


class BridgeLedger:
    """Every command currently in flight.

    Lazily evaluated from elapsed time, like traffic.py and cognition_state:
    no timer, no background sweep, and an idle ORION costs nothing at all.
    """

    __slots__ = ("_bridges", "_ids", "_clock", "_by_key")

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock if clock is not None else time.monotonic
        self._bridges: list[Bridge] = []
        self._ids = itertools.count(1)
        # Correlates a resolve() back to its dispatch. Keyed by whatever the
        # caller used (a call id, a tool name) — several bridges can share a
        # tool name, so the newest unresolved one wins.
        self._by_key: dict[str, int] = {}

    # ── launching ────────────────────────────────────────────────────────────

    def launch(self, route: Iterable[str], label: str = "",
               key: str = "", now: float | None = None) -> int | None:
        """Send a packet along *route*. Returns its id, or None if unusable.

        A one-node "route" is rejected rather than drawn as a degenerate
        pulse sitting on a single node, which would read as a stuck command.
        """
        now = self._clock() if now is None else float(now)
        path = tuple(str(n) for n in route if str(n or "").strip())
        # Collapse repeats: path_for_tool can legitimately produce the same
        # node twice (a module that is also its own cluster's member), and a
        # zero-length hop would make the packet stall for a full segment.
        deduped: list[str] = []
        for node in path:
            if not deduped or deduped[-1] != node:
                deduped.append(node)
        if len(deduped) < 2:
            return None

        self._prune(now)
        if len(self._bridges) >= MAX_LIVE_BRIDGES:
            # Drop the oldest rather than refusing the newest: what ORION is
            # doing NOW is what the user is watching for.
            oldest = min(self._bridges, key=lambda b: b.started_at)
            self._bridges.remove(oldest)

        bridge = Bridge(id=next(self._ids), route=tuple(deduped),
                        label=str(label or deduped[-1]), started_at=now)
        self._bridges.append(bridge)
        if key:
            self._by_key[str(key)] = bridge.id
        return bridge.id

    def resolve(self, key_or_id: int | str, ok: bool = True,
                now: float | None = None) -> bool:
        """The work finished. Starts the return leg, or the failure flare."""
        now = self._clock() if now is None else float(now)
        bridge_id = (key_or_id if isinstance(key_or_id, int)
                     else self._by_key.get(str(key_or_id)))
        if bridge_id is None:
            return False
        for bridge in self._bridges:
            if bridge.id == bridge_id and bridge.resolved_at is None:
                bridge.resolved_at = now
                bridge.ok = bool(ok)
                return True
        return False

    # ── reading ──────────────────────────────────────────────────────────────

    def _prune(self, now: float) -> None:
        self._bridges = [b for b in self._bridges if b.phase_at(now) is not Phase.DONE]
        live = {b.id for b in self._bridges}
        self._by_key = {k: v for k, v in self._by_key.items() if v in live}

    def pulses(self, now: float | None = None) -> list[Pulse]:
        """Every packet's position this frame."""
        now = self._clock() if now is None else float(now)
        self._prune(now)
        out: list[Pulse] = []
        for bridge in self._bridges:
            phase = bridge.phase_at(now)
            elapsed = now - bridge.started_at
            if phase is Phase.OUTBOUND:
                progress = elapsed / max(1e-6, bridge.outbound_s)
                source, target, t = _segment(bridge.route, progress)
                intensity = 1.0
            elif phase is Phase.HOLDING:
                # Parked on the destination. A slow tool is visible as a packet
                # that has stopped moving, and the pulse breathes so a long
                # wait still reads as alive rather than frozen.
                source = target = bridge.route[-1]
                t = 1.0
                waited = elapsed - bridge.outbound_s
                intensity = 0.62 + 0.38 * abs(((waited * 1.4) % 2.0) - 1.0)
            elif phase is Phase.RETURNING:
                start = max(bridge.resolved_at or now,
                            bridge.started_at + bridge.outbound_s)
                span = max(1e-6, bridge.outbound_s / RETURN_SPEEDUP)
                progress = (now - start) / span
                # Reversed route: the answer comes back the way it went, which
                # is what makes the round trip legible as one event.
                source, target, t = _segment(tuple(reversed(bridge.route)), progress)
                intensity = 0.85
            elif phase is Phase.FAILED:
                source = target = bridge.route[-1]
                t = 1.0
                age = now - (bridge.resolved_at or now)
                intensity = max(0.0, 1.0 - age / FLARE_S)
            else:
                continue
            out.append(Pulse(bridge_id=bridge.id, source=source, target=target,
                             t=float(t), phase=phase,
                             intensity=float(min(1.0, max(0.0, intensity))),
                             label=bridge.label))
        return out

    def active_edges(self, now: float | None = None) -> set[tuple[str, str]]:
        """Edges carrying a packet right now, so the renderer can brighten the
        line the command is actually crossing rather than the whole route."""
        return {(p.source, p.target) for p in self.pulses(now)
                if p.source != p.target}

    def live(self, now: float | None = None) -> int:
        now = self._clock() if now is None else float(now)
        self._prune(now)
        return len(self._bridges)

    def stats(self, now: float | None = None) -> dict[str, Any]:
        pulses = self.pulses(now)
        return {
            "live": len(self._bridges),
            "outbound": sum(1 for p in pulses if p.phase is Phase.OUTBOUND),
            "holding": sum(1 for p in pulses if p.phase is Phase.HOLDING),
            "returning": sum(1 for p in pulses if p.phase is Phase.RETURNING),
            "failed": sum(1 for p in pulses if p.phase is Phase.FAILED),
        }

    def clear(self) -> None:
        self._bridges.clear()
        self._by_key.clear()


# ── the routes the deck itself produces ──────────────────────────────────────

def deck_route(page: str, tool: str = "", args: dict | None = None) -> list[str]:
    """The full path a command takes when it starts at a Command Deck page.

    The deck migrated into the swarm, so a page is a node like any other and a
    command issued from it genuinely begins there. Prefixing the tool's own
    route with the page is what turns "something happened" into "you did this,
    and here is where it went"."""
    from .cognition_state import path_for_tool
    from .clusters import normalise_cluster
    from .sources import cluster_id

    page_name = str(page or "").strip()
    tail = path_for_tool(tool, args or {}) if tool else []
    if not page_name:
        return tail
    head = [f"page:{page_name}", cluster_id(normalise_cluster(page_name))]
    # The tool route starts at core; splicing it on after the page's own
    # cluster gives deck -> zone -> ORION -> ... -> destination, which is the
    # order the request really travels.
    return head + list(tail)
