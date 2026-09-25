"""
cognition_state.py — what ORION is actually doing, as visual state (Mark XXIII).

This is the single source of truth behind every "living" behaviour in the
swarm: thinking mode, command paths, memory illumination, and the whole
event-driven animation language. Those were specified as four features but
they are one system — each is the same question asked differently ("which
parts of ORION are engaged right now, and how strongly?"), so they share one
model rather than four parallel ones that could disagree about whether the
research agent is busy.

The hard rule this exists to enforce: NOTHING here is a timer or a loop.
Every value is raised by a real runtime event — a dispatched tool, a bus
state change, an agent invocation, a memory query — and then DECAYS. An
animation that runs on its own schedule would look identical on a busy system
and an idle one, which is the difference between a readout and a screensaver.

Decay is exponential and evaluated lazily from elapsed time (the same
approach traffic.py uses), so an idle model costs nothing: no timer, no
background sweep, and reading it is always correct without having been
ticked.

Pure Python — no Qt, no GL, injectable clock, so the whole cognition
language is unit-testable without a display.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable


class Mode(str, Enum):
    """ORION's overall posture. Drives the global feel of the field —
    velocity, brightness, colour temperature — while per-node activation
    below drives which specific parts light up."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    RESEARCHING = "researching"
    CODING = "coding"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    STANDBY = "standby"
    ERROR = "error"


# How the whole field behaves in each mode: (speed multiplier, glow
# multiplier). Standby deliberately slows AND dims — the user asked for quiet,
# and a field still racing while he is meant to be resting would contradict
# the state he put ORION in.
_MODE_FEEL: dict[Mode, tuple[float, float]] = {
    Mode.IDLE:        (1.00, 0.85),
    Mode.LISTENING:   (1.15, 1.00),
    Mode.THINKING:    (1.60, 1.30),
    Mode.RESEARCHING: (1.45, 1.20),
    Mode.CODING:      (1.40, 1.18),
    Mode.SPEAKING:    (1.25, 1.35),
    Mode.EXECUTING:   (1.70, 1.40),
    Mode.STANDBY:     (0.35, 0.45),
    Mode.ERROR:       (1.30, 1.25),
}

# Seconds for an untouched node's activation to halve. Short enough that a
# finished task visibly subsides, long enough that a burst of rapid calls
# reads as sustained work rather than flicker.
ACTIVATION_HALF_LIFE_S = 2.2

# One event's contribution. Below 1.0 so repeated involvement accumulates and
# a single call is a flicker rather than a flare.
ACTIVATION_BOOST = 0.42

# Below this an activation is treated as zero and dropped, so the map does
# not grow without bound over a long session.
_FLOOR = 0.02

# A node this far below the busiest one is dimmed rather than merely
# un-brightened — "unrelated agents softly dim" needs a reference point.
_DIM_THRESHOLD = 0.12


@dataclass(frozen=True, slots=True)
class NodeCognition:
    """How a single node should currently be drawn."""

    activation: float      # 0..1, how engaged it is right now
    brightness: float      # multiplier on its emissive
    speed: float           # multiplier on its particle orbit rate
    dimmed: bool           # actively suppressed because something else is busy


class CognitionState:
    """Live cognition, driven by real events."""

    __slots__ = ("_activation", "_stamps", "_mode", "_mode_at", "_clock",
                 "_half_life", "_events", "_max_tracked")

    def __init__(self, clock: Callable[[], float] | None = None,
                 half_life_s: float = ACTIVATION_HALF_LIFE_S,
                 max_tracked: int = 2048) -> None:
        self._activation: dict[str, float] = {}
        self._stamps: dict[str, float] = {}
        self._mode = Mode.IDLE
        self._clock = clock if clock is not None else time.monotonic
        self._mode_at = self._clock()
        self._half_life = max(1e-3, float(half_life_s))
        self._events = 0
        self._max_tracked = max(32, int(max_tracked))

    # ── mode ─────────────────────────────────────────────────────────────────

    @property
    def mode(self) -> Mode:
        return self._mode

    def set_mode(self, mode: Mode | str) -> None:
        """Set the posture. An unrecognised value leaves the current mode
        alone rather than resetting to idle — a stray string must not blank
        out a real cognitive state."""
        if isinstance(mode, Mode):
            resolved = mode
        else:
            # NOT str(mode): for a (str, Enum) member Python 3.11+ returns
            # "Mode.THINKING" rather than "thinking", so every enum-typed
            # call silently fell through to IDLE and the field never changed
            # posture at all.
            try:
                resolved = Mode(str(mode).strip().lower())
            except ValueError:
                return
        if resolved is not self._mode:
            self._mode = resolved
            self._mode_at = self._clock()

    def observe_bus_state(self, state: str) -> None:
        """Map ORION's own broadcast state onto a cognition mode.

        Deliberately tolerant: the bus carries a free-form string that has
        grown over many passes, so an unrecognised value leaves the mode
        alone rather than snapping the whole field back to idle."""
        text = str(state or "").strip().upper()
        mapping = {
            "LISTENING": Mode.LISTENING,
            "THINKING": Mode.THINKING,
            "PROCESSING": Mode.THINKING,
            "SPEAKING": Mode.SPEAKING,
            "EXECUTING": Mode.EXECUTING,
            "RESEARCHING": Mode.RESEARCHING,
            "STANDBY": Mode.STANDBY,
            "IDLE": Mode.IDLE,
            "ERROR": Mode.ERROR,
        }
        if text in mapping:
            self.set_mode(mapping[text])

    @property
    def feel(self) -> tuple[float, float]:
        """(speed multiplier, glow multiplier) for the whole field."""
        return _MODE_FEEL.get(self._mode, (1.0, 1.0))

    # ── activation ───────────────────────────────────────────────────────────

    def _decayed(self, node_id: str, now: float) -> float:
        level = self._activation.get(node_id)
        if level is None:
            return 0.0
        elapsed = max(0.0, now - self._stamps.get(node_id, now))
        if elapsed == 0.0:
            return level
        return level * math.pow(0.5, elapsed / self._half_life)

    def engage(self, node_id: str, weight: float = 1.0) -> None:
        """Note that a node was genuinely involved in something."""
        node_id = str(node_id or "")
        if not node_id:
            return
        now = self._clock()
        level = self._decayed(node_id, now) + ACTIVATION_BOOST * max(0.0, float(weight))
        self._activation[node_id] = min(1.0, level)
        self._stamps[node_id] = now
        self._events += 1
        if len(self._activation) > self._max_tracked:
            self._evict(now)

    def engage_path(self, node_ids: Iterable[str],
                    decay_along_path: float = 0.82) -> None:
        """Light a whole reasoning/command route at once.

        Weight falls along the path so the origin reads brightest and the
        pulse has a visible direction — the difference between "these nodes
        are busy" and "the command travelled THIS way"."""
        weight = 1.0
        for node_id in node_ids:
            self.engage(node_id, weight)
            weight *= max(0.0, min(1.0, decay_along_path))

    def _evict(self, now: float) -> None:
        ranked = sorted(self._activation, key=lambda k: self._decayed(k, now))
        for key in ranked[:len(ranked) // 4 or 1]:
            self._activation.pop(key, None)
            self._stamps.pop(key, None)

    def activation(self, node_id: str) -> float:
        return self._decayed(str(node_id or ""), self._clock())

    def peak(self) -> float:
        """The busiest activation right now — the reference other nodes are
        dimmed against."""
        now = self._clock()
        if not self._activation:
            return 0.0
        return max(self._decayed(k, now) for k in self._activation)

    # ── the rendering answer ─────────────────────────────────────────────────

    def for_node(self, node_id: str) -> NodeCognition:
        """How this node should be drawn right now."""
        speed_mul, glow_mul = self.feel
        level = self.activation(node_id)
        peak = self.peak()
        # Only dim when something ELSE is clearly busy. With nothing engaged
        # the whole field sits at its resting brightness rather than every
        # node dimming itself against a peak of zero.
        dimmed = bool(peak > _DIM_THRESHOLD and level < peak * 0.35)
        brightness = glow_mul * (1.0 + 1.5 * level)
        if dimmed:
            brightness *= 0.45
        return NodeCognition(
            activation=level,
            brightness=brightness,
            speed=speed_mul * (1.0 + 1.2 * level),
            dimmed=dimmed,
        )

    def active_nodes(self, floor: float = 0.08) -> dict[str, float]:
        now = self._clock()
        return {k: round(v, 4) for k in self._activation
                if (v := self._decayed(k, now)) >= floor}

    def prune(self) -> int:
        now = self._clock()
        dead = [k for k in self._activation if self._decayed(k, now) < _FLOOR]
        for key in dead:
            self._activation.pop(key, None)
            self._stamps.pop(key, None)
        return len(dead)

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self._mode.value,
            "events": self._events,
            "tracked": len(self._activation),
            "active": len(self.active_nodes()),
            "peak": round(self.peak(), 3),
            "speed_multiplier": round(self.feel[0], 2),
            "glow_multiplier": round(self.feel[1], 2),
        }

    def reset(self) -> None:
        self._activation.clear()
        self._stamps.clear()
        self._mode = Mode.IDLE


def path_for_tool(tool: str, args: dict[str, Any] | None = None) -> list[str]:
    """The route a dispatched tool call travels through ORION.

    This is what makes a command physically visible: the pulse runs core ->
    reasoning -> the responsible cluster -> the specific node, so the user can
    follow the decision rather than just seeing an endpoint light up. Node ids
    match the ones sources.compose emits, so every hop resolves to something
    really on screen.
    """
    from .clusters import (
        AUTOMATION, COMMUNICATION, INTELLIGENCE, MEMORY, cluster_of_agent)
    from .sources import cluster_id

    args = args or {}
    name = str(tool or "")
    route = ["core", cluster_id(INTELLIGENCE), "module:reasoning"]

    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) >= 2 and parts[1]:
            route.append(cluster_id(COMMUNICATION))
            route.append(f"mcp:{parts[1]}")
            if len(parts) >= 3 and parts[2]:
                route.append(f"mcp_tool:{parts[1]}:{parts[2]}")
        return route

    if name in ("agent_dispatch", "reason") and args.get("agent"):
        agent = str(args["agent"])
        route.append(cluster_id(cluster_of_agent(agent)))
        route.append(f"agent:{agent}")
        return route

    if name == "workflow" and args.get("name"):
        route.append(cluster_id(AUTOMATION))
        route.append(f"workflow:{args['name']}")
        return route

    if name in ("memory", "remember", "recall", "search_memory"):
        # A recall reaches OUTWARD through the constellation: working memory
        # first, then further back. The caller lights this path in order, so
        # the user watches him reach rather than seeing every tier flash.
        route.append(cluster_id(MEMORY))
        return route

    route.append(f"module:{name}" if name else "module:dispatcher")
    return route


def memory_recall_path(tiers: list, query: str = "") -> list[str]:
    """The route a memory search takes through the constellation.

    Separated from path_for_tool because a recall's shape depends on what
    ORION actually HOLDS — only tiers with rows in them are lit, since
    illuminating an empty constellation would misrepresent where an answer
    came from."""
    from .clusters import MEMORY as _MEMORY
    from .memory_field import recall_targets
    from .sources import cluster_id as _cluster_id

    return ["core", _cluster_id(_MEMORY)] + recall_targets(tiers, query)
