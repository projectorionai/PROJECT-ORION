"""
Command bridges (Mark XXIII) — commands you can watch travel.

The property that makes this a readout rather than an animation: the packet
holds at its destination for exactly as long as the real work takes, and only
comes back when the work really resolved. These tests pin that, and pin that
a failure never fakes a successful return.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm.bridges import (  # noqa: E402
    FLARE_S,
    MAX_HOLD_S,
    MAX_LIVE_BRIDGES,
    RETURN_SPEEDUP,
    SEGMENT_S,
    BridgeLedger,
    Phase,
    deck_route,
)

ROUTE = ["core", "cluster:INTELLIGENCE", "module:reasoning", "agent:research"]


class _Clock:
    def __init__(self) -> None: self.t = 500.0
    def __call__(self) -> float: return self.t
    def advance(self, dt: float) -> None: self.t += dt


def _ledger():
    clock = _Clock()
    return BridgeLedger(clock=clock), clock


def _phase(ledger, clock):
    pulses = ledger.pulses()
    return pulses[0].phase if pulses else None


# ── the journey ──────────────────────────────────────────────────────────────

def test_a_packet_leaves_orion_and_moves_outward():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, label="research")
    first = ledger.pulses()[0]
    assert first.phase is Phase.OUTBOUND
    assert first.source == "core"

    clock.advance(SEGMENT_S * 1.5)
    later = ledger.pulses()[0]
    assert later.source == "cluster:INTELLIGENCE"
    assert later.target == "module:reasoning"


def test_a_longer_route_genuinely_takes_longer_to_cross():
    """The distance a command travels through ORION is real information; a
    six-hop MCP call must not look as direct as a local lookup."""
    ledger, clock = _ledger()
    ledger.launch(["core", "module:a"])
    ledger.launch(ROUTE)
    clock.advance(SEGMENT_S * 1.2)
    phases = {p.bridge_id: p.phase for p in ledger.pulses()}
    assert Phase.HOLDING in phases.values()      # the short one has arrived
    assert Phase.OUTBOUND in phases.values()     # the long one has not


def test_the_packet_holds_at_its_destination_while_the_work_runs():
    """Latency shown as duration — the one place in the interface that does."""
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="call-1")
    clock.advance(SEGMENT_S * 3 + 2.0)
    pulse = ledger.pulses()[0]
    assert pulse.phase is Phase.HOLDING
    assert pulse.source == pulse.target == "agent:research"


def test_a_held_packet_keeps_breathing_so_a_long_wait_reads_as_alive():
    ledger, clock = _ledger()
    ledger.launch(ROUTE)
    clock.advance(SEGMENT_S * 3 + 0.1)
    first = ledger.pulses()[0].intensity
    clock.advance(0.5)
    second = ledger.pulses()[0].intensity
    assert first != second


def test_the_result_comes_back_only_when_the_work_really_resolved():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="call-1")
    clock.advance(SEGMENT_S * 3 + 5.0)
    assert _phase(ledger, clock) is Phase.HOLDING

    ledger.resolve("call-1", ok=True)
    clock.advance(0.02)
    pulse = ledger.pulses()[0]
    assert pulse.phase is Phase.RETURNING
    assert pulse.is_return is True


def test_the_return_travels_back_the_way_it_came():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    clock.advance(SEGMENT_S * 3)
    ledger.resolve("c")
    clock.advance(0.02)
    pulse = ledger.pulses()[0]
    assert pulse.source == "agent:research"
    assert pulse.target == "module:reasoning"


def test_the_bridge_clears_once_the_result_is_home():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    clock.advance(SEGMENT_S * 3)
    ledger.resolve("c")
    clock.advance(SEGMENT_S * 3 / RETURN_SPEEDUP + 0.05)
    assert ledger.pulses() == []
    assert ledger.live() == 0


def test_resolving_before_the_packet_arrives_still_shows_the_full_journey():
    """A tool that returns instantly must not make the packet teleport home —
    the outbound leg is how the user sees where the command went."""
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    ledger.resolve("c")                       # resolved immediately
    clock.advance(SEGMENT_S)
    assert _phase(ledger, clock) is Phase.OUTBOUND


# ── failure ──────────────────────────────────────────────────────────────────

def test_a_failure_stops_where_it_broke_and_never_returns():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    clock.advance(SEGMENT_S * 3)
    ledger.resolve("c", ok=False)
    pulse = ledger.pulses()[0]
    assert pulse.phase is Phase.FAILED
    assert pulse.source == "agent:research"

    clock.advance(FLARE_S / 2)
    assert _phase(ledger, clock) is Phase.FAILED
    clock.advance(FLARE_S)
    assert ledger.pulses() == []


def test_the_failure_flare_fades_rather_than_blinking_out():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    clock.advance(SEGMENT_S * 3)
    ledger.resolve("c", ok=False)
    bright = ledger.pulses()[0].intensity
    clock.advance(FLARE_S * 0.6)
    assert 0.0 < ledger.pulses()[0].intensity < bright


# ── it can never leak ────────────────────────────────────────────────────────

def test_a_command_that_never_resolves_is_eventually_released():
    ledger, clock = _ledger()
    ledger.launch(ROUTE)
    clock.advance(MAX_HOLD_S + 1.0)
    assert ledger.pulses() == []
    assert ledger.live() == 0


def test_concurrent_bridges_are_capped_keeping_the_newest():
    ledger, clock = _ledger()
    for i in range(MAX_LIVE_BRIDGES + 6):
        ledger.launch(["core", f"module:{i}"], label=f"t{i}")
        clock.advance(0.001)
    assert ledger.live() == MAX_LIVE_BRIDGES
    labels = {p.label for p in ledger.pulses()}
    assert f"t{MAX_LIVE_BRIDGES + 5}" in labels
    assert "t0" not in labels


def test_parallel_commands_all_travel_at_once():
    """ORION dispatches tools in parallel; one bridge at a time would misreport
    what he is doing."""
    ledger, _ = _ledger()
    ledger.launch(ROUTE, key="a")
    ledger.launch(["core", "cluster:MEMORY", "memory:working"], key="b")
    assert len({p.bridge_id for p in ledger.pulses()}) == 2


# ── degenerate input ─────────────────────────────────────────────────────────

def test_a_route_with_nowhere_to_go_is_refused():
    ledger, _ = _ledger()
    assert ledger.launch(["core"]) is None
    assert ledger.launch([]) is None
    assert ledger.pulses() == []


def test_a_repeated_node_does_not_stall_the_packet_for_a_whole_hop():
    ledger, clock = _ledger()
    ledger.launch(["core", "core", "module:a", "module:a"])
    clock.advance(SEGMENT_S * 1.1)
    assert _phase(ledger, clock) is Phase.HOLDING


def test_resolving_something_that_never_launched_is_harmless():
    ledger, _ = _ledger()
    assert ledger.resolve("nope") is False
    assert ledger.resolve(999) is False


def test_resolving_twice_does_not_restart_the_journey():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="c")
    clock.advance(SEGMENT_S * 3)
    assert ledger.resolve("c") is True
    assert ledger.resolve("c") is False


# ── what the renderer asks for ───────────────────────────────────────────────

def test_only_the_edge_actually_being_crossed_is_reported_as_active():
    """Brightening the whole route would put us back to a static glow."""
    ledger, clock = _ledger()
    ledger.launch(ROUTE)
    clock.advance(SEGMENT_S * 1.5)
    assert ledger.active_edges() == {("cluster:INTELLIGENCE", "module:reasoning")}


def test_a_held_packet_is_on_no_edge_at_all():
    ledger, clock = _ledger()
    ledger.launch(ROUTE)
    clock.advance(SEGMENT_S * 3 + 0.5)
    assert ledger.active_edges() == set()


def test_stats_report_the_mix_of_phases():
    ledger, clock = _ledger()
    ledger.launch(ROUTE, key="a")
    ledger.launch(["core", "module:x"], key="b")
    clock.advance(SEGMENT_S * 1.2)
    stats = ledger.stats()
    assert stats["live"] == 2
    assert stats["outbound"] == 1
    assert stats["holding"] == 1


def test_clear_empties_the_ledger():
    ledger, _ = _ledger()
    ledger.launch(ROUTE)
    ledger.clear()
    assert ledger.pulses() == []


# ── routes that start at the deck ────────────────────────────────────────────

def test_a_command_issued_from_a_deck_page_starts_at_that_page():
    route = deck_route("AUTOMATION", "workflow", {"name": "nightly"})
    assert route[0] == "page:AUTOMATION"
    assert "core" in route
    assert route[-1] == "workflow:nightly"


def test_the_page_comes_before_orion_because_that_is_the_real_order():
    route = deck_route("MEMORY", "recall", {})
    assert route.index("page:MEMORY") < route.index("core")


def test_a_deck_route_with_no_tool_is_just_the_navigation():
    route = deck_route("LIBRARY")
    assert route[0] == "page:LIBRARY"
    assert len(route) == 2


def test_a_deck_route_with_no_page_falls_back_to_the_tool_route():
    from orion_core.swarm.cognition_state import path_for_tool
    assert deck_route("", "recall", {}) == path_for_tool("recall", {})


def test_a_deck_route_is_launchable():
    ledger, _ = _ledger()
    assert ledger.launch(deck_route("AUTOMATION", "workflow",
                                    {"name": "nightly"})) is not None
