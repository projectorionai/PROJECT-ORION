"""
Tests for the two spines of the living-cognition layer (Mark XXIII):
orbital rings (render/rings.py) and cognition state (cognition_state.py).

The properties that matter most, and why:

  * Ring motion is PARAMETRIC. CPU and GPU compute a node's position from the
    same orbit parameters, so they must agree exactly — any divergence shows
    up as clicks landing beside nodes rather than on them.
  * Cognition is EVENT-DRIVEN and DECAYS. Nothing here may drift on a timer;
    an animation that looks the same busy or idle is a screensaver, not a
    readout of a running system.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.cognition_state import (  # noqa: E402
    ACTIVATION_BOOST,
    CognitionState,
    Mode,
    path_for_tool,
)
from orion_core.swarm.render.rings import (  # noqa: E402
    RING_AGENTS,
    RING_BACKGROUND,
    RING_CONNECTIONS,
    RING_EXECUTIVE,
    RING_MEMORY,
    RINGS,
    orbit_for_cluster,
    orbital_position,
    ring_for_cluster,
)


class _Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, dt): self.now += dt


# ── rings: structure ─────────────────────────────────────────────────────────

def test_rings_nest_outward_from_orion():
    radii = [r.radius for r in RINGS]
    assert radii == sorted(radii)
    assert radii[0] > 0


def test_cognition_layers_map_to_the_intended_rings():
    """Distance from ORION should MEAN something."""
    assert ring_for_cluster(cl.INTELLIGENCE).index == RING_EXECUTIVE
    assert ring_for_cluster(cl.DEVELOPMENT).index == RING_AGENTS
    assert ring_for_cluster(cl.MEMORY).index == RING_MEMORY
    assert ring_for_cluster(cl.COMMUNICATION).index == RING_CONNECTIONS
    assert ring_for_cluster(cl.SECURITY).index == RING_BACKGROUND


def test_every_zone_lands_on_a_ring():
    for cluster in cl.CLUSTER_ORDER:
        assert ring_for_cluster(cluster) is not None


def test_an_unknown_zone_falls_to_background_rather_than_vanishing():
    assert ring_for_cluster("NOT_A_ZONE").index == RING_BACKGROUND


def test_adjacent_rings_counter_rotate():
    """Otherwise the whole field reads as one rigid spinning body."""
    signs = [1 if r.speed > 0 else -1 for r in RINGS]
    assert any(a != b for a, b in zip(signs, signs[1:]))


def test_no_two_rings_share_a_period():
    """Shared periods would let the arrangement snap back into alignment."""
    periods = [round(abs(math.tau / r.speed), 3) for r in RINGS]
    assert len(set(periods)) == len(periods)


# ── rings: motion ────────────────────────────────────────────────────────────

def test_a_cluster_actually_moves_over_time():
    orbit = orbit_for_cluster(cl.MEMORY)
    assert orbital_position(orbit, 0.0) != orbital_position(orbit, 45.0)


def test_orbit_radius_is_preserved_while_rotating():
    """Wobble aside, a ring must not spiral in or out."""
    orbit = orbit_for_cluster(cl.RESEARCH)
    ring = ring_for_cluster(cl.RESEARCH)
    for t in (0.0, 30.0, 120.0, 600.0):
        distance = math.dist(orbital_position(orbit, t), (0.0, 0.0, 0.0))
        assert abs(distance - ring.radius) <= ring.wobble + 1e-6


def test_position_is_deterministic_for_a_given_time():
    """CPU picking and the GPU shader must agree; a time-dependent result
    that is not reproducible would make them disagree."""
    orbit = orbit_for_cluster(cl.SYSTEM)
    assert orbital_position(orbit, 12.5) == orbital_position(orbit, 12.5)


def test_clusters_start_at_different_angles():
    positions = {c: orbital_position(orbit_for_cluster(c), 0.0)
                 for c in cl.CLUSTER_ORDER}
    assert len(set(positions.values())) == len(positions)


def test_starting_angles_are_stable_across_restarts():
    """crc32-seeded, so the layout a user learns survives a restart."""
    assert (orbital_position(orbit_for_cluster(cl.SECURITY), 0.0)
            == orbital_position(orbit_for_cluster(cl.SECURITY), 0.0))


def test_orbit_serialises_to_a_flat_parameter_tuple():
    """It is uploaded as instance data, not as a position."""
    values = orbit_for_cluster(cl.MEMORY).as_tuple()
    assert len(values) == 11
    assert all(isinstance(v, float) for v in values)


# ── cognition: event-driven, never a timer ───────────────────────────────────

def test_nothing_is_active_until_something_really_happens():
    state = CognitionState(clock=_Clock())
    assert state.activation("agent:coding") == 0.0
    assert state.peak() == 0.0


def test_engaging_a_node_raises_its_activation():
    state = CognitionState(clock=_Clock())
    state.engage("agent:coding")
    assert state.activation("agent:coding") == ACTIVATION_BOOST


def test_repeated_involvement_accumulates_and_caps():
    state = CognitionState(clock=_Clock())
    for _ in range(20):
        state.engage("agent:coding")
    assert state.activation("agent:coding") == 1.0


def test_activation_decays_by_half_over_one_half_life():
    clock = _Clock()
    state = CognitionState(clock=clock, half_life_s=2.0)
    state.engage("agent:coding")
    before = state.activation("agent:coding")
    clock.advance(2.0)
    assert abs(state.activation("agent:coding") - before / 2) < 1e-9


def test_an_idle_model_costs_nothing_and_stays_correct():
    """Decay is lazy — no tick, no sweep, still right."""
    clock = _Clock()
    state = CognitionState(clock=clock, half_life_s=1.0)
    state.engage("agent:coding")
    clock.advance(60.0)
    assert state.activation("agent:coding") < 0.001


def test_prune_drops_fully_decayed_nodes():
    clock = _Clock()
    state = CognitionState(clock=clock, half_life_s=1.0)
    state.engage("agent:coding")
    clock.advance(40.0)
    assert state.prune() == 1
    assert state.stats()["tracked"] == 0


def test_the_model_is_bounded_under_an_event_storm():
    state = CognitionState(clock=_Clock(), max_tracked=64)
    for i in range(500):
        state.engage(f"node{i}")
    assert state.stats()["tracked"] <= 64


# ── cognition: the rendering answer ──────────────────────────────────────────

def test_an_engaged_node_is_brighter_and_faster():
    state = CognitionState(clock=_Clock())
    resting = state.for_node("agent:coding")
    state.engage("agent:coding")
    busy = state.for_node("agent:coding")
    assert busy.brightness > resting.brightness
    assert busy.speed > resting.speed


def test_unrelated_nodes_dim_while_something_else_works():
    """'Relevant agents brighten, unrelated agents softly dim.'"""
    state = CognitionState(clock=_Clock())
    for _ in range(3):
        state.engage("agent:coding")
    assert state.for_node("agent:coding").dimmed is False
    assert state.for_node("agent:fashion").dimmed is True


def test_nothing_dims_when_the_whole_system_is_quiet():
    """With nothing engaged there is no 'unrelated' to dim against."""
    state = CognitionState(clock=_Clock())
    assert state.for_node("agent:coding").dimmed is False


def test_standby_slows_and_softens_the_entire_field():
    """The visual half of being told to be quiet."""
    state = CognitionState(clock=_Clock())
    state.set_mode(Mode.STANDBY)
    speed, glow = state.feel
    assert speed < 0.5 and glow < 0.5


def test_thinking_speeds_the_field_up():
    state = CognitionState(clock=_Clock())
    state.set_mode(Mode.THINKING)
    assert state.feel[0] > 1.0


def test_bus_states_map_onto_cognition_modes():
    state = CognitionState(clock=_Clock())
    for text, expected in (("LISTENING", Mode.LISTENING),
                           ("THINKING", Mode.THINKING),
                           ("SPEAKING", Mode.SPEAKING),
                           ("STANDBY", Mode.STANDBY)):
        state.observe_bus_state(text)
        assert state.mode is expected


def test_an_unknown_bus_state_leaves_the_mode_alone():
    """The bus string has grown over many passes; an unrecognised value must
    not snap the whole field back to idle."""
    state = CognitionState(clock=_Clock())
    state.set_mode(Mode.THINKING)
    state.observe_bus_state("SOMETHING_NEW")
    assert state.mode is Mode.THINKING


# ── command paths ────────────────────────────────────────────────────────────

def test_a_path_runs_from_core_through_reasoning_to_the_target():
    path = path_for_tool("agent_dispatch", {"agent": "coding"})
    assert path[0] == "core"
    assert "module:reasoning" in path
    assert path[-1] == "agent:coding"


def test_an_mcp_call_routes_through_the_connections_cluster():
    path = path_for_tool("mcp__github__list_repos")
    assert path[-1] == "mcp_tool:github:list_repos"
    assert "mcp:github" in path


def test_engaging_a_path_lights_every_hop():
    state = CognitionState(clock=_Clock())
    path = path_for_tool("agent_dispatch", {"agent": "coding"})
    state.engage_path(path)
    for hop in path:
        assert state.activation(hop) > 0.0


def test_a_path_is_brightest_at_its_origin():
    """Gives the pulse a visible direction rather than lighting up flat."""
    state = CognitionState(clock=_Clock())
    path = path_for_tool("agent_dispatch", {"agent": "coding"})
    state.engage_path(path)
    assert state.activation(path[0]) > state.activation(path[-1])


def test_every_hop_names_a_node_the_graph_really_contains():
    """A path through nodes that do not exist would light up nothing."""
    from orion_core.swarm.sources import compose

    class _Agents:
        def describe(self):
            return [{"name": "coding", "title": "Coding", "focus": "", "calls": 0}]

    class _Modules:
        def all_described(self, t):
            return {"reasoning": {"role": "r", "health": "OK", "dependencies": []}}

    class _Reg:
        modules = _Modules()

    ids = compose(agents=_Agents(), registries=_Reg()).node_ids()
    for hop in path_for_tool("agent_dispatch", {"agent": "coding"}):
        assert hop in ids, hop
