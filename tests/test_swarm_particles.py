"""
Tests for the neural particle field (Mark XXIII).

Two properties carry the design and are pinned hardest here:

  * The field is EMITTED ONCE and animated entirely in the vertex shader, so
    the CPU-side data must contain orbit PARAMETERS (centre/radius/axis/
    speed/phase) and no positions. If positions ever appear here it means
    motion moved back onto the CPU, which at several thousand particles is
    the "no CPU spikes / 60 FPS" requirement lost.
  * Density is DERIVED from real system state, so a bigger ORION is visibly
    denser and nothing in the field is decorative filler.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.model import NodeKind, SwarmNode  # noqa: E402
from orion_core.swarm.render.layout import LayoutSolver  # noqa: E402
from orion_core.swarm.render.particles import (  # noqa: E402
    LAYER_COSMIC,
    LAYER_HALO,
    LAYER_MICRO,
    LAYER_SYNAPSE,
    PARTICLE_FLOATS,
    build_field,
    estimate_count,
)
from orion_core.swarm.sources import compose  # noqa: E402


def _graph(extra: int = 0):
    nodes = [SwarmNode(id="core", kind=NodeKind.CORE, label="ORION",
                       cluster=cl.INTELLIGENCE),
             SwarmNode(id="cluster:MEMORY", kind=NodeKind.SUBSYSTEM,
                       label="Memory", cluster=cl.MEMORY)]
    nodes += [SwarmNode(id=f"module:m{i}", kind=NodeKind.SUBSYSTEM,
                        label=f"m{i}", cluster=cl.MEMORY)
              for i in range(3 + extra)]
    positions = LayoutSolver().solve(nodes).positions
    return nodes, positions


# ── shape and layering ───────────────────────────────────────────────────────

def test_the_field_has_every_layer():
    nodes, positions = _graph()
    field = build_field(nodes, positions)
    for layer in (LAYER_HALO, LAYER_MICRO, LAYER_SYNAPSE, LAYER_COSMIC):
        assert field.layers[layer][1] > 0, layer


def test_layer_spans_tile_the_buffer_exactly():
    """Spans are used to draw layers separately; a gap or overlap would drop
    or double-draw particles."""
    nodes, positions = _graph()
    field = build_field(nodes, positions)
    spans = sorted(field.layers.values())
    cursor = 0
    for start, count in spans:
        assert start == cursor
        cursor += count
    assert cursor == field.count


def test_the_buffer_is_upload_ready():
    nodes, positions = _graph()
    data = build_field(nodes, positions).data
    assert data.dtype == np.float32
    assert data.flags["C_CONTIGUOUS"]
    assert data.shape[1] == PARTICLE_FLOATS


# ── motion lives in the shader ───────────────────────────────────────────────

def test_particles_carry_orbit_parameters_not_positions():
    """The whole performance argument: the CPU emits parameters once and the
    shader derives position from them plus time."""
    nodes, positions = _graph()
    data = build_field(nodes, positions).data
    radius = data[:, 3]
    speed = data[:, 4]
    axis = data[:, 6:9]
    assert (radius > 0).all()                       # every particle orbits
    assert np.abs(speed).min() > 0.0                # every particle moves
    assert np.allclose(np.linalg.norm(axis, axis=1), 1.0, atol=1e-4)


def test_orbits_run_both_directions():
    """A field that all spins one way reads as a rigid carousel."""
    nodes, positions = _graph(extra=20)
    speed = build_field(nodes, positions).data[:, 4]
    assert (speed > 0).any() and (speed < 0).any()


def test_phases_are_spread_so_nothing_pulses_in_lockstep():
    nodes, positions = _graph(extra=20)
    phase = build_field(nodes, positions).data[:, 5]
    assert phase.max() - phase.min() > 3.0


# ── driven by real state ─────────────────────────────────────────────────────

def test_a_bigger_graph_produces_a_denser_field():
    small, small_pos = _graph()
    big, big_pos = _graph(extra=40)
    assert build_field(big, big_pos).count > build_field(small, small_pos).count


def test_density_scales_the_whole_field():
    nodes, positions = _graph(extra=10)
    full = build_field(nodes, positions, density=1.0).count
    half = build_field(nodes, positions, density=0.5).count
    assert half < full


def test_zero_density_yields_an_empty_field_without_crashing():
    """Adaptive quality's floor — it must thin to nothing, not explode."""
    nodes, positions = _graph()
    field = build_field(nodes, positions, density=0.0)
    assert field.count == 0
    assert field.data.shape == (0, PARTICLE_FLOATS)


def test_the_core_emits_nothing():
    """ORION's face occupies the centre; a cloud there would sit on top of
    him instead of radiating from him."""
    nodes, positions = _graph()
    field = build_field(nodes, positions)
    centres = field.data[:, 0:3]
    at_origin = np.all(np.abs(centres) < 1e-6, axis=1)
    # only the cosmic layer is centred on the origin
    cosmic_start, cosmic_count = field.layers[LAYER_COSMIC]
    assert at_origin.sum() == cosmic_count


def test_particles_inherit_their_nodes_colour():
    nodes, positions = _graph()
    gold = (1.0, 0.8, 0.2)
    field = build_field(nodes, positions,
                        colours={n.id: gold for n in nodes})
    halo_start, halo_count = field.layers[LAYER_HALO]
    halo = field.data[halo_start:halo_start + halo_count]
    # jittered around the source colour, but unmistakably it
    assert halo[:, 9].mean() > halo[:, 11].mean()


# ── determinism ──────────────────────────────────────────────────────────────

def test_the_field_is_identical_across_runs():
    """crc32-seeded, never Python's randomised hash — the field must not be
    different on every launch."""
    nodes, positions = _graph()
    assert np.array_equal(build_field(nodes, positions).data,
                          build_field(nodes, positions).data)


# ── robustness ───────────────────────────────────────────────────────────────

def test_a_node_with_no_position_is_skipped():
    nodes, positions = _graph()
    positions.pop("module:m1", None)
    assert build_field(nodes, positions).count > 0     # must not raise


def test_an_empty_graph_still_yields_the_background():
    field = build_field([], {})
    assert field.layers[LAYER_COSMIC][1] > 0
    assert field.layers[LAYER_HALO][1] == 0


def test_estimate_matches_the_real_count_closely():
    """Adaptive quality picks a density from the estimate before paying to
    emit, so it must not be wildly wrong."""
    nodes, positions = _graph(extra=30)
    real = build_field(nodes, positions).count
    # core emits nothing, so the estimate is one node's worth high
    assert abs(estimate_count(len(nodes)) - real) < estimate_count(2)


def test_a_realistic_graph_reaches_thousands_of_particles():
    """The brief's actual target: ~85 nodes should yield several thousand."""
    snapshot = compose()
    positions = LayoutSolver().solve(snapshot.nodes).positions
    field = build_field(list(snapshot.nodes), positions)
    assert field.count > 1000
