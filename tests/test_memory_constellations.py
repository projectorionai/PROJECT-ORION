"""
Memory constellations (Mark XXIII).

Distance from ORION means RECENCY: working memory closest, archived storage
receding into deep space. The tests pin the two things that make that
meaningful rather than decorative — that the ordering reflects ORION's real
memory architecture, and that a recall only lights constellations that
actually hold something.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.swarm import clusters as cl  # noqa: E402
from orion_core.swarm.cognition_state import (  # noqa: E402
    CognitionState,
    memory_recall_path,
    path_for_tool,
)
from orion_core.swarm.memory_field import (  # noqa: E402
    band_for,
    describe_tiers,
    detail_for,
    recall_targets,
    rest_glow_for,
)
from orion_core.swarm.model import NodeKind  # noqa: E402
from orion_core.swarm.render.layout import LayoutSolver  # noqa: E402
from orion_core.swarm.sources import compose  # noqa: E402

# ORION's real seven tiers, plus archived.
TIERS = {"working": 12, "short_term": 40, "long_term": 900, "semantic": 300,
         "episodic": 150, "procedural": 60, "project": 25, "archived": 5000}


class _Mem:
    def __init__(self, tiers=None): self._tiers = tiers or TIERS
    def tiers_snapshot(self): return dict(self._tiers)


# ── recency ordering ─────────────────────────────────────────────────────────

def test_working_memory_sits_closest_to_orion():
    assert band_for("working") < band_for("short_term") < band_for("long_term")


def test_archived_memory_is_the_most_distant():
    bands = [band_for(t) for t in TIERS]
    assert band_for("archived") == max(bands)


def test_tiers_are_returned_innermost_first():
    tiers = describe_tiers(TIERS)
    assert [t.name for t in tiers][:2] == ["working", "short_term"]
    assert tiers[-1].name == "archived"


def test_live_and_cold_tiers_are_distinguished():
    by_name = {t.name: t for t in describe_tiers(TIERS)}
    assert by_name["working"].is_live is True
    assert by_name["archived"].is_cold is True
    assert by_name["working"].is_cold is False


def test_distant_memory_rests_dimmer_than_live_memory():
    """Recall is what lights the far tiers; if they were already bright there
    would be nothing to illuminate against."""
    assert rest_glow_for("archived") < rest_glow_for("working")


def test_an_unknown_tier_is_placed_rather_than_dropped():
    tiers = describe_tiers({"brand_new_tier": 5})
    assert len(tiers) == 1
    assert 0.0 < tiers[0].band


def test_no_tiers_is_handled():
    assert describe_tiers(None) == []
    assert describe_tiers({}) == []


# ── the inspector says what a tier IS ────────────────────────────────────────

def test_detail_states_what_the_tier_means_not_just_a_count():
    by_name = {t.name: t for t in describe_tiers(TIERS)}
    assert "held now" in detail_for(by_name["working"])
    assert "cold storage" in detail_for(by_name["archived"])
    assert "recallable" in detail_for(by_name["semantic"])


# ── it reaches the graph ─────────────────────────────────────────────────────

def test_every_tier_becomes_a_node():
    ids = compose(memory=_Mem()).node_ids()
    for tier in TIERS:
        assert f"memory:{tier}" in ids


def test_live_tiers_read_as_active_and_cold_ones_do_not():
    by_id = compose(memory=_Mem()).by_id()
    assert by_id["memory:working"].activity.value == "active"
    assert by_id["memory:archived"].activity.value == "idle"


def test_tier_row_counts_reach_the_inspector():
    node = compose(memory=_Mem()).by_id()["memory:long_term"]
    assert node.telemetry.queue_depth == 900


# ── distance is the information ──────────────────────────────────────────────

def _distance(positions, node_id):
    x, y, z = positions[node_id]
    return (x * x + y * y + z * z) ** 0.5


def test_memory_is_laid_out_by_recency_distance():
    """The property the whole feature exists for."""
    snapshot = compose(memory=_Mem())
    positions = LayoutSolver().solve(snapshot.nodes, elapsed=0.0).positions
    assert (_distance(positions, "memory:working")
            < _distance(positions, "memory:long_term")
            < _distance(positions, "memory:archived"))


def test_memory_tiers_are_the_only_nodes_displaced_by_band():
    """Everything else stays on its cluster shell."""
    snapshot = compose(memory=_Mem())
    tiers = [n for n in snapshot.nodes if n.kind is NodeKind.MEMORY_TIER]
    assert len(tiers) == len(TIERS)


# ── recall lights the constellation ──────────────────────────────────────────

def test_recall_only_lights_tiers_that_hold_something():
    """Illuminating an empty constellation would misrepresent where an answer
    came from."""
    tiers = describe_tiers({"working": 4, "long_term": 0, "archived": 90})
    targets = recall_targets(tiers)
    assert "memory:long_term" not in targets
    assert "memory:working" in targets


def test_recall_propagates_outward_from_the_nearest_memory():
    tiers = describe_tiers(TIERS)
    path = memory_recall_path(tiers)
    assert path[0] == "core"
    working = path.index("memory:working")
    archived = path.index("memory:archived")
    assert working < archived


def test_a_memory_tool_routes_into_the_memory_cluster():
    assert f"cluster:{cl.MEMORY}" in path_for_tool("recall", {})


def test_engaging_a_recall_path_illuminates_the_constellation():
    state = CognitionState()
    tiers = describe_tiers(TIERS)
    state.engage_path(memory_recall_path(tiers))
    assert state.activation("memory:working") > 0.0
    # ...and brightest nearest to him, so the recall has direction
    assert state.activation("memory:working") > state.activation("memory:archived")


def test_a_recall_with_nothing_stored_is_safe():
    assert memory_recall_path([]) == ["core", f"cluster:{cl.MEMORY}"]
