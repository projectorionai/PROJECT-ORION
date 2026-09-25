"""
Persistent knowledge-graph visualisation:

* KnowledgeGraphEngine.graph_snapshot() returns a renderable {nodes, edges,
  totals} slice weighted by connectivity.
* KnowledgeGraphWidget merges snapshots, preserving the positions of entities
  that persist so the on-screen graph evolves rather than redrawing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ORION_REMOTE_ACCESS", "0")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.knowledge_graph import KnowledgeGraphEngine


class _Sig:
    def emit(self, *a):
        pass


class _StubBus:
    def __getattr__(self, name):
        s = _Sig()
        object.__setattr__(self, name, s)
        return s


@pytest.fixture(scope="module")
def _app():
    # Returning the QApplication keeps a live reference for the whole module so
    # it is not garbage-collected out from under the widgets under test.
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_graph_snapshot_returns_nodes_edges_and_totals(tmp_path):
    kg = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")
    hub = kg.upsert_entity("Hub Project", "project")
    a = kg.upsert_entity("Alice", "person")
    b = kg.upsert_entity("A Concept", "concept")
    kg.link_entities(hub.id, a.id, weight=2.0)
    kg.link_entities(hub.id, b.id, weight=1.0)

    snap = kg.graph_snapshot()
    assert {n["name"] for n in snap["nodes"]} == {"Hub Project", "Alice", "A Concept"}
    assert len(snap["edges"]) == 2
    assert snap["totals"]["relationships"] == 2
    # The hub is the most-connected node.
    assert max(snap["nodes"], key=lambda n: n["degree"])["name"] == "Hub Project"


def test_snapshot_edges_only_between_included_nodes(tmp_path):
    kg = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    kg.link_entities(a.id, b.id, weight=1.0)
    # A self-link and a dangling target must never appear.
    kg.link_entities(a.id, a.id, weight=1.0)      # ignored by link_entities
    snap = kg.graph_snapshot(limit=42)
    for e in snap["edges"]:
        assert e["source"] != e["target"]
        ids = {n["id"] for n in snap["nodes"]}
        assert e["source"] in ids and e["target"] in ids


def test_widget_merges_snapshots_preserving_persistent_nodes(_app):
    from orion_core.gui.knowledge_graph_view import KnowledgeGraphWidget

    w = KnowledgeGraphWidget()
    w.resize(400, 300)
    w.set_snapshot({
        "nodes": [{"id": "1", "name": "A", "kind": "person", "degree": 2},
                  {"id": "2", "name": "B", "kind": "concept", "degree": 1}],
        "edges": [{"source": "1", "target": "2", "weight": 1.0}],
        "totals": {"entities": 2, "relationships": 1, "events": 0},
    })
    assert len(w._nodes) == 2 and len(w._edges) == 1
    node1 = w._nodes["1"]

    # A later snapshot: node 1 persists (same object, updated), 2 drops, 3 appears.
    w.set_snapshot({
        "nodes": [{"id": "1", "name": "A", "kind": "person", "degree": 3},
                  {"id": "3", "name": "C", "kind": "product", "degree": 1}],
        "edges": [],
        "totals": {},
    })
    assert set(w._nodes) == {"1", "3"}
    assert w._nodes["1"] is node1          # position preserved (same object)
    assert w._nodes["1"].degree == 3       # metadata updated in place


# ── confidence / clusters / contradictions (Mark XX design-spec §3/§9) ──────

def test_widget_reads_confidence_contradicts_and_cluster_from_a_snapshot(_app):
    from orion_core.gui.knowledge_graph_view import KnowledgeGraphWidget

    w = KnowledgeGraphWidget()
    w.resize(400, 300)
    w.set_snapshot({
        "nodes": [{"id": "1", "name": "A", "kind": "person", "degree": 2, "cluster": 0},
                  {"id": "2", "name": "B", "kind": "concept", "degree": 1, "cluster": 0}],
        "edges": [{"source": "1", "target": "2", "weight": 1.0,
                   "confidence": 0.4, "contradicts": True}],
        "totals": {},
    })
    assert w._nodes["1"].cluster == 0
    assert w._edges[0] == ("1", "2", 1.0, 0.4, True)


def test_widget_defaults_confidence_cluster_and_contradicts_when_absent(_app):
    # Older-shaped snapshots (no confidence/contradicts/cluster keys) must
    # keep working exactly as before this feature existed.
    from orion_core.gui.knowledge_graph_view import KnowledgeGraphWidget

    w = KnowledgeGraphWidget()
    w.resize(400, 300)
    w.set_snapshot({
        "nodes": [{"id": "1", "name": "A", "kind": "person", "degree": 1},
                  {"id": "2", "name": "B", "kind": "concept", "degree": 1}],
        "edges": [{"source": "1", "target": "2", "weight": 1.0}],
        "totals": {},
    })
    assert w._nodes["1"].cluster == -1
    assert w._edges[0] == ("1", "2", 1.0, 1.0, False)


def test_widget_paints_without_crashing_with_contradictions_and_clusters(_app):
    from orion_core.gui.knowledge_graph_view import KnowledgeGraphWidget

    w = KnowledgeGraphWidget()
    w.resize(400, 300)
    w.set_snapshot({
        "nodes": [
            {"id": "1", "name": "A", "kind": "person", "degree": 2, "cluster": 0},
            {"id": "2", "name": "B", "kind": "concept", "degree": 1, "cluster": 0},
            {"id": "3", "name": "C", "kind": "project", "degree": 1, "cluster": 1},
        ],
        "edges": [
            {"source": "1", "target": "2", "weight": 2.0, "confidence": 0.9, "contradicts": False},
            {"source": "1", "target": "3", "weight": 1.0, "confidence": 0.3, "contradicts": True},
        ],
        "totals": {"entities": 3, "relationships": 2, "events": 0},
    })
    w.grab()   # must paint without raising
