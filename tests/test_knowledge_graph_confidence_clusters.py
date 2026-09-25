"""
Tests for the knowledge graph's confidence/cluster/contradiction extensions
(Mark XX design-spec §3/§9, Medium item 9): the backend previously had no
per-relationship confidence field, no clustering, and no contradiction
detection — this closes those three gaps (citations are not a schema
change: they're just relationships with kind="cites", exercised by the
separate literature-ingestion wiring item).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.knowledge_graph import KnowledgeGraphEngine, _connected_components, _has_negation


class _Sig:
    def emit(self, *a):
        pass


class _StubBus:
    def __getattr__(self, name):
        s = _Sig()
        object.__setattr__(self, name, s)
        return s


def _engine(tmp_path) -> KnowledgeGraphEngine:
    return KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")


# ── confidence ────────────────────────────────────────────────────────────

def test_link_entities_records_the_given_confidence(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    kg.link_entities(a.id, b.id, confidence=0.6)
    edge = kg.graph_snapshot()["edges"][0]
    assert edge["confidence"] == 0.6


def test_link_entities_defaults_confidence_to_one(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    kg.link_entities(a.id, b.id)
    edge = kg.graph_snapshot()["edges"][0]
    assert edge["confidence"] == 1.0


def test_confidence_is_clamped_to_the_0_1_range(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    c = kg.upsert_entity("C", "concept")
    kg.link_entities(a.id, b.id, confidence=5.0)
    kg.link_entities(a.id, c.id, confidence=-2.0)
    edges = {(e["source"], e["target"]): e["confidence"] for e in kg.graph_snapshot()["edges"]}
    assert edges[(a.id, b.id)] == 1.0
    assert edges[(a.id, c.id)] == 0.0


# ── contradiction detection ──────────────────────────────────────────────

def test_has_negation_detects_common_negation_cues():
    assert _has_negation("the project is not complete")
    assert _has_negation("this is no longer accurate")
    assert not _has_negation("the project is complete")


def test_contradicting_evidence_is_flagged_on_both_relationships(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("Project X", "project")
    b = kg.upsert_entity("Launch Date", "concept")
    first_contradicted = kg.link_entities(
        a.id, b.id, kind="status", evidence="the launch date is confirmed for March")
    second_contradicted = kg.link_entities(
        a.id, b.id, kind="status", evidence="the launch date is not confirmed anymore")
    assert first_contradicted is False    # nothing to disagree with yet
    assert second_contradicted is True
    edges = kg.graph_snapshot()["edges"]
    assert len(edges) == 2
    assert all(e["contradicts"] for e in edges)


def test_agreeing_evidence_is_not_flagged(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("Project X", "project")
    b = kg.upsert_entity("Launch Date", "concept")
    kg.link_entities(a.id, b.id, kind="status", evidence="the launch date is confirmed")
    second = kg.link_entities(a.id, b.id, kind="status", evidence="the launch date is confirmed again")
    assert second is False
    assert not any(e["contradicts"] for e in kg.graph_snapshot()["edges"])


def test_different_relationship_kinds_are_never_compared_for_contradiction(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("Project X", "project")
    b = kg.upsert_entity("Launch Date", "concept")
    kg.link_entities(a.id, b.id, kind="status", evidence="the launch is confirmed")
    contradicted = kg.link_entities(a.id, b.id, kind="mentioned_with", evidence="not related at all")
    assert contradicted is False


# ── clustering ────────────────────────────────────────────────────────────

def test_connected_components_groups_linked_nodes_and_separates_the_rest():
    edges = [{"source": "a", "target": "b"}, {"source": "c", "target": "d"}]
    clusters = _connected_components(["a", "b", "c", "d", "e"], edges)
    assert clusters["a"] == clusters["b"]
    assert clusters["c"] == clusters["d"]
    assert clusters["a"] != clusters["c"]
    assert clusters["e"] not in (clusters["a"], clusters["c"])   # isolated, own cluster


def test_graph_snapshot_assigns_cluster_ids_to_nodes(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    c = kg.upsert_entity("C", "concept")
    d = kg.upsert_entity("D", "concept")
    kg.link_entities(a.id, b.id)
    kg.link_entities(c.id, d.id)
    snap = kg.graph_snapshot()
    by_id = {n["id"]: n["cluster"] for n in snap["nodes"]}
    assert by_id[a.id] == by_id[b.id]
    assert by_id[c.id] == by_id[d.id]
    assert by_id[a.id] != by_id[c.id]


def test_graph_snapshot_gives_an_isolated_node_its_own_cluster(tmp_path):
    kg = _engine(tmp_path)
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    lonely = kg.upsert_entity("Lonely", "concept")
    kg.link_entities(a.id, b.id)
    snap = kg.graph_snapshot()
    by_id = {n["id"]: n["cluster"] for n in snap["nodes"]}
    assert by_id[lonely.id] != by_id[a.id]


# ── migration of a pre-existing database ─────────────────────────────────

def test_engine_migrates_a_database_created_before_the_confidence_columns_existed(tmp_path):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            evidence TEXT NOT NULL DEFAULT '',
            at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    kg = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=db_path)   # must not raise
    a = kg.upsert_entity("A", "concept")
    b = kg.upsert_entity("B", "concept")
    kg.link_entities(a.id, b.id, confidence=0.8)
    edge = kg.graph_snapshot()["edges"][0]
    assert edge["confidence"] == 0.8
    assert edge["contradicts"] is False
