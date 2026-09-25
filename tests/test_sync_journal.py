"""Tests for the C2 sync journal + conflict/reconcile core (Mark X.12 §4.1)."""

from __future__ import annotations

from orion_core.bus import OrionBus
from orion_core.memory import OrionMemoryMatrix
from orion_core.sync import (
    JournalEntry,
    SyncJournal,
    is_additive,
    materialise_into,
    reconcile,
    resolve_key,
)


# ── the journal store ──────────────────────────────────────────────────────────

def test_journal_records_with_stable_node_id_and_rising_lamport(tmp_path):
    j = SyncJournal(tmp_path / "j.db")
    assert j.node_id  # a stable node id was minted
    e1 = j.record("notes", "notes", "a", "one")
    e2 = j.record("notes", "notes", "b", "two")
    assert e1.lamport_ts == 1 and e2.lamport_ts == 2
    assert e1.node_id == e2.node_id == j.node_id
    assert [e.payload for e in j.all_entries()] == ["one", "two"]


def test_node_id_persists_across_reopen(tmp_path):
    first = SyncJournal(tmp_path / "j.db")
    node_id = first.node_id
    first.close()
    again = SyncJournal(tmp_path / "j.db")
    assert again.node_id == node_id


def test_apply_remote_is_idempotent(tmp_path):
    src = SyncJournal(tmp_path / "src.db", node_id="src")
    src.record("notes", "notes", "k", "v")
    segment = [e.as_dict() for e in src.segments_since(0)]

    dst = SyncJournal(tmp_path / "dst.db", node_id="dst")
    assert dst.apply_remote(segment) == 1          # applied
    assert dst.apply_remote(segment) == 0          # replay is a no-op
    assert len(dst.all_entries()) == 1


def test_apply_remote_advances_lamport_clock(tmp_path):
    dst = SyncJournal(tmp_path / "dst.db", node_id="dst")
    dst.apply_remote([JournalEntry("peer", 9, "notes", "notes", "k", "h", "v")])
    # A subsequent local write must be causally after the observed remote ts.
    assert dst.record("notes", "notes", "z", "w").lamport_ts == 10


# ── conflict rule ──────────────────────────────────────────────────────────────

def _entry(node, ts, tier, key, payload):
    from orion_core.sync.journal import value_hash
    return JournalEntry(node, ts, tier, tier, key, value_hash(payload), payload)


def test_last_writer_wins_for_ordinary_tiers():
    entries = [_entry("a", 1, "notes", "k", "old"), _entry("b", 2, "notes", "k", "new")]
    assert resolve_key(entries) == ["new"]


def test_additive_tiers_keep_all_distinct_values_newest_last():
    assert is_additive("knowledge") and is_additive("long_term")
    entries = [
        _entry("a", 1, "knowledge", "k", "alpha"),
        _entry("b", 2, "knowledge", "k", "beta"),
        _entry("a", 3, "knowledge", "k", "alpha"),   # re-asserted → dedup, but newest
    ]
    result = resolve_key(entries)
    assert set(result) == {"alpha", "beta"}   # both distinct values kept
    assert result[-1] == "alpha"              # re-asserted at ts3 → newest pointed-to


def test_tiebreak_is_deterministic_on_node_id():
    # Equal Lamport ts → higher node id wins, so every node agrees.
    entries = [_entry("aaa", 5, "notes", "k", "from-a"), _entry("zzz", 5, "notes", "k", "from-z")]
    assert resolve_key(entries) == ["from-z"]


def test_reconcile_resolves_each_key_independently():
    entries = [
        _entry("a", 1, "notes", "x", "x-old"), _entry("b", 2, "notes", "x", "x-new"),
        _entry("a", 1, "knowledge", "y", "y1"), _entry("b", 2, "knowledge", "y", "y2"),
    ]
    resolved = reconcile(entries)
    assert resolved[("notes", "notes", "x")] == ["x-new"]
    assert resolved[("knowledge", "knowledge", "y")] == ["y1", "y2"]


# ── convergence: the whole point ───────────────────────────────────────────────

def test_two_nodes_converge_after_exchanging_segments(tmp_path):
    a = SyncJournal(tmp_path / "a.db", node_id="nodeA")
    b = SyncJournal(tmp_path / "b.db", node_id="nodeB")

    a.record("notes", "notes", "shared", "a-writes")
    a.record("knowledge", "knowledge", "fact", "from-a")
    b.record("notes", "notes", "shared", "b-writes")
    b.record("knowledge", "knowledge", "fact", "from-b")

    # Full bidirectional reconcile.
    a.apply_remote([e.as_dict() for e in b.segments_since(0)])
    b.apply_remote([e.as_dict() for e in a.segments_since(0)])

    # Both nodes now materialise to an IDENTICAL state — convergence.
    assert reconcile(a.all_entries()) == reconcile(b.all_entries())
    both = reconcile(a.all_entries())
    # LWW picked one writer for the note; additive kept both facts.
    assert len(both[("notes", "notes", "shared")]) == 1
    assert sorted(both[("knowledge", "knowledge", "fact")]) == ["from-a", "from-b"]


# ── matrix integration ─────────────────────────────────────────────────────────

def _matrix(tmp_path, name="core.db"):
    return OrionMemoryMatrix(tmp_path / name, tmp_path, OrionBus())


def test_compaction_shrinks_journal_without_changing_reconcile(tmp_path):
    j = SyncJournal(tmp_path / "j.db", node_id="n")
    # LWW key with three versions (only the newest matters after compaction).
    j.record("notes", "notes", "status", "draft")
    j.record("notes", "notes", "status", "review")
    j.record("notes", "notes", "status", "final")
    # Additive key: two distinct values + a duplicate of the first.
    j.record("knowledge", "knowledge", "fact", "aaa")
    j.record("knowledge", "knowledge", "fact", "bbb")
    j.record("knowledge", "knowledge", "fact", "aaa")

    before = reconcile(j.all_entries())
    removed = j.compact()
    after = reconcile(j.all_entries())

    assert removed > 0
    assert after == before                       # reconcile is invariant
    # LWW collapsed to 1 entry; additive kept 2 distinct hashes → 3 total.
    assert len(j.all_entries()) == 3


def test_matrix_save_records_a_journal_entry(tmp_path):
    matrix = _matrix(tmp_path)
    matrix.journal = SyncJournal(tmp_path / "j.db", node_id="desk")
    matrix.save("knowledge", "capital_of_france", "Paris")

    entries = matrix.journal.all_entries()
    assert len(entries) == 1
    e = entries[0]
    assert e.tier == "knowledge" and e.key == "capital_of_france" and e.payload == "Paris"


def test_knowledge_graph_upsert_records_a_knowledge_journal_entry(tmp_path):
    from orion_core.knowledge_graph import KnowledgeGraphEngine
    graph = KnowledgeGraphEngine(bus=OrionBus(), memory=None,
                                 db_path=tmp_path / "graph.db")
    graph.journal = SyncJournal(tmp_path / "j.db", node_id="desk")
    graph.upsert_entity("Neuralink", kind="organisation")

    entries = graph.journal.all_entries()
    assert len(entries) == 1
    e = entries[0]
    assert e.tier == "knowledge"           # graph writes are additive-tier
    assert e.category == "graph_entity"
    assert "Neuralink" in e.payload


def test_materialise_writes_newest_values_without_relooping(tmp_path):
    # A journal populated as if from a peer, applied into a fresh matrix.
    peer = SyncJournal(tmp_path / "peer.db", node_id="peer")
    peer.record("notes", "notes", "goal", "ship v1")
    peer.record("notes", "notes", "goal", "ship v2")   # newer wins

    matrix = _matrix(tmp_path)
    local_journal = SyncJournal(tmp_path / "local.db", node_id="local")
    matrix.journal = local_journal
    # Bring the peer's history into the local journal, then materialise it.
    local_journal.apply_remote([e.as_dict() for e in peer.all_entries()])
    written = materialise_into(local_journal, matrix)

    assert written == 1
    rows = matrix.query("goal")
    assert any(r["value"] == "ship v2" for r in rows)
    # Materialisation used journal=False, so it did NOT create new local entries.
    assert all(e.node_id == "peer" for e in local_journal.all_entries())
