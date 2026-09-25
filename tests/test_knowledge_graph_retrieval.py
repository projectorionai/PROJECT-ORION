"""
Knowledge-graph retrieval — the search index that had never once been used.

``semantic_retrieve`` had no test coverage, and it was broken in four separate
ways at the same time. The worst was invisible:

    SELECT id, source_type, title, text, at, metadata_json
    FROM events_fts JOIN events ON events_fts.rowid = events.rowid

``source_type``, ``title``, ``text`` and ``at`` exist in **both** tables, so
every one of those queries raised ``ambiguous column name`` — straight into a
bare ``except sqlite3.Error: rows = []``. The FTS index was never once consulted
in the entire life of the method. Every call silently fell through to a LIKE
fallback which asked for the whole question as a substring, so it could not
match either.

The other three were the same defects found in ``OrionMemoryMatrix``: implicit
AND between terms, a whole-query LIKE fallback, and no word stemming.

Measured on six ordinary questions about six recorded events: **1/6 -> 6/6.**

The lesson worth keeping is in the exception handler, not the SQL. A swallowed
error turned a total failure into a mild-looking degradation, and no test
noticed for as long as the method existed. It now logs.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.knowledge_graph import KnowledgeGraphEngine, _tokens  # noqa: E402


class _Bus:
    def __init__(self):
        self.messages: list[str] = []

    def __getattr__(self, _name):
        outer = self

        class _Signal:
            def emit(self, *args):
                outer.messages.append(" ".join(str(a) for a in args))

            def connect(self, *a):
                pass
        return _Signal()


class _Memory:
    def __getattr__(self, _name):
        return lambda *a, **k: []


EVENTS = [
    ("note", "Raleigh bicycle",
     "Sam bought a Raleigh bicycle in March for commuting"),
    ("note", "ibuprofen", "Sam is allergic to ibuprofen"),
    ("note", "agency rate",
     "The standard client rate is 450 pounds per short-form video"),
    ("note", "exam", "The neural engineering exam is on the 14th of May"),
    ("note", "supervisor",
     "Dr Harper supervises the dissertation on cortical plasticity"),
    ("note", "rent", "Sam lives in Northgate and pays 700 a month in rent"),
]

QUESTIONS = [
    ("who is my supervisor", "supervisor"),
    ("what is my rent", "rent"),
    ("when is the exam", "exam"),
    ("how much do I charge clients", "agency rate"),
    ("what bicycle did I buy", "Raleigh bicycle"),
    ("am I allergic to anything", "ibuprofen"),
]


@pytest.fixture
def graph(tmp_path):
    bus = _Bus()
    engine = KnowledgeGraphEngine(bus, _Memory(), tmp_path / "kg.db")
    engine._test_bus = bus
    for source, title, text in EVENTS:
        engine.ingest_record(source_type=source, title=title, text=text)
    yield engine
    engine.close()


# ── the defect that hid for the life of the method ──────────────────────────

def test_the_fts_query_is_not_ambiguous(graph):
    """The exact production query must execute. It raised 'ambiguous column
    name' on every call, and the error was swallowed."""
    fts = " OR ".join(sorted(_tokens("who is my supervisor")))
    rows = graph.conn.execute(
        """
        SELECT events.id, events.source_type, events.title,
               events.text, events.at, events.metadata_json
        FROM events_fts JOIN events ON events_fts.rowid = events.rowid
        WHERE events_fts MATCH ? ORDER BY rank LIMIT ?
        """,
        (fts, 5),
    ).fetchall()
    assert rows, "the qualified query returned nothing"


def test_unqualified_columns_would_still_be_ambiguous(graph):
    """Proves the bug was real rather than theoretical — this is the query that
    shipped, and it must still fail, which is why qualification is required."""
    with pytest.raises(sqlite3.OperationalError):
        graph.conn.execute(
            """
            SELECT id, source_type, title, text, at, metadata_json
            FROM events_fts JOIN events ON events_fts.rowid = events.rowid
            WHERE events_fts MATCH ? LIMIT 1
            """,
            ("supervisor",),
        ).fetchall()


def test_a_search_failure_is_reported_not_swallowed(graph):
    """A search index that quietly stops working is worse than one that is
    loudly broken."""
    graph.conn.execute("DROP TABLE events_fts")
    graph.conn.commit()
    graph.semantic_retrieve("who is my supervisor", limit=5)
    assert any("full-text search failed" in m.lower()
               for m in graph._test_bus.messages), (
        "the FTS failure was silent again")


# ── retrieval quality ───────────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", QUESTIONS)
def test_natural_questions_retrieve_the_right_event(graph, question, expected):
    titles = [e.title for e in graph.semantic_retrieve(question, limit=5)]
    assert expected in titles, "%r -> %s" % (question, titles)


def test_overall_recall_is_complete(graph):
    hits = sum(
        expected in [e.title for e in graph.semantic_retrieve(q, limit=5)]
        for q, expected in QUESTIONS)
    assert hits == len(QUESTIONS), "recall %d/%d" % (hits, len(QUESTIONS))


def test_terms_are_ored_not_anded():
    """A question needed every one of its words present in a single event."""
    import inspect
    src = inspect.getsource(KnowledgeGraphEngine.semantic_retrieve)
    assert '" OR ".join' in src
    assert '" ".join(sorted(query_tokens))' not in src


def test_stemming_matches_word_endings(graph):
    """'clients' against an event that says 'client'."""
    titles = [e.title for e in graph.semantic_retrieve("clients", limit=5)]
    assert "agency rate" in titles


def test_a_new_graph_is_created_with_stemming(graph):
    row = graph.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='events_fts'"
    ).fetchone()
    assert "porter" in str(row[0]).lower()


def test_an_unrelated_question_does_not_invent_an_answer(graph):
    found = graph.semantic_retrieve("photosynthesis in tropical orchids", limit=5)
    assert len(found) <= 2, "an unrelated query matched %d events" % len(found)


def test_an_empty_query_is_safe(graph):
    graph.semantic_retrieve("", limit=5)
    graph.semantic_retrieve("   ", limit=5)


def test_a_hostile_query_cannot_crash_retrieval(graph):
    for hostile in ('" OR NEAR/2 *', "'; DROP TABLE events; --", "((((", "a" * 600):
        graph.semantic_retrieve(hostile, limit=3)
    assert graph.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == len(EVENTS)


# ── the fallback ────────────────────────────────────────────────────────────

def test_the_fallback_matches_terms_not_the_whole_question(graph):
    """With the index gone, retrieval must degrade rather than go blank."""
    graph.conn.execute("DROP TABLE events_fts")
    graph.conn.commit()
    titles = [e.title for e in graph.semantic_retrieve("who is my supervisor", limit=5)]
    assert "supervisor" in titles, "the degraded path returned nothing useful"


# ── the migration ───────────────────────────────────────────────────────────

def test_an_existing_unstemmed_graph_is_upgraded_without_data_loss(tmp_path):
    path = tmp_path / "legacy_kg.db"
    bus = _Bus()
    first = KnowledgeGraphEngine(bus, _Memory(), path)
    for source, title, text in EVENTS:
        first.ingest_record(source_type=source, title=title, text=text)
    # force it back to the pre-change tokenizer
    first.conn.executescript(
        "DROP TABLE events_fts;"
        "CREATE VIRTUAL TABLE events_fts USING fts5("
        "title, text, source_type, at, content='events', content_rowid='rowid');")
    first.conn.execute("INSERT INTO events_fts(events_fts) VALUES('rebuild')")
    first.conn.commit()
    first.close()

    reopened = KnowledgeGraphEngine(_Bus(), _Memory(), path)
    try:
        row = reopened.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='events_fts'"
        ).fetchone()
        assert "porter" in str(row[0]).lower(), "the index was not upgraded"
        kept = reopened.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert kept == len(EVENTS), "the migration lost events"
        titles = [e.title for e in reopened.semantic_retrieve("clients", limit=5)]
        assert "agency rate" in titles, "the rebuilt index does not search"
    finally:
        reopened.close()


def test_the_migration_is_idempotent(tmp_path):
    path = tmp_path / "kg.db"
    for _ in range(3):
        engine = KnowledgeGraphEngine(_Bus(), _Memory(), path)
        engine.ingest_record(source_type="note", title="probe", text="a probe event")
        engine.close()
    engine = KnowledgeGraphEngine(_Bus(), _Memory(), path)
    try:
        assert [e.title for e in engine.semantic_retrieve("probe", limit=5)]
    finally:
        engine.close()
