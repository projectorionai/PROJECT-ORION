"""
Memory recall — ORION being able to answer questions about what he was told.

This file exists because of a measured, shipped defect. FTS5 treats
space-separated terms as an implicit **AND**, so the query builder's
``' '.join(tokens)`` required every word of a question to appear in one stored
row. Against 34 ordinary questions about 16 stored facts:

    shipped behaviour   11.8% recall
    after the fix       88.2% recall on the HELD-OUT half, answer ranked first

Only exact single keywords worked. ORION could be told something and then be
unable to retrieve it when asked normally — memory was effectively write-only.

Three changes, each measured separately:

* **OR instead of implicit AND**, ranked by FTS5's BM25 (``ORDER BY rank``).
  This is the ordinary way to run a bag-of-words query. 33% -> 83% on the first
  probe set.
* **Porter stemming** so "supervisor" matches "supervises". 83% -> 92%.
* **Stopword removal**, which does not change which rows match but sharply cuts
  ranking noise: "when is my exam" returned two unrelated rows until "is" and
  "my" were dropped.

Prefix matching was measured too and made **no difference at all**, so it was
not added.

The same query builder serves ``intelligence_fts`` and ``episodes_fts``, so
conversation recall had the identical defect and the same fix repairs it.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from data.memory_recall_cases import (  # noqa: E402
    CASES,
    FACTS,
    dev_cases,
    holdout_cases,
)
from orion_core.memory import OrionMemoryMatrix  # noqa: E402


class _Bus:
    def __getattr__(self, _name):
        class _Signal:
            def emit(self, *a):
                pass

            def connect(self, *a):
                pass
        return _Signal()


@pytest.fixture
def matrix(tmp_path):
    made = OrionMemoryMatrix(tmp_path / "mem.db", tmp_path, _Bus())
    for category, key, value in FACTS:
        made.save(category, key, value)
    yield made
    made.close()


def _score(matrix, cases):
    hits = first = 0
    missed: list[str] = []
    for question, expected in cases:
        keys = [row["key_ref"] for row in matrix.query(question, limit=8)]
        if expected in keys:
            hits += 1
            first += keys[0] == expected
        else:
            missed.append(question)
    n = max(1, len(cases))
    return hits / n, first / n, missed


# ── the evaluation set ───────────────────────────────────────────────────────

def test_the_split_is_deterministic_and_balanced():
    dev, hold = dev_cases(), holdout_cases()
    assert len(dev) + len(hold) == len(CASES)
    assert abs(len(dev) - len(hold)) <= 1
    assert not (set(dev) & set(hold))
    assert dev_cases() == dev


def test_every_case_names_a_stored_fact():
    keys = {key for _c, key, _v in FACTS}
    unknown = sorted({k for _q, k in CASES if k not in keys})
    assert not unknown, "cases reference facts that are never stored: %s" % unknown


def test_the_questions_are_not_just_the_stored_words():
    """If the questions echoed the stored sentences this would prove nothing —
    the defect was precisely that people do not speak in schema vocabulary.

    The handful of bare single-keyword probes are excluded: those are there on
    purpose, as the guard that the queries which DID work before still work.
    """
    stored = " ".join(v for _c, _k, v in FACTS).lower()
    phrases = [q for q, _ in CASES if len(q.split()) > 2]
    echoes = [q for q in phrases if q.lower() in stored]
    assert not echoes, "these questions are quoted from the stored text: %s" % echoes
    assert len(phrases) >= 25, "too few real questions to mean anything"


# ── recall: the property that was broken ─────────────────────────────────────

def test_recall_on_the_development_half(matrix):
    recall, first, missed = _score(matrix, dev_cases())
    assert recall >= 0.90, "dev recall %.1f%%; missed %s" % (recall * 100, missed)
    assert first >= 0.85, "the answer is found but ranked poorly (%.1f%% first)" % (first * 100)


def test_recall_on_the_held_out_half(matrix):
    """The honest score — never inspected while tuning retrieval."""
    recall, first, missed = _score(matrix, holdout_cases())
    assert recall >= 0.85, (
        "HOLDOUT recall %.1f%% — ORION cannot answer questions about what he "
        "was told: %s" % (recall * 100, missed))
    assert first >= 0.80, (
        "HOLDOUT: the right fact is retrieved but ranked below noise (%.1f%%)"
        % (first * 100))


def test_a_natural_question_returns_something(matrix):
    """The shipped behaviour returned literally nothing for these."""
    for question in ("who is my supervisor", "what is my rent",
                     "how much do I charge clients", "when is my exam"):
        assert matrix.query(question, limit=5), (
            "%r retrieved nothing at all" % question)


def test_single_keyword_lookups_never_regress(matrix):
    """These were the only queries that worked before. They must keep working."""
    for keyword, expected in (("ibuprofen", "allergy"), ("PrintWorks", "supplier"),
                              ("dissertation", "supervisor")):
        keys = [r["key_ref"] for r in matrix.query(keyword, limit=5)]
        assert keys and keys[0] == expected, (keyword, keys)


def test_stemming_matches_word_endings(matrix):
    """'supervisor' against a fact that says 'supervises' — worth 9 points."""
    keys = [r["key_ref"] for r in matrix.query("supervisor", limit=5)]
    assert "supervisor" in keys


def test_an_unrelated_question_does_not_invent_an_answer(matrix):
    """OR semantics must not turn memory into a random-fact generator."""
    rows = matrix.query("photosynthesis in tropical orchids", limit=5)
    assert len(rows) <= 2, "an unrelated query matched %d facts" % len(rows)


# ── the query builder ────────────────────────────────────────────────────────

def test_terms_are_joined_with_or_not_implicit_and():
    built = OrionMemoryMatrix._sanitise_fts_query("who is my supervisor today")
    assert " OR " in built or built.count(" ") == 0
    assert " AND " not in built


def test_stopwords_are_dropped():
    assert OrionMemoryMatrix._sanitise_fts_query("what is my rent") == "rent"


def test_a_query_of_only_stopwords_still_searches():
    """Matching on a weak term beats returning nothing."""
    built = OrionMemoryMatrix._sanitise_fts_query("what is this")
    assert built, "an all-stopword question produced an empty query"


def test_a_command_verb_is_not_treated_as_a_stopword():
    """tool_resolver drops 'set'/'show'/'run' because a tool query is an
    instruction. Here they are often the point."""
    built = OrionMemoryMatrix._sanitise_fts_query("what did I set the rate to")
    assert "set" in built and "rate" in built


def test_an_empty_query_builds_nothing():
    assert OrionMemoryMatrix._sanitise_fts_query("") == ""
    assert OrionMemoryMatrix._sanitise_fts_query("   ") == ""


# ── injection safety must not have been weakened ────────────────────────────

@pytest.mark.parametrize("hostile", [
    'value: "secret" OR 1',
    "supervisor NEAR/5 password",
    'rent" OR "a',
    "exam AND (rent OR flat)",
    "^start",
    "col:value",
    "*",
    "NOT allergy",
])
def test_fts5_operators_are_still_neutralised(matrix, hostile):
    """The sanitiser strips FTS5 syntax so a query cannot become an expression.
    Building an OR query must not have reopened that."""
    built = OrionMemoryMatrix._sanitise_fts_query(hostile)
    for operator in ('"', "*", "^", "(", ")", ":"):
        assert operator not in built, (hostile, built)
    assert "NEAR" not in built.upper()
    matrix.query(hostile, limit=3)          # and it must not raise


def test_a_hostile_query_cannot_crash_retrieval(matrix):
    for hostile in ("'; DROP TABLE intelligence; --", "\\", "((((", "a" * 500):
        matrix.query(hostile, limit=3)      # must not raise


# ── the LIKE fallback ────────────────────────────────────────────────────────

def test_the_fallback_matches_terms_not_the_whole_question():
    """It used LIKE '%<entire query>%', which could essentially never match
    anything a person actually asked."""
    terms = OrionMemoryMatrix._like_terms("who is my supervisor")
    assert terms == ["supervisor"]


def test_the_fallback_is_bounded():
    terms = OrionMemoryMatrix._like_terms(" ".join("term%d" % i for i in range(40)))
    assert len(terms) <= 6, "an unbounded OR chain per term would scan badly"


def test_the_fallback_still_works_when_fts_is_unavailable(matrix):
    """If the index is missing, memory must degrade rather than go blank."""
    matrix.conn.execute("DROP TABLE intelligence_fts")
    matrix.conn.commit()
    rows = matrix.query("who is my supervisor", limit=5)
    assert any(r["key_ref"] == "supervisor" for r in rows), (
        "with no FTS index memory returned nothing")


# ── the migration ────────────────────────────────────────────────────────────

def _tokenizer_of(conn, table):
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return "porter" if row and "porter" in str(row[0]).lower() else "default"


def test_a_new_database_is_created_with_stemming(matrix):
    assert _tokenizer_of(matrix.conn, "intelligence_fts") == "porter"
    assert _tokenizer_of(matrix.conn, "episodes_fts") == "porter"


def test_an_existing_unstemmed_database_is_upgraded_without_data_loss(tmp_path):
    path = tmp_path / "legacy.db"
    # build a database the way it looked before this change
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL,
            key_ref TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE VIRTUAL TABLE intelligence_fts USING fts5(
            category, key_ref, value, updated_at,
            content='intelligence', content_rowid='id');
    """)
    for i, (category, key, value) in enumerate(FACTS, 1):
        conn.execute("INSERT INTO intelligence VALUES (?,?,?,?,?)",
                     (i, category, key, value, "2026-01-01"))
    conn.execute("INSERT INTO intelligence_fts(intelligence_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()

    opened = OrionMemoryMatrix(path, tmp_path, _Bus())
    try:
        assert _tokenizer_of(opened.conn, "intelligence_fts") == "porter"
        kept = opened.conn.execute("SELECT COUNT(*) FROM intelligence").fetchone()[0]
        assert kept == len(FACTS), "the migration lost rows"
        keys = [r["key_ref"] for r in opened.query("who is my supervisor", limit=5)]
        assert "supervisor" in keys, "the rebuilt index does not search"
    finally:
        opened.close()


def test_the_migration_is_idempotent(tmp_path):
    path = tmp_path / "m.db"
    for _ in range(3):
        opened = OrionMemoryMatrix(path, tmp_path, _Bus())
        opened.save("personal", "probe", "a stored probe value")
        opened.close()
    opened = OrionMemoryMatrix(path, tmp_path, _Bus())
    try:
        rows = opened.conn.execute(
            "SELECT COUNT(*) FROM intelligence WHERE key_ref='probe'").fetchone()[0]
        assert rows == 1
        assert opened.query("probe", limit=3)
    finally:
        opened.close()


def test_a_failed_migration_does_not_stop_startup(tmp_path, monkeypatch):
    """An unstemmed index still works; it just recalls a little less. That must
    never be a reason ORION will not start.

    The failure is forced through the table spec rather than by patching
    sqlite3.Connection, which is an immutable type.
    """
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE intelligence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL,
            key_ref TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE VIRTUAL TABLE intelligence_fts USING fts5(
            category, key_ref, value, updated_at,
            content='intelligence', content_rowid='id');
    """)
    conn.execute("INSERT INTO intelligence VALUES (1,'personal','probe','still usable','x')")
    conn.execute("INSERT INTO intelligence_fts(intelligence_fts) VALUES('rebuild')")
    conn.commit()
    conn.close()

    # a spec whose CREATE cannot succeed -> the rebuild raises inside the migration
    monkeypatch.setattr(
        OrionMemoryMatrix, "_FTS_TABLES",
        (("intelligence_fts", "no_such_content_table", "not, real, columns"),))

    opened = OrionMemoryMatrix(path, tmp_path, _Bus())
    try:
        assert opened.query("probe", limit=3), "startup degraded into an unusable store"
    finally:
        opened.close()


# ── conversation recall shares the fix ───────────────────────────────────────

def test_episode_search_benefits_from_the_same_fix(matrix):
    """The same builder serves episodes_fts, so conversation recall had the
    identical defect."""
    matrix.log_episode("user", "I decided to switch the dissertation topic to "
                                  "cortical plasticity after speaking to Harper")
    matrix.log_episode("assistant", "Noted — plasticity it is.")
    found = matrix.recall_episodes("what did I decide about my dissertation", limit=5)
    assert found, "conversation recall returned nothing for a natural question"
    assert any("plasticity" in r["content"] for r in found)


def test_episode_search_survives_a_hostile_query(matrix):
    matrix.log_episode("user", "ordinary line")
    matrix.recall_episodes('" OR NEAR/2 *', limit=3)      # must not raise


# ── a broken index must be visible, not merely mediocre ─────────────────────

class _RecordingBus:
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


def test_a_broken_search_index_is_reported(tmp_path):
    """The knowledge graph hid a totally broken FTS query for the life of its
    retrieval method because the error went into a bare `except: pass` and the
    LIKE fallback made the failure look like mediocrity."""
    bus = _RecordingBus()
    opened = OrionMemoryMatrix(tmp_path / "m.db", tmp_path, bus)
    try:
        opened.save("personal", "probe", "a stored probe value")
        opened.conn.execute("DROP TABLE intelligence_fts")
        opened.conn.commit()
        bus.messages.clear()
        rows = opened.query("probe", limit=3)
        assert rows, "memory must still answer from the fallback"
        assert any("full-text search failed" in m.lower() for m in bus.messages), (
            "the index is broken and nothing said so")
    finally:
        opened.close()


def test_the_fault_is_reported_once_not_on_every_query(tmp_path):
    """A fault that fires on every query would drown the log it appears in."""
    bus = _RecordingBus()
    opened = OrionMemoryMatrix(tmp_path / "m.db", tmp_path, bus)
    try:
        opened.save("personal", "probe", "a stored probe value")
        opened.conn.execute("DROP TABLE intelligence_fts")
        opened.conn.commit()
        bus.messages.clear()
        for _ in range(12):
            opened.query("probe", limit=3)
        complaints = [m for m in bus.messages if "full-text search failed" in m.lower()]
        assert len(complaints) == 1, "reported %d times" % len(complaints)
    finally:
        opened.close()


def test_episodes_report_their_own_index_separately(tmp_path):
    bus = _RecordingBus()
    opened = OrionMemoryMatrix(tmp_path / "m.db", tmp_path, bus)
    try:
        opened.log_episode("user", "an ordinary line about the dissertation")
        opened.conn.execute("DROP TABLE episodes_fts")
        opened.conn.commit()
        bus.messages.clear()
        assert opened.recall_episodes("dissertation", limit=3)
        assert any("episodes_fts" in m for m in bus.messages)
    finally:
        opened.close()
