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


def test_a_typing_slip_in_the_question_still_finds_the_fact(tmp_path, monkeypatch):
    # Measured: one or two slips per question took recall top-1 from 90.9% to
    # 47.0%; the words of what is remembered put them right (spelling.py).
    from orion_core import spelling
    monkeypatch.setattr(spelling, "_english_words", lambda: {"the": 1, "does": 2, "where": 3,
                                                             "live": 4, "my": 5})
    monkeypatch.setattr(spelling, "_ENGLISH", None)
    made = OrionMemoryMatrix(tmp_path / "mem.db", tmp_path, _Bus())
    try:
        made.save("people", "father_home", "The user's father lives in Swansea")
        made.save("people", "dentist", "The user's dentist is Dr Okafor in Selly Oak")
        made._speller.get()
        keys = [row["key_ref"] for row in made.query("who is my dentsit")]
        assert keys and keys[0] == "dentist"                  # nothing at all, uncorrected
        assert made._speller.correct("where does my fathre live") == "where does my father live"
        assert made._speller.correct("swansea") == "swansea"      # remembered words stay
    finally:
        made.close()


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


# ── meaning-based recall: snapshots, staleness and concurrency ──────────────
#
# Recall by meaning ranks against an in-memory matrix of every stored vector.
# It used to be rebuilt INLINE by whichever question first noticed a change
# (334 ms at 50k rows; during a backfill every indexed batch cost the next
# question a rebuild), and a fact edited or forgotten kept its old vector until
# the indexer came round, so the old meaning could still retrieve it. The
# matrix is now an immutable snapshot built off the question's path and
# swapped whole, and every candidate is checked against its row as it is now.

import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from orion_core import semantic  # noqa: E402
from orion_core.memory import _top_positions  # noqa: E402


class _MeaningStub:
    """A stand-in encoder: each text its own seeded unit vector; paraphrases
    are registered to sit at a chosen cosine from the text they mean."""

    DIM = semantic.MODEL.dim

    def __init__(self):
        self.near: dict[str, np.ndarray] = {}
        self.per_row_s = 0.0
        self.on_encode = None

    def ready(self):
        return True

    def vector(self, text):
        import hashlib
        seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")
        v = np.random.default_rng(seed).standard_normal(self.DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, batch=16):
        hook = self.on_encode
        if hook is not None:
            hook(list(texts))
        if self.per_row_s:
            time.sleep(self.per_row_s * len(texts))
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        return np.vstack([self.vector(t) for t in texts])

    def encode_one(self, text):
        found = self.near.get(text)
        return found if found is not None else self.vector(text)

    def paraphrase(self, question, of, cosine=0.8):
        base = self.vector(of)
        noise = self.vector("noise:" + question)
        noise = noise - noise.dot(base) * base
        noise /= np.linalg.norm(noise)
        v = base + ((1 / cosine ** 2 - 1) ** 0.5) * noise
        self.near[question] = (v / np.linalg.norm(v)).astype(np.float32)
        return question


def _embedded(key, value):
    """What memory embeds for a fact: its label and its words (_embed_text)."""
    return OrionMemoryMatrix._embed_text("intelligence", {"key_ref": key, "value": value})


@pytest.fixture
def meaning(monkeypatch):
    stub = _MeaningStub()
    monkeypatch.setattr(semantic, "ENCODER", stub)
    return stub


@pytest.fixture
def fresh(tmp_path, meaning):
    made = OrionMemoryMatrix(tmp_path / "meaning.db", tmp_path, _Bus())
    yield made
    made.wait_semantic_idle(10)
    made.close()


def _vector_digest(matrix, key):
    row = matrix.conn.execute(
        "SELECT v.digest FROM semantic_vectors v JOIN intelligence i "
        "ON v.source = 'intelligence' AND v.row_id = i.id WHERE i.key_ref = ?",
        (key,)).fetchone()
    return None if row is None else row[0]


def _recording_builds(matrix, monkeypatch):
    """Wrap both ways of making a snapshot (a full read, or the change log
    applied) so a test can see which thread paid for it."""
    built_on: list[str] = []
    full, incremental = matrix._build_snapshot, matrix._apply_changes

    def recorded_full(source):
        built_on.append(threading.current_thread().name)
        return full(source)

    def recorded_incremental(base, changes, generation):
        built_on.append(threading.current_thread().name)
        return incremental(base, changes, generation)

    monkeypatch.setattr(matrix, "_build_snapshot", recorded_full)
    monkeypatch.setattr(matrix, "_apply_changes", recorded_incremental)
    return built_on


def _same_snapshot(left, right):
    return (left.ids.tolist() == right.ids.tolist()
            and np.array_equal(left.matrix, right.matrix)
            and left.digests.tolist() == right.digests.tolist())


@pytest.mark.parametrize("seed", range(4))
def test_partial_selection_matches_a_stable_full_sort(seed):
    """argpartition-based selection must give exactly the full sort's answer,
    ties included, or recall would quietly change with the speed-up."""
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 20, size=500).astype(np.float32)      # many ties
    for k in (0, 1, 7, 8, 32, 499, 500, 600):
        expected = np.argsort(-scores, kind="stable")[:k]
        assert _top_positions(scores, k).tolist() == expected.tolist(), k
    pool = np.flatnonzero(scores >= 12)
    for k in (1, 5, len(pool), len(pool) + 3):
        expected = pool[np.argsort(-scores[pool], kind="stable")[:k]]
        assert _top_positions(scores, k, pool).tolist() == expected.tolist(), k


def test_meaning_recall_is_served_from_a_snapshot_the_indexer_built(fresh, meaning, monkeypatch):
    fact = "Sam reacts badly to ibuprofen tablets"
    fresh.save("personal", "allergy", fact)
    assert fresh.wait_semantic_idle(10)
    assert "intelligence" in fresh._snapshots, "the indexer should publish it"
    built_on = _recording_builds(fresh, monkeypatch)
    question = meaning.paraphrase("qzpain xvwhich", of=_embedded("allergy", fact))
    assert fresh.query(question)[0]["key_ref"] == "allergy"
    assert fresh.relevant(question)[0]["key_ref"] == "allergy"
    assert built_on == [], "a question rebuilt the snapshot"


def test_a_fact_the_request_names_is_volunteered_a_little_below_the_floor(fresh, meaning):
    # Between WORD_SUPPORTED_FLOOR and RELEVANT_FLOOR meaning alone is not
    # enough; the request must also name what the fact is about (its label).
    fact = "The user pays 700 pounds a month in rent, due on the 1st"
    fresh.save("home", "rent", fact)
    assert fresh.wait_semantic_idle(10)
    cosine = (OrionMemoryMatrix.WORD_SUPPORTED_FLOOR + OrionMemoryMatrix.RELEVANT_FLOOR) / 2
    named = meaning.paraphrase("remind me to pay the rent", of=_embedded("rent", fact), cosine=cosine)
    unnamed = meaning.paraphrase("remind me about the first of the month",
                                 of=_embedded("rent", fact), cosine=cosine)
    assert [r["key_ref"] for r in fresh.relevant(named)] == ["rent"]
    assert fresh.relevant(unnamed) == []


def test_an_edited_fact_is_not_found_by_its_old_meaning(fresh, meaning, monkeypatch):
    """The snapshot still holds the old vector until the indexer re-embeds the
    row; the old meaning must not retrieve the new words meanwhile."""
    old, new = "Sam lives in Northgate near the river", "Sam moved to Manchester in August"
    fresh.save("personal", "home", old)
    assert fresh.wait_semantic_idle(10)
    old_q = meaning.paraphrase("qzhome xvold", of=_embedded("home", old))
    assert fresh.query(old_q)[0]["key_ref"] == "home"

    monkeypatch.setattr(fresh, "_index_soon", lambda *_a, **_k: None)     # the indexer lags
    fresh.save("personal", "home", new)
    assert fresh.query(old_q) == []
    assert fresh.relevant(old_q) == []

    fresh.index_semantics()
    new_q = meaning.paraphrase("qzhome xvnew", of=_embedded("home", new))
    assert fresh.query(new_q)[0]["value"] == new
    assert fresh.query(old_q) == []


def test_a_forgotten_fact_is_gone_at_once_even_from_a_stale_snapshot(fresh, meaning, monkeypatch):
    secret = "The spare key is under the blue flowerpot"
    fresh.save("personal", "spare_key", secret)
    fresh.save("personal", "gym", "Sam trains at FitZone early mornings")
    assert fresh.wait_semantic_idle(10)
    question = meaning.paraphrase("qzdoor xvwhere", of=_embedded("spare_key", secret))
    assert fresh.query(question)[0]["key_ref"] == "spare_key"
    stale = fresh._snapshots["intelligence"]

    monkeypatch.setattr(fresh, "_refresh_in_background", lambda source: None)
    assert fresh.forget(category="personal", key_prefix="spare_key") == 1
    assert fresh._snapshots["intelligence"] is stale, "the test needs the stale one"
    assert fresh.query(question) == []
    assert fresh.relevant(question) == []
    assert _vector_digest(fresh, "spare_key") is None
    assert fresh.conn.execute("SELECT COUNT(*) FROM semantic_vectors").fetchone()[0] == 1

    rebuilt = fresh.refresh_semantic_snapshot("intelligence")
    assert len(rebuilt) == 1 and rebuilt.generation == fresh._vector_generation["intelligence"]


def test_a_question_never_pays_for_a_rebuild(fresh, meaning, monkeypatch):
    fresh.save("notes", "boiler", "The boiler was serviced in March")
    assert fresh.wait_semantic_idle(10)
    monkeypatch.setattr(fresh, "_index_soon", lambda *_a, **_k: None)
    late = "The chimney sweep comes every October"
    fresh.save("notes", "chimney", late)
    late_id = fresh.conn.execute(
        "SELECT id FROM intelligence WHERE key_ref='chimney'").fetchone()[0]
    late_text = _embedded("chimney", late)
    fresh._store_vectors("intelligence", [(late_id, late_text)], meaning.encode([late_text]))

    built_on = _recording_builds(fresh, monkeypatch)
    question = meaning.paraphrase("qzsweep xvwhen", of=late_text)
    fresh.query(question)                        # served from the stale snapshot
    assert threading.current_thread().name not in built_on
    assert fresh.wait_semantic_idle(10)
    assert built_on and set(built_on) == {"orion-memory-snapshot"}
    assert fresh.query(question)[0]["key_ref"] == "chimney"


@pytest.mark.parametrize("path", ["full read", "change log"])
def test_a_vector_written_during_a_rebuild_is_published_by_the_next_round(
        fresh, meaning, monkeypatch, path):
    fresh.save("notes", "first", "The first note is about the garden fence")
    assert fresh.wait_semantic_idle(10)
    monkeypatch.setattr(fresh, "_index_soon", lambda *_a, **_k: None)
    rows = {}
    for key, text in (("second", "A second note about the kitchen tap"),
                      ("late", "A late note about the car insurance renewal")):
        fresh.save("notes", key, text)
        rows[key] = (fresh.conn.execute(
            "SELECT id FROM intelligence WHERE key_ref=?", (key,)).fetchone()[0],
            _embedded(key, text))
    # The published snapshot is now behind by one write...
    fresh._store_vectors("intelligence", [rows["second"]], meaning.encode([rows["second"][1]]))
    builds: list[int] = []

    def land_the_late_write():
        if not builds:          # ...and another lands while the first rebuild runs
            fresh._store_vectors("intelligence", [rows["late"]],
                                 meaning.encode([rows["late"][1]]))

    if path == "full read":
        monkeypatch.setattr(fresh, "_pending_changes", lambda *_a: None)
        original = fresh._build_snapshot

        def racing(source):
            snapshot = original(source)
            land_the_late_write()
            builds.append(snapshot.generation)
            return snapshot

        monkeypatch.setattr(fresh, "_build_snapshot", racing)
    else:
        original = fresh._apply_changes

        def racing(base, changes, generation):
            snapshot = original(base, changes, generation)
            land_the_late_write()
            builds.append(snapshot.generation)
            return snapshot

        monkeypatch.setattr(fresh, "_apply_changes", racing)
    final = fresh.refresh_semantic_snapshot("intelligence")
    assert len(builds) == 2, "the write during the rebuild was not picked up"
    assert rows["late"][0] in final.ids.tolist()
    assert final.generation == fresh._vector_generation["intelligence"]


def test_the_change_log_gives_exactly_what_a_full_read_gives(fresh, meaning):
    """Applying logged changes is only a shortcut if it is bit-for-bit the
    snapshot reading the table would have produced: inserts, edits and
    deletes, interleaved, in one refresh and across several."""
    for n in range(12):
        fresh.save("notes", f"n{n:02d}", f"note number {n} about the allotment")
    assert fresh.wait_semantic_idle(10)
    used: list[int] = []
    original = fresh._apply_changes
    fresh._apply_changes = lambda base, changes, gen: (used.append(len(changes)),
                                                       original(base, changes, gen))[1]
    for n in range(0, 12, 3):
        fresh.save("notes", f"n{n:02d}", f"note {n} rewritten about the greenhouse")
    for n in range(12, 16):
        fresh.save("notes", f"n{n:02d}", f"a new note {n} about seed trays")
    fresh.forget(category="notes", key_prefix="n01")
    fresh.forget(category="notes", key_prefix="n05")
    assert fresh.wait_semantic_idle(10)
    published = fresh._snapshots["intelligence"]
    assert used, "the change log was never used"
    assert published.generation == fresh._vector_generation["intelligence"]
    assert _same_snapshot(published, fresh._build_snapshot("intelligence"))


def test_a_log_that_cannot_cover_the_gap_falls_back_to_a_full_read(fresh, meaning, monkeypatch):
    fresh.save("notes", "a", "a note about the shed roof")
    assert fresh.wait_semantic_idle(10)
    monkeypatch.setattr(fresh, "DELTA_LOG_ROWS", 0)       # every change overflows it
    fresh.save("notes", "b", "a note about the water butt")
    assert fresh.wait_semantic_idle(10)
    published = fresh._snapshots["intelligence"]
    assert len(published) == 2
    assert _same_snapshot(published, fresh._build_snapshot("intelligence"))


def test_a_save_embeds_its_own_row_without_scanning_the_table(fresh, meaning, monkeypatch):
    """At 50k facts the full scan held the store's lock for ~250 ms after
    every save — every conversation turn's episode log waited behind it."""
    fresh.save("notes", "first", "The first note is about the garden fence")
    assert fresh.wait_semantic_idle(10)
    scans: list[int] = []
    monkeypatch.setattr(fresh, "index_semantics", lambda *a, **k: scans.append(1) or 0)
    fresh.save("notes", "second", "The second note is about the bird feeder")
    fresh.log_episode("user", "remind me about the bird feeder")
    assert fresh.wait_semantic_idle(10)
    assert scans == []
    count = fresh.conn.execute("SELECT COUNT(*) FROM semantic_vectors").fetchone()[0]
    assert count == 3, "the saved fact and the episode should both be embedded"
    question = meaning.paraphrase(
        "qzbirds xvwhat", of=_embedded("second", "The second note is about the bird feeder"))
    assert fresh.query(question)[0]["key_ref"] == "second"


def test_an_edit_during_encoding_does_not_file_old_words_under_new_text(fresh, meaning, monkeypatch):
    monkeypatch.setattr(fresh, "_index_soon", lambda *_a, **_k: None)
    old, new = "Old plan: fly to Lisbon in May", "New plan: take the train to Paris"
    fresh.save("notes", "plan", old)

    def edit_midway(texts):
        if _embedded("plan", old) in texts:
            meaning.on_encode = None
            fresh.save("notes", "plan", new)

    meaning.on_encode = edit_midway
    fresh.index_semantics()
    assert _vector_digest(fresh, "plan") != semantic.text_digest(_embedded("plan", old))
    fresh.index_semantics()                      # the pass the edit asked for
    assert _vector_digest(fresh, "plan") == semantic.text_digest(_embedded("plan", new))


def _words(key, version):
    syllables = ("ka", "lo", "mi", "su", "ve", "ra", "ti", "no", "pe", "du", "bo", "ge")
    rng = np.random.default_rng(key * 1000 + version)
    return " ".join("".join(syllables[int(i)] for i in rng.integers(0, 12, 3))
                    for _ in range(6))


def _key(number):
    """Fixed width, so forgetting one key by prefix never takes another."""
    return f"fact{number:03d}"


def test_concurrent_edits_deletes_and_questions_never_surface_stale_facts(fresh, meaning):
    """Real threads: two editors, a deleter and an inserter against three readers.

    Every question paraphrases words a fact NO LONGER holds (or a fact already
    forgotten), with no word in common, so any hit could only have come from a
    stale vector. Encoding is slowed (simulated) so the indexer lags and stale
    snapshots are the normal case during the run.
    """
    meaning.per_row_s = 0.0005
    keys = list(range(24))
    for key in keys:
        fresh.save("facts", _key(key), _words(key, 0))
    assert fresh.wait_semantic_idle(20)

    guard = threading.Lock()
    committed = {key: 0 for key in keys}
    gone: dict[int, int] = {}
    errors: list[BaseException] = []
    checked = [0]
    stop = threading.Event()

    def editor(mine):
        while not stop.is_set():
            for key in mine:
                with guard:
                    if key in gone:
                        continue
                    version = committed[key] + 1
                fresh.save("facts", _key(key), _words(key, version))
                with guard:
                    committed[key] = version
            time.sleep(0.002)

    def deleter():
        for key in keys[:6]:
            time.sleep(0.05)
            with guard:
                version = committed[key]
            fresh.forget(category="facts", key_prefix=_key(key))
            with guard:
                gone[key] = version

    def inserter():
        for key in range(100, 130):
            fresh.save("facts", _key(key), _words(key, 0))
            with guard:
                committed[key] = 0
            time.sleep(0.01)

    def reader(seed):
        rng = np.random.default_rng(seed)
        try:
            while not stop.is_set():
                with guard:
                    stale = [(k, int(rng.integers(0, v))) for k, v in committed.items()
                             if v > 0 and k not in gone]
                    forgotten = dict(gone)
                if stale:
                    key, version = stale[int(rng.integers(0, len(stale)))]
                    q = meaning.paraphrase(f"qz{key}x{version} xvold",
                                          of=_embedded(_key(key), _words(key, version)))
                    for rows in (fresh.query(q), fresh.relevant(q)):
                        assert _key(key) not in [r["key_ref"] for r in rows], (key, version)
                    checked[0] += 1
                for key, version in forgotten.items():
                    q = meaning.paraphrase(f"qz{key}x{version} xvgone",
                                          of=_embedded(_key(key), _words(key, version)))
                    for rows in (fresh.query(q), fresh.relevant(q)):
                        assert _key(key) not in [r["key_ref"] for r in rows], ("gone", key)
        except BaseException as exc:          # surfaced below
            errors.append(exc)

    workers = [threading.Thread(target=editor, args=(keys[6:15],)),
               threading.Thread(target=editor, args=(keys[15:] + keys[:6],)),
               threading.Thread(target=deleter), threading.Thread(target=inserter)]
    readers = [threading.Thread(target=reader, args=(s,)) for s in (1, 2, 3)]
    for thread in workers + readers:
        thread.start()
    time.sleep(1.5)
    stop.set()
    for thread in workers + readers:
        thread.join(20)
    assert not errors, errors[0]
    assert checked[0] > 20, "the readers barely ran; the test proved nothing"

    # Once the writers stop, everything settles: the snapshot matches the
    # store, every live fact is found by the meaning of its CURRENT words, and
    # nothing forgotten is left in the vectors or the snapshot.
    assert fresh.wait_semantic_idle(30)
    snapshot = fresh._snapshots["intelligence"]
    assert snapshot.generation == fresh._vector_generation["intelligence"]
    live = {int(r[0]): r[1] for r in fresh.conn.execute("SELECT id, key_ref FROM intelligence")}
    assert set(snapshot.ids.tolist()) == set(live)
    assert set(gone) == set(keys[:6])
    for key, version in committed.items():
        if key in gone:
            continue
        q = meaning.paraphrase(f"qz{key} xvnow", of=_embedded(_key(key), _words(key, version)))
        assert fresh.query(q)[0]["key_ref"] == _key(key)


def test_the_question_cache_survives_concurrent_eviction(monkeypatch):
    """Recall runs on several threads. An unguarded get -> move_to_end raced
    another thread's eviction into a KeyError, silently dropping that question
    to words-only recall."""
    encoder = semantic.SemanticEncoder()
    encoder.CACHE_SIZE = 4
    monkeypatch.setattr(encoder, "ready", lambda: True)
    monkeypatch.setattr(encoder, "encode",
                        lambda texts, batch=16: np.ones((len(texts), 384), np.float32))
    errors: list[BaseException] = []

    def hammer(step):
        try:
            for n in range(4000):
                encoder.encode_one(f"question {(n * step) % 9}")
        except BaseException as exc:
            errors.append(exc)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=hammer, args=(s,)) for s in (1, 2, 3, 5, 7, 11)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
    finally:
        sys.setswitchinterval(previous)
    assert not errors, errors[0]
    assert len(encoder._cache) <= 4
    assert not encoder.encode_one("question 1").flags.writeable, (
        "a cached vector is shared; a caller must not be able to change it")


class _PaddedTokens:
    """encode_batch padded to the longest text, as the real tokenizer is."""

    def encode_batch(self, texts):
        from types import SimpleNamespace
        width = max(len(t.split()) for t in texts) + 2
        return [SimpleNamespace(ids=[1] * width, attention_mask=[1] * width) for _ in texts]


class _CountingSession:
    def __init__(self, on_gpu):
        self.on_gpu, self.rows = on_gpu, []

    def run(self, _names, feed, _options=None):
        rows, width = feed["input_ids"].shape
        self.rows.append(rows)
        return [np.ones((rows, width, semantic.MODEL.dim), np.float32)]


@pytest.fixture
def two_sessions(monkeypatch, tmp_path):
    from orion_core import accelerator
    encoder = semantic.SemanticEncoder()
    cpu, gpu = _CountingSession(False), _CountingSession(True)
    encoder._session, encoder._tokenizer = cpu, _PaddedTokens()
    encoder._input_names = {"input_ids", "attention_mask"}
    monkeypatch.setattr(encoder, "model_dir", lambda: tmp_path)
    built = []
    monkeypatch.setattr(accelerator, "wants_gpu", lambda model: model == "semantic_batch")
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.setattr(accelerator, "create_session",
                        lambda model, path, **kw: built.append((model, kw["threads"])) or gpu)
    return encoder, cpu, gpu, built


def test_questions_stay_on_the_cpu_and_backfill_goes_to_the_gpu(two_sessions):
    """MEASURED (accelerator.POLICY): a question is faster on one CPU thread,
    5.4 ms against 8.9 ms, and runs at its caller's priority there; backfill
    is 4-6x faster on the GPU in batches of 64. The GPU session is built once,
    by the first big batch, from a worker thread."""
    encoder, cpu, gpu, built = two_sessions
    texts = [f"note {i} " + "word " * i for i in range(100)]
    assert encoder.encode(["which painkiller can't I take?"]).shape == (1, semantic.MODEL.dim)
    assert cpu.rows == [1] and gpu.rows == [] and built == []
    vectors = encoder.encode(texts)
    assert vectors.shape == (100, semantic.MODEL.dim)
    assert gpu.rows == [64, 36] and built == [("semantic_batch", 1)]
    encoder.encode(texts[:encoder.GPU_MIN_ROWS])
    assert gpu.rows[-1] == encoder.GPU_MIN_ROWS and len(built) == 1, "built once, then reused"
    encoder.encode(texts[:encoder.GPU_MIN_ROWS - 1])
    assert cpu.rows[-1] == encoder.GPU_MIN_ROWS - 1, "a handful of rows stays on the CPU"


def test_without_a_gpu_or_on_the_gui_thread_batches_stay_on_the_cpu(two_sessions, monkeypatch):
    from orion_core import accelerator
    encoder, cpu, gpu, built = two_sessions
    texts = [f"row {i}" for i in range(40)]
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: True)
    encoder.encode(texts)
    assert cpu.rows == [16, 16, 8] and gpu.rows == [] and built == [], \
        "never built on the GUI thread; the CPU's own batch size"
    monkeypatch.setattr(accelerator, "on_gui_thread", lambda: False)
    monkeypatch.setattr(accelerator, "wants_gpu", lambda model: False)    # ORION_GPU=0
    encoder.encode(texts)
    assert gpu.rows == [] and built == []
    monkeypatch.setattr(accelerator, "wants_gpu", lambda model: True)
    encoder.encode(texts)
    assert gpu.rows == [40]
    gpu.on_gpu = False                       # its GPU failed: accelerator moved it
    encoder.encode(texts)
    assert gpu.rows == [40] and cpu.rows[-3:] == [16, 16, 8]


def test_meaning_recall_is_measured_on_the_request_it_serves(fresh, meaning):
    from orion_core.request_trace import RequestTrace, bound

    fact = "The recycling goes out on Tuesday nights"
    fresh.save("notes", "bins", fact)
    assert fresh.wait_semantic_idle(10)
    trace = RequestTrace("test-1", origin="test")
    with bound(trace):
        fresh.query(meaning.paraphrase("qzbins xvwhen", of=fact))
        fresh.relevant(meaning.paraphrase("qzrubbish xvday", of=fact))
    kinds = [m.fields.get("kind") for m in trace.marks_named("memory_meaning")]
    assert kinds == ["query", "relevant"]
    assert all(m.fields.get("vectors") == 1 for m in trace.marks_named("memory_meaning"))


def test_offline_fact_learning_keeps_only_the_users_own_statements():
    """3 Oct, no text model: "los ang said i have a i'm just chilling so" (an
    overheard fragment) passed the gate on its inner "i have" and was filed as
    a personal fact, because offline the sentence itself is the fact."""
    import asyncio
    from orion_core.fact_memory import FactHarvester
    harvester = FactHarvester(memory=None, router=None)
    assert asyncio.run(harvester._extract("los ang said i have a i'm just chilling so")) == []
    for said in ("My sister Emma lives in Leeds", "So, I'm allergic to penicillin",
                 "Ryan, I work at the hospital on weekends"):
        facts = asyncio.run(harvester._extract(said))
        assert facts and facts[0][1] == said, said


def test_fact_extraction_records_only_what_was_said():
    """'one thing you should know about me is I don't know' became the stored
    fact 'The user does not like poetry'."""
    from orion_core.fact_memory import _EXTRACT_INSTRUCTION
    assert "in so many words" in _EXTRACT_INSTRUCTION and "never infer" in _EXTRACT_INSTRUCTION


@pytest.mark.parametrize("evidence", [None, "I don't know", "My sister likes poetry"])
def test_model_fact_requires_a_durable_quote_from_this_turn(evidence):
    import asyncio
    import json
    from types import SimpleNamespace
    from orion_core.fact_memory import FactHarvester

    async def generate(*args, **kwargs):
        return None, json.dumps({"facts": [{"key": "poetry", "fact": "The user does not like poetry",
                                             "evidence": evidence}]})
    harvester = FactHarvester(memory=None, router=SimpleNamespace(generate_text=generate))
    assert asyncio.run(harvester._extract("My sister Emma lives in Leeds")) == []


def test_explicit_model_fact_keeps_its_source_and_vague_turn_never_calls_model():
    import asyncio
    import json
    from types import SimpleNamespace
    from orion_core.fact_memory import FactHarvester

    said = "My sister Emma lives in Leeds"
    calls = []
    async def generate(*args, **kwargs):
        calls.append(args)
        return None, json.dumps({"facts": [{"key": "sister_home", "fact": "The user's sister Emma lives in Leeds.",
                                             "evidence": said}]})
    harvester = FactHarvester(memory=None, router=SimpleNamespace(generate_text=generate))
    assert asyncio.run(harvester._extract(said)) == [("sister_home", "The user's sister Emma lives in Leeds.", "")]
    assert asyncio.run(harvester._extract("one thing you should know about me is I don't know")) == []
    assert asyncio.run(harvester._extract("I don't know")) == []
    assert len(calls) == 1


# ── what the user said, as recall (query_intelligence + MemoryAgent) ────────
#
# recall_said searched the user's own turns, but nothing called it: the tool
# the system prompt names ("it searches these facts, their records and
# everything they have said to you") only searched facts, and MemoryAgent -
# the object the dispatcher holds - had no way to reach it.

class _NotReady:
    """An encoder that is not loaded: recall runs on words alone."""

    def ready(self):
        return False


class _KnowledgeTools:
    """query_intelligence on a bare object carrying only the memory."""

    def __init__(self, memory):
        self.memory = memory


def _knowledge_tools(memory):
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    class Tools(_KnowledgeTools, KnowledgeDispatchMixin):
        pass
    return Tools(memory)


def _memory_agent(matrix):
    """The dispatcher's MemoryAgent over *matrix*, without its transcript file."""
    from orion_core.memory import MemoryAgent
    agent = MemoryAgent.__new__(MemoryAgent)
    agent.matrix = matrix
    return agent


@pytest.fixture
def said(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "ENCODER", _NotReady())
    made = OrionMemoryMatrix(tmp_path / "said.db", tmp_path, _Bus())
    made.save("personal", "dog", "The user's dog is called Lucy")
    made.log_episode("user", "My budget for the trip to Lisbon is 900 pounds")
    made.log_episode("user", "what is my budget for the Lisbon trip?")
    made.log_episode("assistant", "Your budget for the Lisbon trip is 900 pounds.")
    yield made
    made.close()


def test_recall_said_returns_only_what_the_user_stated(said):
    found = said.recall_said("budget for the Lisbon trip")
    assert [row["content"] for row in found] == ["My budget for the trip to Lisbon is 900 pounds"]
    assert found[0]["role"] == "user" and found[0]["created_at"]


def test_the_memory_agent_reaches_statement_recall(said):
    agent = _memory_agent(said)
    assert agent.recall_said("budget for the Lisbon trip", limit=2) == said.recall_said(
        "budget for the Lisbon trip", limit=2)


def test_query_intelligence_answers_from_the_users_own_words(said):
    import asyncio
    tools = _knowledge_tools(_memory_agent(said))
    text = asyncio.run(tools.query_intelligence({"query": "what was my budget for the Lisbon trip"})).text
    assert "No matching intelligence" not in text, "a statement the user made is an answer"
    assert "900 pounds" in text and "in their own words" in text
    assert "what is my budget" not in text, "the user's question is not evidence"
    assert "Your budget" not in text, "only the user's own turns are statements"


def test_facts_lead_and_statements_follow(said):
    import asyncio
    said.save("travel", "lisbon_trip", "The user is flying to Lisbon in May")
    text = asyncio.run(_knowledge_tools(_memory_agent(said)).query_intelligence(
        {"query": "Lisbon trip budget"})).text
    lines = text.splitlines()
    assert lines[0].startswith("travel/lisbon_trip:"), lines
    header = next(i for i, line in enumerate(lines) if "in their own words" in line)
    assert any("900 pounds" in line for line in lines[header + 1:])


def test_a_filtered_listing_is_facts_only(said):
    import asyncio
    text = asyncio.run(_knowledge_tools(_memory_agent(said)).query_intelligence(
        {"query": "budget", "category": "personal"})).text
    assert "own words" not in text


def test_a_question_nobody_answered_is_still_a_miss(said):
    import asyncio
    text = asyncio.run(_knowledge_tools(_memory_agent(said)).query_intelligence(
        {"query": "dentist appointment"})).text
    assert text == "No matching intelligence records."


def test_a_statement_fault_never_costs_the_facts(said, monkeypatch):
    import asyncio
    agent = _memory_agent(said)

    def broken(*_args, **_kwargs):
        raise RuntimeError("episode index unavailable")
    monkeypatch.setattr(agent, "recall_said", broken, raising=False)
    text = asyncio.run(_knowledge_tools(agent).query_intelligence({"query": "dog called"})).text
    assert text.startswith("personal/dog:")


def test_a_paraphrase_finds_the_statement_by_meaning_and_never_orions_words(fresh, meaning):
    statement = "I go to the gym on Mondays and Wednesdays after work"
    fresh.log_episode("user", statement)
    # ORION repeating the very same words is just as close by meaning, and is
    # still not something the user said.
    fresh.log_episode("assistant", statement)
    assert fresh.wait_semantic_idle(10)
    target = OrionMemoryMatrix._embed_text("episodes", {"content": statement})
    question = meaning.paraphrase("qzwhich xvexercise ywdays", of=target, cosine=0.8)
    found = fresh.recall_said(question)
    assert [(row["role"], row["content"]) for row in found] == [("user", statement)]
    faint = meaning.paraphrase("qzfaint xvguess", of=target, cosine=0.5)
    assert fresh.recall_said(faint) == [], "a meaning-only match below SAID_FLOOR is not an answer"
