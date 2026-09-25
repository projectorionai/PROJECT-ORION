"""
Tests for the unified ingestion engine (Command Deck priority #2).

Covers the directive's non-negotiables: SHA256 fingerprinting gates
reprocessing (skip unchanged / re-index changed / resume partial), chunk
deduplication never double-writes identical content, the embedding cache reuses
identical chunks, folders and zips fan out, unsupported types are reported (not
raised), and offline search ranks stored chunks.  Every test uses throw-away
databases under tmp_path.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.ingestion import (
    IngestionEngine,
    KnowledgeDeduplicationService,
    _hash_embed,
    _cosine,
)


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _FakeEvent:
    def __init__(self, ids):
        self.entity_ids = ids


class _FakeGraph:
    """Records ingest_record calls and returns fixed entity ids."""

    def __init__(self):
        self.calls = []

    def ingest_record(self, source_type, title, text, metadata=None, at=""):
        self.calls.append((source_type, title, text, metadata))
        return _FakeEvent(["ent_alpha", "ent_beta"])

    def entity_name(self, eid):
        return {"ent_alpha": "Alpha", "ent_beta": "Beta"}.get(eid, eid)


class _FakeMemory:
    def __init__(self):
        self.records = {}

    def remember(self, tier, key, value, project=""):
        self.records[key] = value
        return "ok"


@pytest.fixture
def engine(tmp_path):
    eng = IngestionEngine(
        bus=_StubBus(),
        memory=_FakeMemory(),
        knowledge_graph=_FakeGraph(),
        db_path=tmp_path / "ingestion.db",
    )
    yield eng
    eng.close()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# ── embedding primitives ──────────────────────────────────────────────────────

def test_hash_embed_is_unit_and_deterministic():
    a = _hash_embed("membrane potential sodium channel")
    b = _hash_embed("membrane potential sodium channel")
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6
    assert _cosine(a, b) == pytest.approx(1.0, abs=1e-6)


def test_similar_text_scores_higher_than_unrelated():
    q = _hash_embed("action potential depolarisation")
    near = _hash_embed("the action potential drives depolarisation of the membrane")
    far = _hash_embed("quarterly marketing budget spreadsheet totals")
    assert _cosine(q, near) > _cosine(q, far)


# ── fingerprint gating ────────────────────────────────────────────────────────

def test_first_ingest_indexes_then_second_skips(engine, tmp_path):
    f = _write(tmp_path, "note.md", "# Title\n\nSome durable prose about neurons and synapses.")
    first = engine.ingest_file(f)
    assert first.status == "indexed"
    assert first.chunks >= 1
    second = engine.ingest_file(f)
    assert second.status == "skipped"
    assert "unchanged" in second.note.lower()


def test_modified_file_reindexes_and_bumps_version(engine, tmp_path):
    f = _write(tmp_path, "doc.txt", "Original content about the hippocampus.")
    engine.ingest_file(f)
    f.write_text("Rewritten content about the cortex and plasticity.", encoding="utf-8")
    updated = engine.ingest_file(f)
    assert updated.status == "updated"
    assert updated.version == 2


def test_resume_after_partial_status(engine, tmp_path):
    f = _write(tmp_path, "resume.txt", "Paragraph one.\n\nParagraph two about axons.")
    res = engine.ingest_file(f)
    doc_id = res.doc_id
    # Simulate an interruption: mark the row partial without changing bytes.
    engine.fingerprints.set_status(doc_id, "partial")
    sha = engine.fingerprints.lookup(doc_id)["sha256"]
    assert engine.fingerprints.decide(doc_id, sha) == "resume"
    again = engine.ingest_file(f)
    assert again.status in {"indexed", "updated"}
    assert engine.fingerprints.lookup(doc_id)["status"] == "complete"


# ── deduplication + embedding cache ───────────────────────────────────────────

def test_identical_chunks_dedup_across_files(engine, tmp_path):
    body = "Shared paragraph repeated verbatim across two separate documents here."
    a = _write(tmp_path, "a.txt", body)
    b = _write(tmp_path, "b.txt", body)
    engine.ingest_file(a)
    misses_before = engine.embeddings.misses
    engine.ingest_file(b)
    stats = engine.stats()
    # Two documents, but the shared chunk is unique in the dedup ledger.
    assert stats["documents"] == 2
    assert stats["unique_chunks"] < stats["chunks"]
    # Second file's identical chunk is served from the embedding cache.
    assert engine.embeddings.hits >= 1
    assert engine.embeddings.misses == misses_before


def test_dedup_service_marks_once(tmp_path):
    from orion_core.ingestion import _Store
    store = _Store(tmp_path / "d.db")
    dedup = KnowledgeDeduplicationService(store)
    assert dedup.is_new("hash123")
    dedup.mark("hash123", "doc1")
    assert not dedup.is_new("hash123")
    store.close()


# ── folder + zip fan-out ──────────────────────────────────────────────────────

def test_ingest_folder_walks_supported_and_skips_noise(engine, tmp_path):
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "__pycache__").mkdir()
    _write(root, "readme.md", "Project readme with meaningful description text.")
    _write(root / "src", "main.py", "def run():\n    return 42\n")
    _write(root / "__pycache__", "junk.py", "cached noise")
    batch = engine.ingest_folder(root)
    paths = {Path(r.path).name for r in batch.results}
    assert "readme.md" in paths
    assert "main.py" in paths
    assert "junk.py" not in paths          # __pycache__ filtered
    assert batch.indexed == 2


def test_ingest_zip_fans_out(engine, tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inside/a.txt", "Zipped document alpha about neurons.")
        zf.writestr("inside/b.md", "# Beta\nZipped document beta about cortex.")
        zf.writestr("inside/skip.bin", b"\x00\x01binary")
    result = engine.ingest_file(archive)
    assert result.status == "indexed"
    assert result.category == "zip"
    assert result.chunks >= 2               # two supported members ingested


# ── unsupported + empty handling ──────────────────────────────────────────────

def test_unsupported_type_is_reported_not_raised(engine, tmp_path):
    f = _write(tmp_path, "thing.bin", "irrelevant")
    f = f.rename(tmp_path / "thing.xyz")
    res = engine.ingest_file(f)
    assert res.status == "unsupported"
    assert res.ok is False


def test_empty_file_marked_empty(engine, tmp_path):
    f = _write(tmp_path, "blank.txt", "   \n  \n")
    res = engine.ingest_file(f)
    assert res.status == "empty"


# ── integration: graph + memory + search ──────────────────────────────────────

def test_ingest_feeds_graph_and_memory(engine, tmp_path):
    f = _write(tmp_path, "brief.md",
               "# Supplier Report\n\nProject Orion depends on supplier Acme for widgets.")
    res = engine.ingest_file(f)
    assert res.entities == ["Alpha", "Beta"]      # resolved via entity_name
    assert engine.graph.calls, "graph.ingest_record should be called"
    assert any(k.startswith("doc_") for k in engine.memory.records)


def test_search_ranks_relevant_document(engine, tmp_path):
    _write(tmp_path, "neuro.txt", "Dendrites integrate synaptic input across the membrane.")
    _write(tmp_path, "budget.txt", "The marketing budget covers advertising spend this quarter.")
    engine.ingest_folder(tmp_path)
    hits = engine.search("synaptic membrane input", limit=3)
    assert hits
    assert Path(hits[0]["path"]).name == "neuro.txt"


def test_library_and_stats_report(engine, tmp_path):
    _write(tmp_path, "one.txt", "First document about plasticity.")
    _write(tmp_path, "two.txt", "Second document about decoding.")
    engine.ingest_folder(tmp_path)
    lib = engine.library()
    assert len(lib) >= 2
    stats = engine.stats()
    assert stats["documents"] >= 2
    assert stats["chunks"] >= 2


# ── directive memory (#13) ────────────────────────────────────────────────────

def test_prose_directive_captures_instructions(engine, tmp_path):
    f = _write(tmp_path, "policy.md",
               "# Deployment Policy\n\n"
               "You must never commit secrets to the repository. "
               "Always run the full test suite before shipping. "
               "The build is deterministic and reproducible.")
    res = engine.ingest_file(f)
    assert res.directive
    low = res.directive.lower()
    assert "policy.md" in low
    assert "must" in low or "always" in low          # a real instruction surfaced


def test_code_directive_names_declarations(engine, tmp_path):
    f = _write(tmp_path, "widget.py",
               "def build():\n    return 1\n\n"
               "class Widget:\n    pass\n")
    res = engine.ingest_file(f)
    assert res.category == "code"
    assert "build" in res.directive and "Widget" in res.directive


def test_json_directive_lists_keys(engine, tmp_path):
    f = _write(tmp_path, "settings.json",
               '{"api_key": "x", "timeout": 30, "retries": 3}')
    res = engine.ingest_file(f)
    assert "settings.json" in res.directive
    assert "api_key" in res.directive or "keys:" in res.directive


def test_catalog_lists_files_with_directives(engine, tmp_path):
    _write(tmp_path, "guide.txt", "Please follow the onboarding steps carefully every day.")
    _write(tmp_path, "mod.py", "def handler():\n    return True\n")
    engine.ingest_folder(tmp_path)
    catalog = engine.catalog()
    names = {c["filename"] for c in catalog}
    assert {"guide.txt", "mod.py"} <= names
    assert all(c["directive"] for c in catalog)


def test_memory_record_states_the_directive(engine, tmp_path):
    f = _write(tmp_path, "rules.md",
               "Rules: you must always validate input before use.")
    engine.ingest_file(f)
    stored = " ".join(engine.memory.records.values())
    assert "Directive:" in stored


def test_library_row_includes_directive(engine, tmp_path):
    _write(tmp_path, "doc.txt", "Ensure the pipeline is idempotent and side-effect free.")
    engine.ingest_folder(tmp_path)
    lib = engine.library()
    assert lib and "directive" in lib[0]
    assert lib[0]["directive"]


def test_skip_preserves_directive(engine, tmp_path):
    f = _write(tmp_path, "keep.md", "You should keep this configuration stable.")
    first = engine.ingest_file(f)
    second = engine.ingest_file(f)              # unchanged → skipped
    assert second.status == "skipped"
    assert second.directive == first.directive
