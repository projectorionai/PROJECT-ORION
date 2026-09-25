"""
Tests for LiteratureIntakeService — previously entirely uncovered — with a
focus on the Mark XX design-spec §3 addition: ingested papers now ALSO
become real knowledge-graph structure (a research entity for the paper,
concept entities for its mechanisms linked by 'describes', citation
entities for its DOIs/references linked by 'cites') instead of only ever
landing in the flat, disconnected Literature Digest log.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.knowledge_graph import KnowledgeGraphEngine
from orion_core.literature import LiteratureIntakeService

_PAPER_TEXT = """Abstract

This paper describes a mechanism involving the sodium channel and the
resulting action potential in cortical neurons under study. Separately,
membrane potential changes are governed by voltage-gated ion channel
activity described in detail below across several recorded preparations.

DOI: 10.1038/s41586-020-12345-6

References
[1] Smith J, Doe A. A comprehensive study of neural signaling mechanisms in cortex.
"""


class _Sig:
    def emit(self, *a, **k):
        pass


class _StubBus:
    def __getattr__(self, name):
        s = _Sig()
        object.__setattr__(self, name, s)
        return s


class _StubMemory:
    def __init__(self) -> None:
        self.remembered: list[tuple] = []

    def remember(self, tier, key, value):
        self.remembered.append((tier, key, value))

    def query(self, query, limit=8):
        return []


def _paper_file(tmp_path) -> Path:
    path = tmp_path / "paper.txt"
    path.write_text(_PAPER_TEXT, encoding="utf-8")
    return path


# ── ingest_paper: baseline behaviour (previously untested) ──────────────────

def test_ingest_paper_succeeds_and_seeds_memory(tmp_path):
    memory = _StubMemory()
    service = LiteratureIntakeService(_StubBus(), memory)
    result = asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))
    assert result.ok
    assert any(key == "lit_test_paper" for _tier, key, _val in memory.remembered)


def test_ingest_paper_reports_failure_for_a_missing_file(tmp_path):
    service = LiteratureIntakeService(_StubBus(), _StubMemory())
    result = asyncio.run(service.ingest_paper(str(tmp_path / "missing.pdf")))
    assert not result.ok
    assert "not found" in result.text.lower()


def test_analyse_extracts_the_doi_and_a_mechanism(tmp_path):
    service = LiteratureIntakeService(_StubBus(), _StubMemory())
    asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))
    record = service._papers["test_paper"]
    assert "10.1038/s41586-020-12345-6" in record.dois
    assert any("sodium channel" in m.lower() for m in record.mechanisms)


# ── graph seeding (Mark XX design-spec §3) ──────────────────────────────────

def test_ingest_paper_with_no_graph_attached_behaves_exactly_as_before(tmp_path):
    service = LiteratureIntakeService(_StubBus(), _StubMemory())   # graph=None
    result = asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))
    assert result.ok   # no crash, no change in observable behaviour


def test_seed_graph_creates_a_research_entity_for_the_paper(tmp_path):
    graph = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")
    service = LiteratureIntakeService(_StubBus(), _StubMemory(), graph=graph)
    asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))

    snap = graph.graph_snapshot()
    research_nodes = [n for n in snap["nodes"] if n["kind"] == "research"]
    assert any(n["name"] == "Test Paper" for n in research_nodes)


def test_seed_graph_links_mechanisms_as_concepts_with_describes_edges(tmp_path):
    graph = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")
    service = LiteratureIntakeService(_StubBus(), _StubMemory(), graph=graph)
    asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))

    snap = graph.graph_snapshot()
    concept_nodes = {n["id"]: n for n in snap["nodes"] if n["kind"] == "concept"}
    assert concept_nodes   # at least one mechanism became a concept node
    describes_edges = [e for e in snap["edges"] if e["kind"] == "describes"]
    assert describes_edges
    assert all(e["target"] in concept_nodes for e in describes_edges)


def test_seed_graph_links_dois_as_citations_with_cites_edges(tmp_path):
    graph = KnowledgeGraphEngine(_StubBus(), memory=None, db_path=tmp_path / "kg.db")
    service = LiteratureIntakeService(_StubBus(), _StubMemory(), graph=graph)
    asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))

    snap = graph.graph_snapshot()
    citation_nodes = [n for n in snap["nodes"] if n["kind"] == "citation"]
    assert any(n["name"] == "10.1038/s41586-020-12345-6" for n in citation_nodes)
    cites_edges = [e for e in snap["edges"] if e["kind"] == "cites"]
    assert cites_edges


def test_seed_graph_failure_never_breaks_ingestion(tmp_path):
    class _BrokenGraph:
        def upsert_entity(self, *a, **k):
            raise RuntimeError("graph offline")

    memory = _StubMemory()
    service = LiteratureIntakeService(_StubBus(), memory, graph=_BrokenGraph())
    result = asyncio.run(service.ingest_paper(str(_paper_file(tmp_path)), title="Test Paper"))
    assert result.ok   # the graph side-effect failing must not fail ingestion
    assert memory.remembered   # memory seeding still happened
