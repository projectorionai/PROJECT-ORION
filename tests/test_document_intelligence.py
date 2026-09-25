"""
Document intelligence (CAP-06) — read a folder, retrieve offline, answer cited.

    "File scanning must be absolute and lightning-fast, and ORION must read the
     files to me in full detail."
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.document_intelligence import (  # noqa: E402
    DocumentIntelligence, chunk, read_document,
)


@pytest.fixture()
def folder(tmp_path):
    (tmp_path / "mars.md").write_text(
        "# Mars\n"
        "Mars is the fourth planet from the Sun.\n"
        "It has two small moons, Phobos and Deimos.\n"
        "The Martian day is called a sol and lasts about 24 hours 39 minutes.\n",
        encoding="utf-8")
    (tmp_path / "oceans.txt").write_text(
        "The Pacific is the largest and deepest ocean on Earth.\n"
        "The Mariana Trench is the deepest known point in any ocean.\n"
        "Oceans regulate the planet's climate by storing heat.\n",
        encoding="utf-8")
    (tmp_path / "budget.csv").write_text(
        "item,cost\nserver,40\ndomain,12\nmarketing,300\n", encoding="utf-8")
    (tmp_path / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n not really an image")
    return tmp_path


@pytest.fixture()
def di(folder):
    engine = DocumentIntelligence()
    engine.ingest_folder(folder)
    return engine


# ── chunking ─────────────────────────────────────────────────────────────────

def test_chunk_anchors_line_numbers():
    text = "line one\nline two\nline three\n"
    passages = chunk("x.txt", text)
    assert passages
    assert passages[0].line_start == 1
    assert passages[0].line_end >= 1
    assert "line one" in passages[0].text


def test_chunk_of_empty_text_is_empty():
    assert chunk("x.txt", "") == []


def test_citation_is_filename_and_lines():
    passages = chunk("/tmp/report.txt", "alpha beta\ngamma delta\n")
    assert passages[0].citation.startswith("report.txt:")


# ── ingestion ────────────────────────────────────────────────────────────────

def test_reads_the_text_files_and_skips_the_image(di):
    assert di.file_count == 3          # mars, oceans, budget — not the png
    assert di.passages


def test_ingest_report_describes_what_it_did(folder):
    engine = DocumentIntelligence()
    report = engine.ingest_folder(folder)
    assert report.files_read == 3          # folder ingest pre-filters the .png out
    assert report.total_words > 0
    assert "passages" in report.describe()


def test_ingest_paths_reports_an_unsupported_file_as_skipped(folder):
    engine = DocumentIntelligence()
    report = engine.ingest_paths([folder / "photo.png", folder / "mars.md"])
    assert report.files_read == 1          # only mars.md is readable
    assert any("photo.png" in s for s in report.skipped)


def test_ingesting_a_missing_folder_is_graceful():
    engine = DocumentIntelligence()
    report = engine.ingest_folder("/no/such/folder/here")
    assert report.files_read == 0
    assert report.skipped


def test_suffix_filter_limits_what_is_read(folder):
    engine = DocumentIntelligence()
    engine.ingest_folder(folder, suffixes={".csv"})
    assert engine.file_count == 1


# ── retrieval ────────────────────────────────────────────────────────────────

def test_search_finds_the_right_document(di):
    hits = di.search("deepest ocean trench")
    assert hits
    assert "oceans.txt" in hits[0].passage.source


def test_search_on_mars_finds_mars(di):
    hits = di.search("moons of Mars Phobos")
    assert hits
    assert "mars.md" in hits[0].passage.source


def test_search_with_no_match_returns_empty(di):
    assert di.search("quantum chromodynamics tokamak") == []


def test_search_empty_query_returns_empty(di):
    assert di.search("") == []


# ── answering ────────────────────────────────────────────────────────────────

async def test_extractive_answer_quotes_with_citations(di):
    answer = await di.ask("What is the deepest ocean point?")
    assert not answer.grounded
    assert answer.sources
    assert "oceans.txt" in answer.sources[0]
    assert "Mariana" in answer.text


async def test_model_answer_is_grounded_and_prompt_carries_passages(di):
    captured = {}

    async def fake_generate(prompt: str) -> str:
        captured["prompt"] = prompt
        return "The Mariana Trench is the deepest point [1]."

    answer = await di.ask("deepest ocean point", generate=fake_generate)
    assert answer.grounded
    assert "[1]" in answer.text
    # the model must have been handed the retrieved passages + a cite instruction
    assert "PASSAGES:" in captured["prompt"]
    assert "Mariana" in captured["prompt"]
    assert "[1], [2]" in captured["prompt"]


async def test_model_failure_falls_back_to_extractive(di):
    async def broken(prompt: str) -> str:
        raise RuntimeError("model down")

    answer = await di.ask("deepest ocean point", generate=broken)
    assert not answer.grounded
    assert "Mariana" in answer.text


async def test_ask_with_no_documents_says_so():
    engine = DocumentIntelligence()
    answer = await engine.ask("anything")
    assert not answer.grounded
    assert answer.sources == []


async def test_digest_extractive_lists_each_file(di):
    text = await di.digest()
    assert "mars.md" in text and "oceans.txt" in text


async def test_digest_uses_the_model_when_present(di):
    async def fake_generate(prompt: str) -> str:
        return "These documents cover Mars, oceans and a small budget."
    text = await di.digest(generate=fake_generate)
    assert "budget" in text.lower()


# ── the tool wiring ──────────────────────────────────────────────────────────

def test_tool_is_registered_and_schema_present():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"read_documents"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "read_documents")
    props = tool["parameters"]["properties"]
    assert "action" in props and "path" in props and "question" in props
