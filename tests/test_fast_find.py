"""
Finding a file at the speed Windows already can.

  "he must be efficient at finding files faster at lightning speeds, or if he's
   struggling he can take his time and find other methods."

The old search walked the user's folders with rglob('*') — correct, and the
slowest possible way to answer, because Windows has already done the work. Its
Search indexer keeps a live catalogue of the profile and answers SQL over ADO.

Measured on this machine: the index returned matches across the ENTIRE profile,
OneDrive tree included, in 0.12-0.35 s. The equivalent walk is minutes.

Both halves of the request are in the design. Fast when it can be (index),
thorough when it must be (walk) — and which one answered is always reported,
because "no matches" from an index that happens to be disabled means something
very different from "no matches" after reading every directory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import fast_find  # noqa: E402
from orion_core.fast_find import Hit, Search, find, search_walk  # noqa: E402


@pytest.fixture
def tree(tmp_path):
    """A small filesystem with the shapes that matter."""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "quarterly-orion.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "reports" / "unrelated.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "orion-notes").mkdir()
    # A directory the walk must refuse to descend into.
    heavy = tmp_path / "node_modules" / "deep" / "deeper"
    heavy.mkdir(parents=True)
    (heavy / "orion-buried.txt").write_text("x", encoding="utf-8")
    return tmp_path


# ── the walk ─────────────────────────────────────────────────────────────────

def test_it_finds_files_by_name(tree):
    hits, _ = search_walk("orion", roots=[tree])
    names = {Path(h.path).name for h in hits}
    assert "quarterly-orion.txt" in names


def test_it_finds_folders_too(tree):
    hits, _ = search_walk("orion", roots=[tree])
    assert any(h.kind == "folder" and h.name == "orion-notes" for h in hits)


def test_it_does_not_match_unrelated_files(tree):
    hits, _ = search_walk("orion", roots=[tree])
    assert not any(h.name == "unrelated.txt" for h in hits)


def test_heavy_directories_are_pruned(tree):
    """os.walk with in-place pruning rather than rglob: rglob cannot skip a
    subtree, so it descends into node_modules and site-packages and spends most
    of its time there."""
    hits, _ = search_walk("orion", roots=[tree])
    assert not any("node_modules" in h.path for h in hits), (
        "walked into node_modules")


def test_the_skip_list_covers_the_usual_offenders():
    for name in ("node_modules", "__pycache__", ".git", ".venv", "site-packages"):
        assert name in fast_find.SKIP_DIRS


def test_the_walk_is_time_boxed(tree):
    """A search that never returns is worse than one that returns partial
    results and says so."""
    started = time.monotonic()
    _hits, note = search_walk("orion", roots=[Path(tree)], timeout=0.001)
    assert time.monotonic() - started < 5.0
    assert note == "" or "time limit" in note


def test_the_result_limit_is_honoured(tmp_path):
    for index in range(40):
        (tmp_path / f"orion-{index}.txt").write_text("x", encoding="utf-8")
    hits, _ = search_walk("orion", roots=[tmp_path], limit=10)
    assert len(hits) <= 10


def test_a_missing_root_does_not_raise(tmp_path):
    hits, _ = search_walk("orion", roots=[tmp_path / "nope"])
    assert hits == []


# ── the ladder ───────────────────────────────────────────────────────────────

def test_an_empty_query_is_refused_not_guessed():
    result = find("")
    assert result.hits == []
    assert "no search term" in result.note


def test_the_result_says_which_strategy_answered(tree):
    result = find("orion", folder=str(tree))
    assert result.strategy, "did not say how it searched"
    assert "match" in result.describe() or "No matches" in result.describe()


def test_results_are_deduplicated():
    """Index and walk overlap; the same path must not appear twice."""
    duplicated = [Hit(r"C:\a\b.txt"), Hit(r"C:\a\b.txt"), Hit(r"C:\a\c.txt")]
    assert len(fast_find._dedupe(duplicated)) == 2


def test_deduplication_is_case_insensitive_on_windows():
    """Windows paths differ in case and are the same file."""
    hits = [Hit(r"C:\Users\A\File.txt"), Hit(r"c:\users\a\file.txt")]
    expected = 1 if os.name == "nt" else 2
    assert len(fast_find._dedupe(hits)) == expected


def test_a_thorough_search_walks_as_well_as_indexing(tree):
    """The index does not cover network drives, excluded folders, or a file
    created seconds ago — so 'it must be there' has to be able to escalate."""
    result = find("orion", folder=str(tree), thorough=True)
    assert "walk" in result.strategy


def test_a_search_reports_its_own_duration(tree):
    result = find("orion", folder=str(tree))
    assert result.seconds >= 0.0


# ── SQL safety ───────────────────────────────────────────────────────────────

def test_a_quote_in_the_query_cannot_break_the_sql():
    """The query goes into a SQL string literal, and it comes from speech —
    a filename with an apostrophe in it is completely ordinary."""
    assert fast_find._escape("O'Brien's notes") == "O''Brien''s notes"


def test_a_quoted_query_is_searched_not_rejected(tree):
    (tree / "o'brien-orion.txt").write_text("x", encoding="utf-8")
    hits, _ = search_walk("o'brien", roots=[tree])
    assert any("brien" in h.name for h in hits)


# ── the index, when this machine has one ─────────────────────────────────────

def _indexed_or_skip(limit: int):
    if not fast_find.index_available():
        pytest.skip("Windows Search is unavailable")
    started = time.monotonic()
    hits, error = fast_find.search_index("orion", limit=limit)
    elapsed = time.monotonic() - started
    if error:
        pytest.skip(f"Windows Search cannot answer queries here: {error}")
    if not hits:
        pytest.skip("Windows Search has no indexed ORION files here")
    return hits, elapsed


def test_the_index_is_much_faster_than_a_walk():
    """The whole justification for the feature, asserted rather than assumed."""
    hits, elapsed = _indexed_or_skip(20)
    assert elapsed < 5.0, f"the index took {elapsed:.1f}s"
    assert hits


def test_the_index_returns_real_paths():
    hits, _ = _indexed_or_skip(10)
    assert all(hit.path for hit in hits)


def test_a_machine_without_an_index_still_searches(monkeypatch, tree):
    """Losing search entirely because a Windows service is off would be a poor
    trade for the speed."""
    monkeypatch.setattr(fast_find, "index_available", lambda: False)
    result = find("orion", folder=str(tree))
    assert result.hits, "no fallback happened"
    assert "walk" in result.strategy


def test_an_index_failure_is_explained_not_swallowed(monkeypatch, tree):
    monkeypatch.setattr(fast_find, "search_index",
                        lambda *a, **k: ([], "the index query failed (test)"))
    result = find("nothing-matches-this-at-all", folder=str(tree))
    assert result.note, "fell back silently"


# ── it is actually wired in ──────────────────────────────────────────────────

def test_find_files_uses_the_fast_path():
    import inspect

    from orion_core.dispatch_files import FilesDispatchMixin

    source = inspect.getsource(FilesDispatchMixin.find_files)
    # Whole-PC search with filters (file_search), the index's full-text search
    # for "inside" queries (fast_find), never the old slow profile walk.
    assert "file_search.search" in source
    assert "fast_find.find" in source
    assert "_scan_user_files" not in source, "still using the slow walk directly"
def test_a_ranking_alone_is_a_question():
    """10 Oct (capability-gap ledger, twice): "Can you find the biggest files
    currently on the PC" was refused with "tell me what to look for"."""
    from orion_core import file_search as fs
    biggest = fs.build_query({"sort": "largest"})
    assert not biggest.is_empty() and biggest.kind == "file"
    assert not fs.build_query({"sort": "newest"}).is_empty()
    assert fs.build_query({"sort": "largest", "type": "folders"}).kind == "folder"
    assert fs.build_query({}).is_empty() and fs.build_query({"sort": "name"}).is_empty()


def test_folder_labels_use_the_real_redirected_location(tmp_path, monkeypatch):
    from orion_core import app_resolver, file_search as fs
    redirected = tmp_path / "OneDrive" / "Downloads"
    redirected.mkdir(parents=True)
    monkeypatch.setattr(app_resolver, "known_folder",
                        lambda text, **_: redirected if text in {"my downloads", "downloads"} else None)
    query = fs.build_query({"query": "my downloads"})
    monkeypatch.setattr(fs, "_everything_query", lambda *_: pytest.fail("known label invoked whole-PC index"))
    results = fs.search(query)
    assert [hit.path for hit in results.hits] == [str(redirected)]
    assert results.hits[0].is_folder
    scoped = fs.build_query({"query": "invoice", "folder": "downloads"})
    assert scoped.folder == str(redirected)


def test_spoken_folder_query_keeps_the_topic_and_explicit_quoted_names():
    from orion_core import file_search as fs
    query = fs.build_query({"query": "the research folder about solar energy"})
    assert query.kind == "folder" and query.name == "solar energy"
    assert query.research_topic == "solar energy"
    assert fs.build_query({"query": "folder called Tax 2026"}).name == "Tax 2026"
    assert fs.build_query({"query": '"the report"'}).name == "the report"


def test_research_folder_is_found_from_its_saved_report_title(tmp_path, monkeypatch):
    from orion_core import constants, file_search as fs
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "program")
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "private")
    reports = constants.CONFIG_DIR / "research" / "dossiers"
    reports.mkdir(parents=True)
    (reports / "report-42.md").write_text("# Solar energy\n\nSynthetic report", encoding="utf-8")
    (reports / "unrelated.md").write_text("# Tax reporting\n", encoding="utf-8")
    query = fs.build_query({"query": "the research folder about solar energy"})
    result = fs.search(query, engines=())
    assert [hit.path for hit in result.hits] == [str(reports)]
    assert "titles match" in result.note
    assert not fs.search(fs.build_query({"query": "research folder about lunar rocks"}), engines=()).hits


def test_multiple_saved_research_folders_remain_multiple_candidates(tmp_path, monkeypatch):
    from orion_core import constants, file_search as fs
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "program")
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "private")
    for parent in (constants.CONFIG_DIR, constants.BASE_DIR):
        folder = parent / "research" / "solar-study"
        folder.mkdir(parents=True)
        (folder / "solar-energy.md").write_text("# Solar energy", encoding="utf-8")
    result = fs.search(fs.build_query({"query": "research folder about solar energy"}), engines=())
    assert len(result.hits) == 2


def test_recognised_folder_does_not_escape_an_explicit_search_scope(tmp_path, monkeypatch):
    from orion_core import app_resolver, file_search as fs
    downloads, documents = tmp_path / "Downloads", tmp_path / "Documents"
    downloads.mkdir()
    documents.mkdir()
    monkeypatch.setattr(app_resolver, "known_folder",
                        lambda text, **_: downloads if text == "downloads" else None)
    query = fs.build_query({"query": "downloads", "folder": str(documents)})
    assert not fs.search(query).hits


def test_research_topic_can_be_scoped_to_a_subfolder_of_the_output_root(tmp_path, monkeypatch):
    from orion_core import constants, file_search as fs
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "program")
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "private")
    reports = constants.CONFIG_DIR / "research" / "dossiers"
    reports.mkdir(parents=True)
    (reports / "solar-energy.md").write_text("# Solar energy", encoding="utf-8")
    query = fs.build_query({"query": "research folder about solar energy", "folder": str(reports)})
    assert [hit.path for hit in fs.search(query, engines=()).hits] == [str(reports)]


def test_research_folder_search_honours_its_deadline_in_empty_subtrees(tmp_path, monkeypatch):
    from orion_core import constants, file_search as fs
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path / "program")
    monkeypatch.setattr(constants, "CONFIG_DIR", tmp_path / "private")
    ticks = iter(range(1000))
    monkeypatch.setattr(fs.time, "monotonic", lambda: float(next(ticks)))
    visited = []

    def empty_tree(root, onerror):
        for index in range(100):
            visited.append(index)
            yield str(root / str(index)), [], []

    monkeypatch.setattr(fs.os, "walk", empty_tree)
    assert fs.find_research_folders("solar energy", budget=2.0) == []
    assert len(visited) <= 2, "empty directories bypassed the search deadline"
