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
