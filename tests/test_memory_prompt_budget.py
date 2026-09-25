"""
The memory core in the system prompt — and the index of what did not fit.

The defect, measured rather than suspected
------------------------------------------
``prompt_context`` previously selected rows by recency alone. This synthetic
fixture reproduces how frequently updated bulk knowledge can crowd out less
recent identity, relationship, project and note entries.

Two fixes, and the second is the subtle one
-------------------------------------------
Categories are interleaved, with the ones describing the user going round twice
before the bulk ones go round once. A thousand rows of one kind can no longer
starve the one row of another.

And what did not fit is listed BY KEY, because a model cannot look something up
if it does not know the thing exists. Without the index, "who is Sample Relative?" gets
"I don't know" while ``relationship/sample_relative`` sits on disk unread. The
index is interleaved for exactly the same reason the core is: sorted by
category, four hundred knowledge keys would push the one relationship key off
the end of the very list that exists to surface it.

Offline: an in-memory SQLite store, no Qt, no model.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.memory import OrionMemoryMatrix


def _matrix(rows):
    """A matrix over an in-memory store holding exactly *rows*.

    Built with __new__ so the test exercises prompt_context itself rather than
    the constructor's migrations and background wiring.
    """
    matrix = OrionMemoryMatrix.__new__(OrionMemoryMatrix)
    matrix._lock = threading.RLock()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE intelligence (id INTEGER PRIMARY KEY, "
                 "category TEXT, key_ref TEXT, value TEXT, updated_at TEXT)")
    for index, (category, key, value, stamp) in enumerate(rows):
        conn.execute("INSERT INTO intelligence (category, key_ref, value, "
                     "updated_at) VALUES (?,?,?,?)", (category, key, value, stamp))
    conn.commit()
    matrix.conn = conn
    return matrix


def _lopsided():
    """A synthetic store with one category swamping everything else."""
    rows = [("knowledge", f"scraped_{i}", f"fact {i}", f"2026-09-18T10:{i:02d}:00Z")
            for i in range(200)]
    # Three rows per user-facing category, all OLDER than every knowledge row,
    # so recency alone would bury every one of them.
    for category, keys in (
        ("identity", ("orion_name", "orion_role", "orion_voice")),
        ("relationship", ("sample_relative", "sam_colleague", "pat_neighbour")),
        ("personal", ("coffee", "allergy", "commute")),
        ("projects", ("orion", "sample_game", "studies")),
        ("notes", ("reminder", "shopping", "idea")),
    ):
        for key in keys:
            rows.append((category, key, f"{category} fact {key}",
                         "2026-01-01T00:00:00Z"))
    return rows


def _body(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.startswith("- ")]


def _index_line(text: str) -> str:
    for line in text.split("\n"):
        if line.startswith("[ALSO REMEMBERED"):
            return line
    return ""


# ── the core ─────────────────────────────────────────────────────────────────

def test_an_empty_store_contributes_nothing():
    assert _matrix([]).prompt_context() == ""


def test_the_budget_is_respected():
    rows = _body(_matrix(_lopsided()).prompt_context(limit=18))
    assert len(rows) == 18


def test_bulk_rows_can_no_longer_starve_the_user_facing_ones():
    """THE regression. Two hundred knowledge rows, every one of them newer than
    every user fact, previously took all eighteen slots."""
    text = _matrix(_lopsided()).prompt_context(limit=18)
    categories = Counter(line[2:].split("/")[0] for line in _body(text))
    for essential in ("identity", "relationship", "personal", "projects", "notes"):
        assert categories[essential] >= 1, f"{essential} was shut out again"
    assert categories["knowledge"] < 18, "knowledge took the whole prompt"


CORE_LABELS = ("identity", "personal", "relationship", "long term",
               "projects", "notes")


def _core_count(text: str) -> int:
    return sum(1 for line in _body(text)
               if line[2:].split("/")[0] in CORE_LABELS)


def test_the_user_facing_categories_get_the_larger_share():
    """Where there ARE user facts to carry, they should dominate the prompt —
    a plain rotation gave `identity` and `pack_tiktok_shop` a slot each, which
    is equal treatment of things that are not equal."""
    text = _matrix(_lopsided()).prompt_context(limit=18)
    assert _core_count(text) > len(_body(text)) / 2, "the prompt is still mostly bulk"


def test_bulk_still_fills_the_budget_when_there_is_little_else():
    """The weighting must not waste the prompt. With only a handful of user
    facts in the store, the remaining slots SHOULD go to knowledge — there is
    nothing else to carry, and an empty slot helps nobody."""
    rows = [("knowledge", f"k{i}", f"v{i}", f"2026-09-18T10:{i:02d}:00Z")
            for i in range(60)]
    rows += [("identity", "name", "Orion", "2026-01-01T00:00:00Z"),
             ("personal", "coffee", "black", "2026-01-01T00:00:00Z")]
    text = _matrix(rows).prompt_context(limit=18)
    assert len(_body(text)) == 18
    assert _core_count(text) == 2, "both user facts should still be present"


def test_a_single_category_store_still_fills_the_prompt():
    """Interleaving must not starve a store that has only one kind of row."""
    rows = [("knowledge", f"k{i}", f"v{i}", "2026-01-01T00:00:00Z") for i in range(50)]
    assert len(_body(_matrix(rows).prompt_context(limit=18))) == 18


def test_a_store_smaller_than_the_budget_is_carried_whole():
    rows = [("identity", "a", "1", "x"), ("personal", "b", "2", "x")]
    text = _matrix(rows).prompt_context(limit=18)
    assert len(_body(text)) == 2
    assert _index_line(text) == "", "nothing was omitted, so there is no index"


def test_values_are_carried_for_the_rows_that_made_it():
    text = _matrix([("relationship", "sample_relative", "Sample Relative is the user's sister",
                     "x")]).prompt_context()
    assert "Sample Relative is the user's sister" in text


# ── the index ────────────────────────────────────────────────────────────────

def test_what_did_not_fit_is_named_so_it_can_be_asked_for():
    """A model cannot look up something it does not know exists."""
    index = _index_line(_matrix(_lopsided()).prompt_context(limit=6))
    assert index, "no index of omitted keys"
    assert "query_intelligence" in index, "the model is not told how to fetch them"


def test_the_index_names_keys_not_values():
    """Keys are a few words each; values are the expensive part. Naming keys is
    what makes carrying an index of a thousand rows affordable at all."""
    text = _matrix(_lopsided()).prompt_context(limit=6)
    index = _index_line(text)
    assert "fact 1" not in index, "a value leaked into the index"


def test_the_index_interleaves_rather_than_sorting():
    """Sorted by category, two hundred knowledge keys would push the one
    relationship key off the end of the list that exists to surface it."""
    index = _index_line(_matrix(_lopsided()).prompt_context(limit=2))
    named = index.split("]")[-1]
    assert "relationship" in named, "the rare key was buried under the bulk"


def test_the_index_says_how_many_more_there_are():
    index = _index_line(_matrix(_lopsided()).prompt_context(limit=6))
    assert "more)" in index


def test_the_index_is_bounded():
    text = _matrix(_lopsided()).prompt_context(limit=6)
    named = _index_line(text).split("]")[-1].split(",")
    assert len(named) <= OrionMemoryMatrix.INDEX_KEYS + 1


def test_the_whole_block_stays_a_reasonable_size():
    """This goes into every request; it has to be a core, not a dump."""
    text = _matrix(_lopsided()).prompt_context(limit=18)
    assert len(text) < 12_000, len(text)


# ── the tool the index points at ─────────────────────────────────────────────

def test_the_recall_tool_tells_the_model_when_to_use_it():
    """Its description was five words — "Search local SQLite FTS5 memory." —
    which gives a model no reason to reach for it and no idea what is in it."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    spec = next(t for t in TOOL_DECLARATIONS if t["name"] == "query_intelligence")
    description = spec["description"]
    assert "ALSO REMEMBERED" in description, "it does not point at the index"
    assert "before saying you do not know" in description.lower()
    assert len(description) > 120, "too terse to change the model's behaviour"
