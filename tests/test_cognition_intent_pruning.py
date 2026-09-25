"""
Tests for IntentTracker's bounded request ledger.

Regression cover for a real unbounded-growth fault: every distinct request
signature was retained forever, so config/cognitive_state.json reached 1046
entries / 267 KB (75% of the file) within weeks. Because the state manager
rewrites the whole document on every mutation, each observed request was
re-serialising a quarter-megabyte to disk, and every launch parsed it back.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.cognition import IntentTracker


def _ledger(n: int, *, count: int = 1, prefix: str = "req") -> dict:
    """A request ledger with *n* entries, each repeated *count* times."""
    return {
        f"{prefix}_{i}": {
            "id": f"{prefix}_{i}",
            "count": count,
            "last_seen": f"2026-08-02T00:{i % 60:02d}:00Z",
        }
        for i in range(n)
    }


def test_prune_is_a_noop_below_the_cap():
    root = _ledger(10)
    IntentTracker._prune(root)
    assert len(root) == 10


def test_prune_is_a_noop_exactly_at_the_cap():
    root = _ledger(IntentTracker.MAX_TRACKED_REQUESTS)
    IntentTracker._prune(root)
    assert len(root) == IntentTracker.MAX_TRACKED_REQUESTS


def test_prune_trims_to_prune_to_once_over_the_cap():
    root = _ledger(IntentTracker.MAX_TRACKED_REQUESTS + 50)
    IntentTracker._prune(root)
    assert len(root) == IntentTracker.PRUNE_TO


def test_prune_keeps_repeated_patterns_over_one_offs():
    """Repetition is the whole signal an intent tracker exists to find, so a
    frequently-repeated entry must survive a flood of single-shot requests."""
    root = _ledger(IntentTracker.MAX_TRACKED_REQUESTS + 100, count=1)
    root["important"] = {
        "id": "important",
        "count": 99,
        "last_seen": "2026-01-01T00:00:00Z",   # old, but heavily repeated
    }
    IntentTracker._prune(root)
    assert "important" in root


def test_prune_breaks_ties_on_recency():
    """Among equally-repeated entries, the more recent one is the better bet."""
    root = _ledger(IntentTracker.MAX_TRACKED_REQUESTS + 100, count=2)
    root["stale"] = {"id": "stale", "count": 2, "last_seen": "2000-01-01T00:00:00Z"}
    root["fresh"] = {"id": "fresh", "count": 2, "last_seen": "2099-01-01T00:00:00Z"}
    IntentTracker._prune(root)
    assert "fresh" in root
    assert "stale" not in root


def test_prune_tolerates_missing_fields():
    """A record written by an older schema (no count/last_seen) must not raise."""
    root = _ledger(IntentTracker.MAX_TRACKED_REQUESTS + 10)
    root["legacy"] = {"id": "legacy"}
    IntentTracker._prune(root)
    assert len(root) == IntentTracker.PRUNE_TO
