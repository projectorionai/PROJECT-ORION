"""
Change-awareness tests (#15).

ORION fingerprints his own source tree and reports, at the file level, exactly
which of his modules changed / were added / were removed between runs.  These
tests drive the tracker against a throwaway source tree so nothing touches the
real package or config.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.change_awareness import SourceChangeTracker


def _tracker(tmp_path: Path) -> SourceChangeTracker:
    root = tmp_path / "src"
    root.mkdir()
    (root / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (root / "beta.py").write_text("print('beta')\n", encoding="utf-8")
    store = tmp_path / "store.json"
    return SourceChangeTracker(root=root, store_path=store)


def test_first_run_is_a_silent_baseline(tmp_path):
    tracker = _tracker(tmp_path)
    change = tracker.detect()
    assert change.first_snapshot is True
    assert change.count == 0
    assert change.total_files == 2
    assert tracker.store_path.exists()          # baseline persisted
    assert "baseline" in tracker.render(change).lower()


def test_detects_modified_added_and_removed(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.detect()                             # establish baseline

    (tracker.root / "alpha.py").write_text("print('ALPHA changed')\n", encoding="utf-8")
    (tracker.root / "gamma.py").write_text("print('gamma')\n", encoding="utf-8")
    (tracker.root / "beta.py").unlink()

    change = tracker.detect()
    assert change.modified == ["alpha.py"]
    assert change.added == ["gamma.py"]
    assert change.removed == ["beta.py"]
    assert change.count == 3
    rendered = tracker.render(change)
    assert "alpha.py" in rendered and "gamma.py" in rendered and "beta.py" in rendered


def test_no_changes_after_commit(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.detect()
    change = tracker.detect()
    assert not change.changed
    assert "have changed" not in tracker.render(change) or "None" in tracker.render(change)
    assert tracker.render(change).lower().startswith("none")


def test_scan_without_commit_does_not_move_baseline(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.detect()
    (tracker.root / "alpha.py").write_text("changed\n", encoding="utf-8")

    peek = tracker.detect(commit=False)
    assert peek.modified == ["alpha.py"]
    # Baseline untouched — a second non-committing scan still sees the change.
    again = tracker.detect(commit=False)
    assert again.modified == ["alpha.py"]


def test_pycache_and_non_python_ignored(tmp_path):
    tracker = _tracker(tmp_path)
    (tracker.root / "__pycache__").mkdir()
    (tracker.root / "__pycache__" / "alpha.cpython-313.pyc").write_text("x", encoding="utf-8")
    (tracker.root / "notes.txt").write_text("not source", encoding="utf-8")
    manifest = tracker.scan()
    assert set(manifest) == {"alpha.py", "beta.py"}


def test_history_records_update_events(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.detect()                             # baseline (no event)
    (tracker.root / "alpha.py").write_text("v2\n", encoding="utf-8")
    tracker.detect()                             # event 1
    (tracker.root / "beta.py").write_text("v2\n", encoding="utf-8")
    tracker.detect()                             # event 2

    events = tracker.recent_events(10)
    assert len(events) == 2
    history = tracker.render_history()
    assert "recent updates" in history.lower()


def test_nested_module_paths_are_tracked(tmp_path):
    tracker = _tracker(tmp_path)
    sub = tracker.root / "gui"
    sub.mkdir()
    (sub / "face.py").write_text("print('face')\n", encoding="utf-8")
    tracker.detect()
    (sub / "face.py").write_text("print('face v2')\n", encoding="utf-8")
    change = tracker.detect()
    assert "gui/face.py" in change.modified
    assert "face.py" in tracker.render(change)


def test_corrupt_store_degrades_to_baseline(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.store_path.write_text("{ not valid json", encoding="utf-8")
    change = tracker.detect()                    # must not raise
    assert change.first_snapshot is True


def test_render_latest_uses_cached_boot_change(tmp_path):
    tracker = _tracker(tmp_path)
    tracker.detect()
    (tracker.root / "alpha.py").write_text("boot change\n", encoding="utf-8")
    tracker.detect()                             # caches last_change
    assert "alpha.py" in tracker.render_latest()
