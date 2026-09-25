"""
Tests for WorkspaceSnapshot.active_context (Mark XX architectural-audit
pass, Track I) — wires app_context.describe_context() into the real
workspace snapshot path, not just as a standalone unused function.
Hardware-touching bits (_active_title, _enumerate_windows) are monkeypatched
so this stays hermetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.workspace import WorkspaceManager, WorkspaceSnapshot


class _Signal:
    def emit(self, *a, **k):
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubMemory:
    active_project = ""


def _manager() -> WorkspaceManager:
    return WorkspaceManager(_StubBus(), _StubMemory())


# ── WorkspaceSnapshot itself ────────────────────────────────────────────────

def test_active_context_defaults_to_blank():
    snap = WorkspaceSnapshot(at="now", active_window="x")
    assert snap.active_context == ""


def test_summary_includes_active_context_when_present():
    snap = WorkspaceSnapshot(at="now", active_window="VS Code", active_context="programming")
    assert "programming" in snap.summary()


def test_summary_omits_active_context_when_blank():
    snap = WorkspaceSnapshot(at="now", active_window="")
    assert snap.summary().count(";") == 0   # no project, no context — no trailing clauses


def test_from_json_tolerates_a_snapshot_saved_before_active_context_existed():
    # Old persisted snapshots in the WORKSPACE memory tier won't have this
    # key at all — from_json must not blow up on them.
    old_payload = json.dumps({
        "at": "2026-01-01T00:00:00Z", "active_window": "Notepad", "windows": [],
        "browser_tabs": [], "documents": [], "dev_sessions": [], "active_project": "",
        "open_applications": [], "monitor_layout": [], "notion_sessions": [],
        "outlook_sessions": [], "research_environments": [],
    })
    snap = WorkspaceSnapshot.from_json(old_payload)
    assert snap.active_context == ""


def test_to_json_round_trips_active_context():
    snap = WorkspaceSnapshot(at="now", active_window="x", active_context="playing chess")
    restored = WorkspaceSnapshot.from_json(snap.to_json())
    assert restored.active_context == "playing chess"


# ── wired into the real capture path ────────────────────────────────────────

def test_capture_sync_populates_active_context_from_the_foreground_title(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(manager, "_active_title", lambda: "main.py - ORION - Visual Studio Code")
    monkeypatch.setattr(manager, "_enumerate_windows", lambda: [])
    snap = manager._capture_sync()
    assert snap.active_context == "programming"


def test_capture_sync_with_no_foreground_window_has_blank_context(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(manager, "_active_title", lambda: "")
    monkeypatch.setattr(manager, "_enumerate_windows", lambda: [])
    snap = manager._capture_sync()
    assert snap.active_context == ""


def test_capture_sync_with_an_unrecognised_app_still_gets_a_context(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(manager, "_active_title", lambda: "Untitled - Notepad")
    monkeypatch.setattr(manager, "_enumerate_windows", lambda: [])
    snap = manager._capture_sync()
    assert snap.active_context == "using Notepad"
