"""
Tests for the Mission Deck MEMORY VIEWER consolidation (Mark XX design-spec
§3, Medium item 12): this dock used to be an independent, less capable
text-only renderer (tier counts + latest six records, no search) — a third
reimplementation of "browse memory" alongside OPS's MemoryViewerPanel
(which has search) and the standalone MemoryMatrixView deck page. It now
embeds the actual MemoryViewerPanel widget instead of duplicating it.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.gui.mission_deck import MissionDeckView
from orion_core.gui.ops_deck import MemoryViewerPanel


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Signal:
    def emit(self, *a, **k) -> None:
        pass

    def connect(self, *a, **k) -> None:
        pass


class _StubBus:
    def __getattr__(self, name):
        return _Signal()


class _StubMemory:
    def __init__(self, records) -> None:
        self._records = records
        self.queries: list[str] = []

    def records(self, query="", limit=40):
        self.queries.append(query)
        q = query.lower()
        return [r for r in self._records if q in r["value"].lower()] if q else self._records

    def tiers_snapshot(self):
        return {"session": 3, "knowledge": 12}


def test_memory_viewer_dock_embeds_the_real_memory_viewer_panel(_app):
    view = MissionDeckView(_StubBus())
    assert isinstance(view._memory_widget, MemoryViewerPanel)


def test_memory_viewer_is_excluded_from_the_generic_text_panel_machinery(_app):
    view = MissionDeckView(_StubBus())
    assert "MEMORY VIEWER" not in view._panels
    assert "MEMORY VIEWER" not in view._renderers


def test_old_text_only_render_memory_method_is_gone(_app):
    assert not hasattr(MissionDeckView, "_render_memory")


def test_embedded_memory_widget_actually_searches(_app):
    memory = _StubMemory([
        {"category": "knowledge", "key_ref": "a", "value": "orion loves chess"},
        {"category": "knowledge", "key_ref": "b", "value": "unrelated fact"},
    ])
    view = MissionDeckView(_StubBus(), memory=memory)
    view._memory_widget.query.setText("chess")
    view._memory_widget.refresh()
    text = view._memory_widget.listing.item(0).text()
    assert "chess" in text.lower()


def test_periodic_refresh_re_runs_the_current_search(_app):
    memory = _StubMemory([{"category": "k", "key_ref": "a", "value": "alpha"}])
    view = MissionDeckView(_StubBus(), memory=memory)
    view._memory_widget.query.setText("alpha")
    view.show()
    view._refresh()
    assert "alpha" in memory.queries


def test_memory_widget_degrades_cleanly_with_no_memory_agent(_app):
    view = MissionDeckView(_StubBus())   # memory=None
    assert view._memory_widget.listing.item(0).text() == "Memory matrix not available."
