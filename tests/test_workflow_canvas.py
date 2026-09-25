"""
Tests for WorkflowChainCanvas (visual workflow builder, Mark XX design-spec)
and its wiring into AutomationDeckView's step editor: clicking a node
selects the matching table row, selecting a table row highlights the
matching node, and the canvas redraws whenever the table's contents change.

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

from orion_core.bus import OrionBus
from orion_core.constants import C
from orion_core.gui.automation_deck import AutomationDeckView
from orion_core.gui.workflow_canvas import WorkflowChainCanvas


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubEngine:
    def __init__(self):
        self.definitions = {}
        self.runs = {}


# ── WorkflowChainCanvas in isolation ────────────────────────────────────────

def test_set_steps_with_no_steps_shows_a_placeholder(_app):
    canvas = WorkflowChainCanvas()
    canvas.set_steps([])
    assert canvas._rects == []


def test_set_steps_creates_one_node_per_step(_app):
    canvas = WorkflowChainCanvas()
    canvas.set_steps([
        {"tool": "web_search", "args": {"query": "x"}},
        {"tool": "send_message", "args": {}},
        {"tool": "note", "args": {}},
    ])
    assert len(canvas._rects) == 3


def test_set_steps_clears_previous_nodes(_app):
    canvas = WorkflowChainCanvas()
    canvas.set_steps([{"tool": "a", "args": {}}, {"tool": "b", "args": {}}])
    assert len(canvas._rects) == 2
    canvas.set_steps([{"tool": "a", "args": {}}])
    assert len(canvas._rects) == 1


def test_highlight_step_changes_the_pen_of_the_selected_node(_app):
    canvas = WorkflowChainCanvas()
    canvas.set_steps([{"tool": "a", "args": {}}, {"tool": "b", "args": {}}])
    canvas.highlight_step(1)
    assert canvas._rects[1].pen().color().name() != canvas._rects[0].pen().color().name()


def test_highlight_step_with_out_of_range_index_selects_nothing(_app):
    canvas = WorkflowChainCanvas()
    canvas.set_steps([{"tool": "a", "args": {}}])
    canvas.highlight_step(99)   # must not raise
    # Asserted against the palette rather than a literal: what matters is that
    # nothing ended up wearing the selected colour, not which hex that is.
    assert canvas._rects[0].pen().color().name().lower() != C.PRI.lower()


# ── wired into AutomationDeckView ───────────────────────────────────────────

def test_adding_a_step_row_populates_the_canvas(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("web_search", '{"query": "x"}', 1)
    assert len(view.canvas._rects) == 1


def test_removing_a_step_row_updates_the_canvas(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("a", "{}", 1)
    view._add_step_row("b", "{}", 1)
    view.steps_table.setCurrentCell(0, 0)
    view._remove_selected_step()
    assert len(view.canvas._rects) == 1


def test_new_workflow_clears_the_canvas(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("a", "{}", 1)
    view._new_workflow()
    assert view.canvas._rects == []


def test_clicking_a_canvas_node_selects_the_matching_table_row(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("a", "{}", 1)
    view._add_step_row("b", "{}", 1)
    view._on_canvas_step_clicked(1)
    assert view.steps_table.currentRow() == 1


def test_selecting_a_table_row_highlights_the_matching_node(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("a", "{}", 1)
    view._add_step_row("b", "{}", 1)
    view.steps_table.selectRow(1)
    assert view.canvas._rects[1].pen().color().name().lower() == C.PRI.lower(), (
        "the selected step should carry the accent")
    assert view.canvas._rects[0].pen().color().name().lower() != C.PRI.lower(), (
        "an unselected step is structure and stays neutral")


def test_invalid_json_in_the_args_column_does_not_crash_the_canvas(_app):
    view = AutomationDeckView(OrionBus(), _StubEngine())
    view._add_step_row("a", "{not valid json", 1)
    assert len(view.canvas._rects) == 1


def test_selecting_a_workflow_from_the_library_populates_the_canvas(_app):
    engine = _StubEngine()
    engine.definitions["greet"] = {
        "description": "", "steps": [{"tool": "note", "args": {}, "retries": 1}]}
    view = AutomationDeckView(OrionBus(), engine)
    view.show()
    view._refresh_library()
    view.library_list.setCurrentRow(0)
    assert len(view.canvas._rects) == 1
