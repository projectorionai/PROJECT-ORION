"""
Tests for AutomationDeckView (Mark XX design-spec, "Automation + Development
pages" follow-up) — a real workspace over the already-working WorkflowEngine
backend, not a decorative page. Uses the REAL WorkflowEngine (writing to a
tmp_path JSON file) rather than a stub, so these tests exercise the actual
GUI<->backend wiring, not a mock of it.

Headless (offscreen Qt) — no display required.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.gui.automation_deck import AutomationDeckView
from orion_core.workflow_engine import WorkflowEngine


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _StubDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def handler_table(self):
        return {"web_search": None, "open_app": None, "save_memory": None}

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return ToolResult(f"ran {name}")


def _engine(tmp_path, dispatcher=None) -> WorkflowEngine:
    return WorkflowEngine(OrionBus(), dispatcher=dispatcher, path=tmp_path / "workflows.json")


# ── library ──────────────────────────────────────────────────────────────────

def test_library_lists_predefined_workflows(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.define("morning_check", [{"tool": "web_search", "args": {"query": "news"}, "retries": 1}],
                  "Check the morning news")
    view = AutomationDeckView(OrionBus(), engine)
    assert view.library_list.count() == 1
    assert "morning_check" in view.library_list.item(0).text()
    assert "Check the morning news" in view.library_list.item(0).text()


def test_selecting_a_workflow_populates_the_step_editor(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.define("greet", [{"tool": "open_app", "args": {"name": "notepad"}, "retries": 2}], "say hi")
    view = AutomationDeckView(OrionBus(), engine)
    view.library_list.setCurrentRow(0)
    assert view.name_input.text() == "greet"
    assert view.description_input.text() == "say hi"
    assert view.steps_table.rowCount() == 1
    assert view.steps_table.item(0, 0).text() == "open_app"
    assert view.steps_table.item(0, 2).text() == "2"


# ── step editor / save ───────────────────────────────────────────────────────

def test_new_clears_the_editor(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.define("existing", [{"tool": "web_search", "args": {}, "retries": 1}])
    view = AutomationDeckView(OrionBus(), engine)
    view.library_list.setCurrentRow(0)
    view._new_workflow()
    assert view.name_input.text() == ""
    assert view.steps_table.rowCount() == 0


def test_add_step_row_appends_a_row_with_the_picked_tool(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine, dispatcher=_StubDispatcher())
    view.tool_picker.setCurrentText("save_memory")
    view._add_step_row(view.tool_picker.currentText(), "{}", 1)
    assert view.steps_table.rowCount() == 1
    assert view.steps_table.item(0, 0).text() == "save_memory"


def test_tool_picker_is_populated_from_the_dispatcher_handler_table(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine, dispatcher=_StubDispatcher())
    items = [view.tool_picker.itemText(i) for i in range(view.tool_picker.count())]
    assert "web_search" in items
    assert "save_memory" in items


def test_save_workflow_defines_it_on_the_engine_and_refreshes_the_library(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine)
    view.name_input.setText("new_flow")
    view.description_input.setText("does a thing")
    view._add_step_row("web_search", '{"query": "test"}', 1)
    view._save_workflow()
    assert "new_flow" in engine.definitions
    assert view.library_list.count() == 1


def test_save_workflow_without_a_name_reports_an_error_and_does_not_save(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine)
    view._add_step_row("web_search", "{}", 1)
    view._save_workflow()
    assert engine.definitions == {}
    assert "name" in view.status_label.text().lower()


def test_save_workflow_with_invalid_json_args_reports_an_error(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine)
    view.name_input.setText("bad_flow")
    view._add_step_row("web_search", "{not valid json", 1)
    view._save_workflow()
    assert "bad_flow" not in engine.definitions
    assert "json" in view.status_label.text().lower()


def test_delete_selected_workflow_removes_it(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.define("throwaway", [{"tool": "web_search", "args": {}, "retries": 1}])
    view = AutomationDeckView(OrionBus(), engine)
    view.library_list.setCurrentRow(0)
    view._delete_selected_workflow()
    assert "throwaway" not in engine.definitions
    assert view.library_list.count() == 0


# ── running + live status ────────────────────────────────────────────────────

def test_run_without_a_dispatcher_reports_the_engines_own_error(_app, tmp_path):
    engine = _engine(tmp_path, dispatcher=None)
    engine.define("flow", [{"tool": "web_search", "args": {}, "retries": 1}])
    view = AutomationDeckView(OrionBus(), engine)
    view.library_list.setCurrentRow(0)
    view._run_selected_workflow()
    assert "dispatcher is not attached" in view.status_label.text().lower()


def test_run_with_a_dispatcher_starts_a_background_run(_app, tmp_path):
    async def _scenario():
        dispatcher = _StubDispatcher()
        engine = _engine(tmp_path, dispatcher=dispatcher)
        engine.define("flow", [{"tool": "web_search", "args": {"query": "{input}"}, "retries": 0}])
        view = AutomationDeckView(OrionBus(), engine, dispatcher=dispatcher)
        view.show()
        view.library_list.setCurrentRow(0)
        view.run_input.setText("orion news")
        view._run_selected_workflow()
        assert len(engine.runs) == 1
        await asyncio.sleep(0)   # let the background task run at least once
        assert view.runs_list.count() == 1

    asyncio.run(_scenario())


def test_refresh_runs_shows_seeded_run_state(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.runs["r1"] = {"id": "r1", "workflow": "flow", "status": "running", "step": 1, "total": 3}
    view = AutomationDeckView(OrionBus(), engine)
    view.show()
    view._refresh_runs()
    assert view.runs_list.count() == 1
    assert "flow" in view.runs_list.item(0).text()
    assert "1/3" in view.runs_list.item(0).text()


def test_workflow_bus_event_triggers_a_runs_refresh(_app, tmp_path):
    bus = OrionBus()
    engine = _engine(tmp_path)
    view = AutomationDeckView(bus, engine)
    view.show()
    engine.runs["r2"] = {"id": "r2", "workflow": "flow2", "status": "complete", "step": 2, "total": 2}
    bus.dashboard_event.emit("workflow", {"run": "r2", "workflow": "flow2"})
    assert view.runs_list.count() == 1
    assert "flow2" in view.runs_list.item(0).text()


def test_cancel_selected_run_calls_the_engine(_app, tmp_path):
    engine = _engine(tmp_path)
    engine.runs["r3"] = {"id": "r3", "workflow": "flow3", "status": "running", "step": 1, "total": 2}
    view = AutomationDeckView(OrionBus(), engine)
    view.show()   # _refresh_runs() only populates the list while visible
    view._refresh_runs()
    assert view.runs_list.count() == 1   # the selection this test exercises
    view.runs_list.setCurrentRow(0)
    view._cancel_run()
    assert "cancelling 1" in view.status_label.text().lower()


def test_cancel_with_nothing_selected_reports_none_running(_app, tmp_path):
    engine = _engine(tmp_path)
    view = AutomationDeckView(OrionBus(), engine)
    view._cancel_run()
    assert "no running workflow" in view.status_label.text().lower()
