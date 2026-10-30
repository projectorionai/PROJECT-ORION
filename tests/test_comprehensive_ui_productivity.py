"""Automation controls must preserve the visible target and complete edits."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from orion_core import jobs
from orion_core.bus import OrionBus
from orion_core.cognition import CognitiveStateManager
from orion_core.gui.automation_deck import AutomationDeckView
from orion_core.gui.ops_deck import TaskQueuePanel
from orion_core.gui.tasks_view import TasksView
from orion_core.job_store import JobStore, State
from orion_core.workflow_engine import WorkflowEngine


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def deck(app, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    store = JobStore(tmp_path / "jobs.db")
    dispatcher = SimpleNamespace(handler_table=lambda: {"web_search": None, "write_file": None})
    engine = WorkflowEngine(OrionBus(), dispatcher, path=tmp_path / "workflows.json", store=store)
    view = AutomationDeckView(OrionBus(), engine, dispatcher)
    view._timer.stop()
    yield view, engine
    view.close()
    view.deleteLater()
    app.processEvents()
    store.close()


def running_pair(view, engine):
    """Real cancellation selection, with harmless tasks recording effects."""
    cancelled = []
    for key in ("first", "second"):
        engine.runs[key] = {"id": key, "workflow": key, "status": "running",
                            "step": 1, "total": 2,
                            "task": SimpleNamespace(done=lambda: False,
                                cancel=lambda key=key: cancelled.append(key))}
    view.show()
    view._refresh_runs()
    return cancelled


def test_no_selected_run_never_invokes_cancel_all(deck):
    view, engine = deck
    cancelled = running_pair(view, engine)
    view._cancel_run()
    assert cancelled == []
    assert "select" in view.status_label.text().lower()


def test_refresh_preserves_selected_identity_when_run_order_changes(deck):
    view, engine = deck
    cancelled = running_pair(view, engine)
    view.runs_list.setCurrentRow(1)
    engine.runs = dict(reversed(list(engine.runs.items())))
    view._refresh_runs()
    item = view.runs_list.currentItem()
    assert item is not None and item.data(Qt.ItemDataRole.UserRole) == "second"
    view._cancel_run()
    assert cancelled == ["second"]


def test_disappeared_selection_does_not_target_a_different_run(deck):
    view, engine = deck
    cancelled = running_pair(view, engine)
    view.runs_list.setCurrentRow(1)
    del engine.runs["second"]
    view._refresh_runs()
    view._cancel_run()
    assert cancelled == []
    assert "select" in view.status_label.text().lower()


def test_cleared_selection_is_not_the_old_current_item(deck):
    view, engine = deck
    cancelled = running_pair(view, engine)
    view.runs_list.setCurrentRow(1)
    view.runs_list.clearSelection()
    view._cancel_run()
    assert cancelled == []
    assert "select" in view.status_label.text().lower()


def saved_workflow(view, engine):
    assert engine.define("saved", [{"tool": "web_search", "args": {"query": "keep me"},
                                    "retries": 0}], "original description").ok
    view._refresh_library()
    view.library_list.setCurrentRow(0)
    return copy.deepcopy(engine.definitions), engine.path.read_bytes()


@pytest.mark.parametrize("empty_row", [0, 1, 2])
def test_blank_tool_row_refuses_entire_edit_instead_of_dropping_step(deck, empty_row):
    view, engine = deck
    original, saved_bytes = saved_workflow(view, engine)
    view.steps_table.setRowCount(0)
    for row in range(3):
        view._add_step_row("" if row == empty_row else "web_search", '{"query":"edited"}', 0)
    view._save_workflow()
    assert engine.definitions == original
    assert engine.path.read_bytes() == saved_bytes
    assert view.steps_table.rowCount() == 3
    assert f"Row {empty_row + 1}" in view.status_label.text()


@pytest.mark.parametrize("retries", ["many", "1.5", ""])
def test_invalid_retry_text_is_not_silently_changed_to_one(deck, retries):
    view, engine = deck
    original, saved_bytes = saved_workflow(view, engine)
    view.steps_table.item(0, 2).setText(retries)
    view._save_workflow()
    assert engine.definitions == original
    assert engine.path.read_bytes() == saved_bytes
    assert view.steps_table.item(0, 2).text() == retries
    assert "retries" in view.status_label.text().lower()


@pytest.mark.parametrize("failure", ["save_denied", "external_change"])
def test_delete_failure_keeps_editor_selection_and_backend_message(deck, monkeypatch, failure):
    view, engine = deck
    original, _saved_bytes = saved_workflow(view, engine)
    view.description_input.setText("unsaved notes I still need")
    if failure == "save_denied":
        monkeypatch.setattr(engine, "_save", lambda *_: False)
    else:
        engine.path.write_text("{externally changed", encoding="utf-8")
    before = engine.path.read_bytes()
    view._delete_selected_workflow()
    assert engine.definitions == original
    assert engine.path.read_bytes() == before
    assert view._selected_workflow_name() == "saved"
    assert view.name_input.text() == "saved"
    assert view.description_input.text() == "unsaved notes I still need"
    assert view.steps_table.rowCount() == 1
    assert "could not be removed" in view.status_label.text()


def test_successful_delete_clears_editor_but_retains_confirmation(deck):
    view, engine = deck
    saved_workflow(view, engine)
    view._delete_selected_workflow()
    assert engine.definitions == {}
    assert view.name_input.text() == "" and view.steps_table.rowCount() == 0
    assert "removed" in view.status_label.text()


def test_valid_nested_arguments_and_zero_retries_round_trip_unchanged(deck):
    view, engine = deck
    view.name_input.setText("complete")
    view._add_step_row("write_file", '{"content":"literal {prev}","items":[0,{"x":false}]}', 0)
    view._save_workflow()
    assert engine.definitions["complete"]["steps"] == [{"tool": "write_file", "retries": 0,
        "args": {"content": "literal {prev}", "items": [0, {"x": False}]}}]


@pytest.mark.parametrize("title", ["Budget [draft]", "Budget  (due tomorrow)"])
def test_ops_completion_uses_exact_task_key_not_decorated_title(app, tmp_path, title):
    cognition = CognitiveStateManager(OrionBus(), SimpleNamespace(), path=tmp_path / "cognitive.json")
    decoy = cognition.add_task("Budget")
    target = cognition.add_task(title, project="finances", due="2026-11-01")
    panel = TaskQueuePanel(OrionBus(), cognition)
    try:
        row = next(i for i in range(panel.listing.count()) if title in panel.listing.item(i).text())
        panel.listing.setCurrentRow(row)
        panel._complete()
        tasks = cognition.snapshot()["pending_tasks"]
        assert target["id"] not in tasks
        assert decoy["id"] in tasks
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_ops_refresh_keeps_the_selected_task_identity(app, tmp_path):
    cognition = CognitiveStateManager(OrionBus(), SimpleNamespace(), path=tmp_path / "cognitive.json")
    cognition.add_task("First", due="2026-11-02")
    target = cognition.add_task("Second", due="2026-11-01")
    panel = TaskQueuePanel(OrionBus(), cognition)
    try:
        panel.listing.setCurrentRow(0)
        cognition.add_task("New earlier task", due="2026-10-31")
        panel.refresh()
        item = panel.listing.currentItem()
        assert item is not None and item.data(Qt.ItemDataRole.UserRole) == target["id"]
    finally:
        panel.deleteLater()
        app.processEvents()


def test_ops_empty_placeholder_never_calls_complete_task(app):
    completed = []
    cognition = SimpleNamespace(snapshot=lambda: {"pending_tasks": {}},
                                complete_task=lambda key: completed.append(key))
    panel = TaskQueuePanel(OrionBus(), cognition)
    try:
        panel.listing.setCurrentRow(0)
        panel._complete()
        assert completed == []
    finally:
        panel.deleteLater()
        app.processEvents()


def test_ops_stale_row_cannot_fuzzy_complete_another_task(app, tmp_path):
    cognition = CognitiveStateManager(OrionBus(), SimpleNamespace(), path=tmp_path / "cognitive.json")
    target = cognition.add_task("Budget draft")
    other = cognition.add_task("Budget draft extra")
    panel = TaskQueuePanel(OrionBus(), cognition)
    try:
        panel.listing.setCurrentRow(0)
        cognition.remove_task(target["id"])
        panel._complete()
        assert other["id"] in cognition.snapshot()["pending_tasks"]
    finally:
        panel.deleteLater()
        app.processEvents()


def test_exact_task_completion_keeps_default_fuzzy_voice_contract(tmp_path):
    cognition = CognitiveStateManager(OrionBus(), SimpleNamespace(), path=tmp_path / "cognitive.json")
    target = cognition.add_task("Water plants")
    assert not cognition.complete_task("Water", exact=True)
    assert target["id"] in cognition.snapshot()["pending_tasks"]
    assert cognition.complete_task("Water")  # voice/name convenience remains supported
    next_task = cognition.add_task("Feed cat")
    assert cognition.complete_task(next_task["id"], exact=True)


@pytest.fixture
def approval_manager(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    store = JobStore(tmp_path / "approval_jobs.db")
    async def never_dispatch(*_args):
        raise AssertionError("Approval display tests must not execute tools")
    manager = jobs.JobManager(OrionBus(), never_dispatch, store=store)
    yield manager, store
    store.close()


def make_approval(store, kind, index):
    record = store.create(kind=kind, name="review this action", state=State.RUNNING,
        steps=[{"index": index, "tool": "write_file", "args": {"content": "reviewed"}}])
    approval = store.request_approval(record["id"], index, "write_file", {"content": "reviewed"})
    return record, approval


@pytest.mark.parametrize("kind,index,display", [("workflow", 1, 1), ("plan", 2, 2), ("tool", 0, 1)])
def test_tasks_approval_labels_follow_the_job_index_basis(app, approval_manager, kind, index, display):
    manager, store = approval_manager
    record, approval = make_approval(store, kind, index)
    view = TasksView(OrionBus(), jobs=manager)
    view._timer.stop()
    try:
        view.refresh()
        assert view._approval_ids == [approval["id"]]
        assert f"(step {display})" in view.approvals.item(0).text()
        assert store.get(record["id"])["state"] == "waiting_for_approval"
        assert store.approval(approval["id"])["state"] == "pending"
    finally:
        view.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("kind,index,display", [("workflow", 1, 1), ("tool", 0, 1)])
def test_approval_label_can_resolve_kind_outside_limited_task_rows(app, approval_manager, monkeypatch, kind, index, display):
    manager, store = approval_manager
    _record, approval = make_approval(store, kind, index)
    snapshot = manager.snapshot()
    snapshot["jobs"] = []  # pending approval is older than the table's limited rows
    monkeypatch.setattr(manager, "snapshot", lambda **_: snapshot)
    view = TasksView(OrionBus(), jobs=manager)
    view._timer.stop()
    try:
        view.refresh()
        assert view._approval_ids == [approval["id"]]
        assert f"(step {display})" in view.approvals.item(0).text()
    finally:
        view.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("kind,index,display", [("workflow", 1, 1), ("plan", 2, 2), ("tool", 0, 1)])
@pytest.mark.parametrize("approve", [True, False])
def test_approval_confirmation_uses_correct_step_and_keeps_exact_binding(approval_manager, kind, index, display, approve):
    manager, store = approval_manager
    record, approval = make_approval(store, kind, index)
    result = manager.approve(approval["id"], approve=approve)
    assert result.ok and f"step {display} (write_file)" in result.text
    decision = store.approval(approval["id"])
    assert decision["state"] == ("approved" if approve else "denied")
    assert decision["job_id"] == record["id"] and decision["step_idx"] == index
    assert decision["digest"] == approval["digest"]
