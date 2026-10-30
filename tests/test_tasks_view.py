"""
The TASKS page reads the durable job store and must say what is true.

States in words (never colour alone), progress only when the work reported
some, approvals surfaced at the top with a decision that goes back through the
job manager, and no work at all while the page is hidden.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui.tasks_view import STATE_WORDS, TasksView, progress_text  # noqa: E402
from orion_core.job_store import JobStore, State  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


class _Jobs:
    """The JobManager surface the page uses, over a real in-memory store."""

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.calls: list[tuple] = []

    def snapshot(self, limit: int = 50, since_seq: int = 0):
        snap = self.store.snapshot(limit=limit, since_seq=since_seq)
        snap["approvals"] = self.store.pending_approvals()
        snap["admission"] = {"resources": {"memory": "normal", "cpu": "ok", "gpu": "unknown"},
                             "classes": {"user": {"running": 1, "pending": 2}}}
        return snap

    def pause(self, job_id):
        self.calls.append(("pause", job_id))
        return "Paused."

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))
        return "Cancelled."

    def reprioritise(self, job_id, cls):
        self.calls.append(("priority", job_id, cls))
        return "Reprioritised."

    def approve(self, approval_id, approve=True):
        self.calls.append(("approve", approval_id, approve))
        return "Recorded."


def _store_with_jobs() -> tuple[JobStore, dict[str, str]]:
    store = JobStore(None)
    ids = {}
    running = store.create(kind="plan", name="tidy downloads", total=4)
    store.transition(running["id"], State.RUNNING)
    store.update(running["id"], fraction=0.5, step=2, note="moving invoices")
    ids["running"] = running["id"]
    vague = store.create(kind="tool", name="deep research")
    store.transition(vague["id"], State.RUNNING)
    ids["vague"] = vague["id"]
    waiting = store.create(kind="workflow", name="send the weekly report",
                           steps=[{"tool": "send_email", "args": {"to": "x"}}])
    store.transition(waiting["id"], State.RUNNING)
    store.request_approval(waiting["id"], 0, "send_email", {"to": "x"},
                           reason="sends an e-mail")
    ids["waiting"] = waiting["id"]
    return store, ids


def test_progress_is_only_a_number_when_the_work_reported_one():
    assert progress_text({"state": "running", "fraction": 0.5, "step": 2, "total": 4}) == \
        "50% (step 2 of 4)"
    assert progress_text({"state": "running"}) == "in progress"
    assert progress_text({"state": "queued"}) == "not started"
    assert progress_text({"state": "completed", "fraction": 1.0}) == "—"


def test_the_page_lists_every_job_with_its_state_in_words(app):
    store, ids = _store_with_jobs()
    view = TasksView(OrionBus(), jobs=_Jobs(store))
    view.refresh()
    rows = {view.table.item(r, 0).text(): [view.table.item(r, c).text() for c in range(6)]
            for r in range(view.table.rowCount())}
    assert rows["tidy downloads"][2] == "Running"
    assert rows["tidy downloads"][3] == "50% (step 2 of 4)"
    assert rows["deep research"][3] == "in progress"
    assert rows["send the weekly report"][2] == STATE_WORDS["waiting_for_approval"]
    assert "running" in view.summary.text() and "waiting for approval" in view.summary.text()
    assert "graphics unknown" in view.capacity.text()
    assert view.empty_state.isHidden()


def test_empty_page_points_to_the_task_composer(app):
    view = TasksView(OrionBus(), jobs=_Jobs(JobStore(None)))
    view.refresh()
    assert view.table.isHidden()
    assert not view.empty_state.isHidden()
    assert [label.text() for label in view.empty_state.findChildren(QLabel)] == ["No tasks yet"]


def test_an_approval_is_shown_and_the_decision_goes_back_through_the_manager(app):
    store, _ids = _store_with_jobs()
    jobs = _Jobs(store)
    view = TasksView(OrionBus(), jobs=jobs)
    view.refresh()
    assert not view.approvals.isHidden() and view.approvals.count() == 1
    assert "send_email" in view.approvals.item(0).text()
    view.approvals.setCurrentRow(0)
    view._decide(False)
    approval_id = store.pending_approvals()[0]["id"]
    assert jobs.calls[-1] == ("approve", approval_id, False)


def test_actions_apply_to_the_selected_job(app):
    store, ids = _store_with_jobs()
    jobs = _Jobs(store)
    view = TasksView(OrionBus(), jobs=jobs)
    view.refresh()
    row = view._job_ids.index(ids["running"])
    view.table.selectRow(row)
    view._act("pause")
    view._act("priority")
    assert ("pause", ids["running"]) in jobs.calls
    assert ("priority", ids["running"], "user") in jobs.calls


def test_a_hidden_page_does_no_work(app):
    store, _ids = _store_with_jobs()
    reads = []

    class _Counting(_Jobs):
        def snapshot(self, *a, **k):
            reads.append(1)
            return super().snapshot(*a, **k)

    view = TasksView(OrionBus(), jobs=_Counting(store))
    view._tick()
    view._tick()
    assert reads == []
