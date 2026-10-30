"""
TASKS — everything ORION is doing, waiting on, or has finished, in one place.

Plans, workflows and background tool jobs used to report through three
different surfaces (the log, the automation page, a spoken summary), and a job
waiting for approval was only discoverable by asking. The durable job store is
now the single owner of all three, so this page reads that one record:

  * each job's state in words — queued, running, waiting for approval, paused,
    completed, failed, cancelled, interrupted — never colour alone;
  * progress as a real fraction when the work reports one, and an honest
    "in progress" when it does not (no invented percentages);
  * pending approvals with Approve and Decline, bound to the exact step they
    were raised for (the job store revalidates them before anything runs);
  * pause, resume, cancel and priority for the selected job;
  * the most recent tool actions and the machine's admission pressure.

Cheap by construction: updates arrive as bus events and only mark the page
dirty; it redraws at most twice a second, and not at all while hidden.
"""

from __future__ import annotations

import time
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus

#: The words each state is shown with. The store's own values are stable
#: identifiers; these are what a person reads.
STATE_WORDS = {
    "queued": "Queued",
    "running": "Running",
    "waiting_for_approval": "Waiting for approval",
    "paused": "Paused",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "interrupted": "Interrupted — can be resumed",
}

REFRESH_MS = 500


def progress_text(job: dict[str, Any]) -> str:
    """A job's progress in words. A fraction only when the work reported one."""
    state = str(job.get("state") or "")
    if state in {"completed", "failed", "cancelled"}:
        return "—"
    fraction = job.get("fraction")
    step, total = int(job.get("step") or 0), int(job.get("total") or 0)
    if isinstance(fraction, (int, float)):
        text = f"{max(0.0, min(1.0, float(fraction))) * 100:.0f}%"
        return f"{text} (step {step} of {total})" if total else text
    if total:
        return f"step {step} of {total}"
    return "in progress" if state == "running" else "not started"


def _ago(stamp: Any) -> str:
    try:
        seconds = max(0.0, time.time() - float(stamp))
    except (TypeError, ValueError):
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    return time.strftime("%H:%M", time.localtime(float(stamp)))


class TasksView(QWidget):
    """The deck's TASKS page."""

    COLUMNS = ("Task", "Kind", "State", "Progress", "Priority", "Updated")

    def __init__(self, bus: OrionBus, jobs: Any = None, dispatcher: Any = None,
                 router: Any = None) -> None:
        super().__init__()
        self.bus = bus
        self.jobs = jobs
        self.dispatcher = dispatcher
        self.router = router
        self._planned_steps: list[dict[str, Any]] = []
        self._planned_goal = ""
        self._planning = False
        self._dirty = True
        self._job_ids: list[str] = []
        self._approval_ids: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        title = QLabel("Tasks")
        title.setObjectName("titleLabel")
        root.addWidget(title)
        self.summary = QLabel("")
        self.summary.setObjectName("mutedLabel")
        self.summary.setAccessibleName("Task summary")
        root.addWidget(self.summary)

        composer = QFrame()
        composer.setObjectName("panelFrame")
        compose_layout = QVBoxLayout(composer)
        compose_layout.addWidget(QLabel("New Task"))
        entry = QHBoxLayout()
        self.task_entry = QLineEdit()
        self.task_entry.setMaxLength(4000)
        self.task_entry.setPlaceholderText("Describe the task")
        self.task_entry.setAccessibleName("New task instruction")
        self.task_entry.textChanged.connect(self._invalidate_plan)
        self.task_entry.returnPressed.connect(self._prepare_task)
        self.plan_button = QPushButton("Plan Task")
        self.plan_button.clicked.connect(self._prepare_task)
        self.run_button = QPushButton("Start Reviewed Task")
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._start_task)
        entry.addWidget(self.task_entry, 1)
        entry.addWidget(self.plan_button)
        entry.addWidget(self.run_button)
        compose_layout.addLayout(entry)
        self.plan_preview = QPlainTextEdit()
        self.plan_preview.setReadOnly(True)
        self.plan_preview.setMaximumHeight(100)
        self.plan_preview.setAccessibleName("Task plan for review")
        # Shown only while there is a plan to review: empty, it held 100 px of
        # an 800x600 page while the task table got one row (audit 6 Oct 2026).
        self.plan_preview.setVisible(False)
        compose_layout.addWidget(self.plan_preview)
        self.compose_status = QLabel("")
        self.compose_status.setWordWrap(True)
        compose_layout.addWidget(self.compose_status)
        root.addWidget(composer)

        # ── approvals: first, because they are waiting on the user ──────────
        self.approvals_heading = QLabel("Waiting for Your Approval")
        self.approvals_heading.setObjectName("panelHeading")
        root.addWidget(self.approvals_heading)
        self.approvals = QListWidget()
        self.approvals.setAccessibleName("Steps waiting for approval")
        self.approvals.setMaximumHeight(110)
        root.addWidget(self.approvals)
        approval_row = QHBoxLayout()
        self.approve_button = QPushButton("Approve Step")
        self.approve_button.setAccessibleName("Approve the selected step")
        self.approve_button.clicked.connect(lambda: self._decide(True))
        self.decline_button = QPushButton("Decline Step")
        self.decline_button.setAccessibleName("Decline the selected step")
        self.decline_button.clicked.connect(lambda: self._decide(False))
        approval_row.addWidget(self.approve_button)
        approval_row.addWidget(self.decline_button)
        approval_row.addStretch(1)
        root.addLayout(approval_row)

        # ── the jobs ──────────────────────────────────────────────────────────
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setAccessibleName("Tasks")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, len(self.COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._show_detail)
        # Room for the header and four tasks however small the page gets.
        self.table.setMinimumHeight(
            self.table.horizontalHeader().sizeHint().height()
            + 4 * self.table.verticalHeader().defaultSectionSize() + 4)
        root.addWidget(self.table, 1)

        self.empty_state = QFrame()
        self.empty_state.setObjectName("panelFrame")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.addStretch(1)
        empty_heading = QLabel("No tasks yet")
        empty_heading.setObjectName("panelHeading")
        empty_heading.setFont(QFont("Segoe UI", 18, QFont.Weight.DemiBold))
        empty_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_heading)
        empty_layout.addStretch(1)
        root.addWidget(self.empty_state, 1)

        controls = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key, label, tip in (
                ("pause", "Pause", "Pause the selected task at its next safe point"),
                ("resume", "Resume", "Resume the selected paused or interrupted task"),
                ("cancel", "Cancel", "Cancel the selected task"),
                ("priority", "Run Sooner", "Give the selected task user priority")):
            button = QPushButton(label)
            button.setAccessibleName(tip)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, k=key: self._act(k))
            controls.addWidget(button)
            self.buttons[key] = button
        controls.addStretch(1)
        root.addLayout(controls)

        self.detail = QLabel("")
        self.detail.setObjectName("mutedLabel")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setAccessibleName("Selected task detail")
        root.addWidget(self.detail)

        # ── recent actions and machine pressure ──────────────────────────────
        lower = QHBoxLayout()
        recent_box = QVBoxLayout()
        recent_heading = QLabel("Recent Actions")
        recent_heading.setObjectName("panelHeading")
        recent_box.addWidget(recent_heading)
        self.recent = QListWidget()
        self.recent.setAccessibleName("Recent tool actions")
        self.recent.setMaximumHeight(120)
        recent_box.addWidget(self.recent)
        lower.addLayout(recent_box, 2)
        pressure = QFrame()
        pressure.setObjectName("panelFrame")
        pressure_box = QVBoxLayout(pressure)
        heading = QLabel("Capacity")
        heading.setObjectName("panelHeading")
        pressure_box.addWidget(heading)
        self.capacity = QLabel("")
        self.capacity.setWordWrap(True)
        self.capacity.setAccessibleName("Machine capacity")
        pressure_box.addWidget(self.capacity)
        pressure_box.addStretch(1)
        lower.addWidget(pressure, 1)
        root.addLayout(lower)

        try:
            bus.dashboard_event.connect(self._on_event)
        except Exception:
            pass
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── updates ──────────────────────────────────────────────────────────────

    def _on_event(self, channel: str, _payload: Any) -> None:
        if channel == "task":
            self._dirty = True

    def _tick(self) -> None:
        # Tool activity changes the recent list too, so a visible page also
        # redraws on a slow clock; a hidden one does nothing at all.
        if not self.isVisible():
            return
        self.refresh()

    def showEvent(self, event: Any) -> None:  # noqa: N802 — Qt override
        super().showEvent(event)
        self.refresh()

    def _snapshot(self) -> dict[str, Any]:
        if self.jobs is not None and hasattr(self.jobs, "snapshot"):
            return self.jobs.snapshot(limit=60)
        from .. import job_store
        return job_store.default_store().snapshot(limit=60)

    def refresh(self) -> None:
        try:
            snap = self._snapshot()
        except Exception as exc:
            self.summary.setText(f"Tasks are unavailable — {exc}")
            return
        if not isinstance(snap, dict):
            self.summary.setText("Tasks are unavailable")
            return
        self._dirty = False
        self._render_summary(snap)
        self._render_approvals(snap.get("approvals") or [], snap.get("jobs") or [])
        self._render_jobs(snap.get("jobs") or [])
        self._render_recent()
        self._render_capacity(snap.get("admission") or {})

    def _render_summary(self, snap: dict[str, Any]) -> None:
        counts = snap.get("counts") or {}
        live = [(state, counts.get(state, 0)) for state in
                ("running", "queued", "waiting_for_approval", "paused", "interrupted")]
        parts = [f"{n} {STATE_WORDS[s].split(' —')[0].lower()}" for s, n in live if n]
        failed = counts.get("failed", 0)
        if failed:
            parts.append(f"{failed} failed")
        self.summary.setText(", ".join(parts) if parts else "Nothing in progress.")

    def _render_approvals(self, approvals: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> None:
        from ..jobs import approval_step_number
        kinds = {str(job.get("id")): str(job.get("kind") or "") for job in jobs}
        ids = [str(a.get("id")) for a in approvals]
        if ids != self._approval_ids:
            self._approval_ids = ids
            self.approvals.clear()
            for approval in approvals:
                job_id = str(approval.get("job_id"))
                kind = kinds.get(job_id)
                # Approvals can outnumber the limited task-table snapshot.
                # Look up only missing job metadata rather than infer its basis.
                if kind is None:
                    store = getattr(self.jobs, "store", None)
                    if store is not None:
                        try:
                            kind = str((store.get(job_id) or {}).get("kind") or "")
                        except Exception:
                            kind = ""
                reason = str(approval.get("reason") or "").strip()
                expires = approval.get("expires_at")
                left = ""
                try:
                    left = f" — expires in {max(0, int(float(expires) - time.time())) // 60} min"
                except (TypeError, ValueError):
                    pass
                raw_step = int(approval.get("step_idx", 0))
                label = (f"step {approval_step_number(raw_step, kind)}" if kind else
                         f"recorded step {raw_step}")
                self.approvals.addItem(
                    f"{approval.get('tool', 'a step')} ({label})"
                    + (f": {reason}" if reason else "") + left)
        visible = bool(approvals)
        for widget in (self.approvals_heading, self.approvals,
                       self.approve_button, self.decline_button):
            widget.setVisible(visible)

    def _render_jobs(self, jobs: list[dict[str, Any]]) -> None:
        selected = self._selected_job_id()
        self._job_ids = [str(j.get("id")) for j in jobs]
        self.table.setVisible(bool(jobs))
        self.empty_state.setVisible(not jobs)
        self.table.blockSignals(True)
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            state = str(job.get("state") or "")
            name = str(job.get("label") or job.get("name") or job.get("id"))
            position = job.get("position")
            if state == "queued" and position:
                state_text = f"Queued — number {position} in line"
            else:
                state_text = STATE_WORDS.get(state, state or "unknown")
            values = (name, str(job.get("kind") or ""), state_text, progress_text(job),
                      str(job.get("cls") or ""), _ago(job.get("updated_at")))
            for column, value in enumerate(values):
                item = self.table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(row, column, item)
                if item.text() != value:
                    item.setText(value)
        if selected in self._job_ids:
            self.table.selectRow(self._job_ids.index(selected))
        elif selected:
            self.table.clearSelection()
        self.table.blockSignals(False)
        self._show_detail()

    def _render_recent(self) -> None:
        recent = list(getattr(self.dispatcher, "recent_tools", []) or [])[-12:]
        lines = [f"{r.get('at', '')}  {r.get('tool', '?')}  "
                 f"{'ok' if r.get('ok') else 'failed'}  {float(r.get('ms', 0)):.0f} ms"
                 for r in reversed(recent)] or ["No actions yet this session."]
        current = [self.recent.item(i).text() for i in range(self.recent.count())]
        if lines != current:
            self.recent.clear()
            self.recent.addItems(lines)

    def _render_capacity(self, admission: dict[str, Any]) -> None:
        resources = admission.get("resources") or {}
        classes = admission.get("classes") or {}
        lines = []
        if resources:
            lines.append(
                f"Memory {resources.get('memory', 'unknown')}, "
                f"processor {resources.get('cpu', 'unknown')}, "
                f"graphics {resources.get('gpu', 'unknown')}.")
        busy = [f"{name}: {info.get('running', 0)} running, {info.get('pending', 0)} waiting"
                for name, info in classes.items()
                if info.get("running") or info.get("pending")]
        lines.extend(busy or ["No queued work."])
        self.capacity.setText("\n".join(lines))

    # ── actions ──────────────────────────────────────────────────────────────

    def _selected_job_id(self) -> str:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return ""
        index = rows[0].row()
        return self._job_ids[index] if 0 <= index < len(self._job_ids) else ""

    def _show_detail(self) -> None:
        job_id = self._selected_job_id()
        enabled = bool(job_id) and self.jobs is not None
        for button in self.buttons.values():
            button.setEnabled(False)
        if not job_id:
            self.detail.setText("")
            return
        try:
            snap_jobs = {str(j.get("id")): j for j in self._snapshot().get("jobs") or []}
        except Exception:
            return
        job = snap_jobs.get(job_id) or {}
        state = job.get("state")
        for key, states in {
            "pause": {"running", "queued", "waiting_for_approval"},
            "resume": {"paused", "interrupted"},
            "cancel": {"running", "queued", "waiting_for_approval", "paused", "interrupted"},
            "priority": {"running", "queued", "paused", "interrupted"},
        }.items():
            self.buttons[key].setEnabled(enabled and state in states)
        note = str(job.get("note") or "").strip()
        result = str(job.get("result") or "").strip()
        text = f"{job.get('label') or job.get('name') or job_id} ({job_id})"
        if note:
            text += f"\nLatest: {note[:300]}"
        if result:
            text += f"\nResult: {result[:600]}"
        self.detail.setText(text)

    def _invalidate_plan(self, *_args: Any) -> None:
        self._planned_steps = []
        self._planned_goal = ""
        self.run_button.setEnabled(False)
        self.plan_preview.clear()
        self.plan_preview.setVisible(False)

    def _prepare_task(self) -> None:
        goal = self.task_entry.text().strip()
        if not goal or self._planning:
            return
        executor = getattr(self.dispatcher, "plan_executor", None)
        if executor is None:
            self.compose_status.setText("Tasks are unavailable")
            return
        from .. import background
        self._invalidate_plan()
        self._planning = True
        self.plan_button.setEnabled(False)
        self.compose_status.setText("Planning…")

        async def prepare() -> None:
            import asyncio
            from ..plan_executor import validate_plan
            from ..remote_capability import describe_action
            from ..remote_tasks import notepad_plan, plan_request
            try:
                steps = notepad_plan(goal)
                if steps is None:
                    if self.router is None:
                        raise ValueError("The task planner is unavailable. Nothing has run.")
                    async with asyncio.timeout(60):
                        steps = await plan_request(self.router, self.dispatcher, goal,
                                                   "The user is at the PC. Plan only the requested actions.",
                                                   target="desktop")
                _parsed, problems = validate_plan(steps, executor.catalogue())
                if problems:
                    raise ValueError(problems[0])
                if sip.isdeleted(self):
                    return
                if self.task_entry.text().strip() != goal:
                    self.compose_status.setText("The task changed - plan it again")
                    return
                self._planned_steps, self._planned_goal = steps, goal
                self.plan_preview.setPlainText("\n".join(
                    f"{i}. {describe_action(s['tool'], s['args'])}"
                    for i, s in enumerate(steps, 1)))
                self.plan_preview.setVisible(True)
                self.run_button.setEnabled(True)
                self.compose_status.setText("Review the plan, then start")
            except asyncio.TimeoutError:
                if not sip.isdeleted(self):
                    self.compose_status.setText("Planning timed out")
            except Exception as exc:
                if not sip.isdeleted(self):
                    self.compose_status.setText(f"Could not prepare the task: {str(exc)[:300]}")
            finally:
                self._planning = False
                if not sip.isdeleted(self):
                    self.plan_button.setEnabled(True)
        background.spawn(prepare(), name="orion-task-plan-preview")

    def _start_task(self) -> None:
        if not self._planned_steps or self.task_entry.text().strip() != self._planned_goal:
            return
        from .. import background
        steps, goal = self._planned_steps, self._planned_goal
        executor = getattr(self.dispatcher, "plan_executor", None)
        if executor is None:
            return
        self._invalidate_plan()
        self.compose_status.setText("Task started")

        async def run() -> None:
            try:
                # A retry could duplicate a text edit. Reviewed tasks run each
                # step once and preserve their durable result.
                result = await executor.execute(steps, objective=goal, max_retries=0)
                if not sip.isdeleted(self):
                    self.compose_status.setText(str(result.text)[:500])
                self._report(result)
            except Exception as exc:
                message = f"Task failed: {str(exc)[:300]}"
                if not sip.isdeleted(self):
                    self.compose_status.setText(message)
                self._report(message)
        background.spawn(run(), name="orion-reviewed-task")

    def _report(self, result: Any) -> None:
        text = str(getattr(result, "text", result) or "").strip()
        if text:
            try:
                self.bus.log.emit(f"TASKS: {text.splitlines()[0][:160]}")
            except Exception:
                pass
        # The durable work belongs to the job manager, not this page. Let it
        # finish and report to the bus after the deck has disposed its widgets.
        if sip.isdeleted(self):
            return
        if text:
            self.detail.setText(text[:600])
        self.refresh()

    def _act(self, action: str) -> None:
        job_id = self._selected_job_id()
        if not job_id or self.jobs is None:
            return
        try:
            if action == "pause":
                self._report(self.jobs.pause(job_id))
            elif action == "cancel":
                self._report(self.jobs.cancel(job_id))
            elif action == "priority":
                self._report(self.jobs.reprioritise(job_id, "user"))
            elif action == "resume":
                from .. import background

                async def resume() -> None:
                    try:
                        self._report(await self.jobs.resume(job_id))
                    except Exception as exc:
                        self._report(f"Could not resume: {exc}")
                background.spawn(resume(), name="orion-resume-task")
        except Exception as exc:
            self._report(f"Could not {action} that task: {exc}")

    def _decide(self, approve: bool) -> None:
        index = self.approvals.currentRow()
        if not (0 <= index < len(self._approval_ids)) or self.jobs is None:
            return
        try:
            self._report(self.jobs.approve(self._approval_ids[index], approve=approve))
        except Exception as exc:
            self._report(f"Could not record that decision: {exc}")


__all__ = ["TasksView", "STATE_WORDS", "progress_text"]
