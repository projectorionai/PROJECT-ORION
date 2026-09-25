"""
AutomationDeckView — a real workspace over WorkflowEngine/AutomationManager
(workflow_engine.py), not a decorative page.

Mark XX design-spec: the AUTOMATION zone existed as a disabled, empty tab in
the nav — there was no page to enable it with. The backend it wraps was
never the gap: WorkflowEngine already runs named, ordered tool-call chains
with per-step retries, {input}/{prev} placeholder resolution and live
progress on the bus; it only ever had a read-only text listing inside
Mission Deck. This page makes the whole lifecycle interactive — browse the
library, inspect and edit a workflow's steps, run one, watch it execute,
cancel it, define a new one — reading/writing the SAME WorkflowEngine
instance the dispatcher's `automation` tool already uses, so anything built
or run here is identical to what a voice/text command would produce.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from .workflow_canvas import WorkflowChainCanvas

_REFRESH_MS = 2000


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return frame, layout


class AutomationDeckView(QWidget):
    """The AUTOMATION zone page: library, step editor, live runs."""

    def __init__(self, bus: OrionBus, workflow_engine: Any,
                 dispatcher: Any | None = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.engine = workflow_engine
        self.dispatcher = dispatcher
        self._build_ui()

        try:
            self.bus.dashboard_event.connect(self._on_event)
        except Exception:
            pass
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh_runs)
        self._timer.start()

        self._refresh_library()
        self._refresh_runs()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        title = QLabel("AUTOMATION — WORKFLOW ENGINE")
        title.setObjectName("titleLabel")
        outer.addWidget(title)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_library_panel(), 0, 0)
        grid.addWidget(self._build_editor_panel(), 0, 1)
        grid.addWidget(self._build_runs_panel(), 1, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        outer.addLayout(grid, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

    def _build_library_panel(self) -> QFrame:
        frame, layout = _panel("WORKFLOW LIBRARY")
        self.library_list = QListWidget()
        self.library_list.currentItemChanged.connect(self._on_library_selection)
        layout.addWidget(self.library_list, 1)

        run_row = QHBoxLayout()
        self.run_input = QLineEdit()
        self.run_input.setPlaceholderText("Optional input text ({input} in steps)")
        run_row.addWidget(self.run_input, 1)
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._run_selected_workflow)
        run_row.addWidget(run_btn)
        layout.addLayout(run_row)

        button_row = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new_workflow)
        button_row.addWidget(new_btn)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_selected_workflow)
        button_row.addWidget(delete_btn)
        layout.addLayout(button_row)
        return frame

    def _build_editor_panel(self) -> QFrame:
        frame, layout = _panel("STEP EDITOR")

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self.name_input = QLineEdit()
        name_row.addWidget(self.name_input, 1)
        layout.addLayout(name_row)

        self.description_input = QLineEdit()
        self.description_input.setPlaceholderText("Description (optional)")
        layout.addWidget(self.description_input)

        self.canvas = WorkflowChainCanvas()
        self.canvas.step_clicked.connect(self._on_canvas_step_clicked)
        layout.addWidget(self.canvas)

        self.steps_table = QTableWidget(0, 3)
        self.steps_table.setHorizontalHeaderLabels(["Tool", "Args (JSON)", "Retries"])
        self.steps_table.horizontalHeader().setStretchLastSection(False)
        self.steps_table.setColumnWidth(0, 150)
        self.steps_table.setColumnWidth(2, 60)
        self.steps_table.itemChanged.connect(lambda *_a: self._refresh_canvas())
        self.steps_table.itemSelectionChanged.connect(self._on_step_table_selection)
        layout.addWidget(self.steps_table, 1)

        step_row = QHBoxLayout()
        self.tool_picker = QComboBox()
        self.tool_picker.setEditable(True)
        if self.dispatcher is not None:
            try:
                self.tool_picker.addItems(sorted(self.dispatcher.handler_table().keys()))
            except Exception:
                pass
        step_row.addWidget(self.tool_picker, 1)
        add_step_btn = QPushButton("Add Step")
        add_step_btn.clicked.connect(
            lambda: self._add_step_row(self.tool_picker.currentText(), "{}", 1))
        step_row.addWidget(add_step_btn)
        remove_step_btn = QPushButton("Remove Step")
        remove_step_btn.clicked.connect(self._remove_selected_step)
        step_row.addWidget(remove_step_btn)
        layout.addLayout(step_row)

        save_btn = QPushButton("Save Workflow")
        save_btn.clicked.connect(self._save_workflow)
        layout.addWidget(save_btn)
        return frame

    def _build_runs_panel(self) -> QFrame:
        frame, layout = _panel("RUNS")
        self.runs_list = QListWidget()
        layout.addWidget(self.runs_list, 1)
        cancel_btn = QPushButton("Cancel Selected Run")
        cancel_btn.clicked.connect(self._cancel_run)
        layout.addWidget(cancel_btn)
        return frame

    # ── library ───────────────────────────────────────────────────────────────

    def _refresh_library(self) -> None:
        selected = self._selected_workflow_name()
        self.library_list.clear()
        for name, definition in sorted(self.engine.definitions.items()):
            steps = definition.get("steps") or []
            desc = definition.get("description") or ""
            label = f"{name}  ({len(steps)} step(s))" + (f" — {desc}" if desc else "")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.library_list.addItem(item)
            if name == selected:
                self.library_list.setCurrentItem(item)

    def _selected_workflow_name(self) -> str:
        item = self.library_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""

    def _on_library_selection(self, current: QListWidgetItem, _previous: QListWidgetItem) -> None:
        if current is None:
            return
        name = str(current.data(Qt.ItemDataRole.UserRole))
        definition = self.engine.definitions.get(name, {})
        self.name_input.setText(name)
        self.description_input.setText(str(definition.get("description") or ""))
        self.steps_table.setRowCount(0)
        for step in definition.get("steps") or []:
            self._add_step_row(
                str(step.get("tool") or ""),
                json.dumps(step.get("args") or {}),
                int(step.get("retries", 1)),
            )
        self._refresh_canvas()

    # ── step editor ───────────────────────────────────────────────────────────

    def _add_step_row(self, tool: str, args_json: str, retries: int) -> None:
        row = self.steps_table.rowCount()
        self.steps_table.insertRow(row)
        self.steps_table.setItem(row, 0, QTableWidgetItem(tool))
        self.steps_table.setItem(row, 1, QTableWidgetItem(args_json))
        self.steps_table.setItem(row, 2, QTableWidgetItem(str(retries)))
        self._refresh_canvas()

    def _remove_selected_step(self) -> None:
        row = self.steps_table.currentRow()
        if row >= 0:
            self.steps_table.removeRow(row)
        self._refresh_canvas()

    def _new_workflow(self) -> None:
        self.library_list.clearSelection()
        self.library_list.setCurrentItem(None)
        self.name_input.clear()
        self.description_input.clear()
        self.steps_table.setRowCount(0)
        self.status_label.setText("")
        self._refresh_canvas()

    # ── visual chain canvas ───────────────────────────────────────────────────

    def _steps_for_canvas(self) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for row in range(self.steps_table.rowCount()):
            tool_item = self.steps_table.item(row, 0)
            tool = tool_item.text().strip() if tool_item else ""
            args_item = self.steps_table.item(row, 1)
            args_text = args_item.text().strip() if args_item else "{}"
            try:
                args = json.loads(args_text) if args_text else {}
            except (json.JSONDecodeError, ValueError):
                args = {"...": args_text[:20]}
            steps.append({"tool": tool or "(no tool)", "args": args})
        return steps

    def _refresh_canvas(self) -> None:
        self.canvas.set_steps(self._steps_for_canvas())

    def _on_canvas_step_clicked(self, index: int) -> None:
        if 0 <= index < self.steps_table.rowCount():
            self.steps_table.selectRow(index)

    def _on_step_table_selection(self) -> None:
        self.canvas.highlight_step(self.steps_table.currentRow())

    def _collect_steps(self) -> list[dict[str, Any]] | None:
        steps: list[dict[str, Any]] = []
        for row in range(self.steps_table.rowCount()):
            tool_item = self.steps_table.item(row, 0)
            tool = tool_item.text().strip() if tool_item else ""
            if not tool:
                continue
            args_item = self.steps_table.item(row, 1)
            args_text = args_item.text().strip() if args_item else "{}"
            try:
                args = json.loads(args_text) if args_text else {}
            except (json.JSONDecodeError, ValueError):
                self.status_label.setText(f"Row {row + 1}: args must be valid JSON.")
                return None
            retries_item = self.steps_table.item(row, 2)
            try:
                retries = int(retries_item.text().strip()) if retries_item else 1
            except ValueError:
                retries = 1
            steps.append({"tool": tool, "args": args, "retries": retries})
        return steps

    def _save_workflow(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            self.status_label.setText("A workflow needs a name.")
            return
        steps = self._collect_steps()
        if steps is None:
            return
        result = self.engine.define(name, steps, self.description_input.text().strip())
        self.status_label.setText(result.text)
        if result.ok:
            self._refresh_library()

    def _delete_selected_workflow(self) -> None:
        name = self._selected_workflow_name()
        if not name:
            self.status_label.setText("Select a workflow to delete.")
            return
        result = self.engine.remove(name)
        self.status_label.setText(result.text)
        self._refresh_library()
        self._new_workflow()

    # ── runs ──────────────────────────────────────────────────────────────────

    def _run_selected_workflow(self) -> None:
        name = self._selected_workflow_name()
        if not name:
            self.status_label.setText("Select a workflow to run.")
            return
        result = self.engine.start(name, self.run_input.text().strip())
        self.status_label.setText(result.text)
        self._refresh_runs()

    def _cancel_run(self) -> None:
        item = self.runs_list.currentItem()
        run_id = str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""
        result = self.engine.cancel(run_id)
        self.status_label.setText(result.text)

    def _refresh_runs(self) -> None:
        if not self.isVisible():
            return
        self.runs_list.clear()
        for run in list(self.engine.runs.values())[-12:]:
            text = f"{run.get('workflow', '?')} — {run.get('status', '?')}  " \
                  f"(step {run.get('step', 0)}/{run.get('total', 0)})"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, run.get("id", ""))
            self.runs_list.addItem(item)

    def _on_event(self, channel: str, payload: Any) -> None:
        if channel == "workflow":
            self._refresh_runs()
