"""
DevelopmentDeckView — a real workspace over dev_workbench (dispatch_files.py),
not a decorative page.

Mark XX design-spec: the DEVELOPMENT zone existed as a disabled, empty tab
in the nav — there was no page to enable it with. The backend it wraps was
never the gap: dev_workbench already analyses a repository (language
breakdown, key files, TODO/FIXME counts), reads code with line numbers, and
runs allow-listed development commands (python/pytest/pip/git/node/npm/…)
— it was only ever reachable by the model issuing a tool call, with zero
GUI surface. This page calls the exact same dispatcher tool a voice/text
command would, through the same security allowlist, so nothing here is a
parallel or weaker implementation.

Three tabs:
  Workbench — repo analysis / command runner / code viewer (original page).
  Debugger  — a real pdb-backed debug session (debugger.py), console-style:
              start a script, step through it, print variables, set
              breakpoints. Scoped deliberately to pdb's own command surface
              rather than a full gutter-breakpoint IDE debugger — that
              would be a much larger, riskier undertaking; this is real
              and immediately useful, not a placeholder.
  Docker    — container/image control over the real `docker` CLI
              (docker_control.py): list, start/stop/restart/remove, tail
              logs. Degrades to an explanatory status line when Docker
              isn't installed or the daemon isn't running.

Every tab calls through self.dispatcher.dispatch(...) — the identical path
a voice/text command takes — so nothing here is a second, weaker
implementation of the same capability.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from .. import background
from .widgets import describe_control

_DEV_COMMAND_HINT = (
    "Allow-listed: python, pytest, pip, git, node, npm, npx, tsc, cargo, go, dotnet, rustc"
)


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


def _mono_output() -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setObjectName("logBox")
    box.setFont(QFont("Cascadia Mono", 9))
    return box


class DevelopmentDeckView(QWidget):
    """The DEVELOPMENT zone page: Workbench / Debugger / Docker tabs."""

    def __init__(self, bus: OrionBus, dispatcher: Any | None = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.dispatcher = dispatcher
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        title = QLabel("DEVELOPMENT — WORKBENCH")
        title.setObjectName("titleLabel")
        outer.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_workbench_tab(), "Workbench")
        tabs.addTab(self._build_debugger_tab(), "Debugger")
        tabs.addTab(self._build_docker_tab(), "Docker")
        outer.addWidget(tabs, 1)

    # ── workbench tab ─────────────────────────────────────────────────────────

    def _build_workbench_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_repo_panel(), 0, 0)
        grid.addWidget(self._build_command_panel(), 0, 1)
        grid.addWidget(self._build_code_panel(), 1, 0, 1, 2)
        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 3)
        layout.addLayout(grid, 1)
        return page

    def _build_repo_panel(self) -> QFrame:
        frame, layout = _panel("REPOSITORY ANALYSIS")
        row = QHBoxLayout()
        self.repo_path = QLineEdit()
        self.repo_path.setPlaceholderText("Repository path (blank = ORION's own root)")
        self.repo_path.returnPressed.connect(self._analyse_repo)
        row.addWidget(self.repo_path, 1)
        analyse_btn = QPushButton("Analyse")
        analyse_btn.clicked.connect(self._analyse_repo)
        row.addWidget(analyse_btn)
        layout.addLayout(row)
        self.repo_output = _mono_output()
        layout.addWidget(self.repo_output, 1)
        return frame

    def _build_command_panel(self) -> QFrame:
        frame, layout = _panel("COMMAND RUNNER")
        hint = QLabel(_DEV_COMMAND_HINT)
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        row = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("e.g. pytest -q")
        self.command_input.returnPressed.connect(self._run_command)
        row.addWidget(self.command_input, 1)
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._run_command)
        row.addWidget(run_btn)
        layout.addLayout(row)
        self.command_output = _mono_output()
        layout.addWidget(self.command_output, 1)
        return frame

    def _build_code_panel(self) -> QFrame:
        frame, layout = _panel("CODE VIEWER")
        row = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("File path")
        self.file_path.returnPressed.connect(self._read_file)
        row.addWidget(self.file_path, 2)
        row.addWidget(QLabel("from"))
        self.start_line = QLineEdit("1")
        self.start_line.setMaximumWidth(60)
        row.addWidget(self.start_line)
        row.addWidget(QLabel("lines"))
        self.line_count = QLineEdit("200")
        self.line_count.setMaximumWidth(60)
        row.addWidget(self.line_count)
        read_btn = QPushButton("Read")
        read_btn.clicked.connect(self._read_file)
        row.addWidget(read_btn)
        layout.addLayout(row)
        self.code_output = _mono_output()
        layout.addWidget(self.code_output, 1)
        return frame

    # ── debugger tab ──────────────────────────────────────────────────────────

    def _build_debugger_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 8, 0, 0)
        frame, layout = _panel("PDB SESSION — real Python debugger, one session at a time")

        start_row = QHBoxLayout()
        self.debug_script = QLineEdit()
        self.debug_script.setPlaceholderText("Script path to debug")
        start_row.addWidget(self.debug_script, 1)
        self.debug_args = QLineEdit()
        self.debug_args.setPlaceholderText("Args (optional, space-separated)")
        self.debug_args.setMaximumWidth(200)
        start_row.addWidget(self.debug_args)
        start_btn = describe_control(
            QPushButton("Start"), "Start debugger",
            "Run the named script under the debugger")
        start_btn.clicked.connect(self._debug_start)
        start_row.addWidget(start_btn)
        stop_btn = describe_control(
            QPushButton("Stop"), "Stop debugger", "End the debugging session")
        stop_btn.clicked.connect(self._debug_stop)
        start_row.addWidget(stop_btn)
        layout.addLayout(start_row)

        step_row = QHBoxLayout()
        for label, command in (("Next", "next"), ("Step", "step"), ("Continue", "continue")):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, c=command: self._debug_send(c))
            step_row.addWidget(btn)
        layout.addLayout(step_row)

        command_row = QHBoxLayout()
        self.debug_command = QLineEdit()
        self.debug_command.setPlaceholderText("pdb command, e.g. p some_variable, break file.py:12")
        self.debug_command.returnPressed.connect(self._debug_send_input)
        command_row.addWidget(self.debug_command, 1)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._debug_send_input)
        command_row.addWidget(send_btn)
        layout.addLayout(command_row)

        self.debug_output = _mono_output()
        layout.addWidget(self.debug_output, 1)
        outer.addWidget(frame, 1)
        return page

    def _debug_start(self) -> None:
        self._run(self._debug_start_async)

    async def _debug_start_async(self) -> None:
        path = self.debug_script.text().strip()
        if not path:
            return
        args = self.debug_args.text().split()
        self._append_debug(f"$ python -m pdb {path} {' '.join(args)}")
        result = await self.dispatcher.dispatch(
            "debugger", {"action": "start", "path": path, "args": args})
        self._append_debug(result.text)

    def _debug_stop(self) -> None:
        self._run(self._debug_stop_async)

    async def _debug_stop_async(self) -> None:
        result = await self.dispatcher.dispatch("debugger", {"action": "stop"})
        self._append_debug(result.text)

    def _debug_send(self, command: str) -> None:
        self._run(lambda: self._debug_send_async(command))

    def _debug_send_input(self) -> None:
        command = self.debug_command.text().strip()
        if not command:
            return
        self.debug_command.clear()
        self._debug_send(command)

    async def _debug_send_async(self, command: str) -> None:
        self._append_debug(f"(Pdb) {command}")
        result = await self.dispatcher.dispatch(
            "debugger", {"action": "command", "command": command})
        self._append_debug(result.text)

    def _append_debug(self, text: str) -> None:
        self.debug_output.appendPlainText(text)

    # ── docker tab ────────────────────────────────────────────────────────────

    def _build_docker_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 8, 0, 0)

        self.docker_status = QLabel("")
        self.docker_status.setObjectName("mutedLabel")
        self.docker_status.setWordWrap(True)
        outer.addWidget(self.docker_status)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_docker_containers_panel(), 0, 0)
        grid.addWidget(self._build_docker_logs_panel(), 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        outer.addLayout(grid, 1)
        return page

    def _build_docker_containers_panel(self) -> QFrame:
        frame, layout = _panel("CONTAINERS")
        self.docker_list = QListWidget()
        layout.addWidget(self.docker_list, 1)

        row1 = QHBoxLayout()
        refresh_btn = describe_control(
            QPushButton("Refresh"), "Refresh containers",
            "Reload the Docker container and image lists")
        refresh_btn.clicked.connect(self._docker_refresh)
        row1.addWidget(refresh_btn)
        start_btn = describe_control(
            QPushButton("Start"), "Start container",
            "Start the selected Docker container")
        start_btn.clicked.connect(lambda: self._docker_lifecycle("start"))
        row1.addWidget(start_btn)
        stop_btn = describe_control(
            QPushButton("Stop"), "Stop container",
            "Stop the selected Docker container")
        stop_btn.clicked.connect(lambda: self._docker_lifecycle("stop"))
        row1.addWidget(stop_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        restart_btn = QPushButton("Restart")
        restart_btn.clicked.connect(lambda: self._docker_lifecycle("restart"))
        row2.addWidget(restart_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._docker_lifecycle("remove"))
        row2.addWidget(remove_btn)
        logs_btn = QPushButton("Logs")
        logs_btn.clicked.connect(self._docker_show_logs)
        row2.addWidget(logs_btn)
        layout.addLayout(row2)
        return frame

    def _build_docker_logs_panel(self) -> QFrame:
        frame, layout = _panel("LOGS / OUTPUT")
        self.docker_output = _mono_output()
        layout.addWidget(self.docker_output, 1)
        return frame

    def _selected_container(self) -> str:
        item = self.docker_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else ""

    def _docker_refresh(self) -> None:
        self._run(self._docker_refresh_async)

    async def _docker_refresh_async(self) -> None:
        status = await self.dispatcher.dispatch("docker", {"action": "status"})
        self.docker_status.setText(status.text)
        result = await self.dispatcher.dispatch("docker", {"action": "list"})
        self.docker_list.clear()
        for line in result.text.splitlines():
            line = line.strip()
            if not line or line.lower().startswith(("containers:", "docker")):
                continue
            item = QListWidgetItem(line)
            name = line.split()[0] if line.split() else ""
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.docker_list.addItem(item)

    def _docker_lifecycle(self, action: str) -> None:
        self._run(lambda: self._docker_lifecycle_async(action))

    async def _docker_lifecycle_async(self, action: str) -> None:
        name = self._selected_container()
        if not name:
            self.docker_output.setPlainText("Select a container first.")
            return
        result = await self.dispatcher.dispatch(
            "docker", {"action": action, "container": name})
        self.docker_output.setPlainText(result.text)
        await self._docker_refresh_async()

    def _docker_show_logs(self) -> None:
        self._run(self._docker_show_logs_async)

    async def _docker_show_logs_async(self) -> None:
        name = self._selected_container()
        if not name:
            self.docker_output.setPlainText("Select a container first.")
            return
        result = await self.dispatcher.dispatch(
            "docker", {"action": "logs", "container": name, "tail": 200})
        self.docker_output.setPlainText(result.text)

    # ── shared async dispatch helper ─────────────────────────────────────────

    def _run(self, factory) -> None:
        # Takes the zero-arg coroutine FUNCTION, not an already-constructed
        # coroutine — constructing one just to discard it unused (when
        # there's no dispatcher, or no running loop) is what produced a
        # "coroutine was never awaited" warning; this way nothing gets
        # built until both guards have passed.
        if self.dispatcher is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        background.spawn(factory())

    def _analyse_repo(self) -> None:
        self._run(self._analyse_repo_async)

    async def _analyse_repo_async(self) -> None:
        self.repo_output.setPlainText("Analysing…")
        path = self.repo_path.text().strip()
        result = await self.dispatcher.dispatch(
            "dev_workbench", {"action": "analyse_repo", "path": path or "."})
        self.repo_output.setPlainText(result.text)

    def _run_command(self) -> None:
        self._run(self._run_command_async)

    async def _run_command_async(self) -> None:
        command = self.command_input.text().strip()
        if not command:
            return
        self.command_output.setPlainText(f"$ {command}\nRunning…")
        result = await self.dispatcher.dispatch(
            "dev_workbench",
            {"action": "run_command", "command": command,
             "path": self.repo_path.text().strip() or "."})
        self.command_output.setPlainText(result.text)

    def _read_file(self) -> None:
        self._run(self._read_file_async)

    async def _read_file_async(self) -> None:
        path = self.file_path.text().strip()
        if not path:
            return
        try:
            start = max(1, int(self.start_line.text().strip() or "1"))
        except ValueError:
            start = 1
        try:
            count = max(1, int(self.line_count.text().strip() or "200"))
        except ValueError:
            count = 200
        result = await self.dispatcher.dispatch(
            "dev_workbench",
            {"action": "read_file", "path": path, "start_line": start, "line_count": count})
        self.code_output.setPlainText(result.text)
