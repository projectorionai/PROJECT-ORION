"""Stacked views hosted inside the core window."""

from __future__ import annotations

import asyncio
import webbrowser  # noqa: F401  (kept: see ..browser.open_url)
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import BASE_DIR
from ..memory import MemoryAgent
from ..security import SecuritySanitiser, SecurityViolation
from ..utils import now_stamp
from ..browser import open_url
from .hud import CentralHud
from .widgets import MetricBar, MetricGraph
from .. import background


class HudView(QWidget):
    """View 0 — Main Core HUD (the full CentralHud widget)."""

    def __init__(self, hud: CentralHud, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(hud)


class LogConsoleView(QWidget):
    """View 1 — Overt Log & Text Dispatch Console (the conversation surface)."""

    def __init__(self, bus: OrionBus, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus    = bus
        self.worker: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("INTELLIGENCE MONITOR")
        heading.setObjectName("panelHeading")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setObjectName("logBox")

        browser_heading = QLabel("RESEARCH BROWSER")
        browser_heading.setObjectName("panelHeading")

        self.browser_line = QLineEdit()
        self.browser_line.setPlaceholderText("Search query or URL")
        self.browser_line.returnPressed.connect(self._browser_search)
        self.browser_search_btn = QPushButton("SEARCH")
        self.browser_search_btn.clicked.connect(self._browser_search)
        self.browser_open_btn   = QPushButton("OPEN")
        self.browser_open_btn.clicked.connect(self._browser_open)

        browser_row = QHBoxLayout()
        browser_row.addWidget(self.browser_line, 1)
        browser_row.addWidget(self.browser_search_btn)
        browser_row.addWidget(self.browser_open_btn)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Manual command uplink  (Enter / Ctrl+Return)")
        self.input_line.returnPressed.connect(self._send_manual)
        self.send_btn  = QPushButton("SEND")
        self.send_btn.clicked.connect(self._send_manual)
        self.file_btn  = QPushButton("SCAN FILE")
        self.file_btn.clicked.connect(self._scan_file)

        self.mic_toggle = QCheckBox("MICROPHONE ACTIVE")
        self.mic_toggle.setChecked(True)
        self.mic_toggle.stateChanged.connect(self._toggle_microphone)
        self.bus.mic_enabled.connect(self.mic_toggle.setChecked)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_line, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.file_btn)

        layout.addWidget(heading)
        layout.addWidget(self.log_box, 1)
        layout.addWidget(browser_heading)
        layout.addLayout(browser_row)
        layout.addLayout(input_row)
        layout.addWidget(self.mic_toggle)

        self.bus.log.connect(self.write_log)

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker
        worker.set_microphone_enabled(self.mic_toggle.isChecked())

    def write_log(self, message: str) -> None:
        line = f"{now_stamp()}  {message}"
        self.log_box.appendPlainText(line)
        self.log_box.moveCursor(QTextCursor.MoveOperation.End)
        self.log_box.ensureCursorVisible()

    def _send_manual(self) -> None:
        text = self.input_line.text().strip()
        if not text:
            return
        self.input_line.clear()
        self.write_log(f"YOU: {text}")
        if self.worker is None:
            self.write_log("SYS: Live session not ready.")
            return
        background.spawn(self.worker.submit_text(text))

    def _scan_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select file for Orion scan",
            str(BASE_DIR),
            "All supported files (*.txt *.md *.py *.json *.csv *.tsv *.pdf "
            "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;All files (*.*)",
        )
        if not file_path:
            return
        focus = self.input_line.text().strip()
        if focus:
            self.input_line.clear()
        self.write_log(f"FILE: queued scan for {Path(file_path).name}")
        if self.worker is None:
            self.write_log("FILE: Live session not ready.")
            return
        background.spawn(self.worker.submit_file_for_review(file_path, focus))

    def _browser_search(self) -> None:
        query = self.browser_line.text().strip()
        if not query:
            return
        try:
            SecuritySanitiser.guard_text(query, "browser.query")
        except SecurityViolation as exc:
            self.write_log(f"SEC: {exc}")
            return
        url = query if self._looks_like_url(query) else f"https://www.bing.com/search?q={quote_plus(query)}"
        if "://" not in url:
            url = f"https://{url}"
        open_url(url)
        self.write_log(f"WEB: research opened for {query}")
        if self.worker is not None and not self._looks_like_url(query):
            background.spawn(
                self.worker.submit_text(
                    f"Research this with me and keep the answer concise: {query}"
                )
            )

    def _browser_open(self) -> None:
        target = self.browser_line.text().strip()
        if not target:
            return
        try:
            SecuritySanitiser.guard_text(target, "browser.url")
        except SecurityViolation as exc:
            self.write_log(f"SEC: {exc}")
            return
        url    = target if "://" in target else f"https://{target}"
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            self.write_log("WEB: blocked unsupported URL scheme.")
            return
        open_url(url)
        self.write_log(f"WEB: opened {url}")

    def _toggle_microphone(self) -> None:
        enabled = self.mic_toggle.isChecked()
        if self.worker is not None:
            self.worker.set_microphone_enabled(enabled)
        self.write_log(f"SYS: microphone {'active' if enabled else 'muted'}.")

    def _looks_like_url(self, text: str) -> bool:
        parsed = urlparse(text if "://" in text else f"https://{text}")
        return "." in parsed.netloc and " " not in parsed.netloc


class MemoryMatrixView(QWidget):
    """View 2 — Memory Matrix (SQLite FTS5 browser through the MemoryAgent)."""

    def __init__(self, memory: MemoryAgent, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.memory = memory
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QLabel("MEMORY MATRIX — SESSION + PERSISTENT INTELLIGENCE STORE")
        heading.setObjectName("panelHeading")

        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Full-text search (leave blank to list all)")
        self.search_line.returnPressed.connect(self.refresh)
        self.search_btn  = QPushButton("SEARCH")
        self.search_btn.clicked.connect(self.refresh)
        self.clear_btn   = QPushButton("CLEAR FILTER")
        self.clear_btn.clicked.connect(self._clear_filter)

        filter_row = QHBoxLayout()
        filter_row.addWidget(self.search_line, 1)
        filter_row.addWidget(self.search_btn)
        filter_row.addWidget(self.clear_btn)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Category", "Key", "Value", "Updated"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        self.status_label = QLabel("Awaiting query.")
        self.status_label.setObjectName("mutedLabel")

        layout.addWidget(heading)
        layout.addLayout(filter_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status_label)

    def refresh(self) -> None:
        query   = self.search_line.text().strip()
        records = self.memory.records(query=query, limit=500)
        self.table.setRowCount(0)
        for row in records:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for col, key in enumerate(("category", "key_ref", "value", "updated_at")):
                item = QTableWidgetItem(str(row.get(key, "")))
                self.table.setItem(r, col, item)
        self.table.resizeColumnsToContents()
        self.status_label.setText(
            f"{len(records)} record(s) retrieved."
            + (f"  Filter: '{query}'" if query else "  Displaying all records.")
        )

    def _clear_filter(self) -> None:
        self.search_line.clear()
        self.refresh()


class TelemetryView(QWidget):
    """View 3 — System Telemetry & Process Governor."""

    def __init__(self, bus: OrionBus, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.bus    = bus
        self.worker: Any = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        heading = QLabel("SYSTEM TELEMETRY & PROCESS GOVERNOR")
        heading.setObjectName("panelHeading")
        sub = QLabel(
            "Live host graphs and the process governor. Host bars, calendar and "
            "the geographical node keep their home in the left panel."
        )
        sub.setObjectName("mutedLabel")
        sub.setWordWrap(True)

        # Real-time scrolling graphs — the detailed system-telemetry view (the
        # compact bars, calendar and environment stay in the persistent left
        # panel; they are no longer duplicated here).
        self.cpu_graph = MetricGraph("CPU LOAD")
        self.ram_graph = MetricGraph("RAM CONSUMPTION")
        self.net_graph = MetricGraph("NETWORK ACTIVITY")

        # Quick-action buttons the user asked to keep on this page — environment
        # and calendar themselves live in the left panel.
        self.env_refresh_btn = QPushButton("⟳ REFRESH ENVIRONMENT")
        self.env_refresh_btn.setToolTip("Refresh the geographical node in the left panel.")
        action_row = QHBoxLayout()
        action_row.addWidget(self.env_refresh_btn)
        action_row.addStretch(1)

        # Process governor
        proc_heading = QLabel("PROCESS INVENTORY")
        proc_heading.setObjectName("panelHeading")
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(4)
        self.proc_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "RAM %"])
        self.proc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proc_table.setMinimumHeight(240)
        self.proc_refresh_btn = QPushButton("⟳ REFRESH PROCESSES")
        self.proc_refresh_btn.clicked.connect(self._refresh_processes)

        for widget in [heading, sub,
                       self.cpu_graph, self.ram_graph, self.net_graph]:
            layout.addWidget(widget)
        layout.addLayout(action_row)
        for widget in [proc_heading, self.proc_table, self.proc_refresh_btn]:
            layout.addWidget(widget)
        layout.addStretch(1)

        # Keep the governor live while the page is on screen; the guard makes it
        # a no-op behind the face view so it never costs CPU when hidden.
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(5000)
        self._proc_timer.timeout.connect(self._auto_refresh_processes)
        self._proc_timer.start()

        # Host telemetry (cpu/ram/net) is sampled by the Core Window's
        # background loop and broadcast on the bus — this view just renders it.
        self.bus.telemetry_sample.connect(self._on_telemetry_sample)

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker

    def _on_telemetry_sample(self, sample: Any) -> None:
        if not isinstance(sample, dict):
            return
        self.cpu_graph.set_value(float(sample.get("cpu", 0.0)))
        self.ram_graph.set_value(float(sample.get("ram", 0.0)))
        self.net_graph.set_value(float(sample.get("net_percent", 0.0)))

    def _auto_refresh_processes(self) -> None:
        if self.isVisible():
            self._refresh_processes()

    def attach_env_refresh(self, slot: Callable) -> None:
        self.env_refresh_btn.clicked.connect(slot)

    def _refresh_processes(self) -> None:
        self.proc_table.setRowCount(0)
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                r    = self.proc_table.rowCount()
                self.proc_table.insertRow(r)
                self.proc_table.setItem(r, 0, QTableWidgetItem(str(info.get("pid", ""))))
                self.proc_table.setItem(r, 1, QTableWidgetItem(str(info.get("name") or "")))
                self.proc_table.setItem(r, 2, QTableWidgetItem(f"{info.get('cpu_percent') or 0:.1f}"))
                self.proc_table.setItem(r, 3, QTableWidgetItem(f"{info.get('memory_percent') or 0:.1f}"))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.proc_table.resizeColumnsToContents()
