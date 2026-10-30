"""Stacked views hosted inside the core window."""

from __future__ import annotations

import asyncio
import webbrowser  # noqa: F401  (kept: see ..browser.open_url)
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote_plus, urlparse

import psutil
from PyQt6.QtCore import QTimer
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
from ..constants import BASE_DIR, SPACE
from ..memory import MemoryAgent
from ..security import SecuritySanitiser, SecurityViolation
from ..utils import now_stamp
from ..browser import open_url
from .hud import CentralHud
from .line_icons import apply as apply_icon
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
        # The record first, filling the page; the two things you can DO here
        # below it, one row each, with one primary action (Send). The two
        # shouted section titles and four capitalised buttons are gone; the
        # placeholders say what each row is for.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE.MD, SPACE.SM, SPACE.MD, SPACE.MD)
        layout.setSpacing(SPACE.SM + 2)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setObjectName("logBox")
        self.log_box.setAccessibleName("ORION's log")

        self.browser_line = QLineEdit()
        self.browser_line.setPlaceholderText("Search the web, or type an address to open")
        self.browser_line.setAccessibleName("Web search or address")
        self.browser_line.returnPressed.connect(self._browser_search)
        self.browser_search_btn = QPushButton("Search the Web")
        apply_icon(self.browser_search_btn, "search")
        self.browser_search_btn.clicked.connect(self._browser_search)
        self.browser_open_btn   = QPushButton("Open Address")
        self.browser_open_btn.clicked.connect(self._browser_open)

        browser_row = QHBoxLayout()
        browser_row.setSpacing(SPACE.SM)
        browser_row.addWidget(self.browser_line, 1)
        browser_row.addWidget(self.browser_search_btn)
        browser_row.addWidget(self.browser_open_btn)

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Message ORION…")
        self.input_line.setAccessibleName("Message to ORION")
        self.input_line.returnPressed.connect(self._send_manual)
        self.send_btn  = QPushButton("Send")
        self.send_btn.setObjectName("primaryButton")
        apply_icon(self.send_btn, "send", tone="inverse")
        self.send_btn.clicked.connect(self._send_manual)
        self.file_btn  = QPushButton("Scan a File…")
        apply_icon(self.file_btn, "document")
        self.file_btn.clicked.connect(self._scan_file)

        self.mic_toggle = QCheckBox("Microphone On")
        self.mic_toggle.setChecked(True)
        self.mic_toggle.stateChanged.connect(self._toggle_microphone)
        self.bus.mic_enabled.connect(self.mic_toggle.setChecked)

        input_row = QHBoxLayout()
        input_row.setSpacing(SPACE.SM)
        input_row.addWidget(self.input_line, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.file_btn)
        browser_row.addSpacing(SPACE.MD)
        browser_row.addWidget(self.mic_toggle)

        layout.addWidget(self.log_box, 1)
        layout.addLayout(input_row)
        layout.addLayout(browser_row)

        self.bus.log.connect(self.write_log)

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker
        worker.set_microphone_enabled(self.mic_toggle.isChecked())

    #: Within this many lines of the end counts as "following the log".
    FOLLOW_SLACK_LINES = 2

    def write_log(self, message: str) -> None:
        """Append a line. The view follows new lines only if the reader was
        already at the end: it used to jump to the end on every line, so
        anyone reading back was thrown to the bottom by the next event
        (audit 6 Oct 2026, A10). Their selection is left alone too."""
        line = f"{now_stamp()}  {message}"
        bar = self.log_box.verticalScrollBar()
        following = bar.value() >= bar.maximum() - self.FOLLOW_SLACK_LINES
        held = bar.value()
        blocks = self.log_box.document().blockCount()
        self.log_box.appendPlainText(line)
        if following:
            bar.setValue(bar.maximum())
            return
        # At the retention cap the oldest lines are pruned from the top;
        # move up with the text being read rather than drift past it.
        pruned = max(0, blocks + 1 - self.log_box.document().blockCount())
        bar.setValue(max(0, held - pruned))

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
        from ..file_review import supported_suffixes

        # 30 Oct 2026: the filter listed fifteen types and hid Word, Excel,
        # PowerPoint, audio, video and most code - all of which ORION reads.
        patterns = " ".join(f"*{suffix}" for suffix in supported_suffixes())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File for ORION Scan",
            str(BASE_DIR),
            f"Files ORION Can Read ({patterns});;All Files (*.*)",
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
        background.spawn(self._review_chosen_file(self.worker, file_path, focus))

    @staticmethod
    async def _review_chosen_file(worker: Any, file_path: str, focus: str) -> None:
        """The owner chose this file himself: the Live model reviews the
        extraction it is sent, so no second model reads it first."""
        from .. import file_review

        with file_review.handed_over(offer=False):
            await worker.submit_file_for_review(file_path, focus)

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
        layout.setContentsMargins(SPACE.MD, SPACE.SM, SPACE.MD, SPACE.MD)
        layout.setSpacing(SPACE.SM + 2)

        self.search_line = QLineEdit()
        self.search_line.setPlaceholderText("Search memories")
        self.search_line.setAccessibleName("Search saved records")
        self.search_line.returnPressed.connect(self.refresh)
        self.search_btn  = QPushButton("Search")
        self.search_btn.setObjectName("primaryButton")
        apply_icon(self.search_btn, "search", tone="inverse")
        self.search_btn.clicked.connect(self.refresh)
        self.clear_btn   = QPushButton("Clear")
        self.clear_btn.setAccessibleName("Clear the search")
        self.clear_btn.clicked.connect(self._clear_filter)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(SPACE.SM)
        filter_row.addWidget(self.search_line, 1)
        filter_row.addWidget(self.search_btn)
        filter_row.addWidget(self.clear_btn)

        from PyQt6.QtWidgets import QHeaderView

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Category", "Key", "Value", "Updated"])
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        # The value is what you came to read: it takes the width, the rest fit.
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        # Fit the columns to the first 40 rows, not all 500: measuring every
        # cell took 59-106 ms of the GUI thread at boot (28 Oct 2026).
        header.setResizeContentsPrecision(40)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        self.status_label = QLabel("")
        self.status_label.setObjectName("mutedLabel")

        # His memory by category, as Orion's stars (28 Oct 2026).
        from .hud_instruments import MemoryConstellation
        self.constellation = MemoryConstellation()
        self.constellation.setFixedHeight(240)
        layout.addWidget(self.constellation)
        layout.addLayout(filter_row)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status_label)

    def refresh(self) -> None:
        query   = self.search_line.text().strip()
        counts = getattr(self.memory, "category_counts", None)
        if callable(counts):
            try:
                self.constellation.set_categories(counts().items())
            except Exception:
                pass
        records = self.memory.records(query=query, limit=500)
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(records))
            for r, row in enumerate(records):
                for col, key in enumerate(("category", "key_ref", "value", "updated_at")):
                    self.table.setItem(r, col, QTableWidgetItem(str(row.get(key, ""))))
            self.table.resizeColumnsToContents()
        finally:
            self.table.setUpdatesEnabled(True)
        found = f"{len(records)} record{'' if len(records) == 1 else 's'}"
        self.status_label.setText(
            f"{found} matching “{query}”." if query else f"{found} — everything saved.")

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
        layout.setContentsMargins(SPACE.MD, SPACE.SM, SPACE.MD, SPACE.MD)
        layout.setSpacing(SPACE.SM + 2)

        # Real-time scrolling graphs — the detailed system-telemetry view. In
        # sentence case and named as a person would ("Processor", not "CPU
        # LOAD"), matching the Brain page and the Command Centre.
        self.cpu_graph = MetricGraph("Processor")
        self.ram_graph = MetricGraph("Memory")
        self.net_graph = MetricGraph("Network")

        # Quick action the user asked to keep on this page, at its top right.
        self.env_refresh_btn = QPushButton("Refresh Environment")
        self.env_refresh_btn.setObjectName("ghostButton")
        apply_icon(self.env_refresh_btn, "refresh")
        self.env_refresh_btn.setToolTip("Refresh the weather and location readings.")
        lead_row = QHBoxLayout()
        lead_row.addStretch(1)
        lead_row.addWidget(self.env_refresh_btn)

        # Process governor: its heading and its one action on one row.
        from PyQt6.QtWidgets import QHeaderView

        proc_heading = QLabel("Processes")
        proc_heading.setObjectName("panelHeading")
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(4)
        self.proc_table.setHorizontalHeaderLabels(["PID", "Name", "CPU %", "RAM %"])
        self.proc_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.proc_table.setMinimumHeight(240)
        self.proc_refresh_btn = QPushButton("Refresh")
        self.proc_refresh_btn.setObjectName("ghostButton")
        self.proc_refresh_btn.setAccessibleName("Refresh the process list")
        apply_icon(self.proc_refresh_btn, "refresh")
        self.proc_refresh_btn.clicked.connect(self._refresh_processes)
        proc_head = QHBoxLayout()
        proc_head.addWidget(proc_heading)
        proc_head.addStretch(1)
        proc_head.addWidget(self.proc_refresh_btn)

        layout.addLayout(lead_row)
        from .hud_instruments import ReactorGauge
        self.reactor = ReactorGauge(bus)
        self.reactor.setFixedHeight(230)
        gauges = QHBoxLayout()
        gauges.setSpacing(SPACE.MD)
        gauges.addWidget(self.reactor, 2)
        graphs = QVBoxLayout()
        for widget in (self.cpu_graph, self.ram_graph, self.net_graph):
            graphs.addWidget(widget)
        gauges.addLayout(graphs, 3)
        layout.addLayout(gauges)
        layout.addSpacing(SPACE.SM)
        layout.addLayout(proc_head)
        layout.addWidget(self.proc_table, 1)

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

    @staticmethod
    def _read_processes() -> list[tuple[str, str, str, str]]:
        rows = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                rows.append((str(info.get("pid", "")), str(info.get("name") or ""),
                             f"{info.get('cpu_percent') or 0:.1f}",
                             f"{info.get('memory_percent') or 0:.1f}"))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rows

    def _refresh_processes(self) -> None:
        """Read the process list off the GUI thread, then fill the table.

        28 Oct 2026: process_iter took 330-360 ms on the GUI thread every
        five seconds while this page was open; the table itself is ~12 ms."""
        if getattr(self, "_proc_busy", False):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:                      # no loop (tests, early start)
            self._fill_processes(self._read_processes())
            return
        self._proc_busy = True

        async def read() -> None:
            try:
                rows = await asyncio.to_thread(self._read_processes)
                self._fill_processes(rows)
            except RuntimeError:                  # the page was closed meanwhile
                pass
            finally:
                self._proc_busy = False
        background.spawn(read(), name="orion-process-list")

    def _fill_processes(self, rows: list[tuple[str, str, str, str]]) -> None:
        table = self.proc_table
        table.setUpdatesEnabled(False)
        try:
            table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, text in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(text))
            table.resizeColumnsToContents()
        finally:
            table.setUpdatesEnabled(True)
