"""
ORION Command Centre (Phase 10) — the persistent operating dashboard.

A third window (alongside the Core Window and the Widget Dashboard) that gives
a live, at-a-glance picture of the whole system:

    • system metrics    — CPU, RAM, GPU (best effort), network
    • audio state        — speech state, active streams, queue depth, latency
    • memory state       — per-tier row counts
    • workspace          — window count, active window, active project
    • agent health       — component heartbeats (green/amber/red)
    • task queue         — active tool count + recent tool executions w/ timings
    • live logs          — the structured log ring buffer

Everything is *pulled* from the Telemetry facade (and a few cheap psutil reads)
on a 1 second QTimer, so nothing on a hot path ever touches this window.  The
Command Centre holds only read-only references and renders — it never mutates
runtime state.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from typing import Any, Optional

import psutil
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..constants import APP_NAME, C
from .widgets import MetricBar


class CommandCentreWindow(QMainWindow):
    def __init__(
        self,
        bus: Any,
        telemetry: Any,
        worker: Any,
        memory: Any,
        display: Any,
        workspace: Any,
        dispatcher: Any,
        control: Any = None,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.telemetry = telemetry
        self.worker = worker
        self.memory = memory
        self.display = display
        self.workspace = workspace
        self.dispatcher = dispatcher
        self.control = control

        self.setWindowTitle(f"{APP_NAME} — Command Centre")
        self.setMinimumSize(1040, 700)
        self.resize(1240, 800)

        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.monotonic()
        self._gpu_cache = ("n/a", 0.0)
        psutil.cpu_percent(interval=None)

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)
        outer.addWidget(self._build_header())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_system_panel(),   0, 0)
        grid.addWidget(self._build_audio_panel(),     0, 1)
        grid.addWidget(self._build_memory_panel(),    0, 2)
        grid.addWidget(self._build_health_panel(),    1, 0)
        grid.addWidget(self._build_tasks_panel(),     1, 1, 1, 2)
        grid.addWidget(self._build_logs_panel(),      2, 0, 1, 3)
        grid.setRowStretch(2, 1)
        outer.addLayout(grid, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)
        title = QLabel("COMMAND CENTRE")
        title.setObjectName("titleLabel")
        subtitle = QLabel("LIVE SYSTEM • AGENTS • TASKS • LOGS")
        subtitle.setObjectName("subtitleLabel")
        box = QVBoxLayout()
        box.setSpacing(0)
        box.addWidget(title)
        box.addWidget(subtitle)
        self.autonomy_btn = QPushButton("AUTONOMY: ?")
        self.autonomy_btn.setToolTip("Toggle the autonomous control layer on/off.")
        self.autonomy_btn.clicked.connect(self._toggle_autonomy)
        self.clock = QLabel("")
        self.clock.setObjectName("clockLabel")
        layout.addLayout(box, 1)
        layout.addWidget(self.autonomy_btn)
        layout.addWidget(self.clock)
        return frame

    @staticmethod
    def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("panelFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)
        return frame, layout

    def _build_system_panel(self) -> QFrame:
        frame, layout = self._panel("SYSTEM")
        self.cpu_bar = MetricBar("CPU")
        self.ram_bar = MetricBar("RAM")
        self.gpu_bar = MetricBar("GPU")
        self.net_bar = MetricBar("NETWORK")
        for b in (self.cpu_bar, self.ram_bar, self.gpu_bar, self.net_bar):
            layout.addWidget(b)
        self.sys_note = QLabel("")
        self.sys_note.setObjectName("mutedLabel")
        self.sys_note.setWordWrap(True)
        layout.addWidget(self.sys_note)
        layout.addStretch(1)
        return frame

    def _build_audio_panel(self) -> QFrame:
        frame, layout = self._panel("AUDIO STATE")
        self.audio_state = QLabel("—")
        self.audio_state.setObjectName("stateLabel")
        self.audio_detail = QLabel("")
        self.audio_detail.setObjectName("mutedLabel")
        self.audio_detail.setWordWrap(True)
        layout.addWidget(self.audio_state)
        layout.addWidget(self.audio_detail)
        layout.addStretch(1)
        return frame

    def _build_memory_panel(self) -> QFrame:
        frame, layout = self._panel("MEMORY STATE")
        self.mem_detail = QLabel("")
        self.mem_detail.setObjectName("mutedLabel")
        self.mem_detail.setWordWrap(True)
        self.ws_detail = QLabel("")
        self.ws_detail.setObjectName("mutedLabel")
        self.ws_detail.setWordWrap(True)
        self.display_detail = QLabel("")
        self.display_detail.setObjectName("mutedLabel")
        self.display_detail.setWordWrap(True)
        layout.addWidget(QLabel("Tiers:"))
        layout.addWidget(self.mem_detail)
        layout.addWidget(QLabel("Workspace:"))
        layout.addWidget(self.ws_detail)
        layout.addWidget(QLabel("Displays:"))
        layout.addWidget(self.display_detail)
        layout.addStretch(1)
        return frame

    def _build_health_panel(self) -> QFrame:
        frame, layout = self._panel("AGENT / COMPONENT HEALTH")
        self.health_table = QTableWidget()
        self.health_table.setColumnCount(3)
        self.health_table.setHorizontalHeaderLabels(["Component", "Status", "Detail"])
        self.health_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.health_table.verticalHeader().setVisible(False)
        layout.addWidget(self.health_table, 1)
        return frame

    def _build_tasks_panel(self) -> QFrame:
        frame, layout = self._panel("TASK QUEUE & TOOL EXECUTION")
        self.task_note = QLabel("")
        self.task_note.setObjectName("mutedLabel")
        layout.addWidget(self.task_note)
        self.tool_table = QTableWidget()
        self.tool_table.setColumnCount(4)
        self.tool_table.setHorizontalHeaderLabels(["Time", "Tool", "OK", "ms"])
        self.tool_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tool_table.verticalHeader().setVisible(False)
        layout.addWidget(self.tool_table, 1)
        return frame

    def _build_logs_panel(self) -> QFrame:
        frame, layout = self._panel("LIVE LOGS")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        self.log_box.setMaximumBlockCount(400)
        layout.addWidget(self.log_box, 1)
        return frame

    # ── refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self.clock.setText(datetime.now().strftime("%H:%M:%S"))
        self._refresh_system()
        self._refresh_audio()
        self._refresh_memory()
        self._refresh_health()
        self._refresh_tasks()
        self._refresh_logs()
        self._refresh_autonomy_button()

    def _refresh_system(self) -> None:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        now = time.monotonic()
        net = psutil.net_io_counters()
        elapsed = max(0.001, now - self._last_net_t)
        bps = ((net.bytes_sent + net.bytes_recv)
               - (self._last_net.bytes_sent + self._last_net.bytes_recv)) / elapsed
        self._last_net, self._last_net_t = net, now
        net_pct = min(100.0, (bps / 12_500_000.0) * 100.0)
        self.cpu_bar.set_value(cpu)
        self.ram_bar.set_value(ram)
        self.net_bar.set_value(net_pct)
        gpu_label, gpu_pct = self._gpu()
        self.gpu_bar.set_value(gpu_pct)
        self.sys_note.setText(
            f"Net {bps/1_048_576:.2f} MB/s • GPU {gpu_label} • "
            f"{psutil.cpu_count(logical=True)} cores"
        )

    def _gpu(self) -> tuple[str, float]:
        # nvidia-smi every ~5 s (cached) so the 1 s timer never spawns a process.
        if time.monotonic() - self._gpu_cache[1] < 5.0 and self._gpu_cache[0] != "n/a":
            try:
                return (self._gpu_cache[0], float(self._gpu_cache[0].rstrip("%")))
            except Exception:
                return (self._gpu_cache[0], 0.0)
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                val = float(proc.stdout.strip().splitlines()[0])
                self._gpu_cache = (f"{val:.0f}%", time.monotonic())
                return (f"{val:.0f}%", val)
        except Exception:
            pass
        self._gpu_cache = ("n/a", time.monotonic())
        return ("n/a", 0.0)

    def _refresh_audio(self) -> None:
        try:
            snap = self.worker.speech.telemetry_snapshot()
        except Exception:
            snap = {}
        state = str(snap.get("state", "—"))
        self.audio_state.setText(state)
        self.audio_detail.setText(
            f"Active streams: {snap.get('active_streams', 0)}  "
            f"(native={snap.get('native_active', False)}, tts={snap.get('tts_active', False)})\n"
            f"Playback queue: {snap.get('playback_queue_depth', 0)} chunk(s)\n"
            f"TTS queue: {snap.get('tts_queue_depth', 0)}\n"
            f"Local voice: {snap.get('local_voice', '?')}\n"
            f"{self._playback_latency()}"
        )

    def _playback_latency(self) -> str:
        try:
            timers = self.telemetry.metrics.snapshot().get("timers", {})
            pl = timers.get("audio.playback.write_latency_ms")
            if pl:
                return f"Playback write latency: p50 {pl['p50_ms']}ms, p95 {pl['p95_ms']}ms"
        except Exception:
            pass
        return "Playback write latency: —"

    def _refresh_memory(self) -> None:
        try:
            tiers = self.memory.tiers_snapshot()
            self.mem_detail.setText(
                "  ".join(f"{k}:{v}" for k, v in tiers.items())
                + (f"\nActive project: {self.memory.active_project or 'none'}")
            )
        except Exception:
            self.mem_detail.setText("—")
        try:
            snap = self.workspace.last_snapshot() if self.workspace else None
            if snap is not None:
                self.ws_detail.setText(f"{snap.summary()}\nActive: {snap.active_window[:60]}")
            else:
                self.ws_detail.setText("No snapshot yet.")
        except Exception:
            self.ws_detail.setText("—")
        try:
            self.display_detail.setText(self.display.summary())
        except Exception:
            self.display_detail.setText("—")

    def _refresh_health(self) -> None:
        rows = self.telemetry.health.snapshot()
        self.health_table.setRowCount(0)
        colours = {"OK": "#39ff88", "DEGRADED": "#ffcc44", "DOWN": "#ff4d4d", "UNKNOWN": C.MUTED}
        for r in rows:
            i = self.health_table.rowCount()
            self.health_table.insertRow(i)
            self.health_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            status_item = QTableWidgetItem(r["status"])
            from PyQt6.QtGui import QColor
            status_item.setForeground(QColor(colours.get(r["status"], C.MUTED)))
            self.health_table.setItem(i, 1, status_item)
            self.health_table.setItem(i, 2, QTableWidgetItem(str(r.get("detail", ""))[:70]))
        self.health_table.resizeColumnsToContents()

    def _refresh_tasks(self) -> None:
        active = getattr(self.dispatcher, "active_tools", 0)
        recent = list(getattr(self.dispatcher, "recent_tools", []))
        metrics = self.telemetry.metrics.snapshot().get("counters", {})
        self.task_note.setText(
            f"Active tool executions: {active}  •  "
            f"total calls: {int(metrics.get('tool.calls', 0))}  •  "
            f"failures: {int(metrics.get('tool.failures', 0))}"
        )
        self.tool_table.setRowCount(0)
        for entry in reversed(recent[-40:]):
            i = self.tool_table.rowCount()
            self.tool_table.insertRow(i)
            self.tool_table.setItem(i, 0, QTableWidgetItem(str(entry.get("at", ""))))
            self.tool_table.setItem(i, 1, QTableWidgetItem(str(entry.get("tool", ""))))
            self.tool_table.setItem(i, 2, QTableWidgetItem("✓" if entry.get("ok") else "✗"))
            self.tool_table.setItem(i, 3, QTableWidgetItem(str(entry.get("ms", ""))))
        self.tool_table.resizeColumnsToContents()

    def _refresh_logs(self) -> None:
        records = self.telemetry.log.recent(limit=60)
        text = "\n".join(
            f"{r['at'][11:19]} {r['level'][:4]:4} {r['component']}: {r['message']}"
            for r in records
        )
        # Only rewrite when the tail changed, to keep scrolling smooth.
        if text and text != getattr(self, "_last_log_text", ""):
            self._last_log_text = text
            self.log_box.setPlainText(text)
            self.log_box.verticalScrollBar().setValue(
                self.log_box.verticalScrollBar().maximum()
            )

    # ── autonomy toggle ───────────────────────────────────────────────────────

    def _refresh_autonomy_button(self) -> None:
        if self.control is None:
            self.autonomy_btn.setText("AUTONOMY: n/a")
            self.autonomy_btn.setEnabled(False)
            return
        self.autonomy_btn.setText(f"AUTONOMY: {'ON' if self.control.enabled else 'OFF'}")

    def _toggle_autonomy(self) -> None:
        if self.control is not None:
            self.control.set_enabled(not self.control.enabled)
            self._refresh_autonomy_button()

    def closeEvent(self, event: Any) -> None:
        event.ignore()
        self.hide()
