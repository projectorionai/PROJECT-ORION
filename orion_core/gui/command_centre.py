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

import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

import psutil
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QComboBox,
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

from ..constants import APP_NAME, C, HEALTH_COLOURS
from .token_graph import TokenUsageGraph
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
        grid.addWidget(self._build_tasks_panel(),     1, 1)
        grid.addWidget(self._build_tokens_panel(),    1, 2)
        grid.addWidget(self._build_logs_panel(),      2, 0, 1, 2)
        grid.addWidget(self._build_thoughts_panel(),  2, 2)
        grid.addWidget(self._build_capabilities_panel(), 3, 0, 1, 3)
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

        # ── manual device selection ───────────────────────────────────────────
        # ORION sometimes locks onto the wrong microphone/speaker; let the user
        # pick the right one directly.  Selecting a device applies live and
        # persists; "System default" follows the Windows default again.
        picker = QLabel("DEVICE CONTROL")
        picker.setObjectName("panelHeading")
        layout.addWidget(picker)

        self.input_combo = QComboBox()
        self.input_combo.setToolTip("Choose which microphone ORION listens on.")
        self.output_combo = QComboBox()
        self.output_combo.setToolTip("Choose which speaker/headphones ORION speaks through.")
        for label, combo in (("🎙 Mic", self.input_combo), ("🔊 Voice", self.output_combo)):
            row = QHBoxLayout()
            row.setSpacing(6)
            tag = QLabel(label)
            tag.setObjectName("mutedLabel")
            tag.setMinimumWidth(56)
            row.addWidget(tag)
            row.addWidget(combo, 1)
            layout.addLayout(row)

        refresh = QPushButton("⟳  Refresh devices")
        refresh.setToolTip("Re-scan available audio devices.")
        refresh.clicked.connect(self._populate_audio_devices)
        layout.addWidget(refresh)

        self._populate_audio_devices()
        self.input_combo.activated.connect(self._on_input_device_picked)
        self.output_combo.activated.connect(self._on_output_device_picked)

        layout.addStretch(1)
        return frame

    def _populate_audio_devices(self) -> None:
        """Fill the mic/speaker dropdowns, marking the device currently in use."""
        try:
            from ..audio_devices import _devices, resolve_effective
        except Exception:
            return
        try:
            devices = _devices()
        except Exception:
            devices = []
        specs = (
            (self.input_combo, "input", "max_input_channels"),
            (self.output_combo, "output", "max_output_channels"),
        )
        for combo, kind, channel_key in specs:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("System default", "default")
            try:
                current = resolve_effective(kind)
            except Exception:
                current = None
            selected = 0
            for index, dev in enumerate(devices):
                if dev.get(channel_key, 0) > 0:
                    combo.addItem(f"[{index}] {dev.get('name', '?')}", str(index))
                    if index == current:
                        selected = combo.count() - 1
            combo.setCurrentIndex(selected)
            combo.blockSignals(False)

    def _on_input_device_picked(self, _idx: int) -> None:
        spec = self.input_combo.currentData()
        if spec is not None:
            self.bus.audio_device_request.emit("input", spec)

    def _on_output_device_picked(self, _idx: int) -> None:
        spec = self.output_combo.currentData()
        if spec is not None:
            self.bus.audio_device_request.emit("output", spec)

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

    def _build_capabilities_panel(self) -> QFrame:
        """Every registered subsystem, its role, live health and what it
        depends on.

        The ModuleRegistry has carried this (role/dependencies/version, plus
        health cross-referenced from Telemetry) for a while with no GUI
        surface at all — it was only reachable as a text blob through the
        `capabilities` tool. This is the same data, legible at a glance, next
        to the component-heartbeat table it complements: that one answers
        "is this thing beating", this one answers "what is this thing, what
        does it need, and what version is it".
        """
        frame, layout = self._panel("SUBSYSTEM REGISTRY")
        self.capabilities_note = QLabel("")
        self.capabilities_note.setObjectName("mutedLabel")
        layout.addWidget(self.capabilities_note)
        self.capabilities_table = QTableWidget()
        self.capabilities_table.setColumnCount(5)
        self.capabilities_table.setHorizontalHeaderLabels(
            ["Module", "Role", "Health", "Depends on", "Version"])
        self.capabilities_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.capabilities_table.verticalHeader().setVisible(False)
        self.capabilities_table.setMaximumHeight(190)
        layout.addWidget(self.capabilities_table, 1)
        return frame

    def _refresh_capabilities(self) -> None:
        registries = getattr(self.dispatcher, "registries", None)
        if registries is None:
            self.capabilities_note.setText("Registries are not available.")
            return
        try:
            described = registries.modules.all_described(self.telemetry)
        except Exception:
            return
        unavailable = sum(1 for r in described.values() if not r.get("available"))
        self.capabilities_note.setText(
            f"{len(described)} subsystem(s) registered  •  "
            f"{len(described) - unavailable} available")
        from PyQt6.QtGui import QColor
        self.capabilities_table.setRowCount(0)
        for name, record in sorted(described.items()):
            i = self.capabilities_table.rowCount()
            self.capabilities_table.insertRow(i)
            self.capabilities_table.setItem(i, 0, QTableWidgetItem(name))
            self.capabilities_table.setItem(
                i, 1, QTableWidgetItem(str(record.get("role", ""))[:70]))
            status = str(record.get("health", "UNKNOWN"))
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor(HEALTH_COLOURS.get(status, C.MUTED)))
            self.capabilities_table.setItem(i, 2, status_item)
            self.capabilities_table.setItem(
                i, 3, QTableWidgetItem(", ".join(record.get("dependencies") or [])))
            self.capabilities_table.setItem(
                i, 4, QTableWidgetItem(str(record.get("version", ""))))
        self.capabilities_table.resizeColumnsToContents()

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

    def _build_tokens_panel(self) -> QFrame:
        frame, layout = self._panel("TOKEN USAGE · 24H · ALL PROVIDERS")
        ledger = getattr(getattr(self.worker, "router", None), "token_ledger", None)
        self.token_graph = TokenUsageGraph(ledger)
        layout.addWidget(self.token_graph, 1)
        return frame

    def _build_logs_panel(self) -> QFrame:
        frame, layout = self._panel("LIVE LOGS")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("logBox")
        self.log_box.setMaximumBlockCount(400)
        layout.addWidget(self.log_box, 1)
        return frame

    def _build_thoughts_panel(self) -> QFrame:
        frame, layout = self._panel("THOUGHT STREAM · INNER VOICE")
        self.thought_box = QPlainTextEdit()
        self.thought_box.setReadOnly(True)
        self.thought_box.setObjectName("logBox")
        self.thought_box.setMaximumBlockCount(200)
        self.thought_box.setPlaceholderText(
            "ORION's free thoughts appear here — reflections on what he is "
            "noticing and why he acts, typed out live on his dedicated "
            "thoughts model.")
        layout.addWidget(self.thought_box, 1)

        # Live "typing" state: thoughts are revealed character-by-character so
        # the stream reads like ORION thinking in real time rather than pasting
        # finished paragraphs.  Overlapping thoughts queue behind the current one.
        self._thought_queue: deque[str] = deque()
        self._typing_text = ""
        self._typing_pos = 0
        self._caret_on = False
        self._type_timer = QTimer(self)
        self._type_timer.setInterval(22)          # ~45 chars/sec, natural cadence
        self._type_timer.timeout.connect(self._type_tick)

        # Live streaming state: while a thought is being generated its tokens
        # arrive on bus.thought_delta and are written to the panel immediately,
        # so the user watches ORION think in real time.  _stream_active suppresses
        # the canned typewriter for the same thought when its final `thought`
        # event lands.
        self._stream_active = False
        self._stream_id: int | None = None
        self._streamed_ids: set[int] = set()

        try:
            self.bus.thought_delta.connect(self._on_thought_delta)
        except Exception:
            pass
        try:
            self.bus.thought.connect(self._on_thought)
        except Exception:
            pass
        return frame

    def _on_thought_delta(self, payload: Any) -> None:
        """Render a thought token-by-token as it streams off the model."""
        if not isinstance(payload, dict):
            return
        phase = payload.get("phase")
        tid = payload.get("id")
        if phase == "start":
            # Stop any canned typewriter and open a fresh line with a header.
            self._type_timer.stop()
            self._clear_caret()
            self._stream_active = True
            self._stream_id = tid
            marker = "⚙" if payload.get("kind") == "decision" else "◈"
            self.thought_box.appendPlainText(f"{payload.get('at', '')} {marker} ")
            self._caret_on = False
        elif phase == "delta":
            if not self._stream_active or tid != self._stream_id:
                return
            self._insert_at_end(str(payload.get("text", "")))
            self.thought_box.verticalScrollBar().setValue(
                self.thought_box.verticalScrollBar().maximum())
        elif phase == "end":
            if tid is not None:
                self._streamed_ids.add(tid)
            self._stream_active = False
            self._stream_id = None

    def _on_thought(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        # A thought already shown live via the streaming deltas is not re-typed.
        tid = payload.get("id")
        if tid in self._streamed_ids:
            self._streamed_ids.discard(tid)
            return
        marker = "⚙" if payload.get("kind") == "decision" else "◈"
        line = f"{payload.get('at', '')} {marker} {payload.get('text', '')}"
        self._thought_queue.append(line)
        if not self._type_timer.isActive() and not self._stream_active:
            self._begin_next_thought()

    def _begin_next_thought(self) -> None:
        if not self._thought_queue:
            return
        self._typing_text = self._thought_queue.popleft()
        self._typing_pos = 0
        # Start each thought on its own fresh line.
        self.thought_box.appendPlainText("")
        self._type_timer.start()

    def _type_tick(self) -> None:
        # Reveal a couple of characters per tick and trail a blinking caret so
        # the panel visibly "types".
        #
        # Nobody watching means nothing to animate. Measured at 1.53 ms per
        # tick (insert + repaint) and running at 45 Hz, the effect costs about
        # 70 ms of every second on the thread that also carries audio — and
        # ORION thinks continuously, so it is rarely idle. The text still
        # arrives in full; it simply stops being revealed one character at a
        # time into a widget on a page that is not on screen.
        if not self.thought_box.isVisible():
            self._finish_typing_now()
            return
        self._clear_caret()
        if self._typing_pos >= len(self._typing_text):
            self._type_timer.stop()
            # Brief beat before the next queued thought starts typing.
            if self._thought_queue:
                QTimer.singleShot(360, self._begin_next_thought)
            return
        chunk = self._typing_text[self._typing_pos:self._typing_pos + 2]
        self._typing_pos += len(chunk)
        self._insert_at_end(chunk)
        # Draw a caret while more remains.
        self._caret_on = self._typing_pos < len(self._typing_text)
        if self._caret_on:
            self._insert_at_end("▌")
        self.thought_box.verticalScrollBar().setValue(
            self.thought_box.verticalScrollBar().maximum())

    def _finish_typing_now(self) -> None:
        """Drop the remaining text in at once and stop the timer."""
        self._clear_caret()
        remaining = self._typing_text[self._typing_pos:]
        if remaining:
            self._insert_at_end(remaining)
        self._typing_pos = len(self._typing_text)
        self._caret_on = False
        self._type_timer.stop()
        if self._thought_queue:
            QTimer.singleShot(360, self._begin_next_thought)

    def _clear_caret(self) -> None:
        if not self._caret_on:
            return
        cursor = self.thought_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.deletePreviousChar()               # remove the trailing caret glyph
        self._caret_on = False

    def _insert_at_end(self, text: str) -> None:
        cursor = self.thought_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.thought_box.setTextCursor(cursor)

    # ── refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        # closeEvent hides this window rather than destroying it, so without
        # this guard the 1 Hz sweep below (psutil sampling, memory-tier counts,
        # health, task and log rebuilds, a telemetry history write) keeps
        # running forever after the user "closes" the window. Same one-line
        # convention as automation_deck/mission_deck's refreshes.
        if not self.isVisible():
            return
        self.clock.setText(datetime.now().strftime("%H:%M:%S"))
        self._refresh_system()
        self._refresh_audio()
        self._refresh_memory()
        self._refresh_health()
        self._refresh_capabilities()
        self._refresh_tasks()
        self._refresh_logs()
        self._refresh_autonomy_button()
        # History sampling is NOT done here any more: this refresh is skipped
        # while the page is hidden (above), which silently stopped recording.
        # app._sample_metrics_history drives it regardless of what is showing.

    def _refresh_system(self) -> None:
        # One reading for the whole application: the core window's telemetry
        # loop asks for the same figures, and net_io_counters costs 7.5 ms a
        # call on this machine.
        from ..system_metrics import sample

        reading = sample()
        bps = reading.bytes_per_second
        self.cpu_bar.set_value(reading.cpu)
        self.ram_bar.set_value(reading.ram)
        self.net_bar.set_value(reading.network)
        gpu_label, gpu_pct = self._gpu()
        self.gpu_bar.set_value(gpu_pct)
        self.sys_note.setText(
            f"Net {bps/1_048_576:.2f} MB/s • GPU {gpu_label} • "
            f"{psutil.cpu_count(logical=True)} cores"
        )

    def _gpu(self) -> tuple[str, float]:
        # This runs on the Qt thread, which under qasync IS the event loop that
        # paints the face. It used to run nvidia-smi here every five seconds:
        # MEASURED ~80 ms each (a 2 s timeout at worst), a visible hitch every
        # time. gpu_stats.latest() reads NVML in-process (0.02 ms) or returns
        # the last reading the background telemetry loop took — never a
        # process launched on this thread.
        try:
            from .. import gpu_stats
            reading = gpu_stats.latest()
            if not reading.get("available"):
                return ("n/a", 0.0)
            value = float(reading.get("util", 0.0))
            return (f"{value:.0f}%", value)
        except Exception:
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
        # HEALTH_COLOURS (constants.py, Mark XX design-spec §9) is the one
        # source for this triad now — this file used to define its own
        # #39ff88/#ffcc44/#ff4d4d, a third value for the same OK/DEGRADED/
        # DOWN concept C.GOOD/WARN/BAD and diagnostics_centre.py each also
        # defined independently.
        for r in rows:
            i = self.health_table.rowCount()
            self.health_table.insertRow(i)
            self.health_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            status_item = QTableWidgetItem(r["status"])
            from PyQt6.QtGui import QColor
            status_item.setForeground(QColor(HEALTH_COLOURS.get(r["status"], C.MUTED)))
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
