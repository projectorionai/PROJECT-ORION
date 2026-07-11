"""
ORION Core Window — conversation, voice activity, system health and status.

One of the two Mark VIII windows (the other is the Widget Dashboard).  Also
implements the immersive display modes:

    FULLSCREEN (F11)         — borderless full-monitor operating-system feel.
    OVERLAY    (Ctrl+Shift+O) — compact frameless always-on-top orb that
                                floats over the desktop while ORION keeps
                                listening; drag to reposition, Esc to exit.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Optional

import psutil
from aiohttp import ClientSession
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QGuiApplication, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCalendarWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import APP_NAME, APP_SUBTITLE
from ..memory import MemoryAgent
from ..utils import aiohttp_client_timeout, utc_stamp, weather_code_label
from .face import HologramFace
from .hud import CentralHud
from .style import APP_STYLESHEET
from .views import HudView, LogConsoleView, MemoryMatrixView, TelemetryView
from .widgets import MetricBar, MiniOrb


class OrionCoreWindow(QMainWindow):
    def __init__(self, bus: OrionBus, memory: MemoryAgent) -> None:
        super().__init__()
        self.bus          = bus
        self.memory       = memory
        self.worker: Any  = None
        self.dashboard: QMainWindow | None = None
        self.command_centre: QMainWindow | None = None
        self.active_state = "INITIALISING"
        self.telemetry: dict[str, Any] = {
            "cpu":          0.0,
            "ram":          0.0,
            "net_bps":      0.0,
            "net_percent":  0.0,
            "state":        self.active_state,
            "mic_active":   True,
            "updated_at":   utc_stamp(),
        }
        # Overlay-mode bookkeeping
        self._overlay_active = False
        self._saved_geometry: Any = None
        self._saved_min_size: Any = None
        self._drag_offset: Any = None
        # Speaking face: auto-morph the orb into the cyberpunk face while ORION
        # speaks, then back — only when we did the switch ourselves.
        self._auto_face  = True
        self._auto_faced = False

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1180, 720)
        self.resize(1360, 820)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self._connect_bus()

    # ── attachments ───────────────────────────────────────────────────────────

    def attach_worker(self, worker: Any) -> None:
        self.worker = worker
        self.log_view.attach_worker(worker)
        self.telemetry_view.attach_worker(worker)
        worker.set_microphone_enabled(self.log_view.mic_toggle.isChecked())

    def attach_dashboard(self, dashboard: QMainWindow) -> None:
        self.dashboard = dashboard

    def attach_command_centre(self, centre: QMainWindow) -> None:
        self.command_centre = centre

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root   = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.header_frame = self._build_header()
        layout.addWidget(self.header_frame)

        # ── horizontal splitter: left telemetry | stacked views ──────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.left_panel = self._build_left_panel()
        self.splitter.addWidget(self.left_panel)

        # Stacked widget — houses all four views (memory injected properly;
        # the Mark VII dummy-memory workaround is gone).
        self.hud            = CentralHud()
        self.hud_view       = HudView(self.hud)
        self.log_view       = LogConsoleView(self.bus)
        self.memory_view    = MemoryMatrixView(self.memory)
        self.telemetry_view = TelemetryView(self.bus)

        # ORION's avatar — the true 3-D quantum voxel face (Three.js/WebGL,
        # Phase 8).  Falls back to the 2-D HologramFace only where PyQt6-
        # WebEngine is unavailable, so the interface and wiring never change.
        try:
            from .face3d import QuantumFace3D
            self.face = QuantumFace3D() if QuantumFace3D.available else HologramFace()
        except Exception:
            self.face = HologramFace()

        # Stack order MUST match the segmented switcher's view order.
        self.stack = QStackedWidget()
        self.stack.addWidget(self.hud_view)          # 0  HUD (orb)
        self.stack.addWidget(self.face)              # 1  FACE (cyberpunk skull)
        self.stack.addWidget(self.log_view)          # 2  LOG
        self.stack.addWidget(self.memory_view)       # 3  MEMORY
        self.stack.addWidget(self.telemetry_view)    # 4  TELEMETRY

        self.splitter.addWidget(self.stack)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([270, 1090])
        layout.addWidget(self.splitter, 1)

        # ── mini orb (bottom-right overlay for non-HUD views) ────────────────
        self.mini_orb = MiniOrb(root)
        self.mini_orb.hide()

        # ── overlay controls — the ONLY way back is otherwise hidden ──────────
        # In overlay mode the header disappears, so these floating glass chips
        # (restore / shut down) are shown over the orb; double-clicking the orb
        # also restores. Without this there was no way to un-shrink ORION.
        self.overlay_bar = QWidget(root)
        overlay_row = QHBoxLayout(self.overlay_bar)
        overlay_row.setContentsMargins(0, 0, 0, 0)
        overlay_row.setSpacing(6)
        self.overlay_restore_btn = QPushButton("⤢")
        self.overlay_restore_btn.setObjectName("overlayChip")
        self.overlay_restore_btn.setToolTip("Restore ORION (or double-click the orb · Ctrl+Shift+O)")
        self.overlay_restore_btn.clicked.connect(self.exit_overlay_mode)
        self.overlay_quit_btn = QPushButton("⏻")
        self.overlay_quit_btn.setObjectName("overlayChip")
        self.overlay_quit_btn.setToolTip("Shut ORION down completely")
        self.overlay_quit_btn.clicked.connect(self._quit)
        overlay_row.addWidget(self.overlay_restore_btn)
        overlay_row.addWidget(self.overlay_quit_btn)
        self.overlay_bar.hide()

        self.setCentralWidget(root)

        # ── keyboard shortcuts ────────────────────────────────────────────────
        send_action = QAction(self)
        send_action.setShortcut(QKeySequence("Ctrl+Return"))
        send_action.triggered.connect(self.log_view._send_manual)
        self.addAction(send_action)

        fullscreen_action = QAction(self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        self.addAction(fullscreen_action)

        overlay_action = QAction(self)
        overlay_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        overlay_action.triggered.connect(self.toggle_overlay_mode)
        self.addAction(overlay_action)

        dashboard_action = QAction(self)
        dashboard_action.setShortcut(QKeySequence("Ctrl+D"))
        dashboard_action.triggered.connect(self.toggle_dashboard)
        self.addAction(dashboard_action)

        centre_action = QAction(self)
        centre_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        centre_action.triggered.connect(self.toggle_command_centre)
        self.addAction(centre_action)

        pause_action = QAction(self)
        pause_action.setShortcut(QKeySequence("Ctrl+Space"))
        pause_action.triggered.connect(self._toggle_pause)
        self.addAction(pause_action)

        quit_action = QAction(self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self._quit)
        self.addAction(quit_action)

    def _build_header(self) -> QFrame:
        frame  = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)

        self.title_label    = QLabel("O.R.I.O.N. MARK X.5")
        self.title_label.setObjectName("titleLabel")
        self.subtitle_label = QLabel(APP_SUBTITLE)
        self.subtitle_label.setObjectName("subtitleLabel")
        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        text_box.addWidget(self.title_label)
        text_box.addWidget(self.subtitle_label)

        # ── View switcher — a clean segmented control (replaces the dropdown) ──
        # One scannable control for the four core views, styled identically to
        # the Command Deck's tabs so navigation reads the same everywhere.
        view_defs = [("⬡", "HUD"), ("◉", "FACE"), ("⌨", "LOG"),
                     ("⊞", "MEMORY"), ("◈", "TELEMETRY")]
        self.view_switcher = QFrame()
        self.view_switcher.setObjectName("segmented")
        switch_row = QHBoxLayout(self.view_switcher)
        switch_row.setContentsMargins(3, 3, 3, 3)
        switch_row.setSpacing(2)
        self._view_buttons: list[QPushButton] = []
        for index, (glyph, name) in enumerate(view_defs):
            btn = QPushButton(f"{glyph}  {name}")
            btn.setObjectName("segItem")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, i=index: self._select_view(i))
            self._view_buttons.append(btn)
            switch_row.addWidget(btn)
        self._view_buttons[0].setChecked(True)

        # ── Window controls — one compact icon cluster (was five loose buttons)
        self.pause_btn = QPushButton("⏸ PAUSE")
        self.pause_btn.setToolTip("Pause ORION's speech and listening (say 'pause'); "
                                  "say 'resume' or 'Orion' to zone back in.")
        self.pause_btn.clicked.connect(self._toggle_pause)

        self.dashboard_btn = QPushButton("⧉")
        self.dashboard_btn.setObjectName("iconButton")
        self.dashboard_btn.setToolTip("Widget Dashboard (Ctrl+D)")
        self.dashboard_btn.clicked.connect(self.toggle_dashboard)
        self.centre_btn = QPushButton("◉")
        self.centre_btn.setObjectName("iconButton")
        self.centre_btn.setToolTip("Command Centre (Ctrl+Shift+C)")
        self.centre_btn.clicked.connect(self.toggle_command_centre)
        self.overlay_btn = QPushButton("◱")
        self.overlay_btn.setObjectName("iconButton")
        self.overlay_btn.setToolTip("Compact overlay orb (Ctrl+Shift+O; Esc to exit)")
        self.overlay_btn.clicked.connect(self.toggle_overlay_mode)
        self.fullscreen_btn = QPushButton("⛶")
        self.fullscreen_btn.setObjectName("iconButton")
        self.fullscreen_btn.setToolTip("Fullscreen (F11)")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)

        self.control_cluster = QFrame()
        self.control_cluster.setObjectName("controlCluster")
        cluster_row = QHBoxLayout(self.control_cluster)
        cluster_row.setContentsMargins(3, 3, 3, 3)
        cluster_row.setSpacing(2)
        for button in (self.dashboard_btn, self.centre_btn,
                       self.overlay_btn, self.fullscreen_btn):
            cluster_row.addWidget(button)

        # Explicit shutdown — closing this window or Ctrl+Q stops ORION fully.
        self.quit_btn = QPushButton("⏻")
        self.quit_btn.setObjectName("quitButton")
        self.quit_btn.setToolTip("Shut ORION down completely (Ctrl+Q)")
        self.quit_btn.clicked.connect(self._quit)

        # ── Status chips — grouped on the right ───────────────────────────────
        self.voice_led = QLabel("VOICE ○")
        self.voice_led.setObjectName("voiceLed")
        self.voice_led.setProperty("speaking", "false")
        self.state_label = QLabel("INITIALISING")
        self.state_label.setObjectName("stateLabel")
        self.clock_label  = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.clock_label.setObjectName("clockLabel")

        # Clear left→right zones: brand · views ——— pause · windows · status · quit
        layout.addLayout(text_box, 0)
        layout.addSpacing(8)
        layout.addWidget(self.view_switcher)
        layout.addStretch(1)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.control_cluster)
        layout.addSpacing(6)
        layout.addWidget(self.voice_led)
        layout.addWidget(self.state_label)
        layout.addWidget(self.clock_label)
        layout.addWidget(self.quit_btn)

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(
            lambda: self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))
        )
        self.clock_timer.start()
        return frame

    def _select_view(self, index: int) -> None:
        """Switch core view from the segmented control (and keep it in sync)."""
        index = max(0, min(len(self._view_buttons) - 1, index))
        for i, button in enumerate(self._view_buttons):
            button.setChecked(i == index)
        self._on_nav_changed(index)

    def _build_left_panel(self) -> QWidget:
        """
        Persistent left panel — host telemetry bars, calendar, geographical
        node.  These remain visible across all stacked views.
        """
        frame  = QFrame()
        frame.setObjectName("panelFrame")
        frame.setMinimumWidth(250)
        frame.setMaximumWidth(320)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        heading = QLabel("HOST TELEMETRY")
        heading.setObjectName("panelHeading")

        # These bars are updated by start_telemetry() regardless of active view
        self.cpu_bar  = MetricBar("CPU LOAD")
        self.ram_bar  = MetricBar("RAM CONSUMPTION")
        self.net_bar  = MetricBar("NETWORK ACTIVITY")

        self.telemetry_note = QLabel("Local psutil monitor active.")
        self.telemetry_note.setObjectName("mutedLabel")
        self.telemetry_note.setWordWrap(True)

        cal_heading = QLabel("CALENDAR")
        cal_heading.setObjectName("panelHeading")
        self.calendar_widget = QCalendarWidget()
        self.calendar_widget.setGridVisible(False)
        self.calendar_widget.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        self.calendar_widget.setMaximumHeight(215)

        env_heading = QLabel("GEOGRAPHICAL NODE")
        env_heading.setObjectName("panelHeading")
        self.location_label = QLabel("Location: awaiting refresh.")
        self.location_label.setObjectName("mutedLabel")
        self.location_label.setWordWrap(True)
        self.weather_label  = QLabel("Weather: awaiting refresh.")
        self.weather_label.setObjectName("mutedLabel")
        self.weather_label.setWordWrap(True)
        self.weather_button = QPushButton("REFRESH ENVIRONMENT")
        self.weather_button.clicked.connect(
            lambda: asyncio.create_task(self.refresh_environment_widgets())
        )

        for widget in [
            heading,
            self.cpu_bar, self.ram_bar, self.net_bar,
            cal_heading, self.calendar_widget,
            env_heading,
            self.location_label, self.weather_label, self.weather_button,
        ]:
            layout.addWidget(widget)
        layout.addStretch(1)
        layout.addWidget(self.telemetry_note)
        return frame

    # ── navigation ────────────────────────────────────────────────────────────

    def _on_nav_changed(self, index: int) -> None:
        """
        Switch the visible stacked view.

        When navigating away from the main HUD (index > 0):
          • Throttle the full CentralHud animation timer (reduce to 4 fps).
          • Show the 64×64 MiniOrb in the bottom-right corner so the orb
            continues to listen and pulse visually.
        """
        self.stack.setCurrentIndex(index)
        HUD, FACE, MEMORY = 0, 1, 3

        # Full frame-rate only for the avatar that is actually on screen.
        self.hud.timer.setInterval(33 if index == HUD else 250)
        self.face.timer.setInterval(33 if index == FACE else 250)

        # The mini orb persists only over the static views (log/memory/telemetry),
        # not over the two full-size avatars (orb HUD / face).
        if index in (HUD, FACE):
            self.mini_orb.hide()
        else:
            self._reposition_mini_orb()
            self.mini_orb.show()
            self.mini_orb.raise_()

        # Refresh the memory view whenever it becomes visible
        if index == MEMORY:
            self.memory_view.refresh()

    def _reposition_mini_orb(self) -> None:
        """Place the mini orb in the bottom-right corner of the central widget."""
        parent = self.centralWidget()
        if parent is None:
            return
        margin = 12
        self.mini_orb.move(
            parent.width()  - self.mini_orb.width()  - margin,
            parent.height() - self.mini_orb.height() - margin,
        )

    def _reposition_overlay_bar(self) -> None:
        """Float the restore/quit chips at the top-right of the overlay orb."""
        parent = self.centralWidget()
        if parent is None:
            return
        self.overlay_bar.adjustSize()
        self.overlay_bar.move(parent.width() - self.overlay_bar.width() - 10, 10)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if not self.mini_orb.isHidden():
            self._reposition_mini_orb()
        if self._overlay_active:
            self._reposition_overlay_bar()

    # ── display modes ─────────────────────────────────────────────────────────

    def toggle_fullscreen(self) -> None:
        """F11 — immersive full-monitor mode (exits overlay first if active)."""
        if self._overlay_active:
            self.exit_overlay_mode()
        if self.isFullScreen():
            self.showNormal()
            self.bus.log.emit("SYS: fullscreen disengaged.")
        else:
            self.showFullScreen()
            self.bus.log.emit("SYS: fullscreen engaged - press F11 or Esc to exit.")

    def toggle_dashboard(self) -> None:
        """Ctrl+D — show/hide the Command Deck on the Widgets page."""
        if self.dashboard is None:
            self.bus.log.emit("SYS: dashboard window not attached.")
            return
        # The deck is a swipeable UnifiedDashboard; jump to its Widgets page.
        if hasattr(self.dashboard, "show_page_named") and self.dashboard.isVisible():
            self.dashboard.show_page_named("WIDGETS")
            self.dashboard.raise_()
            self.dashboard.activateWindow()
            return
        if self.dashboard.isVisible():
            self.dashboard.hide()
        else:
            self.dashboard.show()
            if hasattr(self.dashboard, "show_page_named"):
                self.dashboard.show_page_named("WIDGETS")
            self.dashboard.raise_()
            self.dashboard.activateWindow()

    def _toggle_pause(self) -> None:
        """Pause/resume ORION from the header button."""
        if self.worker is not None:
            self.worker.toggle_pause()
        else:
            self.bus.log.emit("SYS: worker not ready for pause.")

    def _on_paused(self, paused: bool) -> None:
        self.pause_btn.setText("▶ RESUME" if paused else "⏸ PAUSE")

    def toggle_command_centre(self) -> None:
        """Ctrl+Shift+C — show the Command Deck on the Command Centre page."""
        if self.command_centre is None:
            self.bus.log.emit("SYS: command centre not attached.")
            return
        if hasattr(self.command_centre, "show_page_named") and self.command_centre.isVisible():
            self.command_centre.show_page_named("COMMAND")
            self.command_centre.raise_()
            self.command_centre.activateWindow()
            return
        if self.command_centre.isVisible():
            self.command_centre.hide()
        else:
            self.command_centre.show()
            if hasattr(self.command_centre, "show_page_named"):
                self.command_centre.show_page_named("COMMAND")
            self.command_centre.raise_()
            self.command_centre.activateWindow()

    def toggle_overlay_mode(self) -> None:
        if self._overlay_active:
            self.exit_overlay_mode()
        else:
            self.enter_overlay_mode()

    def enter_overlay_mode(self) -> None:
        """
        Compact frameless always-on-top orb.  The HUD keeps animating, the
        microphone keeps listening; drag anywhere to reposition, Esc to exit.
        """
        if self._overlay_active:
            return
        if self.isFullScreen():
            self.showNormal()
        self._overlay_active = True
        self._saved_geometry = self.saveGeometry()
        self._saved_min_size = self.minimumSize()
        self.header_frame.hide()
        self.left_panel.hide()
        self._select_view(0)
        self.stack.setCurrentIndex(0)
        self.mini_orb.hide()
        self.hud.timer.setInterval(33)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setMinimumSize(300, 300)
        self.setWindowOpacity(0.94)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(380, 400)
            self.move(available.right() - 400, available.bottom() - 430)
        self.show()
        # Show the floating restore/quit chips and tell the user how to get back
        # — the header (with its controls) is hidden in overlay mode.
        self.overlay_bar.show()
        self.overlay_bar.raise_()
        self._reposition_overlay_bar()
        self.hud.set_banner("Double-click to restore  ·  ⤢ button  ·  Ctrl+Shift+O", 3)
        self.bus.log.emit("SYS: overlay mode engaged - double-click, the ⤢ chip, "
                          "Esc or Ctrl+Shift+O to restore.")

    def exit_overlay_mode(self) -> None:
        if not self._overlay_active:
            return
        self._overlay_active = False
        self.overlay_bar.hide()
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowOpacity(1.0)
        if self._saved_min_size is not None:
            self.setMinimumSize(self._saved_min_size)
        self.header_frame.show()
        self.left_panel.show()
        if self._saved_geometry is not None:
            self.restoreGeometry(self._saved_geometry)
        self.show()
        self.bus.log.emit("SYS: overlay mode disengaged.")

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Escape:
            if self._overlay_active:
                self.exit_overlay_mode()
                return
            if self.isFullScreen():
                self.showNormal()
                return
        super().keyPressEvent(event)

    # Frameless overlay dragging ------------------------------------------------

    def mousePressEvent(self, event: Any) -> None:
        if self._overlay_active and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if (
            self._overlay_active
            and self._drag_offset is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        # Double-clicking the compact orb restores ORION to the full window.
        if self._overlay_active:
            self.exit_overlay_mode()
            return
        super().mouseDoubleClickEvent(event)

    # ── shutdown ────────────────────────────────────────────────────────────────

    def _quit(self) -> None:
        """Shut ORION down completely — every window and background task."""
        self.bus.log.emit("SYS: shutdown requested by user.")
        self.bus.request_shutdown.emit()

    def closeEvent(self, event: Any) -> None:
        """
        Closing the core window shuts the whole assistant down.

        Other surfaces (the floating pill, the Command Deck) are separate
        top-level windows, so without this ORION would keep running headless
        after the main window closed — which is why `py orion.py` from a
        terminal appeared to 'never quit'. Requesting shutdown here stops the
        background tasks and quits the Qt application cleanly.
        """
        self.bus.request_shutdown.emit()
        super().closeEvent(event)

    # ── bus connections ───────────────────────────────────────────────────────

    def _connect_bus(self) -> None:
        # bus.log is already connected inside LogConsoleView.__init__;
        # connecting it here as well duplicated every console line.
        self.bus.state.connect(self.set_state)
        self.bus.state.connect(self.hud.set_state)
        self.bus.state.connect(self.mini_orb.set_state)
        self.bus.state.connect(self.face.set_state)
        self.bus.amplitude.connect(self.hud.set_amplitude)
        self.bus.amplitude.connect(self.mini_orb.set_amplitude)
        self.bus.amplitude.connect(self.face.set_amplitude)
        self.bus.banner.connect(self.hud.set_banner)
        self.bus.speaking.connect(self.face.set_speaking)
        self.bus.speaking.connect(self._on_speaking_changed)
        self.bus.paused.connect(self._on_paused)
        # Mark X.7: the EmotionStateManager broadcasts full rendering
        # parameter sets; the face morphs to whatever arrives.
        self.bus.emotion_changed.connect(self.face.apply_emotion)
        _app = QApplication.instance()
        if _app is not None:
            self.bus.request_shutdown.connect(_app.quit)

    def _on_speaking_changed(self, active: bool) -> None:
        """Voice-activity LED driven by the SpeechQueueManager."""
        self.voice_led.setText("VOICE ●" if active else "VOICE ○")
        self.voice_led.setProperty("speaking", "true" if active else "false")
        style = self.voice_led.style()
        if style is not None:
            style.unpolish(self.voice_led)
            style.polish(self.voice_led)

        # Auto-morph the orb into the speaking face while ORION talks, then back.
        # Only when we're on the orb HUD (index 0) — never yank the user out of
        # the log, memory or telemetry views — and only undo a switch we made.
        HUD, FACE = 0, 1
        if self._auto_face and not self._overlay_active:
            if active and self.stack.currentIndex() == HUD:
                self._auto_faced = True
                self._select_view(FACE)
            elif not active and self._auto_faced and self.stack.currentIndex() == FACE:
                self._auto_faced = False
                self._select_view(HUD)

    # ── state / logging ───────────────────────────────────────────────────────

    def write_log(self, message: str) -> None:
        self.log_view.write_log(message)

    def set_state(self, state: str) -> None:
        self.active_state       = str(state).upper()
        self.telemetry["state"] = self.active_state
        self.state_label.setText(self.active_state)

    # ── telemetry snapshot ────────────────────────────────────────────────────

    def telemetry_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self.telemetry)
        snapshot["clock"] = datetime.now().strftime("%H:%M:%S")
        if self.worker is not None:
            snapshot["queue_depth"]    = self.worker.out_queue.qsize()
            snapshot["live_connected"] = self.worker.connected
            if hasattr(self.worker, "router"):
                snapshot["providers"] = self.worker.router.provider_snapshot()
        else:
            snapshot["queue_depth"]    = 0
            snapshot["live_connected"] = False
        return snapshot

    # ── environment refresh (async, non-blocking) ─────────────────────────────

    async def refresh_environment_widgets(self) -> None:
        self.location_label.setText("Location: resolving.")
        self.weather_label.setText("Weather: resolving.")
        try:
            timeout = aiohttp_client_timeout()
            async with ClientSession(timeout=timeout) as session:
                async with session.get("https://ipapi.co/json/") as response:
                    if response.status != 200:
                        raise RuntimeError(f"location service returned {response.status}")
                    location = await response.json()
                latitude  = float(location.get("latitude"))
                longitude = float(location.get("longitude"))
                city      = str(location.get("city") or "unknown locality")
                region    = str(location.get("region") or "")
                country   = str(location.get("country_name") or location.get("country") or "")
                self.location_label.setText(
                    f"Location: {city}, {region}, {country}\n"
                    f"{latitude:.4f}, {longitude:.4f}"
                )
                weather_url = (
                    "https://api.open-meteo.com/v1/forecast"
                    f"?latitude={latitude:.5f}&longitude={longitude:.5f}"
                    "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                    "&timezone=auto"
                )
                async with session.get(weather_url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"weather service returned {response.status}")
                    weather = await response.json()
                current  = weather.get("current") or {}
                temp     = current.get("temperature_2m")
                humidity = current.get("relative_humidity_2m")
                wind     = current.get("wind_speed_10m")
                code     = int(current.get("weather_code") or 0)
                self.weather_label.setText(
                    f"Weather: {weather_code_label(code)}; {temp} °C; "
                    f"humidity {humidity}%; wind {wind} km/h."
                )
                self.write_log(f"GEO: environment synchronised for {city}.")
        except Exception as exc:
            self.location_label.setText("Location: unavailable.")
            self.weather_label.setText(
                f"Weather: unavailable - {str(exc).splitlines()[0][:90]}"
            )
            self.write_log(
                f"GEO: environment refresh failed - {str(exc).splitlines()[0][:120]}"
            )

    # ── telemetry loop (async, 0.75 s interval) ───────────────────────────────

    async def start_telemetry(self) -> None:
        last_net  = psutil.net_io_counters()
        last_time = time.monotonic()
        psutil.cpu_percent(interval=None)
        while True:
            try:
                await asyncio.sleep(0.75)
                cpu         = psutil.cpu_percent(interval=None)
                ram         = psutil.virtual_memory().percent
                current_net = psutil.net_io_counters()
                current_time = time.monotonic()
                delta_bytes = (
                    (current_net.bytes_sent + current_net.bytes_recv)
                    - (last_net.bytes_sent + last_net.bytes_recv)
                )
                elapsed       = max(0.001, current_time - last_time)
                bytes_per_sec = max(0.0, delta_bytes / elapsed)
                net_percent   = min(100.0, (bytes_per_sec / 12_500_000.0) * 100.0)
                self.telemetry.update({
                    "cpu":         float(cpu),
                    "ram":         float(ram),
                    "net_bps":     float(bytes_per_sec),
                    "net_percent": float(net_percent),
                    "state":       self.active_state,
                    "updated_at":  utc_stamp(),
                })
                # Update both the left-panel bars and the telemetry view bars
                for bar_set in ((self.cpu_bar, self.ram_bar, self.net_bar),
                                (self.telemetry_view.cpu_bar,
                                 self.telemetry_view.ram_bar,
                                 self.telemetry_view.net_bar)):
                    bar_set[0].set_value(cpu)
                    bar_set[1].set_value(ram)
                    bar_set[2].set_value(net_percent)
                last_net  = current_net
                last_time = current_time
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.write_log(f"TEL: telemetry loop recovered - {exc}")
                await asyncio.sleep(1.0)


# Legacy alias — external references to OrionMainWindow keep working.
OrionMainWindow = OrionCoreWindow
