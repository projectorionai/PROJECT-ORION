"""
OperationsDeckView — the OPS page of the Command Deck.

The remaining directive panels in one two-column scroll surface:

    • Companion       — the continuity brief, goals with progress, habits and
                        achievements from the CompanionEngine.
    • Agent Monitor   — every registered specialist agent and its focus.
    • Task Queue      — cognition's pending tasks; add and complete inline.
    • Project Explorer— active projects from the cognitive state + memory.
    • Memory Viewer   — searchable window over the persistent memory matrix.
    • Security Centre — sentinel status, paired phone devices with revoke,
                        and QR pairing for mobile access.
    • Desktop Control — open/close applications and media transport.

Every panel takes its backing service as an optional constructor argument and
renders a clear "not available" state when unwired, so the page never breaks
in partial assemblies (tests, headless boots).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from ..lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QTextCursor
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..utils import aiohttp_client_timeout, weather_code_label
from .. import background
from .widgets import describe_control


def _panel_into(frame: QFrame, title: str) -> QVBoxLayout:
    """Give *frame* the shared panel chrome; return the layout to fill."""
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return layout


# ──────────────────────────────────────────────────────────────────────────────
# COMPANION
# ──────────────────────────────────────────────────────────────────────────────

class CompanionPanel(QFrame):
    def __init__(self, bus: OrionBus, companion: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.companion = companion
        layout = _panel_into(self, "COMPANION")

        self.brief = QLabel("No continuity yet.")
        self.brief.setWordWrap(True)
        layout.addWidget(self.brief)

        self.lists = QListWidget()
        self.lists.setMaximumHeight(130)
        layout.addWidget(self.lists)

        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self) -> None:
        if self.companion is None:
            self.brief.setText("Companion engine not available.")
            return
        try:
            brief = self.companion.continuity_brief()
            self.brief.setText(brief or "Nothing tracked yet — I will keep notes as you work.")
            self.lists.clear()
            for goal in self.companion.goals()[:4]:
                self.lists.addItem(f"◎ {goal.title} — {goal.progress}%")
            for habit in self.companion.habits()[:3]:
                self.lists.addItem(f"↻ {habit['name']} — {habit['streak']}-day streak")
            for ach in self.companion.achievements(limit=3):
                self.lists.addItem(f"★ {ach['title']}")
        except Exception as exc:
            self.brief.setText(f"Companion unavailable: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# AGENT MONITOR
# ──────────────────────────────────────────────────────────────────────────────

class AgentMonitorPanel(QFrame):
    def __init__(self, bus: OrionBus, agents: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.agents = agents
        layout = _panel_into(self, "AGENT MONITOR")
        self.listing = QListWidget()
        layout.addWidget(self.listing)
        self.refresh()

    def refresh(self) -> None:
        self.listing.clear()
        if self.agents is None:
            self.listing.addItem("Agent manager not available.")
            return
        try:
            for info in self.agents.describe():
                name = info.get("name", "?")
                focus = info.get("focus") or info.get("description") or ""
                calls = info.get("calls", 0)
                last_active = info.get("last_active") or ""
                activity = f" · {calls} call(s), last {last_active}" if calls else " · not yet consulted"
                self.listing.addItem(f"● {name} — {focus}{activity}"[:160])
        except Exception as exc:
            self.listing.addItem(f"Agent listing failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# TASK QUEUE
# ──────────────────────────────────────────────────────────────────────────────

class TaskQueuePanel(QFrame):
    def __init__(self, bus: OrionBus, cognition: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.cognition = cognition
        layout = _panel_into(self, "TASK QUEUE")

        self.listing = QListWidget()
        layout.addWidget(self.listing)

        row = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("New task…")
        self.entry.returnPressed.connect(self._add)
        row.addWidget(self.entry, 1)
        add = QPushButton("Add")
        add.clicked.connect(self._add)
        row.addWidget(add)
        done = QPushButton("Complete ✓")
        done.clicked.connect(self._complete)
        row.addWidget(done)
        layout.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        self.listing.clear()
        if self.cognition is None:
            self.listing.addItem("Cognitive state not available.")
            return
        try:
            state = self.cognition.snapshot()
            tasks = [t for t in (state.get("pending_tasks") or {}).values()
                     if str(t.get("status", "pending")) != "completed"]
            if not tasks:
                self.listing.addItem("No open tasks.")
            for task in sorted(tasks, key=lambda t: str(t.get("due") or "9999")):
                due = f"  (due {task['due']})" if task.get("due") else ""
                proj = f" [{task['project']}]" if task.get("project") else ""
                self.listing.addItem(f"☐ {task.get('title', '?')}{proj}{due}")
        except Exception as exc:
            self.listing.addItem(f"Task listing failed: {exc}")

    def _add(self) -> None:
        text = self.entry.text().strip()
        if not text or self.cognition is None:
            return
        try:
            self.cognition.add_task(text)
            self.entry.clear()
            self.refresh()
        except Exception:
            pass

    def _complete(self) -> None:
        item = self.listing.currentItem()
        if item is None or self.cognition is None:
            return
        title = item.text().lstrip("☐ ").split("  (due")[0].split(" [")[0].strip()
        try:
            if self.cognition.complete_task(title):
                self.refresh()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# PROJECT EXPLORER
# ──────────────────────────────────────────────────────────────────────────────

class ProjectExplorerPanel(QFrame):
    def __init__(self, bus: OrionBus, cognition: Any | None, memory: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.cognition = cognition
        self.memory = memory
        layout = _panel_into(self, "PROJECT EXPLORER")
        self.listing = QListWidget()
        layout.addWidget(self.listing)
        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self) -> None:
        self.listing.clear()
        active = ""
        if self.memory is not None:
            try:
                active = self.memory.active_project
            except Exception:
                active = ""
        if self.cognition is None:
            self.listing.addItem("Cognitive state not available.")
            return
        try:
            state = self.cognition.snapshot()
            projects = state.get("active_projects") or {}
            if not projects:
                self.listing.addItem("No projects tracked yet.")
            for key, proj in projects.items():
                marker = "▶" if key == active else "·"
                goal = f" — {proj.get('goal')}" if proj.get("goal") else ""
                self.listing.addItem(f"{marker} {proj.get('name', key)}{goal}"[:120])
        except Exception as exc:
            self.listing.addItem(f"Project listing failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# MEMORY VIEWER
# ──────────────────────────────────────────────────────────────────────────────

class MemoryViewerPanel(QFrame):
    def __init__(self, bus: OrionBus, memory: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.memory = memory
        layout = _panel_into(self, "MEMORY VIEWER")

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search persistent memory…")
        self.query.returnPressed.connect(self.refresh)
        row.addWidget(self.query, 1)
        go = QPushButton("Search")
        go.clicked.connect(self.refresh)
        row.addWidget(go)
        layout.addLayout(row)

        self.listing = QListWidget()
        layout.addWidget(self.listing)
        self.refresh()

    def refresh(self) -> None:
        self.listing.clear()
        if self.memory is None:
            self.listing.addItem("Memory matrix not available.")
            return
        try:
            rows = self.memory.records(query=self.query.text().strip(), limit=40)
            if not rows:
                self.listing.addItem("No matching memories.")
            for rec in rows:
                cat = rec.get("category", "")
                self.listing.addItem(f"[{cat}] {rec.get('key_ref', '')}: {rec.get('value', '')}"[:160])
        except Exception as exc:
            self.listing.addItem(f"Memory query failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# SECURITY CENTRE (sentinel + mobile pairing with QR)
# ──────────────────────────────────────────────────────────────────────────────

class SecurityCentrePanel(QFrame):
    def __init__(self, bus: OrionBus, sentinel: Any | None,
                 gateway: Any | None = None) -> None:
        super().__init__()
        self.bus = bus
        self.sentinel = sentinel
        self.gateway = gateway
        layout = _panel_into(self, "SECURITY CENTRE")

        self.status = QLabel("Sentinel status unknown.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.devices = QListWidget()
        self.devices.setMaximumHeight(90)
        layout.addWidget(self.devices)

        row = QHBoxLayout()
        pair = QPushButton("▦  Pair phone (QR)")
        pair.clicked.connect(self._pair)
        row.addWidget(pair)
        revoke = QPushButton("Revoke device")
        revoke.clicked.connect(self._revoke)
        row.addWidget(revoke)
        refresh = describe_control(
            QPushButton("⟳"), "Refresh", "Reload this panel now")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout.addLayout(row)

        self.qr = QLabel("")
        self.qr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr.setMinimumHeight(10)
        layout.addWidget(self.qr)
        self.refresh()

    def attach_gateway(self, gateway: Any) -> None:
        self.gateway = gateway
        self.refresh()

    def refresh(self) -> None:
        if self.sentinel is not None:
            try:
                result = self.sentinel.status()
                self.status.setText(getattr(result, "text", str(result))[:400])
            except Exception as exc:
                self.status.setText(f"Sentinel unavailable: {exc}")
        else:
            self.status.setText("Security sentinel not wired.")
        self.devices.clear()
        if self.gateway is None:
            self.devices.addItem("Remote uplink offline — pairing unavailable.")
            return
        try:
            for dev in self.gateway.auth.list_devices():
                flag = "✗ revoked" if dev["revoked"] else "✓ active"
                self.devices.addItem(
                    f"{dev['name']}  ({dev['device_id'][:8]}…)  {flag}")
            if self.devices.count() == 0:
                self.devices.addItem("No paired devices yet.")
        except Exception as exc:
            self.devices.addItem(f"Device listing failed: {exc}")

    def _pair(self) -> None:
        if self.gateway is None:
            return
        try:
            code = self.gateway.begin_pairing()
            # The phone page reads ?pair= and pairs itself; the scheme must
            # match what the uplink serves or the link does not even load.
            scheme = "https" if getattr(self.gateway, "_https_enabled", False) else "http"
            url = f"{scheme}://{self.gateway._lan_ip()}:{self.gateway.port}/?pair={code}"
            image = _qr_image(url)
            if image is not None:
                self.qr.setPixmap(QPixmap.fromImage(image).scaled(
                    180, 180, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation))
                self.qr.setToolTip(url)
            self.qr.setText("" if image is not None else
                            f"Scan unavailable — open {url} and enter code {code}")
            self.status.setText(
                f"Pairing window open (10 min). Code: {code} — scan the QR "
                f"or open {url} on the phone.")
        except Exception as exc:
            self.status.setText(f"Pairing failed: {exc}")

    def _revoke(self) -> None:
        item = self.devices.currentItem()
        if item is None or self.gateway is None:
            return
        text = item.text()
        if "(" not in text:
            return
        short = text.split("(")[1].split("…")[0]
        try:
            for dev in self.gateway.auth.list_devices():
                if dev["device_id"].startswith(short):
                    self.gateway.auth.revoke_device(dev["device_id"])
                    break
            self.refresh()
        except Exception:
            pass


def _qr_image(payload: str) -> Optional[QImage]:
    """QR via the optional 'qrcode' package; None when unavailable."""
    try:
        import qrcode
    except Exception:
        return None
    try:
        qr = qrcode.QRCode(border=1, box_size=1)
        qr.add_data(payload)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        size = len(matrix)
        image = QImage(size, size, QImage.Format.Format_RGB32)
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                image.setPixel(x, y, 0xFF000000 if cell else 0xFFFFFFFF)
        return image
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DESKTOP CONTROL
# ──────────────────────────────────────────────────────────────────────────────

class DesktopControlPanel(QFrame):
    def __init__(self, bus: OrionBus, desktop: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.desktop = desktop
        layout = _panel_into(self, "DESKTOP CONTROL")

        row = QHBoxLayout()
        self.app_entry = QLineEdit()
        self.app_entry.setPlaceholderText("Application name…")
        row.addWidget(self.app_entry, 1)
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(lambda: self._app("open"))
        row.addWidget(open_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: self._app("close"))
        row.addWidget(close_btn)
        layout.addLayout(row)

        media = QHBoxLayout()
        for label, action in (("⏯", "play_pause"), ("⏮", "previous"),
                              ("⏭", "next"), ("🔉", "volume_down"),
                              ("🔊", "volume_up"), ("🔇", "mute")):
            btn = QPushButton(label)
            btn.setObjectName("iconButton")
            btn.clicked.connect(lambda _c=False, a=action: self._media(a))
            media.addWidget(btn)
        layout.addLayout(media)

        self.feedback = QLabel("")
        self.feedback.setObjectName("mutedLabel")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)

    def _app(self, action: str) -> None:
        name = self.app_entry.text().strip()
        if not name or self.desktop is None:
            self.feedback.setText("Desktop agent not available." if self.desktop is None else "")
            return
        try:
            result = (self.desktop.open_app(name) if action == "open"
                      else self.desktop.close_app(name))
            self.feedback.setText(getattr(result, "text", str(result))[:200])
        except Exception as exc:
            self.feedback.setText(f"Desktop control failed: {exc}")

    def _media(self, action: str) -> None:
        if self.desktop is None:
            self.feedback.setText("Desktop agent not available.")
            return
        try:
            result = self.desktop.media_control(action)
            self.feedback.setText(getattr(result, "text", str(result))[:200])
        except Exception as exc:
            self.feedback.setText(f"Media control failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# EXECUTIVE
# ──────────────────────────────────────────────────────────────────────────────
# ExecutiveCore (decision/priority/strategy engine, executive_core.py) was
# constructed and registered with the dispatcher but never given a GUI panel
# — zero visibility beyond chatting with ORION about it, same category of
# gap the Diagnostics Centre panel had. focus() is its richest single call:
# priority queue + strategic recommendations + blind spots in one report.

class ExecutivePanel(QFrame):
    def __init__(self, bus: OrionBus, executive_core: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.executive_core = executive_core
        layout = _panel_into(self, "EXECUTIVE")

        self.summary = QLabel("No executive focus captured yet.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.refresh()

    def refresh(self) -> None:
        if self.executive_core is None:
            self.summary.setText("Executive core not available.")
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop (e.g. constructed outside qasync) — the
            # panel just keeps showing its last-known text rather than
            # raising out of a Qt slot. Checked before building the
            # coroutine so there is never an unawaited one to warn about.
            return
        background.spawn(self._refresh_async())

    async def _refresh_async(self) -> None:
        try:
            result = await self.executive_core.focus()
            self.summary.setText(getattr(result, "text", str(result))[:1600])
        except Exception as exc:
            self.summary.setText(f"Executive core unavailable: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# INNER VOICE (streaming thought mirror — moved off the face-only Core Window)
# ──────────────────────────────────────────────────────────────────────────────

class InnerVoicePanel(QFrame):
    """ORION's live thought stream, rendering tokens as they arrive and
    falling back to a typewriter reveal for any thought delivered whole."""

    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        self.bus = bus
        layout = _panel_into(self, "INNER VOICE")

        self.thought_box = QPlainTextEdit()
        self.thought_box.setReadOnly(True)
        self.thought_box.setObjectName("thoughtBox")
        self.thought_box.setMaximumBlockCount(400)
        self.thought_box.setMinimumHeight(180)
        self.thought_box.setPlaceholderText(
            "ORION's free thoughts stream here — reflections on what he notices "
            "and why he acts, typed out live as he thinks.")
        layout.addWidget(self.thought_box, 1)

        self._thought_queue: list[str] = []
        self._typing_text = ""
        self._typing_pos = 0
        self._caret_on = False
        self._stream_active = False
        self._stream_id: Any = None
        self._stream_anchor: Any = None
        self._streamed_ids: set[Any] = set()
        self._type_timer = QTimer(self)
        self._type_timer.setInterval(24)
        self._type_timer.timeout.connect(self._thought_type_tick)
        for signal, slot in ((self.bus.thought_delta, self._on_thought_delta),
                             (self.bus.thought, self._on_thought_full)):
            try:
                signal.connect(slot)
            except Exception:
                pass

    def _thought_insert(self, text: str) -> None:
        cursor = self.thought_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.thought_box.setTextCursor(cursor)
        self.thought_box.verticalScrollBar().setValue(
            self.thought_box.verticalScrollBar().maximum())

    def _thought_clear_caret(self) -> None:
        if not self._caret_on:
            return
        cursor = self.thought_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.deletePreviousChar()
        self._caret_on = False

    def _on_thought_delta(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        phase = payload.get("phase")
        tid = payload.get("id")
        if phase == "start":
            self._type_timer.stop()
            self._thought_clear_caret()
            self._stream_active = True
            self._stream_id = tid
            # Remember where this streamed line begins, so a mid-stream failure
            # ("abort") can cleanly drop the partial before the full thought is
            # typewriter-revealed instead.
            cursor = self.thought_box.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._stream_anchor = cursor.position()
            marker = "⚙" if payload.get("kind") == "decision" else "◈"
            self.thought_box.appendPlainText(f"{payload.get('at', '')} {marker} ")
        elif phase == "delta":
            if self._stream_active and tid == self._stream_id:
                self._thought_insert(str(payload.get("text", "")))
        elif phase == "abort":
            # Streaming failed part-way through; remove the partial line so the
            # complete thought (arriving next on bus.thought) reveals in full
            # rather than leaving "just what he'd noted so far" on screen.
            if self._stream_active and tid == self._stream_id:
                self._thought_clear_caret()
                if self._stream_anchor is not None:
                    cursor = self.thought_box.textCursor()
                    cursor.setPosition(self._stream_anchor)
                    cursor.movePosition(QTextCursor.MoveOperation.End,
                                        QTextCursor.MoveMode.KeepAnchor)
                    cursor.removeSelectedText()
            self._stream_active = False
            self._stream_id = None
            self._stream_anchor = None
        elif phase == "end":
            if tid is not None:
                self._streamed_ids.add(tid)
            self._stream_active = False
            self._stream_id = None
            self._stream_anchor = None

    def _on_thought_full(self, payload: Any) -> None:
        """A completed thought — only typewriter-reveal it if it was not already
        shown live via the streaming deltas."""
        if not isinstance(payload, dict):
            return
        tid = payload.get("id")
        if tid in self._streamed_ids:
            self._streamed_ids.discard(tid)
            return
        marker = "⚙" if payload.get("kind") == "decision" else "◈"
        self._thought_queue.append(
            f"{payload.get('at', '')} {marker} {payload.get('text', '')}")
        if not self._type_timer.isActive() and not self._stream_active:
            self._thought_begin_next()

    def _thought_begin_next(self) -> None:
        if not self._thought_queue:
            return
        self._typing_text = self._thought_queue.pop(0)
        self._typing_pos = 0
        self.thought_box.appendPlainText("")
        self._type_timer.start()

    def _thought_type_tick(self) -> None:
        # See command_centre._type_tick: ~64 ms of every second on the event
        # loop, spent revealing characters one at a time into a widget that
        # may not be on screen. The thought still lands in full.
        if not self._thought_visible():
            self._thought_finish_now()
            return
        self._thought_clear_caret()
        if self._typing_pos >= len(self._typing_text):
            self._type_timer.stop()
            if self._thought_queue:
                QTimer.singleShot(320, self._thought_begin_next)
            return
        chunk = self._typing_text[self._typing_pos:self._typing_pos + 2]
        self._typing_pos += len(chunk)
        self._thought_insert(chunk)
        self._caret_on = self._typing_pos < len(self._typing_text)
        if self._caret_on:
            self._thought_insert("▌")

    def _thought_visible(self) -> bool:
        """Whether the thought panel is actually on screen. Never raises: a
        widget torn down mid-tick must not take the tick with it."""
        try:
            box = getattr(self, "thought_box", None)
            return bool(box is not None and box.isVisible())
        except Exception:
            return False

    def _thought_finish_now(self) -> None:
        """Drop the remaining text in at once and stop the timer."""
        try:
            self._thought_clear_caret()
            remaining = self._typing_text[self._typing_pos:]
            if remaining:
                self._thought_insert(remaining)
            self._typing_pos = len(self._typing_text)
            self._caret_on = False
            self._type_timer.stop()
            if self._thought_queue:
                QTimer.singleShot(320, self._thought_begin_next)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT (calendar + geographical node — moved off the face-only Core Window)
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentPanel(QFrame):
    """Calendar plus IP-geolocation + weather, refreshed on demand and on a
    background loop (started externally via run_environment_loop())."""

    # Two keyless IP-geolocation providers, tried in order — ipapi.co rate-limits
    # aggressively (HTTP 429), which was a common reason the node stuck on
    # "awaiting refresh"; ipwho.is is the fallback so a single provider's outage
    # no longer blanks the panel.
    _GEO_PROVIDERS = ("https://ipapi.co/json/", "https://ipwho.is/")

    def __init__(self, bus: OrionBus) -> None:
        super().__init__()
        self.bus = bus
        layout = _panel_into(self, "ENVIRONMENT")

        self.calendar_widget = QCalendarWidget()
        self.calendar_widget.setGridVisible(False)
        self.calendar_widget.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        self.calendar_widget.setMaximumHeight(200)
        layout.addWidget(self.calendar_widget)

        self.location_label = QLabel("Location: awaiting refresh.")
        self.location_label.setObjectName("mutedLabel")
        self.location_label.setWordWrap(True)
        self.weather_label = QLabel("Weather: awaiting refresh.")
        self.weather_label.setObjectName("mutedLabel")
        self.weather_label.setWordWrap(True)
        layout.addWidget(self.location_label)
        layout.addWidget(self.weather_label)

        refresh = QPushButton("⟳  Refresh environment")
        refresh.clicked.connect(
            lambda: background.spawn(self.refresh_environment_widgets())
        )
        layout.addWidget(refresh)

    async def _resolve_location(self, session: Any) -> tuple[float, float, str, str, str]:
        last_error: Exception | None = None
        for url in self._GEO_PROVIDERS:
            try:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise RuntimeError(f"{url} returned {response.status}")
                    data = await response.json()
                lat = data.get("latitude", data.get("lat"))
                lon = data.get("longitude", data.get("lon", data.get("longitude")))
                if lat is None or lon is None:
                    raise RuntimeError(f"{url} returned no coordinates")
                city    = str(data.get("city") or "unknown locality")
                region  = str(data.get("region") or data.get("region_name") or "")
                country = str(data.get("country_name") or data.get("country") or "")
                return float(lat), float(lon), city, region, country
            except Exception as exc:
                last_error = exc
        raise RuntimeError(str(last_error) if last_error else "no location provider responded")

    async def refresh_environment_widgets(self) -> bool:
        """Resolve the host's location + current weather.  Returns True on
        success so the autonomous loop can back off appropriately."""
        self.location_label.setText("Location: resolving.")
        self.weather_label.setText("Weather: resolving.")
        try:
            timeout = aiohttp_client_timeout()
            async with ClientSession(timeout=timeout) as session:
                latitude, longitude, city, region, country = await self._resolve_location(session)
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
                self.bus.log.emit(f"GEO: environment synchronised for {city}.")
                return True
        except Exception as exc:
            self.location_label.setText("Location: unavailable — retrying.")
            self.weather_label.setText(
                f"Weather: unavailable - {str(exc).splitlines()[0][:90]}"
            )
            self.bus.log.emit(
                f"GEO: environment refresh failed - {str(exc).splitlines()[0][:120]}"
            )
            return False

    async def run_environment_loop(self) -> None:
        """Keep the geographical node current autonomously.  Refreshes every 15
        minutes on success; retries every 60 s while it can't reach a provider
        (so a failed boot-time lookup — no network yet — heals itself instead of
        sticking on 'awaiting refresh')."""
        while True:
            ok = False
            try:
                ok = await self.refresh_environment_widgets()
            except Exception:
                ok = False
            await asyncio.sleep(900 if ok else 60)


# ──────────────────────────────────────────────────────────────────────────────
# THE PAGE
# ──────────────────────────────────────────────────────────────────────────────

class OperationsDeckView(QWidget):
    """Two-column scroll page hosting the remaining Command Deck panels."""

    REFRESH_MS = 30_000

    def __init__(
        self,
        bus: OrionBus,
        *,
        companion: Any | None = None,
        agents: Any | None = None,
        cognition: Any | None = None,
        memory: Any | None = None,
        sentinel: Any | None = None,
        desktop: Any | None = None,
        gateway: Any | None = None,
        executive_core: Any | None = None,
    ) -> None:
        super().__init__()
        self.bus = bus

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(10)

        self.companion_panel = CompanionPanel(bus, companion)
        self.agent_panel = AgentMonitorPanel(bus, agents)
        self.task_panel = TaskQueuePanel(bus, cognition)
        self.project_panel = ProjectExplorerPanel(bus, cognition, memory)
        self.memory_panel = MemoryViewerPanel(bus, memory)
        self.security_panel = SecurityCentrePanel(bus, sentinel, gateway)
        self.desktop_panel = DesktopControlPanel(bus, desktop)
        self.executive_panel = ExecutivePanel(bus, executive_core)
        self.inner_voice_panel = InnerVoicePanel(bus)
        self.environment_panel = EnvironmentPanel(bus)

        cells = [
            self.companion_panel, self.security_panel,
            self.task_panel, self.project_panel,
            self.agent_panel, self.memory_panel,
            self.desktop_panel, self.executive_panel,
            self.inner_voice_panel, self.environment_panel,
        ]
        for index, panel in enumerate(cells):
            grid.addWidget(panel, index // 2, index % 2)
        grid.setRowStretch((len(cells) + 1) // 2, 1)
        scroll.setWidget(content)

        # Gentle periodic refresh of read-only panels (never while hidden).
        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def attach_gateway(self, gateway: Any) -> None:
        """The remote uplink starts after the deck is built; attach late."""
        self.security_panel.attach_gateway(gateway)

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self.companion_panel.refresh()
        self.task_panel.refresh()
        self.project_panel.refresh()
        self.security_panel.refresh()
        self.executive_panel.refresh()
