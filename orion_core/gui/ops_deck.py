"""
OperationsDeckView — the OPS page of the Command Deck.

The remaining directive panels in one two-column scroll surface:

    • Companion       — the continuity brief, goals with progress, habits and
                        achievements from the CompanionEngine.
    • Agent Monitor   — every registered specialist agent and its focus.
    • Task Queue      — cognition's pending tasks; add and complete inline.
    • Project Explorer— active projects from the cognitive state + memory.
    • Memory Viewer   — searchable window over the persistent memory matrix.
    • Desktop Control — open/close applications and media transport.

Every panel takes its backing service as an optional constructor argument and
renders a clear "not available" state when unwired, so the page never breaks
in partial assemblies (tests, headless boots).
"""

from __future__ import annotations

import asyncio
from typing import Any

# aiohttp costs ~178 ms to import and is only used inside coroutines;
# deferring it keeps it off ORION's startup path (see lazy_import).
from ..lazy_import import lazy_attr

ClientSession = lazy_attr("aiohttp", "ClientSession")
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..utils import aiohttp_client_timeout, weather_code_label
from .. import background
from .labels import plural
from .widgets import describe_control
from .reflections import ReflectionsPanel


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
        layout = _panel_into(self, "Companion")

        self.brief = QLabel("No continuity yet.")
        self.brief.setWordWrap(True)
        layout.addWidget(self.brief)

        self.lists = QListWidget()
        self.lists.setMaximumHeight(130)
        layout.addWidget(self.lists)

        self.status = QLabel("")
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        refresh = QPushButton("⟳  Refresh")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self._forgetting = False
        self.refresh()

    #: Shown only when there is truly nothing (25 Oct 2026: this line sat
    #: above three starred achievements, contradicting them).
    EMPTY = "Nothing tracked yet — I will keep notes as you work."

    def refresh(self) -> None:
        if self.companion is None:
            self.brief.setText("Companion engine not available.")
            return
        try:
            brief = self.companion.continuity_brief()
            entries: list[tuple[str, dict[str, Any]]] = []
            for goal in self.companion.goals()[:4]:
                entries.append((f"◎ {goal.title} — {goal.progress}%",
                                {"store": "companion", "table": "goals", "id": goal.slug,
                                 "title": goal.title}))
            for habit in self.companion.habits()[:3]:
                entries.append((f"↻ {habit['name']} — {habit['streak']}-day streak",
                                {"store": "companion", "table": "habits", "id": habit["name"],
                                 "title": habit["name"]}))
            for ach in self.companion.achievements(limit=3):
                entries.append((f"★ {ach['title']}",
                                {"store": "companion", "table": "achievements", "id": ach.get("id"),
                                 "title": ach["title"], "detail": ach.get("detail", "")}))
            self.brief.setText(brief or ("No recent activity — goals, habits and wins on record:"
                                         if entries else self.EMPTY))
            self.lists.clear()
            for text, item in entries:
                self._add_entry(text, item)
        except Exception as exc:
            self.brief.setText(f"Companion unavailable: {exc}")

    def _add_entry(self, text: str, item: dict[str, Any]) -> None:
        """One ledger line with its own Forget control."""
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(2, 0, 2, 0)
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.PlainText)
        line.addWidget(label, 1)
        forget = QPushButton("Forget")
        forget.setObjectName("ghostButton")
        forget.setAccessibleName(f"Forget {item.get('title') or text}")
        forget.setToolTip("Remove this, and every copy of it ORION holds, after you confirm")
        forget.clicked.connect(lambda _=False, i=dict(item): self._forget(i))
        line.addWidget(forget)
        entry = QListWidgetItem()
        entry.setSizeHint(row.sizeHint())
        entry.setData(Qt.ItemDataRole.UserRole, dict(item))
        self.lists.addItem(entry)
        self.lists.setItemWidget(entry, row)

    def _forget_engine(self) -> Any:
        from ..forget import ForgetEngine
        return ForgetEngine(memory=getattr(self.companion, "memory", None),
                            companion=self.companion,
                            cognition=getattr(self.companion, "cognition", None),
                            log=self.bus.log.emit)

    def _confirm_box(self, title: str, summary: str) -> tuple[Any, Any]:
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Forget This")
        box.setText(f"Forget {title}?")
        box.setInformativeText(summary)
        forget = box.addButton("Forget", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(box.buttons()[-1])      # Cancel is the safe default
        return box, forget

    def _confirm(self, title: str, summary: str) -> bool:
        """Blocking - only where no asyncio task is running (see _ask)."""
        box, forget = self._confirm_box(title, summary)
        box.exec()
        return box.clickedButton() is forget

    async def _ask(self, title: str, summary: str) -> bool:
        """The same question without a nested event loop.

        28 Oct 2026: box.exec() inside the forget task ran a second Qt loop
        INSIDE an asyncio task step; every other task the loop tried to wake
        meanwhile failed with "Cannot enter into task" and was never resumed -
        orion-send-realtime among them, so Gemini Live got no more microphone
        audio and ORION answered whatever the small local recogniser misheard.
        open() shows the same window-modal box and returns at once."""
        box, forget = self._confirm_box(title, summary)
        answered: asyncio.Future = asyncio.get_running_loop().create_future()

        def done(_result: int) -> None:
            if not answered.done():
                answered.set_result(box.clickedButton() is forget)

        box.finished.connect(done)
        box.open()
        try:
            return await answered
        finally:
            box.deleteLater()

    def _forget(self, item: dict[str, Any]) -> None:
        """Survey every store (off the GUI thread), show the owner exactly
        what goes, and remove it only on their explicit approval."""
        if self.companion is None or self._forgetting:
            return
        engine = self._forget_engine()
        title = f"'{item.get('title') or 'this'}'"

        def finish(receipt: dict[str, Any]) -> None:
            removed = sum(receipt["removed"].values())
            redacted = sum(receipt["redacted"].values())
            text = f"Forgot {title}: {plural(removed, 'record')} removed"
            if redacted:
                text += f", {redacted} redacted"
            stores = sorted(set(receipt["removed"]) | set(receipt["redacted"]))
            if stores:
                text += " from " + ", ".join(stores)
            if receipt["problems"]:
                text += ". Could not finish: " + "; ".join(receipt["problems"])
            self.status.setText(text + ".")
            self.refresh()

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            plan = engine.survey(item=item)
            if self._confirm(title, plan.summary()):
                finish(engine.execute(plan))
            return

        async def flow() -> None:
            from PyQt6 import sip
            self._forgetting = True
            try:
                plan = await asyncio.to_thread(engine.survey, item=item)
                if sip.isdeleted(self) or not await self._ask(title, plan.summary()):
                    return
                receipt = await asyncio.to_thread(engine.execute, plan)
                if not sip.isdeleted(self):
                    finish(receipt)
            except Exception as exc:
                if not sip.isdeleted(self):
                    self.status.setText(f"Could not forget {title}: {exc}")
            finally:
                self._forgetting = False

        background.spawn(flow(), name="orion-companion-forget")


# ──────────────────────────────────────────────────────────────────────────────
# AGENT MONITOR
# ──────────────────────────────────────────────────────────────────────────────

class AgentMonitorPanel(QFrame):
    def __init__(self, bus: OrionBus, agents: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.agents = agents
        layout = _panel_into(self, "Agent Monitor")
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
                activity = f" · {plural(calls, 'call')}, last {last_active}" if calls else " · not yet consulted"
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
        layout = _panel_into(self, "Task Queue")

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
        selected = self.listing.selectedItems()
        selected_key = selected[0].data(Qt.ItemDataRole.UserRole) if selected else None
        self.listing.clear()
        if self.cognition is None:
            self.listing.addItem("Cognitive state not available.")
            return
        try:
            state = self.cognition.snapshot()
            tasks = [(key, t) for key, t in (state.get("pending_tasks") or {}).items()
                     if str(t.get("status", "pending")) != "completed"]
            if not tasks:
                self.listing.addItem("No open tasks.")
            for key, task in sorted(tasks, key=lambda pair: str(pair[1].get("due") or "9999")):
                due = f"  (due {task['due']})" if task.get("due") else ""
                proj = f" [{task['project']}]" if task.get("project") else ""
                item = QListWidgetItem(f"☐ {task.get('title', '?')}{proj}{due}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.listing.addItem(item)
                if key == selected_key:
                    self.listing.setCurrentItem(item)
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
        selected = self.listing.selectedItems()
        key = selected[0].data(Qt.ItemDataRole.UserRole) if selected else None
        if not key or self.cognition is None:
            return
        try:
            if self.cognition.complete_task(key, exact=True):
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
        layout = _panel_into(self, "Project Explorer")
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
        layout = _panel_into(self, "Memory Viewer")

        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search memory…")
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
# DESKTOP CONTROL
# ──────────────────────────────────────────────────────────────────────────────

class DesktopControlPanel(QFrame):
    def __init__(self, bus: OrionBus, desktop: Any | None) -> None:
        super().__init__()
        self.bus = bus
        self.desktop = desktop
        layout = _panel_into(self, "Desktop Control")

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
        # Words, not media glyphs: Windows draws those as colour emoji, and
        # the glyph-only buttons had no name for a screen reader either.
        for label, action in (("Play / Pause", "play_pause"), ("Previous", "previous"),
                              ("Next", "next"), ("Volume Down", "volume_down"),
                              ("Volume Up", "volume_up"), ("Mute", "mute")):
            btn = QPushButton(label)
            btn.setObjectName("ghostButton")
            btn.setAccessibleName(f"Media: {label.lower()}")
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
        layout = _panel_into(self, "Executive")

        self.summary = QLabel("No focus set")
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
# Compatibility name for the reflections panel, now mounted in Cognition.
# ──────────────────────────────────────────────────────────────────────────────

class InnerVoicePanel(ReflectionsPanel):
    """Compatibility name for integrations using the old activity-notes panel.

    The live panel now belongs to Cognition and shows completed, bounded notes.
    """


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
        layout = _panel_into(self, "Environment")

        self.calendar_widget = QCalendarWidget()
        self.calendar_widget.setGridVisible(False)
        self.calendar_widget.setVerticalHeaderFormat(
            QCalendarWidget.VerticalHeaderFormat.NoVerticalHeader
        )
        self.calendar_widget.setMaximumHeight(200)
        layout.addWidget(self.calendar_widget)

        self.location_label = QLabel("Location: —")
        self.location_label.setObjectName("mutedLabel")
        self.location_label.setWordWrap(True)
        self.weather_label = QLabel("Weather: —")
        self.weather_label.setObjectName("mutedLabel")
        self.weather_label.setWordWrap(True)
        layout.addWidget(self.location_label)
        layout.addWidget(self.weather_label)

        refresh = QPushButton("⟳  Refresh Environment")
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
        self.gateway = gateway
        self.desktop_panel = DesktopControlPanel(bus, desktop)
        self.executive_panel = ExecutivePanel(bus, executive_core)
        self.environment_panel = EnvironmentPanel(bus)

        cells = [
            self.companion_panel, self.agent_panel,
            self.task_panel, self.project_panel,
            self.memory_panel,
            self.desktop_panel, self.executive_panel,
            self.environment_panel,
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
        self.gateway = gateway

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self.companion_panel.refresh()
        self.task_panel.refresh()
        self.project_panel.refresh()
        self.executive_panel.refresh()
