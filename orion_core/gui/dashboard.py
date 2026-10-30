"""
Widget Dashboard Window — the second window of the Mark VIII dual-window
architecture: productivity tools, agent controls and information panels.

Four live panels in a 2×2 grid:

    SPECIALIST AGENTS   — consult any registered agent (or auto-route) and
                          watch the activity feed.
    OUTLOOK COMMAND     — inbox/priority digests and the approval queue:
                          drafts composed by ORION are listed here and are
                          only transmitted through the explicit SEND button
                          (with a confirmation dialog) or a confirmed voice
                          command.
    NOTION WORKSPACE    — task list, agenda and quick task capture.
    BRIEFINGS FOR TODAY — run the briefing on demand and read the latest
                          composed intelligence picture.

Panels update through ``bus.dashboard_event`` so services never touch
widgets directly, keeping the dashboard fully decoupled from the runtime.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any

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
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:          # annotation-only; keeps agents off the startup path
    from ..agents import AgentManager
from ..bus import OrionBus
from ..constants import APP_NAME
from ..data import ToolResult
from ..notion import NotionService
from ..outlook import OutlookService
from ..utils import now_stamp
from .. import background


class WidgetDashboardWindow(QMainWindow):
    def __init__(
        self,
        bus: OrionBus,
        agent_manager: AgentManager,
        outlook: OutlookService,
        notion: NotionService,
        dispatcher: Any,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.agent_manager = agent_manager
        self.outlook = outlook
        self.notion = notion
        self.dispatcher = dispatcher

        self.setWindowTitle(f"{APP_NAME} — Widget Dashboard")
        self.setMinimumSize(980, 660)
        self.resize(1160, 760)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        outer.addWidget(self._build_header())

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(self._build_today_panel(),    0, 0, 1, 2)
        grid.addWidget(self._build_agents_panel(),   1, 0)
        grid.addWidget(self._build_outlook_panel(),  1, 1)
        grid.addWidget(self._build_notion_panel(),   2, 0)
        grid.addWidget(self._build_briefing_panel(), 2, 1)
        grid.setRowStretch(0, 0)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        outer.addLayout(grid, 1)

        # Scrolls rather than squeezes: on a deck shorter than the panels
        # need, Qt crushed the buttons below their height and cut their
        # labels in half ("REFRESH INBOX" lost its lower half).
        scroll = QScrollArea()
        scroll.setObjectName("zoneScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        # Word-wrapped labels make the scroll area size the page to its
        # PREFERRED height, and a text box prefers ~200 px: every panel
        # ballooned past the screen. The boxes take what is left instead,
        # never less than a few lines.
        for box in (self.agent_output, self.email_list, self.notion_list, self.briefing_box):
            policy = box.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Policy.Ignored)
            box.setSizePolicy(policy)
            box.setMinimumHeight(72)

        # ── bus wiring (services publish, panels render) ─────────────────────
        self.bus.dashboard_event.connect(self._on_dashboard_event)
        self.bus.agent_activity.connect(self._on_agent_activity)
        self.bus.state.connect(self._on_state)
        self.bus.speaking.connect(self._on_speaking)

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)

        title = QLabel("Widget Dashboard")
        title.setObjectName("titleLabel")
        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        text_box.addWidget(title)

        self.voice_led = QLabel("VOICE ○")
        self.voice_led.setObjectName("voiceLed")
        self.state_label = QLabel("INITIALISING")
        self.state_label.setObjectName("stateLabel")
        self.clock_label = QLabel(datetime.now().strftime("%H:%M:%S"))
        self.clock_label.setObjectName("clockLabel")

        layout.addLayout(text_box, 1)
        layout.addWidget(self.voice_led)
        layout.addWidget(self.state_label)
        layout.addWidget(self.clock_label)

        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(
            lambda: self.clock_label.setText(datetime.now().strftime("%H:%M:%S"))
        )
        self._clock_timer.start()
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

    # ── today at a glance ─────────────────────────────────────────────────────
    # The weather where the user is and the market indices, without asking
    # for a spoken briefing. Fetched when the page is shown (and every 15
    # minutes while it stays on screen), never more than once in ten.

    TODAY_STALE_S = 600.0

    def _build_today_panel(self) -> QFrame:
        frame, layout = self._panel("Today")
        row = QHBoxLayout()
        row.setSpacing(24)
        weather_box = QVBoxLayout()
        weather_box.setSpacing(2)
        self.today_weather = QLabel("")
        self.today_weather.setObjectName("todayHeadline")
        self.today_weather.setWordWrap(True)
        self.today_detail = QLabel("")
        self.today_detail.setObjectName("mutedLabel")
        self.today_detail.setWordWrap(True)
        weather_box.addWidget(self.today_weather)
        weather_box.addWidget(self.today_detail)
        # Indices, ETFs and stocks as a chart (Mark XXXX.4; markets.watchlist).
        from .market_chart import MarketChart
        self.market_chart = MarketChart()
        self.today_refresh = QPushButton("Refresh")
        self.today_refresh.setToolTip("Fetch the weather and the markets again.")
        self.today_refresh.clicked.connect(lambda: self.refresh_today(force=True))
        row.addLayout(weather_box, 1)
        row.addWidget(self.today_refresh, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)
        layout.addSpacing(8)
        # The same markets as a heat map; a tile opens its chart (28 Oct 2026).
        from PyQt6.QtWidgets import QButtonGroup, QStackedWidget
        from .hud_instruments import MarketHeatMap
        self.market_heat = MarketHeatMap(on_pick=self._open_market_chart)
        self.market_stack = QStackedWidget()
        self.market_stack.addWidget(self.market_chart)
        self.market_stack.addWidget(self.market_heat)
        view_toggle = QFrame()
        view_toggle.setObjectName("segmented")
        toggle_row = QHBoxLayout(view_toggle)
        toggle_row.setContentsMargins(3, 3, 3, 3)
        toggle_row.setSpacing(2)
        self._market_views = QButtonGroup(view_toggle)
        self._market_views.setExclusive(True)
        self.market_view_buttons: dict[str, QPushButton] = {}
        for index, label in enumerate(("Chart", "Heat Map")):
            button = QPushButton(label)
            button.setObjectName("formItem")
            button.setCheckable(True)
            button.setChecked(index == 0)
            self._market_views.addButton(button, index)
            toggle_row.addWidget(button)
            self.market_view_buttons[label] = button
        self._market_views.idClicked.connect(self.market_stack.setCurrentIndex)
        markets_head = QHBoxLayout()
        markets_head.addStretch(1)
        markets_head.addWidget(view_toggle)
        layout.addLayout(markets_head)
        layout.addWidget(self.market_stack, 1)
        self._today_at: float | None = None
        self._today_busy = False
        self._today_timer = QTimer(self)
        self._today_timer.setInterval(15 * 60 * 1000)
        self._today_timer.timeout.connect(lambda: self.isVisible() and self.refresh_today(force=True))
        self._today_timer.start()
        return frame

    def showEvent(self, event: Any) -> None:  # noqa: N802 — Qt override
        super().showEvent(event)
        # A bare window (tests, no dispatcher) never reaches the network.
        if self.dispatcher is not None:
            self.refresh_today()

    def refresh_today(self, force: bool = False) -> bool:
        """Fetch the glance unless one is in flight or fresh. True if started."""
        import time
        if self._today_busy:
            return False
        if not force and self._today_at is not None and time.monotonic() - self._today_at < self.TODAY_STALE_S:
            return False
        self._today_busy = True
        self.today_refresh.setEnabled(False)
        background.spawn(self._load_today())
        return True

    async def _load_today(self) -> None:
        import time
        try:
            weather_result, market_result = await asyncio.gather(
                self._today_weather_text(), self._today_markets(), return_exceptions=True)
            if isinstance(weather_result, tuple):
                self.today_weather.setText(weather_result[0])
                self.today_detail.setText(weather_result[1])
            else:
                self.today_weather.setText("Weather unavailable right now.")
                self.today_detail.setText("")
            groups = market_result if isinstance(market_result, dict) else {}
            self.market_chart.set_quotes(groups)
            self.market_heat.set_quotes(groups)
        finally:
            self._today_at = time.monotonic()
            self._today_busy = False
            self.today_refresh.setEnabled(True)

    async def _today_weather_text(self) -> tuple[str, str]:
        from .. import weather
        temporal = getattr(getattr(self.dispatcher, "live_worker", None), "temporal", None)
        point = temporal.coordinates() if temporal is not None else None
        if point is None:
            return ("Weather: your location isn't known yet.",
                    "Ask “what's the weather in …” with your town and ORION will check it.")
        place = str(temporal.locality() or "").split(",")[0].strip()
        forecast = await weather.fetch(point[0], point[1], place)
        return weather.glance(forecast)

    def _open_market_chart(self, group: str, symbol: str) -> None:
        """A heat-map tile opens that market's chart."""
        self.market_chart.choose(group=group)
        self.market_chart.choose(symbol=symbol)
        self.market_view_buttons["Chart"].setChecked(True)
        self.market_stack.setCurrentWidget(self.market_chart)

    async def _today_markets(self) -> dict[str, list[Any]]:
        """Every watched group's quotes with today's prices, fetched together."""
        from aiohttp import ClientSession, ClientTimeout
        from .. import markets
        groups = await asyncio.to_thread(markets.watchlist)
        async with ClientSession(timeout=ClientTimeout(total=10.0, connect=4.0)) as session:
            fetched = await asyncio.gather(
                *(markets.fetch(session, rows, intraday=True) for rows in groups.values()))
        return {name: quotes for name, quotes in zip(groups, fetched) if quotes}

    # ── specialist agents panel ───────────────────────────────────────────────

    def _build_agents_panel(self) -> QFrame:
        frame, layout = self._panel("Specialist Agents")

        roster = QLabel(
            "  •  ".join(a["title"] for a in self.agent_manager.describe())
        )
        roster.setObjectName("mutedLabel")
        roster.setWordWrap(True)

        self.agent_combo = QComboBox()
        self.agent_combo.addItem("auto — route by content", "auto")
        for agent in self.agent_manager.describe():
            self.agent_combo.addItem(agent["title"], agent["name"])

        self.agent_input = QLineEdit()
        self.agent_input.setPlaceholderText("Ask a specialist…")
        self.agent_input.returnPressed.connect(self._consult_agent)
        consult_btn = QPushButton("Consult")
        consult_btn.clicked.connect(self._consult_agent)

        input_row = QHBoxLayout()
        input_row.addWidget(self.agent_combo)
        input_row.addWidget(self.agent_input, 1)
        input_row.addWidget(consult_btn)

        self.agent_output = QPlainTextEdit()
        self.agent_output.setReadOnly(True)
        self.agent_output.setObjectName("logBox")
        self.agent_output.setPlaceholderText("No agent activity yet")

        layout.addWidget(roster)
        layout.addLayout(input_row)
        layout.addWidget(self.agent_output, 1)
        return frame

    def _consult_agent(self) -> None:
        request = self.agent_input.text().strip()
        if not request:
            return
        self.agent_input.clear()
        agent_name = str(self.agent_combo.currentData() or "auto")
        self.agent_output.appendPlainText(f"{now_stamp()}  YOU → [{agent_name}] {request}")

        async def _run() -> None:
            result = await self.agent_manager.dispatch(request, agent_name=agent_name)
            self.agent_output.appendPlainText(f"{now_stamp()}  {result.text}")
            # Mark XX design-spec §6: the specialist's toolbelt evidence
            # trail used to be gathered and thrown away at the dispatch()
            # boundary — this is the one place in the whole app that now
            # shows it, so grounding a request in real lookups (memory, the
            # knowledge graph, live web…) is visible, not just invisible
            # extra latency.
            if result.evidence:
                self.agent_output.appendPlainText(
                    f"  Sources consulted ({len(result.evidence)}):")
                for item in result.evidence:
                    marker = "✓" if item.get("ok") else "✗"
                    self.agent_output.appendPlainText(f"    {marker} {item.get('summary', '')}")
            self.agent_output.appendPlainText("")

        background.spawn(_run())

    def _on_agent_activity(self, agent_title: str, summary: str) -> None:
        self.agent_output.appendPlainText(
            f"{now_stamp()}  ACTIVITY: {agent_title} handled '{summary}'"
        )

    # ── outlook panel ─────────────────────────────────────────────────────────

    def _build_outlook_panel(self) -> QFrame:
        frame, layout = self._panel("Outlook Command")

        self.outlook_status = QLabel(
            "Outlook COM bridge ready." if self.outlook.available
            else "Outlook unavailable — install pywin32 and configure the desktop client."
        )
        self.outlook_status.setObjectName("mutedLabel")
        self.outlook_status.setWordWrap(True)

        inbox_btn = QPushButton("Refresh Inbox")
        inbox_btn.clicked.connect(lambda: self._run_mail_action("inbox"))
        priority_btn = QPushButton("Priority Mail")
        priority_btn.clicked.connect(lambda: self._run_mail_action("priority"))
        button_row = QHBoxLayout()
        button_row.addWidget(inbox_btn)
        button_row.addWidget(priority_btn)
        button_row.addStretch(1)

        self.email_list = QListWidget()
        self.email_list.setWordWrap(True)

        drafts_heading = QLabel("Drafts Awaiting Approval")
        drafts_heading.setObjectName("panelHeading")
        self.drafts_list = QListWidget()
        self.drafts_list.setMaximumHeight(96)

        send_btn = QPushButton("Send Selected")
        send_btn.setToolTip("Transmits the selected draft after confirmation — the approval gate.")
        send_btn.clicked.connect(self._send_selected_draft)
        discard_btn = QPushButton("Discard")
        discard_btn.clicked.connect(self._discard_selected_draft)
        drafts_row = QHBoxLayout()
        drafts_row.addWidget(send_btn)
        drafts_row.addWidget(discard_btn)
        drafts_row.addStretch(1)

        layout.addWidget(self.outlook_status)
        layout.addLayout(button_row)
        layout.addWidget(self.email_list, 1)
        layout.addWidget(drafts_heading)
        layout.addWidget(self.drafts_list)
        layout.addLayout(drafts_row)
        return frame

    def _run_mail_action(self, kind: str) -> None:
        self.outlook_status.setText("Contacting Outlook…")

        async def _run() -> None:
            if kind == "priority":
                result = await self.outlook.priority_emails(limit=8)
            else:
                result = await self.outlook.read_inbox(limit=10)
            # Success populates the list via the dashboard_event feed;
            # here we surface the status line (and failures).
            self.outlook_status.setText(result.text.splitlines()[0][:140])

        background.spawn(_run())

    def _render_emails(self, emails: Any) -> None:
        self.email_list.clear()
        if not isinstance(emails, list):
            return
        for mail in emails:
            try:
                flag = "●" if mail.get("unread") else " "
                pri  = "⚑ " if mail.get("high_importance") else ""
                item = QListWidgetItem(
                    f"{flag} {pri}{mail.get('sender', '?')} — {mail.get('subject', '')}\n"
                    f"    {mail.get('preview', '')[:110]}"
                )
                self.email_list.addItem(item)
            except Exception:
                continue

    def _render_drafts(self, drafts: Any) -> None:
        self.drafts_list.clear()
        if not isinstance(drafts, list):
            return
        for draft in drafts:
            try:
                item = QListWidgetItem(
                    f"{draft['ref']}: → {draft['to']}  '{draft['subject']}'"
                )
                item.setData(Qt.ItemDataRole.UserRole, draft["ref"])
                self.drafts_list.addItem(item)
            except Exception:
                continue

    def _selected_draft_ref(self) -> str:
        item = self.drafts_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item else ""

    def _send_selected_draft(self) -> None:
        ref = self._selected_draft_ref()
        if not ref:
            self.outlook_status.setText("Select a draft to send.")
            return
        record = next((d for d in self.outlook.pending_drafts() if d["ref"] == ref), None)
        detail = f"to {record['to']} — '{record['subject']}'" if record else ref
        # The human approval gate: nothing is transmitted without this click.
        answer = QMessageBox.question(
            self,
            "Approve Transmission",
            f"Send {ref} {detail}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        async def _run() -> None:
            result = await self.outlook.send_draft(ref, confirm=True)
            self.outlook_status.setText(result.text.splitlines()[0][:140])

        background.spawn(_run())

    def _discard_selected_draft(self) -> None:
        ref = self._selected_draft_ref()
        if not ref:
            return

        async def _run() -> None:
            result = await self.outlook.discard_draft(ref)
            self.outlook_status.setText(result.text.splitlines()[0][:140])

        background.spawn(_run())

    # ── notion panel ──────────────────────────────────────────────────────────

    def _build_notion_panel(self) -> QFrame:
        frame, layout = self._panel("Notion Workspace")

        self.notion_status = QLabel(
            "Notion connected." if self.notion.available
            else "Notion not configured — add integrations.notion in config/api_keys.json."
        )
        self.notion_status.setObjectName("mutedLabel")
        self.notion_status.setWordWrap(True)

        tasks_btn = QPushButton("Refresh Tasks")
        tasks_btn.clicked.connect(lambda: self._run_notion_action("tasks"))
        agenda_btn = QPushButton("Agenda (7 Days)")
        agenda_btn.clicked.connect(lambda: self._run_notion_action("agenda"))
        button_row = QHBoxLayout()
        button_row.addWidget(tasks_btn)
        button_row.addWidget(agenda_btn)
        button_row.addStretch(1)

        self.notion_list = QListWidget()
        self.notion_list.setWordWrap(True)

        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Quick task…")
        self.task_input.returnPressed.connect(self._quick_add_task)
        self.task_due_input = QLineEdit()
        self.task_due_input.setPlaceholderText("Due (YYYY-MM-DD, optional)")
        self.task_due_input.setMaximumWidth(200)
        add_btn = QPushButton("Add Task")
        add_btn.clicked.connect(self._quick_add_task)
        add_row = QHBoxLayout()
        add_row.addWidget(self.task_input, 1)
        add_row.addWidget(self.task_due_input)
        add_row.addWidget(add_btn)

        layout.addWidget(self.notion_status)
        layout.addLayout(button_row)
        layout.addWidget(self.notion_list, 1)
        layout.addLayout(add_row)
        return frame

    def _run_notion_action(self, kind: str) -> None:
        self.notion_status.setText("Contacting Notion…")

        async def _run() -> None:
            if kind == "agenda":
                result = await self.notion.upcoming_events(days=7, limit=12)
            else:
                result = await self.notion.list_tasks(limit=12)
            self.notion_status.setText(result.text.splitlines()[0][:140])

        background.spawn(_run())

    def _quick_add_task(self) -> None:
        title = self.task_input.text().strip()
        if not title:
            return
        due = self.task_due_input.text().strip()
        self.task_input.clear()

        async def _run() -> None:
            result = await self.notion.create_task(title, due=due)
            self.notion_status.setText(result.text.splitlines()[0][:140])
            if result.ok:
                await self.notion.list_tasks(limit=12)

        background.spawn(_run())

    def _render_workspace_rows(self, rows: Any, kind: str) -> None:
        self.notion_list.clear()
        if not isinstance(rows, list):
            return
        for row in rows:
            try:
                bits = [str(row.get("title") or "(untitled)")]
                if row.get("status"):
                    bits.append(f"[{row['status']}]")
                if row.get("date"):
                    bits.append(
                        f"due {row['date']}" if kind == "tasks" else str(row["date"])
                    )
                self.notion_list.addItem(QListWidgetItem("  ".join(bits)))
            except Exception:
                continue

    # ── briefing panel ────────────────────────────────────────────────────────

    def _build_briefing_panel(self) -> QFrame:
        frame, layout = self._panel("Briefings for Today")

        run_btn = QPushButton("Run Briefing Now")
        run_btn.setToolTip("Compose and deliver the intelligence briefing by voice.")
        run_btn.clicked.connect(self._run_briefing)
        button_row = QHBoxLayout()
        button_row.addWidget(run_btn)
        button_row.addStretch(1)

        self.briefing_box = QPlainTextEdit()
        self.briefing_box.setReadOnly(True)
        self.briefing_box.setObjectName("logBox")
        self.briefing_box.setPlaceholderText(
            "No briefing yet"
        )

        layout.addLayout(button_row)
        layout.addWidget(self.briefing_box, 1)
        return frame

    def _run_briefing(self) -> None:
        self.briefing_box.setPlainText("Preparing the briefing…")

        async def _run() -> None:
            result: ToolResult = await self.dispatcher.dispatch("morning_briefing", {})
            if not result.ok:
                self.briefing_box.setPlainText(result.text)

        background.spawn(_run())

    # ── bus handlers ──────────────────────────────────────────────────────────

    def _on_dashboard_event(self, channel: str, payload: object) -> None:
        if channel == "emails":
            self._render_emails(payload)
        elif channel == "drafts":
            self._render_drafts(payload)
        elif channel == "tasks":
            self._render_workspace_rows(payload, "tasks")
        elif channel == "events":
            self._render_workspace_rows(payload, "events")
        elif channel == "briefing":
            self.briefing_box.setPlainText(str(payload))

    def _on_state(self, state: str) -> None:
        self.state_label.setText(str(state).upper())

    def _on_speaking(self, active: bool) -> None:
        self.voice_led.setText("VOICE ●" if active else "VOICE ○")
        self.voice_led.setProperty("speaking", "true" if active else "false")
        style = self.voice_led.style()
        if style is not None:
            style.unpolish(self.voice_led)
            style.polish(self.voice_led)

    # Hiding rather than destroying keeps panel state alive for the session.
    def closeEvent(self, event: Any) -> None:
        event.ignore()
        self.hide()
