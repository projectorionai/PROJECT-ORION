"""
MissionDeckView — the Command Deck evolved into a true command center.

Where the other deck pages are fixed grids, this page is a full docking
workspace (the Ultron-pass "command center, not chat app" directive): every
panel is a QDockWidget the user can drag, float, nest, resize or close, and
the arrangement persists across sessions via QSettings, so the deck becomes
*their* console rather than a layout we chose.

Fifteen panels, all pull-based (a 2 s QTimer against cheap snapshot reads;
nothing here ever touches a hot path) plus a live Activity Feed pushed from
``bus.dashboard_event``:

    MISSION CONTROL     mission board: progress, risks, recommended next move
    PROJECT EXPLORER    active projects from cognitive state
    MEMORY VIEWER       the same searchable MemoryViewerPanel OPS embeds
    RESEARCH            research agenda + evidence-store statistics
    AGENT MONITOR       the specialist roster: focus, call count, last active
    COMPONENT HEALTH    component health heartbeats
    ACTIVITY FEED       live dashboard events as they happen
    WORKFLOWS           defined workflows + run states
    KNOWLEDGE GRAPH     graph size and latest entities
    FILE LIBRARY        ingestion totals + newest files
    TASK CENTER         pending tasks with due pressure
    COMPANION           continuity ledger brief
    SECURITY CENTER     sentinel status + recent security events
    DESKTOP CONTROL     workspace snapshot (windows, active app)
    SYSTEM HEALTH       CPU / RAM / process vitals

Every service reference is optional — a missing engine renders an "offline"
line, never an error.  All refresh work is wrapped so one faulty snapshot
can never take the deck down.

Design-spec note (Mark XX, §5): AGENT MONITOR used to be the name on a dock
that actually rendered ``telemetry.health.snapshot()`` — component
heartbeats, not agents.  Its ``agents`` constructor argument was assigned to
``self.agents`` and never read again anywhere in the file.  That panel is now
COMPONENT HEALTH (same data, honest name), and AGENT MONITOR is a new panel
that actually reads ``self.agents.describe()``.

Design-spec note (Mark XX, §3, Medium item 12): MEMORY VIEWER used to be an
independent text-only renderer (tier counts + latest six records, no
search) — a third, less capable reimplementation of "browse memory"
alongside OPS's MemoryViewerPanel (which has search) and the standalone
MemoryMatrixView deck page. It now embeds the actual MemoryViewerPanel
widget, the same instance shape OPS uses, refreshed every tick like the
text panels were (re-running whatever search is currently typed).
MemoryMatrixView stays separate deliberately — a dedicated full-page
browser is a different job from a compact dock, not a fourth
reimplementation of the same one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import psutil
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import HEALTH_GLYPHS

_REFRESH_MS = 2000
_FEED_LIMIT = 120

# Which dashboard_event channels the Activity Feed narrates.
_FEED_GLYPHS = {
    "mission": "◎", "research": "◈", "research_director": "◈",
    "workflow": "⚙", "security": "⛨", "forge": "⚒", "briefing": "▤",
    "avatar": "◉", "ingestion": "▥",
}


class _TextPanel(QPlainTextEdit):
    """Read-only monospace surface every deck panel renders into."""

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setObjectName("logBox")
        self.setMaximumBlockCount(400)


class MissionDeckView(QMainWindow):
    """Dockable mission command center (a page inside the Unified deck)."""

    def __init__(
        self,
        bus: OrionBus,
        missions: Any = None,
        cognition: Any = None,
        memory: Any = None,
        research_director: Any = None,
        evidence: Any = None,
        telemetry: Any = None,
        workflow_engine: Any = None,
        graph: Any = None,
        ingestion: Any = None,
        companion: Any = None,
        security: Any = None,
        workspace: Any = None,
        agents: Any = None,
        cognitive_loop: Any = None,
    ) -> None:
        super().__init__()
        self.bus = bus
        self.missions = missions
        self.cognition = cognition
        self.memory = memory
        self.research_director = research_director
        self.evidence = evidence
        self.telemetry = telemetry
        self.workflow_engine = workflow_engine
        self.graph = graph
        self.ingestion = ingestion
        self.companion = companion
        self.security = security
        self.workspace = workspace
        self.agents = agents
        # CognitiveLoopManager (Mark XX design-spec, Critical item 2): runs
        # continuously and broadcasts a focus/project/deadlines digest on
        # bus.dashboard_event channel "awareness" — before this, nothing in
        # the whole GUI package read that channel; it only ever reached the
        # self-improvement heartbeat's own reasoning prompt. This is the
        # first screen it gets.
        self.cognitive_loop = cognitive_loop

        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks)

        self._panels: dict[str, _TextPanel] = {}
        self._renderers: dict[str, Callable[[], str]] = {}
        self._build_panels()
        self._restore_layout()

        try:
            bus.dashboard_event.connect(self._on_event)
        except Exception:
            pass

        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── construction ──────────────────────────────────────────────────────────

    def _build_panels(self) -> None:
        # Mission Control is the centre of gravity; everything else docks
        # around it (and the user re-arranges freely from there).
        centre = QWidget()
        box = QVBoxLayout(centre)
        box.setContentsMargins(8, 8, 8, 8)
        # A fixed, un-closable strip above Mission Control — unlike every
        # dock below it, this widget has no close button, so it stays true
        # to "always visible, never scrolled past" rather than being one
        # more equal-weight panel the user can tab away or dismiss.
        self._awareness_label = QLabel(self._format_awareness(None))
        self._awareness_label.setObjectName("awarenessStrip")
        self._awareness_label.setWordWrap(True)
        box.addWidget(self._awareness_label)
        heading = QLabel("MISSION CONTROL")
        heading.setObjectName("panelHeading")
        box.addWidget(heading)
        self._mission_panel = _TextPanel()
        box.addWidget(self._mission_panel, 1)
        self.setCentralWidget(centre)
        if self.cognitive_loop is not None:
            try:
                seed = self.cognitive_loop.last_digest()
                if seed:
                    self._awareness_label.setText(self._format_awareness(seed))
            except Exception:
                pass

        left = Qt.DockWidgetArea.LeftDockWidgetArea
        right = Qt.DockWidgetArea.RightDockWidgetArea
        bottom = Qt.DockWidgetArea.BottomDockWidgetArea

        docks: tuple[tuple[str, Any, Callable[[], str]], ...] = (
            ("PROJECT EXPLORER", left, self._render_projects),
            ("TASK CENTER", left, self._render_tasks),
            # Renderer callback unused for MEMORY VIEWER — it's special-cased
            # below to embed the real, searchable MemoryViewerPanel instead
            # (Mark XX design-spec §3, Medium item 12: this dock used to be
            # its own independent text-only implementation — tier counts and
            # the latest six records, no search — a third, less capable
            # reimplementation of the same "browse memory" concept OPS's
            # MemoryViewerPanel already covers with search included).
            ("MEMORY VIEWER", left, self._render_feed_placeholder),
            ("RESEARCH", right, self._render_research),
            ("AGENT MONITOR", right, self._render_agent_roster),
            ("COMPONENT HEALTH", right, self._render_component_health),
            ("WORKFLOWS", right, self._render_workflows),
            ("KNOWLEDGE GRAPH", right, self._render_graph),
            ("ACTIVITY FEED", bottom, self._render_feed_placeholder),
            ("FILE LIBRARY", bottom, self._render_library),
            ("COMPANION", bottom, self._render_companion),
            ("SECURITY CENTER", bottom, self._render_security),
            ("DESKTOP CONTROL", bottom, self._render_desktop),
            ("SYSTEM HEALTH", bottom, self._render_health),
        )
        previous_area: dict[Qt.DockWidgetArea, QDockWidget] = {}
        self._graph_widget: Any = None
        self._memory_widget: Any = None
        for name, area, renderer in docks:
            dock = QDockWidget(name, self)
            dock.setObjectName(f"dock_{name.lower().replace(' ', '_')}")
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable)
            if name == "KNOWLEDGE GRAPH" and self.graph is not None:
                # A live node-link visualiser rather than a text panel — it shows
                # the actual entities and links ORION has learned, and grows.
                from .knowledge_graph_view import KnowledgeGraphWidget
                self._graph_widget = KnowledgeGraphWidget()
                dock.setWidget(self._graph_widget)
            elif name == "MEMORY VIEWER":
                # The SAME searchable widget OPS uses, embedded here instead
                # of a second, less capable reimplementation — one canonical
                # "browse memory" component, not three (the third being the
                # standalone MemoryMatrixView deck page, which stays as-is:
                # a dedicated full-page browser is a different job from a
                # compact dock).
                from .ops_deck import MemoryViewerPanel
                self._memory_widget = MemoryViewerPanel(self.bus, self.memory)
                dock.setWidget(self._memory_widget)
            else:
                panel = _TextPanel()
                dock.setWidget(panel)
                self._panels[name] = panel
                self._renderers[name] = renderer
            self.addDockWidget(area, dock)
            # Tab the later arrivals in each area so the panels open tidy.
            if area in previous_area:
                self.tabifyDockWidget(previous_area[area], dock)
            previous_area[area] = dock

    @staticmethod
    def _format_awareness(digest: dict[str, Any] | None) -> str:
        """One line: what ORION is aware of right now.

        The digest is already prose-shaped (focus/active_project/deadlines)
        by CognitiveLoopManager, so this only needs to assemble it — no new
        summarisation logic, matching the design-spec's framing of this as
        the highest-value, lowest-cost fix in the whole redesign."""
        if digest is None:
            return "AWARENESS — building situational context…"
        parts: list[str] = []
        focus = str(digest.get("focus") or "").strip()
        if focus:
            parts.append(f"watching {focus}")
        project = str(digest.get("active_project") or "").strip()
        if project:
            parts.append(f"project: {project}")
        deadlines = digest.get("deadlines") or []
        if deadlines:
            parts.append(f"{len(deadlines)} deadline(s) in view")
        if not parts:
            return "AWARENESS — no active focus recorded yet."
        return "AWARENESS — " + " · ".join(parts)

    # ── live feed ─────────────────────────────────────────────────────────────

    def _on_event(self, channel: str, payload: Any) -> None:
        if channel == "awareness" and isinstance(payload, dict):
            # Ambient state, not a discrete event — updates its own fixed
            # strip rather than scrolling past in the Activity Feed.
            self._awareness_label.setText(self._format_awareness(payload))
            return
        panel = self._panels.get("ACTIVITY FEED")
        if panel is None:
            return
        glyph = _FEED_GLYPHS.get(str(channel or ""), "·")
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("topic") or payload.get("name")
                         or payload.get("workflow") or payload.get("status")
                         or "")[:80]
        panel.appendPlainText(
            f"{datetime.now():%H:%M:%S} {glyph} {channel}: {detail}")
        panel.verticalScrollBar().setValue(panel.verticalScrollBar().maximum())

    # ── refresh ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self.isVisible():
            return
        self._set_text(self._mission_panel, self._render_missions())
        for name, panel in self._panels.items():
            if name == "ACTIVITY FEED":
                continue           # push-driven, never overwritten
            self._set_text(panel, self._renderers[name]())
        if self._graph_widget is not None and self.graph is not None:
            try:
                self._graph_widget.set_snapshot(self.graph.graph_snapshot())
            except Exception:
                pass
        if self._memory_widget is not None:
            try:
                # Re-runs whatever search the user currently has typed (or
                # none), so new matching memories appear without the user
                # having to resubmit — the same freshness the old
                # auto-refreshed text panel had, now with search included.
                self._memory_widget.refresh()
            except Exception:
                pass

    @staticmethod
    def _set_text(panel: _TextPanel, text: str) -> None:
        if text != panel.toPlainText():
            bar = panel.verticalScrollBar()
            position = bar.value()
            panel.setPlainText(text)
            bar.setValue(position)

    # ── renderers (every one exception-proof) ─────────────────────────────────

    def _render_missions(self) -> str:
        if self.missions is None:
            return "Mission engine offline."
        try:
            snap = self.missions.panel_snapshot()
            lines = []
            for m in snap.get("missions", []):
                marker = "▶" if m["name"] == snap.get("current") else "•"
                lines.append(f"{marker} {m['name']}")
                lines.append(f"    {m['progress']:.0f}% · "
                             f"{m['open_tasks']} open · {m['status']}")
                for risk in m.get("risks", [])[:2]:
                    lines.append(f"    ⚠ {risk}")
                if m["name"] == snap.get("current") and m.get("recommendation"):
                    lines.append(f"    → {m['recommendation']}")
            return "\n".join(lines) or "No missions on the board."
        except Exception as exc:
            return f"Mission snapshot fault: {exc}"

    def _render_projects(self) -> str:
        try:
            state = self.cognition.snapshot() if self.cognition else {}
            projects = state.get("active_projects") or {}
            if not projects:
                return "No active projects recorded."
            lines = []
            for name, record in list(projects.items())[:10]:
                detail = ""
                if isinstance(record, dict):
                    detail = str(record.get("status")
                                 or record.get("note") or "")[:50]
                lines.append(f"• {str(name)[:50]}" + (f" — {detail}" if detail else ""))
            return "\n".join(lines)
        except Exception as exc:
            return f"Projects fault: {exc}"

    def _render_research(self) -> str:
        try:
            state = self.cognition.snapshot() if self.cognition else {}
            sessions = state.get("research_sessions") or {}
            lines = []
            for s in list(sessions.values())[:8]:
                if isinstance(s, dict):
                    lines.append(f"[{str(s.get('status', '?'))[:8]}] "
                                 f"{str(s.get('topic', ''))[:48]}")
            if self.evidence is not None:
                stats = self.evidence.stats()
                lines.append(f"Evidence: {stats['claims']} claim(s) across "
                             f"{stats['topics']} topic(s), "
                             f"avg confidence {stats['avg_confidence']:.0%}")
            return "\n".join(lines) or "Research agenda empty."
        except Exception as exc:
            return f"Research fault: {exc}"

    def _render_component_health(self) -> str:
        if self.telemetry is None:
            return "Telemetry offline."
        try:
            rows = self.telemetry.health.snapshot()
            return "\n".join(
                f"{HEALTH_GLYPHS.get(r['status'], '·')} {r['name'][:28]:28} {r['status']}"
                for r in rows[:18]) or "No components registered."
        except Exception as exc:
            return f"Health fault: {exc}"

    def _render_agent_roster(self) -> str:
        """The specialist agent roster — focus, call count, last active.

        Distinct from COMPONENT HEALTH: this reads AgentManager.describe(),
        the same source OPS's AgentMonitorPanel already uses, so the two
        surfaces agree instead of one silently showing telemetry instead."""
        if self.agents is None:
            return "Agent manager offline."
        try:
            rows = self.agents.describe()
            if not rows:
                return "No specialist agents registered."
            lines = []
            for info in rows[:18]:
                name = str(info.get("name", "?"))[:20]
                focus = str(info.get("focus") or "")[:44]
                calls = info.get("calls", 0)
                last_active = info.get("last_active") or ""
                activity = f"{calls} call(s), last {last_active}" if calls else "not yet consulted"
                lines.append(f"● {name:20} {focus:44} {activity}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Agent roster fault: {exc}"

    def _render_workflows(self) -> str:
        if self.workflow_engine is None:
            return "Workflow engine offline."
        try:
            names = []
            definitions = getattr(self.workflow_engine, "definitions", None)
            if isinstance(definitions, dict):
                names = list(definitions)[:12]
            elif hasattr(self.workflow_engine, "list_workflows"):
                names = list(self.workflow_engine.list_workflows())[:12]
            return ("\n".join(f"⚙ {n}" for n in names)
                    or "No workflows defined yet.")
        except Exception as exc:
            return f"Workflow fault: {exc}"

    def _render_graph(self) -> str:
        if self.graph is None:
            return "Knowledge graph offline."
        try:
            for method in ("stats", "summary"):
                if hasattr(self.graph, method):
                    result = getattr(self.graph, method)()
                    if isinstance(result, dict):
                        return "\n".join(f"{k}: {v}" for k, v in
                                         list(result.items())[:8])
                    return str(result)[:400]
            return "Graph online."
        except Exception as exc:
            return f"Graph fault: {exc}"

    @staticmethod
    def _render_feed_placeholder() -> str:
        return "Waiting for system activity…"

    def _render_library(self) -> str:
        if self.ingestion is None:
            return "Ingestion engine offline."
        try:
            for method in ("stats", "summary"):
                if hasattr(self.ingestion, method):
                    result = getattr(self.ingestion, method)()
                    if isinstance(result, dict):
                        return "\n".join(f"{k}: {v}" for k, v in
                                         list(result.items())[:8])
                    return str(result)[:400]
            return "Library online."
        except Exception as exc:
            return f"Library fault: {exc}"

    def _render_tasks(self) -> str:
        try:
            state = self.cognition.snapshot() if self.cognition else {}
            tasks = state.get("pending_tasks") or {}
            lines = []
            for task in list(tasks.values())[:10]:
                if isinstance(task, dict):
                    due = str(task.get("due") or "")[:10]
                    lines.append(f"□ {str(task.get('title', ''))[:44]}"
                                 + (f"  (due {due})" if due else ""))
            return "\n".join(lines) or "Task queue clear."
        except Exception as exc:
            return f"Tasks fault: {exc}"

    def _render_companion(self) -> str:
        if self.companion is None:
            return "Companion offline."
        try:
            brief = self.companion.continuity_brief()
            return str(brief)[:600] or "No continuity data yet."
        except Exception as exc:
            return f"Companion fault: {exc}"

    def _render_security(self) -> str:
        if self.security is None:
            return "Security sentinel offline."
        try:
            for method in ("summary", "status_report", "snapshot"):
                if hasattr(self.security, method):
                    result = getattr(self.security, method)()
                    if isinstance(result, dict):
                        return "\n".join(f"{k}: {v}" for k, v in
                                         list(result.items())[:8])
                    return str(result)[:500]
            return "Sentinel online — no interface exposed."
        except Exception as exc:
            return f"Security fault: {exc}"

    def _render_desktop(self) -> str:
        if self.workspace is None:
            return "Workspace observer offline."
        try:
            snap = self.workspace.last_snapshot()
            if snap is None:
                return "No workspace snapshot yet."
            return (f"{snap.summary()}\n"
                    f"Active: {str(snap.active_window)[:70]}")
        except Exception as exc:
            return f"Workspace fault: {exc}"

    def _render_health(self) -> str:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            lines = [f"CPU {cpu:5.1f}%", f"RAM {mem.percent:5.1f}% "
                     f"({mem.used / 1_073_741_824:.1f} GB used)"]
            if self.telemetry is not None:
                counters = self.telemetry.metrics.snapshot().get("counters", {})
                lines.append(f"Tool calls: {int(counters.get('tool.calls', 0))} "
                             f"(failures {int(counters.get('tool.failures', 0))})")
            return "\n".join(lines)
        except Exception as exc:
            return f"Health fault: {exc}"

    # ── layout persistence ────────────────────────────────────────────────────

    def _settings(self) -> QSettings:
        return QSettings("ORION", "MissionDeck")

    def _restore_layout(self) -> None:
        try:
            state = self._settings().value("layout")
            if state is not None:
                self.restoreState(state)
        except Exception:
            pass

    def save_layout(self) -> None:
        try:
            self._settings().setValue("layout", self.saveState())
        except Exception:
            pass

    def hideEvent(self, event: Any) -> None:  # noqa: N802 (Qt override)
        self.save_layout()
        super().hideEvent(event)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 (Qt override)
        self.save_layout()
        event.ignore()
        self.hide()


__all__ = ["MissionDeckView"]
