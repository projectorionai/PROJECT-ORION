"""
UnifiedDashboard — the widget dashboard, command centre, globe and e-commerce
hub merged into one swipeable window.

Pages are switched three ways, so it is genuinely "swipe-able":
    • a segmented tab bar at the top (click a name);
    • left/right arrow keys, or Ctrl+Tab;
    • a horizontal mouse drag (swipe) across the page area.

Each page transition fades the incoming page in for visual continuity.  The
existing QMainWindow panels are embedded directly as pages, so no panel logic
is duplicated — this window is purely a swipeable container over them.
"""

from __future__ import annotations

import re
import weakref

from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import SPACE
from .widgets import describe_control
from .labels import PAGE_DISPLAY_NAMES, page_display_name
from .line_icons import apply as apply_icon, icon as line_icon

# Distinct from any real zone name (or the None used for an orphan page) so
# the very first _select() call always rebuilds the page row regardless of
# which zone page 0 happens to belong to.
_UNSET_ZONE = object()


def _deck_transitions_enabled() -> bool:
    """False under the Reduced motion preference: pages switch instantly."""
    try:
        from ..appearance import motion_profile
        return bool(motion_profile().deck_transitions)
    except Exception:
        return True


class UnifiedDashboard(QMainWindow):
    SWIPE_THRESHOLD = 90  # px of horizontal drag to change page

    def __init__(self, bus: OrionBus, pages: list[tuple[str, Any]]) -> None:
        """*pages* is ``(name, widget)``, or ``(name, factory)`` to build it
        on first use.

        A factory is any zero-argument callable that is not already a QWidget.
        Deferring one costs nothing until the page is opened, and the heavy
        pages are heavy indeed: a WebEngine surface brings up Chromium
        subprocesses and a GL context whether or not anybody looks at it.
        ORION's code is 82 MB; the rest of his ~750 MB is runtime like this.

        ``_pages`` stays ``(name, widget)`` throughout, so every existing
        reader keeps working — an unbuilt page simply holds a placeholder
        until it is opened, and the real widget replaces it in place.
        """
        super().__init__()
        self.bus = bus
        self.setAcceptDrops(True)
        self._factories: dict[str, Any] = {}
        resolved: list[tuple[str, QWidget]] = []
        for name, page in pages:
            if isinstance(page, QWidget):
                self._fit_page_margins(page)
                resolved.append((name, page))
                continue
            if callable(page):
                self._factories[name] = page
                resolved.append((name, self._placeholder(name)))
                continue
            resolved.append((name, self._placeholder(name)))
        self._pages = resolved
        self._current_zone: Any = _UNSET_ZONE
        # The name without the Mark: the deck shows no Mark at all (owner,
        # 28 Oct 2026); APP_NAME carries one.
        self.setWindowTitle("O.R.I.O.N — Command Deck")
        self.setMinimumSize(1040, 700)
        self.resize(1280, 820)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Operational status strip (Mark XXVI §14): one unobtrusive line telling
        # the user what ORION is doing right now — mode + voice state — driven by
        # the always-present bus.state signal. Enriched (focus/mission) later.
        # It sits INSIDE the header row now (built there), not on a row of its
        # own: one compact strip for status, title and search.
        from .status_strip import StatusStrip
        self.status_strip = StatusStrip(embedded=True)
        layout.addWidget(self._build_header())
        self._op_state = "STANDBY"
        self._status_provider: Any = None
        try:
            self.bus.state.connect(self._on_op_state)
        except Exception:
            pass
        self._on_op_state(self._op_state)

        self.stack = QStackedWidget()
        # self._pages, not the argument: a deferred page is a placeholder here
        # until it is opened, and the argument still holds its factory.
        for _name, widget in self._pages:
            self.stack.addWidget(widget)
        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._sidebar = self._build_sidebar()
        self._content_splitter.addWidget(self._sidebar)
        self._content_splitter.addWidget(self.stack)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes([self.SIDEBAR_WIDTH, 1000])
        # A full step of gutter between the sidebar and the page, and round
        # the outside: with SPACE.SM every page sat 8 px from the sidebar and
        # the window's edge, so panels read as touching both.
        self._content_splitter.setHandleWidth(SPACE.MD)
        body = QWidget()
        body_box = QVBoxLayout(body)
        body_box.setContentsMargins(SPACE.MD, SPACE.MD, SPACE.MD, SPACE.MD)
        body_box.addWidget(self._content_splitter)
        layout.addWidget(body, 1)

        self.setCentralWidget(root)
        self._drag_x: float | None = None
        self._anim: QPropertyAnimation | None = None
        # Pages visited, for Back/Forward (Alt+Left/Right, the mouse's side
        # buttons). Indices into _pages; bounded like a browser's.
        self._back: list[int] = []
        self._forward: list[int] = []
        self._select(0)

        # Swipe detection over the page area.
        self.stack.installEventFilter(self)
        shortcuts: list[tuple[str, Any]] = [
            ("Ctrl+K", self._open_search),
            ("Ctrl+Shift+E", lambda: self.show_page_named("WORKBENCH")),
            # Window-level actions rather than keyPressEvent: Ctrl+Tab is
            # otherwise taken by focus traversal inside a page and never
            # reaches the window, which is why it only worked sometimes.
            ("Ctrl+Tab", self.next_page), ("Ctrl+Shift+Tab", self.prev_page),
            ("Ctrl+PgDown", self.next_page), ("Ctrl+PgUp", self.prev_page),
            ("Alt+Left", self.go_back), ("Alt+Right", self.go_forward),
        ]
        for number in range(1, 10):
            shortcuts.append((f"Ctrl+{number}",
                              lambda _c=False, n=number: self._select_nav_position(n - 1)))
        for shortcut, callback in shortcuts:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self.addAction(action)
        # The sidebar's own: icons-only rail, and "go to page" in its filter.
        for sequence, callback in (
                (QKeySequence("Ctrl+B"), lambda: self.set_sidebar_rail(not self.sidebar_rail)),
                (QKeySequence("Ctrl+G"), self.focus_page_filter)):
            action = QAction(self)
            action.setShortcut(sequence)
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def attach_status_provider(self, provider: Any) -> None:
        """Supply a callable(state) -> OpStatus so the strip can show the full
        picture (network, focus block, mission), not just the voice state.

        Optional by design: with no provider the strip still reports mode and
        voice from the bus alone, so the deck never depends on the engines.
        """
        self._status_provider = provider
        self._on_op_state(self._op_state)

    def _on_op_state(self, state: Any) -> None:
        """Reflect ORION's live pipeline state in the operational status strip."""
        from ..operational_status import compose
        self._op_state = str(state or "")
        provider = getattr(self, "_status_provider", None)
        if provider is not None:
            try:
                self.status_strip.set_status(provider(self._op_state))
                return
            except Exception:
                pass        # a faulting provider must not blank the strip
        self.status_strip.set_status(compose(state=self._op_state))

    #: What a page is CALLED on screen, where that differs from its key. The
    #: keys stay stable (voice navigation, tests and the swarm all use them);
    #: only the label a person reads changes.
    _DISPLAY_NAMES = PAGE_DISPLAY_NAMES

    @classmethod
    def display_name(cls, name: str) -> str:
        return page_display_name(name)

    def page_names(self) -> list[str]:
        """Every deck page's label, in tab order — the source of truth the
        Command Palette indexes (Mark XX design-spec §8) instead of a
        hardcoded copy that could drift from the real page list."""
        return [name for name, _widget in self._pages]

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        reference = getattr(self, "_core_window_ref", None)
        core = reference() if callable(reference) else None
        handler = getattr(core, "_hud_files_dropped", None)
        if paths and callable(handler):
            handler(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ── header (zone row + page tabs + nav) ────────────────────────────────────
    #
    # Mark XX design-spec §4: the old header was one flat row of thirteen
    # co-equal page buttons in alphabetically-arbitrary order — a textbook
    # Hick's Law cost (decision time scales with the log of undifferentiated
    # choices), and the reason "the ROAS calculator" or "the chess thing"
    # had no findable home once a user forgot which of the thirteen tabs it
    # lived under. This replaces it with two rows: twelve intent-based ZONES
    # (Intelligence / Research / Development / Business / Creative /
    # Automation / Operations / Communication / Memory / Monitoring /
    # Security / System) on top, and the active zone's pages below. Clicking
    # a page still works exactly as before; the zone row is purely a
    # coarser first choice that narrows which page row you're looking at.
    #
    # Audit pass (this session): the taxonomy grew from nine to twelve
    # zones — RESEARCH split out of INTELLIGENCE (it's the flagship
    # knowledge-work surface, not a sub-item), MEMORY split out of the old
    # KNOWLEDGE zone (LIBRARY moved in with RESEARCH instead — a document
    # library is research material), COMMAND CENTRE moved from OPERATIONS
    # into a new SYSTEM zone (it's a live system-metrics dashboard, not an
    # operational workflow tool — OPS keeps the companion/agents/tasks/
    # projects panels that actually ARE operational), and SECURITY was
    # added as a genuinely empty zone: security_recon.py is real, tested,
    # tool-callable backend, but has no GUI page yet, so the honest move is
    # to name the zone and leave it disabled rather than either hiding the
    # gap or building a placeholder page to fill it.
    #
    # ZONE_PAGES is a best-fit, page-granularity placement — several pages
    # (WIDGETS, TOOLKIT, OPS) actually span multiple zones' worth of panels
    # per the design-spec's own page-verdict table (§3); a page only moves
    # zones for real once its panels are individually redistributed, tracked
    # separately. Any zone listed with an empty tuple genuinely has no page
    # yet — an honest preview of the full taxonomy, not a claim the
    # migration is finished.
    #
    # CHESS is deliberately absent from every zone (§3's verdict: it
    # shouldn't compete for permanent top-level real estate). It keeps its
    # page and every byte of its functionality — reachable via the Command
    # Palette (§8, already indexes every page) and via the existing
    # chevron/swipe/keyboard navigation, which still cycle the full page
    # list exactly as before. An "orphan" page like this gets its own
    # single-entry row when it becomes active, so it's never simply
    # invisible — see _rebuild_page_row.

    ZONE_ORDER = (
        "INTELLIGENCE", "RESEARCH", "DEVELOPMENT", "BUSINESS", "CREATIVE",
        "AUTOMATION", "OPERATIONS", "COMMUNICATION", "MEMORY", "MONITORING",
        "SECURITY", "SYSTEM",
    )
    # Mark XX design-spec §6: each specialist agent's workspace lands in the
    # zone its domain already best fits (Coding -> Development, Marketing ->
    # Business, Research -> Research; Design/Fashion/Entertainment all
    # share Creative as the "taste and aesthetic judgement" cluster) rather
    # than getting its own top-level zone.
    ZONE_PAGES: dict[str, tuple[str, ...]] = {
        # NEURAL (Mark XXXX.4): his stats as graphs, beside the BRAIN page
        # that draws where his thoughts go.
        "INTELLIGENCE": ("BRAIN", "NEURAL", "MISSION", "GLOBE"),
        "RESEARCH": ("RESEARCH", "LIBRARY"),
        "DEVELOPMENT": ("WORKBENCH", "DEVELOPMENT"),
        # Mark XXXVIII: the Toolkit page is the old Toolkit and Cognition
        # pages in one (gui/deck_tabs.py) - its key stayed COGNITION.
        "BUSINESS": ("COGNITION",),
        # Mark XXXI: STUDIO, DESIGN, FASHION and ENTERTAINMENT were retired
        # (a status log and three chat personas). MARKETING — the owner's
        # creator workspace — is the creative work that remains.
        "CREATIVE": ("MARKETING",),
        # HOME (Mark XXXVII.5): the smart home - lights, scenes, routines.
        "AUTOMATION": ("AUTOMATION", "HOME"),
        # TASKS: every plan, workflow and background job in one list, with
        # approvals, pause/resume/cancel and honest progress (job_store).
        # CALENDAR, FOCUS and FITNESS are the owner's own day.
        "OPERATIONS": ("OPS", "TASKS", "CALENDAR", "FOCUS", "FITNESS"),
        "COMMUNICATION": ("WIDGETS",),
        "MEMORY": ("MEMORY",),
        "MONITORING": ("DIAGNOSTICS", "LOG"),
        # No longer empty: SecurityCentreView (gui/security_centre.py) is the
        # zone's first real page — live posture from SecuritySentinel, a
        # monitoring on/off toggle, and the alert history that previously only
        # ever flashed past on the HUD banner. An empty tuple here renders the
        # zone disabled, which is exactly what "the security tab is unclickable
        # and you can't do anything in it" was describing.
        "SECURITY": ("SECURITY",),
        # CHESS lives here rather than nowhere.
        #
        # It used to belong to no zone at all — the reasoning being that it
        # should not compete for permanent top-level real estate, with the
        # Command Palette and the chevrons left to reach it. That held while
        # the deck still had tab bars. Once swarm navigation retired them
        # (ORION_SWARM_NAV), zone routing became the ONLY way in, and a page
        # in no zone became a page with no route: "I cannot go on the chess
        # page at all and it's having a hard time navigating to it".
        #
        # Listed last in SYSTEM, so it is reachable without displacing the
        # Command Centre as what the SYSTEM zone opens on.
        "SYSTEM": ("COMMAND CENTRE", "PLUGINS", "CHESS"),
    }

    # Concise glyph per page so the tab bar is scannable at a glance.
    _PAGE_GLYPHS = {
        "BRAIN": "✵", "MISSION": "◎", "WIDGET": "⧉", "TOOLKIT": "⚒",
        "LIBRARY": "▤", "OPS": "⬢", "COMMAND": "◉", "DIAGNOSTIC": "◈",
        "GLOBE": "◍", "LOG": "▤", "MEMORY": "◈", "TELEMETRY": "◈", "CHESS": "♞",
        "WORKBENCH": "⌖", "DEVELOPMENT": "⌨", "AUTOMATION": "⚙", "TASKS": "☰",
        "MARKETING": "▲", "RESEARCH": "⌕",
        "SECURITY": "⛨", "PLUGIN": "⧩", "NEURAL": "⌬",
    }

    def _fit_zone_buttons(self) -> None:
        """Size each zone button to the text it actually renders.

        Deferred until the widget is shown because the app stylesheet — which
        sets the font and padding for #segItem — is applied after
        construction. Measuring at build time used the default font, produced
        buttons ~30% too narrow, and Qt then clipped the labels from both ends
        ("Intelligence" -> "elligen"). Re-fitting here uses the real metrics.
        """
        for zone, btn in getattr(self, "_zone_buttons", {}).items():
            metrics = btn.fontMetrics()
            # Padding allowance covers the QSS horizontal padding plus the
            # checked-state border; generous on purpose, since a slightly wide
            # button is invisible while a narrow one shreds its label. Now that
            # the zones own a full-width row, a scroll bar (not clipping)
            # handles the case where even these minimums exceed the width.
            btn.setMinimumWidth(metrics.horizontalAdvance(btn.text()) + 22)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        # Reopened before the release timer fired, so keep the renderers we
        # still have rather than throwing them away and rebuilding.
        timer = getattr(self, "_release_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        if not getattr(self, "_zones_fitted", False):
            self._zones_fitted = True
            self._fit_zone_buttons()
            # Start with the keyboard on the page, not on whichever header
            # control happens to be first in the tab chain: that drew a
            # crimson focus ring round "Find a page" every time the deck
            # opened, the loudest thing on screen for no reason. Tab still
            # reaches every control; Ctrl+K still searches.
            focused = self.focusWidget()
            if focused is None or focused is getattr(self, "_search_btn", None):
                self.stack.setFocus(Qt.FocusReason.OtherFocusReason)
        # The deck is a separate top-level window, so it needs to ask the
        # compositor for its own backdrop — the Core Window's does not extend
        # to it. Cheap, once, and a no-op anywhere that cannot blur.
        if not getattr(self, "_backdrop_asked", False):
            self._backdrop_asked = True
            try:
                from . import depth

                depth.apply_backdrop(self)
            except Exception:
                pass

    def _zone_for_page(self, page_name: str) -> str | None:
        for zone, pages in self.ZONE_PAGES.items():
            if page_name in pages:
                return zone
        return None

    # ── sidebar ──────────────────────────────────────────────────────────────
    #
    # The twelve zones stay the deck's taxonomy (voice, the palette and the
    # Overview's clusters all speak it), but as a SIDEBAR they were eleven
    # headings over twenty-two pages, six of them guarding a single page each
    # (Business > Toolkit, Creative > Marketing ...), and the list scrolled at
    # every window size. The sidebar shows a few broad sections, each one
    # collapsible, with a live filter on top and an icon-rail mode (Ctrl+B)
    # that hands the page almost the whole width.
    #
    # Mark XXXV refinement: the five sections (Command / Explore / Create /
    # Personal / System) still put all twenty-two pages at equal weight, and a
    # RECENT block above them repeated three of those same pages a second
    # time. The sections now follow what a person comes to the deck FOR —
    # today's work, the work itself, what ORION knows, and the machinery —
    # and the pages used least sit under More, folded away until opened. The
    # page on screen is always shown, even inside a folded section, and every
    # page is still one Ctrl+G / Ctrl+K away.

    SIDEBAR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        # GOALS is the goals lane's page; listed here so it lands in Today the
        # moment it is registered (an absent page is simply skipped).
        # One word each: Qt reads "&" in a button label as a mnemonic marker
        # ("WORLD & KNOWLEDGE" rendered as "WORLD _OWLEDGE").
        ("TODAY", ("BRAIN", "CALENDAR", "FOCUS", "FITNESS", "GOALS", "TASKS", "HOME",
                   "NEURAL", "MISSION", "COGNITION")),
        ("WORK", ("RESEARCH", "DEVELOPMENT", "AUTOMATION", "WORKBENCH", "MARKETING")),
        ("KNOWLEDGE", ("MEMORY", "LIBRARY", "GLOBE")),
        ("SYSTEM", ("COMMAND CENTRE", "DIAGNOSTICS", "SECURITY", "PLUGINS")),
        ("MORE", ("OPS", "WIDGETS", "CHESS", "LOG")),
    )
    #: Sections folded until the user opens them (and remembered after that).
    SIDEBAR_FOLDED_BY_DEFAULT = frozenset({"MORE"})
    #: How many recent pages the Workspaces menu lists (the current page excluded).
    SIDEBAR_RECENT = 5
    SIDEBAR_WIDTH = 200
    RAIL_WIDTH = 64

    def _group_for_page(self, page_name: str) -> str | None:
        for group, names in self.SIDEBAR_GROUPS:
            if page_name in names:
                return group
        return None

    _PAGE_ICONS = {
        "BRAIN": "workflow", "MISSION": "flag", "WIDGET": "grid", "TOOLKIT": "sliders",
        "LIBRARY": "book", "OPS": "briefcase", "COMMAND": "gauge", "DIAGNOSTIC": "activity",
        "GLOBE": "globe", "LOG": "terminal", "MEMORY": "database", "TELEMETRY": "chart",
        "CHESS": "pawn", "WORKBENCH": "camera", "DEVELOPMENT": "code", "AUTOMATION": "workflow",
        "TASKS": "checklist", "MARKETING": "trend", "RESEARCH": "search", "SECURITY": "shield",
        "PLUGIN": "plug", "HOME": "bulb", "NEURAL": "chart",
        "CALENDAR": "calendar", "FOCUS": "target", "FITNESS": "dumbbell",
    }

    def _page_icon(self, name: str) -> str:
        return next((icon for key, icon in self._PAGE_ICONS.items() if key in name.upper()), "document")

    def _glyph(self, name: str) -> str:
        upper = name.upper()
        return next((g for key, g in self._PAGE_GLYPHS.items() if key in upper), "•")

    def _build_sidebar(self) -> QScrollArea:
        """Persistent page targets beside the content: pinned pages, then
        every page under its section (recent pages are in the Workspaces
        menu and on Back/Forward, rather than listed a second time here)."""
        scroll = QScrollArea()
        scroll.setObjectName("zoneScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QFrame()
        inner.setObjectName("panelFrame")
        column = QVBoxLayout(inner)
        column.setContentsMargins(SPACE.SM, SPACE.SM + SPACE.XS, SPACE.SM, SPACE.SM + SPACE.XS)
        column.setSpacing(2)
        self._sidebar_buttons: dict[str, QPushButton] = {}
        self._sidebar_groups: dict[str, tuple[QToolButton, list[str]]] = {}
        # The order a person SEES the pages in. Next/previous follow it; they
        # used to follow construction order, so the arrows jumped between
        # sections in an order that matched nothing on screen.
        self._nav_order: list[str] = []
        from .deck_favourites import DeckFavourites
        self._favourites = DeckFavourites(write_in_background=True)
        self._favourites.restrict_to(name for name, _ in self._pages)

        # Rail toggle + filter.
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, SPACE.XS)
        top.setSpacing(SPACE.SM)
        self._rail_btn = describe_control(QPushButton(), "Collapse the sidebar to icons (Ctrl+B)")
        self._rail_btn.setObjectName("iconButton")
        apply_icon(self._rail_btn, "sidebar-collapse")
        self._rail_btn.clicked.connect(lambda: self.set_sidebar_rail(not self.sidebar_rail))
        top.addWidget(self._rail_btn)
        self._filter = QLineEdit()
        self._filter.setObjectName("sidebarFilter")
        self._filter.setPlaceholderText("Go to page…  Ctrl+G")
        self._filter.setAccessibleName("Filter the deck's pages")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        self._filter.returnPressed.connect(self._open_first_match)
        self._filter.installEventFilter(self)
        top.addWidget(self._filter, 1)
        column.addLayout(top)

        self._shortcut_sections: dict[str, tuple[QLabel, QVBoxLayout]] = {}
        for section in ("PINNED",):
            heading = QLabel(section.title())
            heading.setObjectName("sidebarCaption")
            box = QVBoxLayout()
            box.setSpacing(1)
            column.addWidget(heading)
            column.addLayout(box)
            self._shortcut_sections[section] = (heading, box)

        present = [name for name, _ in self._pages]
        groups = [(group, [n for n in names if n in present]) for group, names in self.SIDEBAR_GROUPS]
        # A page no section names (a page added later, say) is never lost:
        # it joins More, the last section, rather than a second More.
        leftovers = [n for n in present if not any(n in names for _, names in groups)]
        if leftovers:
            groups = [(g, names + leftovers if g == "MORE" else names) for g, names in groups]
            if "MORE" not in {g for g, _ in groups}:
                groups.append(("MORE", leftovers))
        for group, names in groups:
            if not names:
                continue
            header = QToolButton()
            header.setObjectName("sidebarGroup")
            header.setText(group.title())
            header.setCheckable(True)
            header.setChecked(not self._favourites.is_collapsed(
                group, default=group in self.SIDEBAR_FOLDED_BY_DEFAULT))
            header.setArrowType(Qt.ArrowType.DownArrow if header.isChecked() else Qt.ArrowType.RightArrow)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setCursor(Qt.CursorShape.PointingHandCursor)
            header.setAccessibleName(f"{group.title()} section")
            header.setToolTip(f"{group.title()}: " + ", ".join(self.display_name(n) for n in names))
            header.toggled.connect(lambda on, g=group: self._toggle_group(g, on))
            column.addWidget(header)
            self._sidebar_groups[group] = (header, names)
            for name in names:
                self._sidebar_page_button(column, name)
                self._sidebar_buttons[name].setVisible(header.isChecked())
        column.addStretch(1)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(self.RAIL_WIDTH)
        self._refresh_shortcuts()
        self._rail = False
        if self._favourites.rail:
            # The dashboard can be destroyed before the next event-loop turn.
            # Parenting the timer cancels that pending restore with its owner.
            self._rail_restore_timer = QTimer(self)
            self._rail_restore_timer.setSingleShot(True)
            self._rail_restore_timer.timeout.connect(self._restore_sidebar_rail)
            self._rail_restore_timer.start(0)
        return scroll

    def _restore_sidebar_rail(self) -> None:
        self.set_sidebar_rail(True)

    def _toggle_group(self, group: str, expanded: bool) -> None:
        header, names = self._sidebar_groups.get(group, (None, []))
        if header is None:
            return
        header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if not self._filter.text().strip():
            current = self._current_page_name()
            for name in names:
                self._sidebar_buttons[name].setVisible(
                    expanded or self.sidebar_rail or name == current)
        self._favourites.set_collapsed(group, not expanded)

    def _current_page_name(self) -> str | None:
        stack = getattr(self, "stack", None)
        if stack is None or not stack.count():
            return None
        index = stack.currentIndex()
        return self._pages[index][0] if 0 <= index < len(self._pages) else None

    def _filter_matches(self, text: str) -> list[str]:
        """Pages whose name, label, section or zone contains every typed word,
        in the order the sidebar shows them."""
        words = [w for w in text.lower().split() if w]
        if not words:
            return list(self._nav_order)
        found = []
        for name in self._nav_order:
            haystack = " ".join((name, self.display_name(name), self._group_for_page(name) or "",
                                 self._zone_for_page(name) or "")).lower()
            if all(word in haystack for word in words):
                found.append(name)
        return found

    def _apply_filter(self, text: str) -> None:
        typed = bool(text.strip())
        matches = set(self._filter_matches(text))
        current = self._current_page_name()
        for group, (header, names) in self._sidebar_groups.items():
            shown = [n for n in names if n in matches]
            header.setVisible(bool(shown) and not self.sidebar_rail)
            for name in names:
                # The page on screen stays listed even inside a folded section,
                # so the sidebar always shows where you are.
                visible = (name in matches) if typed else (
                    header.isChecked() or self.sidebar_rail or name == current)
                self._sidebar_buttons[name].setVisible(visible)
        for heading, box in self._shortcut_sections.values():
            heading.setVisible(not typed and not self.sidebar_rail and box.count() > 0)
            for i in range(box.count()):
                widget = box.itemAt(i).widget()
                if widget is not None:
                    widget.setVisible(not typed)

    def _open_first_match(self) -> None:
        matches = self._filter_matches(self._filter.text())
        if matches:
            self.show_page_named(matches[0])
            self._filter.clear()

    def focus_page_filter(self) -> None:
        """Ctrl+G: jump to the sidebar's page filter (opening the rail)."""
        if self.sidebar_rail:
            self.set_sidebar_rail(False)
        self._filter.setFocus()
        self._filter.selectAll()

    @property
    def sidebar_rail(self) -> bool:
        return bool(getattr(self, "_rail", False))

    def set_sidebar_rail(self, on: bool) -> None:
        """Icons only (on) or icons and names (off); remembered."""
        self._rail = bool(on)
        self._filter.setVisible(not on)
        apply_icon(self._rail_btn, "sidebar-expand" if on else "sidebar-collapse")
        self._rail_btn.setToolTip("Show page names (Ctrl+B)" if on
                                  else "Collapse the sidebar to icons (Ctrl+B)")
        for name, button in self._sidebar_buttons.items():
            button.setText("" if on else self.display_name(name))
            apply_icon(button, self._page_icon(name))
        if on:
            self._filter.clear()
        self._apply_filter("")
        width = self.RAIL_WIDTH if on else self.SIDEBAR_WIDTH
        try:
            sizes = self._content_splitter.sizes()
            total = sum(sizes) or (width + 1000)
            self._content_splitter.setSizes([width, max(200, total - width)])
            self._sidebar.setMaximumWidth(self.RAIL_WIDTH + 8 if on else 16777215)
        except Exception:
            pass
        self._favourites.set_rail(self._rail)

    def recent_pages(self) -> list[str]:
        """The last few distinct pages opened, the current one left out."""
        favourites = getattr(self, "_favourites", None)
        if favourites is None:
            return []
        current = self._current_page_name()
        return [n for n in favourites.recent if n != current][:self.SIDEBAR_RECENT]

    def _refresh_shortcuts(self) -> None:
        """Rebuild the sidebar's pinned section from the stored list."""
        sections = getattr(self, "_shortcut_sections", None)
        if not sections:
            return
        favourites = self._favourites
        current = self._current_page_name()
        recent = [n for n in favourites.recent if n not in favourites.pinned and n != current]
        wanted = {"PINNED": list(favourites.pinned), "RECENT": recent[:self.SIDEBAR_RECENT]}
        rail = self.sidebar_rail
        for section, (heading, box) in sections.items():
            while box.count():
                item = box.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    # Hidden now, deleted later: deleteLater waits for the
                    # event loop, and until then the old row drew underneath
                    # the new one ("Mission" printed over "Chess").
                    widget.hide()
                    widget.setParent(None)
                    widget.deleteLater()
            names = wanted[section]
            heading.setVisible(bool(names) and not rail)
            for name in names:
                button = QPushButton(self.display_name(name))
                apply_icon(button, self._page_icon(name))
                button.setObjectName("sidebarItem")
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                label = "pinned" if section == "PINNED" else "recently opened"
                button.setAccessibleName(f"Open {self.display_name(name)} ({label})")
                button.setToolTip(f"{self.display_name(name)} ({label})")
                button.clicked.connect(
                    lambda _c=False, page=name: self.show_page_named(page))
                self._attach_pin_menu(button, name)
                button.setVisible(not rail)
                box.addWidget(button)

    def _attach_pin_menu(self, button: QPushButton, name: str) -> None:
        """Right-click (or the keyboard's menu key) pins or unpins a page."""
        button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        button.customContextMenuRequested.connect(
            lambda pos, b=button, page=name: self._pin_menu(b, page, pos))

    def _pin_menu(self, button: QPushButton, name: str, pos: Any) -> None:
        menu = QMenu(button)
        pinned = self._favourites.is_pinned(name)
        action = menu.addAction("Unpin from the Top" if pinned else "Pin to the Top")
        action.triggered.connect(lambda _c=False, page=name: self.toggle_pin(page))
        menu.exec(button.mapToGlobal(pos))

    def toggle_pin(self, name: str) -> bool:
        """Pin or unpin a page (also reachable from tests and voice)."""
        resolved = self.resolve_page_name(name) or name
        pinned = self._favourites.toggle_pin(resolved)
        self._refresh_shortcuts()
        return pinned

    def _sidebar_page_button(self, column: QVBoxLayout, name: str) -> None:
        button = QPushButton(self.display_name(name))
        apply_icon(button, self._page_icon(name))
        button.setObjectName("sidebarItem")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName(f"Open {self.display_name(name)}")
        self._nav_order.append(name)
        position = len(self._nav_order)
        button.setToolTip(f"{self.display_name(name)}   Ctrl+{position}" if position <= 9
                          else self.display_name(name))
        button.clicked.connect(lambda _checked=False, page=name: self.show_page_named(page))
        self._sidebar_buttons[name] = button
        self._attach_pin_menu(button, name)
        column.addWidget(button)

    def _build_header(self) -> QFrame:
        # ONE compact row: where you are (section › page), ORION's status, and
        # search. It was three things on two rows — a Workspaces button that
        # repeated the sidebar, the title, and a status rail on a row of its
        # own carrying two words.
        frame = QFrame()
        frame.setObjectName("headerFrame")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(SPACE.MD, SPACE.SM, SPACE.MD, SPACE.SM)
        outer.setSpacing(SPACE.XS)

        top = QHBoxLayout()
        top.setSpacing(SPACE.SM)
        # Every workspace by zone, the recent pages and the legacy tab bars:
        # kept whole, behind one quiet icon at the end of the row.
        self._workspace_btn = QToolButton()
        self._workspace_btn.setObjectName("workspaceMenu")
        self._workspace_btn.setAccessibleName("Browse all workspaces")
        self._workspace_btn.setToolTip("All workspaces and recent pages")
        self._workspace_btn.setIcon(line_icon("grid", owner=self._workspace_btn))
        self._workspace_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._workspace_menu = QMenu(self._workspace_btn)
        self._recent_menu = self._workspace_menu.addMenu("Recent Pages")
        self._recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self._workspace_menu.addSeparator()
        self._page_actions = {}
        assigned = set()
        for zone in self.ZONE_ORDER:
            names = [name for name, _ in self._pages
                     if name in self.ZONE_PAGES.get(zone, ())]
            if not names:
                continue
            group = self._workspace_menu.addMenu(zone.title())
            for name in names:
                action = group.addAction(self.display_name(name))
                action.setCheckable(True)
                action.triggered.connect(lambda _checked=False, n=name: self.show_page_named(n))
                self._page_actions[name] = action
                assigned.add(name)
        for name, _ in self._pages:
            if name not in assigned:
                action = self._workspace_menu.addAction(self.display_name(name))
                action.setCheckable(True)
                action.triggered.connect(lambda _checked=False, n=name: self.show_page_named(n))
                self._page_actions[name] = action
        self._workspace_menu.addSeparator()
        self._expanded_navigation = self._workspace_menu.addAction("Show Navigation Tabs")
        self._expanded_navigation.setCheckable(True)
        self._expanded_navigation.toggled.connect(self._set_expanded_navigation)
        self._workspace_btn.setMenu(self._workspace_menu)

        # No mark tag (owner, 28 Oct 2026): the deck shows no Mark at all -
        # the core window's header carries it. The row is where you are
        # (back, forward, section › page) on the left, ORION's status in the
        # middle, and search on the right; the two outer zones take equal
        # shares so the status sits in the true middle of the bar.
        # The section is a signpost (quiet); the page is the title (the one
        # title on screen — pages no longer repeat it underneath).
        self._crumb_label = QLabel()
        self._crumb_label.setObjectName("crumbLabel")
        self._current_page_label = QLabel()
        self._current_page_label.setObjectName("pageTitle")
        self._current_page_label.setAccessibleName("Current page")
        where = QHBoxLayout()
        where.setSpacing(SPACE.SM)
        where.addSpacing(SPACE.SM)
        where.addWidget(self._crumb_label, 0, Qt.AlignmentFlag.AlignVCenter)
        where.addWidget(self._current_page_label, 0, Qt.AlignmentFlag.AlignVCenter)
        where.addStretch(1)
        top.addLayout(where, 1)
        self.status_strip.setMaximumWidth(460)
        self.status_strip.setMinimumWidth(240)
        top.addSpacing(SPACE.LG)
        top.addWidget(self.status_strip, 0)
        top.addSpacing(SPACE.LG)
        actions = QHBoxLayout()
        actions.setSpacing(SPACE.SM)
        actions.addStretch(1)
        top.addLayout(actions, 1)

        # Back / forward through the pages actually visited, like a browser.
        # These replace two shortcut buttons (Vision lab, Research) that
        # duplicated the sidebar one inch to the left.
        self._back_btn = describe_control(QPushButton(), "Back (Alt+Left)")
        self._back_btn.setObjectName("iconButton")
        apply_icon(self._back_btn, "arrow-left")
        self._back_btn.clicked.connect(self.go_back)
        self._forward_btn = describe_control(QPushButton(), "Forward (Alt+Right)")
        self._forward_btn.setObjectName("iconButton")
        apply_icon(self._forward_btn, "arrow-right")
        self._forward_btn.clicked.connect(self.go_forward)
        for button in (self._back_btn, self._forward_btn):
            button.setEnabled(False)
        where.insertWidget(0, self._back_btn)
        where.insertWidget(1, self._forward_btn)

        # A VISIBLE way to find any page — search takes the middle of the top
        # row so page-finding is the most obvious thing on screen. Compact
        # wording: the old label plus twelve zone buttons in one row is exactly
        # what left the zones colliding ("DevelopmentBusiness").
        self._search_btn = QPushButton("Find a Page or Action    Ctrl+K")
        self._search_btn.setObjectName("deckSearch")
        self._search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._search_btn.setToolTip("Find a page or choose something to do (Ctrl+K)")
        self._search_btn.clicked.connect(self._open_search)
        actions.addWidget(self._search_btn)

        # The compact-navigation shortcut lives in the top row (hidden unless
        # that legacy mode is on) — see set_swarm_navigation.
        self._swarm_home_btn = QPushButton("Overview")
        apply_icon(self._swarm_home_btn, "dashboard")
        self._swarm_home_btn.setObjectName("segItem")
        self._swarm_home_btn.setToolTip(
            "Back to the overview (Esc)")
        self._swarm_home_btn.clicked.connect(lambda: self.show_page_named("BRAIN"))
        self._swarm_home_btn.hide()
        actions.addWidget(self._swarm_home_btn, 0)
        actions.addWidget(self._workspace_btn)

        # Shorter now that Search is visible — it no longer has to advertise
        # Ctrl+K, which was half the clutter on the right.
        self._hint = QLabel("←  →   ·   Ctrl+Tab")
        self._hint.setObjectName("mutedLabel")
        self._hint.hide()   # shortcuts are available in control tooltips
        outer.addLayout(top)

        # ── the zones, on their OWN full-width row ────────────────────────────
        # The twelve zones used to share the top row with the title, search and
        # hint; squeezed into what was left, their labels overflowed and ran
        # into each other. Given the whole width to themselves they fit, and a
        # horizontal scroll area is the safety net so a narrow window scrolls
        # them rather than ever overlapping again.
        zone_frame = QFrame()
        zone_frame.setObjectName("segmented")
        zone_row = QHBoxLayout(zone_frame)
        zone_row.setContentsMargins(3, 3, 3, 3)
        zone_row.setSpacing(4)
        self._zone_buttons: dict[str, QPushButton] = {}
        for zone in self.ZONE_ORDER:
            label = zone.title()
            btn = QPushButton(label)
            btn.setObjectName("zoneItem")
            btn.setCheckable(True)
            has_pages = bool(self.ZONE_PAGES.get(zone))
            btn.setEnabled(has_pages)
            btn.setToolTip("No pages here yet" if not has_pages
                           else f"{label} zone")
            btn.clicked.connect(lambda _c=False, z=zone: self._select_zone(z))
            self._zone_buttons[zone] = btn
            zone_row.addWidget(btn)
        zone_row.addStretch(1)

        zone_scroll = QScrollArea()
        zone_scroll.setObjectName("zoneScroll")
        zone_scroll.setWidget(zone_frame)
        zone_scroll.setWidgetResizable(True)
        zone_scroll.setFrameShape(QFrame.Shape.NoFrame)
        zone_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        zone_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        zone_scroll.setFixedHeight(46)
        self._zone_frame = zone_frame
        self._zone_scroll = zone_scroll
        outer.addWidget(zone_scroll)
        zone_scroll.hide()

        page_bar = QHBoxLayout()
        page_bar.setSpacing(8)
        prev = describe_control(QPushButton("‹"), "Previous page")
        prev.setObjectName("iconButton")
        prev.clicked.connect(self.prev_page)
        page_bar.addWidget(prev)

        self._page_segmented = QFrame()
        self._page_segmented.setObjectName("segmented")
        self._page_seg_row = QHBoxLayout(self._page_segmented)
        self._page_seg_row.setContentsMargins(3, 3, 3, 3)
        self._page_seg_row.setSpacing(2)
        page_bar.addWidget(self._page_segmented, 1)

        nxt = describe_control(QPushButton("›"), "Next page")
        nxt.setObjectName("iconButton")
        nxt.clicked.connect(self.next_page)
        page_bar.addWidget(nxt)
        outer.addLayout(page_bar)
        # Held so swarm-first navigation can hide the whole page row, not just
        # its middle: leaving the ‹ › chevrons behind would be a tab bar with
        # the labels taken away rather than an absence of one.
        self._page_bar_widgets = (prev, self._page_segmented, nxt)
        for widget in self._page_bar_widgets:
            widget.hide()

        self._tabs: dict[int, QPushButton] = {}   # page index -> its tab button
        return frame

    def _fill_recent_menu(self) -> None:
        """Recent pages, listed when the Workspaces menu is opened."""
        menu = self._recent_menu
        menu.clear()
        names = self.recent_pages()
        if not names:
            empty = menu.addAction("Nothing opened yet")
            empty.setEnabled(False)
            return
        for name in names:
            action = menu.addAction(line_icon(self._page_icon(name)), self.display_name(name))
            action.triggered.connect(lambda _checked=False, n=name: self.show_page_named(n))

    def _set_expanded_navigation(self, expanded: bool) -> None:
        """Optional full taxonomy; the compact menu always reaches every page."""
        show_tabs = expanded and not self.swarm_navigation
        self._zone_scroll.setVisible(show_tabs)
        for widget in self._page_bar_widgets:
            widget.setVisible(show_tabs)

    # ── legacy compact navigation ────────────────────────────────────────────

    def set_swarm_navigation(self, enabled: bool) -> None:
        """Keep the old setting while retaining the permanent 2D sidebar."""
        self._swarm_navigation = bool(enabled)
        # Hide the scroll area (the zone row's container), not just the frame
        # inside it — hiding the frame alone would leave an empty scroll strip.
        self._set_expanded_navigation(self._expanded_navigation.isChecked())
        self._swarm_home_btn.setVisible(enabled)
        self._hint.setText(
            "Esc for overview"
            if enabled else "←  →   ·   Ctrl+Tab")
        if enabled:
            self.show_page_named("BRAIN")

    @property
    def swarm_navigation(self) -> bool:
        return getattr(self, "_swarm_navigation", False)


    def _open_search(self) -> None:
        """Ask the core window to open the command palette. Falls back to a
        local Ctrl+K if the signal has no listener (a bare/test deck)."""
        try:
            self.bus.open_palette.emit()
        except Exception:
            pass

    def _select_zone(self, zone: str) -> None:
        """Zone-row click: jump to that zone's first page."""
        pages_in_zone = self.ZONE_PAGES.get(zone, ())
        for i, (name, _widget) in enumerate(self._pages):
            if name in pages_in_zone:
                self._select(i, animate=True)
                return

    def _rebuild_page_row(self, zone: str | None) -> None:
        """Repopulate the page row with whichever pages belong to *zone*
        (or, for an orphan page outside every zone, just that one page —
        see the CHESS note above the class constants)."""
        while self._page_seg_row.count():
            item = self._page_seg_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._tabs = {}
        if zone is not None:
            wanted = self.ZONE_PAGES.get(zone, ())
            indices = [i for i, (name, _w) in enumerate(self._pages) if name in wanted]
        else:
            indices = [i for i, (name, _w) in enumerate(self._pages)
                      if self._zone_for_page(name) is None]
        for index in indices:
            name = self._pages[index][0]
            glyph = next((g for k, g in self._PAGE_GLYPHS.items() if k in name.upper()), "•")
            label = self.display_name(name)
            btn = QPushButton(label)
            apply_icon(btn, self._page_icon(name))
            btn.setObjectName("segItem")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, i=index: self._select(i, animate=True))
            self._tabs[index] = btn
            self._page_seg_row.addWidget(btn)

    # ── page selection ────────────────────────────────────────────────────────

    @staticmethod
    def _fit_page_margins(widget: Any) -> None:
        """Take away a page's own outer margin: the deck's gutter is the one.

        Every page wrote its own (4 px on Research, 9 on Goals, 10 on Tasks,
        16 on the Overview, 20 on Vision Lab), so on top of the deck's gutter
        the content jumped sideways from page to page and sat anywhere from
        12 to 36 px off the sidebar. With this, every page starts on the same
        line, SPACE.MD from the sidebar, the header and the window's edge.
        Only the page's outermost layout is touched; panels keep their own
        padding.
        """
        try:
            root = widget
            central = getattr(widget, "centralWidget", None)
            if callable(central) and central() is not None:
                root = central()
            layout = root.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)
        except Exception:
            pass

    @staticmethod
    def _placeholder(name: str) -> QWidget:
        """Stands in for a page until it is opened. Deliberately plain: it is
        replaced the moment anyone navigates here.

        It also carries the page's readiness honestly: "opening" while the
        page is being built, and if building fails, what went wrong and a way
        to try again — rather than "opening…" for the rest of the session.
        """
        holder = QWidget()
        holder.setObjectName("panelFrame")
        box = QVBoxLayout(holder)
        box.addStretch(1)
        label = QLabel(f"{page_display_name(name)} — opening…")
        label.setObjectName("mutedLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setAccessibleName(f"{page_display_name(name)} page status")
        box.addWidget(label)
        retry = QPushButton("Try Again")
        retry.setAccessibleName(f"Try opening {page_display_name(name)} again")
        retry.hide()
        box.addWidget(retry, 0, Qt.AlignmentFlag.AlignHCenter)
        box.addStretch(1)
        holder._status_label = label        # type: ignore[attr-defined]
        holder._retry_button = retry        # type: ignore[attr-defined]
        return holder

    def _show_unavailable(self, name: str, reason: str) -> None:
        """Turn a page's placeholder into an honest 'could not open' state."""
        holder = next((w for n, w in self._pages if n == name), None)
        label = getattr(holder, "_status_label", None)
        retry = getattr(holder, "_retry_button", None)
        if label is None or retry is None:
            return
        label.setText(f"{self.display_name(name)} could not be opened — {reason}")
        if not getattr(retry, "_orion_wired", False):
            retry.clicked.connect(lambda _c=False, n=name: self._retry_page(n))
            retry._orion_wired = True       # type: ignore[attr-defined]
        retry.show()

    def _retry_page(self, name: str) -> None:
        index = next((i for i, (n, _w) in enumerate(self._pages) if n == name), None)
        if index is not None:
            self._select(index, remember=False)

    def page_is_built(self, name: str) -> bool:
        """Whether *name* has been constructed yet (diagnostics and tests)."""
        return any(n == name for n, _ in self._pages) and name not in self._factories

    def _ensure_built(self, name: str) -> QWidget | None:
        """Build a deferred page and swap it in. Returns the live widget.

        A factory that raises must not take the deck with it: the placeholder
        stays, the reason is logged, and every other page still works. Losing
        one page is a bad afternoon; losing the deck is a broken assistant.
        """
        factory = self._factories.get(name)
        current = next((w for n, w in self._pages if n == name), None)
        if factory is None:
            return current
        try:
            widget = factory()
        except Exception as exc:
            self.bus.log.emit(f"DECK: {name} could not be opened - {exc}")
            reason = (str(exc).splitlines() or [type(exc).__name__])[0][:160]
            self._show_unavailable(name, reason or type(exc).__name__)
            return current
        if not isinstance(widget, QWidget):
            self.bus.log.emit(f"DECK: {name} did not produce a widget.")
            self._show_unavailable(name, "it did not produce a page")
            return current
        self._factories.pop(name, None)
        self._fit_page_margins(widget)
        for i, (page_name, placeholder) in enumerate(self._pages):
            if page_name != name:
                continue
            self._pages[i] = (name, widget)
            self.stack.insertWidget(i, widget)
            self.stack.removeWidget(placeholder)
            placeholder.deleteLater()
            # No stylesheet copy here: the page inherits the deck's own. The
            # copy re-parsed the whole sheet for every page (23% of a heavy
            # page's first open), replaced the page's own style, and kept a
            # later theme change from reaching it (28 Oct 2026).
            return widget
        return widget

    def page_widget(self, name: str) -> QWidget | None:
        """The live widget for *name*, building it if it has been deferred.

        Callers that need the REAL page — starting the camera, say — must go
        through this rather than reading ``_pages`` directly, which would hand
        them the placeholder.
        """
        resolved = self.resolve_page_name(name) or name
        return self._ensure_built(resolved)

    # ── what ORION can read of his deck (gui/deck_reader.py) ─────────────────

    def live_page(self, name: str) -> QWidget | None:
        """The page's widget if it has been built - never builds it, so a
        search does not wake every deferred page (BRAIN is a whole 3-D scene)."""
        if name in self._factories:
            return None
        return next((w for n, w in self._pages if n == name), None)

    def current_page_name(self) -> str | None:
        """The page on screen now."""
        return self._current_page_name()

    def page_terms(self, name: str) -> str:
        """The words a page is known by: its key, its label, its sidebar
        section and its zone - what the sidebar's filter matches too."""
        return " ".join((name, self.display_name(name), self._group_for_page(name) or "",
                         self._zone_for_page(name) or ""))

    def _select(self, index: int, animate: bool = False, *,
                remember: bool = True) -> None:
        index = max(0, min(len(self._pages) - 1, index))
        page_name = self._pages[index][0]
        previous = self.stack.currentIndex() if self.stack.count() else -1
        if remember and previous not in (-1, index) and hasattr(self, "_back"):
            self._back.append(previous)
            del self._back[:-30]
            self._forward.clear()
        # Build it the first time it is actually looked at.
        built = self._ensure_built(page_name)
        if isinstance(built, QWidget):
            from .humane import apply_heading_voice
            apply_heading_voice(built)
        self.stack.setCurrentIndex(index)
        if previous not in (-1, index) and hasattr(self, "_favourites"):
            self._favourites.visited(page_name)
            # Every change, after the switch: RECENT leaves out the page you
            # are on, so it has to know which one that now is.
            self._refresh_shortcuts()
        # A page brought up while this window is in the background must not
        # start at full rate just because it is new.
        self._notify_attention(self.isActiveWindow())
        zone = self._zone_for_page(page_name)
        section = self._group_for_page(page_name) if hasattr(self, "SIDEBAR_GROUPS") else None
        # "More" says nothing about a page; its zone does ("Monitoring › Log").
        crumb = zone if section in (None, "MORE") else section
        self._crumb_label.setText(f"{crumb.title()}  ›" if crumb else "")
        self._crumb_label.setVisible(bool(crumb))
        self._current_page_label.setText(self.display_name(page_name))
        self._current_page_label.setToolTip(
            f"{crumb.title()} › {self.display_name(page_name)}" if crumb else "")
        if hasattr(self, "_back_btn"):
            self._back_btn.setEnabled(bool(self._back))
            self._forward_btn.setEnabled(bool(self._forward))
        for name, action in self._page_actions.items():
            action.setChecked(name == page_name)
        if zone != self._current_zone:
            self._current_zone = zone
            self._rebuild_page_row(zone)
        for i, btn in self._tabs.items():
            btn.setChecked(i == index)
        for z, btn in self._zone_buttons.items():
            btn.setChecked(z == zone)
        for name, btn in self._sidebar_buttons.items():
            btn.setChecked(name == self._pages[index][0])
        page_filter = getattr(self, "_filter", None)
        if page_filter is not None:
            # A page inside a folded section is still listed while it is open.
            self._apply_filter(page_filter.text())
        current_button = self._sidebar_buttons.get(page_name)
        if current_button is not None:
            try:
                self._sidebar.ensureWidgetVisible(current_button, 0, 24)
            except Exception:
                pass
        if animate and _deck_transitions_enabled():
            self._fade_in(self.stack.currentWidget())
        self.bus.log.emit(f"DECK: {self._pages[index][0]}")

    #: Pages whose content is drawn by a native surface (QWebEngineView's
    #: Chromium compositor, a QOpenGLWidget) rather than by Qt's raster
    #: painter.  A QGraphicsEffect forces the whole subtree through software
    #: compositing, which those surfaces do not participate in: the page goes
    #: black, grey or blank for the duration and can fail to come back.  They
    #: are switched instantly instead — a fade is not worth a broken page.
    _NATIVE_SURFACE_PAGES = frozenset({"GLOBE", "CHESS", "MISSION", "WORKBENCH"})

    def _uses_native_surface(self, widget: QWidget | None) -> bool:
        if widget is None:
            return False
        for name, page in self._pages:
            if page is widget:
                if name in self._NATIVE_SURFACE_PAGES:
                    return True
                break
        # Anything hosting a WebEngine view or a GL widget anywhere beneath it.
        try:
            from PyQt6.QtOpenGLWidgets import QOpenGLWidget
            if widget.findChildren(QOpenGLWidget):
                return True
        except Exception:
            pass
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            if widget.findChildren(QWebEngineView):
                return True
        except Exception:
            pass
        return False

    def _fade_in(self, widget: QWidget | None) -> None:
        if widget is None or sip.isdeleted(widget):
            return
        # A stopped animation remains a child of the dashboard until deleted.
        # Release both it and its effect when a page switch interrupts a fade.
        previous = self._anim
        self._anim = None
        if previous is not None:
            try:
                previous.stop()
                previous_effect = previous.targetObject()
                previous.deleteLater()
                if previous_effect is not None and not sip.isdeleted(previous_effect):
                    previous_page = previous_effect.parent()
                    if (isinstance(previous_page, QWidget)
                            and not sip.isdeleted(previous_page)
                            and previous_page.graphicsEffect() is previous_effect):
                        previous_page.setGraphicsEffect(None)
            except RuntimeError:
                pass
        if self._uses_native_surface(widget):
            widget.setGraphicsEffect(None)
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(160)
        anim.setStartValue(0.25)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        owner_ref, page_ref = weakref.ref(self), weakref.ref(widget)
        animation_ref, effect_ref = weakref.ref(anim), weakref.ref(effect)

        def finished() -> None:
            owner, page = owner_ref(), page_ref()
            animation, fading_effect = animation_ref(), effect_ref()
            if owner is None or sip.isdeleted(owner) or animation is None:
                return
            # A stale completion must not clear a newer fade on the same page.
            if owner._anim is not animation:
                return
            owner._anim = None
            animation.deleteLater()
            if page is not None and not sip.isdeleted(page):
                if page.graphicsEffect() is fading_effect:
                    page.setGraphicsEffect(None)

        anim.finished.connect(finished)
        self._anim = anim
        anim.start()

    def _step(self, delta: int) -> None:
        """Move through the pages in the order the sidebar shows, wrapping."""
        order = [n for n in getattr(self, "_nav_order", []) if n in self.page_names()]
        if not order:
            self._select(self.stack.currentIndex() + delta, animate=True)
            return
        current = self._pages[self.stack.currentIndex()][0]
        position = order.index(current) if current in order else -delta
        target = order[(position + delta) % len(order)]
        self._select(self.page_names().index(target), animate=True)

    def next_page(self) -> None:
        self._step(+1)

    def prev_page(self) -> None:
        self._step(-1)

    def _select_nav_position(self, position: int) -> None:
        order = getattr(self, "_nav_order", [])
        if 0 <= position < len(order):
            self._select(self.page_names().index(order[position]), animate=True)

    def go_back(self) -> bool:
        """The page before this one (Alt+Left, or the mouse's back button)."""
        if not self._back:
            return False
        self._forward.append(self.stack.currentIndex())
        self._select(self._back.pop(), animate=True, remember=False)
        return True

    def go_forward(self) -> bool:
        if not self._forward:
            return False
        self._back.append(self.stack.currentIndex())
        self._select(self._forward.pop(), animate=True, remember=False)
        return True

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        # The side buttons most mice have. A page that uses them itself
        # accepts the event and it never arrives here.
        button = event.button()
        if button == Qt.MouseButton.BackButton and self.go_back():
            event.accept()
            return
        if button == Qt.MouseButton.ForwardButton and self.go_forward():
            event.accept()
            return
        super().mousePressEvent(event)

    #: Words people say that are not part of any page's name.
    _NAV_FILLER = frozenset({
        "the", "a", "an", "my", "page", "pages", "tab", "deck", "screen",
        "view", "window", "panel", "to", "on", "open", "show", "go", "goto",
        "bring", "up", "please", "me", "orion", "switch", "change",
    })

    #: What people call a page when they do not use its exact name.
    _NAV_ALIASES = {
        "spellscape": "BRAIN", "swarm": "BRAIN",
        # Mark XXXX.4: his stats as graphs have their own page.
        "neural": "NEURAL", "neuro": "NEURAL", "statistics": "NEURAL",
        "metrics": "NEURAL", "performance": "NEURAL",
        "mind": "BRAIN", "brains": "BRAIN",
        "command": "COMMAND CENTRE", "centre": "COMMAND CENTRE",
        "center": "COMMAND CENTRE", "console": "COMMAND CENTRE",
        "diagnostic": "DIAGNOSTICS", "health": "DIAGNOSTICS",
        "camera": "WORKBENCH", "workbench": "WORKBENCH", "scanner": "WORKBENCH",
        "lab": "WORKBENCH", "researching": "RESEARCH",
        "map": "GLOBE", "world": "GLOBE", "earth": "GLOBE",
        "logs": "LOG", "memories": "MEMORY", "plugin": "PLUGINS",
        "widget": "WIDGETS", "mission": "MISSION", "missions": "MISSION",
        # Stats are the NEURAL page's; a deck without it has them on Load.
        "security": "SECURITY", "stats": ("NEURAL", "DIAGNOSTICS"),
        "lights": "HOME", "light": "HOME", "lamps": "HOME", "house": "HOME",
        # Mark XXXVIII merges: the old names still lead to the page that holds them.
        "toolkit": "COGNITION", "tools": "COGNITION", "focus": ("FOCUS", "COGNITION"),
        "study": "COGNITION", "telemetry": "DIAGNOSTICS", "load": "DIAGNOSTICS",
        "smarthome": "HOME", "hue": "HOME", "routines": "HOME",
        # The owner's day: his calendar, focus timer and workouts.
        "schedule": "CALENDAR", "agenda": "CALENDAR", "diary": "CALENDAR",
        "week": "CALENDAR", "timer": "FOCUS", "pomodoro": "FOCUS", "concentrate": "FOCUS",
        "fitness": "FITNESS", "workout": "FITNESS", "workouts": "FITNESS",
        "exercise": "FITNESS", "gym": "FITNESS",
    }

    #: Words that name a TAB of a merged page (gui/deck_tabs.py), so "open
    #: the toolkit" lands on the tools, not on the focus timer beside them.
    _NAV_TABS = {
        "toolkit": "Tools", "tools": "Tools", "calculator": "Tools",
        "calculators": "Tools", "telemetry": "Load", "load": "Load",
        "stats": "Load", "graphs": "Load", "focus": "Focus", "study": "Focus",
        "health": "Health", "diagnostics": "Health", "diagnostic": "Health",
    }

    def resolve_page_name(self, name: str) -> str | None:
        """The page *name* means, or None if it names nothing.

        This used to be ``if name.lower() in page_name.lower()`` over the page
        list, taking the first hit. That is backwards for anything a person
        actually says: the REQUEST had to be a substring of the PAGE, so "the
        command centre page" matched nothing at all, while a one-letter
        request matched whichever page happened to contain that letter first.
        Both failure modes look the same from outside — ORION says he has
        navigated and the deck does not move.

        Resolved in order of confidence: exact, then alias, then whole-word,
        then prefix, then substring either way. Ambiguity between equally good
        matches is a failure rather than a guess.
        """
        wanted = str(name or "").strip().lower()
        if not wanted:
            return None
        names = [page_name for page_name, _w in self._pages]
        lower = {n.lower(): n for n in names}

        if wanted in lower:                                  # exact
            return lower[wanted]

        words = [w for w in re.split(r"[^a-z0-9]+", wanted)
                 if w and w not in self._NAV_FILLER]
        if not words:
            return None
        stripped = " ".join(words)
        if stripped in lower:
            return lower[stripped]

        for word in words:                                   # alias
            alias = self._NAV_ALIASES.get(word)
            for target in ((alias,) if isinstance(alias, str) else alias or ()):
                if target.lower() in lower:
                    return lower[target.lower()]

        def _unique(matches: list[str]) -> str | None:
            found = sorted(set(matches))
            return found[0] if len(found) == 1 else None

        # Exact names and aliases may be short; a FUZZY match may not. "o"
        # prefix-matched OPS, which is the old over-eager behaviour wearing a
        # new coat — one or two stray letters from the recogniser are noise,
        # not a request to navigate.
        if len(stripped) < 3:
            return None

        # whole word of the page name, e.g. "centre" -> COMMAND CENTRE
        hit = _unique([n for n in names
                       if set(re.split(r"[^a-z0-9]+", n.lower())) & set(words)])
        if hit:
            return hit
        hit = _unique([n for n in names if n.lower().startswith(stripped)])
        if hit:
            return hit
        hit = _unique([n for n in names
                       if stripped in n.lower() or n.lower() in stripped])
        return hit

    def show_page_named(self, name: str) -> bool:
        """Show the page *name* refers to. Returns whether one was found.

        The return value matters: the caller used to have no way to tell a
        successful navigation from a silent miss, so ORION reported "done"
        either way.
        """
        resolved = self.resolve_page_name(name)
        if resolved is None:
            return False
        for i, (page_name, _w) in enumerate(self._pages):
            if page_name == resolved:
                self._select(i, animate=True)
                self._show_named_tab(resolved, name)
                return True
        return False

    def _show_named_tab(self, page: str, name: str) -> None:
        words = re.split(r"[^a-z0-9]+", str(name or "").lower())
        tab = next((self._NAV_TABS[w] for w in words if w in self._NAV_TABS), None)
        if tab is None:
            return
        widget = self.live_page(page)
        show_tab = getattr(widget, "show_tab", None)
        if callable(show_tab):
            show_tab(tab)

    # ── swipe + keyboard ──────────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: Any) -> bool:
        # Read through getattr, never self.stack. This filter watches the page
        # stack, and the stack can outlive this window's Python state: the
        # unified shell moves it into the core window, and the garbage
        # collector breaks a reference cycle by emptying this object's
        # __dict__ BEFORE the C++ side is destroyed. The stack still gets
        # events in that gap (it is hidden and deleted along with whichever
        # window now owns it), they still arrive here, and an AttributeError
        # raised inside a Qt virtual is not an exception under PyQt6 — it is
        # qFatal, which kills the process with 0xC0000409 and no traceback.
        # That is what intermittently took the whole test suite down at ~93%.
        et = event.type()
        from PyQt6.QtCore import QEvent
        page_filter = getattr(self, "_filter", None)
        if page_filter is not None and obj is page_filter and et == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Escape:
                # Escape in the filter clears it and hands the keyboard back
                # to the page, instead of reaching the window's own Escape.
                page_filter.clear()
                stack = getattr(self, "stack", None)
                if stack is not None:
                    stack.setFocus()
                return True
            return False
        if obj is getattr(self, "stack", None):
            if et == QEvent.Type.MouseButtonPress:
                self._drag_x = event.position().x()
            elif et == QEvent.Type.MouseButtonRelease and self._drag_x is not None:
                dx = event.position().x() - self._drag_x
                self._drag_x = None
                if dx <= -self.SWIPE_THRESHOLD:
                    self.next_page()
                    return True
                if dx >= self.SWIPE_THRESHOLD:
                    self.prev_page()
                    return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        # Escape returns to the swarm — the keyboard half of the escape hatch.
        # With the tab bars retired, a user who opens a page from the graph
        # needs a way back that does not depend on finding one button.
        # A page that navigates with these keys itself (the chess board steps
        # through its moves with Left/Right) gets them first; the deck only
        # turns the page when the page declines. They used to flip the user
        # off the chess page mid-review.
        handler = getattr(self.stack.currentWidget(), "handle_navigation_key", None)
        if key != Qt.Key.Key_Escape and callable(handler) and handler(event):
            return
        modifiers = event.modifiers()
        plain = not (modifiers & (Qt.KeyboardModifier.ControlModifier
                                  | Qt.KeyboardModifier.AltModifier))
        if key == Qt.Key.Key_Escape and plain:
            # Home, from anywhere. It used to work only in the legacy
            # swarm mode, so from most pages Esc did nothing.
            if self._pages[self.stack.currentIndex()][0] != "BRAIN":
                self.show_page_named("BRAIN")
        elif key == Qt.Key.Key_Right and plain:
            self.next_page()
        elif key == Qt.Key.Key_Left and plain:
            self.prev_page()
        else:
            super().keyPressEvent(event)

    # ── window behaviour ──────────────────────────────────────────────────────

    def toggle(self, page: str = "") -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
        if page:
            self.show_page_named(page)

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        """Tell a 3-D page when this window stops being the one in front.

        Qt hides a page when it is swapped out or the window is minimised, and
        a hidden 3-D page already stops rendering. But the deck's ordinary
        home is a SECOND MONITOR, where it is visible, never hidden, and not
        being watched -- and there it kept a whole three.js scene running at
        the refresh rate, competing with ORION's face for the one GPU they
        share. The face's frame rate being unstable is what that felt like.

        Throttled rather than stopped: a second monitor is exactly where
        people leave something running to glance at.
        """
        try:
            from PyQt6.QtCore import QEvent

            if event.type() in (QEvent.Type.WindowActivate,
                                QEvent.Type.WindowDeactivate):
                self._notify_attention(self.isActiveWindow())
        except Exception:
            pass
        super().changeEvent(event)

    def _notify_attention(self, attended: bool) -> None:
        """Pass the window's attention state to whichever page can use it.

        Only the page on screen is told. A page that is not current is either
        unbuilt or already hidden, and both of those stop on their own.
        """
        try:
            widget = self.stack.currentWidget()
        except Exception:
            return
        setter = getattr(widget, "set_attended", None)
        if callable(setter):
            try:
                setter(bool(attended))
            except Exception:
                pass

    #: How long a closed deck keeps its heavy 3-D renderers before giving
    #: them back. Long enough that closing and reopening to check something
    #: costs nothing; short enough that a deck closed and forgotten is not
    #: still holding hundreds of megabytes an hour later.
    RELEASE_AFTER_SECONDS = 180

    def closeEvent(self, event: Any) -> None:
        # Hide rather than destroy so panel state survives the session.
        event.ignore()
        self.hide()
        # Hiding already stops the 3-D pages rendering, but it does not give
        # back what they HOLD -- a WebEngine renderer and its few hundred
        # megabytes stay for the whole session. That is worth reclaiming on a
        # machine under memory pressure, and costs only a rebuild the next
        # time the deck is opened.
        self._arm_release()

    def _arm_release(self) -> None:
        try:
            from PyQt6.QtCore import QTimer
        except Exception:
            return
        timer = getattr(self, "_release_timer", None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._release_heavy_pages)
            self._release_timer = timer
        timer.start(int(self.RELEASE_AFTER_SECONDS * 1000))

    def _release_heavy_pages(self) -> None:
        """Give back the renderers of any built page that can rebuild itself.

        Only pages that offer release() are touched, and only while the deck
        is still hidden -- reopening it in the meantime cancels this.
        """
        if self.isVisible():
            return
        for name, widget in self._pages:
            release = getattr(widget, "release", None)
            if not callable(release):
                continue
            try:
                release()
            except Exception:
                continue

