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

from typing import Any

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
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
from ..constants import APP_NAME
from .widgets import describe_control

# Distinct from any real zone name (or the None used for an orphan page) so
# the very first _select() call always rebuilds the page row regardless of
# which zone page 0 happens to belong to.
_UNSET_ZONE = object()


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
        self._factories: dict[str, Any] = {}
        resolved: list[tuple[str, QWidget]] = []
        for name, page in pages:
            if isinstance(page, QWidget):
                resolved.append((name, page))
                continue
            if callable(page):
                self._factories[name] = page
                resolved.append((name, self._placeholder(name)))
                continue
            resolved.append((name, self._placeholder(name)))
        self._pages = resolved
        self._current_zone: Any = _UNSET_ZONE
        self.setWindowTitle(f"{APP_NAME} — Command Deck")
        self.setMinimumSize(1040, 700)
        self.resize(1280, 820)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(self._build_header())

        # Operational status strip (Mark XXVI §14): one unobtrusive line telling
        # the user what ORION is doing right now — mode + voice state — driven by
        # the always-present bus.state signal. Enriched (focus/mission) later.
        from .status_strip import StatusStrip
        self.status_strip = StatusStrip()
        layout.addWidget(self.status_strip)
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
        self._content_splitter.setSizes([210, 1000])
        layout.addWidget(self._content_splitter, 1)

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
    _DISPLAY_NAMES = {
        "BRAIN": "Brain",
        "WORKBENCH": "Vision lab", "RESEARCH": "Research console",
        "COMMAND CENTRE": "Command centre", "OPS": "Operations",
    }

    @classmethod
    def display_name(cls, name: str) -> str:
        return cls._DISPLAY_NAMES.get(str(name), str(name).title())

    def page_names(self) -> list[str]:
        """Every deck page's label, in tab order — the source of truth the
        Command Palette indexes (Mark XX design-spec §8) instead of a
        hardcoded copy that could drift from the real page list."""
        return [name for name, _widget in self._pages]

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
        "INTELLIGENCE": ("BRAIN", "MISSION", "GLOBE"),
        "RESEARCH": ("RESEARCH", "LIBRARY"),
        "DEVELOPMENT": ("WORKBENCH", "DEVELOPMENT"),
        "BUSINESS": ("TOOLKIT",),
        # Mark XXXI: STUDIO, DESIGN, FASHION and ENTERTAINMENT were retired
        # (a status log and three chat personas). MARKETING — the Maynards
        # Media creator workspace — is the creative work that remains.
        "CREATIVE": ("MARKETING",),
        "AUTOMATION": ("AUTOMATION",),
        # COGNITION (Mark XXVI): the personal learning loop — deep-focus blocks
        # and spaced-repetition review — a first-class operational surface for
        # the study/focus engines that until now had only voice tools.
        "OPERATIONS": ("OPS", "COGNITION"),
        "COMMUNICATION": ("WIDGETS",),
        "MEMORY": ("MEMORY",),
        "MONITORING": ("DIAGNOSTICS", "LOG", "TELEMETRY"),
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
        "WORKBENCH": "⌖", "DEVELOPMENT": "⌨", "AUTOMATION": "⚙",
        "MARKETING": "▲", "RESEARCH": "⌕",
        "SECURITY": "🛡", "PLUGIN": "⧩",
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

    def _build_sidebar(self) -> QScrollArea:
        """Persistent page targets beside the content, including every zone."""
        scroll = QScrollArea()
        scroll.setObjectName("zoneScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QFrame()
        inner.setObjectName("panelFrame")
        column = QVBoxLayout(inner)
        column.setContentsMargins(8, 12, 8, 12)
        column.setSpacing(3)
        self._sidebar_buttons: dict[str, QPushButton] = {}
        # The order a person SEES the pages in. Next/previous follow it; they
        # used to follow construction order, so the arrows jumped between
        # sections in an order that matched nothing on screen.
        self._nav_order: list[str] = []
        assigned: set[str] = set()
        for zone in self.ZONE_ORDER:
            names = [name for name, _ in self._pages
                     if name in self.ZONE_PAGES.get(zone, ())]
            if not names:
                continue
            heading = QLabel(zone)
            heading.setObjectName("panelHeading")
            column.addWidget(heading)
            for name in names:
                self._sidebar_page_button(column, name)
                assigned.add(name)
        for name, _ in self._pages:
            if name not in assigned:
                self._sidebar_page_button(column, name)
        column.addStretch(1)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(180)
        return scroll

    def _sidebar_page_button(self, column: QVBoxLayout, name: str) -> None:
        button = QPushButton(self.display_name(name))
        button.setObjectName("deckTab")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAccessibleName(f"Open {self.display_name(name)}")
        self._nav_order.append(name)
        position = len(self._nav_order)
        if position <= 9:
            button.setToolTip(f"{self.display_name(name)}   Ctrl+{position}")
        button.clicked.connect(lambda _checked=False, page=name: self.show_page_named(page))
        self._sidebar_buttons[name] = button
        column.addWidget(button)

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerFrame")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(14, 8, 14, 8)
        outer.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._workspace_btn = QToolButton()
        self._workspace_btn.setObjectName("workspaceMenu")
        self._workspace_btn.setText("Workspaces")
        self._workspace_btn.setAccessibleName("Browse all workspaces")
        self._workspace_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._workspace_menu = QMenu(self._workspace_btn)
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
        self._expanded_navigation = self._workspace_menu.addAction("Show navigation tabs")
        self._expanded_navigation.setCheckable(True)
        self._expanded_navigation.toggled.connect(self._set_expanded_navigation)
        self._workspace_btn.setMenu(self._workspace_menu)
        top.addWidget(self._workspace_btn)

        from ..constants import APP_MARK

        mark = QLabel(APP_MARK.upper())
        mark.setObjectName("markTag")
        mark.setToolTip(f"{APP_NAME} — Command Deck")
        top.addWidget(mark, 0, Qt.AlignmentFlag.AlignVCenter)
        self._current_page_label = QLabel()
        self._current_page_label.setObjectName("workspaceTitle")
        top.addWidget(self._current_page_label)
        top.addStretch(1)

        self._workbench_btn = QPushButton("Vision lab")
        self._workbench_btn.setObjectName("workbenchShortcut")
        self._workbench_btn.setToolTip("Scan anything with the camera, on a grid")
        self._workbench_btn.clicked.connect(lambda: self.show_page_named("WORKBENCH"))
        self._workbench_btn.setVisible(any(name == "WORKBENCH" for name, _ in self._pages))
        top.addWidget(self._workbench_btn)
        self._research_btn = QPushButton("Research")
        self._research_btn.setObjectName("workbenchShortcut")
        self._research_btn.setToolTip("The live research console")
        self._research_btn.clicked.connect(lambda: self.show_page_named("RESEARCH"))
        self._research_btn.setVisible(any(name == "RESEARCH" for name, _ in self._pages))
        top.addWidget(self._research_btn)

        # A VISIBLE way to find any page — search takes the middle of the top
        # row so page-finding is the most obvious thing on screen. Compact
        # wording: the old label plus twelve zone buttons in one row is exactly
        # what left the zones colliding ("DevelopmentBusiness").
        self._search_btn = QPushButton("Search    Ctrl+K")
        self._search_btn.setObjectName("deckSearch")
        self._search_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._search_btn.setToolTip("Find any page or run any tool (Ctrl+K)")
        self._search_btn.clicked.connect(self._open_search)
        top.addWidget(self._search_btn)

        # The compact-navigation shortcut lives in the top row (hidden unless
        # that legacy mode is on) — see set_swarm_navigation.
        self._swarm_home_btn = QPushButton("✦  OVERVIEW")
        self._swarm_home_btn.setObjectName("segItem")
        self._swarm_home_btn.setToolTip(
            "Back to the overview (Esc)")
        self._swarm_home_btn.clicked.connect(lambda: self.show_page_named("BRAIN"))
        self._swarm_home_btn.hide()
        top.addWidget(self._swarm_home_btn, 0)

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
            label = "CENTRE" if "COMMAND" in name.upper() else name
            btn = QPushButton(f"{glyph}  {label}")
            btn.setObjectName("segItem")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, i=index: self._select(i, animate=True))
            self._tabs[index] = btn
            self._page_seg_row.addWidget(btn)

    # ── page selection ────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder(name: str) -> QWidget:
        """Stands in for a page until it is opened. Deliberately plain: it is
        replaced the moment anyone navigates here."""
        holder = QWidget()
        holder.setObjectName("panelFrame")
        box = QVBoxLayout(holder)
        label = QLabel(f"{name.title()} — opening…")
        label.setObjectName("mutedLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(label)
        return holder

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
            return current
        if not isinstance(widget, QWidget):
            self.bus.log.emit(f"DECK: {name} did not produce a widget.")
            return current
        self._factories.pop(name, None)
        for i, (page_name, placeholder) in enumerate(self._pages):
            if page_name != name:
                continue
            self._pages[i] = (name, widget)
            self.stack.insertWidget(i, widget)
            self.stack.removeWidget(placeholder)
            placeholder.deleteLater()
            # The deck copies the window's stylesheet once, at startup, so a
            # page built later would otherwise arrive unstyled.
            try:
                widget.setStyleSheet(self.styleSheet())
            except Exception:
                pass
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
        self._ensure_built(page_name)
        self.stack.setCurrentIndex(index)
        # A page brought up while this window is in the background must not
        # start at full rate just because it is new.
        self._notify_attention(self.isActiveWindow())
        zone = self._zone_for_page(page_name)
        self._current_page_label.setText(
            f"{zone.title()}  ›  {self.display_name(page_name)}" if zone
            else self.display_name(page_name))
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
        current_button = self._sidebar_buttons.get(page_name)
        if current_button is not None:
            try:
                self._sidebar.ensureWidgetVisible(current_button, 0, 24)
            except Exception:
                pass
        if animate:
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
        if widget is None:
            return
        # Stop the previous animation before starting another. Without this a
        # fast run through several pages leaves multiple animations driving
        # effects on widgets that may already have been cleared — one of the
        # ways deck navigation felt unstable.
        previous = self._anim
        if previous is not None:
            try:
                previous.stop()
            except RuntimeError:
                pass
            self._anim = None
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
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start()
        self._anim = anim

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
        "spellscape": "BRAIN", "swarm": "BRAIN", "neural": "BRAIN",
        "mind": "BRAIN", "brains": "BRAIN",
        "command": "COMMAND CENTRE", "centre": "COMMAND CENTRE",
        "center": "COMMAND CENTRE", "console": "COMMAND CENTRE",
        "diagnostic": "DIAGNOSTICS", "health": "DIAGNOSTICS",
        "camera": "WORKBENCH", "workbench": "WORKBENCH", "scanner": "WORKBENCH",
        "lab": "WORKBENCH", "researching": "RESEARCH",
        "map": "GLOBE", "world": "GLOBE", "earth": "GLOBE",
        "logs": "LOG", "memories": "MEMORY", "plugin": "PLUGINS",
        "widget": "WIDGETS", "mission": "MISSION", "missions": "MISSION",
        "security": "SECURITY", "stats": "TELEMETRY",
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
            if alias and alias.lower() in lower:
                return lower[alias.lower()]

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
                return True
        return False

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

