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

from typing import Any

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import APP_NAME


class UnifiedDashboard(QMainWindow):
    SWIPE_THRESHOLD = 90  # px of horizontal drag to change page

    def __init__(self, bus: OrionBus, pages: list[tuple[str, QWidget]]) -> None:
        super().__init__()
        self.bus = bus
        self._pages = pages
        self.setWindowTitle(f"{APP_NAME} — Command Deck")
        self.setMinimumSize(1040, 700)
        self.resize(1280, 820)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        for _name, widget in pages:
            self.stack.addWidget(widget)
        layout.addWidget(self.stack, 1)

        self.setCentralWidget(root)
        self._drag_x: float | None = None
        self._anim: QPropertyAnimation | None = None
        self._select(0)

        # Swipe detection over the page area.
        self.stack.installEventFilter(self)

    # ── header (segmented tabs + nav) ─────────────────────────────────────────

    # Concise glyph per page so the tab bar is scannable at a glance.
    _PAGE_GLYPHS = {
        "WIDGET": "⧉", "TOOLKIT": "⚒", "STUDIO": "♪",
        "COMMAND": "◉", "DIAGNOSTIC": "◈", "GLOBE": "◍",
    }

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("headerFrame")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(8)

        title = QLabel("COMMAND DECK")
        title.setObjectName("titleLabel")
        layout.addWidget(title)
        layout.addStretch(1)

        prev = QPushButton("‹")
        prev.setObjectName("iconButton")
        prev.clicked.connect(self.prev_page)
        layout.addWidget(prev)

        # Segmented tabs — the same control language as the core window.
        segmented = QFrame()
        segmented.setObjectName("segmented")
        seg_row = QHBoxLayout(segmented)
        seg_row.setContentsMargins(3, 3, 3, 3)
        seg_row.setSpacing(2)
        self._tabs: list[QPushButton] = []
        for index, (name, _widget) in enumerate(self._pages):
            glyph = next((g for k, g in self._PAGE_GLYPHS.items() if k in name.upper()), "•")
            label = "CENTRE" if "COMMAND" in name.upper() else name
            btn = QPushButton(f"{glyph}  {label}")
            btn.setObjectName("segItem")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, i=index: self._select(i, animate=True))
            self._tabs.append(btn)
            seg_row.addWidget(btn)
        layout.addWidget(segmented)

        nxt = QPushButton("›")
        nxt.setObjectName("iconButton")
        nxt.clicked.connect(self.next_page)
        layout.addWidget(nxt)
        layout.addStretch(1)

        self._hint = QLabel("swipe · ← → · Ctrl+Tab")
        self._hint.setObjectName("mutedLabel")
        layout.addWidget(self._hint)
        return frame

    # ── page selection ────────────────────────────────────────────────────────

    def _select(self, index: int, animate: bool = False) -> None:
        index = max(0, min(len(self._pages) - 1, index))
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tabs):
            btn.setChecked(i == index)
        if animate:
            self._fade_in(self.stack.currentWidget())
        self.bus.log.emit(f"DECK: {self._pages[index][0]}")

    def _fade_in(self, widget: QWidget | None) -> None:
        if widget is None:
            return
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(220)
        anim.setStartValue(0.25)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start()
        self._anim = anim

    def next_page(self) -> None:
        self._select(self.stack.currentIndex() + 1, animate=True)

    def prev_page(self) -> None:
        self._select(self.stack.currentIndex() - 1, animate=True)

    def show_page_named(self, name: str) -> None:
        for i, (page_name, _w) in enumerate(self._pages):
            if name.lower() in page_name.lower():
                self._select(i, animate=True)
                return

    # ── swipe + keyboard ──────────────────────────────────────────────────────

    def eventFilter(self, obj: Any, event: Any) -> bool:
        et = event.type()
        from PyQt6.QtCore import QEvent
        if obj is self.stack:
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
        if key == Qt.Key.Key_Right:
            self.next_page()
        elif key == Qt.Key.Key_Left:
            self.prev_page()
        elif key == Qt.Key.Key_Tab and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.next_page()
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

    def closeEvent(self, event: Any) -> None:
        # Hide rather than destroy so panel state survives the session.
        event.ignore()
        self.hide()
