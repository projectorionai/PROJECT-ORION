"""
A place for what ORION found, instead of a sentence about it.

ORION already fetches real material — the briefing pulls news stories with
titles and links, the research engine gathers sources the same way — and all of
it arrived as spoken prose and a line in the activity log. A headline read
aloud cannot be clicked, and a log line is not a reading surface: by the time
three stories have gone past, the first has scrolled away and the URL was never
somewhere you could reach.

So results that have a title and a link get a surface. Nothing here fetches,
ranks or summarises anything; it renders rows another part of ORION already
produced, and opens one when it is clicked.

Deliberately modest
-------------------
Cards, not a browser. A title, where it came from, and a click. Embedding pages
would mean a second Chromium render process beside the one the globe already
uses, which is precisely the GPU contention that broke the globe's context
once before.

It also shows its emptiness honestly rather than pretending: an empty panel
says nothing has been found yet, because a blank rectangle reads as broken.
"""

from __future__ import annotations

from typing import Any, Iterable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .style import C, rgba
from .widgets import describe_control

#: Most cards kept on screen. Older ones fall off the bottom; this is a view of
#: what just happened, not an archive — the knowledge store is the archive.
MAX_CARDS = 40

#: Longest title rendered before it is elided. Real headlines run long and a
#: card that grows to six lines stops being scannable.
TITLE_CHARS = 150


class ContentCard(QFrame):
    """One result: what it is, where it came from, and a click."""

    def __init__(self, title: str, url: str = "", source: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.url = str(url or "")
        self.title = str(title or "").strip()
        self.source = str(source or "").strip()

        self.setObjectName("contentCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.url
                       else Qt.CursorShape.ArrowCursor)
        self.setStyleSheet(
            f"#contentCard {{ background: {rgba(C.PANEL_HI, 0.82)};"
            f" border: 1px solid {rgba(C.WHITE, 0.055)};"
            f" border-top: 1px solid {rgba(C.WHITE, 0.10)};"
            f" border-radius: 14px; }}"
            f"#contentCard:hover {{ background: {rgba(C.PANEL_HI, 0.94)};"
            f" border-color: {rgba(C.WHITE, 0.18)}; }}"
            f"QLabel#cardTitle {{ color: {C.WHITE}; }}"
            f"QLabel#cardSource {{ color: {C.MUTED}; font-size: 11px; }}"
        )

        shown = self.title if len(self.title) <= TITLE_CHARS else (
            self.title[:TITLE_CHARS].rstrip() + "…")
        title_label = QLabel(shown)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)
        title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))

        trail = " · ".join(part for part in (self.source, _host(self.url)) if part)
        source_label = QLabel(trail or "no source given")
        source_label.setObjectName("cardSource")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(3)
        layout.addWidget(title_label)
        layout.addWidget(source_label)

        if self.url:
            describe_control(self, f"Open: {self.title[:70]}",
                             f"Opens {self.url} in your browser.")
        else:
            describe_control(self, self.title[:70], "No link for this item.")

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802  (Qt naming)
        if self.url and event.button() == Qt.MouseButton.LeftButton:
            try:
                QDesktopServices.openUrl(QUrl(self.url))
            except Exception:
                pass
        super().mouseReleaseEvent(event)


def _host(url: str) -> str:
    """The readable part of a link. A full URL on a card is noise."""
    raw = str(url or "")
    if "://" not in raw:
        return ""
    try:
        host = raw.split("://", 1)[1].split("/", 1)[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


class ContentPanel(QWidget):
    """A scrollable stack of cards, newest first."""

    #: Emitted when a card is added, so a face can glance down at it.
    arrived = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPanel")
        # The scroll area's VIEWPORT is a separate widget from the QScrollArea
        # and from the widget inside it, and it keeps the platform's default
        # light grey unless it is named explicitly. Styling only "QScrollArea"
        # leaves a pale slab under the cards wherever they do not reach the
        # bottom, which on a near-black shell is the most obvious thing on
        # screen.
        self.setStyleSheet(
            f"#contentPanel {{ background: {C.BG}; }}"
            f"QScrollArea {{ background: {C.BG}; border: 0; }}"
            f"QScrollArea > QWidget > QWidget {{ background: {C.BG}; }}"
            f"#contentInner {{ background: {C.BG}; }}"
            f"QLabel#contentEmpty {{ color: {C.FAINT}; }}"
            f"QScrollBar:vertical {{ background: {C.BG}; width: 9px; }}"
            f"QScrollBar::handle:vertical {{ background: {C.ACCENT_DEEP};"
            f"  border-radius: 4px; min-height: 30px; }}"
            f"QScrollBar::handle:vertical:hover {{ background: {C.PRI_DIM}; }}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            "{ height: 0; }"
        )

        self._cards: list[ContentCard] = []

        self._inner = QWidget()
        self._inner.setObjectName("contentInner")
        self._stack = QVBoxLayout(self._inner)
        self._stack.setContentsMargins(8, 8, 8, 8)
        self._stack.setSpacing(6)

        self._empty = QLabel("Nothing found yet.")
        self._empty.setObjectName("contentEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._empty)
        self._stack.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("contentScroll")
        self._scroll.viewport().setObjectName("contentViewport")
        self._scroll.setWidget(self._inner)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll)

    # ── content ──────────────────────────────────────────────────────────────

    def show_results(self, rows: Iterable[Any]) -> int:
        """Render *rows* as cards, newest at the top. Returns how many landed.

        Rows are whatever the producing subsystem already has: the briefing's
        articles and the research engine's sources both carry title and url, so
        neither has to be reshaped to appear here.
        """
        cards = []
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("headline") or "").strip()
            if not title:
                continue                    # a card with no title says nothing
            cards.append(ContentCard(
                title,
                str(row.get("url") or row.get("link") or ""),
                str(row.get("topic") or row.get("source") or ""),
            ))
        if not cards:
            return 0

        self._empty.hide()
        for card in reversed(cards):
            self._stack.insertWidget(0, card)
            self._cards.insert(0, card)

        # Oldest fall off the bottom. This is a view of what just happened, not
        # an archive — the knowledge store is the archive.
        while len(self._cards) > MAX_CARDS:
            stale = self._cards.pop()
            stale.setParent(None)
            stale.deleteLater()

        self._scroll.verticalScrollBar().setValue(0)
        self.arrived.emit(len(cards))
        return len(cards)

    def clear(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._empty.show()

    @property
    def count(self) -> int:
        return len(self._cards)

    def titles(self) -> list[str]:
        return [card.title for card in self._cards]


__all__ = ["MAX_CARDS", "TITLE_CHARS", "ContentCard", "ContentPanel"]
