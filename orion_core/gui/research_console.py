"""
ResearchConsole — ORION's research, shown while it happens.

  "ORION must SHOW me his working (for example, actually researching online)"

Every stage of a run was already announced — the search being run, the page
being opened, the note being taken, the section being written — but it was
announced into the activity log, interleaved with everything else ORION does,
where it read as noise rather than work. This page is where that stream
lands, in order, with the pages he actually read listed as links you can open.

Three columns:

    THE WORK     every step as it happens: questions, searches, pages opened,
                 notes taken, sections written.
    NOW READING  the page currently being read — its title, address and the
                 text ORION pulled out of it — then the note he took from it.
    SOURCES      every page read this run, clickable, in the order read.

It also STARTS research: a topic box with depth and style, so the page is not
only a window onto work begun by voice. The request goes through the same
``research`` tool the voice path calls, never a parallel implementation.

Nothing here is a source of truth. It renders ``dashboard_event
("research_step", …)`` and forgets it; the notes and the paper are filed by
the research engine itself.
"""

from __future__ import annotations

import html
import time
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import background

#: Glyph per step kind — scannable at a glance without colour.
_GLYPHS = {
    "start": "▶", "paper_start": "▶", "planning": "◇", "question": "?", "follow_up": "↳",
    "search": "⌕", "search_backend": "·", "search_done": "·",
    "search_failed": "✕", "search_unavailable": "✕",
    "reading": "▤", "read": "▤", "thin": "∅", "unreachable": "✕",
    "note": "✎", "satisfied": "✓", "time": "◷", "done": "✓",
    "outline": "§", "planned": "§", "writing": "✍", "section_done": "✓",
    "framing": "✍", "finished": "★", "saved": "⤓", "writing_up": "✍",
}

#: Most rows kept in the feed. A 50-page run produces a few hundred steps;
#: the widget stays cheap because old rows are dropped, not hidden.
MAX_FEED_ROWS = 600

_DEPTHS = (("Brief · ~5 pages", "brief"), ("Standard · ~15 pages", "standard"),
           ("Thesis · ~35 pages", "thesis"), ("Exhaustive · ~50 pages", "exhaustive"))
_STYLES = (("Literature review", "literature_review"), ("Technical deep-dive", "technical"),
           ("Comparative", "comparative"), ("Critical appraisal", "critical"),
           ("Systematic", "systematic"), ("Teaching primer", "primer"))
_MODES = (("Write a paper (Word)", "paper"), ("Investigate & summarise", "browse"))


def _panel(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return frame, layout


class ResearchConsole(QWidget):
    """The live research page: start a run, watch it work, open what it read."""

    def __init__(self, bus: Any, dispatcher: Any = None) -> None:
        super().__init__()
        self.bus = bus
        self.dispatcher = dispatcher
        self._sources: list[str] = []
        self._running = False
        self._build()
        try:
            bus.dashboard_event.connect(self._on_dashboard_event)
        except Exception:
            pass

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(10)

        head = QHBoxLayout()
        title = QLabel("Live research")
        title.setObjectName("titleLabel")
        head.addWidget(title)
        self.status = QLabel("Idle — ask ORION to research something, or start a run here.")
        self.status.setObjectName("mutedLabel")
        head.addWidget(self.status, 1)
        root.addLayout(head)

        # ── the request ──────────────────────────────────────────────────────
        ask = QFrame()
        ask.setObjectName("panelFrame")
        row = QHBoxLayout(ask)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        self.topic = QLineEdit()
        self.topic.setPlaceholderText("What should ORION research?  e.g. neural engineering")
        self.topic.setAccessibleName("Research topic")
        self.topic.returnPressed.connect(self._start)
        row.addWidget(self.topic, 3)
        self.mode = QComboBox()
        self.mode.setAccessibleName("Research mode")
        for label, value in _MODES:
            self.mode.addItem(label, value)
        row.addWidget(self.mode, 1)
        self.depth = QComboBox()
        self.depth.setAccessibleName("Research depth")
        for label, value in _DEPTHS:
            self.depth.addItem(label, value)
        self.depth.setCurrentIndex(1)
        row.addWidget(self.depth, 1)
        self.style_box = QComboBox()
        self.style_box.setAccessibleName("Research style")
        for label, value in _STYLES:
            self.style_box.addItem(label, value)
        row.addWidget(self.style_box, 1)
        self.start_btn = QPushButton("Research")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self._start)
        row.addWidget(self.start_btn)
        root.addWidget(ask)

        self.progress = QProgressBar()
        self.progress.setObjectName("researchProgress")
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m sections")
        self.progress.setMaximumHeight(14)
        self.progress.hide()
        root.addWidget(self.progress)

        # ── the three columns ────────────────────────────────────────────────
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        work, work_layout = _panel("THE WORK")
        self.feed = QListWidget()
        self.feed.setObjectName("researchFeed")
        self.feed.setWordWrap(True)
        self.feed.setUniformItemSizes(False)
        self.feed.itemActivated.connect(self._open_item)
        self.feed.itemDoubleClicked.connect(self._open_item)
        work_layout.addWidget(self.feed, 1)
        split.addWidget(work)

        now, now_layout = _panel("NOW READING")
        self.page_title = QLabel("—")
        self.page_title.setObjectName("sectionTitle")
        self.page_title.setWordWrap(True)
        now_layout.addWidget(self.page_title)
        self.page_url = QLabel("")
        self.page_url.setObjectName("mutedLabel")
        self.page_url.setWordWrap(True)
        self.page_url.setOpenExternalLinks(True)
        self.page_url.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        now_layout.addWidget(self.page_url)
        self.page_text = QTextBrowser()
        self.page_text.setObjectName("logBox")
        self.page_text.setOpenExternalLinks(True)
        self.page_text.setPlaceholderText("The text of the page ORION is reading appears here.")
        now_layout.addWidget(self.page_text, 3)
        note_head = QLabel("HIS NOTE")
        note_head.setObjectName("panelHeading")
        now_layout.addWidget(note_head)
        self.note_text = QTextBrowser()
        self.note_text.setObjectName("logBox")
        self.note_text.setPlaceholderText("What ORION wrote down from that page.")
        now_layout.addWidget(self.note_text, 2)
        split.addWidget(now)

        read, read_layout = _panel("SOURCES READ")
        self.sources = QListWidget()
        self.sources.setObjectName("researchSources")
        self.sources.setWordWrap(True)
        self.sources.itemActivated.connect(self._open_item)
        self.sources.itemDoubleClicked.connect(self._open_item)
        read_layout.addWidget(self.sources, 1)
        hint = QLabel("Double-click a source to open it in your browser.")
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)
        read_layout.addWidget(hint)
        split.addWidget(read)

        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)
        root.addWidget(split, 1)

    # ── starting a run ───────────────────────────────────────────────────────

    def _start(self) -> None:
        topic = self.topic.text().strip()
        if not topic:
            self.status.setText("Type a topic first.")
            return
        dispatcher = self.dispatcher
        if dispatcher is None or not hasattr(dispatcher, "dispatch"):
            self.status.setText("The research engine is not attached yet.")
            return
        args = {"action": self.mode.currentData(), "topic": topic,
                "depth": self.depth.currentData(), "style": self.style_box.currentData()}
        self.reset()
        self.status.setText(f"Starting research on “{topic}”…")
        self.start_btn.setEnabled(False)

        async def _run() -> None:
            try:
                result = await dispatcher.dispatch("research", args)
                message = str(getattr(result, "text", "") or getattr(result, "message", "") or result)
            except Exception as exc:
                message = f"The research run failed: {exc}"
            try:
                self.status.setText(message.splitlines()[0][:220] if message else "Done.")
                self.start_btn.setEnabled(True)
            except RuntimeError:
                pass    # the page was destroyed while the run was going

        background.spawn(_run(), name="orion-research-console")

    def reset(self) -> None:
        """Clear the page for a new run."""
        self.feed.clear()
        self.sources.clear()
        self._sources = []
        self.page_title.setText("—")
        self.page_url.setText("")
        self.page_text.clear()
        self.note_text.clear()
        self.progress.hide()

    # ── rendering the stream ─────────────────────────────────────────────────

    def _on_dashboard_event(self, channel: str, payload: Any) -> None:
        if channel != "research_step" or not isinstance(payload, dict):
            return
        try:
            self.add_step(payload)
        except RuntimeError:
            pass    # destroyed mid-signal

    def add_step(self, step: dict) -> None:
        """Render one research step. Public so a test can drive it directly."""
        kind = str(step.get("kind") or "")
        message = str(step.get("message") or "").strip()
        if not message:
            return
        if kind == "paper_start":
            # A paper run: its investigation announces its own "start" a
            # moment later, which must not wipe this line.
            self.reset()
            self._running = True
        elif kind == "start" and not self._running:
            self.reset()
        stamp = time.strftime("%H:%M:%S", time.localtime(float(step.get("at") or time.time())))
        item = QListWidgetItem(f"{stamp}  {_GLYPHS.get(kind, '·')}  {message}")
        url = str(step.get("url") or "")
        if url:
            item.setData(Qt.ItemDataRole.UserRole, url)
            item.setToolTip(url)
        self.feed.addItem(item)
        while self.feed.count() > MAX_FEED_ROWS:
            self.feed.takeItem(0)
        self.feed.scrollToBottom()
        self.status.setText(message[:200])

        if kind == "reading" and url:
            self.page_title.setText(str(step.get("title") or url))
            from ..constants import C

            self.page_url.setText(
                f'<a href="{html.escape(url)}" style="color:{C.PRI_HI};'
                f'text-decoration:none">{html.escape(url)}</a>')
            self.page_text.setPlainText("Opening…")
            self.note_text.clear()
        elif kind == "read" and url:
            self.page_text.setPlainText(str(step.get("excerpt") or ""))
            if url not in self._sources:
                self._sources.append(url)
                title = str(step.get("title") or url)
                source = QListWidgetItem(f"{len(self._sources)}. {title}\n{url}")
                source.setData(Qt.ItemDataRole.UserRole, url)
                source.setToolTip(url)
                self.sources.addItem(source)
        elif kind == "note":
            self.note_text.setPlainText(str(step.get("note") or ""))
        elif kind == "planned":
            total = int(step.get("sections") or 0)
            if total:
                self.progress.setRange(0, total)
                self.progress.setValue(0)
                self.progress.show()
        elif kind == "section_done":
            total = int(step.get("total") or 0)
            if total:
                self.progress.setRange(0, total)
                self.progress.setValue(int(step.get("done") or 0))
                self.progress.show()
        elif kind == "finished":
            self._running = False

    def _open_item(self, item: QListWidgetItem) -> None:
        url = item.data(Qt.ItemDataRole.UserRole)
        if url:
            QDesktopServices.openUrl(QUrl(str(url)))


__all__ = ["MAX_FEED_ROWS", "ResearchConsole"]
