"""
Clipboard intelligence — copy something, and ORION offers to do the obvious.

Copy a paragraph of German, a stack trace, a clause of a contract, and the
thing you want next is nearly always one of four: translate it, shorten it,
explain it, or fix it. Asking for that means switching window, composing a
sentence and pasting — enough friction that most of the time nobody bothers.
This puts the four where the text already is.

The part that needed the most care
----------------------------------
A clipboard watcher is a privacy hazard wearing a convenience hat. People copy
passwords out of password managers, API keys out of dashboards, card numbers
off statements. A panel that cheerfully offered to send any of those to a cloud
model would be the single worst thing in this codebase.

So:

  * **Nothing is ever sent automatically.** The panel only appears; the text
    goes nowhere until a specific button is pressed. Appearing is free,
    sending is not.
  * **Credentials are never offered at all.** Every capture is put through
    ``SpillageGuard`` first — the same recogniser that stops ORION leaking a
    key outbound — and anything holding a password, an API key, a private key
    block or a card number is skipped silently. Not warned about: silently. A
    panel that says "I noticed your password" is its own kind of alarming.
  * **Off by default**, because a watcher nobody asked for is a watcher.
  * **Local-looking content stays local where it can.** Nothing here inspects
    the text beyond the guard; the panel holds it and hands it over only on a
    press.

It also stays out of the way: short text is ignored (a copied filename needs no
help), repeats are ignored, and the panel dismisses itself.

Styled from ORION's own tokens rather than carrying a palette of its own.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor, QFont, QGuiApplication
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .style import C, rgba
from .widgets import describe_control

#: Shorter than this and there is nothing to do with it — a filename, a number,
#: a single word. Offering help there is noise.
MIN_LENGTH = 48

#: Longest excerpt shown on the panel itself. The full text is what gets sent.
PREVIEW_CHARS = 90

#: How long the panel waits before dismissing itself.
VISIBLE_SECONDS = 9.0

#: Ignore a repeat of the same text within this window. Copying twice is
#: common and does not mean "ask me again".
REPEAT_SECONDS = 25.0

#: The four things people actually want, and the instruction each one sends.
ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Translate", "translate",
     "Translate this into my language. Give only the translation."),
    ("Summarise", "summarise",
     "Summarise this in a few sentences. Lead with the point."),
    ("Explain", "explain",
     "Explain this plainly — what it is, and what matters about it."),
    ("Fix", "fix",
     "Correct this: grammar, spelling and clarity. Return the corrected text "
     "and nothing else."),
)


class ClipboardPanel(QFrame):
    """A small floating offer, shown near the cursor after a copy.

    ``requested`` carries (instruction, text) when the user presses one of the
    four. Nothing is emitted until they do.
    """

    requested = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setObjectName("clipboardPanel")
        self.setStyleSheet(
            # It floats over whatever you were reading, so it is a raised
            # surface — not a panel outlined in crimson. The accent means
            # state, focus or danger; "here is your clipboard" is none of them.
            f"#clipboardPanel {{ background: {rgba(C.PANEL_HI, 0.94)};"
            f" border: 1px solid {rgba(C.WHITE, 0.055)};"
            f" border-top: 1px solid {rgba(C.WHITE, 0.10)};"
            f" border-radius: 14px; }}"
            f"QLabel {{ color: {C.WHITE}; }}"
            f"QLabel#excerpt {{ color: {C.MUTED}; font-size: 11px; }}"
            "QPushButton {"
            f"  background: {rgba(C.WHITE, 0.06)}; color: {C.SILVER};"
            f"  border: 1px solid {rgba(C.WHITE, 0.055)}; border-radius: 10px;"
            "   padding: 5px 11px;"
            "}"
            f"QPushButton:hover {{ background: {rgba(C.WHITE, 0.12)};"
            f" color: {C.WHITE}; border-color: {rgba(C.WHITE, 0.18)}; }}"
            f"QPushButton#dismiss {{ color: {C.FAINT}; border-color: transparent; }}"
            f"QPushButton#dismiss:hover {{ color: {C.WHITE}; background: transparent; }}"
        )

        self._text = ""

        title = QLabel("Copied — shall I…")
        title.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        self._excerpt = QLabel("")
        self._excerpt.setObjectName("excerpt")
        self._excerpt.setWordWrap(True)
        self._excerpt.setMaximumWidth(330)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        for label, key, instruction in ACTIONS:
            button = QPushButton(label)
            describe_control(button, f"{label} the copied text",
                             f"Sends what you just copied to ORION: "
                             f"{instruction}")
            button.clicked.connect(
                lambda _checked=False, i=instruction: self._request(i))
            buttons.addWidget(button)
        buttons.addStretch(1)
        dismiss = QPushButton("✕")
        dismiss.setObjectName("dismiss")
        describe_control(dismiss, "Dismiss", "Close without sending anything.")
        dismiss.clicked.connect(self.hide)
        buttons.addWidget(dismiss)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(self._excerpt)
        layout.addLayout(buttons)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.hide)

    # ── showing ──────────────────────────────────────────────────────────────

    def offer(self, text: str, at: QPoint | None = None) -> None:
        """Show the panel for *text*. The text is held, not sent."""
        self._text = text
        excerpt = " ".join(text.split())
        if len(excerpt) > PREVIEW_CHARS:
            excerpt = excerpt[:PREVIEW_CHARS].rstrip() + "…"
        self._excerpt.setText(excerpt)
        self.adjustSize()
        self.move(at or (QCursor.pos() + QPoint(16, 18)))
        self.show()
        self.raise_()
        self._dismiss_timer.start(int(VISIBLE_SECONDS * 1000))

    def _request(self, instruction: str) -> None:
        text, self._text = self._text, ""
        self.hide()
        if text:
            self.requested.emit(instruction, text)

    def pending_text(self) -> str:
        return self._text


class ClipboardWatcher:
    """Watches the clipboard and decides whether to offer anything.

    Deliberately not a QObject with its own lifecycle: it attaches to an
    existing clipboard and panel, so it can be exercised without either.
    """

    def __init__(self, panel: Any = None, *, guard: Any = None,
                 clock: Callable[[], float] = time.monotonic,
                 enabled: bool = False) -> None:
        self.panel = panel
        self.enabled = bool(enabled)
        self._clock = clock
        self._last_text = ""
        self._last_at = 0.0
        if guard is not None:
            self._guard = guard
        else:
            try:
                from ..spillage_guard import SpillageGuard

                self._guard = SpillageGuard()
            except Exception:
                self._guard = None

    def attach(self, clipboard: Any = None) -> bool:
        """Subscribe to clipboard changes. Returns whether it took."""
        try:
            clipboard = clipboard or QGuiApplication.clipboard()
            clipboard.dataChanged.connect(
                lambda: self.consider(clipboard.text()))
            return True
        except Exception:
            return False

    # ── the decision ─────────────────────────────────────────────────────────

    def should_offer(self, text: str) -> bool:
        """Whether this capture is worth interrupting anyone for.

        Every reason to say no is cheap; the one reason to say yes has to earn
        it. In particular a credential is never offered — not warned about,
        just skipped, because "I noticed your password" is its own kind of
        alarming.
        """
        if not self.enabled:
            return False
        raw = (text or "").strip()
        if len(raw) < MIN_LENGTH:
            return False                    # a filename needs no help
        now = self._clock()
        if raw == self._last_text and (now - self._last_at) < REPEAT_SECONDS:
            return False                    # copying twice is not a new ask
        # No recogniser means no offers. This gate FAILS CLOSED on purpose: an
        # earlier version skipped the check when the guard was missing, which
        # meant the one configuration with no protection at all was also the
        # one that offered everything, credentials included. A clipboard
        # feature that silently degrades into a credential leak is not a
        # feature that should degrade silently.
        if self._guard is None:
            return False
        try:
            if not self._guard.is_safe(raw):
                return False                # a secret. Silently.
        except Exception:
            return False                    # cannot tell: assume not
        return True

    def consider(self, text: str) -> bool:
        """Offer the panel if the capture warrants it. Returns whether shown."""
        if not self.should_offer(text):
            return False
        raw = text.strip()
        self._last_text = raw
        self._last_at = self._clock()
        if self.panel is not None:
            try:
                self.panel.offer(raw)
            except Exception:
                return False
        return True


__all__ = ["ACTIONS", "MIN_LENGTH", "PREVIEW_CHARS", "REPEAT_SECONDS",
           "VISIBLE_SECONDS", "ClipboardPanel", "ClipboardWatcher"]
