"""
COGNITION — the Command Deck page for the learning loop (Mark XXVI, Phase 1).

Marks XXIV and XXV built the spaced-repetition ``study`` system and the deep-work
``focus`` timer as backend + voice tools with no on-screen home. This page is that
home: one surface where a focus block runs on a live ring, due flashcards are
reviewed and graded, and mastery/attention are shown at a glance.

Design notes honouring the house rules:
  * The page holds the SAME lazy engine instances the tools use
    (``dispatcher.study`` / ``dispatcher.focus``) — the qasync loop unifies the
    GUI and event-loop threads, so a shared instance is safe and there is exactly
    one source of truth.
  * Everything drawn comes from a *pure* view-model reducer (``focus_view`` /
    ``study_view`` / ``insight_view``) so the display logic is unit-testable with
    no Qt at all.
  * A visibility-gated ``QTimer`` drives the ring; a hidden page never repaints
    (the established render-loop discipline that stops the deck taxing the audio
    deadline).
  * The focus ring's ``paintEvent`` guards degenerate sizes, per the Mark XXIII
    black-flicker lesson (a QPainter widget that fills then faults leaves a bare
    frame on screen).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from PyQt6.QtCore import Qt, QRectF, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C

# Custom-painted, so no stylesheet reached it and it grew its own cyan, amber
# and blue-grey. Taken from C now: the ring is a status reading, so it uses
# the same nominal/degraded pair as everything else.
_ACCENT = QColor(C.SILVER)      # nominal — recedes
_AMBER = QColor(C.WARN)         # a break is due — steps forward
_DIM = QColor(C.ACCENT_DEEP)    # the unfilled track
_INK = QColor(C.WHITE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── pure view-model reducers (no Qt — fully unit-testable) ────────────────────

def focus_view(focus: Any, now: datetime | None = None) -> dict[str, Any]:
    """Everything the focus panel needs, as plain data."""
    now = now or _now()
    if focus is None:
        return {"available": False, "active": False, "streak": 0,
                "today_minutes": 0, "today_completed": 0}
    session = focus.active(now)
    today = focus.today(now)
    vm: dict[str, Any] = {
        "available": True,
        "active": session is not None,
        "streak": focus.streak(now),
        "today_minutes": today["focus_minutes"],
        "today_completed": today["completed"],
    }
    if session is not None:
        planned = max(1, session.planned_minutes)
        elapsed = session.elapsed_minutes(now)
        vm.update(
            label=session.label,
            planned=session.planned_minutes,
            break_minutes=session.break_minutes,
            elapsed=elapsed,
            remaining=session.remaining_minutes(now),
            break_due=session.is_break_due(now),
            interruptions=session.interruptions,
            progress=max(0.0, min(1.0, elapsed / planned)),
        )
    return vm


def study_view(study: Any, now: datetime | None = None) -> dict[str, Any]:
    now = now or _now()
    if study is None:
        return {"available": False, "due": 0, "decks": [], "total": 0,
                "new": 0, "learning": 0, "mastered": 0, "retention": None}
    stats = study.stats(now=now)
    return {
        "available": True,
        "due": len(study.due(limit=10_000, now=now)),
        "decks": study.decks(),
        "total": stats["total"],
        "new": stats["new"],
        "learning": stats["learning"],
        "mastered": stats["mastered"],
        "retention": stats["retention"],
    }


def insight_view(study: Any, focus: Any, now: datetime | None = None) -> dict[str, Any]:
    now = now or _now()
    return {
        "focus": focus_view(focus, now),
        "study": study_view(study, now),
        "focus_stats": (focus.stats(now=now) if focus is not None else {}),
    }


# ── the focus ring ────────────────────────────────────────────────────────────

class FocusRing(QWidget):
    """A progress ring for the running block; amber once the break is due."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(150, 150)
        self._vm: dict[str, Any] = {"active": False}

    def set_state(self, vm: dict[str, Any]) -> None:
        self._vm = dict(vm or {})
        self.update()

    def paintEvent(self, _event: Any) -> None:
        w, h = self.width(), self.height()
        if w < 8 or h < 8:            # degenerate size — skip (black-flicker guard)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(w, h) - 14
        rect = QRectF((w - side) / 2, (h - side) / 2, side, side)

        p.setPen(QPen(_DIM, 10))
        p.drawArc(rect, 0, 360 * 16)

        vm = self._vm
        if vm.get("active"):
            colour = _AMBER if vm.get("break_due") else _ACCENT
            sweep = int(-360 * 16 * float(vm.get("progress", 0.0)))
            p.setPen(QPen(colour, 10, cap=Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 90 * 16, sweep)
            centre = (f"{int(round(vm.get('remaining', 0)))}m"
                      if not vm.get("break_due") else "BREAK")
            sub = str(vm.get("label", ""))[:22]
        else:
            centre, sub = "IDLE", "no block running"

        p.setPen(_INK)
        p.setFont(QFont("Segoe UI", max(11, int(side * 0.16)), QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, centre)
        p.setPen(QColor(C.MUTED))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(0, rect.bottom() - side * 0.16, w, side * 0.16),
                   Qt.AlignmentFlag.AlignHCenter, sub)


def _panel(frame: QFrame, title: str) -> QVBoxLayout:
    frame.setObjectName("panelFrame")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(6)
    heading = QLabel(title)
    heading.setObjectName("panelHeading")
    layout.addWidget(heading)
    return layout


# ── panels ────────────────────────────────────────────────────────────────────

class FocusPanel(QFrame):
    def __init__(self, bus: OrionBus, focus: Any) -> None:
        super().__init__()
        self.bus = bus
        self.focus = focus
        layout = _panel(self, "DEEP FOCUS")

        self.ring = FocusRing()
        layout.addWidget(self.ring, 1)

        controls = QHBoxLayout()
        self.intention = QLineEdit()
        self.intention.setPlaceholderText("what are we focusing on?")
        self.preset = QComboBox()
        self.preset.addItems(["deep", "pomodoro", "long", "short"])
        controls.addWidget(self.intention, 1)
        controls.addWidget(self.preset, 0)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.done_btn = QPushButton("Done")
        self.interrupt_btn = QPushButton("Distracted")
        self.cancel_btn = QPushButton("Cancel")
        for b in (self.start_btn, self.done_btn, self.interrupt_btn, self.cancel_btn):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.start_btn.clicked.connect(self._start)
        self.done_btn.clicked.connect(self._done)
        self.interrupt_btn.clicked.connect(self._interrupt)
        self.cancel_btn.clicked.connect(self._cancel)
        self.refresh()

    def _start(self) -> None:
        if self.focus is None:
            return
        label = self.intention.text().strip() or "focus"
        self.focus.start(label=label, preset=self.preset.currentText())
        self.intention.clear()
        self.refresh()

    def _done(self) -> None:
        if self.focus is not None:
            self.focus.complete()
            self.refresh()

    def _interrupt(self) -> None:
        if self.focus is not None:
            self.focus.interrupt()
            self.refresh()

    def _cancel(self) -> None:
        if self.focus is not None:
            self.focus.cancel()
            self.refresh()

    def refresh(self) -> None:
        vm = focus_view(self.focus)
        self.ring.set_state(vm)
        active = vm.get("active", False)
        self.start_btn.setEnabled(vm.get("available", False) and not active)
        for b in (self.done_btn, self.interrupt_btn, self.cancel_btn):
            b.setEnabled(active)
        if not vm.get("available"):
            self.status.setText("Focus engine unavailable.")
        elif active:
            self.status.setText(
                f"{vm['interruptions']} distraction(s) · streak {vm['streak']} day(s) · "
                f"today {vm['today_minutes']}m")
        else:
            self.status.setText(
                f"Streak {vm['streak']} day(s) · today {vm['today_completed']} block(s), "
                f"{vm['today_minutes']}m focused")


class StudyPanel(QFrame):
    def __init__(self, bus: OrionBus, study: Any) -> None:
        super().__init__()
        self.bus = bus
        self.study = study
        self._card: Any = None
        self._revealed = False
        layout = _panel(self, "STUDY & RECALL")

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.prompt = QLabel("Press Review to begin.")
        self.prompt.setWordWrap(True)
        self.prompt.setObjectName("studyPrompt")
        layout.addWidget(self.prompt, 1)

        row = QHBoxLayout()
        self.review_btn = QPushButton("Review")
        self.reveal_btn = QPushButton("Reveal")
        row.addWidget(self.review_btn)
        row.addWidget(self.reveal_btn)
        layout.addLayout(row)

        grades = QHBoxLayout()
        self._grade_btns = []
        for text, quality in (("Again", 1), ("Hard", 3), ("Good", 4), ("Easy", 5)):
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, q=quality: self._grade(q))
            grades.addWidget(b)
            self._grade_btns.append(b)
        layout.addLayout(grades)

        self.review_btn.clicked.connect(self._next)
        self.reveal_btn.clicked.connect(self._reveal)
        self.refresh()

    def _next(self) -> None:
        if self.study is None:
            return
        self._card = self.study.next_due()
        self._revealed = False
        self.refresh()

    def _reveal(self) -> None:
        self._revealed = True
        self.refresh()

    def _grade(self, quality: int) -> None:
        if self.study is None or self._card is None:
            return
        self.study.grade(quality)
        self._card = self.study.next_due()
        self._revealed = False
        self.refresh()

    def refresh(self) -> None:
        vm = study_view(self.study)
        if not vm.get("available"):
            self.summary.setText("Study engine unavailable.")
            return
        decks = ", ".join(f"{d['deck']} ({d['due']}/{d['total']})" for d in vm["decks"][:4])
        self.summary.setText(
            f"{vm['due']} due · {vm['total']} cards · {vm['mastered']} mastered"
            + (f" · {vm['retention']}% recall" if vm["retention"] is not None else "")
            + (f"\nDecks: {decks}" if decks else ""))
        if self._card is None:
            self.prompt.setText("Press Review to begin."
                                if vm["due"] else "Nothing due — nicely ahead.")
        elif self._revealed:
            self.prompt.setText(f"Q: {self._card.front}\n\nA: {self._card.back}")
        else:
            self.prompt.setText(f"Q: {self._card.front}")
        reviewing = self._card is not None
        self.review_btn.setEnabled(vm["due"] > 0 or not reviewing)
        self.reveal_btn.setEnabled(reviewing and not self._revealed)
        for b in self._grade_btns:
            b.setEnabled(reviewing and self._revealed)


class InsightPanel(QFrame):
    def __init__(self, bus: OrionBus, study: Any, focus: Any) -> None:
        super().__init__()
        self.study = study
        self.focus = focus
        layout = _panel(self, "INSIGHT")
        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.body, 1)
        self.refresh()

    def refresh(self) -> None:
        vm = insight_view(self.study, self.focus)
        s, fst = vm["study"], vm["focus_stats"]
        bar = self._bar(s["new"], s["learning"], s["mastered"])
        lines = [
            f"<b>Mastery</b> {bar}",
            f"new {s['new']} · learning {s['learning']} · mastered {s['mastered']}",
        ]
        if s["retention"] is not None:
            lines.append(f"<b>Recall</b> {s['retention']}%")
        if fst:
            lines.append(
                f"<b>Focus ({fst.get('days', 7)}d)</b> {fst.get('completed', 0)} blocks, "
                f"{fst.get('focus_minutes', 0)}m, {fst.get('completion_rate', 0)}% done, "
                f"streak {fst.get('streak', 0)}")
        self.body.setText("<br>".join(lines))

    @staticmethod
    def _bar(new: int, learning: int, mastered: int, width: int = 20) -> str:
        total = max(1, new + learning + mastered)
        m = round(width * mastered / total)
        l = round(width * learning / total)
        n = width - m - l
        return "█" * m + "▓" * l + "░" * n


# ── the page ──────────────────────────────────────────────────────────────────

class CognitionDeckView(QWidget):
    """The COGNITION Command-Deck page: focus, study and insight in one surface."""

    REFRESH_MS = 1000

    def __init__(self, bus: OrionBus, *, study: Any = None, focus: Any = None) -> None:
        super().__init__()
        self.bus = bus
        self.study = study
        self.focus = focus

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

        self.focus_panel = FocusPanel(bus, focus)
        self.study_panel = StudyPanel(bus, study)
        self.insight_panel = InsightPanel(bus, study, focus)
        grid.addWidget(self.focus_panel, 0, 0)
        grid.addWidget(self.study_panel, 0, 1)
        grid.addWidget(self.insight_panel, 1, 0, 1, 2)
        grid.setRowStretch(2, 1)
        scroll.setWidget(content)

        # Second-granularity refresh so the focus ring counts down — but never
        # while the page is hidden (render-loop discipline).
        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self.refresh_all()

    def refresh_all(self) -> None:
        self.focus_panel.refresh()
        self.study_panel.refresh()
        self.insight_panel.refresh()

    def __probe_state__(self) -> dict[str, Any]:
        """A headless inspection hook for the render test: the exact data the
        page is displaying, without scraping widgets."""
        return insight_view(self.study, self.focus)


__all__ = [
    "CognitionDeckView", "FocusRing", "FocusPanel", "StudyPanel", "InsightPanel",
    "focus_view", "study_view", "insight_view",
]
