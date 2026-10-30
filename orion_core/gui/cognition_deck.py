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
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C
from .labels import plural, title_case
from .reflections import ReflectionsPanel

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
        self.setFixedSize(200, 200)
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
        else:
            centre = "IDLE"

        p.setPen(_INK)
        p.setFont(QFont("Segoe UI", max(11, int(side * 0.16)), QFont.Weight.Bold))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, centre)


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
        layout = _panel(self, "Deep Focus")

        self.ring = FocusRing()
        overview = QHBoxLayout()
        overview.setSpacing(24)
        overview.addWidget(self.ring)
        description = QVBoxLayout()
        self.focus_title = QLabel("No focus session")
        self.focus_title.setFont(QFont("Segoe UI", 17, QFont.Weight.DemiBold))
        self.focus_title.setWordWrap(True)
        self.focus_hint = QLabel("")
        self.focus_hint.setObjectName("mutedLabel")
        self.focus_hint.setWordWrap(True)
        description.addStretch()
        description.addWidget(self.focus_title)
        description.addWidget(self.focus_hint)
        description.addStretch()
        overview.addLayout(description, 1)
        layout.addLayout(overview)

        controls = QHBoxLayout()
        self.intention = QLineEdit()
        self.intention.setPlaceholderText("What are we focusing on?")
        self.preset = QComboBox()
        for preset in ("deep", "pomodoro", "long", "short"):
            self.preset.addItem(title_case(preset), preset)     # shown as a name, used as the key
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
        self.status.setObjectName("mutedLabel")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.start_btn.clicked.connect(self._start)
        self.intention.returnPressed.connect(self._start)
        self.done_btn.clicked.connect(self._done)
        self.interrupt_btn.clicked.connect(self._interrupt)
        self.cancel_btn.clicked.connect(self._cancel)
        self.refresh()

    def _start(self) -> None:
        if self.focus is None or self.focus.active() is not None:
            return
        label = self.intention.text().strip() or "focus"
        self.focus.start(label=label, preset=str(self.preset.currentData() or "deep"))
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
            self.focus_title.setText(vm["label"])
            self.focus_hint.setText("Time for a break." if vm["break_due"] else
                                    f"{vm['planned']}-minute block · {plural(round(vm['remaining']), 'minute')} remaining")
            self.status.setText(
                f"{plural(vm['interruptions'], 'distraction')} · streak {plural(vm['streak'], 'day')} · "
                f"today {vm['today_minutes']}m")
        else:
            self.focus_title.setText("No focus session")
            self.focus_hint.setText("")
            self.status.setText(
                f"{plural(vm['streak'], 'day')} streak · {vm['today_completed']} completed today · "
                f"{plural(vm['today_minutes'], 'minute')} focused")


class StudyPanel(QFrame):
    def __init__(self, bus: OrionBus, study: Any) -> None:
        super().__init__()
        self.bus = bus
        self.study = study
        self._card: Any = None
        self._revealed = False
        layout = _panel(self, "Study & Recall")

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.prompt = QLabel("Press Review to begin.")
        self.prompt.setWordWrap(True)
        self.prompt.setObjectName("studyPrompt")
        layout.addWidget(self.prompt, 1)

        row = QHBoxLayout()
        self.add_btn = QPushButton("Add Card")
        self.review_btn = QPushButton("Review")
        self.reveal_btn = QPushButton("Reveal")
        row.addWidget(self.add_btn)
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
        self.add_btn.clicked.connect(self._add_card)
        self.reveal_btn.clicked.connect(self._reveal)
        self.refresh()

    def _add_card(self) -> None:
        if self.study is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Add a Study Card")
        dialog.resize(460, 320)
        layout = QVBoxLayout(dialog)
        front, back, deck = QLineEdit(), QTextEdit(), QLineEdit("General")
        front.setPlaceholderText("Question or prompt")
        back.setPlaceholderText("Answer")
        for label, widget in (("Question", front), ("Answer", back), ("Deck", deck)):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setEnabled(False)
        def changed() -> None:
            save.setEnabled(bool(front.text().strip() and back.toPlainText().strip()))
        front.textChanged.connect(changed)
        back.textChanged.connect(changed)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                self.study.add(front.text().strip(), back.toPlainText().strip(),
                               deck=deck.text().strip() or "General")
                self.refresh()
            except Exception as exc:
                self.prompt.setText(f"Could not save the card: {str(exc)[:180]}")

    def _next(self) -> None:
        if self.study is None:
            return
        # next_due() changes the engine's voice-review cursor. The desktop
        # keeps its own selection so opening this panel cannot redirect a
        # spoken grade to a different card.
        cards = self.study.due(limit=1)
        self._card = cards[0] if cards else None
        self._revealed = False
        self.refresh()

    def _reveal(self) -> None:
        self._revealed = True
        self.refresh()

    def _grade(self, quality: int) -> None:
        if self.study is None or self._card is None:
            return
        # The engine is shared with voice review. Its implicit current card may
        # have changed since this panel displayed its own card, so always grade
        # the card the person can actually see here.
        self.study.grade(quality, card_id=self._card.id)
        cards = self.study.due(limit=1)
        self._card = cards[0] if cards else None
        self._revealed = False
        self.refresh()

    def refresh(self) -> None:
        vm = study_view(self.study)
        if not vm.get("available"):
            self.summary.setText("Study engine unavailable.")
            for button in (self.add_btn, self.review_btn, self.reveal_btn, *self._grade_btns):
                button.setEnabled(False)
            return
        decks = ", ".join(f"{d['deck']} ({d['due']}/{d['total']})" for d in vm["decks"][:4])
        self.summary.setText(
            f"{vm['due']} due · {plural(vm['total'], 'card')} · {vm['mastered']} mastered"
            + (f" · {vm['retention']}% recall" if vm["retention"] is not None else "")
            + (f"\nDecks: {decks}" if decks else ""))
        if self._card is None:
            self.prompt.setText("Your next review is ready. Select Review to begin."
                                if vm["due"] else
                                ("All caught up. Add a card or return when your next review is due."
                                 if vm["total"] else "Build your first deck. Add a question and answer to start learning."))
        elif self._revealed:
            self.prompt.setText(f"Q: {self._card.front}\n\nA: {self._card.back}")
        else:
            self.prompt.setText(f"Q: {self._card.front}")
        reviewing = self._card is not None
        self.add_btn.setEnabled(True)
        self.review_btn.setEnabled(vm["due"] > 0)
        self.reveal_btn.setEnabled(reviewing and not self._revealed)
        for b in self._grade_btns:
            b.setEnabled(reviewing and self._revealed)


class InsightPanel(QFrame):
    def __init__(self, bus: OrionBus, study: Any, focus: Any) -> None:
        super().__init__()
        self.study = study
        self.focus = focus
        layout = _panel(self, "Insight")
        metrics = QHBoxLayout()
        self.metrics: dict[str, QLabel] = {}
        for key, label in (("minutes", "Focus · 7 Days"), ("blocks", "Completed Blocks"),
                           ("recall", "Recall"), ("mastered", "Cards Mastered")):
            card = QFrame()
            column = QVBoxLayout(card)
            heading, value = QLabel(label), QLabel("—")
            heading.setObjectName("mutedLabel")
            value.setFont(QFont("Segoe UI", 22, QFont.Weight.DemiBold))
            column.addWidget(heading)
            column.addWidget(value)
            metrics.addWidget(card, 1)
            self.metrics[key] = value
        layout.addLayout(metrics)
        self.mastery = QProgressBar()
        self.mastery.setRange(0, 100)
        self.mastery.setAccessibleName("Proportion of cards mastered")
        layout.addWidget(self.mastery)
        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.body, 1)
        self.refresh()

    def refresh(self) -> None:
        vm = insight_view(self.study, self.focus)
        s, fst = vm["study"], vm["focus_stats"]
        self.metrics["minutes"].setText(f"{fst.get('focus_minutes', 0)} min")
        self.metrics["blocks"].setText(str(fst.get("completed", 0)))
        self.metrics["recall"].setText(f"{s['retention']}%" if s["retention"] is not None else "—")
        self.metrics["mastered"].setText(f"{s['mastered']} / {s['total']}")
        self.mastery.setValue(round(100 * s["mastered"] / max(1, s["total"])))
        self.mastery.setFormat("%p% mastered" if s["total"] else "Add study cards to begin")
        lines = [
            f"{s['new']} new · {s['learning']} learning · {s['mastered']} mastered",
        ]
        if s["retention"] is not None:
            lines.append(f"<b>Recall</b> {s['retention']}%")
        if fst:
            lines.append(
                f"<b>Focus ({fst.get('days', 7)}d)</b> {plural(fst.get('completed', 0), 'block')}, "
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
    """The Cognition page: focus, study, insight and useful activity notes."""

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
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)

        self.focus_panel = FocusPanel(bus, focus)
        self.study_panel = StudyPanel(bus, study)
        self.insight_panel = InsightPanel(bus, study, focus)
        self.reflections_panel = ReflectionsPanel(bus)
        grid.addWidget(self.focus_panel, 0, 0)
        grid.addWidget(self.study_panel, 0, 1)
        # Today round a 24-hour clock: conversations, focus, study (28 Oct 2026).
        from .hud_instruments import DayRing
        self.day_ring = DayRing()
        self.day_ring.setFixedHeight(260)
        day_panel = QFrame()
        day_panel.setObjectName("panelFrame")
        day_box = QVBoxLayout(day_panel)
        day_box.setContentsMargins(14, 10, 14, 10)
        day_heading = QLabel("Today")
        day_heading.setObjectName("panelHeading")
        day_box.addWidget(day_heading)
        day_box.addWidget(self.day_ring)
        grid.addWidget(day_panel, 1, 0, 1, 2)
        grid.addWidget(self.insight_panel, 2, 0, 1, 2)
        grid.addWidget(self.reflections_panel, 3, 0, 1, 2)
        grid.setRowStretch(4, 1)
        self._day_ticks = 0
        self._day_busy = False
        scroll.setWidget(content)

        # Second-granularity refresh so the focus ring counts down — but never
        # while the page is hidden (render-loop discipline).
        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    #: How often (in refresh ticks) the day ring reads today again.
    DAY_EVERY = 60

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self.refresh_all()
        self._day_ticks += 1
        if self._day_ticks == 1 or self._day_ticks % self.DAY_EVERY == 0:
            self.refresh_day()
        self.day_ring.update()                  # the hour hand

    def refresh_day(self) -> None:
        """Read today's focus, study and conversations off the GUI thread."""
        import asyncio

        from ..constants import CONFIG_DIR
        from .hud_instruments import day_activity
        stores = {"focus_store": getattr(self.focus, "store", None),
                  "study_store": getattr(self.study, "store", None),
                  "log_path": CONFIG_DIR / "diagnostics" / "session.log"}
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.day_ring.set_activity(day_activity(**stores))
            return
        if self._day_busy:
            return
        self._day_busy = True
        from .. import background

        async def read() -> None:
            try:
                activity = await asyncio.to_thread(lambda: day_activity(**stores))
                self.day_ring.set_activity(activity)
            except RuntimeError:
                pass
            finally:
                self._day_busy = False
        background.spawn(read(), name="orion-day-ring")

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
