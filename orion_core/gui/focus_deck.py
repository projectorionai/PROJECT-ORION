"""
Focus: one big countdown, a few lengths to pick from, the task in small
steps, and ORION keeping you company if you want him to (body doubling).

Built for attention that wanders. Starting is one press; stopping never
scolds - a block stopped past its halfway point still counts as done; the
nudges are the reminder service's (a break when the block is up, a check-in
halfway through when he is body doubling), so they arrive even with the deck
closed. The timer itself is focus.FocusEngine, the same one the focus tool
uses by voice; its writes run off the GUI thread.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from ..constants import C, SPACE
from ..focus import chunk_task, resolve_cadence
from .cards import card, clear_layout, label, off_thread, run_async


@dataclass(frozen=True)
class Mode:
    key: str
    name: str
    minutes: int
    body_double: bool = False


MODES = (Mode("15", "15 min", 15), Mode("25", "25 min", 25), Mode("45", "45 min", 45),
         Mode("body", "Body Double", 25, body_double=True))
BREAK_TEXT = "Time for a short break"
HALFWAY_TEXT = "Halfway there"
BACK_TEXT = "Break's over"
#: Ticks (seconds) between reads of the running block, which the voice can change.
RESYNC_TICKS = 15


def _now() -> datetime:
    from ..time_service import TIME
    return TIME.now()


def clock_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, rest = divmod(seconds, 60)
    return f"{minutes:02d}:{rest:02d}"


def dial_state(session: Any, now: datetime, *, idle_minutes: int,
               break_until: datetime | None = None, break_seconds: float = 0.0,
               caption: str = "") -> dict[str, Any]:
    """What the dial shows, as plain data: idle, running, due (the block is
    up) or break."""
    if session is None and break_until is not None and now < break_until:
        left = (break_until - now).total_seconds()
        total = max(1.0, break_seconds)
        return {"mode": "break", "left": left, "total": total, "progress": 1 - left / total,
                "text": clock_text(left), "caption": "Break"}
    if session is None:
        total = idle_minutes * 60.0
        return {"mode": "idle", "left": total, "total": total, "progress": 0.0,
                "text": clock_text(total), "caption": caption}
    total = max(60.0, session.planned_minutes * 60.0)
    left = max(0.0, total - session.elapsed_minutes(now) * 60.0)
    if left <= 0:
        return {"mode": "due", "left": 0.0, "total": total, "progress": 1.0,
                "text": "00:00", "caption": BREAK_TEXT}
    return {"mode": "running", "left": left, "total": total, "progress": 1 - left / total,
            "text": clock_text(left), "caption": caption or session.label}


def gentle_stop_counts(session: Any, now: datetime) -> bool:
    """A block stopped past its halfway point is kept as done."""
    return session is not None and session.elapsed_minutes(now) >= session.planned_minutes / 2


class CountdownDial(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._state: dict[str, Any] = {"mode": "idle", "progress": 0.0, "text": "25:00",
                                       "caption": ""}
        self.setAccessibleName("Focus timer")

    def set_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state)
        self.setAccessibleDescription(f"{state.get('text', '')} {state.get('caption', '')}".strip())
        self.update()

    def state(self) -> dict[str, Any]:
        return dict(self._state)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        w, h = self.width(), self.height()
        side = min(w, h) - 24
        if side < 60:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ring = max(8.0, side * 0.035)
        rect = QRectF((w - side) / 2, (h - side) / 2, side, side)
        p.setPen(QPen(QColor(255, 255, 255, 16), ring))
        p.drawEllipse(rect)
        mode = self._state.get("mode", "idle")
        progress = max(0.0, min(1.0, float(self._state.get("progress", 0.0))))
        if mode != "idle":
            colour = {"due": QColor(C.PRI), "break": QColor(C.MUTED)}.get(mode, QColor(C.WHITE))
            p.setPen(QPen(colour, ring, cap=Qt.PenCapStyle.RoundCap))
            # What is left, shrinking clockwise from the top.
            p.drawArc(rect, 90 * 16, int(-360 * 16 * (1.0 - progress)) if mode != "due"
                      else -360 * 16)
        text_colour = QColor(C.WHITE if mode != "idle" else C.SILVER)
        p.setPen(text_colour)
        big = QFont("Cascadia Mono", max(18, int(side * 0.17)))
        big.setStyleHint(QFont.StyleHint.Monospace)
        big.setWeight(QFont.Weight.Light)
        p.setFont(big)
        p.drawText(QRectF(rect.left(), rect.top(), rect.width(), rect.height() * 0.56),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                   str(self._state.get("text", "")))
        caption = str(self._state.get("caption", "") or "")
        if caption:
            p.setPen(QColor(C.MUTED))
            p.setFont(QFont("Segoe UI", max(9, int(side * 0.038))))
            inner = side * 0.62
            p.drawText(QRectF(rect.center().x() - inner / 2, rect.top() + rect.height() * 0.6,
                              inner, rect.height() * 0.2),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                       | Qt.TextFlag.TextWordWrap, caption)
        p.end()


class FocusDeckView(QWidget):
    REFRESH_MS = 1000

    def __init__(self, bus: Any, focus: Any = None, *, reminders: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus, self.focus, self.reminders = bus, focus, reminders
        self.mode = MODES[1]
        self.session: Any = None
        self._break_until: datetime | None = None
        self._break_seconds = 0.0
        self._nudges: list[int] = []
        self._busy = False
        self._ticks = 0
        self._plan_loaded = False
        self.task = ""
        self.steps: list[tuple[str, bool]] = []
        self.face: Any = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        grid = QGridLayout(body)
        grid.setContentsMargins(SPACE.SM, SPACE.SM, SPACE.SM, SPACE.SM)
        grid.setSpacing(SPACE.LG)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)

        timer_card, timer_box = card()
        timer_box.setContentsMargins(SPACE.LG, SPACE.LG, SPACE.LG, SPACE.LG)
        presets = QHBoxLayout()
        presets.setSpacing(SPACE.SM)
        presets.addStretch(1)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode in MODES:
            chip = QPushButton(mode.name)
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setChecked(mode is self.mode)
            chip.clicked.connect(lambda _c=False, m=mode: self.choose(m))
            self.mode_group.addButton(chip)
            self.mode_buttons[mode.key] = chip
            presets.addWidget(chip)
        presets.addStretch(1)
        timer_box.addLayout(presets)
        self.dial = CountdownDial()
        timer_box.addWidget(self.dial, 1)

        controls = QHBoxLayout()
        controls.setSpacing(SPACE.SM)
        controls.addStretch(1)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.start)
        self.done_button = QPushButton("Done")
        self.done_button.setObjectName("primaryButton")
        self.done_button.clicked.connect(self.done)
        self.break_button = QPushButton("Take a Break")
        self.break_button.setObjectName("primaryButton")
        self.break_button.clicked.connect(self.take_break)
        self.away_button = QPushButton("Distracted")
        self.away_button.setObjectName("ghostButton")
        self.away_button.clicked.connect(self.distracted)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("ghostButton")
        self.stop_button.clicked.connect(self.stop)
        for button in (self.start_button, self.done_button, self.break_button,
                       self.away_button, self.stop_button):
            button.setMinimumWidth(96)
            controls.addWidget(button)
        controls.addStretch(1)
        timer_box.addLayout(controls)

        self.company = QHBoxLayout()
        self.company.setSpacing(SPACE.MD)
        self.company.addStretch(1)
        self.company_text = label("I'm here. I'll check in halfway.", "mutedLabel")
        self.company.addWidget(self.company_text)
        self.company.addStretch(1)
        self.company_text.setVisible(False)
        timer_box.addLayout(self.company)
        self.status = label("", "mutedLabel", wrap=True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        timer_box.addWidget(self.status)
        grid.addWidget(timer_card, 0, 0, 2, 1)

        steps_card, steps_box = card()
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("What are you working on?")
        self.task_input.setAccessibleName("Task")
        self.task_input.editingFinished.connect(self._task_named)
        steps_box.addWidget(self.task_input)
        self.step_list = QVBoxLayout()
        self.step_list.setSpacing(0)
        steps_box.addLayout(self.step_list)
        self.step_input = QLineEdit()
        self.step_input.setPlaceholderText("Add a step")
        self.step_input.setAccessibleName("New step")
        self.step_input.returnPressed.connect(self.add_steps)
        steps_box.addWidget(self.step_input)
        tidy = QHBoxLayout()
        tidy.addStretch(1)
        self.clear_button = QPushButton("Clear Done")
        self.clear_button.setObjectName("ghostButton")
        self.clear_button.clicked.connect(self.clear_done)
        tidy.addWidget(self.clear_button)
        steps_box.addLayout(tidy)
        grid.addWidget(steps_card, 0, 1)

        stats_card, stats_box = card()
        row = QHBoxLayout()
        row.setSpacing(SPACE.LG)
        self.stats: dict[str, Any] = {}
        for key, caption in (("minutes", "Today"), ("blocks", "Blocks"), ("streak", "Day Streak")):
            column = QVBoxLayout()
            value = label("0", "readout")
            column.addWidget(value)
            column.addWidget(label(caption, "readoutKey"))
            row.addLayout(column)
            self.stats[key] = value
        row.addStretch(1)
        stats_box.addLayout(row)
        grid.addWidget(stats_card, 1, 1, Qt.AlignmentFlag.AlignTop)
        grid.setRowStretch(1, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._draw_steps()
        self.redraw()

    # ── state ────────────────────────────────────────────────────────────────

    def caption(self) -> str:
        return next((text for text, done in self.steps if not done), self.task)

    def redraw(self, now: datetime | None = None) -> None:
        now = now or _now()
        if self._break_until is not None and now >= self._break_until:
            self._break_until = None
        state = dial_state(self.session, now, idle_minutes=self.mode.minutes,
                           break_until=self._break_until, break_seconds=self._break_seconds,
                           caption=self.caption())
        self.dial.set_state(state)
        mode = state["mode"]
        available = self.focus is not None
        self.start_button.setVisible(mode in {"idle", "break"})
        self.start_button.setEnabled(available and not self._busy)
        self.done_button.setVisible(mode == "running")
        self.away_button.setVisible(mode == "running")
        self.stop_button.setVisible(mode in {"running", "due"})
        self.break_button.setVisible(mode == "due")
        for button in (self.done_button, self.away_button, self.stop_button, self.break_button):
            button.setEnabled(not self._busy)
        for chip in self.mode_buttons.values():
            chip.setEnabled(mode in {"idle", "break"})
        company = self.mode.body_double and mode in {"running", "due"}
        self._show_company(company)

    def _show_company(self, on: bool) -> None:
        if on and self.face is None:
            try:
                from .screen_face import ScreenFacePanel
                self.face = ScreenFacePanel(fps=20)
                self.face.setFixedSize(132, 92)
                self.company.insertWidget(1, self.face)
            except Exception:
                self.face = None
        if self.face is not None:
            self.face.setVisible(on)
        self.company_text.setVisible(on)

    def _tick(self) -> None:
        if not self.isVisible():
            return
        self._ticks += 1
        if self._ticks % RESYNC_TICKS == 0:
            self.load()
        self.redraw()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self.load()

    def load(self) -> None:
        """Read the running block, today's numbers and the saved steps."""
        if self.focus is None or self._busy or sip.isdeleted(self):
            return
        focus, want_plan = self.focus, not self._plan_loaded

        def read() -> dict[str, Any]:
            today = focus.today()
            out = {"session": focus.active(), "minutes": today["focus_minutes"],
                   "blocks": today["completed"], "streak": focus.streak()}
            if want_plan:
                out["plan"] = focus.store.load_plan()
            return out
        off_thread(self, read, done=self._loaded, name="orion-focus-page")

    def _loaded(self, data: dict[str, Any]) -> None:
        if self._busy:
            return
        self.session = data.get("session")
        if self.session is not None:
            self._break_until = None
        self.stats["minutes"].setText(f"{data.get('minutes', 0)}m")
        self.stats["blocks"].setText(str(data.get("blocks", 0)))
        self.stats["streak"].setText(str(data.get("streak", 0)))
        if "plan" in data and not self._plan_loaded:
            self._plan_loaded = True
            self.task, self.steps = data["plan"][0], list(data["plan"][1])
            self.task_input.setText(self.task)
            self._draw_steps()
        self.redraw()

    # ── the timer ────────────────────────────────────────────────────────────

    def choose(self, mode: Mode) -> None:
        self.mode = mode
        self.mode_buttons[mode.key].setChecked(True)
        self.redraw()

    def _act(self, work: Any, then: Any) -> None:
        self._busy = True
        self.redraw()

        def finished(result: Any) -> None:
            self._busy = False
            then(result)
            self.load()

        def failed(exc: BaseException) -> None:
            self._busy = False
            self.status.setText(str(exc)[:160])
            self.redraw()
        off_thread(self, work, done=finished, failed=failed, name="orion-focus-action")

    def start(self) -> None:
        if self.focus is None or self._busy:
            return
        self._break_until = None
        mode, title = self.mode, (self.caption() or self.task or "Focus")
        work, rest = resolve_cadence(None, mode.minutes)
        focus = self.focus

        def begin() -> Any:
            return focus.start(label=title, minutes=work, break_minutes=rest,
                               kind="body double" if mode.body_double else "deep")

        def started(session: Any) -> None:
            self.session = session
            self.status.setText("")
            nudges = [(BREAK_TEXT, work)]
            if mode.body_double and work >= 10:
                nudges.append((HALFWAY_TEXT, work / 2))
            self._schedule(nudges)
        self._act(begin, started)

    def done(self) -> None:
        if self.focus is None or self._busy:
            return
        focus = self.focus
        self._cancel_nudges()
        self._act(focus.complete, lambda s: self._ended(s, kept=True))

    def stop(self) -> None:
        """Stop without fuss: past halfway it still counts."""
        if self.focus is None or self._busy:
            return
        focus, keep = self.focus, gentle_stop_counts(self.session, _now())
        self._cancel_nudges()
        self._act(focus.complete if keep else focus.cancel, lambda s: self._ended(s, kept=keep))

    def _ended(self, session: Any, *, kept: bool) -> None:
        self.session = None
        minutes = int(round(session.elapsed_minutes())) if session is not None else 0
        self.status.setText(f"{minutes} min done." if kept else f"Stopped after {minutes} min.")

    def take_break(self) -> None:
        if self.focus is None or self._busy:
            return
        rest = getattr(self.session, "break_minutes", 5) or 5
        focus = self.focus
        self._cancel_nudges()

        def on_break(session: Any) -> None:
            self.session = None
            self._break_seconds = rest * 60.0
            self._break_until = _now() + timedelta(minutes=rest)
            self.status.setText("")
            self._schedule([(BACK_TEXT, rest)])
        self._act(focus.complete, on_break)

    def distracted(self) -> None:
        if self.focus is None or self._busy:
            return
        self._act(self.focus.interrupt, lambda _s: self.status.setText("Noted."))

    # ── nudges, through the reminder service ─────────────────────────────────

    def _schedule(self, nudges: list[tuple[str, float]]) -> None:
        reminders = self.reminders
        if reminders is None or not nudges:
            return

        def add() -> list[int]:
            ids = []
            for text, minutes in nudges:
                result = reminders.add(text=text, minutes=minutes)
                if getattr(result, "ok", False):
                    found = [r.id for r in reminders.active() if r.text == text]
                    if found:
                        ids.append(max(found))
            return ids
        off_thread(self, add, done=lambda ids: self._nudges.extend(ids),
                   name="orion-focus-nudges")

    def _cancel_nudges(self) -> None:
        reminders, ids = self.reminders, list(self._nudges)
        self._nudges.clear()
        if reminders is None or not ids:
            return

        def cancel() -> None:
            live = {r.id for r in reminders.active()}
            for rid in ids:
                if rid in live:
                    reminders.cancel(rid)
        off_thread(self, cancel, name="orion-focus-nudges")

    # ── the task in steps ────────────────────────────────────────────────────

    def _task_named(self) -> None:
        text = self.task_input.text().strip()
        if text != self.task:
            self.task = text
            self._save_plan()
            self.redraw()

    def add_steps(self) -> None:
        added = chunk_task(self.step_input.text())
        if not added:
            return
        known = {text.casefold() for text, _done in self.steps}
        self.steps += [(text, False) for text in added if text.casefold() not in known]
        self.step_input.clear()
        self._draw_steps()
        self._save_plan()
        self.redraw()

    def set_step_done(self, index: int, done: bool) -> None:
        if 0 <= index < len(self.steps):
            self.steps[index] = (self.steps[index][0], bool(done))
            # Rebuilt on the next turn: the box that was ticked is still
            # inside its own toggled signal here.
            QTimer.singleShot(0, self._draw_steps)
            self._save_plan()
            self.redraw()

    def clear_done(self) -> None:
        self.steps = [step for step in self.steps if not step[1]]
        self._draw_steps()
        self._save_plan()
        self.redraw()

    def _draw_steps(self) -> None:
        clear_layout(self.step_list)
        current = next((i for i, (_t, done) in enumerate(self.steps) if not done), None)
        for index, (text, done) in enumerate(self.steps):
            box = QCheckBox(text)
            box.setObjectName("step")
            box.setChecked(done)
            font = box.font()
            font.setStrikeOut(done)
            font.setWeight(QFont.Weight.DemiBold if index == current else QFont.Weight.Normal)
            box.setFont(font)
            box.toggled.connect(lambda on, i=index: self.set_step_done(i, on))
            self.step_list.addWidget(box)
        self.clear_button.setVisible(any(done for _t, done in self.steps))

    def _save_plan(self) -> None:
        store = getattr(self.focus, "store", None)
        if store is None or not hasattr(store, "save_plan"):
            return
        task, steps = self.task, list(self.steps)
        off_thread(self, lambda: store.save_plan(task, steps), name="orion-focus-plan")


__all__ = ["CountdownDial", "FocusDeckView", "MODES", "Mode", "clock_text", "dial_state",
           "gentle_stop_counts"]
