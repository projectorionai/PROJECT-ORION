"""
Fitness: workouts logged in a few taps, the week they add up to, the streak,
and the health goals from goals.py. A workout can also count towards those
goals - the goals tool reads it the way it reads "I ran 5 km" said aloud.
Storage is fitness.FitnessStore; every read and write runs off the GUI thread.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .. import fitness
from ..constants import C, SPACE
from .cards import card, clear_layout, label, off_thread, run_async
from .line_icons import apply as apply_icon

#: Shown as chips, in this order.
KIND_NAMES = {"run": "Run", "walk": "Walk", "cycle": "Cycle", "strength": "Strength",
              "yoga": "Yoga", "swim": "Swim", "sport": "Sport", "other": "Other"}


def goal_sentence(kind: str, minutes: float, distance_km: float | None) -> str:
    """The workout as the owner would say it to the goals tool."""
    mins = int(round(minutes))
    if distance_km:
        verb = {"run": "ran", "walk": "walked", "cycle": "cycled", "swim": "swam"}.get(kind)
        if verb:
            return f"I {verb} {distance_km:g} km"
    if kind == "yoga":
        return f"I did {mins} minutes of yoga"
    return f"I did a {mins} minute {KIND_NAMES.get(kind, 'workout').lower()} workout"


def when_text(stamp: str, now: datetime) -> str:
    try:
        at = datetime.fromisoformat(stamp).astimezone(now.tzinfo)
    except (TypeError, ValueError):
        return ""
    days = (now.date() - at.date()).days
    if days == 0:
        return f"Today, {at:%H:%M}"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return at.strftime("%A")
    return f"{at.day} {at:%b}"


def amount_text(workout: fitness.Workout) -> str:
    parts = [f"{int(round(workout.minutes))} min"]
    if workout.distance_km:
        parts.append(f"{workout.distance_km:g} km")
    return "  ·  ".join(parts)


class WeekBars(QWidget):
    """Minutes per day, Monday to Sunday; today in the accent."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(190)
        self._days: list[tuple[date, float]] = []
        self._today: date | None = None
        self.setAccessibleName("Minutes trained each day this week")

    def set_days(self, days: list[tuple[date, float]], today: date) -> None:
        self._days, self._today = list(days), today
        self.update()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        w, h = self.width(), self.height()
        if w < 40 or h < 60 or not self._days:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        top, bottom = 22.0, 24.0
        peak = max(30.0, max(m for _d, m in self._days))
        slot = w / len(self._days)
        bar = min(44.0, slot * 0.5)
        small = QFont("Segoe UI", 8)
        for i, (day, minutes) in enumerate(self._days):
            cx = slot * i + slot / 2
            height = max(3.0, (h - top - bottom) * minutes / peak)
            rect = QRectF(cx - bar / 2, h - bottom - height, bar, height)
            today = day == self._today
            colour = QColor(C.PRI) if today else QColor(255, 255, 255, 38)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colour)
            p.drawRoundedRect(rect, 6, 6)
            p.setFont(small)
            p.setPen(QColor(C.WHITE if today else C.FAINT))
            p.drawText(QRectF(cx - slot / 2, h - bottom + 4, slot, bottom - 4),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, day.strftime("%a"))
            if minutes:
                p.setPen(QColor(C.MUTED))
                p.drawText(QRectF(cx - slot / 2, rect.top() - 18, slot, 16),
                           Qt.AlignmentFlag.AlignCenter, fitness.clock(minutes))
        p.end()


class FitnessDeckView(QWidget):
    def __init__(self, bus: Any, dispatcher: Any = None, *, store: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus, self.dispatcher, self.store = bus, dispatcher, store
        self._busy = False
        self._summary: dict[str, Any] = {}
        self._health_goals: list[dict[str, Any]] = []

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
        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 2)

        tiles = QHBoxLayout()
        tiles.setSpacing(SPACE.SM + SPACE.XS)
        self.tiles: dict[str, Any] = {}
        for key, caption in (("minutes", "This Week"), ("workouts", "Workouts"),
                             ("distance", "Kilometres"), ("streak", "Day Streak")):
            tile, box = card()
            value = label("0", "readout")
            box.addWidget(value)
            box.addWidget(label(caption, "readoutKey"))
            self.tiles[key] = value
            tiles.addWidget(tile, 1)
        grid.addLayout(tiles, 0, 0)

        week_card, week_box = card("This Week")
        self.bars = WeekBars()
        week_box.addWidget(self.bars)
        grid.addWidget(week_card, 1, 0)

        log_card, log_box = card()
        chips = QHBoxLayout()
        chips.setSpacing(SPACE.SM)
        self.kinds = QButtonGroup(self)
        self.kinds.setExclusive(True)
        for index, kind in enumerate(fitness.KINDS[:6]):
            chip = QPushButton(KIND_NAMES[kind])
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setProperty("kind", kind)
            chip.setChecked(index == 0)
            self.kinds.addButton(chip)
            chips.addWidget(chip)
        chips.addStretch(1)
        log_box.addLayout(chips)
        row = QHBoxLayout()
        row.setSpacing(SPACE.SM)
        self.minutes = QSpinBox()
        self.minutes.setRange(1, 600)
        self.minutes.setValue(30)
        self.minutes.setSuffix(" min")
        self.minutes.setAccessibleName("Minutes")
        self.distance = QDoubleSpinBox()
        self.distance.setRange(0.0, 500.0)
        self.distance.setDecimals(1)
        self.distance.setSingleStep(0.5)
        self.distance.setSuffix(" km")
        self.distance.setAccessibleName("Distance in kilometres")
        self.note = QLineEdit()
        self.note.setPlaceholderText("Note")
        self.note.returnPressed.connect(self.log_workout)
        self.log_button = QPushButton("Log")
        self.log_button.setObjectName("primaryButton")
        self.log_button.setAccessibleName("Log the workout")
        self.log_button.clicked.connect(self.log_workout)
        row.addWidget(self.minutes, 1)
        row.addWidget(self.distance, 1)
        row.addWidget(self.note, 2)
        row.addWidget(self.log_button)
        log_box.addLayout(row)
        self.count_goals = QCheckBox("Count Towards Goals")
        self.count_goals.setChecked(True)
        self.count_goals.setVisible(False)
        log_box.addWidget(self.count_goals)
        self.status = label("", "mutedLabel", wrap=True)
        log_box.addWidget(self.status)
        grid.addWidget(log_card, 2, 0)
        grid.setRowStretch(3, 1)

        side = QVBoxLayout()
        side.setSpacing(SPACE.LG)
        goals_card, goals_box = card("Goals")
        self.goals_list = QVBoxLayout()
        self.goals_list.setSpacing(SPACE.MD)
        goals_box.addLayout(self.goals_list)
        target_row = QHBoxLayout()
        target_row.addWidget(label("Workouts a week", "entryTitle"))
        target_row.addStretch(1)
        self.target = QSpinBox()
        self.target.setRange(1, 14)
        self.target.setValue(fitness.DEFAULT_WEEKLY_WORKOUTS)
        self.target.setAccessibleName("Workouts a week")
        self.target.valueChanged.connect(self._set_target)
        target_row.addWidget(self.target)
        goals_box.addSpacing(SPACE.XS)
        goals_box.addLayout(target_row)
        side.addWidget(goals_card)
        recent_card, recent_box = card("Recent")
        self.recent_list = QVBoxLayout()
        self.recent_list.setSpacing(0)
        recent_box.addLayout(self.recent_list)
        side.addWidget(recent_card)
        side.addStretch(1)
        grid.addLayout(side, 0, 1, 4, 1)

        self._recent: list[fitness.Workout] = []
        self._show_goals([])
        self._show_recent([], fitness._now())

    # ── store ────────────────────────────────────────────────────────────────

    def _open(self) -> Any:
        if self.store is None:
            self.store = fitness.FitnessStore()
        return self.store

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self.load()

    def load(self) -> None:
        if self._busy or sip.isdeleted(self):
            return
        self._busy = True
        run_async(self, self._gather, done=self._loaded, failed=self._failed,
                  name="orion-fitness-page")

    async def _gather(self) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
        import asyncio

        def read() -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
            store = self._open()
            return (fitness.week_summary(store), store.recent(6), self._read_goals())
        return await asyncio.to_thread(read)

    def _read_goals(self) -> list[dict[str, Any]]:
        engine = getattr(self.dispatcher, "goal_tracker", None)
        if engine is None:
            return []
        try:
            rows = engine.overview()
        except Exception:
            return []
        out = []
        for row in rows:
            goal = row.get("goal")
            if goal is not None and str(getattr(goal, "area", "")).lower() in {"health", "fitness"}:
                out.append({"title": goal.title, "percent": int(row.get("percent") or 0),
                            "pace": str(row.get("pace") or "")})
        return out[:4]

    def _loaded(self, result: tuple[dict[str, Any], list[Any], list[dict[str, Any]]]) -> None:
        self._busy = False
        summary, recent, goals = result
        self.show_summary(summary)
        self._show_recent(recent, fitness._now())
        self._show_goals(goals)

    def _failed(self, exc: BaseException) -> None:
        self._busy = False
        self.status.setText(f"Workouts could not be read: {str(exc)[:120]}")

    # ── drawing ──────────────────────────────────────────────────────────────

    def show_summary(self, summary: dict[str, Any]) -> None:
        self._summary = dict(summary)
        self.tiles["minutes"].setText(fitness.clock(summary.get("minutes", 0)))
        self.tiles["workouts"].setText(f"{summary.get('workouts', 0)}/{summary.get('target_workouts', 0)}")
        self.tiles["distance"].setText(f"{summary.get('distance_km', 0):g}")
        self.tiles["streak"].setText(str(summary.get("streak", 0)))
        self.bars.set_days(summary.get("days", []), summary.get("today"))
        target = int(summary.get("target_workouts") or fitness.DEFAULT_WEEKLY_WORKOUTS)
        if self.target.value() != target:
            self.target.blockSignals(True)
            self.target.setValue(target)
            self.target.blockSignals(False)

    def _show_goals(self, goals: list[dict[str, Any]]) -> None:
        self._health_goals = list(goals)
        clear_layout(self.goals_list)
        self.count_goals.setVisible(bool(goals) and self.dispatcher is not None)
        for goal in goals:
            row = QVBoxLayout()
            row.setSpacing(SPACE.XS + 2)
            head = QHBoxLayout()
            head.addWidget(label(goal["title"], "entryTitle", wrap=True), 1)
            head.addWidget(label(f"{goal['percent']}%", "entryTime"))
            row.addLayout(head)
            meter = QProgressBar()
            meter.setObjectName("goalMeter")
            meter.setRange(0, 100)
            meter.setValue(max(0, min(100, goal["percent"])))
            meter.setTextVisible(False)
            meter.setFixedHeight(6)
            meter.setAccessibleName(f"{goal['title']}, {goal['percent']} percent")
            row.addWidget(meter)
            self.goals_list.addLayout(row)
        if not goals:
            self.goals_list.addWidget(label("No health goals yet", "readoutKey"))

    def _show_recent(self, workouts: list[Any], now: datetime) -> None:
        self._recent = list(workouts)
        clear_layout(self.recent_list)
        for workout in workouts:
            line = QFrame()
            box = QHBoxLayout(line)
            box.setContentsMargins(0, SPACE.SM, 0, SPACE.SM)
            names = QVBoxLayout()
            names.setSpacing(0)
            names.addWidget(label(KIND_NAMES.get(workout.kind, "Workout"), "entryTitle"))
            names.addWidget(label(when_text(workout.at, now), "readoutKey"))
            box.addLayout(names, 1)
            box.addWidget(label(amount_text(workout), "entryTime"))
            remove = QPushButton()
            remove.setObjectName("iconButton")
            apply_icon(remove, "close")
            remove.setAccessibleName(f"Remove the {KIND_NAMES.get(workout.kind, 'workout')} "
                                     f"from {when_text(workout.at, now) or 'the log'}")
            remove.clicked.connect(lambda _c=False, wid=workout.id: self.remove(wid))
            box.addWidget(remove)
            self.recent_list.addWidget(line)
        if not workouts:
            self.recent_list.addWidget(label("Nothing logged yet", "readoutKey"))

    # ── writing ──────────────────────────────────────────────────────────────

    def selected_kind(self) -> str:
        button = self.kinds.checkedButton()
        return str(button.property("kind")) if button is not None else "other"

    def log_workout(self) -> None:
        if self._busy:
            return
        kind, minutes = self.selected_kind(), float(self.minutes.value())
        distance = float(self.distance.value()) or None
        if kind not in fitness.DISTANCE_KINDS:
            distance = None
        note = self.note.text().strip()
        count = self.count_goals.isVisible() and self.count_goals.isChecked()
        self._busy = True
        self.log_button.setEnabled(False)

        async def write() -> str:
            import asyncio
            workout = await asyncio.to_thread(
                lambda: self._open().log(kind, minutes, distance_km=distance, note=note))
            said = f"{KIND_NAMES.get(workout.kind, 'Workout')} logged, {amount_text(workout)}."
            tool = getattr(self.dispatcher, "goals_tool", None)
            if count and callable(tool):
                result = await tool({"action": "log_progress",
                                     "text": goal_sentence(workout.kind, workout.minutes,
                                                           workout.distance_km)})
                if getattr(result, "ok", False):
                    said += " " + str(getattr(result, "text", "")).split("\n")[0]
            return said

        run_async(self, write, done=self._logged, failed=self._log_failed,
                  name="orion-fitness-log")

    def _logged(self, said: str) -> None:
        self._busy = False
        self.log_button.setEnabled(True)
        self.note.clear()
        self.status.setText(said)
        try:
            self.bus.dashboard_event.emit("fitness", said)
        except Exception:
            pass
        self.load()

    def _log_failed(self, exc: BaseException) -> None:
        self._busy = False
        self.log_button.setEnabled(True)
        self.status.setText(str(exc)[:160])

    def remove(self, workout_id: int | None) -> None:
        if workout_id is None or self._busy:
            return
        off_thread(self, lambda: self._open().remove(workout_id),
                   done=lambda _ok: self.load(), name="orion-fitness-remove")

    def _set_target(self, value: int) -> None:
        off_thread(self, lambda: self._open().set_target("workouts", int(value)),
                   done=lambda _r: self.load(), name="orion-fitness-target")


__all__ = ["FitnessDeckView", "WeekBars", "amount_text", "goal_sentence", "when_text"]
