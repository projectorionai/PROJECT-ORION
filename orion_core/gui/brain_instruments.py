"""
The Brain page's two instruments (Mark XXXII): the Iris and the Signal Lanes.

Both are drawn with QPainter from one ``BrainModel`` that the page feeds from
signals ORION already emits — his state, speaking on/off, telemetry, the Live
connection, the dispatcher's report after every tool call, the request traces
and the first words of each thing he hears. Nothing here reaches into the
voice or tool paths.

Everything drawn is something that happened. A moment with no measured
duration ("he heard you") is drawn as a tick, never as an invented bar.

Cost: the instruments draw only while something is moving. With ORION idle
the timer stops and the last frame stays on screen — the deck usually lives
on a second monitor, sharing the one GPU with the face.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C

_UI_FAMILY = "Segoe UI Variable Display"
_MONO_FAMILY = "Cascadia Mono"


def _colour(hex_colour: str, alpha: float = 1.0) -> QColor:
    colour = QColor(hex_colour)
    colour.setAlphaF(max(0.0, min(1.0, alpha)))
    return colour


def _mono(px: int) -> QFont:
    font = QFont(_MONO_FAMILY)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(px)
    return font


# ── the model ────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    lane: str
    start: float
    end: float | None = None      # None while it is still happening
    label: str = ""
    ok: bool = True
    turn: int = 0
    tick: bool = False            # an instant, not a duration


LANES = ("HEAR", "THINK", "ACT", "SPEAK", "RECALL")
IRIS_SEGMENTS = ("HEAR", "SIGHT", "MEMORY", "REASON", "TOOLS", "AGENTS",
                 "VOICE", "WEB", "DESKTOP", "CHESS", "SYSTEM", "NET")

#: Tool name fragments -> the Iris segment they light. First match wins.
_TOOL_SEGMENT = (
    (("vision", "camera", "screen", "ocr", "watch", "face_track", "image"), "SIGHT"),
    (("memory", "recall", "remember", "knowledge", "second_brain", "awareness",
      "library", "decision"), "MEMORY"),
    (("research", "search", "fetch", "news", "web", "flight", "aviation",
      "weather", "browse"), "WEB"),
    (("desktop", "open_app", "type_text", "edit_text", "file", "clipboard",
      "browser", "window", "control"), "DESKTOP"),
    (("chess",), "CHESS"),
    (("agent", "reason", "workflow", "mission", "specialist"), "AGENTS"),
)
_RECALL_WORDS = ("memory", "recall", "remember", "knowledge", "second_brain",
                 "awareness", "library")
_BUSY_STATES = frozenset({"PROCESSING", "THINKING", "SPEAKING", "CONNECTING", "ACTING"})


def segment_for_tool(name: str) -> str:
    lowered = str(name or "").lower()
    if lowered.startswith("mcp__"):
        return "TOOLS"
    for words, segment in _TOOL_SEGMENT:
        if any(word in lowered for word in words):
            return segment
    return "TOOLS"


class BrainModel(QObject):
    """What ORION is doing, kept just long enough to draw it."""

    changed = pyqtSignal()

    WINDOW_S = 20.0          # how much history the lanes show
    HEAT_DECAY_S = 4.0       # time constant of an Iris segment fading
    HEAR_GAP_S = 1.6         # silence that ends a heard utterance

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self.clock = clock
        self.state = "STANDBY"
        self.speaking = False
        self.cpu: float | None = None
        self.ram: float | None = None
        self.gpu: float | None = None
        self.link = "Waiting for connection"
        self.link_ok = False
        self.first_word_s: float | None = None
        self.segments: deque[Segment] = deque(maxlen=400)
        self._open: dict[str, Segment] = {}
        self._heat: dict[str, tuple[float, float]] = {}   # segment -> (value, at)
        self._turn = 0
        self._turn_open = False
        self._last_heard = 0.0
        self.last_event_at = -1e9

    # ── reading ──────────────────────────────────────────────────────────────

    def heat(self, segment: str, now: float | None = None) -> float:
        now = self.clock() if now is None else now
        value, at = self._heat.get(segment, (0.0, now))
        decayed = value * math.exp(-(now - at) / self.HEAT_DECAY_S)
        if segment == "SYSTEM" and self.cpu is not None:
            decayed = max(decayed, min(1.0, self.cpu / 100.0) * 0.8)
        return decayed

    def busy(self) -> bool:
        return self.state in _BUSY_STATES or self.speaking or bool(self._open)

    def needs_frames(self, now: float | None = None) -> bool:
        """Whether anything on screen is still moving."""
        now = self.clock() if now is None else now
        if self.busy():
            return True
        return now - self.last_event_at < self.WINDOW_S + 1.0

    def visible_segments(self, now: float | None = None) -> list[Segment]:
        now = self.clock() if now is None else now
        horizon = now - self.WINDOW_S
        return [s for s in self.segments if (s.end if s.end is not None else now) >= horizon]

    # ── events ───────────────────────────────────────────────────────────────

    def _touch(self, segment: str, amount: float = 1.0) -> None:
        now = self.clock()
        self._heat[segment] = (max(self.heat(segment, now), amount), now)
        self.last_event_at = now

    def _begin_turn(self) -> None:
        if not self._turn_open:
            self._turn += 1
            self._turn_open = True

    def _open_lane(self, lane: str, label: str = "") -> Segment:
        seg = self._open.get(lane)
        if seg is None:
            self._begin_turn()
            seg = Segment(lane, self.clock(), None, label, True, self._turn)
            self._open[lane] = seg
            self.segments.append(seg)
        elif label and not seg.label:
            seg.label = label
        self.last_event_at = self.clock()
        return seg

    def _close_lane(self, lane: str, at: float | None = None, *, ok: bool = True,
                    label: str = "") -> Segment | None:
        seg = self._open.pop(lane, None)
        if seg is not None:
            seg.end = max(seg.start, self.clock() if at is None else at)
            seg.ok = ok
            if label:
                seg.label = label
            self.last_event_at = self.clock()
        return seg

    def _maybe_end_turn(self) -> None:
        if not self._open and not self.speaking and self.state not in _BUSY_STATES:
            self._turn_open = False

    def heard(self) -> None:
        """A fragment of the user's speech was transcribed."""
        now = self.clock()
        self._last_heard = now
        self._open_lane("HEAR")
        self._touch("HEAR")
        self.changed.emit()

    def heard_utterance(self) -> None:
        """A whole utterance was recognised locally, with no running segment."""
        if "HEAR" in self._open:
            return
        self._begin_turn()
        now = self.clock()
        self.segments.append(Segment("HEAR", now, now, "", True, self._turn, tick=True))
        self._last_heard = now
        self._touch("HEAR")
        self.changed.emit()

    def settle(self) -> None:
        """Close a heard utterance once the speaker has gone quiet."""
        if "HEAR" in self._open and self.clock() - self._last_heard > self.HEAR_GAP_S:
            self._close_lane("HEAR", self._last_heard)
            self._maybe_end_turn()
            self.changed.emit()

    def set_state(self, state: str) -> None:
        state = str(state or "STANDBY").upper()
        if state == self.state:
            return
        self.state = state
        if state in {"PROCESSING", "THINKING"}:
            if "HEAR" in self._open:
                self._close_lane("HEAR", self._last_heard)
            self._open_lane("THINK")
            self._touch("REASON")
        else:
            self._close_lane("THINK")
        self._maybe_end_turn()
        self.last_event_at = self.clock()
        self.changed.emit()

    def set_speaking(self, active: bool) -> None:
        active = bool(active)
        if active == self.speaking:
            return
        self.speaking = active
        if active:
            now = self.clock()
            if "HEAR" in self._open:
                self._close_lane("HEAR", self._last_heard)
            self._close_lane("THINK")
            if self._last_heard and now - self._last_heard < 15.0:
                self.first_word_s = max(0.0, now - self._last_heard)
                self._last_heard = 0.0
            self._open_lane("SPEAK")
            self._touch("VOICE")
        else:
            self._close_lane("SPEAK")
            self._maybe_end_turn()
        self.changed.emit()

    def tool_started(self, names: str) -> None:
        self._open_lane("ACT", str(names or "")[:60])
        self._touch("TOOLS", 0.7)
        self.changed.emit()

    def tool_finished(self, name: str, ok: bool = True,
                      elapsed_ms: float | None = None) -> None:
        name = str(name or "tool")
        lane = "RECALL" if any(w in name.lower() for w in _RECALL_WORDS) else "ACT"
        open_seg = self._open.get("ACT")
        if lane == "ACT" and open_seg is not None and (
                not open_seg.label or name in open_seg.label):
            self._close_lane("ACT", ok=ok, label=name)
        else:
            now = self.clock()
            start = now - max(0.0, float(elapsed_ms or 0.0)) / 1000.0
            self._begin_turn()
            tick = elapsed_ms is None
            self.segments.append(Segment(lane, start, now, name, bool(ok), self._turn, tick=tick))
        self._touch(segment_for_tool(name))
        self._maybe_end_turn()
        self.changed.emit()

    def set_telemetry(self, cpu: float | None, ram: float | None,
                      gpu: float | None = None) -> None:
        self.cpu, self.ram, self.gpu = cpu, ram, gpu
        self.changed.emit()

    def set_link(self, provider: str, state: str) -> None:
        state = str(state or "").replace("_", " ").lower()
        self.link_ok = state in {"connected", "synchronised", "live"}
        self.link = f"{provider or 'Local'}  ·  {state.title() or 'Unknown'}"
        self._touch("NET", 0.6 if self.link_ok else 1.0)
        self.changed.emit()


# ── shared animation clock ───────────────────────────────────────────────────

class _Animator(QObject):
    """One timer for both instruments; it runs only while they are moving."""

    def __init__(self, model: BrainModel, widgets: list[QWidget]) -> None:
        super().__init__()
        self.model = model
        self.widgets = widgets
        self.attended = True
        self.visible = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        model.changed.connect(self.kick)

    def interval_ms(self) -> int:
        return 33 if self.attended else 90

    def kick(self) -> None:
        if not self.visible:
            return
        for widget in self.widgets:
            widget.update()
        if not self.timer.isActive() and self.model.needs_frames():
            self.timer.start(self.interval_ms())

    def set_attended(self, attended: bool) -> None:
        self.attended = bool(attended)
        if self.timer.isActive():
            self.timer.setInterval(self.interval_ms())

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.visible:
            self.kick()
        else:
            self.timer.stop()

    def _tick(self) -> None:
        self.model.settle()
        for widget in self.widgets:
            widget.update()
        if not self.model.needs_frames():
            self.timer.stop()        # the last frame stays on screen


# ── the Iris ─────────────────────────────────────────────────────────────────

class IrisGauge(QWidget):
    """ORION's state and load as one circular instrument, echoing his orb."""

    _SWEEP_SPEED = {"LISTENING": 0.35, "PROCESSING": 2.4, "THINKING": 2.4,
                    "ACTING": 1.6, "SPEAKING": 1.0, "CONNECTING": 1.2}

    def __init__(self, model: BrainModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._sweep = 0.0
        self._last_paint = time.monotonic()
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("ORION state gauge")

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        model = self.model
        now = model.clock()
        wall = time.monotonic()
        dt = min(0.1, wall - self._last_paint)
        self._last_paint = wall
        state = model.state if not model.speaking else "SPEAKING"
        if model.needs_frames(now):
            self._sweep = (self._sweep + dt * self._SWEEP_SPEED.get(state, 0.0)) % (2 * math.pi)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        radius = max(40.0, min(w, h) * 0.40)
        busy = state in _BUSY_STATES

        # Bezel ticks, like an instrument's.
        for i in range(72):
            angle = i / 72 * 2 * math.pi
            long_tick = i % 6 == 0
            p.setPen(QPen(_colour(C.WHITE, 0.20 if long_tick else 0.08), 1))
            r0, r1 = radius + 6, radius + (14 if long_tick else 10)
            p.drawLine(QPointF(cx + math.cos(angle) * r0, cy + math.sin(angle) * r0),
                       QPointF(cx + math.cos(angle) * r1, cy + math.sin(angle) * r1))

        # The twelve system segments, clockwise from twelve o'clock.
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        count = len(IRIS_SEGMENTS)
        step = 360.0 / count
        gap = 2.6
        p.setFont(_mono(10))
        for i, name in enumerate(IRIS_SEGMENTS):
            heat = model.heat(name, now)
            if name == "NET" and not model.link_ok:
                colour = _colour(C.AMBER, 0.55)
            elif heat > 0.2:
                colour = _colour(C.PRI, 0.35 + 0.65 * heat)
            else:
                colour = _colour(C.WHITE, 0.12)
            pen = QPen(colour, 5)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            start_deg = 90.0 - i * step - gap
            p.drawArc(rect, int(start_deg * 16), int(-(step - 2 * gap) * 16))
            if heat > 0.45:
                mid = math.radians(-(90.0 - i * step - step / 2.0))
                tx = cx + math.cos(mid) * (radius - 22)
                ty = cy + math.sin(mid) * (radius - 22)
                p.setPen(_colour(C.ACCENT))
                p.drawText(QRectF(tx - 40, ty - 8, 80, 16),
                           int(Qt.AlignmentFlag.AlignCenter), name)

        # Inner ring and its sweeping comet.
        r2 = radius * 0.66
        p.setPen(QPen(_colour(C.WHITE, 0.06), 2))
        p.drawEllipse(QPointF(cx, cy), r2, r2)
        if state in self._SWEEP_SPEED:
            inner = QRectF(cx - r2, cy - r2, r2 * 2, r2 * 2)
            head = math.degrees(self._sweep)
            for k in range(14):
                fade = 1.0 - k / 14.0
                colour = _colour(C.PRI, 0.9 * fade) if busy else _colour(C.WHITE, 0.5 * fade)
                pen = QPen(colour, 2.4)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawArc(inner, int(-(head - k * 3.4) * 16), int(3.4 * 16))

        glow = QRadialGradient(QPointF(cx, cy), radius * 0.55)
        glow.setColorAt(0.0, _colour(C.PRI, 0.16 if busy else 0.05))
        glow.setColorAt(1.0, _colour(C.PRI, 0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), radius * 0.55, radius * 0.55)

        # Centre readout.
        title = QFont(_UI_FAMILY)
        title.setPixelSize(max(14, int(radius * 0.13)))
        title.setWeight(QFont.Weight.DemiBold)
        p.setFont(title)
        p.setPen(_colour(C.WHITE))
        p.drawText(QRectF(cx - r2, cy - radius * 0.12, r2 * 2, radius * 0.18),
                   int(Qt.AlignmentFlag.AlignCenter), state)
        p.setFont(_mono(10))
        p.setPen(_colour(C.MUTED))
        load = "   ·   ".join(part for part in (
            f"CPU {model.cpu:.0f}%" if model.cpu is not None else "",
            f"RAM {model.ram:.0f}%" if model.ram is not None else "",
            f"GPU {model.gpu:.0f}%" if model.gpu is not None else "") if part)
        p.drawText(QRectF(cx - r2, cy + radius * 0.10, r2 * 2, 16),
                   int(Qt.AlignmentFlag.AlignCenter), load or "load —")
        p.setPen(_colour(C.FAINT if model.link_ok else C.AMBER))
        p.drawText(QRectF(cx - r2, cy + radius * 0.10 + 18, r2 * 2, 16),
                   int(Qt.AlignmentFlag.AlignCenter), model.link)
        p.end()


# ── the Signal Lanes ─────────────────────────────────────────────────────────

class SignalLanes(QWidget):
    """Each request as a thread stepping down through ORION's stages."""

    LEFT = 64
    stalled_count: Callable[[], int] = staticmethod(lambda: 0)

    def __init__(self, model: BrainModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(190)
        self.setAccessibleName("Request timeline: hear, think, act, speak and recall")

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        model = self.model
        now = model.clock()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = self.LEFT, w - 14.0, 30.0, h - 24.0
        lane_h = (bottom - top) / len(LANES)
        span = model.WINDOW_S

        def x_at(t: float) -> float:
            return right - (now - t) / span * (right - left)

        def y_of(lane: str) -> float:
            return top + lane_h * (LANES.index(lane) + 0.5)

        p.setFont(_mono(10))
        for lane in LANES:
            y = y_of(lane)
            p.setPen(_colour(C.FAINT))
            p.drawText(QRectF(12, y - 8, left - 16, 16),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), lane)
            p.setPen(QPen(_colour(C.WHITE, 0.05), 1))
            p.drawLine(QPointF(left, y), QPointF(right, y))
        for s in range(0, int(span) + 1, 5):
            x = right - s / span * (right - left)
            p.setPen(QPen(_colour(C.WHITE, 0.04), 1))
            p.drawLine(QPointF(x, top), QPointF(x, bottom))
            p.setPen(_colour(C.FAINT))
            p.drawText(QRectF(x - 24, h - 20, 48, 14), int(Qt.AlignmentFlag.AlignCenter),
                       "now" if s == 0 else f"-{s}s")

        segments = model.visible_segments(now)
        # Threads: each turn's stages joined in the order they began.
        turns: dict[int, list[Segment]] = {}
        for seg in segments:
            turns.setdefault(seg.turn, []).append(seg)
        p.setPen(QPen(_colour(C.WHITE, 0.14), 1))
        for items in turns.values():
            items.sort(key=lambda s: s.start)
            path = QPainterPath()
            for index, seg in enumerate(items):
                point = QPointF(max(left, x_at(seg.start)), y_of(seg.lane))
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            if len(items) > 1:
                p.drawPath(path)

        for seg in segments:
            y = y_of(seg.lane)
            live = seg.end is None
            end = now if live else seg.end
            x0, x1 = max(left, x_at(seg.start)), min(right, x_at(end))
            if x1 < left:
                continue
            if seg.tick:
                p.setPen(QPen(_colour(C.WHITE, 0.55) if seg.ok else _colour(C.AMBER), 2))
                p.drawLine(QPointF(x1, y - 6), QPointF(x1, y + 6))
            else:
                colour = (_colour(C.PRI) if live else
                          _colour(C.WHITE, 0.24) if seg.ok else _colour(C.AMBER, 0.8))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(colour)
                p.drawRoundedRect(QRectF(x0, y - 3.5, max(3.0, x1 - x0), 7), 3.5, 3.5)
                if live:
                    glow = QRadialGradient(QPointF(x1, y), 14)
                    glow.setColorAt(0.0, _colour(C.PRI, 0.5))
                    glow.setColorAt(1.0, _colour(C.PRI, 0.0))
                    p.setBrush(glow)
                    p.drawEllipse(QPointF(x1, y), 14, 14)
            if seg.label and seg.lane in {"ACT", "RECALL"}:
                # Beside the bar, on its own lane: above it, the label ran
                # into the lane over it. Near "now" it flips to the left.
                text = seg.label if seg.ok else f"{seg.label} · failed"
                width = p.fontMetrics().horizontalAdvance(text) + 4
                p.setPen(_colour(C.ACCENT if live else C.MUTED))
                if x1 + 8 + width <= right:
                    box = QRectF(x1 + 8, y - 7, width, 14)
                    align = Qt.AlignmentFlag.AlignLeft
                else:
                    box = QRectF(max(left, x0 - 8 - width), y - 7, width, 14)
                    align = Qt.AlignmentFlag.AlignRight
                p.drawText(box, int(align | Qt.AlignmentFlag.AlignVCenter), text)

        p.setPen(QPen(_colour(C.PRI, 0.5), 1))
        p.drawLine(QPointF(right, top - 6), QPointF(right, bottom))

        p.setPen(_colour(C.MUTED))
        p.drawText(QRectF(12, 6, 200, 16), int(Qt.AlignmentFlag.AlignLeft), f"LAST {int(span)} s")
        try:
            stalled = int(self.stalled_count())
        except Exception:
            stalled = 0
        first = (f"first word {model.first_word_s:.2f} s" if model.first_word_s is not None
                 else "first word —")
        p.setPen(_colour(C.AMBER) if stalled else _colour(C.MUTED))
        p.drawText(QRectF(w - 330, 6, 316, 16), int(Qt.AlignmentFlag.AlignRight),
                   f"{first}   ·   {stalled} stalled")
        p.end()
