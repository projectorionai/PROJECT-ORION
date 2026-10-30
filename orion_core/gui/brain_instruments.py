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
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QRadialGradient
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
    key: str = ""                 # a WORK bar's identity (agent:, job:, tool:)


#: WORK (Mark XXXX.4): what he does in the background - a specialist agent, a
#: job, a tool still running after its turn answered - one bar each, from
#: start to finish, so nothing he is doing is off the page.
LANES = ("HEAR", "THINK", "ACT", "SPEAK", "RECALL", "WORK")
#: While only background work moves, the lanes redraw this often (ms).
WORK_FRAME_MS = 250
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

#: The words the page shows for ORION's states (shared with the Brain host).
STATE_WORDS = {"STANDBY": "Ready", "PROCESSING": "Thinking", "SPEAKING": "Speaking",
               "LISTENING": "Listening", "PAUSED": "Paused", "ERROR": "Needs Attention"}


def display_state(state: str) -> str:
    state = str(state or "STANDBY")
    return STATE_WORDS.get(state, state.replace("_", " ").capitalize())


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
        #: The two halves of `link`, so a long provider name can be shortened
        #: on the ring without losing the state after it.
        self.link_parts: tuple[str, str] | None = None
        self.first_word_s: float | None = None
        self.segments: deque[Segment] = deque(maxlen=400)
        self._open: dict[str, Segment] = {}
        self._heat: dict[str, tuple[float, float]] = {}   # segment -> (value, at)
        self._turn = 0
        self._turn_open = False
        self._last_heard = 0.0
        self.last_event_at = -1e9
        #: Background work in progress, by key. Kept apart from the turn's
        #: lanes: an agent running for minutes must not hold a turn open.
        self._work: dict[str, Segment] = {}
        self._work_seq = 0

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
        if self.busy() or self._work:
            return True
        return now - self.last_event_at < self.WINDOW_S + 1.0

    def working(self) -> list[Segment]:
        """Background work still in progress, oldest first."""
        return sorted(self._work.values(), key=lambda s: s.start)

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

    def work_started(self, key: str, label: str = "") -> None:
        """A piece of background work began (an agent, a job, a detached tool)."""
        key = str(key or "")
        if not key or key in self._work:
            return
        self._work_seq -= 1                # its own thread: never joined to a turn
        seg = Segment("WORK", self.clock(), None, str(label or key)[:60], True,
                      self._work_seq, key=key)
        self._work[key] = seg
        self.segments.append(seg)
        self.last_event_at = self.clock()
        self._touch("TOOLS", 0.5)
        self.changed.emit()

    def work_finished(self, key: str, ok: bool = True) -> None:
        seg = self._work.pop(str(key or ""), None)
        if seg is None:
            return
        seg.end = max(seg.start, self.clock())
        seg.ok = bool(ok)
        self.last_event_at = self.clock()
        self.changed.emit()

    def set_telemetry(self, cpu: float | None, ram: float | None,
                      gpu: float | None = None) -> None:
        self.cpu, self.ram, self.gpu = cpu, ram, gpu
        self.changed.emit()

    def set_link(self, provider: str, state: str) -> None:
        state = str(state or "").replace("_", " ").lower()
        self.link_ok = state in {"connected", "synchronised", "live"}
        self.link = f"{provider or 'Local'}  ·  {state.title() or 'Unknown'}"
        self.link_parts = (provider or "Local", state.title() or "Unknown")
        self._touch("NET", 0.6 if self.link_ok else 1.0)
        self.changed.emit()

    # ── handing the picture to another process (gui/page_host) ──────────────

    def snapshot(self) -> dict[str, Any]:
        """Everything on screen now, with times as ages - two processes do
        not share a clock - so a Brain host started late begins in sync."""
        now = self.clock()
        open_ids = {id(seg) for seg in (*self._open.values(), *self._work.values())}
        return {
            "state": self.state, "speaking": self.speaking,
            "cpu": self.cpu, "ram": self.ram, "gpu": self.gpu,
            "link": self.link, "link_ok": self.link_ok,
            "link_parts": list(self.link_parts) if self.link_parts else None,
            "first_word_s": self.first_word_s,
            "segments": [[s.lane, now - s.start, None if s.end is None else now - s.end,
                          s.label, s.ok, s.turn, s.tick, id(s) in open_ids, s.key]
                         for s in self.visible_segments(now)
                         + [w for w in self._work.values()
                            if w not in self.visible_segments(now)]],
            "heat": {name: [value, now - at] for name, (value, at) in self._heat.items()},
            "turn": self._turn, "turn_open": self._turn_open,
            "last_heard_age": (now - self._last_heard) if self._last_heard else None,
            "last_event_age": now - self.last_event_at,
        }

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        if not isinstance(snap, dict):
            return
        now = self.clock()
        self.state = str(snap.get("state") or "STANDBY")
        self.speaking = bool(snap.get("speaking"))
        self.cpu, self.ram, self.gpu = snap.get("cpu"), snap.get("ram"), snap.get("gpu")
        self.link = str(snap.get("link") or self.link)
        self.link_ok = bool(snap.get("link_ok"))
        parts = snap.get("link_parts")
        self.link_parts = (str(parts[0]), str(parts[1])) if isinstance(parts, list) and len(parts) == 2 else None
        self.first_word_s = snap.get("first_word_s")
        self.segments.clear()
        self._open.clear()
        self._work.clear()
        for row in snap.get("segments") or []:
            try:
                lane, start_age, end_age, label, ok, turn, tick, is_open = row[:8]
                key = str(row[8]) if len(row) > 8 else ""
                seg = Segment(str(lane), now - float(start_age),
                              None if end_age is None else now - float(end_age),
                              str(label), bool(ok), int(turn), bool(tick), key=key)
            except (TypeError, ValueError):
                continue
            self.segments.append(seg)
            if is_open and seg.end is None:
                if seg.lane == "WORK":
                    self._work[seg.key or seg.label] = seg
                else:
                    self._open[seg.lane] = seg
        self._heat = {}
        for name, pair in (snap.get("heat") or {}).items():
            try:
                self._heat[str(name)] = (float(pair[0]), now - float(pair[1]))
            except (TypeError, ValueError, IndexError):
                continue
        self._turn = int(snap.get("turn") or 0)
        self._turn_open = bool(snap.get("turn_open"))
        heard = snap.get("last_heard_age")
        self._last_heard = (now - float(heard)) if heard is not None else 0.0
        self.last_event_at = now - float(snap.get("last_event_age") or 1e9)
        self.changed.emit()


def _fit_link(model: "BrainModel", font: QFont, width: float) -> str:
    """The link line as it fits inside the ring: the state ("Disconnected")
    is kept whole and a long provider name is shortened with an ellipsis -
    it used to be clipped at both ends, losing the state (audit 6 Oct 2026).
    The side panel still shows the full name."""
    metrics = QFontMetricsF(font)
    if metrics.horizontalAdvance(model.link) <= width or not model.link_parts:
        return metrics.elidedText(model.link, Qt.TextElideMode.ElideRight, width)
    provider, state = model.link_parts
    tail = f"  ·  {state}"
    room = width - metrics.horizontalAdvance(tail)
    if room < metrics.horizontalAdvance("M…"):
        return metrics.elidedText(state, Qt.TextElideMode.ElideRight, width)
    return metrics.elidedText(provider, Qt.TextElideMode.ElideRight, room) + tail


# ── shared animation clock ───────────────────────────────────────────────────

class _Animator(QObject):
    """One timer for both instruments; it runs only while they are moving.

    In ORION's process the instruments draw at 30 fps (*attended_ms* 33): the
    frames come out of the same thread as his Python. The Brain host, whose
    thread does nothing else, draws them at 60 (16)."""

    def __init__(self, model: BrainModel, widgets: list[QWidget],
                 attended_ms: int = 33) -> None:
        super().__init__()
        self.model = model
        self.widgets = widgets
        self.attended = True
        self.attended_ms = max(8, int(attended_ms))
        self.visible = False
        self.timer = QTimer(self)
        if self.attended_ms < 30:
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        model.changed.connect(self.kick)

    def interval_ms(self) -> int:
        if not self.model.busy() and self.model.working():
            return WORK_FRAME_MS            # only background work moving: slow frames
        return self.attended_ms if self.attended else 90

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
        elif self.timer.interval() != self.interval_ms():
            self.timer.setInterval(self.interval_ms())


# ── the Iris ─────────────────────────────────────────────────────────────────

class IrisGauge(QWidget):
    """ORION's state and load as one circular instrument, echoing his orb."""

    _SWEEP_SPEED = {"LISTENING": 0.35, "PROCESSING": 2.4, "THINKING": 2.4,
                    "ACTING": 1.6, "SPEAKING": 1.0, "CONNECTING": 1.2}

    def __init__(self, model: BrainModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self._sweep = 0.0
        self.state_label: Callable[[str], str] = lambda state: state
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
                   int(Qt.AlignmentFlag.AlignCenter), self.state_label(state))
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
                   int(Qt.AlignmentFlag.AlignCenter), _fit_link(model, p.font(), r2 * 2))
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


# ── the ECG monitor (Mark XXXX.4) ──────────────────────────────────────────────
#
# Owner, 28 Oct 2026: "a graph ... kind of looking like an ECG machine" for
# what his brain is doing. Five leads sweep left to right over ECG paper, the
# way a bedside monitor draws: each thing he really does is one heartbeat on
# its lead (hearing, thinking, acting, speaking, background work), work in
# progress keeps beating until it ends, and a quiet lead is a flat line.
#
# Cost: like a real monitor it only draws the newest sliver each frame into a
# kept image (a few short line segments per lead), clears the gap ahead of
# the write-head, and paints that image - well under a millisecond a frame.

ECG_LEADS = ("HEAR", "THINK", "ACT", "SPEAK", "WORK")
#: Seconds one sweep takes from left to right.
ECG_SWEEP_S = 8.0
#: A lead that is working beats this often (seconds).
ECG_BEAT_EVERY_S = 0.55


def ecg_beat(dt: float) -> float:
    """One heartbeat (P wave, QRS complex, T wave) *dt* seconds after it began."""
    if dt < 0.0 or dt > 0.65:
        return 0.0

    def bump(centre: float, width: float, height: float) -> float:
        return height * math.exp(-((dt - centre) / width) ** 2)
    return (bump(0.08, 0.030, 0.12) + bump(0.195, 0.010, -0.16) + bump(0.215, 0.011, 1.0)
            + bump(0.240, 0.012, -0.28) + bump(0.43, 0.050, 0.24))


class EcgMonitor(QWidget):
    """Brain activity as a five-lead ECG trace."""

    FRAME_MS = 33
    STEP_S = 1.0 / 120.0                 # trace resolution

    def __init__(self, parent: QWidget | None = None, clock: Callable[[], float] = time.monotonic
                 ) -> None:
        super().__init__(parent)
        self.clock = clock
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Brain activity monitor")
        self._beats: dict[str, list[float]] = {lead: [] for lead in ECG_LEADS}
        self._open: dict[str, dict[str, float]] = {lead: {} for lead in ECG_LEADS}
        self._last_beat: dict[str, float] = {lead: -99.0 for lead in ECG_LEADS}
        self._canvas: Any = None
        self._drawn_to: float | None = None
        self._last_y: dict[str, float] = {}
        self.rate: dict[str, int] = {lead: 0 for lead in ECG_LEADS}
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # ── what happened ────────────────────────────────────────────────────────

    def beat(self, lead: str, at: float | None = None) -> None:
        if lead in self._beats:
            when = self.clock() if at is None else at
            beats = self._beats[lead]
            beats.append(when)
            del beats[:-12]                     # hidden, nothing prunes them
            self._last_beat[lead] = when

    def open(self, lead: str, key: str) -> None:
        if lead in self._open and key not in self._open[lead]:
            self._open[lead][key] = self.clock()
            self.beat(lead)

    def close(self, lead: str, key: str) -> None:
        if lead in self._open and self._open[lead].pop(key, None) is not None:
            self.beat(lead)

    def active(self, lead: str) -> bool:
        return bool(self._open.get(lead))

    #: Trace stages that open a lead, and the stages that close it.
    _OPEN = {"API_REQUEST_START": "THINK", "TOOL_EXECUTION_START": "ACT",
             "PLAYBACK_START": "SPEAK"}
    _CLOSE = {"API_RESPONSE_RECEIVED": ("THINK",), "TOOL_EXECUTION_COMPLETE": ("ACT",),
              "PLAYBACK_COMPLETE": ("SPEAK",), "LISTENING_RESTORED": ("SPEAK", "THINK"),
              "REPLY_DELIVERED": ("THINK", "SPEAK")}

    def feed(self, message: str) -> None:
        """One live log line onto the leads: request TRACE stages, and the
        background work the BRAIN page's WORK lane reads."""
        text = str(message)
        if text.startswith("TRACE: "):
            parts = text.split()
            if len(parts) < 3:
                return
            rid, stage = parts[1], parts[2]
            if stage == "TRANSCRIPTION_COMPLETE":
                self.beat("HEAR")
            elif stage in self._OPEN:
                self.open(self._OPEN[stage], rid)
            elif stage in self._CLOSE:
                for lead in self._CLOSE[stage]:
                    self.close(lead, rid)
            elif stage.startswith("FAILED_STAGE"):
                for lead in ("THINK", "ACT", "SPEAK"):
                    self.close(lead, rid)
            return
        from .command_overview import _WORK_LINES
        for pattern, key_of, _label_of, outcome in _WORK_LINES:
            match = pattern.match(text)
            if match is not None:
                (self.open if outcome is None else self.close)("WORK", key_of(match))
                return

    def value(self, lead: str, at: float) -> float:
        """The trace's height (about -0.3 to 1) on *lead* at time *at*."""
        return sum(ecg_beat(at - start) for start in self._beats[lead][-6:])

    # ── drawing ──────────────────────────────────────────────────────────────

    def _layout(self) -> tuple[float, float, float]:
        left = 64.0
        return left, max(1.0, self.width() - left - 10), max(1.0, (self.height() - 8) / len(ECG_LEADS))

    def _paper(self, painter: QPainter, x0: float, x1: float) -> None:
        """ECG paper between x0 and x1: fine and bold grid lines."""
        left, _width, _lane = self._layout()
        painter.fillRect(QRectF(x0, 0, x1 - x0, self.height()), _colour(C.INK))
        minor, major = 8, 40
        start = int((x0 - left) // minor) * minor + left
        x = start
        while x <= x1:
            if x >= x0:
                bold = int(round((x - left) / minor)) % (major // minor) == 0
                painter.setPen(QPen(_colour(C.PRI, 0.10 if bold else 0.035), 1))
                painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            x += minor
        y = 0
        while y <= self.height():
            bold = (y // minor) % (major // minor) == 0
            painter.setPen(QPen(_colour(C.PRI, 0.10 if bold else 0.035), 1))
            painter.drawLine(QPointF(x0, y), QPointF(x1, y))
            y += minor

    def _x_at(self, at: float) -> float:
        left, width, _lane = self._layout()
        return left + (at % ECG_SWEEP_S) / ECG_SWEEP_S * width

    def _ensure_canvas(self) -> None:
        from PyQt6.QtGui import QImage
        ratio = max(1.0, self.devicePixelRatioF())
        size = (max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        if self._canvas is None or (self._canvas.width(), self._canvas.height()) != size:
            self._canvas = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
            self._canvas.setDevicePixelRatio(ratio)
            painter = QPainter(self._canvas)
            self._paper(painter, 0, self.width())
            painter.end()
            self._drawn_to = None
            self._last_y = {}

    def _tick(self) -> None:
        if not self.isVisible():
            self._timer.stop()
            return
        now = self.clock()
        for lead in ECG_LEADS:
            if self._open[lead] and now - self._last_beat[lead] >= ECG_BEAT_EVERY_S:
                self.beat(lead, now)
            self._beats[lead] = [b for b in self._beats[lead] if now - b < 1.0]
            self.rate[lead] = int(round(60.0 / ECG_BEAT_EVERY_S)) if self._open[lead] else 0
        self.advance(now)
        self.update()

    def advance(self, now: float) -> None:
        """Draw the trace from where it stopped up to *now* into the canvas."""
        if self.width() < 80 or self.height() < 60:
            return
        self._ensure_canvas()
        left, width, lane = self._layout()
        start = self._drawn_to if self._drawn_to is not None else now - self.STEP_S
        start = max(start, now - ECG_SWEEP_S * 0.5)
        painter = QPainter(self._canvas)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # The gap ahead of the write-head, as on a bedside monitor.
            head = self._x_at(now)
            gap_end = min(left + width, head + 14)
            self._paper(painter, head + 1, gap_end)
            if head + 14 > left + width:
                self._paper(painter, left, left + (head + 14 - (left + width)))
            t = start
            while t < now:
                t2 = min(now, t + self.STEP_S)
                x1, x2 = self._x_at(t), self._x_at(t2)
                if x2 >= x1:                       # not across the wrap
                    for index, lead in enumerate(ECG_LEADS):
                        base = 4 + lane * (index + 0.62)
                        amp = lane * 0.52
                        y1 = self._last_y.get(lead, base - amp * self.value(lead, t))
                        y2 = base - amp * self.value(lead, t2)
                        busy = self.active(lead) or any(t2 - b < 0.7 for b in self._beats[lead][-2:])
                        if busy:
                            painter.setPen(QPen(_colour(C.PRI, 0.25), 3.2))
                            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                        painter.setPen(QPen(_colour(C.PRI_HI if busy else C.PRI_DIM, 0.95), 1.4))
                        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                        self._last_y[lead] = y2
                else:
                    self._last_y = {}
                t = t2
        finally:
            painter.end()
        self._drawn_to = now

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        if self.width() < 80 or self.height() < 60:
            return
        if self._canvas is None:
            self.advance(self.clock())
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._canvas is not None:
                painter.drawImage(0, 0, self._canvas)
            left, _width, lane = self._layout()
            painter.fillRect(QRectF(0, 0, left - 4, self.height()), _colour(C.PANEL))
            painter.setFont(_mono(8))
            for index, lead in enumerate(ECG_LEADS):
                y = 4 + lane * (index + 0.62)
                painter.setPen(_colour(C.SILVER if self.active(lead) else C.FAINT))
                painter.drawText(QRectF(8, y - 14, left - 12, 12),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), lead)
                rate = self.rate.get(lead, 0)
                painter.setPen(_colour(C.PRI_HI if rate else C.FAINT))
                painter.drawText(QRectF(8, y - 1, left - 12, 12),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 f"{rate} bpm" if rate else "--")
            head = self._x_at(self.clock())
            painter.setPen(Qt.PenStyle.NoPen)
            for index, lead in enumerate(ECG_LEADS):
                y = self._last_y.get(lead)
                if y is not None:
                    painter.setBrush(_colour(C.CORE, 0.9))
                    painter.drawEllipse(QPointF(head, y), 2.2, 2.2)
        finally:
            painter.end()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._drawn_to = None
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)
