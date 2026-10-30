"""
The Command Deck's instruments beyond the ECG (owner, 28 Oct 2026).

  VoiceRidgeline      the owner's voice and ORION's as a live ridgeline (BRAIN)
  ReactorGauge        CPU, GPU, RAM and network as nested arcs (Diagnostics)
  MemoryConstellation his memory by category, drawn as Orion's stars (Memory)
  DayRing             today's conversations, focus and study round a clock
  ToolRadar           every tool he uses, pinged as he uses it (NEURAL)
  MarketHeatMap       the watched markets by move (Widgets)

All of them draw only real data, in the interface's two lights (crimson and
cold white), and none animates while it is off screen. What does not move
between frames is drawn once into a cached pixmap.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ..constants import C
from .brain_instruments import _colour, _mono


def _stable_unit(text: str) -> float:
    """A repeatable number in [0, 1) for *text* (positions that never jump)."""
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


class _CachedInstrument(QWidget):
    """A widget whose still layer is drawn once per size and data version."""

    FRAME_MS = 66

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._version = 0
        self._cache: QPixmap | None = None
        self._cache_key: tuple[Any, ...] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self._frame)

    def _changed(self) -> None:
        self._version += 1
        self.update()

    def _still(self) -> QPixmap:
        key = (self.width(), self.height(), self.devicePixelRatioF(), self._version)
        if self._cache is None or self._cache_key != key:
            ratio = max(1.0, self.devicePixelRatioF())
            pixmap = QPixmap(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                self._draw_still(painter)
            finally:
                painter.end()
            self._cache, self._cache_key = pixmap, key
        return self._cache

    def _draw_still(self, painter: QPainter) -> None:  # pragma: no cover - overridden
        pass

    def _draw_moving(self, painter: QPainter) -> None:
        pass

    def _animating(self) -> bool:
        return False

    def _frame(self) -> None:
        if not self.isVisible():
            self._timer.stop()
            return
        self._advance()
        self.update()
        if not self._animating():
            self._timer.stop()

    def _advance(self) -> None:
        pass

    def _wake(self) -> None:
        if self.isVisible() and not self._timer.isActive():
            self._timer.start()

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        if self.width() < 40 or self.height() < 40:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.drawPixmap(0, 0, self._still())
            self._draw_moving(painter)
        finally:
            painter.end()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        if self._animating():
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def _ui(self, px: int, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(px)
        font.setBold(bold)
        return font


# ── the voice ridgeline ──────────────────────────────────────────────────────

RIDGE_BANDS = 48
RIDGE_ROWS = 22
RIDGE_LOW_HZ, RIDGE_HIGH_HZ = 80.0, 7000.0


def mic_spectrum(pcm: bytes, rate: int = 16_000, bands: int = RIDGE_BANDS) -> list[float]:
    """The newest 64 ms of microphone audio as *bands* levels in [0, 1],
    log-spaced from 80 Hz to 7 kHz (where speech lives)."""
    import numpy as np
    count = len(pcm) // 2
    if count < 256:
        return [0.0] * bands
    samples = np.frombuffer(pcm[-2 * min(count, 1024):], dtype=np.int16).astype(np.float32) / 32768.0
    window = np.hanning(len(samples))
    magnitude = np.abs(np.fft.rfft(samples * window)) / (len(samples) / 4.0)
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    edges = np.geomspace(RIDGE_LOW_HZ, RIDGE_HIGH_HZ, bands + 1)
    # A band's level is its strongest bin; a band narrower than one bin
    # (the lowest ones) reads the spectrum at its centre instead.
    level = np.interp(np.sqrt(edges[:-1] * edges[1:]), freqs, magnitude)
    starts = np.searchsorted(freqs, edges)
    for band in range(bands):
        low, high = starts[band], starts[band + 1]
        if high > low:
            level[band] = max(level[band], float(magnitude[low:high].max()))
    level = np.convolve(level, [0.25, 0.5, 0.25], mode="same")
    decibels = 20.0 * np.log10(level + 1e-9)
    return [float(v) for v in np.clip((decibels + 62.0) / 48.0, 0.0, 1.0)]


def voice_spectrum(bands_in: dict, amplitude: float, bands: int = RIDGE_BANDS) -> list[float]:
    """ORION's voice as *bands* levels, shaped from the low/mid/high energies
    the playback visualiser already measures (audio._VisualiserThread)."""
    low = float(bands_in.get("low", 0.0) or 0.0)
    mid = float(bands_in.get("mid", 0.0) or 0.0)
    high = float(bands_in.get("high", 0.0) or 0.0)
    loud = max(0.0, min(1.0, float(amplitude or 0.0) * 1.6))
    out = []
    for index in range(bands):
        t = index / max(1, bands - 1)
        shape = (low * math.exp(-((t - 0.16) / 0.13) ** 2) + mid * math.exp(-((t - 0.45) / 0.16) ** 2)
                 + high * math.exp(-((t - 0.76) / 0.15) ** 2))
        out.append(max(0.0, min(1.0, shape * loud * 1.4)))
    return out


def _recent_microphone() -> bytes:
    try:
        from ..audio import RECENT_AUDIO
    except Exception:
        return b""
    if not RECENT_AUDIO.fresh(0.6):
        return b""
    return RECENT_AUDIO.snapshot(0.08)


class VoiceRidgeline(_CachedInstrument):
    """The owner's voice (crimson) and ORION's (white) as a moving ridgeline:
    the newest line at the front, each older one a step further back."""

    def __init__(self, bus: Any = None, parent: QWidget | None = None,
                 source: Callable[[], bytes] | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setAccessibleName("Voice ridgeline")
        self.source = source or _recent_microphone
        self.rows: deque[tuple[list[float], list[float]]] = deque(maxlen=RIDGE_ROWS)
        self._you = [0.0] * RIDGE_BANDS
        self._bands: dict = {}
        self._amplitude = 0.0
        self._quiet = 0
        if bus is not None:
            for name, slot in (("voice_spectrum", self._on_bands), ("amplitude", self._on_amplitude)):
                signal = getattr(bus, name, None)
                if signal is not None:
                    signal.connect(slot)

    def _on_bands(self, bands: Any) -> None:
        if isinstance(bands, dict):
            self._bands = bands

    def _on_amplitude(self, value: float) -> None:
        self._amplitude = float(value or 0.0)
        if self._amplitude > 0.01:
            self._wake()

    def _animating(self) -> bool:
        return True

    def _advance(self) -> None:
        heard = mic_spectrum(self.source() or b"")
        # Fast to rise, slow to fall, as a level meter does.
        self._you = [max(h, y * 0.8) for h, y in zip(heard, self._you)]
        own = voice_spectrum(self._bands, self._amplitude)
        self._amplitude *= 0.9
        self.rows.append((list(self._you), own))
        loud = max(max(self._you), max(own))
        self._quiet = self._quiet + 1 if loud < 0.02 else 0

    def update(self, *args: Any) -> None:  # noqa: D401 - Qt override
        # A silent ridgeline stops repainting once it has settled flat.
        if self._quiet <= RIDGE_ROWS:
            super().update(*args)

    def _draw_still(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), _colour(C.INK))
        painter.setFont(_mono(8))
        painter.setPen(_colour(C.FAINT))
        left, right, bottom = 54.0, self.width() - 12.0, self.height() - 8.0
        for label, hz in (("100 Hz", 100.0), ("1 kHz", 1000.0), ("5 kHz", 5000.0)):
            t = math.log(hz / RIDGE_LOW_HZ) / math.log(RIDGE_HIGH_HZ / RIDGE_LOW_HZ)
            x = left + t * (right - left)
            painter.drawText(QRectF(x - 24, bottom - 12, 48, 12), int(Qt.AlignmentFlag.AlignCenter), label)
        for index, (label, colour) in enumerate((("You", C.PRI), ("ORION", C.ACCENT))):
            y = 14 + index * 16
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(colour))
            painter.drawEllipse(QPointF(12, y), 3.0, 3.0)
            painter.setPen(_colour(C.SILVER))
            painter.drawText(QRectF(20, y - 7, 40, 14),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)

    def _draw_moving(self, painter: QPainter) -> None:
        if not self.rows:
            return
        left, right = 54.0, self.width() - 12.0
        top, bottom = 18.0, self.height() - 26.0
        step = (bottom - top) / RIDGE_ROWS
        height = step * 3.4
        count = len(self.rows)
        for index, (you, own) in enumerate(self.rows):
            depth = (count - 1 - index) / max(1, RIDGE_ROWS - 1)      # 0 front .. 1 back
            base = bottom - (count - 1 - index) * step
            values = [max(a, b) for a, b in zip(you, own)]
            colour = C.ACCENT if sum(own) > sum(you) else C.PRI
            path = QPainterPath(QPointF(left, base))
            for band, value in enumerate(values):
                x = left + band / (RIDGE_BANDS - 1) * (right - left)
                path.lineTo(QPointF(x, base - value * height))
            path.lineTo(QPointF(right, base))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(C.INK))
            painter.drawPath(path)                          # hides the lines behind
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(_colour(colour, 0.95 - 0.8 * depth), 1.5 if depth < 0.05 else 1.1))
            painter.drawPath(path)


# ── the reactor gauge ────────────────────────────────────────────────────────

REACTOR_RINGS = (("CPU", "cpu", C.PRI), ("GPU", "gpu", C.PRI_HI),
                 ("RAM", "ram", C.CORE), ("Network", "net_percent", C.ACCENT_DIM))


class ReactorGauge(_CachedInstrument):
    """CPU, GPU, RAM and network as four nested arcs, eased between samples."""

    FRAME_MS = 33

    def __init__(self, bus: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 200)
        self.setAccessibleName("System load gauge")
        self.target = {key: 0.0 for _label, key, _colour in REACTOR_RINGS}
        self.shown = dict(self.target)
        self.available = {key: True for key in self.target}
        if bus is not None and getattr(bus, "telemetry_sample", None) is not None:
            bus.telemetry_sample.connect(self.set_sample)

    def set_sample(self, sample: Any) -> None:
        if not isinstance(sample, dict):
            return
        for _label, key, _colour_name in REACTOR_RINGS:
            value = sample.get(key)
            self.available[key] = value is not None
            self.target[key] = max(0.0, min(100.0, float(value or 0.0)))
        self.setAccessibleDescription(", ".join(
            f"{label} {self.target[key]:.0f}%" for label, key, _c in REACTOR_RINGS if self.available[key]))
        self._wake()

    def _animating(self) -> bool:
        return any(abs(self.shown[k] - self.target[k]) > 0.2 for k in self.target)

    def _advance(self) -> None:
        for key in self.shown:
            self.shown[key] += (self.target[key] - self.shown[key]) * 0.25

    def _geometry(self) -> tuple[QPointF, float]:
        size = min(self.width() * 0.62, self.height()) - 16
        return QPointF(8 + size / 2 + 8, self.height() / 2), max(30.0, size / 2)

    def _draw_still(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index in range(len(REACTOR_RINGS)):
            r = radius - index * radius * 0.2
            painter.setPen(QPen(_colour(C.BORDER), radius * 0.11, cap=Qt.PenCapStyle.FlatCap))
            painter.drawEllipse(centre, r, r)

    def _draw_moving(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        for index, (_label, key, colour) in enumerate(REACTOR_RINGS):
            if not self.available[key]:
                continue
            r = radius - index * radius * 0.2
            span = -self.shown[key] / 100.0 * 360.0 * 16
            pen = QPen(_colour(colour), radius * 0.11)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r), 90 * 16, int(span))
        painter.setPen(_colour(C.WHITE))
        painter.setFont(_mono(max(12, int(radius * 0.32))))
        painter.drawText(QRectF(centre.x() - radius, centre.y() - radius * 0.3, 2 * radius, radius * 0.4),
                         int(Qt.AlignmentFlag.AlignCenter), f"{self.shown['cpu']:.0f}%")
        painter.setFont(_mono(9))
        painter.setPen(_colour(C.FAINT))
        painter.drawText(QRectF(centre.x() - radius, centre.y() + radius * 0.08, 2 * radius, 14),
                         int(Qt.AlignmentFlag.AlignCenter), "CPU")
        x = centre.x() + radius + 22
        for index, (label, key, colour) in enumerate(REACTOR_RINGS):
            y = centre.y() - 34 + index * 22
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(colour if self.available[key] else C.BORDER))
            painter.drawRoundedRect(QRectF(x, y - 4, 10, 8), 2, 2)
            painter.setPen(_colour(C.SILVER))
            painter.setFont(self._ui(11))
            painter.drawText(QRectF(x + 16, y - 8, 70, 16),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)
            painter.setFont(_mono(11))
            painter.setPen(_colour(C.WHITE if self.available[key] else C.FAINT))
            painter.drawText(QRectF(x + 84, y - 8, 50, 16),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             f"{self.shown[key]:.0f}%" if self.available[key] else "--")


# ── the memory constellation ─────────────────────────────────────────────────

#: Orion's brightest stars (x, y in a unit box) and the lines between them.
ORION_STARS = (("Meissa", 0.48, 0.10), ("Betelgeuse", 0.22, 0.24), ("Bellatrix", 0.74, 0.27),
               ("Alnitak", 0.40, 0.53), ("Alnilam", 0.50, 0.50), ("Mintaka", 0.60, 0.47),
               ("Saiph", 0.30, 0.88), ("Rigel", 0.80, 0.85))
ORION_LINES = ((0, 1), (0, 2), (1, 3), (2, 5), (3, 4), (4, 5), (3, 6), (5, 7))


def _category_name(category: str) -> str:
    """'pack_ai' -> 'Pack AI': the deck's heading rules (acronyms kept)."""
    from .humane import heading_case
    return heading_case(str(category).replace("_", " "))


class MemoryConstellation(_CachedInstrument):
    """His memory by category: the eight largest are Orion's bright stars
    (bigger for more memories), the rest a faint field around them."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setAccessibleName("Memory constellation")
        self.categories: list[tuple[str, int]] = []
        self._points: list[tuple[QPointF, float, str, int]] = []

    def set_categories(self, counts: Iterable[tuple[str, int]]) -> None:
        self.categories = sorted(((str(n), int(c)) for n, c in counts if int(c) > 0),
                                 key=lambda item: (-item[1], item[0]))
        self.setAccessibleDescription(", ".join(f"{n} {c}" for n, c in self.categories[:8]))
        self._changed()

    def _box(self) -> QRectF:
        return QRectF(24, 18, max(1.0, self.width() - 48), max(1.0, self.height() - 36))

    def _draw_still(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), _colour(C.INK))
        self._points = []
        if not self.categories:
            painter.setPen(_colour(C.FAINT))
            painter.setFont(self._ui(12))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "No Memories Yet")
            return
        box = self._box()
        biggest = max(count for _n, count in self.categories)
        for name, count in self.categories[8:]:                      # the faint field
            point = QPointF(box.left() + _stable_unit(name) * box.width(),
                            box.top() + _stable_unit(name[::-1] + "y") * box.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(C.ACCENT_DIM, 0.55))
            painter.drawEllipse(point, 1.6, 1.6)
            self._points.append((point, 4.0, name, count))
        stars = []
        for (name, count), (_star, x, y) in zip(self.categories[:8], ORION_STARS):
            stars.append((QPointF(box.left() + x * box.width(), box.top() + y * box.height()), name, count))
        painter.setPen(QPen(_colour(C.ACCENT_DEEP), 1.0))
        for a, b in ORION_LINES:
            if a < len(stars) and b < len(stars):
                painter.drawLine(stars[a][0], stars[b][0])
        painter.setFont(_mono(9))
        for rank, (point, name, count) in enumerate(stars):
            size = 2.5 + 6.5 * math.sqrt(count / biggest)
            colour = C.PRI if rank == 0 else C.WHITE if rank == 1 else C.PRI_HI
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(colour, 0.18))
            painter.drawEllipse(point, size * 2.2, size * 2.2)
            painter.setBrush(_colour(colour))
            painter.drawEllipse(point, size, size)
            painter.setPen(_colour(C.SILVER))
            label = f"{_category_name(name)}  {count}"
            painter.drawText(QRectF(point.x() + size + 6, point.y() - 7, 180, 14),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)
            self._points.append((point, size + 6, name, count))

    def star_at(self, point: QPointF) -> tuple[str, int] | None:
        self._still()
        for centre, reach, name, count in self._points:
            if math.hypot(point.x() - centre.x(), point.y() - centre.y()) <= reach:
                return name, count
        return None

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        found = self.star_at(event.position())
        if found is None:
            QToolTip.hideText()
        else:
            QToolTip.showText(event.globalPosition().toPoint(),
                              f"{_category_name(found[0])}: {found[1]} memories", self)


# ── the day ring ─────────────────────────────────────────────────────────────

def day_activity(*, focus_store: Any = None, study_store: Any = None, log_path: Any = None,
                 now: datetime | None = None) -> dict[str, list]:
    """Today, as minutes after local midnight: what he and the owner did.

    conversations - each line the owner said (session.log "YOU:" lines)
    focus         - (start, end, kind) for each focus session
    study         - each flashcard review
    """
    now = now or datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    out: dict[str, list] = {"conversations": [], "focus": [], "study": []}

    def minute(moment: datetime) -> float:
        if moment.tzinfo is not None:
            moment = moment.astimezone().replace(tzinfo=None)
        return (moment - midnight).total_seconds() / 60.0

    if focus_store is not None:
        try:
            for session in focus_store.since(midnight.astimezone()):
                start = minute(datetime.fromisoformat(session.started_at))
                end = minute(datetime.fromisoformat(session.ended_at)) if session.ended_at else minute(now)
                out["focus"].append((max(0.0, start), min(1440.0, end), str(session.kind or "deep")))
        except Exception:
            pass
    if study_store is not None:
        try:
            out["study"] = [m for m in (minute(at) for at in study_store.reviews_since(midnight.astimezone()))
                            if 0 <= m <= 1440]
        except Exception:
            pass
    if log_path is not None:
        prefix = midnight.strftime("%Y-%m-%d ")
        try:
            from pathlib import Path
            with Path(log_path).open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith(prefix) and line[21:26] == "YOU: ":
                        stamp = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                        out["conversations"].append(minute(stamp))
        except (OSError, ValueError):
            pass
    return out


class DayRing(_CachedInstrument):
    """A 24-hour clock of today: conversations on the outer track, focus
    sessions in the middle, flashcard reviews inside, and the hour hand."""

    TRACKS = (("Conversations", C.ACCENT_DIM), ("Focus", C.PRI), ("Study", C.PRI_HI))

    def __init__(self, parent: QWidget | None = None, clock: Callable[[], datetime] = datetime.now) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 220)
        self.setAccessibleName("Today as a 24-hour ring")
        self.clock = clock
        self.activity: dict[str, list] = {"conversations": [], "focus": [], "study": []}

    def set_activity(self, activity: dict[str, list]) -> None:
        self.activity = {key: list(activity.get(key) or []) for key in ("conversations", "focus", "study")}
        focus = sum(max(0.0, e - s) for s, e, _k in self.activity["focus"])
        self.setAccessibleDescription(
            f"{len(self.activity['conversations'])} messages, {focus:.0f} focus minutes, "
            f"{len(self.activity['study'])} reviews today")
        self._changed()

    def _geometry(self) -> tuple[QPointF, float]:
        size = min(self.width() * 0.6, self.height()) - 20
        return QPointF(10 + size / 2 + 10, self.height() / 2), max(40.0, size / 2)

    @staticmethod
    def _angle(minute: float) -> float:
        """Qt degrees (counter-clockwise from three o'clock), midnight at the top."""
        return 90.0 - minute / 1440.0 * 360.0

    def _draw_still(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        widths = radius * 0.13
        tracks = [radius - i * radius * 0.22 for i in range(3)]
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for r in tracks:
            painter.setPen(QPen(_colour(C.BORDER), widths))
            painter.drawEllipse(centre, r, r)
        painter.setFont(_mono(9))
        for hour in range(24):
            angle = math.radians(self._angle(hour * 60))
            outer = radius + widths
            inner = outer + (7 if hour % 6 == 0 else 3)
            painter.setPen(QPen(_colour(C.FAINT if hour % 6 else C.SILVER), 1))
            painter.drawLine(QPointF(centre.x() + outer * math.cos(angle), centre.y() - outer * math.sin(angle)),
                             QPointF(centre.x() + inner * math.cos(angle), centre.y() - inner * math.sin(angle)))
            if hour % 6 == 0:
                at = inner + 9
                painter.drawText(QRectF(centre.x() + at * math.cos(angle) - 12,
                                        centre.y() - at * math.sin(angle) - 7, 24, 14),
                                 int(Qt.AlignmentFlag.AlignCenter), f"{hour:02d}")

        def arc(r: float, start: float, end: float, colour: str, alpha: float = 1.0) -> None:
            pen = QPen(_colour(colour, alpha), widths)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            span = -max(0.8, (end - start) / 1440.0 * 360.0)
            painter.drawArc(QRectF(centre.x() - r, centre.y() - r, 2 * r, 2 * r),
                            int(self._angle(start) * 16), int(span * 16))

        slots: dict[int, int] = {}
        for minute in self.activity["conversations"]:
            slots[int(minute // 10)] = slots.get(int(minute // 10), 0) + 1
        busiest = max(slots.values(), default=1)
        for slot, count in slots.items():
            arc(tracks[0], slot * 10, slot * 10 + 10, C.ACCENT, 0.35 + 0.65 * count / busiest)
        for start, end, _kind in self.activity["focus"]:
            arc(tracks[1], start, end, C.PRI)
        for minute in self.activity["study"]:
            arc(tracks[2], minute - 1, minute + 1, C.PRI_HI)
        x = centre.x() + radius + 30
        focus = sum(max(0.0, e - s) for s, e, _k in self.activity["focus"])
        rows = ((self.TRACKS[0], f"{len(self.activity['conversations'])}"),
                (self.TRACKS[1], f"{int(focus) // 60} h {int(focus) % 60} m"),
                (self.TRACKS[2], f"{len(self.activity['study'])}"))
        for index, ((label, colour), value) in enumerate(rows):
            y = centre.y() - 30 + index * 26
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(colour))
            painter.drawRoundedRect(QRectF(x, y - 4, 10, 8), 2, 2)
            painter.setPen(_colour(C.SILVER))
            painter.setFont(self._ui(11))
            painter.drawText(QRectF(x + 16, y - 8, 100, 16),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), label)
            painter.setFont(_mono(11))
            painter.setPen(_colour(C.WHITE))
            painter.drawText(QRectF(x + 112, y - 8, 70, 16),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), value)

    def _draw_moving(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        now = self.clock()
        angle = math.radians(self._angle(now.hour * 60 + now.minute + now.second / 60.0))
        tip = QPointF(centre.x() + (radius + 4) * math.cos(angle), centre.y() - (radius + 4) * math.sin(angle))
        painter.setPen(QPen(_colour(C.WHITE), 1.6))
        painter.drawLine(centre, tip)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_colour(C.WHITE))
        painter.drawEllipse(centre, 3.2, 3.2)
        painter.drawEllipse(tip, 2.6, 2.6)
        painter.setFont(_mono(max(11, int(radius * 0.2))))
        painter.setPen(_colour(C.WHITE))
        painter.drawText(QRectF(centre.x() - radius, centre.y() + radius * 0.12, 2 * radius, radius * 0.3),
                         int(Qt.AlignmentFlag.AlignCenter), now.strftime("%H:%M"))


# ── the tool radar ───────────────────────────────────────────────────────────

class ToolRadar(_CachedInstrument):
    """Every tool he has used, placed round a radar by name (it never moves)
    and nearer the centre the more he uses it; a sweep turns, and a tool
    pings the moment he calls it."""

    PING_S = 1.6
    SWEEP_S = 4.0

    def __init__(self, bus: Any = None, parent: QWidget | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setAccessibleName("Tool radar")
        self.clock = clock
        self.counts: dict[str, int] = {}
        self.pings: dict[str, float] = {}
        #: The current research question while a real run is in progress
        #: (neural_stats.ResearchActivity), "" otherwise (Mark XXXX.9).
        self.research = ""
        if bus is not None and getattr(bus, "log", None) is not None:
            bus.log.connect(self._on_log)

    def set_counts(self, counts: dict[str, int]) -> None:
        self.counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
        self._changed()

    def set_research(self, caption: str) -> None:
        """Show research in progress - a slow pulse at the centre and its
        question along the foot - or clear it ("" when the run finishes)."""
        caption = " ".join(str(caption or "").split())[:160]
        if caption == self.research:
            return
        self.research = caption
        self._changed()
        if caption:
            self._wake()

    def _on_log(self, message: str) -> None:
        text = str(message)
        if " TOOL_EXECUTION_START" in text and " — " in text:
            for name in text.split(" — ", 1)[1].replace(",", " ").split():
                self.ping(name.strip())

    def ping(self, tool: str) -> None:
        if not tool:
            return
        self.counts[tool] = self.counts.get(tool, 0) + 1
        self.pings[tool] = self.clock()
        self._changed()
        self._wake()

    def _animating(self) -> bool:
        return True

    def _geometry(self) -> tuple[QPointF, float]:
        radius = max(30.0, min(self.width(), self.height()) / 2 - 18)
        return QPointF(self.width() / 2, self.height() / 2), radius

    def position(self, tool: str) -> QPointF:
        centre, radius = self._geometry()
        biggest = max(self.counts.values(), default=1)
        share = self.counts.get(tool, 0) / biggest
        r = radius * (0.92 - 0.72 * math.sqrt(share))
        angle = _stable_unit(tool) * 2 * math.pi
        return QPointF(centre.x() + r * math.cos(angle), centre.y() - r * math.sin(angle))

    def _draw_still(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for fraction in (0.33, 0.66, 1.0):
            painter.setPen(QPen(_colour(C.BORDER), 1))
            painter.drawEllipse(centre, radius * fraction, radius * fraction)
        painter.drawLine(QPointF(centre.x() - radius, centre.y()), QPointF(centre.x() + radius, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - radius), QPointF(centre.x(), centre.y() + radius))
        top = sorted(self.counts, key=lambda k: -self.counts[k])[:5]
        painter.setFont(_mono(8))
        for tool in self.counts:
            point = self.position(tool)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(C.PRI_DIM if tool not in top else C.PRI, 0.9))
            painter.drawEllipse(point, 2.4 if tool not in top else 3.2, 2.4 if tool not in top else 3.2)
            if tool in top:
                painter.setPen(_colour(C.SILVER))
                painter.drawText(QRectF(point.x() + 6, point.y() - 7, 140, 14),
                                 int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 tool.replace("_", " "))

    def _draw_moving(self, painter: QPainter) -> None:
        centre, radius = self._geometry()
        now = self.clock()
        heading = (now % self.SWEEP_S) / self.SWEEP_S * 360.0
        wedge = QPainterPath(centre)
        wedge.arcTo(QRectF(centre.x() - radius, centre.y() - radius, 2 * radius, 2 * radius),
                    -heading, 28.0)
        wedge.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_colour(C.PRI, 0.16))
        painter.drawPath(wedge)
        edge = math.radians(-heading)
        painter.setPen(QPen(_colour(C.PRI, 0.85), 1.4))
        painter.drawLine(centre, QPointF(centre.x() + radius * math.cos(edge),
                                         centre.y() - radius * math.sin(edge)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for tool, at in list(self.pings.items()):
            age = now - at
            if age > self.PING_S:
                del self.pings[tool]
                continue
            phase = age / self.PING_S
            painter.setPen(QPen(_colour(C.CORE, 0.9 * (1 - phase)), 1.6))
            painter.drawEllipse(self.position(tool), 4 + 18 * phase, 4 + 18 * phase)
        if self.research:
            # Research in progress: a ring breathing out from the centre, and
            # the question being worked on along the foot.
            phase = (now % 2.4) / 2.4
            painter.setPen(QPen(_colour(C.PRI_HI, 0.8 * (1 - phase)), 1.8))
            painter.drawEllipse(centre, radius * (0.1 + 0.85 * phase), radius * (0.1 + 0.85 * phase))
            painter.setFont(_mono(8))
            painter.setPen(_colour(C.SILVER))
            band = QRectF(4, self.height() - 16, self.width() - 8, 14)
            painter.drawText(band, int(Qt.AlignmentFlag.AlignCenter),
                             QFontMetricsF(painter.font()).elidedText(
                                 "Researching: " + self.research, Qt.TextElideMode.ElideRight,
                                 band.width()))


# ── the market heat map ──────────────────────────────────────────────────────

class MarketHeatMap(_CachedInstrument):
    """Every watched market as a tile: a row per group, a wider tile for a
    bigger move, white for up and crimson for down, deeper for more."""

    def __init__(self, parent: QWidget | None = None,
                 on_pick: Callable[[str, str], Any] | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setAccessibleName("Market heat map")
        self.on_pick = on_pick
        self.groups: dict[str, list[Any]] = {}
        self._tiles: list[tuple[QRectF, str, str]] = []

    def set_quotes(self, groups: dict[str, list[Any]]) -> None:
        self.groups = {name: list(rows) for name, rows in groups.items() if rows}
        self._changed()

    def _draw_still(self, painter: QPainter) -> None:
        from ..markets import price_text
        self._tiles = []
        if not self.groups:
            painter.setPen(_colour(C.FAINT))
            painter.setFont(self._ui(12))
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Markets Unavailable")
            return
        rows = list(self.groups.items())
        gap, label_w = 4.0, 70.0
        height = (self.height() - gap * (len(rows) - 1)) / len(rows)
        biggest = max((abs(q.change or 0.0) for _n, quotes in rows for q in quotes), default=1.0) or 1.0
        for index, (group, quotes) in enumerate(rows):
            y = index * (height + gap)
            painter.setPen(_colour(C.MUTED))
            painter.setFont(self._ui(11, bold=True))
            painter.drawText(QRectF(0, y, label_w - 8, height),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), group)
            weights = [0.6 + abs(q.change or 0.0) / biggest for q in quotes]
            total = sum(weights)
            x = label_w
            width = self.width() - label_w
            for quote, weight in zip(quotes, weights):
                w = width * weight / total
                tile = QRectF(x, y, w - gap, height)
                change = quote.change or 0.0
                strength = 0.18 + 0.7 * min(1.0, abs(change) / biggest)
                fill = C.ACCENT if change >= 0 else C.PRI
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_colour(C.PANEL_HI))
                painter.drawRoundedRect(tile, 5, 5)
                painter.setBrush(_colour(fill, strength * (0.55 if change >= 0 else 1.0)))
                painter.drawRoundedRect(tile, 5, 5)
                inner = tile.adjusted(8, 6, -8, -6)
                painter.setPen(_colour(C.WHITE))
                painter.setFont(self._ui(11, bold=True))
                painter.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                                 painter.fontMetrics().elidedText(quote.label, Qt.TextElideMode.ElideRight,
                                                                  int(inner.width())))
                painter.setFont(_mono(12))
                painter.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                                 f"{change:+.2f}%" if quote.change is not None else "--")
                painter.setFont(_mono(9))
                painter.setPen(_colour(C.SILVER))
                painter.drawText(inner, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom),
                                 price_text(quote))
                self._tiles.append((tile, group, quote.symbol))
                x += w

    def tile_at(self, point: QPointF) -> tuple[str, str] | None:
        self._still()
        for rect, group, symbol in self._tiles:
            if rect.contains(point):
                return group, symbol
        return None

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        hit = self.tile_at(event.position())
        if hit is not None and self.on_pick is not None:
            self.on_pick(*hit)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.PointingHandCursor if self.tile_at(event.position())
                       else Qt.CursorShape.ArrowCursor)


__all__ = ["DayRing", "MarketHeatMap", "MemoryConstellation", "ReactorGauge", "ToolRadar",
           "VoiceRidgeline", "day_activity", "mic_spectrum", "voice_spectrum"]
