"""
Markets as a chart: indices, ETFs and stocks over today's session.

Owner, 28 Oct 2026: the stocks, the ETFs and the FTSE "must represent as an
amazing graph". One instrument is drawn large - today's line against the
previous close, the session's hours along the bottom, the price on the right
- and the rest of its group sit beside it as small lines to pick from.
Rising is cold white and falling is crimson, the two lights the interface
already uses; the arrow says the same in shape.

Cost: everything except the live dot and the crosshair is drawn once into a
pixmap when the data, the choice or the size changes; a frame is that pixmap,
the dot's pulse and (while the mouse is over the plot) the crosshair.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C
from ..markets import Quote, price_text
from .brain_instruments import _colour, _mono

GROUP_ORDER = ("Indices", "ETFs", "Stocks")
_STATE_TEXT = {"open": "Open", "closed": "Closed", "previous": "Last Session"}


def tone(quote: Quote) -> str:
    """The light a quote is drawn in: white rising, crimson falling."""
    if quote.change is None:
        return C.ACCENT_DIM
    return C.ACCENT if quote.change >= 0 else C.PRI


def change_text(quote: Quote) -> str:
    if quote.change is None:
        return "--"
    arrow = "▲" if quote.change > 0.005 else "▼" if quote.change < -0.005 else "■"
    return f"{arrow} {quote.change:+.2f}%"


def domain(quote: Quote) -> tuple[float, float]:
    """The time span the plot covers: the whole regular session when the
    prices are from it (so the line grows across the day), else the prices'."""
    times = quote.times
    if not times:
        return 0.0, 1.0
    lo, hi = float(times[0]), float(times[-1])
    if quote.period is not None:
        start, end = quote.period
        if start - 3600 <= times[0] <= end:
            lo, hi = min(lo, float(start)), max(hi, float(end))
    return lo, max(hi, lo + 60.0)


def value_range(quote: Quote) -> tuple[float, float]:
    values = list(quote.points)
    if quote.previous:
        values.append(quote.previous)
    if not values:
        return quote.price - 1.0, quote.price + 1.0
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.12, abs(hi) * 0.0005, 1e-6)
    return lo - pad, hi + pad


class MarketChart(QWidget):
    """Today's markets: one instrument large, its group beside it."""

    FRAME_MS = 66
    CHIP_H = 26.0
    HEADER_H = 50.0
    AXIS_W = 70.0
    TIME_H = 18.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Markets chart")
        self.groups: dict[str, list[Quote]] = {}
        self.group = GROUP_ORDER[0]
        self.symbol = ""
        self.status = "Loading"
        self._version = 0
        self._cache: QPixmap | None = None
        self._cache_key: tuple[Any, ...] | None = None
        self._hover: QPointF | None = None
        self._hits: list[tuple[QRectF, str, str]] = []
        self._plot = QRectF()
        self._timer = QTimer(self)
        self._timer.setInterval(self.FRAME_MS)
        self._timer.timeout.connect(self.update)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_quotes(self, groups: dict[str, list[Quote]]) -> None:
        self.groups = {name: list(rows) for name, rows in groups.items() if rows}
        self.status = "" if self.groups else "Markets Unavailable"
        if self.group not in self.groups and self.groups:
            self.group = next(iter(self._group_names()))
        if self.selected() is None:
            rows = self.groups.get(self.group) or []
            self.symbol = rows[0].symbol if rows else ""
        self._changed()

    def set_status(self, text: str) -> None:
        self.status = text
        self._changed()

    def choose(self, *, group: str | None = None, symbol: str | None = None) -> None:
        if group is not None and group in self.groups and group != self.group:
            self.group = group
            rows = self.groups[group]
            self.symbol = rows[0].symbol if rows else ""
        if symbol is not None:
            self.symbol = symbol
        self._changed()

    def selected(self) -> Quote | None:
        for quote in self.groups.get(self.group) or []:
            if quote.symbol == self.symbol:
                return quote
        return None

    def _group_names(self) -> list[str]:
        known = [name for name in GROUP_ORDER if name in self.groups]
        return known + [name for name in self.groups if name not in known]

    def _changed(self) -> None:
        self._version += 1
        quote = self.selected()
        if quote is not None:
            self.setAccessibleDescription(
                f"{quote.label} {price_text(quote)} {change_text(quote)}")
        self.update()

    # ── geometry ─────────────────────────────────────────────────────────────

    def _areas(self) -> tuple[QRectF, QRectF]:
        top = self.CHIP_H + 10
        side_w = min(280.0, max(190.0, self.width() * 0.3)) if self.width() >= 620 else 0.0
        gap = 14.0 if side_w else 0.0
        main = QRectF(0, top, max(1.0, self.width() - side_w - gap), max(1.0, self.height() - top))
        side = QRectF(main.right() + gap, top, side_w, main.height())
        return main, side

    def _x(self, at: float, span: tuple[float, float]) -> float:
        lo, hi = span
        return self._plot.left() + (at - lo) / (hi - lo) * self._plot.width()

    def _y(self, value: float, span: tuple[float, float]) -> float:
        lo, hi = span
        return self._plot.bottom() - (value - lo) / (hi - lo) * self._plot.height()

    # ── the still part, drawn once ───────────────────────────────────────────

    def _render(self) -> QPixmap:
        ratio = max(1.0, self.devicePixelRatioF())
        pixmap = QPixmap(max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        self._hits = []
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_chips(painter)
            main, side = self._areas()
            quote = self.selected()
            if quote is None:
                self._plot = QRectF()
                painter.setPen(_colour(C.FAINT))
                painter.setFont(self._ui(12))
                painter.drawText(main, int(Qt.AlignmentFlag.AlignCenter), self.status or "Loading")
            else:
                self._draw_header(painter, main, quote)
                self._draw_plot(painter, main, quote)
            if side.width() > 0:
                self._draw_side(painter, side)
        finally:
            painter.end()
        return pixmap

    def _ui(self, px: int, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPixelSize(px)
        font.setBold(bold)
        return font

    def _draw_chips(self, painter: QPainter) -> None:
        x = 0.0
        painter.setFont(self._ui(11, bold=True))
        metrics = painter.fontMetrics()
        for name in self._group_names() or list(GROUP_ORDER):
            width = metrics.horizontalAdvance(name) + 26.0
            rect = QRectF(x, 0, width, self.CHIP_H)
            chosen = name == self.group
            painter.setPen(QPen(_colour(C.PRI if chosen else C.BORDER, 0.75 if chosen else 1.0), 1))
            painter.setBrush(_colour(C.PANEL_HI if chosen else C.INK))
            painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 13, 13)
            painter.setPen(_colour(C.WHITE if chosen else C.MUTED))
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), name)
            self._hits.append((rect, "group", name))
            x += width + 8.0

    def _draw_header(self, painter: QPainter, main: QRectF, quote: Quote) -> None:
        painter.setPen(_colour(C.SILVER))
        painter.setFont(self._ui(13, bold=True))
        painter.drawText(QRectF(main.left(), main.top(), main.width() * 0.6, 18),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), quote.label)
        painter.setFont(_mono(22))
        painter.setPen(_colour(C.WHITE))
        price = price_text(quote)
        painter.drawText(QRectF(main.left(), main.top() + 20, main.width(), 28),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), price)
        offset = painter.fontMetrics().horizontalAdvance(price) + 14
        painter.setFont(_mono(13))
        painter.setPen(_colour(tone(quote)))
        painter.drawText(QRectF(main.left() + offset, main.top() + 20, 200, 28),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                         change_text(quote))
        painter.setFont(self._ui(10, bold=True))
        state = _STATE_TEXT.get(quote.state, "")
        painter.setPen(_colour(C.PRI_HI if quote.state == "open" else C.FAINT))
        painter.drawText(QRectF(main.left(), main.top(), main.width() - 4, 18),
                         int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                         ("● " if quote.state == "open" else "") + state.upper())

    def _draw_plot(self, painter: QPainter, main: QRectF, quote: Quote) -> None:
        self._plot = QRectF(main.left() + 2, main.top() + self.HEADER_H + 8,
                            max(1.0, main.width() - 2 - self.AXIS_W),
                            max(1.0, main.height() - self.HEADER_H - 8 - self.TIME_H))
        plot = self._plot
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_colour(C.INK, 0.85))
        painter.drawRoundedRect(plot.adjusted(-2, -2, 2, 2), 6, 6)
        span_t, span_v = domain(quote), value_range(quote)
        # Price grid and the price axis on the right.
        painter.setFont(_mono(9))
        for step in range(5):
            value = span_v[0] + (span_v[1] - span_v[0]) * step / 4
            y = self._y(value, span_v)
            painter.setPen(QPen(_colour(C.ACCENT, 0.05), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(_colour(C.FAINT))
            painter.drawText(QRectF(plot.right() + 8, y - 7, self.AXIS_W - 10, 14),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             price_text(quote, value))
        # The session's hours along the bottom.
        if quote.times:
            hours = (span_t[1] - span_t[0]) / 3600.0
            step = 3600 * (1 if plot.width() / max(hours, 0.1) >= 64 else 2 if
                           plot.width() / max(hours, 0.1) >= 32 else 4)
            first = int(span_t[0] // 3600 + 1) * 3600
            for at in range(first, int(span_t[1]) + 1, step):
                x = self._x(at, span_t)
                painter.setPen(QPen(_colour(C.ACCENT, 0.035), 1))
                painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
                painter.setPen(_colour(C.FAINT))
                painter.drawText(QRectF(x - 24, plot.bottom() + 4, 48, 14),
                                 int(Qt.AlignmentFlag.AlignCenter),
                                 datetime.fromtimestamp(at).strftime("%H:%M"))
        # The previous close, dashed.
        if quote.previous:
            y = self._y(quote.previous, span_v)
            pen = QPen(_colour(C.ACCENT_DIM, 0.55), 1, Qt.PenStyle.DashLine)
            pen.setDashPattern([4, 4])
            painter.setPen(pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setFont(_mono(8))
            painter.setPen(_colour(C.FAINT))
            painter.drawText(QRectF(plot.left() + 6, y - 14, 120, 12),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), "PREV CLOSE")
        if len(quote.points) < 2:
            painter.setFont(self._ui(11))
            painter.setPen(_colour(C.FAINT))
            painter.drawText(plot, int(Qt.AlignmentFlag.AlignCenter), "No Intraday Prices")
            return
        colour = tone(quote)
        line = QPainterPath()
        for index, (at, value) in enumerate(zip(quote.times, quote.points)):
            point = QPointF(self._x(at, span_t), self._y(value, span_v))
            line.moveTo(point) if index == 0 else line.lineTo(point)
        area = QPainterPath(line)
        area.lineTo(QPointF(line.currentPosition().x(), plot.bottom()))
        area.lineTo(QPointF(self._x(quote.times[0], span_t), plot.bottom()))
        area.closeSubpath()
        fill = QLinearGradient(0, plot.top(), 0, plot.bottom())
        fill.setColorAt(0.0, _colour(colour, 0.26))
        fill.setColorAt(0.6, _colour(colour, 0.06))
        fill.setColorAt(1.0, _colour(colour, 0.0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        painter.drawPath(area)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for width, alpha in ((8.0, 0.06), (4.0, 0.16), (1.7, 1.0)):    # glow, then the line
            pen = QPen(_colour(colour, alpha), width)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(line)
        # The last price, tagged on the axis.
        y = self._y(quote.points[-1], span_v)
        tag = QRectF(plot.right() + 4, y - 9, self.AXIS_W - 4, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_colour(colour, 0.92))
        painter.drawRoundedRect(tag, 4, 4)
        painter.setFont(_mono(9))
        painter.setPen(_colour(C.BG if colour == C.ACCENT else C.WHITE))
        painter.drawText(tag, int(Qt.AlignmentFlag.AlignCenter), price_text(quote, quote.points[-1]))

    def _draw_side(self, painter: QPainter, side: QRectF) -> None:
        rows = self.groups.get(self.group) or []
        if not rows:
            return
        gap = 8.0
        height = min(72.0, (side.height() - gap * (len(rows) - 1)) / len(rows))
        for index, quote in enumerate(rows):
            card = QRectF(side.left(), side.top() + index * (height + gap), side.width(), height)
            chosen = quote.symbol == self.symbol
            painter.setPen(QPen(_colour(C.BORDER if not chosen else C.PRI, 0.6 if chosen else 1.0), 1))
            painter.setBrush(_colour(C.PANEL_HI if chosen else C.INK))
            painter.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
            colour = tone(quote)
            if chosen:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_colour(C.PRI))
                painter.drawRoundedRect(QRectF(card.left(), card.top() + 8, 3, card.height() - 16), 1.5, 1.5)
            inner = card.adjusted(12, 7, -10, -7)
            painter.setFont(self._ui(11, bold=True))
            painter.setPen(_colour(C.WHITE if chosen else C.SILVER))
            painter.drawText(QRectF(inner.left(), inner.top(), inner.width() * 0.62, 15),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             painter.fontMetrics().elidedText(
                                 quote.label, Qt.TextElideMode.ElideRight, int(inner.width() * 0.62)))
            painter.setFont(_mono(10))
            painter.setPen(_colour(colour))
            painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), 15),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             change_text(quote))
            painter.setPen(_colour(C.MUTED))
            painter.drawText(QRectF(inner.left(), inner.top() + 16, inner.width() * 0.45, 14),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             price_text(quote))
            spark = QRectF(inner.left() + inner.width() * 0.48, inner.top() + 18,
                           inner.width() * 0.52, max(8.0, inner.height() - 18))
            self._spark(painter, spark, quote, colour)
            self._hits.append((card, "symbol", quote.symbol))

    @staticmethod
    def _spark(painter: QPainter, rect: QRectF, quote: Quote, colour: str) -> None:
        if len(quote.points) < 2:
            return
        lo, hi = value_range(quote)
        t0, t1 = domain(quote)
        path = QPainterPath()
        for index, (at, value) in enumerate(zip(quote.times, quote.points)):
            point = QPointF(rect.left() + (at - t0) / (t1 - t0) * rect.width(),
                            rect.bottom() - (value - lo) / (hi - lo) * rect.height())
            path.moveTo(point) if index == 0 else path.lineTo(point)
        if quote.previous:
            y = rect.bottom() - (quote.previous - lo) / (hi - lo) * rect.height()
            painter.setPen(QPen(_colour(C.ACCENT_DIM, 0.3), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(_colour(colour, 0.18), 3.0))
        painter.drawPath(path)
        painter.setPen(QPen(_colour(colour, 0.95), 1.2))
        painter.drawPath(path)

    # ── a frame ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event: Any) -> None:  # noqa: N802
        key = (self.width(), self.height(), self.devicePixelRatioF(), self._version)
        if self._cache is None or self._cache_key != key:
            self._cache = self._render()
            self._cache_key = key
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.drawPixmap(0, 0, self._cache)
            quote = self.selected()
            if quote is None or len(quote.points) < 2 or self._plot.isEmpty():
                return
            span_t, span_v = domain(quote), value_range(quote)
            colour = tone(quote)
            head = QPointF(self._x(quote.times[-1], span_t), self._y(quote.points[-1], span_v))
            if quote.state == "open":
                phase = (time.monotonic() % 1.6) / 1.6
                painter.setPen(QPen(_colour(colour, 0.55 * (1.0 - phase)), 1.4))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(head, 3.0 + 11.0 * phase, 3.0 + 11.0 * phase)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_colour(colour, 0.25))
            painter.drawEllipse(head, 6.0, 6.0)
            painter.setBrush(_colour(C.WHITE if colour == C.ACCENT else C.CORE))
            painter.drawEllipse(head, 2.8, 2.8)
            if self._hover is not None and self._plot.contains(self._hover):
                self._draw_crosshair(painter, quote, span_t, span_v, colour)
        finally:
            painter.end()

    def _draw_crosshair(self, painter: QPainter, quote: Quote, span_t: tuple[float, float],
                        span_v: tuple[float, float], colour: str) -> None:
        plot = self._plot
        hover_t = span_t[0] + (self._hover.x() - plot.left()) / plot.width() * (span_t[1] - span_t[0])
        index = min(range(len(quote.times)), key=lambda i: abs(quote.times[i] - hover_t))
        at, value = quote.times[index], quote.points[index]
        point = QPointF(self._x(at, span_t), self._y(value, span_v))
        pen = QPen(_colour(C.ACCENT, 0.35), 1, Qt.PenStyle.DashLine)
        pen.setDashPattern([3, 3])
        painter.setPen(pen)
        painter.drawLine(QPointF(point.x(), plot.top()), QPointF(point.x(), plot.bottom()))
        painter.drawLine(QPointF(plot.left(), point.y()), QPointF(plot.right(), point.y()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_colour(colour))
        painter.drawEllipse(point, 3.5, 3.5)
        text = f"{datetime.fromtimestamp(at).strftime('%H:%M')}   {price_text(quote, value)}"
        if quote.previous:
            text += f"   {(value - quote.previous) / quote.previous * 100:+.2f}%"
        painter.setFont(_mono(10))
        width = painter.fontMetrics().horizontalAdvance(text) + 18
        box = QRectF(point.x() + 12, plot.top() + 6, width, 22)
        if box.right() > plot.right():
            box.moveRight(point.x() - 12)
        painter.setPen(QPen(_colour(C.BORDER), 1))
        painter.setBrush(_colour(C.PANEL_HI, 0.96))
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(_colour(C.WHITE))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    # ── the mouse ────────────────────────────────────────────────────────────

    def _hit(self, point: QPointF) -> tuple[str, str] | None:
        for rect, kind, value in self._hits:
            if rect.contains(point):
                return kind, value
        return None

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        hit = self._hit(event.position())
        if hit is None:
            super().mousePressEvent(event)
            return
        kind, value = hit
        self.choose(group=value) if kind == "group" else self.choose(symbol=value)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = event.position()
        self.setCursor(Qt.CursorShape.PointingHandCursor if self._hit(self._hover) is not None
                       else Qt.CursorShape.CrossCursor if self._plot.contains(self._hover)
                       else Qt.CursorShape.ArrowCursor)
        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = None
        self.update()
        super().leaveEvent(event)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)
