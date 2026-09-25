"""
TokenUsageGraph — the command deck's per-provider token consumption chart.

Renders the last 24 hours of the TokenUsageLedger as layered per-provider
area lines (Gemini Live and every text provider), with a legend carrying
each provider's 24-hour totals and estimated cost.  The ledger query runs on
its own 15-second timer and is a single grouped SQL read, so the deck's
1-second refresh never touches the database.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..constants import C

# Provider trace colours: crimson identity first (gemini = the native voice),
# then clearly separable hues for the fallback providers.
_SERIES_COLOURS = (
    C.PRI,      # crimson
    "#00e5ff",  # cyan
    "#ffb020",  # amber
    "#3ddc84",  # green
    "#b388ff",  # violet
    "#ff8a65",  # coral
    "#64b5f6",  # blue
    "#f06292",  # pink
)

_WINDOW_S = 24 * 3600.0
_REFRESH_MS = 15_000


def _fmt_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


class TokenUsageGraph(QWidget):
    """Layered 24-hour token-usage traces, one per provider."""

    def __init__(self, ledger: Any = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.ledger = ledger
        # {provider: ([hourly totals aligned to self._hours], total, cost)}
        self._series: dict[str, tuple[list[float], float, float]] = {}
        self._hours: list[str] = []
        self._peak = 0.0
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        # _do_refresh, not refresh: priming the first paint has to bypass the
        # visibility guard, since nothing is visible yet at construction time.
        self._do_refresh()

    def attach_ledger(self, ledger: Any) -> None:
        self.ledger = ledger
        self._do_refresh()   # same reason — render the new ledger immediately

    # ── data ──────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Timer target — skips the ledger query while this panel is hidden
        (it lives inside the Command Centre window, which is hidden rather
        than destroyed when closed)."""
        if not self.isVisible():
            return
        self._do_refresh()

    def _do_refresh(self) -> None:
        if self.ledger is None:
            return
        now = time.time()
        try:
            raw = self.ledger.timeseries_by(
                "provider", bucket="hour", filters={"since": now - _WINDOW_S})
        except Exception:
            return
        # A fixed 24-slot hourly axis (oldest → newest) that every provider's
        # sparse buckets are aligned onto, so traces are comparable.
        hours = [time.strftime("%Y-%m-%d %H:00", time.gmtime(now - h * 3600.0))
                 for h in range(23, -1, -1)]
        series: dict[str, tuple[list[float], float, float]] = {}
        peak = 0.0
        for provider, buckets in raw.items():
            by_bucket = {b["bucket"]: b for b in buckets}
            values: list[float] = []
            total = 0.0
            cost = 0.0
            for hour in hours:
                entry = by_bucket.get(hour)
                tokens = float(entry.get("total_tokens") or 0) if entry else 0.0
                values.append(tokens)
                total += tokens
                if entry and entry.get("estimated_cost_usd"):
                    cost += float(entry["estimated_cost_usd"])
                peak = max(peak, tokens)
            if total > 0:
                series[provider] = (values, total, cost)
        self._hours = hours
        self._series = dict(sorted(series.items(), key=lambda kv: -kv[1][1]))
        self._peak = peak
        self.update()

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        painter.setPen(QPen(QColor(C.BORDER), 1))
        painter.setBrush(QColor(C.INK))
        painter.drawRoundedRect(rect, 6, 6)

        legend_h = 16.0 + 13.0 * max(1, len(self._series))
        plot = rect.adjusted(34, 10, -10, -legend_h)
        if plot.height() < 24:
            plot = rect.adjusted(34, 10, -10, -16)

        painter.setPen(QPen(QColor(C.BORDER), 1, Qt.PenStyle.DotLine))
        for frac in (0.0, 0.5, 1.0):
            y = plot.bottom() - plot.height() * frac
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))

        peak = max(1.0, self._peak)
        painter.setPen(QColor(C.MUTED))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(QRectF(2, plot.top() - 6, 30, 12),
                         Qt.AlignmentFlag.AlignRight, _fmt_tokens(peak))
        painter.drawText(QRectF(2, plot.bottom() - 6, 30, 12),
                         Qt.AlignmentFlag.AlignRight, "0")

        if not self._series:
            painter.setPen(QColor(C.MUTED))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter,
                             "No model turns recorded in the last 24 hours.")
            return

        slots = max(1, len(self._hours) - 1)
        for index, (provider, (values, total, cost)) in enumerate(self._series.items()):
            colour = QColor(_SERIES_COLOURS[index % len(_SERIES_COLOURS)])
            path = QPainterPath()
            for i, tokens in enumerate(values):
                x = plot.left() + plot.width() * (i / slots)
                y = plot.bottom() - plot.height() * (tokens / peak)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(colour, 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            # Soft area fill under the trace for at-a-glance volume.
            fill = QPainterPath(path)
            fill.lineTo(plot.right(), plot.bottom())
            fill.lineTo(plot.left(), plot.bottom())
            fill.closeSubpath()
            area = QColor(colour)
            area.setAlpha(28)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(area)
            painter.drawPath(fill)

            # Legend row: swatch, provider, 24 h tokens, estimated cost.
            ly = plot.bottom() + 12 + index * 13
            painter.setBrush(colour)
            painter.drawRect(QRectF(plot.left(), ly, 8, 8))
            painter.setPen(QColor(C.WHITE))
            painter.setFont(QFont("Segoe UI", 8))
            label = f"{provider}  ·  {_fmt_tokens(total)} tok"
            if cost > 0:
                label += f"  ·  ~${cost:.4f}"
            painter.drawText(QRectF(plot.left() + 14, ly - 3, plot.width() - 14, 14),
                             Qt.AlignmentFlag.AlignLeft, label)
