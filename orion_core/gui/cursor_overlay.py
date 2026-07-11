"""
CursorOverlay — a small, discreet pointer ORION drives around the screen.

A tiny, frameless, translucent, click-through, always-on-top window follows the
real OS cursor everywhere and draws a **normal small black arrow cursor** at the
pointer, so when ORION moves the mouse, clicks or scrolls you can see where he
is acting — without the large sci-fi "scope" marker of earlier builds.

The pointer:
    • follows the physical cursor via a lightweight poll (ctypes GetCursorPos
      on Windows, pyautogui elsewhere), so it tracks ORION and you alike;
    • **flares** subtly (a small soft ring at the tip) whenever ORION performs a
      control action (``bus.control_activity``), so autonomous moves stand out
      without dominating the screen;
    • is fully click-through (``WA_TransparentForMouseEvents``) — it never
      intercepts a click and never appears in the taskbar.

Because it is a tiny widget that just repositions itself each tick (not a
full-screen surface), it is cheap to run.  Toggle it with the
``cursor_overlay`` tool or ORION_CURSOR_HALO=0 to start hidden.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..bus import OrionBus
from ..constants import C


class CursorOverlay(QWidget):
    SIZE = 34            # px square that trails the cursor (small)
    HOTSPOT = (10, 9)    # arrow tip position inside the window == the cursor

    def __init__(self, bus: OrionBus) -> None:
        super().__init__(None)
        self.bus = bus
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._phase = 0.0
        self._flare = 0.0
        self._enabled = os.getenv("ORION_CURSOR_HALO", "1").strip().lower() not in {"0", "false", "no", "off"}

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

        self.bus.control_activity.connect(lambda _a: self.flare())
        if self._enabled:
            self.start()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._enabled = True
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self) -> None:
        self._enabled = False
        self._timer.stop()
        self.hide()

    def toggle(self) -> bool:
        self.stop() if self._enabled else self.start()
        return self._enabled

    def flare(self) -> None:
        """Momentarily brighten/enlarge the halo (ORION just acted)."""
        self._flare = 1.0

    # ── follow + paint ────────────────────────────────────────────────────────

    def _cursor_pos(self) -> tuple[int, int]:
        if sys.platform == "win32":
            try:
                import ctypes

                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                pt = POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                return (pt.x, pt.y)
            except Exception:
                pass
        try:
            import pyautogui
            pos = pyautogui.position()
            return (int(pos.x), int(pos.y))
        except Exception:
            return (0, 0)

    def _tick(self) -> None:
        x, y = self._cursor_pos()
        # Anchor the arrow tip (hotspot) exactly on the OS cursor point.
        self.move(x - self.HOTSPOT[0], y - self.HOTSPOT[1])
        self._phase = (self._phase + 0.14) % (2 * math.pi)
        if self._flare > 0:
            self._flare = max(0.0, self._flare - 0.03)
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        tip_x, tip_y = self.HOTSPOT
        flare = self._flare

        # Subtle flare ring at the tip when ORION acts — small, not a scope.
        if flare > 0.01:
            r = 5 + flare * 8
            painter.setPen(QPen(QColor(255, 26, 60, int(150 * flare)), 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(tip_x, tip_y), r, r)

        # A normal small black arrow pointer (classic cursor silhouette),
        # white-outlined so it stays visible on dark and light surfaces.
        arrow = QPainterPath()
        pts = [
            (0, 0), (0, 14), (3.6, 10.6), (6.2, 16.4),
            (8.4, 15.4), (5.9, 9.8), (10.2, 9.6),
        ]
        arrow.moveTo(tip_x + pts[0][0], tip_y + pts[0][1])
        for px, py in pts[1:]:
            arrow.lineTo(tip_x + px, tip_y + py)
        arrow.closeSubpath()
        painter.setPen(QPen(QColor(255, 255, 255, 210), 1.2))
        # A hair brighter while flaring so an autonomous move reads at a glance.
        fill = 235 if flare > 0.01 else 255
        painter.setBrush(QColor(10, 10, 12, fill))
        painter.drawPath(arrow)
