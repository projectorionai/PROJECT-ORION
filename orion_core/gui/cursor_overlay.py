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

import os
import sys
from typing import Any

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..bus import OrionBus
from ..constants import C


# ── cursor plumbing, resolved once ───────────────────────────────────────────
# Built at import rather than per tick: see CursorHalo._cursor_pos. A single
# shared POINT buffer is safe here because every read happens on the GUI thread.
_CURSOR_PT: Any = None
_CURSOR_REF: Any = None
_GET_CURSOR_POS: Any = None

if sys.platform == "win32":
    try:
        import ctypes as _ctypes

        class _POINT(_ctypes.Structure):
            _fields_ = [("x", _ctypes.c_long), ("y", _ctypes.c_long)]

        _CURSOR_PT = _POINT()
        _CURSOR_REF = _ctypes.byref(_CURSOR_PT)
        _GET_CURSOR_POS = _ctypes.windll.user32.GetCursorPos
    except Exception:               # no user32, or a locked-down host
        _CURSOR_PT = _CURSOR_REF = _GET_CURSOR_POS = None

def _to_logical_rect(region: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """A rectangle in the PHYSICAL pixels Windows reports (GetWindowRect,
    pygetwindow, the screen grabber) as the LOGICAL pixels QWidget.setGeometry
    takes. Identity at 100% scaling. On Windows, Qt keeps each screen's origin
    where Windows puts it and scales distances from that origin, so the screen
    whose native span contains the point supplies the ratio."""
    x, y, w, h = (int(v) for v in region)
    if not _scaled_display():
        return (x, y, w, h)
    try:
        from PyQt6.QtGui import QGuiApplication
        for screen in QGuiApplication.screens():
            ratio = screen.devicePixelRatio() or 1.0
            geo = screen.geometry()
            ox, oy = geo.x(), geo.y()
            if (ox <= x < ox + geo.width() * ratio) and (oy <= y < oy + geo.height() * ratio):
                return (int(ox + (x - ox) / ratio), int(oy + (y - oy) / ratio),
                        int(w / ratio), int(h / ratio))
    except Exception:
        pass
    return (x, y, w, h)


_SCALE_CHECK: dict[str, Any] = {"at": -1e9, "scaled": False}


def _scaled_display() -> bool:
    """Whether any screen scales (devicePixelRatio != 1), re-checked every few
    seconds so a monitor plugged in or rescaled mid-session is picked up.
    At 100% everywhere the cheap raw GetCursorPos path is exact."""
    import time as _time
    now = _time.monotonic()
    if now - _SCALE_CHECK["at"] < 5.0:
        return bool(_SCALE_CHECK["scaled"])
    _SCALE_CHECK["at"] = now
    try:
        from PyQt6.QtGui import QGuiApplication
        _SCALE_CHECK["scaled"] = any(abs(s.devicePixelRatio() - 1.0) > 1e-3
                                     for s in QGuiApplication.screens())
    except Exception:
        _SCALE_CHECK["scaled"] = False
    return bool(_SCALE_CHECK["scaled"])


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

        self._flare = 0.0
        self._last_cursor: tuple[int, int] = (-1, -1)
        self._enabled = os.getenv("ORION_CURSOR_HALO", "1").strip().lower() not in {"0", "false", "no", "off"}

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

        self.bus.control_activity.connect(lambda _a: self.flare())
        # Showing his working: what he is doing, beside the pointer, and an
        # outline round the window he is switching to. Separate click-through
        # windows so the small pointer stays a tiny, cheap surface.
        self.caption = NarrationBubble(self._cursor_pos)
        self.highlight = ActionHighlight()
        self.bus.control_narration.connect(self._on_narration)
        if self._enabled:
            self.start()

    def _on_narration(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text") or "").strip()
        if text:
            self.caption.show_caption(text)
        region = payload.get("region")
        if isinstance(region, (tuple, list)) and len(region) == 4:
            self.highlight.flash(tuple(int(v) for v in region))
        self.flare()

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
        """Where the OS cursor is, 33 times a second.

        This used to declare a ctypes.Structure subclass INSIDE the function, so
        every tick built a new class through the ctypes metaclass and re-resolved
        windll.user32.GetCursorPos. Measured 9.14 us; hoisting the struct, the
        buffer and the function pointer to module scope makes it 1.18 us — the
        same call, 7.7x cheaper, on the thread that also draws the face.
        """
        if _scaled_display():
            # GetCursorPos is PHYSICAL pixels; QWidget.move takes LOGICAL ones.
            # At 125%/150% scaling the arrow and caption sat offset from the
            # real pointer, drifting further the further right/down it went.
            # Qt's own cursor position is already mapped per monitor.
            try:
                from PyQt6.QtGui import QCursor
                pos = QCursor.pos()
                return (pos.x(), pos.y())
            except Exception:
                pass
        if _GET_CURSOR_POS is not None:
            try:
                _GET_CURSOR_POS(_CURSOR_REF)
                return (_CURSOR_PT.x, _CURSOR_PT.y)
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
        # Anchor the arrow tip (hotspot) exactly on the OS cursor point — but
        # only when it has actually moved. move() on a frameless always-on-top
        # window is a real SetWindowPos call into the compositor; issuing it 33
        # times a second while the cursor sits still is pure waste.
        moved = (x, y) != self._last_cursor
        if moved:
            self._last_cursor = (x, y)
            self.move(x - self.HOTSPOT[0], y - self.HOTSPOT[1])
        if self._flare > 0:
            self._flare = max(0.0, self._flare - 0.03)
            self.update()
        elif moved:
            # Only when something visible changed. The arrow is drawn from
            # fixed points and its only animated property is the flare, so an
            # unconditional update() repainted an identical picture 33 times a
            # second — 6.1% of every sample taken from a running ORION, on the
            # thread that also paints the face.
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


# ── showing his working ──────────────────────────────────────────────────────

def _overlay_window(widget: QWidget) -> None:
    """Frameless, on top, never in the taskbar, never takes a click or focus."""
    widget.setWindowFlags(
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
        | Qt.WindowType.Tool
        | Qt.WindowType.WindowTransparentForInput
    )
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    widget.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)


class NarrationBubble(QWidget):
    """A caption that rides beside the pointer while ORION works.

    It follows the cursor on its own timer, which runs only while the caption
    is showing, and hides itself a few seconds after the last step — so it
    costs nothing when he is not driving the desktop.
    """

    VISIBLE_S = 3.2
    OFFSET = (22, 20)
    PAD_X, PAD_Y = 10, 6
    MAX_W = 380

    def __init__(self, cursor_pos: Any) -> None:
        super().__init__(None)
        _overlay_window(self)
        self._cursor_pos = cursor_pos
        self._text = ""
        self._follow = QTimer(self)
        self._follow.setInterval(30)
        self._follow.timeout.connect(self._reposition)
        self._expire = QTimer(self)
        self._expire.setSingleShot(True)
        self._expire.timeout.connect(self._retire)

    def show_caption(self, text: str) -> None:
        from PyQt6.QtGui import QFont, QFontMetrics
        self._text = text
        font = QFont(self.font())
        font.setPointSizeF(9.0)
        self.setFont(font)
        metrics = QFontMetrics(font)
        width = min(self.MAX_W, metrics.horizontalAdvance(text) + 2 * self.PAD_X + 6)
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight,
                                    width - 2 * self.PAD_X - 6)
        self._text = elided
        self.setFixedSize(width, metrics.height() + 2 * self.PAD_Y)
        self._reposition()
        self.show()
        self.raise_()
        self.update()
        self._follow.start()
        self._expire.start(int(self.VISIBLE_S * 1000))

    def _retire(self) -> None:
        self._follow.stop()
        self.hide()

    def _reposition(self) -> None:
        x, y = self._cursor_pos()
        nx, ny = x + self.OFFSET[0], y + self.OFFSET[1]
        screen = self.screen()
        if screen is not None:
            area = screen.availableGeometry()
            # Flip to the other side of the pointer rather than run off-screen.
            if nx + self.width() > area.right():
                nx = x - self.width() - 8
            if ny + self.height() > area.bottom():
                ny = y - self.height() - 8
        if (nx, ny) != (self.x(), self.y()):
            self.move(nx, ny)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(QColor(12, 14, 18, 225))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(C.ACCENT))
        painter.drawRoundedRect(3, 5, 3, self.height() - 10, 1.5, 1.5)
        painter.setPen(QColor(240, 242, 245))
        painter.drawText(rect.adjusted(self.PAD_X + 4, 0, -self.PAD_X, 0),
                         int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                         self._text)


class ActionHighlight(QWidget):
    """A brief outline round the window ORION has just switched to."""

    FADE_STEPS = 40          # x 30 ms = 1.2 s
    MARGIN = 6

    def __init__(self) -> None:
        super().__init__(None)
        _overlay_window(self)
        self._alpha = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._fade)

    def flash(self, region: tuple[int, int, int, int]) -> None:
        x, y, w, h = _to_logical_rect(region)
        if w <= 0 or h <= 0:
            return
        m = self.MARGIN
        self.setGeometry(x - m, y - m, w + 2 * m, h + 2 * m)
        self._alpha = 1.0
        self.show()
        self.raise_()
        self.update()
        self._timer.start()

    def _fade(self) -> None:
        self._alpha -= 1.0 / self.FADE_STEPS
        if self._alpha <= 0.0:
            self._alpha = 0.0
            self._timer.stop()
            self.hide()
            return
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(C.ACCENT)
        colour.setAlpha(int(230 * self._alpha))
        painter.setPen(QPen(colour, 3.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)
