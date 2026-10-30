"""
ORION's line icons — one small, consistent set, drawn rather than typed.

Why not text glyphs
-------------------
The Core Window and the Command Deck used to label their controls with
characters: ``⚙``, ``⏻``, ``▶``, ``🛡``, ``⧉``. Each one is drawn by whichever font
happens to contain it, so a single row of buttons mixed three type designs,
four sizes and two baselines — and Windows draws several of them (``⚙ ⚒ ⌨ ▶
🛡``) as colour emoji, which put blue and orange pictures in an interface whose
only hue is crimson. Their widths also differed, so every sidebar label started
at a different x.

These are paths on a 24-unit grid, stroked at one weight with round caps and
joins, and painted in the colour of the control they sit on. Nothing here
depends on a font, an SVG module or a file on disk, so the frozen build draws
exactly what the source does.

Colour follows state
--------------------
An icon takes a *tone* ("muted", "text", "accent", "bad", "inverse"). The
colour is resolved when the icon is painted — never baked into a pixmap — so a
button's icon brightens with its label on hover, whitens when the button is
checked, dims when it is disabled, and follows a re-themed accent without
anything being rebuilt.
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Callable

from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QIconEngine, QPainter, QPainterPath, QPen, QPixmap

from ..constants import C

#: The design grid every icon is drawn on.
GRID = 24.0
#: One stroke weight for the whole set, in grid units: 1.5 px at 16 px.
STROKE = 2.25
#: The one display size used across the Core Window and the deck.
SIZE = 16

# ── tones ─────────────────────────────────────────────────────────────────────

_ACCENT = {"pri": C.PRI, "pri_hi": C.PRI_HI, "pri_dim": C.PRI_DIM}


def set_accent(primary: str) -> None:
    """Follow a re-themed accent (core_window.apply_accent calls this).

    The lifted and dimmed shades are derived the same way the stylesheet
    derives them, so an accent-toned icon matches its button's label.
    """
    primary = str(primary or "").strip() or C.PRI
    family = {C.PRI: primary}
    if primary != C.PRI:
        try:
            from .theming import derive_family

            family = derive_family(primary) or family
        except Exception:
            pass
    _ACCENT["pri"] = primary
    _ACCENT["pri_hi"] = family.get(C.PRI_HI, primary) if primary != C.PRI else C.PRI_HI
    _ACCENT["pri_dim"] = family.get(C.PRI_DIM, primary) if primary != C.PRI else C.PRI_DIM


def _tone(tone: str) -> dict[str, str]:
    """rest / hover / on (checked) / off (disabled) colours for *tone*."""
    if tone == "text":
        return {"rest": C.SILVER, "hover": C.WHITE, "on": C.WHITE, "off": C.FAINT}
    if tone == "accent":
        return {"rest": _ACCENT["pri_hi"], "hover": C.WHITE, "on": C.WHITE,
                "off": _ACCENT["pri_dim"]}
    if tone == "bad":
        return {"rest": C.BAD, "hover": C.WHITE, "on": C.BAD, "off": C.FAINT}
    if tone == "inverse":                       # dark mark on an accent fill
        return {"rest": C.BG, "hover": C.BG, "on": C.BG, "off": C.FAINT}
    if tone == "white":
        return {"rest": C.WHITE, "hover": C.WHITE, "on": C.WHITE, "off": C.FAINT}
    return {"rest": C.MUTED, "hover": C.WHITE, "on": C.WHITE, "off": C.FAINT}


TONES = ("muted", "text", "accent", "bad", "inverse", "white")

# ── the drawings ──────────────────────────────────────────────────────────────
# Each returns (stroked path, filled path or None) on the 24-unit grid. Shapes
# are kept inside 3..21 so every icon has the same optical margin.


def _p() -> QPainterPath:
    return QPainterPath()


def _poly(path: QPainterPath, *points: tuple[float, float], close: bool = False) -> None:
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    if close:
        path.closeSubpath()


def _line(path: QPainterPath, x1: float, y1: float, x2: float, y2: float) -> None:
    path.moveTo(x1, y1)
    path.lineTo(x2, y2)


def _arc(path: QPainterPath, cx: float, cy: float, r: float, start: float, sweep: float) -> None:
    """An arc of the circle at (cx, cy); Qt angles (0 = 3 o'clock, +ve = anticlockwise)."""
    rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
    path.arcMoveTo(rect, start)
    path.arcTo(rect, start, sweep)


def _circle(path: QPainterPath, cx: float, cy: float, r: float) -> None:
    path.addEllipse(QPointF(cx, cy), r, r)


def _round_rect(path: QPainterPath, x: float, y: float, w: float, h: float, r: float = 2.0) -> None:
    path.addRoundedRect(QRectF(x, y, w, h), r, r)


def _speaker(path: QPainterPath) -> None:
    _poly(path, (4, 9.5), (7.5, 9.5), (12, 5.5), (12, 18.5), (7.5, 14.5), (4, 14.5), close=True)


def _mic(path: QPainterPath) -> None:
    _round_rect(path, 9, 3, 6, 11, 3)
    _arc(path, 12, 10.5, 6.5, 180, 180)
    _line(path, 12, 17, 12, 21)
    _line(path, 8.5, 21, 15.5, 21)


def _icon_arrow_left():
    s = _p(); _line(s, 19, 12, 5, 12); _poly(s, (11, 6), (5, 12), (11, 18)); return s, None


def _icon_arrow_right():
    s = _p(); _line(s, 5, 12, 19, 12); _poly(s, (13, 6), (19, 12), (13, 18)); return s, None


def _icon_chevron_left():
    s = _p(); _poly(s, (15, 6), (9, 12), (15, 18)); return s, None


def _icon_chevron_right():
    s = _p(); _poly(s, (9, 6), (15, 12), (9, 18)); return s, None


def _icon_send():
    s = _p()
    _poly(s, (21, 3), (14.5, 21), (11, 13), (3, 9.5), close=True)
    _line(s, 21, 3, 11, 13)
    return s, None


def _icon_stop():
    s = _p(); _round_rect(s, 6, 6, 12, 12, 2); return s, None


def _icon_pause():
    s = _p(); _round_rect(s, 6.5, 5, 3.5, 14, 1); _round_rect(s, 14, 5, 3.5, 14, 1); return s, None


def _icon_play():
    s = _p(); _poly(s, (7.5, 5), (18.5, 12), (7.5, 19), close=True); return s, None


def _icon_power():
    s = _p(); _arc(s, 12, 13, 8, 125, 290); _line(s, 12, 3, 12, 11.5); return s, None


def _icon_sliders():
    s = _p()
    _line(s, 4, 7, 8.5, 7); _line(s, 13.5, 7, 20, 7); _circle(s, 11, 7, 2.5)
    _line(s, 4, 12, 13.5, 12); _line(s, 18.5, 12, 20, 12); _circle(s, 16, 12, 2.5)
    _line(s, 4, 17, 5.5, 17); _line(s, 10.5, 17, 20, 17); _circle(s, 8, 17, 2.5)
    return s, None


def _icon_search():
    s = _p(); _circle(s, 10.5, 10.5, 6.5); _line(s, 15.5, 15.5, 20.5, 20.5); return s, None


def _icon_more():
    f = _p()
    for x in (5.5, 12, 18.5):
        _circle(f, x, 12, 1.6)
    return _p(), f


def _icon_compact():
    s = _p(); _round_rect(s, 3, 5, 18, 14, 2); _circle(s, 15.5, 14.5, 2.5); return s, None


def _icon_maximize():
    s = _p()
    _poly(s, (4, 9), (4, 4), (9, 4)); _poly(s, (15, 4), (20, 4), (20, 9))
    _poly(s, (20, 15), (20, 20), (15, 20)); _poly(s, (9, 20), (4, 20), (4, 15))
    return s, None


def _icon_grid():
    s = _p()
    for x, y in ((4, 4), (13, 4), (4, 13), (13, 13)):
        _round_rect(s, x, y, 7, 7, 1.5)
    return s, None


def _icon_dashboard():
    s = _p()
    _round_rect(s, 4, 4, 7, 9, 1.5); _round_rect(s, 13, 4, 7, 5, 1.5)
    _round_rect(s, 13, 11, 7, 9, 1.5); _round_rect(s, 4, 15, 7, 5, 1.5)
    return s, None


def _icon_gauge():
    s = _p(); _arc(s, 12, 14, 8.5, 200, -220); _line(s, 12, 14, 16, 9.5)
    f = _p(); _circle(f, 12, 14, 1.6)
    return s, f


def _icon_flag():
    s = _p(); _line(s, 5.5, 21, 5.5, 3.5); _poly(s, (5.5, 4), (18.5, 4), (15, 8.75), (18.5, 13.5), (5.5, 13.5))
    return s, None


def _icon_checklist():
    s = _p()
    _poly(s, (3.5, 6.5), (5, 8), (7.5, 5)); _line(s, 11, 6.5, 20.5, 6.5)
    _poly(s, (3.5, 12.5), (5, 14), (7.5, 11)); _line(s, 11, 12.5, 20.5, 12.5)
    _line(s, 11, 18.5, 20.5, 18.5)
    f = _p(); _circle(f, 5.5, 18.5, 1.4)
    return s, f


def _icon_briefcase():
    s = _p(); _round_rect(s, 3, 7.5, 18, 12.5, 2)
    s.moveTo(9, 7.5); s.lineTo(9, 5.5); s.quadTo(9, 4, 10.5, 4); s.lineTo(13.5, 4)
    s.quadTo(15, 4, 15, 5.5); s.lineTo(15, 7.5)
    _line(s, 3, 13, 21, 13)
    return s, None


def _icon_globe():
    s = _p(); _circle(s, 12, 12, 9); s.addEllipse(QRectF(8, 3, 8, 18)); _line(s, 3, 12, 21, 12)
    return s, None


def _icon_book():
    s = _p()
    s.moveTo(19, 3); s.lineTo(7.5, 3); s.quadTo(5, 3, 5, 5.5); s.lineTo(5, 18.5)
    s.quadTo(5, 21, 7.5, 21); s.lineTo(19, 21); s.closeSubpath()
    _line(s, 5, 17, 19, 17)
    _line(s, 9, 7.5, 15, 7.5)
    return s, None


def _ellipse_arc(path: QPainterPath, x: float, y: float, w: float, h: float,
                 start: float, sweep: float) -> None:
    rect = QRectF(x, y, w, h)
    path.arcMoveTo(rect, start)
    path.arcTo(rect, start, sweep)


def _icon_database():
    s = _p()
    s.addEllipse(QRectF(5, 3, 14, 5))
    _line(s, 5, 5.5, 5, 18.5); _line(s, 19, 5.5, 19, 18.5)
    _ellipse_arc(s, 5, 9.5, 14, 5, 180, 180)     # the middle band
    _ellipse_arc(s, 5, 16, 14, 5, 180, 180)      # the base
    return s, None


def _icon_camera():
    s = _p(); _round_rect(s, 3, 7, 18, 13, 2.5)
    _poly(s, (8, 7), (9.5, 4.5), (14.5, 4.5), (16, 7))
    _circle(s, 12, 13.5, 3.5)
    return s, None


def _icon_code():
    s = _p(); _poly(s, (8.5, 7), (3.5, 12), (8.5, 17)); _poly(s, (15.5, 7), (20.5, 12), (15.5, 17))
    return s, None


def _icon_workflow():
    s = _p(); _round_rect(s, 3.5, 3.5, 7, 7, 1.5); _round_rect(s, 13.5, 13.5, 7, 7, 1.5)
    s.moveTo(7, 10.5); s.lineTo(7, 14); s.quadTo(7, 17, 10, 17); s.lineTo(13.5, 17)
    return s, None


def _icon_trend():
    s = _p(); _poly(s, (3, 17), (9, 11), (13, 15), (21, 7)); _poly(s, (15, 7), (21, 7), (21, 13))
    return s, None


def _icon_calculator():
    s = _p(); _round_rect(s, 5, 3, 14, 18, 2.5); _line(s, 8.5, 7.5, 15.5, 7.5)
    f = _p()
    for x in (9, 12, 15):
        for y in (12, 15.5):
            _circle(f, x, y, 1.1)
    _line(s, 9, 18.75, 15, 18.75)
    return s, f


def _icon_target():
    s = _p(); _circle(s, 12, 12, 9); _circle(s, 12, 12, 5)
    f = _p(); _circle(f, 12, 12, 1.6)
    return s, f


def _icon_pawn():
    s = _p(); _circle(s, 12, 6.5, 2.75); _line(s, 9, 11, 15, 11)
    _line(s, 10.5, 11, 9.25, 17.5); _line(s, 13.5, 11, 14.75, 17.5)
    _round_rect(s, 6.5, 17.5, 11, 3.5, 1.25)
    return s, None


def _icon_activity():
    s = _p(); _poly(s, (3, 12), (7, 12), (10, 4.5), (14, 19.5), (17, 12), (21, 12)); return s, None


def _icon_shield():
    s = _p()
    s.moveTo(12, 3); s.lineTo(19, 6); s.lineTo(19, 11.5)
    s.cubicTo(19, 16, 15.8, 19.4, 12, 21); s.cubicTo(8.2, 19.4, 5, 16, 5, 11.5)
    s.lineTo(5, 6); s.closeSubpath()
    return s, None


def _icon_plug():
    s = _p(); _line(s, 9, 3, 9, 7.5); _line(s, 15, 3, 15, 7.5)
    s.moveTo(6, 7.5); s.lineTo(18, 7.5); s.lineTo(18, 11)
    s.cubicTo(18, 14.3, 15.3, 17, 12, 17); s.cubicTo(8.7, 17, 6, 14.3, 6, 11); s.closeSubpath()
    _line(s, 12, 17, 12, 21)
    return s, None


def _icon_terminal():
    s = _p(); _round_rect(s, 3, 4, 18, 16, 2); _poly(s, (7, 9), (10, 12), (7, 15)); _line(s, 12.5, 15, 17, 15)
    return s, None


def _icon_chart():
    s = _p(); _line(s, 6, 20, 6, 13); _line(s, 12, 20, 12, 5); _line(s, 18, 20, 18, 10)
    return s, None


def _icon_sidebar():
    s = _p(); _round_rect(s, 3, 4, 18, 16, 2); _line(s, 9, 4, 9, 20); return s, None


def _icon_sidebar_collapse():
    s, _ = _icon_sidebar(); _poly(s, (16, 9), (13, 12), (16, 15)); return s, None


def _icon_sidebar_expand():
    s, _ = _icon_sidebar(); _poly(s, (13, 9), (16, 12), (13, 15)); return s, None


def _icon_refresh():
    s = _p()
    _arc(s, 12, 12, 8, 30, 300)
    end = (12 + 8 * math.cos(math.radians(30)), 12 - 8 * math.sin(math.radians(30)))
    _poly(s, (end[0], end[1] - 5), end, (end[0] - 5, end[1]))
    return s, None


def _icon_plus():
    s = _p(); _line(s, 12, 5, 12, 19); _line(s, 5, 12, 19, 12); return s, None


def _icon_folder():
    s = _p()
    s.moveTo(3, 6.5); s.quadTo(3, 5, 4.5, 5); s.lineTo(9, 5); s.lineTo(11, 7.5)
    s.lineTo(19.5, 7.5); s.quadTo(21, 7.5, 21, 9); s.lineTo(21, 17.5); s.quadTo(21, 19, 19.5, 19)
    s.lineTo(4.5, 19); s.quadTo(3, 19, 3, 17.5); s.closeSubpath()
    return s, None


def _icon_close():
    s = _p(); _line(s, 6, 6, 18, 18); _line(s, 18, 6, 6, 18); return s, None


def _icon_check():
    s = _p(); _poly(s, (5, 12.5), (10, 17), (19, 7)); return s, None


def _icon_swap():
    s = _p()
    _line(s, 4, 8, 19, 8); _poly(s, (15, 4), (19, 8), (15, 12))
    _line(s, 20, 16, 5, 16); _poly(s, (9, 12), (5, 16), (9, 20))
    return s, None


def _icon_swap_vertical():
    s = _p()
    _line(s, 8, 20, 8, 5); _poly(s, (4, 9), (8, 5), (12, 9))
    _line(s, 16, 4, 16, 19); _poly(s, (12, 15), (16, 19), (20, 15))
    return s, None


def _icon_skip_back():
    s = _p(); _line(s, 6, 6, 6, 18); _poly(s, (18, 6), (9.5, 12), (18, 18), close=True); return s, None


def _icon_skip_forward():
    s = _p(); _line(s, 18, 6, 18, 18); _poly(s, (6, 6), (14.5, 12), (6, 18), close=True); return s, None


def _icon_play_pause():
    s = _p(); _poly(s, (4, 6), (11.5, 12), (4, 18), close=True); _line(s, 15.5, 6, 15.5, 18)
    _line(s, 19.5, 6, 19.5, 18)
    return s, None


def _icon_volume_low():
    s = _p(); _speaker(s); _arc(s, 12, 12, 4, -45, 90); return s, None


def _icon_volume_high():
    s = _p(); _speaker(s); _arc(s, 12, 12, 4, -45, 90); _arc(s, 12, 12, 8, -45, 90); return s, None


def _icon_volume_off():
    s = _p(); _speaker(s); _line(s, 15.5, 9.5, 20.5, 14.5); _line(s, 20.5, 9.5, 15.5, 14.5); return s, None


def _icon_mic():
    s = _p(); _mic(s); return s, None


def _icon_mic_off():
    s = _p(); _mic(s); _line(s, 4, 3.5, 20, 20.5); return s, None


def _icon_phone():
    s = _p(); _round_rect(s, 6.5, 2.5, 11, 19, 2.5); _line(s, 10.5, 18, 13.5, 18); return s, None


def _icon_monitor():
    s = _p(); _round_rect(s, 3, 4, 18, 12, 2); _line(s, 12, 16, 12, 20); _line(s, 8, 20, 16, 20)
    return s, None


def _icon_keyboard():
    s = _p(); _round_rect(s, 2.5, 6, 19, 12, 2); _line(s, 8, 14.5, 16, 14.5)
    f = _p()
    for x in (6.5, 9.8, 13.2, 16.5):
        _circle(f, x, 10, 1.05)
    return s, f


def _icon_document():
    s = _p()
    s.moveTo(14, 3); s.lineTo(6.5, 3); s.quadTo(5, 3, 5, 4.5); s.lineTo(5, 19.5)
    s.quadTo(5, 21, 6.5, 21); s.lineTo(17.5, 21); s.quadTo(19, 21, 19, 19.5); s.lineTo(19, 8)
    s.closeSubpath()
    _poly(s, (14, 3), (14, 8), (19, 8))
    _line(s, 8.5, 13, 15.5, 13); _line(s, 8.5, 17, 13.5, 17)
    return s, None


def _icon_login():
    s = _p()
    s.moveTo(14, 3.5); s.lineTo(17.5, 3.5); s.quadTo(19.5, 3.5, 19.5, 5.5); s.lineTo(19.5, 18.5)
    s.quadTo(19.5, 20.5, 17.5, 20.5); s.lineTo(14, 20.5)
    _line(s, 4, 12, 14, 12); _poly(s, (10, 8), (14, 12), (10, 16))
    return s, None


def _icon_clock():
    s = _p(); _circle(s, 12, 12, 9); _poly(s, (12, 7), (12, 12), (15.5, 14)); return s, None


def _icon_calendar():
    s = _p()
    _round_rect(s, 3.5, 5, 17, 15, 3)
    _line(s, 3.5, 10, 20.5, 10); _line(s, 8, 3, 8, 7); _line(s, 16, 3, 16, 7)
    f = _p(); _circle(f, 8.5, 14.5, 1.1); _circle(f, 12, 14.5, 1.1)
    return s, f


def _icon_dumbbell():
    s = _p()
    _line(s, 8, 12, 16, 12)
    _round_rect(s, 5, 7.5, 3, 9, 1.2); _round_rect(s, 16, 7.5, 3, 9, 1.2)
    _line(s, 3, 10, 3, 14); _line(s, 21, 10, 21, 14)
    return s, None


def _icon_bulb():
    s = _p()
    s.moveTo(9, 17); s.lineTo(9, 15.5); s.cubicTo(9, 14, 6, 12.5, 6, 9)
    s.cubicTo(6, 5.7, 8.7, 3, 12, 3); s.cubicTo(15.3, 3, 18, 5.7, 18, 9)
    s.cubicTo(18, 12.5, 15, 14, 15, 15.5); s.lineTo(15, 17)
    _line(s, 9, 17, 15, 17); _line(s, 10, 20.5, 14, 20.5)
    return s, None


def _icon_bell():
    s = _p()
    s.moveTo(6, 16.5); s.lineTo(6, 10.5); s.cubicTo(6, 7.2, 8.7, 4.5, 12, 4.5)
    s.cubicTo(15.3, 4.5, 18, 7.2, 18, 10.5); s.lineTo(18, 16.5); s.lineTo(19.5, 18); s.lineTo(4.5, 18)
    s.closeSubpath()
    _arc(s, 12, 18.5, 2.25, 180, 180)
    return s, None


def _icon_alert():
    s = _p()
    _poly(s, (12, 3.5), (21.5, 20), (2.5, 20), close=True)
    _line(s, 12, 9.5, 12, 14)
    f = _p(); _circle(f, 12, 17, 1.25)
    return s, f


def _icon_info():
    s = _p(); _circle(s, 12, 12, 9); _line(s, 12, 11, 12, 16.5)
    f = _p(); _circle(f, 12, 7.75, 1.25)
    return s, f


ICONS: dict[str, Callable[[], tuple[QPainterPath, QPainterPath | None]]] = {
    "arrow-left": _icon_arrow_left, "arrow-right": _icon_arrow_right,
    "chevron-left": _icon_chevron_left, "chevron-right": _icon_chevron_right,
    "send": _icon_send, "stop": _icon_stop, "pause": _icon_pause, "play": _icon_play,
    "power": _icon_power, "sliders": _icon_sliders, "search": _icon_search,
    "more": _icon_more, "compact": _icon_compact, "maximize": _icon_maximize,
    "grid": _icon_grid, "dashboard": _icon_dashboard, "gauge": _icon_gauge,
    "flag": _icon_flag, "checklist": _icon_checklist, "briefcase": _icon_briefcase,
    "globe": _icon_globe, "book": _icon_book, "database": _icon_database,
    "camera": _icon_camera, "code": _icon_code, "workflow": _icon_workflow,
    "trend": _icon_trend, "calculator": _icon_calculator, "target": _icon_target,
    "pawn": _icon_pawn, "activity": _icon_activity, "shield": _icon_shield,
    "plug": _icon_plug, "terminal": _icon_terminal, "chart": _icon_chart,
    "sidebar": _icon_sidebar, "sidebar-collapse": _icon_sidebar_collapse,
    "sidebar-expand": _icon_sidebar_expand, "refresh": _icon_refresh,
    "plus": _icon_plus, "folder": _icon_folder, "close": _icon_close,
    "check": _icon_check, "swap": _icon_swap, "swap-vertical": _icon_swap_vertical,
    "skip-back": _icon_skip_back, "skip-forward": _icon_skip_forward,
    "play-pause": _icon_play_pause, "volume-low": _icon_volume_low,
    "volume-high": _icon_volume_high, "volume-off": _icon_volume_off,
    "mic": _icon_mic, "mic-off": _icon_mic_off, "phone": _icon_phone,
    "monitor": _icon_monitor, "keyboard": _icon_keyboard, "document": _icon_document,
    "login": _icon_login, "clock": _icon_clock, "bulb": _icon_bulb, "bell": _icon_bell,
    "calendar": _icon_calendar, "dumbbell": _icon_dumbbell,
    "alert": _icon_alert, "info": _icon_info,
}

#: Every icon name, for tests and for anyone choosing one.
NAMES = tuple(sorted(ICONS))

_PATHS: dict[str, tuple[QPainterPath, QPainterPath | None]] = {}


def _paths(name: str) -> tuple[QPainterPath, QPainterPath | None]:
    cached = _PATHS.get(name)
    if cached is None:
        builder = ICONS.get(name)
        if builder is None:
            raise KeyError(f"no line icon called {name!r}")
        cached = _PATHS[name] = builder()
    return cached


def draw(painter: QPainter, rect: QRectF | QRect, name: str, colour: Any) -> None:
    """Paint icon *name* centred in *rect* in *colour*."""
    stroke, fill = _paths(name)
    rect = QRectF(rect)
    side = min(rect.width(), rect.height())
    if side <= 0:
        return
    colour = QColor(colour)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.translate(rect.x() + (rect.width() - side) / 2.0,
                      rect.y() + (rect.height() - side) / 2.0)
    painter.scale(side / GRID, side / GRID)
    if not stroke.isEmpty():
        pen = QPen(colour)
        pen.setWidthF(STROKE)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(stroke)
    if fill is not None and not fill.isEmpty():
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawPath(fill)
    painter.restore()


def pixmap(name: str, size: int = SIZE, colour: Any = C.SILVER, scale: float = 1.0) -> QPixmap:
    """A transparent pixmap of one icon — for previews, tests and labels."""
    side = max(1, int(round(size * scale)))
    image = QPixmap(side, side)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        draw(painter, QRectF(0, 0, side, side), name, colour)
    finally:
        painter.end()
    image.setDevicePixelRatio(scale)
    return image


class _LineIconEngine(QIconEngine):
    """Draws one line icon at any size, in the colour its control is in.

    *owner* is the button the icon sits on, held weakly: it is how the icon
    knows the pointer is over it, so it brightens with the label on hover
    (QIcon's own modes only cover focus, never hover).
    """

    def __init__(self, name: str, tone: str = "muted", owner: Any = None) -> None:
        super().__init__()
        _paths(name)                         # fail now, not at paint time
        self._name = name
        self._tone = tone if tone in TONES else "muted"
        self._owner = weakref.ref(owner) if owner is not None else None

    def clone(self) -> "_LineIconEngine":
        engine = _LineIconEngine(self._name, self._tone)
        engine._owner = self._owner
        return engine

    def iconName(self) -> str:  # noqa: N802  (Qt naming)
        return self._name

    def _hovered(self) -> bool:
        owner = self._owner() if self._owner is not None else None
        if owner is None:
            return False
        try:
            return bool(owner.isEnabled() and owner.underMouse())
        except RuntimeError:                 # the button has been deleted
            return False

    def colour(self, mode: QIcon.Mode, state: QIcon.State) -> str:
        tone = _tone(self._tone)
        if mode == QIcon.Mode.Disabled:
            return tone["off"]
        if state == QIcon.State.On:
            return tone["on"]
        if mode in (QIcon.Mode.Active, QIcon.Mode.Selected) or self._hovered():
            return tone["hover"]
        return tone["rest"]

    def paint(self, painter: QPainter, rect: QRect, mode: QIcon.Mode,  # noqa: D401
              state: QIcon.State) -> None:
        draw(painter, QRectF(rect), self._name, self.colour(mode, state))

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State,  # noqa: N802
                     scale: float) -> QPixmap:
        scale = float(scale or 1.0)
        width = max(1, int(round(size.width() * scale)))
        height = max(1, int(round(size.height() * scale)))
        image = QPixmap(width, height)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        try:
            draw(painter, QRectF(0, 0, width, height), self._name, self.colour(mode, state))
        finally:
            painter.end()
        image.setDevicePixelRatio(scale)
        return image


def icon(name: str, tone: str = "muted", owner: Any = None) -> QIcon:
    """A QIcon for *name*; pass *owner* (the button) to brighten on hover."""
    return QIcon(_LineIconEngine(name, tone, owner))


def apply(button: Any, name: str, tone: str = "muted", size: int = SIZE) -> Any:
    """Give *button* the line icon *name*. Returns the button for chaining.

    The label is left exactly as it is: the icon sits before it, or alone when
    the button has no text. Calling it again swaps the icon (and its tone) —
    that is how a toggle shows its other state.
    """
    try:
        button.setIcon(icon(name, tone, owner=button))
        button.setIconSize(QSize(size, size))
        button.setProperty("lineIcon", name)
    except Exception:
        pass              # a control that cannot take an icon must still work
    return button


def icon_button(name: str, label: str, hint: str = "", *, tone: str = "muted",
                object_name: str = "iconButton", size: int = SIZE) -> Any:
    """An icon-only push button with its spoken name and tooltip set.

    The name is required: an icon is no more readable to a screen reader than
    the glyph it replaced, so an icon-only control without one is not offered.
    """
    from PyQt6.QtWidgets import QPushButton

    from .widgets import describe_control

    if not str(label or "").strip():
        raise ValueError("an icon-only button needs a spoken name")
    button = QPushButton()
    button.setObjectName(object_name)
    apply(button, name, tone, size)
    return describe_control(button, label, hint)


__all__ = ["GRID", "ICONS", "NAMES", "SIZE", "STROKE", "TONES",
           "apply", "draw", "icon", "icon_button", "pixmap", "set_accent"]
