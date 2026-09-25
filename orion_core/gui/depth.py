"""
Depth that is real, rather than drawn.

Qt Style Sheets cannot blur and cannot cast a shadow. The old stylesheet said
so in its own docstring and worked around it with gradient pairs and
bright/dim border tricks — which produces a *picture* of depth. It reads as a
diagram of a lit surface rather than a lit surface, and it is the single thing
that made the interface look like every other dark assistant UI.

Two mechanisms here, and neither is a gradient:

**Shadows.** ``QGraphicsDropShadowEffect`` is a real composited blur. A panel
with one genuinely sits above the field, and the eye reads the elevation
without being told.

**Backdrop.** On Windows 11 the compositor itself can blur what is behind the
window — the same effect the operating system uses on its own surfaces. That
is a DWM feature, not a Qt one, so it is asked for through the Win32 API and
degrades to an ordinary opaque window everywhere else.

Why shadows are OFF by default
------------------------------
Measured, on the real frozen build: turning them on killed the process.
``ORION.exe`` died in under forty seconds with STATUS_HEAP_CORRUPTION
(0xC0000374), and the same build with ``ORION_FLAT=1`` ran indefinitely.

The reason is not a bug in this file. ``QGraphicsEffect`` renders its widget
into an offscreen pixmap, and that is undefined behaviour for a widget that
owns — or contains — a NATIVE window. ORION's Core Window contains the
QWebEngineView face and OpenGL views, so walking the tree and lifting every
panel eventually put an effect over a native child and corrupted the heap.

There is also nothing much to gain. Measured by rendering: against a near-black
field a drop shadow is almost invisible, because a shadow is a darkening and
there is nothing left to darken. What actually makes a panel read as raised
here is surface brightness, which the palette already carries. So the shadow
was buying polish worth roughly nothing at the price of the whole application.

It stays available — ``ORION_SHADOWS=1`` — because on a surface with no native
children it is the real thing rather than a drawn approximation. :func:`lift`
refuses any widget containing a native window whatever that setting says.

The other cost, stated plainly: an effect pushes the widget onto a separate
composited path, and this application has already been bitten once by paint
cost — 441 ms of QPainter per second on the qasync thread, which is also the
thread that feeds audio. :func:`shadow_budget` exists to make it obvious when
that line has been crossed.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from ..constants import ELEV

#: Elevation names, so call sites read as intent rather than as a number.
REST, RAISED, FLOAT = 0, 1, 2

_LEVELS = {REST: ELEV.REST, RAISED: ELEV.RAISED, FLOAT: ELEV.FLOAT}

#: How many shadowed widgets are reasonable at once. Not a hard limit — it is
#: the number above which someone should be asked whether every one of them
#: really needs to float.
SHADOW_BUDGET = 24

_lifted: set[int] = set()


def _truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def shadows_enabled() -> bool:
    """Whether to composite shadows at all. OFF unless asked for.

    Opt-in, not opt-out, because the default has to be the one that cannot
    kill the application — see the module docstring: this crashed the frozen
    build with STATUS_HEAP_CORRUPTION. ``ORION_FLAT=1`` still forces them off
    and wins over ``ORION_SHADOWS``, so an existing instruction to go flat
    keeps working.

    The interface is complete without them: elevation is carried by surface
    brightness and corner radius, so losing the shadow loses polish rather
    than meaning.
    """
    if _truthy("ORION_FLAT"):
        return False
    return _truthy("ORION_SHADOWS")


def _contains_native_window(widget: Any) -> bool:
    """Whether *widget* is, or holds, a window the compositor owns directly.

    QWebEngineView and the OpenGL views are native children. An effect over
    one of those is not merely ignored — it corrupted the heap and took the
    process with it. Cheap to ask, and the answer is the difference between a
    shadow and a crash.
    """
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QWidget

        native = Qt.WidgetAttribute.WA_NativeWindow
        if widget.testAttribute(native) or widget.windowHandle() is not None:
            return True
        for child in widget.findChildren(QWidget):
            if child.testAttribute(native) or child.windowHandle() is not None:
                return True
            # WebEngine and OpenGL views become native the moment they are
            # shown, which may be after this runs — so match them by type too.
            name = type(child).__name__
            if "WebEngine" in name or "OpenGL" in name or "QQuick" in name:
                return True
    except Exception:
        return True          # cannot tell — do not risk it
    return False


def lift(widget: Any, level: int = RAISED, colour: Any = None) -> bool:
    """Raise *widget* off the surface with a real shadow.

    Returns whether a shadow was applied. Never raises: this is presentation,
    and a widget that cannot take an effect should still be a widget.
    """
    if widget is None or level == REST or not shadows_enabled():
        return False
    if _contains_native_window(widget):
        return False        # an effect here is undefined behaviour, not a look
    try:
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
    except Exception:
        return False

    blur, offset, alpha = _LEVELS.get(level, ELEV.RAISED)
    try:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(float(blur))
        effect.setXOffset(0.0)
        effect.setYOffset(float(offset))
        # Black, not a tinted shadow. A coloured shadow is a hue, and the
        # whole point of this palette is that there is only one.
        effect.setColor(colour if colour is not None else QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(effect)
        _lifted.add(id(widget))
        return True
    except Exception:
        return False


def flatten(widget: Any) -> None:
    """Remove a widget's shadow. Cheap, and safe to call on anything."""
    try:
        widget.setGraphicsEffect(None)
        _lifted.discard(id(widget))
    except Exception:
        pass


def lift_panels(root: Any, level: int = RAISED,
                selector: str = "panelFrame") -> int:
    """Raise every panel under *root* in one call.

    There are twenty-six places that build a panelFrame. Editing each one to
    add a shadow would mean twenty-six chances to forget, and the twenty-
    seventh panel would simply look flat for no reason anybody could see.
    Walking the tree once keeps elevation a property of what a panel IS
    rather than of who remembered.

    Returns how many were lifted.
    """
    if root is None or not shadows_enabled():
        return 0
    try:
        from PyQt6.QtWidgets import QWidget
    except Exception:
        return 0
    lifted = 0
    try:
        for widget in root.findChildren(QWidget):
            if widget.objectName() == selector and lift(widget, level):
                lifted += 1
    except Exception:
        pass
    return lifted


def shadow_budget() -> tuple[int, int]:
    """(shadows applied, budget). For diagnostics rather than enforcement.

    A number over budget is not a bug; it is a question. Twenty-four floating
    panels means nothing is floating, because elevation only reads when most
    things are resting.
    """
    return len(_lifted), SHADOW_BUDGET


# ── the compositor's own blur ────────────────────────────────────────────────

#: DWM attribute ids. Named because the raw integers say nothing.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_SYSTEMBACKDROP_TYPE = 38

#: Backdrop kinds the compositor understands.
BACKDROP_NONE, BACKDROP_AUTO, BACKDROP_MICA, BACKDROP_ACRYLIC = 1, 0, 2, 3


def backdrop_supported() -> bool:
    """Whether this Windows can blur behind a window.

    The system backdrop attribute arrived in Windows 11 22H2 (build 22621).
    Asking an older build is harmless — it returns a failure code and the
    window stays opaque — but knowing lets the caller say so rather than
    wonder why nothing looks different.
    """
    if sys.platform != "win32":
        return False
    try:
        return sys.getwindowsversion().build >= 22621
    except Exception:
        return False


def apply_backdrop(window: Any, kind: int = BACKDROP_ACRYLIC) -> bool:
    """Ask Windows to blur whatever is behind *window*.

    This is the one part of the look that cannot be faked in Qt: real
    translucency sampling the actual desktop behind the window. Everything
    else here only *suggests* glass.

    Returns whether the compositor accepted it. A False is not a failure worth
    reporting to anybody — it means an ordinary opaque window, which is what
    every other application has.
    """
    if not backdrop_supported():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        handle = int(window.winId())
        dwm = ctypes.WinDLL("dwmapi")
        value = ctypes.c_int(int(kind))
        dark = ctypes.c_int(1)
        # Dark mode first: without it the compositor tints the backdrop light
        # and a near-black interface ends up sitting on a pale sheet.
        dwm.DwmSetWindowAttribute(
            wintypes.HWND(handle), _DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(dark), ctypes.sizeof(dark))
        result = dwm.DwmSetWindowAttribute(
            wintypes.HWND(handle), _DWMWA_SYSTEMBACKDROP_TYPE,
            ctypes.byref(value), ctypes.sizeof(value))
        return result == 0
    except Exception:
        return False


def describe() -> str:
    """One line on what the depth system is actually doing, for diagnostics."""
    used, budget = shadow_budget()
    parts = [f"{used}/{budget} shadows"]
    parts.append("compositor backdrop available" if backdrop_supported()
                 else "no compositor backdrop on this OS")
    if not shadows_enabled():
        parts.append("shadows OFF (default; ORION_SHADOWS=1 to opt in)")
    return "; ".join(parts)


__all__ = [
    "BACKDROP_ACRYLIC", "BACKDROP_AUTO", "BACKDROP_MICA", "BACKDROP_NONE",
    "FLOAT", "RAISED", "REST", "SHADOW_BUDGET",
    "apply_backdrop", "backdrop_supported", "describe", "flatten", "lift",
    "shadow_budget", "shadows_enabled",
]
