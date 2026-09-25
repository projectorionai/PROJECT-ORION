"""
The compact orb — ORION shrunk to a floating sphere in the corner of the desktop.

Why this is its own window
--------------------------
It used to be the MAIN window with its flags changed: frameless, always on
top, a tool window, 94% opaque. ``setWindowFlags`` destroys and recreates the
native window, and a QWebEngineView cannot survive that — so every trip into
and out of the orb rebuilt ORION's Three.js face from scratch. A fresh page
inside a layered (semi-transparent) window frequently failed to get a WebGL
context at all, the health watcher then gave up and fell back to the old
software-rendered head, and on the way back the restored window often painted
nothing but black around a small face in its corner while the page reloaded —
"it pops up with an old avatar", "I can't uncompact it", "Not Responding".

None of that can happen here. The main window is simply hidden and shown
again, untouched: its face never reloads, never loses its context, and comes
back exactly as it was. The orb is a separate small window drawn with
QPainter only — no WebEngine, no GL — so there is nothing in it that a window
change can break.

Getting back: double-click the orb, press Esc, use the ⤢ chip, or right-click
for a menu. Drag it anywhere.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QPainter, QPainterPath, QRadialGradient
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .hud import CentralHud
from .style import C

#: Side of the orb window, in pixels. Small enough to live in a corner.
ORB_SIZE = 220


class CompactOrb(CentralHud):
    """CentralHud's sphere, satellites and arcs — without the HUD around it.

    The full HUD carries a grid, a starfield, corner brackets, readouts and a
    banner: right for a whole window, clutter at 220 pixels. This paints only
    ORION himself, on a round dark disc, so the window can be transparent
    around him.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(120, 120)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    def paintEvent(self, event: Any) -> None:   # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        side = min(width, height)
        centre = QPointF(width / 2, height / 2)
        radius = side * 0.24

        # The disc: dark in the middle, fading to nothing at the rim, so the
        # orb reads on any wallpaper without a hard square edge.
        disc = QRadialGradient(centre, side / 2)
        bg = QColor(C.BG)
        inner, outer = QColor(bg), QColor(bg)
        inner.setAlpha(235)
        outer.setAlpha(0)
        disc.setColorAt(0.0, inner)
        disc.setColorAt(0.78, inner)
        disc.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(disc)
        painter.drawEllipse(centre, side / 2, side / 2)

        clip = QPainterPath()
        clip.addEllipse(centre, side / 2, side / 2)
        painter.setClipPath(clip)
        self._paint_arcs(painter, centre, radius)
        satellites = self._satellite_points(centre, radius)
        self._paint_orbital_planes(painter, centre, radius)
        self._paint_satellites(painter, [s for s in satellites if s["depth"] < 0])
        self._paint_sphere_orb(painter, centre, radius)
        self._paint_satellites(painter, [s for s in satellites if s["depth"] >= 0])
        self._paint_particles(painter)
        painter.end()


class OrbOverlay(QWidget):
    """The floating orb window. Emits; never reaches into the main window."""

    restore_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Window
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setWindowTitle("ORION")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(ORB_SIZE, ORB_SIZE)
        self.setAccessibleName("ORION compact orb")
        self.setToolTip("ORION — double-click, Esc or ⤢ to restore; drag to move")
        self._drag_offset: Any = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.orb = CompactOrb(self)
        layout.addWidget(self.orb)
        # The orb does not hold keyboard focus; clicks reach this window.
        self.orb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Restore / shut-down chips, shown while the pointer is over the orb.
        self.chips = QWidget(self)
        row = QHBoxLayout(self.chips)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.restore_btn = QPushButton("⤢", self.chips)
        self.restore_btn.setObjectName("overlayChip")
        self.restore_btn.setToolTip("Restore ORION")
        self.restore_btn.setAccessibleName("Restore ORION")
        self.restore_btn.clicked.connect(self.restore_requested.emit)
        self.quit_btn = QPushButton("⏻", self.chips)
        self.quit_btn.setObjectName("overlayChip")
        self.quit_btn.setToolTip("Shut ORION down completely")
        self.quit_btn.setAccessibleName("Shut ORION down")
        self.quit_btn.clicked.connect(self.quit_requested.emit)
        row.addWidget(self.restore_btn)
        row.addWidget(self.quit_btn)
        self.chips.adjustSize()
        self.chips.move((ORB_SIZE - self.chips.width()) // 2, ORB_SIZE - self.chips.height() - 14)
        # Faded out and click-through until the pointer is over the orb — not
        # hidden. A hidden widget is absent from the accessibility tree, so
        # Narrator, Voice Access and UI Automation had no "Restore ORION" to
        # press. Click-through while faded, so a drag cannot land on an
        # invisible shut-down chip.
        self._chip_fade = QGraphicsOpacityEffect(self.chips)
        self.chips.setGraphicsEffect(self._chip_fade)
        self._show_chips(False)

        restore = QAction("Restore ORION", self)
        restore.setShortcut("Esc")
        restore.triggered.connect(self.restore_requested.emit)
        self.addAction(restore)

    # ── feeding it ───────────────────────────────────────────────────────────

    def set_state(self, state: Any) -> None:
        self.orb.set_state(str(state or "STANDBY"))

    def set_amplitude(self, value: Any) -> None:
        try:
            self.orb.set_amplitude(float(value or 0.0))
        except (TypeError, ValueError):
            pass

    def set_palette_colours(self, primary: str, accent: str) -> None:
        setter = getattr(self.orb, "set_palette_colours", None)
        if callable(setter):
            setter(primary, accent)

    # ── placement ────────────────────────────────────────────────────────────

    def place_on(self, screen: Any) -> None:
        """Bottom-right of *screen* — the one ORION was on, not the primary."""
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.right() - self.width() - 24, area.bottom() - self.height() - 24)

    # ── running only while seen ──────────────────────────────────────────────

    def showEvent(self, event: Any) -> None:   # noqa: N802
        self.orb.timer.start()
        super().showEvent(event)

    def hideEvent(self, event: Any) -> None:   # noqa: N802
        # A hidden orb animating at 30 Hz is paint work on the loop that also
        # carries audio, for nobody.
        self.orb.timer.stop()
        super().hideEvent(event)

    # ── interaction ──────────────────────────────────────────────────────────

    def _show_chips(self, shown: bool) -> None:
        self._chip_fade.setOpacity(1.0 if shown else 0.0)
        self.chips.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not shown)
        if shown:
            self.chips.raise_()

    def enterEvent(self, event: Any) -> None:   # noqa: N802
        self._show_chips(True)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:   # noqa: N802
        self._show_chips(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:   # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (event.globalPosition().toPoint()
                                 - self.frameGeometry().topLeft())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:   # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:   # noqa: N802
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:   # noqa: N802
        self.restore_requested.emit()

    def contextMenuEvent(self, event: Any) -> None:   # noqa: N802
        menu = QMenu(self)
        menu.addAction("Restore ORION").triggered.connect(self.restore_requested.emit)
        menu.addSeparator()
        menu.addAction("Shut ORION down").triggered.connect(self.quit_requested.emit)
        menu.exec(event.globalPos())

    def keyPressEvent(self, event: Any) -> None:   # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.restore_requested.emit()
            return
        super().keyPressEvent(event)


__all__ = ["CompactOrb", "ORB_SIZE", "OrbOverlay"]
