"""
The docked Screen Face - ORION's face on the edge of the desktop.

A small always-on-top window holding a :class:`ScreenFacePanel`, parked against
the right (or left) edge of the primary screen at about 70% of its height, the
way a tray of status lights sits at the side of a console. He is there when you
look for him: he listens, speaks and reacts in place, and a click brings him up.

How it behaves
--------------
  * **It never takes focus.** Frameless, a Tool window (no taskbar button, no
    Alt-Tab entry), shown without activating, refusing keyboard focus. Clicking
    it cannot pull the caret out of whatever you were typing into.
  * **It sits on the work area**, not the whole screen: against the taskbar
    rather than behind it, and it follows the work area when that changes.
  * **It grows while he is listening or speaking** (about 1.5x, about a second
    of grace so listening -> thinking -> speaking is one swell, not three) and
    settles back to a small idle size.
  * **It gets out of the way** of full-screen games and presentations: polled
    every five seconds, hidden while Windows reports one, shown again after.
  * **Drag it up or down the edge**; the position is kept in memory only (the
    settings file has no key for it, by design - it is a convenience, not a
    preference). A press that barely moves is a click.
  * **Right-click** for Show ORION and Hide Dock (plus a choice of face when the
    build offers more than the Screen Face - Mark XXXVII offers only that one).
  * **It is not on screen at start**: it appears when asked for (the tray's Show
    Dock, Preferences, "show your dock"); see app._start_windows_integration.

The face inside renders on the face pipeline's own worker thread, so the dock
adds nothing to the qasync loop beyond blitting a frame.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import (
    QEasingCurve, QPoint, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer)
from PyQt6.QtGui import QActionGroup, QColor, QGuiApplication, QPainter, QPainterPath, QRegion
from PyQt6.QtWidgets import QMenu, QVBoxLayout, QWidget

from .screen_face import ScreenFacePanel
from .style import C

#: Idle size, and how much larger he is while listening or speaking.
IDLE_SIZE = QSize(150, 120)
GROW_FACTOR = 1.5

#: Where along the edge he sits by default: the centre line at 70% of the
#: work area's height.
DEFAULT_Y_FRACTION = 0.70

#: States (bus.state values) in which he is the centre of attention.
GROW_STATES = frozenset({"LISTENING", "SPEAKING"})

#: How often to ask Windows whether a game or presentation has taken the screen.
BUSY_POLL_MS = 5000

#: The thin crimson rim around the face, and the corner rounding.
_RIM = 2
_RADIUS = 14

#: Every face the dock's menu knows: (form key, menu label). Only the ones the
#: build offers (appearance.enabled_forms()) are listed, and with just one there
#: is nothing to choose, so no face entries at all.
FACE_FORMS = (("screen", "Screen Face"), ("classic", "Classic Face"), ("orb", "Orb"))


def dock_rect(area: QRect, width: int, height: int, edge: str, y_fraction: float) -> QRect:
    """Where a dock of this size sits: flush against *edge* of the work area,
    its centre line at *y_fraction* of the height, kept fully on screen."""
    x = area.left() if edge == "left" else area.right() + 1 - width
    centre = area.top() + float(y_fraction) * area.height()
    y = int(round(centre - height / 2.0))
    y = max(area.top(), min(y, area.bottom() + 1 - height))
    return QRect(x, y, width, height)


class ScreenDock(QWidget):
    """ORION's face, docked. See the module docstring."""

    def __init__(self, parent: Optional[QWidget] = None, *, edge: str = "right",
                 busy_state: Callable[[], str] | None = None,
                 on_activate: Callable[[], None] | None = None,
                 on_show: Callable[[], None] | None = None,
                 on_face_form: Callable[[str], None] | None = None,
                 on_hide: Callable[[], None] | None = None,
                 face_form: str = "", animate: bool = True,
                 shrink_delay_ms: int = 1100, y_fraction: float = DEFAULT_Y_FRACTION,
                 screen_provider: Callable[[], Any] | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("ORION Dock")
        self.setAccessibleName("ORION docked face")
        self.setAccessibleDescription("ORION's face, docked to the edge of the screen. "
                                      "Click to talk to him; right-click for options.")
        self.setToolTip("ORION - click to talk, drag to move, right-click for options")

        if busy_state is None:
            from ..windows_presence import user_busy_state
            busy_state = user_busy_state
        self._busy_state = busy_state
        self._on_activate = on_activate
        self._on_show = on_show
        self._on_face_form = on_face_form
        self._on_hide = on_hide
        self._face_form = str(face_form or "")
        self._edge = "left" if str(edge).lower() == "left" else "right"
        self._y_fraction = max(0.0, min(1.0, float(y_fraction)))
        self._animate = bool(animate)
        self._screen_provider = screen_provider

        self._state = ""
        self._speaking = False
        self._grown = False
        self._accent = QColor(C.PRI)

        self._wanted = False            # the owner wants it on screen
        self._busy_hidden = False       # ...but a game or presentation is in the way
        self._stopped = False
        self._press_global: QPoint | None = None
        self._press_top = 0
        self._dragged = False
        self._menu: QMenu | None = None
        self._screen_hooked = False
        self._animation: QPropertyAnimation | None = None

        self.panel = ScreenFacePanel(self)
        # Clicks belong to the dock (drag, activate, menu), not to the face.
        self.panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.panel.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(_RIM, _RIM, _RIM, _RIM)
        layout.addWidget(self.panel)

        self._poll = QTimer(self)
        self._poll.setInterval(BUSY_POLL_MS)
        self._poll.timeout.connect(self._poll_busy)

        self._shrink = QTimer(self)
        self._shrink.setSingleShot(True)
        self._shrink.setInterval(max(0, int(shrink_delay_ms)))
        self._shrink.timeout.connect(lambda: self._set_grown(False))

        self.setGeometry(self.target_rect())

    # ── geometry ─────────────────────────────────────────────────────────────

    @property
    def edge(self) -> str:
        return self._edge

    @property
    def y_fraction(self) -> float:
        return self._y_fraction

    @property
    def grown(self) -> bool:
        return self._grown

    def size_for(self, grown: bool) -> QSize:
        if not grown:
            return QSize(IDLE_SIZE)
        return QSize(int(round(IDLE_SIZE.width() * GROW_FACTOR)),
                     int(round(IDLE_SIZE.height() * GROW_FACTOR)))

    def work_area(self) -> QRect:
        """The primary screen's work area (taskbar excluded)."""
        screen = None
        try:
            screen = self._screen_provider() if self._screen_provider else QGuiApplication.primaryScreen()
        except Exception:
            screen = None
        if screen is None:
            return QRect(0, 0, 1280, 720)
        return QRect(screen.availableGeometry())

    def target_rect(self, grown: bool | None = None) -> QRect:
        size = self.size_for(self._grown if grown is None else grown)
        return dock_rect(self.work_area(), size.width(), size.height(),
                         self._edge, self._y_fraction)

    def set_edge(self, edge: str) -> None:
        self._edge = "left" if str(edge).lower() == "left" else "right"
        self.redock()

    def redock(self) -> None:
        """Snap back to the edge - after the work area changed or a drag."""
        self._stop_animation()
        self.setGeometry(self.target_rect())

    def _apply_geometry(self) -> None:
        target = self.target_rect()
        if not (self._animate and self.isVisible()):
            self._stop_animation()
            self.setGeometry(target)
            return
        self._stop_animation()
        animation = QPropertyAnimation(self, b"geometry", self)
        animation.setDuration(180)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.setStartValue(self.geometry())
        animation.setEndValue(target)
        self._animation = animation
        animation.start()

    def _stop_animation(self) -> None:
        animation, self._animation = self._animation, None
        if animation is not None:
            animation.stop()
            animation.deleteLater()

    def resizeEvent(self, event: Any) -> None:   # noqa: N802
        outer = QPainterPath()
        outer.addRoundedRect(QRectF(self.rect()), _RADIUS, _RADIUS)
        self.setMask(QRegion(outer.toFillPolygon().toPolygon()))
        inner = QPainterPath()
        # From the dock's own size, not the panel's: the layout may not have
        # resized the panel yet when this runs.
        inner.addRoundedRect(QRectF(0, 0, max(1, self.width() - 2 * _RIM),
                                    max(1, self.height() - 2 * _RIM)),
                             _RADIUS - _RIM, _RADIUS - _RIM)
        self.panel.setMask(QRegion(inner.toFillPolygon().toPolygon()))
        super().resizeEvent(event)

    def paintEvent(self, event: Any) -> None:   # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rim = QColor(self._accent)
        rim.setAlpha(170)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(rim)
        painter.drawRoundedRect(QRectF(self.rect()), _RADIUS, _RADIUS)
        painter.end()

    # ── feeding him: ORION's face protocol, forwarded ────────────────────────

    def set_state(self, state: Any) -> None:
        self._state = str(state or "").upper()
        self.panel.set_state(state)
        self._reconsider_size()

    def set_speaking(self, active: Any) -> None:
        self._speaking = bool(active)
        self.panel.set_speaking(active)
        self._reconsider_size()

    def set_amplitude(self, value: Any) -> None:
        self.panel.set_amplitude(value)

    def set_viseme(self, openness: Any, width: Any = 0.0, closure: Any = 0.0) -> None:
        self.panel.set_viseme(openness, width, closure)

    def set_viseme_payload(self, payload: Any) -> None:
        """A bus.viseme payload ({"viseme","weight"} or {"open","width","closure"})."""
        if not isinstance(payload, dict):
            return
        try:
            if "open" in payload:
                self.panel.set_viseme(float(payload.get("open", 0.0)),
                                      float(payload.get("width", 0.0)),
                                      float(payload.get("closure", 0.0)))
            else:
                self.panel.set_viseme(payload.get("viseme", "sil"),
                                      float(payload.get("weight", 1.0)))
        except (TypeError, ValueError):
            pass

    def glance(self, dx: Any = 0.0, dy: Any = -0.65, hold: Any = 1.2) -> None:
        self.panel.glance(dx, dy, hold)

    def apply_emotion(self, name: Any, params: Any = None) -> None:
        self.panel.apply_emotion(name, params)

    def set_palette_colours(self, primary: Any, accent: Any, background: Any = None) -> None:
        self.panel.set_palette_colours(primary, accent, background)
        colour = QColor(accent) if not isinstance(accent, QColor) else QColor(accent)
        if colour.isValid():
            self._accent = colour
            self.update()

    def set_face_form(self, form: str) -> None:
        """Which face the menu should tick (the dock always draws a Screen Face)."""
        self._face_form = str(form or "")

    def connect_bus(self, bus: Any) -> None:
        """Follow ORION's bus: state, loudness, speech, emotion and lip-sync."""
        for name, slot in (("state", self.set_state), ("amplitude", self.set_amplitude),
                           ("speaking", self.set_speaking),
                           ("emotion_changed", self.apply_emotion),
                           ("viseme", self.set_viseme_payload)):
            try:
                getattr(bus, name).connect(slot)
            except Exception:
                pass

    # ── growing and shrinking ────────────────────────────────────────────────

    def _wants_grown(self) -> bool:
        return self._speaking or self._state in GROW_STATES

    def _reconsider_size(self) -> None:
        if self._wants_grown():
            self._shrink.stop()
            self._set_grown(True)
        elif self._grown and not self._shrink.isActive():
            if self._shrink.interval() <= 0:
                self._set_grown(False)
            else:
                self._shrink.start()        # a grace period: no flicker between turns

    def _set_grown(self, grown: bool) -> None:
        if grown == self._grown:
            return
        self._grown = grown
        self._apply_geometry()

    # ── visibility ───────────────────────────────────────────────────────────

    @property
    def wanted(self) -> bool:
        return self._wanted

    @property
    def busy_hidden(self) -> bool:
        return self._busy_hidden

    def show_dock(self) -> None:
        """Put him on the edge - unless a game or presentation has the screen,
        in which case he waits and appears when it ends."""
        if self._stopped:
            return
        self._wanted = True
        self._hook_screen_changes()
        self.redock()
        self._poll.start()
        self._poll_busy()

    def hide_dock(self) -> None:
        """Take him off the screen until show_dock()."""
        self._wanted = False
        self._busy_hidden = False
        self._poll.stop()
        self.hide()

    def _poll_busy(self) -> None:
        if not self._wanted or self._stopped:
            return
        try:
            from ..windows_presence import HIDE_DOCK_STATES

            busy = self._busy_state() in HIDE_DOCK_STATES
        except Exception:
            busy = False
        if busy:
            self._busy_hidden = True
            if self.isVisible():
                self.hide()
        else:
            self._busy_hidden = False
            if not self.isVisible():
                self.redock()
                self.show()         # WA_ShowWithoutActivating: shown, not focused

    def _hook_screen_changes(self) -> None:
        """Follow the work area: a taskbar moved or resized, a monitor plugged
        in, the primary screen changing. Bound methods, not lambdas, so Qt
        drops the connection when the dock is destroyed instead of calling a
        dead widget."""
        if self._screen_hooked or self._screen_provider is not None:
            return
        try:
            QGuiApplication.primaryScreenChanged.connect(self._primary_changed)
            self._screen_hooked = True
            self._watch_primary_screen()
        except Exception:
            pass

    def _watch_primary_screen(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            screen.availableGeometryChanged.connect(self._screen_changed)
            screen.geometryChanged.connect(self._screen_changed)

    def _primary_changed(self, *_args: Any) -> None:
        if self._stopped:
            return
        self._watch_primary_screen()            # the new primary screen's own signals
        self.redock()

    def _screen_changed(self, *_args: Any) -> None:
        if not self._stopped:
            self.redock()

    def stop(self) -> None:
        """Stop the timers and the face's render thread, and hide. Idempotent."""
        self._stopped = True
        self._wanted = False
        self._poll.stop()
        self._shrink.stop()
        self._stop_animation()
        try:
            self.hide()
        except RuntimeError:
            pass
        try:
            self.panel.stop_pipeline()
        except Exception:
            pass

    def closeEvent(self, event: Any) -> None:   # noqa: N802
        self.stop()
        super().closeEvent(event)

    # ── mouse: drag along the edge, click to activate ────────────────────────

    def mousePressEvent(self, event: Any) -> None:   # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._press_top = self.y()
            self._dragged = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:   # noqa: N802
        if self._press_global is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        delta = event.globalPosition().toPoint() - self._press_global
        if not self._dragged:
            if abs(delta.y()) <= QGuiApplication.styleHints().startDragDistance():
                return
            self._dragged = True
            self._stop_animation()
        area = self.work_area()
        top = max(area.top(), min(self._press_top + delta.y(), area.bottom() + 1 - self.height()))
        self._y_fraction = max(0.0, min(1.0, (top + self.height() / 2.0 - area.top())
                                        / max(1, area.height())))
        self.move(self.x(), top)            # along the edge only
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:   # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._press_global is not None:
            clicked = not self._dragged
            self._press_global = None
            self._dragged = False
            if clicked:
                self._call(self._on_activate)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ── the menu ─────────────────────────────────────────────────────────────

    def build_menu(self) -> QMenu:
        menu = QMenu(self)
        show = menu.addAction("Show ORION")
        show.triggered.connect(lambda *_a: self._call(self._on_show or self._on_activate))
        menu.addSeparator()
        from ..appearance import enabled_forms

        offered = [(form, label) for form, label in FACE_FORMS if form in enabled_forms()]
        if len(offered) > 1:
            group = QActionGroup(menu)
            group.setExclusive(True)
            for form, label in offered:
                action = menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(form == self._face_form)
                action.setData(form)
                group.addAction(action)
                action.triggered.connect(lambda _checked=False, f=form: self._choose_form(f))
            menu.addSeparator()
        menu.addAction("Hide Dock").triggered.connect(lambda *_a: self._hide_requested())
        return menu

    def contextMenuEvent(self, event: Any) -> None:   # noqa: N802
        # popup(), not a blocking run: a nested event loop under qasync would
        # stall the asyncio loop that carries the audio while the menu is open.
        menu = self.build_menu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._menu = menu
        menu.popup(event.globalPos())

    def _choose_form(self, form: str) -> None:
        self._face_form = form
        if self._on_face_form is not None:
            self._call(lambda: self._on_face_form(form))

    def _hide_requested(self) -> None:
        self.hide_dock()
        self._call(self._on_hide)

    @staticmethod
    def _call(callback: Callable[[], None] | None) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception:
            import logging

            logging.getLogger("orion.screen_dock").exception("a dock callback raised")


__all__ = ["BUSY_POLL_MS", "DEFAULT_Y_FRACTION", "FACE_FORMS", "GROW_FACTOR",
           "GROW_STATES", "IDLE_SIZE", "ScreenDock", "dock_rect"]
