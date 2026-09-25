"""
System-tray presence (Mark XXII) — ORION lives in the notification area.

  "ORION must be in the background, in the hidden-icons list too, as an
   application — there if I need him, like a real OS AI."

A real operating-system assistant is never more than a glance away: minimise his
windows and he stays resident in the tray with his own icon, a right-click menu
to bring him back, an overlay, or shut him down, a tooltip that tracks what he's
doing, and the ability to surface a safety alert as a native notification even
when no window is open.

Everything is guarded: a machine with no system tray (or a headless test host)
gets a silent no-op rather than a crash, and every window action is wrapped so a
tray click can never take ORION down. The label/tooltip text is factored into
pure helpers so the behaviour is tested without a real tray.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .constants import APP_NAME

# Menu labels — named once so the tests and the menu agree.
LABEL_SHOW = "Show ORION"
LABEL_OVERLAY = "Overlay mode"
LABEL_QUIT = "Quit ORION"


def tooltip_for_state(state: str) -> str:
    """The hover tooltip: his name and what he's doing right now."""
    state = str(state or "").strip().upper() or "READY"
    pretty = {
        "INITIALISING": "starting up",
        "LISTENING": "listening",
        "PROCESSING": "thinking",
        "SPEAKING": "speaking",
        "STANDBY": "in standby",
        "FOCUS": "focus mode",
        "PROACTIVE": "watching",
    }.get(state, state.lower())
    return f"{APP_NAME} — {pretty}"


def tray_available() -> bool:
    """Whether this machine actually has a system tray to live in.

    Guarded on there being a QApplication: QSystemTrayIcon's static methods
    touch the platform integration and ACCESS-VIOLATION crash if called before
    one exists (e.g. from a test, or any non-GUI code path). No app → no tray."""
    try:
        from PyQt6.QtWidgets import QApplication, QSystemTrayIcon
        if QApplication.instance() is None:
            return False
        return bool(QSystemTrayIcon.isSystemTrayAvailable())
    except Exception:
        return False


class OrionTray:
    """ORION's notification-area icon, menu and notifications."""

    def __init__(self, app: Any, window: Any, bus: Any,
                 icon_path: Optional[str] = None) -> None:
        self.app = app
        self.window = window
        self.bus = bus
        self._tray: Any = None
        if not tray_available():
            return
        try:
            self._build(icon_path)
        except Exception:
            # A tray is a nicety; it must never stop ORION running.
            self._tray = None

    # ── construction ──────────────────────────────────────────────────────────

    def _build(self, icon_path: Optional[str]) -> None:
        from PyQt6.QtGui import QAction, QIcon
        from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

        icon = None
        if icon_path:
            icon = QIcon(str(icon_path))
        if icon is None or icon.isNull():
            try:
                icon = self.app.windowIcon()
            except Exception:
                icon = QIcon()
        self._tray = QSystemTrayIcon(icon, self.app)
        self._tray.setToolTip(tooltip_for_state("INITIALISING"))

        menu = QMenu()
        self._menu = menu
        self._actions: dict[str, Any] = {}
        for label, slot in (
            (LABEL_SHOW, self.show_window),
            (LABEL_OVERLAY, self.overlay),
        ):
            action = QAction(label, menu)
            action.triggered.connect(slot)
            menu.addAction(action)
            self._actions[label] = action
        menu.addSeparator()
        quit_action = QAction(LABEL_QUIT, menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)
        self._actions[LABEL_QUIT] = quit_action
        self._tray.setContextMenu(menu)

        # Double-click / left-click brings him to the front.
        self._tray.activated.connect(self._on_activated)

        # Track his state on the tooltip, and raise safety alerts as native
        # notifications so a warning reaches you even with every window minimised.
        for signal_name, handler in (("state", self._on_state),
                                     ("safety_alert", self._on_safety)):
            try:
                getattr(self.bus, signal_name).connect(handler)
            except Exception:
                pass

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        if self._tray is not None:
            try:
                self._tray.show()
            except Exception:
                pass

    def hide(self) -> None:
        if self._tray is not None:
            try:
                self._tray.hide()
            except Exception:
                pass

    @property
    def active(self) -> bool:
        return self._tray is not None

    # ── actions ───────────────────────────────────────────────────────────────

    def show_window(self) -> None:
        window = self.window
        try:
            if window.isMinimized():
                window.showNormal()
            else:
                window.show()
            window.raise_()
            window.activateWindow()
        except Exception:
            pass

    def overlay(self) -> None:
        try:
            toggle = getattr(self.window, "toggle_overlay_mode", None)
            if callable(toggle):
                toggle()
            else:
                self.show_window()
        except Exception:
            pass

    def quit(self) -> None:
        try:
            self.bus.request_shutdown.emit()
        except Exception:
            try:
                self.app.quit()
            except Exception:
                pass

    # ── bus reactions ─────────────────────────────────────────────────────────

    def _on_activated(self, reason: Any) -> None:
        # QSystemTrayIcon.ActivationReason: Trigger (single) / DoubleClick.
        try:
            from PyQt6.QtWidgets import QSystemTrayIcon
            trigger = {QSystemTrayIcon.ActivationReason.Trigger,
                       QSystemTrayIcon.ActivationReason.DoubleClick}
            if reason in trigger:
                self.show_window()
        except Exception:
            self.show_window()

    def _on_state(self, state: Any) -> None:
        if self._tray is not None:
            try:
                self._tray.setToolTip(tooltip_for_state(state))
            except Exception:
                pass

    def _on_safety(self, message: Any) -> None:
        if self._tray is None:
            return
        try:
            from PyQt6.QtWidgets import QSystemTrayIcon
            self._tray.showMessage(
                f"{APP_NAME} — attention",
                str(message)[:220],
                QSystemTrayIcon.MessageIcon.Warning, 8000)
        except Exception:
            pass


__all__ = ["OrionTray", "tooltip_for_state", "tray_available",
           "LABEL_SHOW", "LABEL_OVERLAY", "LABEL_QUIT"]
