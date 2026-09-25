"""
SecurityCentreView — the Command Deck's SECURITY tab (Mark X.8).

Before this file existed, the Command Deck had no security page at all:
``SecuritySentinel`` (orion_core/security_sentinel.py) ran a proactive host
monitor and fired alerts only through the shared HUD banner (which fades in
~3 seconds) and spoken word — there was nowhere in the GUI to see the
current posture on demand, review what had already fired this session, or
turn monitoring on/off.  That is the "unclickable, can't do anything in it"
security tab the user meant: not a broken widget, but a missing one.

This view is a thin, read-mostly consumer of SecuritySentinel, matching the
pattern DiagnosticsCentreView already uses elsewhere in this package:

    • POSTURE       — ``SecuritySentinel.status()`` pulled on a timer (only
                      while the tab is visible) — externally-reachable
                      ports, process count, a plain-English verdict.
    • MONITORING    — an ON/OFF toggle over ``SecuritySentinel.enabled`` /
                      ``set_enabled()``, mirroring CommandCentreWindow's
                      existing AUTONOMY toggle.
    • ALERT HISTORY — every 🛡-prefixed banner SecuritySentinel has fired
                      this session, tapped off ``bus.banner`` (its only
                      broadcast channel) so nothing that used to vanish off
                      the HUD in a few seconds is lost anymore.

Strictly a CONSUMER + one control surface, same contract as the rest of the
GUI: it never reaches into SecuritySentinel's internals beyond the two public
methods above, and every render is a plain dictionary/attribute read.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..utils import now_stamp


class SecurityCentreView(QWidget):
    """The SECURITY page: live posture, a monitoring toggle, alert history."""

    SAMPLE_MS = 4000

    def __init__(self, bus: OrionBus, security: Any | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.security = security
        self._alerts: Deque[str] = deque(maxlen=200)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("SECURITY CENTRE")
        title.setObjectName("titleLabel")
        header.addWidget(title, 1)
        self.toggle_btn = QPushButton("MONITORING: ?")
        self.toggle_btn.setToolTip(
            "Toggle SecuritySentinel's proactive host monitoring "
            "(listening ports, suspicious processes, new drives) on/off."
        )
        self.toggle_btn.clicked.connect(self._toggle_monitoring)
        self.toggle_btn.setEnabled(self.security is not None)
        header.addWidget(self.toggle_btn)
        refresh_btn = QPushButton("REFRESH")
        refresh_btn.clicked.connect(self._refresh_status)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        status_frame = QFrame()
        status_frame.setObjectName("panelFrame")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_heading = QLabel("POSTURE")
        status_heading.setObjectName("panelHeading")
        status_layout.addWidget(status_heading)
        self.status_label = QLabel("Awaiting first sample…")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        root.addWidget(status_frame)

        alerts_frame = QFrame()
        alerts_frame.setObjectName("panelFrame")
        alerts_layout = QVBoxLayout(alerts_frame)
        alerts_layout.setContentsMargins(12, 12, 12, 12)
        alerts_heading = QLabel("ALERT HISTORY (this session)")
        alerts_heading.setObjectName("panelHeading")
        alerts_layout.addWidget(alerts_heading)
        self.alert_box = QPlainTextEdit()
        self.alert_box.setReadOnly(True)
        self.alert_box.setObjectName("logBox")
        self.alert_box.setPlaceholderText(
            "Port, process and drive alerts from SecuritySentinel appear "
            "here as they fire — they otherwise only flash on the HUD "
            "banner for a few seconds and are gone."
        )
        alerts_layout.addWidget(self.alert_box, 1)
        root.addWidget(alerts_frame, 1)

        # SecuritySentinel has no dedicated bus channel for alerts — it
        # speaks through the shared banner — so tap that and keep only its
        # own lines (they all start with the shield glyph _alert() uses).
        self.bus.banner.connect(self._on_banner)

        self._timer = QTimer(self)
        self._timer.setInterval(self.SAMPLE_MS)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()
        self._refresh_status()

    # ── bus tap ───────────────────────────────────────────────────────────────

    def _on_banner(self, text: str, _priority: int) -> None:
        if not str(text).startswith("🛡"):
            return
        self._alerts.append(f"{now_stamp()}  {text}")
        self.alert_box.setPlainText("\n".join(reversed(self._alerts)))

    # ── posture + toggle ─────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        if not self.isVisible():
            return  # tab not open — skip the psutil work entirely
        if self.security is None:
            self.status_label.setText("SecuritySentinel is not wired into this build.")
            return
        self.toggle_btn.setText(
            f"MONITORING: {'ON' if self.security.enabled else 'OFF'}"
        )
        try:
            result = self.security.status()
            self.status_label.setText(result.text)
        except Exception as exc:
            self.status_label.setText(
                f"Status unavailable — {str(exc).splitlines()[0][:120]}"
            )

    def _toggle_monitoring(self) -> None:
        if self.security is None:
            return
        self.security.set_enabled(not self.security.enabled)
        self._refresh_status()

    # QTimer keeps sampling even while the tab is hidden (it's cheap to
    # check isVisible each tick); refresh immediately whenever it's shown so
    # the posture line is never stale from before the last SAMPLE_MS tick.
    def showEvent(self, event: Any) -> None:
        self._refresh_status()
        super().showEvent(event)
