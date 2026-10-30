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
    • ALERT HISTORY — every "Security: " banner SecuritySentinel has fired
                      this session, tapped off ``bus.banner`` (its only
                      broadcast channel) so nothing that used to vanish off
                      the HUD in a few seconds is lost anymore.

Mark XXXX.3 brought the rest of his security into one place:

    • AUDIT         — security_audit.gather_checks(): the score and every
                      check, failures first with their fix (Run Audit).
    • DEFENDER      — real-time protection, definitions age, last quick scan
                      and the threats Defender has recorded.
    • PHONE CONTROL — the master switch (phone_control.py) and the last
                      steps the phone made this computer take.
    • PRIVATE LOG   — session_log's switch: the log keeps what he does,
                      never what is said.

The audit and Defender's status run PowerShell and the registry, so they are
read on the page's worker thread when the page is first shown and on Run
Audit, never on a timer.

Strictly a CONSUMER + one control surface, same contract as the rest of the
GUI: it never reaches into SecuritySentinel's internals beyond the two public
methods above, and every render is a plain dictionary/attribute read.
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bus import OrionBus
from ..constants import C
from ..utils import now_stamp


_POSTURE_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orion-security-ui")


class SecurityCentreView(QWidget):
    """The SECURITY page: live posture, a monitoring toggle, alert history."""

    SAMPLE_MS = 4000

    def __init__(self, bus: OrionBus, security: Any | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.bus = bus
        self.security = security
        self.gateway: Any | None = None
        self._alerts: Deque[str] = deque(maxlen=200)
        self._posture_pending: Future | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Security Centre")
        title.setObjectName("titleLabel")
        header.addWidget(title, 1)
        self.toggle_btn = QPushButton("Monitoring: ?")
        self.toggle_btn.setToolTip(
            "Toggle SecuritySentinel's proactive host monitoring "
            "(listening ports, suspicious processes, new drives) on/off."
        )
        self.toggle_btn.clicked.connect(self._toggle_monitoring)
        self.toggle_btn.setEnabled(self.security is not None)
        header.addWidget(self.toggle_btn)
        self.private_btn = QPushButton("Private Log: ?")
        self.private_btn.setAccessibleName("Private log")
        self.private_btn.clicked.connect(self._toggle_private_log)
        header.addWidget(self.private_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_status)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        status_frame = QFrame()
        status_frame.setObjectName("panelFrame")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_heading = QLabel("Posture")
        status_heading.setObjectName("panelHeading")
        status_layout.addWidget(status_heading)
        self.status_label = QLabel("Awaiting first sample…")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        root.addWidget(status_frame)

        audit_frame = QFrame()
        audit_frame.setObjectName("panelFrame")
        audit_layout = QVBoxLayout(audit_frame)
        audit_layout.setContentsMargins(12, 12, 12, 12)
        audit_header = QHBoxLayout()
        audit_heading = QLabel("Audit")
        audit_heading.setObjectName("panelHeading")
        audit_header.addWidget(audit_heading, 1)
        self.audit_score = QLabel("–")
        self.audit_score.setObjectName("summaryValue")
        audit_header.addWidget(self.audit_score)
        self.audit_btn = QPushButton("Run Audit")
        self.audit_btn.clicked.connect(self.run_audit)
        audit_header.addWidget(self.audit_btn)
        audit_layout.addLayout(audit_header)
        self.audit_list = QListWidget()
        self.audit_list.setAccessibleName("Security audit checks")
        self.audit_list.setMaximumHeight(200)
        self.audit_list.setWordWrap(True)
        audit_layout.addWidget(self.audit_list)
        self.defender_label = QLabel("Defender: –")
        self.defender_label.setObjectName("mutedLabel")
        self.defender_label.setWordWrap(True)
        audit_layout.addWidget(self.defender_label)
        root.addWidget(audit_frame)

        phone_frame = QFrame()
        phone_frame.setObjectName("panelFrame")
        phone_layout = QVBoxLayout(phone_frame)
        phone_layout.setContentsMargins(12, 12, 12, 12)
        phone_heading = QLabel("Phone Access")
        phone_heading.setObjectName("panelHeading")
        phone_layout.addWidget(phone_heading)
        self.phone_switch = QPushButton("Phone Control: ?")
        self.phone_switch.setAccessibleName("Phone control master switch")
        self.phone_switch.clicked.connect(self._toggle_phone_control)
        phone_layout.addWidget(self.phone_switch)
        self.phone_activity = QListWidget()
        self.phone_activity.setAccessibleName("Recent phone activity")
        self.phone_activity.setMaximumHeight(110)
        phone_layout.addWidget(self.phone_activity)
        self.phone_status = QLabel("Waiting for the phone gateway…")
        self.phone_status.setObjectName("mutedLabel")
        self.phone_status.setWordWrap(True)
        phone_layout.addWidget(self.phone_status)
        self.pair_phone_btn = QPushButton("Pair Phone")
        self.pair_phone_btn.setEnabled(False)
        self.pair_phone_btn.clicked.connect(self._pair_phone)
        phone_layout.addWidget(self.pair_phone_btn)
        self.devices = QListWidget()
        self.devices.setMaximumHeight(110)
        self.devices.setAccessibleName("Paired phone devices")
        phone_layout.addWidget(self.devices)
        self.revoke_device_btn = QPushButton("Revoke Selected Device")
        self.revoke_device_btn.setEnabled(False)
        self.revoke_device_btn.clicked.connect(self._revoke_device)
        self.devices.itemSelectionChanged.connect(lambda: self.revoke_device_btn.setEnabled(
            self.gateway is not None and self.devices.currentItem() is not None
            and bool(self.devices.currentItem().data(Qt.ItemDataRole.UserRole))))
        phone_layout.addWidget(self.revoke_device_btn)
        root.addWidget(phone_frame)

        alerts_frame = QFrame()
        alerts_frame.setObjectName("panelFrame")
        alerts_layout = QVBoxLayout(alerts_frame)
        alerts_layout.setContentsMargins(12, 12, 12, 12)
        alerts_heading = QLabel("Alerts")
        alerts_heading.setObjectName("panelHeading")
        alerts_layout.addWidget(alerts_heading)
        self.alert_box = QPlainTextEdit()
        self.alert_box.setReadOnly(True)
        self.alert_box.setObjectName("logBox")
        self.alert_box.setMinimumHeight(100)
        self.alert_box.setMaximumHeight(260)
        self.alert_box.hide()
        self.empty_alerts = QLabel("No alerts")
        self.empty_alerts.setObjectName("mutedLabel")
        alerts_layout.addWidget(self.empty_alerts)
        alerts_layout.addWidget(self.alert_box, 1)
        root.addWidget(alerts_frame)
        root.addStretch(1)

        # SecuritySentinel has no dedicated bus channel for alerts — it
        # speaks through the shared banner — so tap that and keep only its
        # own lines (they all start with the shield glyph _alert() uses).
        self.bus.banner.connect(self._on_banner)

        self._timer = QTimer(self)
        self._timer.setInterval(self.SAMPLE_MS)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()
        self._posture_timer = QTimer(self)
        self._posture_timer.setInterval(50)
        self._posture_timer.timeout.connect(self._poll_posture)
        self._audit_pending: Future | None = None
        self._audited = False
        self._audit_timer = QTimer(self)
        self._audit_timer.setInterval(100)
        self._audit_timer.timeout.connect(self._poll_audit)
        self._refresh_switches()
        self._refresh_status()

    def attach_gateway(self, gateway: Any) -> None:
        """Enable pairing when the desktop gateway has finished starting."""
        self.gateway = gateway
        self.pair_phone_btn.setEnabled(gateway is not None)
        self.phone_status.setText(
            "Open the ORION app on your phone, then enter the code shown here."
            if gateway is not None else "Phone gateway unavailable."
        )
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        selected = self.devices.currentItem()
        selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        self.devices.clear()
        if self.gateway is None:
            self.devices.addItem("Phone gateway unavailable.")
            return
        try:
            for device in self.gateway.auth.list_devices():
                state = "revoked" if device.get("revoked") else "active"
                item = QListWidgetItem(f"{device.get('name') or 'Phone'} · {state}")
                if not device.get("revoked"):
                    item.setData(Qt.ItemDataRole.UserRole, device["device_id"])
                self.devices.addItem(item)
                if device.get("device_id") == selected_id:
                    self.devices.setCurrentItem(item)
            if not self.devices.count():
                self.devices.addItem("No paired phones yet. Select PAIR PHONE to add one.")
        except Exception:
            self.devices.addItem("Device listing unavailable. Check PC diagnostics.")

    def _revoke_device(self) -> None:
        item = self.devices.currentItem()
        device_id = item.data(Qt.ItemDataRole.UserRole) if item else None
        if self.gateway is None or not device_id:
            return
        try:
            self.gateway.auth.revoke_device(device_id)
            self.phone_status.setText("Device revoked. Its saved credentials no longer work.")
            self._refresh_devices()
        except Exception:
            self.phone_status.setText("Could not revoke that device. Check PC diagnostics.")

    def _pair_phone(self) -> None:
        """Show a fresh one-time code for the connected Android app."""
        if self.gateway is None:
            return
        try:
            code = self.gateway.begin_pairing()
            hint = ""
            try:
                hint = str(self.gateway.lan_pairing_hint() or "")
            except Exception:
                hint = ""
            self.phone_status.setText(
                f"Enter {code} in the ORION app. This code expires in 10 minutes."
                + (f"\n{hint}" if hint else "")
            )
        except Exception as exc:
            self.phone_status.setText(
                f"Could not start pairing: {str(exc).splitlines()[0][:120]}"
            )

    # ── bus tap ───────────────────────────────────────────────────────────────

    def _on_banner(self, text: str, _priority: int) -> None:
        # "Security: " since 18 Oct (a colour shield emoji before; still
        # accepted, so nothing a stale emitter sends is lost).
        if not str(text).startswith(("Security: ", "🛡")):
            return
        self._alerts.append(f"{now_stamp()}  {text}")
        self.alert_box.setPlainText("\n".join(reversed(self._alerts)))
        self.empty_alerts.hide()
        self.alert_box.show()

    # ── posture + toggle ─────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        if not self.isVisible():
            return  # tab not open — skip the psutil work entirely
        self._refresh_devices()
        if self.security is None:
            self.status_label.setText("Security monitor unavailable")
            return
        self.toggle_btn.setText(
            f"MONITORING: {'ON' if self.security.enabled else 'OFF'}"
        )
        if self._posture_pending is None:
            # Windows connection enumeration may take hundreds of
            # milliseconds. The thread painting ORION's face must never wait
            # for it, and repeated refresh clicks must not queue more scans.
            self._posture_pending = _POSTURE_WORKER.submit(self.security.status)
            self._posture_timer.start()

    def _poll_posture(self) -> None:
        pending = self._posture_pending
        if pending is None or not pending.done():
            return
        self._posture_pending = None
        self._posture_timer.stop()
        try:
            self.status_label.setText(pending.result().text)
        except Exception as exc:
            self.status_label.setText(
                f"Status unavailable — {str(exc).splitlines()[0][:120]}"
            )

    def _toggle_monitoring(self) -> None:
        if self.security is None:
            return
        self.security.set_enabled(not self.security.enabled)
        self._refresh_status()

    # ── switches: phone control, private log ─────────────────────────────────

    def _refresh_switches(self) -> None:
        try:
            from .. import phone_control
            on = phone_control.enabled()
            self.phone_switch.setText(f"Phone Control: {'ON' if on else 'OFF'}")
            self.phone_activity.clear()
            steps = phone_control.recent(7)[-8:]
            for entry in reversed(steps):
                stamp = time.strftime("%d %b %H:%M", time.localtime(float(entry.get("at") or 0)))
                tool = str(entry.get("tool") or "?").replace("_", " ")
                self.phone_activity.addItem(f"{stamp}  ·  {tool}  ·  {entry.get('outcome') or ''}")
            if not steps:
                self.phone_activity.addItem("No phone activity in 7 days")
        except Exception:
            self.phone_switch.setText("Phone Control: ?")
        try:
            from .. import session_log
            self.private_btn.setText(f"Private Log: {'ON' if session_log.private() else 'OFF'}")
        except Exception:
            pass

    def _toggle_phone_control(self) -> None:
        try:
            from .. import phone_control
            phone_control.set_enabled(not phone_control.enabled())
        except Exception:
            pass
        self._refresh_switches()

    def _toggle_private_log(self) -> None:
        try:
            from .. import session_log
            session_log.set_private(not session_log.private())
        except Exception:
            pass
        self._refresh_switches()

    # ── audit + Defender (worker thread) ─────────────────────────────────────

    def run_audit(self) -> None:
        """Read the audit and Defender's state off the GUI thread."""
        if self._audit_pending is not None:
            return
        self._audited = True
        self.audit_btn.setEnabled(False)
        self.audit_score.setText("…")
        self._audit_pending = _POSTURE_WORKER.submit(_read_audit, self.security)
        self._audit_timer.start()

    def _poll_audit(self) -> None:
        pending = self._audit_pending
        if pending is None or not pending.done():
            return
        self._audit_pending = None
        self._audit_timer.stop()
        self.audit_btn.setEnabled(True)
        try:
            checks, defender = pending.result()
        except Exception as exc:
            self.audit_score.setText("–")
            self.defender_label.setText(f"Audit unavailable — {str(exc).splitlines()[0][:120]}")
            return
        self.show_audit(checks, defender)

    def show_audit(self, checks: list[Any], defender: str) -> None:
        from ..security_audit import score
        passed, checked = score(checks)
        self.audit_score.setText(f"{passed} / {checked}")
        self.audit_list.clear()
        ordered = sorted(checks, key=lambda c: {False: 0, None: 1, True: 2}[c.ok])
        for check in ordered:
            mark = {True: "✓", False: "✗", None: "–"}[check.ok]
            text = f"{mark}  {check.name} — {check.detail}"
            if check.ok is False and check.fix:
                text += f"\n     Fix: {check.fix}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, check.ok)
            if check.ok is False:
                item.setForeground(QColor(C.PRI_HI))
            self.audit_list.addItem(item)
        self.defender_label.setText(defender)

    # QTimer keeps sampling even while the tab is hidden (it's cheap to
    # check isVisible each tick); refresh immediately whenever it's shown so
    # the posture line is never stale from before the last SAMPLE_MS tick.
    def showEvent(self, event: Any) -> None:
        self._refresh_status()
        self._refresh_switches()
        if not self._audited:
            self.run_audit()
        super().showEvent(event)


def _days(count: int) -> str:
    return f"{count} Day" if count == 1 else f"{count} Days"


def _read_audit(security: Any) -> tuple[list[Any], str]:
    """(audit checks, one Defender line) - blocking, for the worker thread."""
    from ..security_audit import gather_checks
    checks = gather_checks(security)
    parts: list[str] = []
    try:
        from .. import antivirus
        status = antivirus.get_protection_status_sync()
        if status is None:
            parts.append("Defender: not readable")
        else:
            parts.append("Real-Time Protection " + ("On" if status.realtime_protection else "Off"))
            if status.signature_age_days is not None:
                parts.append(f"Definitions {_days(status.signature_age_days)} Old")
            if status.quick_scan_age_days is not None:
                parts.append(f"Quick Scan {_days(status.quick_scan_age_days)} Ago")
            threats = antivirus.get_recent_threats_sync()
            parts.append(f"{len(threats)} {'Threat' if len(threats) == 1 else 'Threats'} On Record"
                         if threats else "No Threats On Record")
            for threat in threats[:3]:
                parts.append(f"{threat.name} ({threat.severity}, {threat.status})")
    except Exception as exc:
        parts.append(f"Defender: {type(exc).__name__}")
    return checks, "  ·  ".join(parts)
