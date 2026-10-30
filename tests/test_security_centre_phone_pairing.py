"""The desktop Security Centre must expose the Android pairing code."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6.QtWidgets import QApplication

from orion_core.bus import OrionBus
from orion_core.gui.security_centre import SecurityCentreView


def test_phone_pairing_button_opens_a_fresh_code() -> None:
    app = QApplication.instance() or QApplication([])

    class Gateway:
        calls = 0

        def begin_pairing(self) -> str:
            self.calls += 1
            return "ABCD-1234"

    gateway = Gateway()
    view = SecurityCentreView(OrionBus())
    try:
        assert not view.pair_phone_btn.isEnabled()
        view.attach_gateway(gateway)
        assert view.pair_phone_btn.isEnabled()
        view.pair_phone_btn.click()
        assert gateway.calls == 1
        assert "ABCD-1234" in view.phone_status.text()
        assert "ORION app" in view.phone_status.text()
    finally:
        view.close()
        del app


def test_posture_scan_never_blocks_or_queues_repeated_gui_refreshes():
    from threading import Event, get_ident
    from orion_core.data import ToolResult

    app = QApplication.instance() or QApplication([])
    entered, release = Event(), Event()
    calls = []
    caller = get_ident()

    class Security:
        enabled = True

        def status(self):
            calls.append(get_ident())
            entered.set()
            assert release.wait(2)
            return ToolResult("Posture captured")

    view = SecurityCentreView(OrionBus(), Security())
    try:
        view.isVisible = lambda: True
        view._refresh_status()
        assert entered.wait(1)
        pending = view._posture_pending
        for _ in range(100):
            view._refresh_status()
        assert view._posture_pending is pending
        assert len(calls) == 1
        assert calls[0] != caller
        release.set()
        assert pending.result(timeout=2).text == "Posture captured"
        view._poll_posture()
        assert view.status_label.text() == "Posture captured"
        assert view._posture_pending is None
    finally:
        release.set()
        view.close()
    assert app is not None


def test_alert_history_stays_compact_until_an_alert_arrives():
    app = QApplication.instance() or QApplication([])
    view = SecurityCentreView(OrionBus())
    try:
        assert view.alert_box.isHidden()
        assert view.alert_box.placeholderText() == ""
        view._on_banner("Ordinary notification", 1)
        assert view.alert_box.isHidden()
        view._on_banner("Security: Unexpected listening port", 1)
        assert not view.alert_box.isHidden()
        assert view.empty_alerts.isHidden()
        view._on_banner("🛡 Legacy prefix still recorded", 1)
        assert "Legacy prefix still recorded" in view.alert_box.toPlainText()
    finally:
        view.close()
    assert app is not None

# ── folded in from test_security_page.py (Mark XXXX.3: the Security page shows the audit, Defender, phone control and) ────

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

_QT_APP = QApplication.instance() or QApplication([])
globals().setdefault("_QT_APP", _QT_APP)

from orion_core import phone_control, session_log  # noqa: E402
from orion_core.bus import OrionBus  # noqa: E402
from orion_core.gui.security_centre import SecurityCentreView  # noqa: E402
from orion_core.security_audit import Check, score  # noqa: E402


def test_failures_come_first_with_their_fix():
    view = SecurityCentreView(OrionBus())
    checks = [Check("Firewall", True, "on"), Check("Remote Assistance", False, "on", "Untick it."),
              Check("SMBv1", None, "not readable")]
    view.show_audit(checks, "Real-Time Protection On")
    rows = [view.audit_list.item(i).text() for i in range(view.audit_list.count())]
    assert rows[0].startswith("✗  Remote Assistance") and "Fix: Untick it." in rows[0]
    assert rows[-1].startswith("✓  Firewall")
    assert view.audit_score.text() == "1 / 2"
    assert view.defender_label.text() == "Real-Time Protection On"
    assert score(checks) == (1, 2)


def test_the_phone_switch_flips_phone_control():
    view = SecurityCentreView(OrionBus())
    phone_control.set_enabled(True)
    view._refresh_switches()
    assert view.phone_switch.text() == "Phone Control: ON"
    view._toggle_phone_control()
    assert not phone_control.enabled()
    assert view.phone_switch.text() == "Phone Control: OFF"
    view._toggle_phone_control()
    assert phone_control.enabled()


def test_the_private_log_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(session_log, "_privacy_path", lambda: tmp_path / "privacy.json")
    monkeypatch.setattr(session_log, "_private", False)
    view = SecurityCentreView(OrionBus())
    view._toggle_private_log()
    assert session_log.private() and view.private_btn.text() == "Private Log: ON"
    view._toggle_private_log()
    assert not session_log.private() and view.private_btn.text() == "Private Log: OFF"
