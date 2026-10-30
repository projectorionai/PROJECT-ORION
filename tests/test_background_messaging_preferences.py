"""Account setup grants originate at trusted Preferences actions only."""
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import sip
from PyQt6.QtWidgets import QApplication

from orion_core.data import ToolResult
from orion_core.execution_policy import foreground_authorized, is_background
from orion_core.gui import preferences


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def queued(monkeypatch):
    work = []
    monkeypatch.setattr(preferences.threading, "Thread", lambda target, **_kwargs:
                        SimpleNamespace(start=lambda: work.append(target)))
    return work


def test_construction_does_not_query_accounts_or_launch_browser(app, queued):
    widget = preferences._MessagingConnections(lambda: pytest.fail("queried account at boot"))
    try:
        assert not queued
        assert all(status.text() == "Not checked" for status, *_ in widget._rows.values())
    finally:
        sip.delete(widget)


@pytest.mark.parametrize("action", ["connect", "check"])
def test_authority_is_created_inside_physical_connection_worker(app, queued, action):
    seen = []
    def request(platform):
        seen.append((platform, foreground_authorized(), is_background()))
        return ToolResult("synthetic ready", ok=True)
    gateway = SimpleNamespace(browser_connect=request, browser_check_connection=request)
    widget = preferences._MessagingConnections(gateway)
    try:
        widget._operate("whatsapp", action)
        assert not seen and len(queued) == 1
        assert not foreground_authorized()
        widget._operate("whatsapp", action)
        assert len(queued) == 1
        queued.pop()()
        assert seen == [("whatsapp", action == "connect", action == "check")]
        assert not foreground_authorized()
        status, connect, check = widget._rows["whatsapp"]
        assert status.text() == "synthetic ready"
        assert connect.isEnabled() and check.isEnabled()
    finally:
        sip.delete(widget)


def test_late_connection_result_survives_closed_preferences(app, queued):
    calls = []
    widget = preferences._MessagingConnections(SimpleNamespace(
        browser_check_connection=lambda platform: calls.append(platform) or ToolResult("ready", ok=True)))
    widget._operate("discord", "check")
    sip.delete(widget)
    queued.pop()()
    assert calls == ["discord"]


def test_unready_provider_keeps_controls_usable(app, queued):
    def unavailable():
        raise RuntimeError("dispatcher retiring")
    widget = preferences._MessagingConnections(unavailable)
    try:
        widget._operate("discord", "check")
        status, connect, check = widget._rows["discord"]
        assert "unavailable" in status.text()
        assert connect.isEnabled() and check.isEnabled()
        assert not queued
    finally:
        sip.delete(widget)
