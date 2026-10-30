"""Nonmodal approvals use real Qt and guard tokens, with no live executors."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QPushButton, QTextEdit

from orion_core.data import ToolResult
from orion_core.execution_policy import foreground_authorized
from orion_core.gui import core_window
from orion_core.system_guard import ActionIntent, DecisionKind, SystemActionGuard


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def host(app, monkeypatch):
    created = []
    class QuietBox(QMessageBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
        def exec(self):
            pytest.fail("Background approval entered a modal loop")
        def activateWindow(self):
            pytest.fail("Background approval activated a window")
        def raise_(self):
            pytest.fail("Background approval raised a window")
        def setFocus(self, *args):
            pytest.fail("Background approval requested keyboard focus")
    monkeypatch.setattr(core_window, "QMessageBox", QuietBox)
    clock = [100.0]
    guard = SystemActionGuard("synthetic", ttl_s=60, clock=lambda: clock[0])
    confirmed, cancelled, grants, logs = [], [], [], []
    def confirm(token):
        grants.append(foreground_authorized())
        decision = guard.confirm(token)
        if decision.kind is DecisionKind.EXECUTE:
            confirmed.append((token, deepcopy(decision.payload)))
        return ToolResult(decision.message, ok=decision.kind is DecisionKind.EXECUTE)
    def cancel(token):
        cancelled.append(token)
        guard.cancel(token)
        return ToolResult("Cancelled")
    dispatcher = SimpleNamespace(system_guard=guard, confirm_system_action=confirm,
                                 cancel_system_action=cancel)
    # Initialise the real Qt base without starting the avatar/account/runtime
    # construction paths; this is the actual CoreWindow confirmation handler.
    window = core_window.OrionCoreWindow.__new__(core_window.OrionCoreWindow)
    QMainWindow.__init__(window)
    window.worker = SimpleNamespace(dispatcher=dispatcher)
    window.bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append),
                                 request_shutdown=SimpleNamespace(emit=lambda: None))
    window._shutting_down = False
    window._orb_hands_enabled = False
    state = SimpleNamespace(window=window, dispatcher=dispatcher, guard=guard, clock=clock,
                            created=created, confirmed=confirmed, cancelled=cancelled, grants=grants)
    yield state
    if not sip.isdeleted(window):
        window._cancel_background_plugin_prompts()
        sip.delete(window)
    app.processEvents()


def _payload(host, arguments=None, *, intent=ActionIntent.SEND_MESSAGE):
    bound = {"plugin": "synthetic_message", "arguments": deepcopy(arguments or {
        "action": "send", "to": "synthetic-recipient", "message": "exact body"})}
    decision = host.guard.request_confirmation(intent, origin="plugin:synthetic_message", payload=bound)
    return {"token": decision.token, "intent": "plugin_action", "background": True,
            "plugin": bound["plugin"], "arguments": deepcopy(bound["arguments"])}


def _show(host, payload):
    host.window._on_confirm_action(payload)
    return host.window._background_plugin_prompts[payload["token"]]


def _button(box, label):
    return next(button for button in box.buttons() if button.text() == label)


def test_creation_is_nonmodal_nonactivating_and_does_not_execute_an_account_action(host):
    payload = _payload(host)
    state = _show(host, payload)
    box = state["box"]
    assert box.windowModality() == Qt.WindowModality.NonModal
    assert not box.isModal()
    assert box.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert box.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert box.defaultButton() is _button(box, "Cancel")
    assert host.confirmed == host.cancelled == host.grants == []
    assert payload["token"] in host.guard._pending


def test_full_exact_arguments_are_read_only_plain_text_and_not_truncated(host):
    body = "  <b>literal</b> É🙂\n" * 500 + "END  "
    arguments = {"action": "send", "to": "synthetic-recipient", "message": body,
                 "nested": {"enabled": False, "count": 0}}
    payload = _payload(host, arguments)
    box = _show(host, payload)["box"]
    assert json.loads(box.detailedText()) == {"plugin": "synthetic_message", "arguments": arguments}
    assert box.textFormat() == Qt.TextFormat.PlainText
    editors = box.findChildren(QTextEdit)
    assert editors and all(editor.isReadOnly() for editor in editors)
    assert body == json.loads(box.detailedText())["arguments"]["message"]


def test_approve_releases_exact_bound_token_once_without_a_foreground_grant(host):
    payload = _payload(host)
    state = _show(host, payload)
    original = deepcopy(payload["arguments"])
    payload["arguments"]["message"] = "GUI payload changed after presentation"
    _button(state["box"], "Approve").click()
    state["finish"](approved=True)
    assert host.confirmed == [(payload["token"], {"plugin": "synthetic_message", "arguments": original})]
    assert host.grants == [False]
    assert host.cancelled == []
    assert host.window._background_plugin_prompts == {}
    assert payload["token"] not in host.guard._pending


def test_details_button_does_not_cancel_or_approve_the_action(host):
    payload = _payload(host)
    state = _show(host, payload)
    unrelated = QPushButton("Show Details", state["box"])
    state["box"].buttonClicked.emit(unrelated)
    assert not state["done"]
    assert host.confirmed == host.cancelled == []


@pytest.mark.parametrize("closing", ["cancel", "close", "expire", "host_close", "host_delete"])
def test_cancel_expiry_and_host_lifetime_never_release_an_action(host, closing):
    payload = _payload(host)
    state = _show(host, payload)
    if closing == "cancel":
        _button(state["box"], "Cancel").click()
    elif closing == "close":
        state["box"].close()
    elif closing == "expire":
        host.clock[0] += 61
        state["timer"].timeout.emit()
    elif closing == "host_close":
        host.window.close()
    else:
        sip.delete(host.window)
    state["finish"](approved=True)  # A retained late callback is inert.
    assert host.confirmed == []
    assert host.cancelled == [payload["token"]]
    assert payload["token"] not in host.guard._pending


def test_expired_approve_is_refused_even_before_the_expiry_timer_fires(host):
    payload = _payload(host)
    state = _show(host, payload)
    host.clock[0] += 61
    _button(state["box"], "Approve").click()
    assert host.confirmed == []
    assert host.cancelled == [payload["token"]]


def test_mutated_guard_payload_cannot_release_a_different_action(host):
    payload = _payload(host)
    state = _show(host, payload)
    host.guard._pending[payload["token"]].payload["arguments"]["message"] = "different action"
    _button(state["box"], "Approve").click()
    assert host.confirmed == []
    assert host.cancelled == [payload["token"]]


def test_duplicate_signals_do_not_create_multiple_prompts_and_overflow_is_cancelled(host):
    payloads = [_payload(host) for _ in range(5)]
    for payload in payloads[:4]:
        _show(host, payload)
    host.window._on_confirm_action(payloads[0])
    host.window._on_confirm_action(payloads[4])
    assert len(host.created) == len(host.window._background_plugin_prompts) == 4
    assert host.cancelled == [payloads[4]["token"]]
    assert host.confirmed == []


@pytest.mark.parametrize("invalid", ["arguments", "plugin", "intent", "token", "bound_intent", "oversize", "expired"])
def test_invalid_background_requests_are_cancelled_without_a_modal_fallback(host, invalid):
    payload = _payload(host, intent=ActionIntent.PROCESS_KILL if invalid == "bound_intent" else ActionIntent.SEND_MESSAGE)
    token = payload["token"]
    if invalid in {"arguments", "plugin", "intent"}:
        payload[invalid] = {} if invalid == "arguments" else "different"
    elif invalid == "token":
        payload["token"] = "unknown-token"
    elif invalid == "oversize":
        arguments = {"message": "x" * (256 * 1024)}
        payload["arguments"] = deepcopy(arguments)
        host.guard._pending[token].payload["arguments"] = arguments
    elif invalid == "expired":
        host.clock[0] += 61
    host.window._on_confirm_action(payload)
    assert host.created == host.confirmed == []
    assert host.cancelled == [payload["token"]]


def test_shutdown_cancels_pending_boxes_before_late_approval(host):
    payload = _payload(host)
    state = _show(host, payload)
    host.window._mark_shutting_down()
    state["finish"](approved=True)
    assert host.confirmed == []
    assert host.cancelled == [payload["token"]]


def test_other_destructive_foreground_dialogs_keep_the_existing_modal_path(host, monkeypatch):
    payload = _payload(host, intent=ActionIntent.OS_SHUTDOWN)
    payload.update(background=False, intent="os_shutdown", message="Synthetic power action")
    modal = []
    def decline(box):
        modal.append(box)
        return 0
    monkeypatch.setattr(core_window.QMessageBox, "exec", decline)
    host.window._on_confirm_action(payload)
    assert len(modal) == 1
    assert host.confirmed == []
    assert host.cancelled == [payload["token"]]


def test_settled_prompts_release_host_receivers_and_delete_boxes_and_timers(host, app):
    baseline = host.window.receivers(host.window.destroyed)
    retired = []
    for index in range(8):
        payload = _payload(host)
        state = _show(host, payload)
        retired.append((state["box"], state["timer"]))
        _button(state["box"], "Approve" if index % 2 else "Cancel").click()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert host.window.receivers(host.window.destroyed) == baseline
        assert host.window._background_plugin_prompts == {}
        assert payload["token"] not in host.guard._pending
    assert len(host.confirmed) == len(host.cancelled) == 4
    assert all(sip.isdeleted(box) and sip.isdeleted(timer) for box, timer in retired)


# ── installing software only ever on the owner's click (Mark XXXX.4) ──────────

WINGET_SEARCH = (
    "   - \r\n"
    "Name                Id                              Version Match\r\n"
    "---------------------------------------------------------------------\r\n"
    "VLC media player    VideoLAN.VLC                    3.0.24\r\n"
    "BTChargeTrayWatcher peterandree.BTChargeTrayWatcher 3.2.0   Tag: ntfy\r\n"
)


def test_winget_search_output_is_read_into_packages():
    from orion_core.software_install import ID_RE, parse_table
    packages = parse_table(WINGET_SEARCH)
    assert [(p.name, p.id, p.version) for p in packages] == [
        ("VLC media player", "VideoLAN.VLC", "3.0.24"),
        ("BTChargeTrayWatcher", "peterandree.BTChargeTrayWatcher", "3.2.0")]
    assert not ID_RE.match("VLC & del C:\\") and not ID_RE.match("--source msstore")


def test_install_only_asks_and_the_click_installs(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from orion_core import software_install as si
    from orion_core.dispatch_desktop import DesktopDispatchMixin
    from orion_core.system_guard import SystemActionGuard

    ran = []
    monkeypatch.setattr(si, "winget", lambda: "winget.exe")
    monkeypatch.setattr(si, "resolve", lambda target: (si.Package("VLC media player",
                                                                   "VideoLAN.VLC", "3.0.24"), []))
    monkeypatch.setattr(si, "install", lambda package_id: (ran.append(package_id) or True,
                                                           f"Installed {package_id}."))
    asked, logged = [], []
    host = SimpleNamespace(
        system_guard=SystemActionGuard(),
        bus=SimpleNamespace(confirm_action=SimpleNamespace(emit=asked.append),
                            log=SimpleNamespace(emit=logged.append),
                            banner=SimpleNamespace(emit=lambda *a: None)))
    result = asyncio.run(DesktopDispatchMixin.software_install_tool(host, {"package": "vlc"}))
    assert result.ok is False and "Nothing is installed until you click" in result.text
    assert ran == [] and asked and "VideoLAN.VLC" in asked[0]["message"]
    assert "licence" in asked[0]["message"]

    async def approve():
        host._run_confirmed_install = lambda payload: DesktopDispatchMixin._run_confirmed_install(
            host, payload)
        done = DesktopDispatchMixin.confirm_system_action(host, asked[0]["token"])
        await asyncio.sleep(0.2)
        return done

    done = asyncio.run(approve())
    assert done.ok and ran == ["VideoLAN.VLC"]
    assert any(line.startswith("INSTALL: VideoLAN.VLC finished") for line in logged)
    again = DesktopDispatchMixin.confirm_system_action(host, asked[0]["token"])
    assert again.ok is False                         # a token installs once
