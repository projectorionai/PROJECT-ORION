"""Offline remote chat routes phone commands without changing desktop routing."""

from __future__ import annotations

import asyncio

from orion_core.local_brain import LocalBrain
from orion_core.remote_capability import RemoteConfirmationRegistry, RemoteToolGate
from orion_core.remote_mobile_ws import ViewportDom


class _Desktop:
    def __init__(self):
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        from orion_core.data import ToolResult
        return ToolResult("desktop done")


class _Bridge:
    def __init__(self):
        self.calls = []
        self.snapshot = {
            "snapshotId": "snap-1", "package": "com.android.settings",
            "root": {"id": "0", "children": [
                {"id": "0/1", "text": "Wi-Fi", "clickable": True}]},
        }

    async def request_device_action(self, device_id, method, params):
        self.calls.append((device_id, method, params))
        return {"authorized": True, "ok": True, "snapshot": self.snapshot}

    def latest_dom(self, device_id):
        return ViewportDom(self.snapshot, 1.0)


def test_remote_chat_can_read_phone_controls_and_propose_action():
    desktop = _Desktop()
    bridge = _Bridge()
    registry = RemoteConfirmationRegistry()
    gate = RemoteToolGate(desktop, registry)
    gate.mobile_viewport = bridge
    brain = LocalBrain(None, None, gate, None)

    async def scenario():
        with gate.for_device("owner"):
            summary = await brain.try_task("what is on my phone screen?")
            opened = await brain.try_task("open settings on my phone")
            tapped = await brain.try_task("tap node 0/1 on my phone")
            called = await brain.try_task("call +441234567890")
        assert "0/1: Wi-Fi" in summary
        # Each action is proposed to the phone, which asks for its own
        # slide-to-confirm before acting; the reply is its authorised result.
        assert opened == "Open app completed on the phone."
        assert tapped == "Click completed on the phone."
        assert called == "Make call completed on the phone."

    asyncio.run(scenario())
    assert bridge.calls == [
        ("owner", "viewport.capture_dom", {}),
        ("owner", "device.open_app", {"package": "com.android.settings"}),
        ("owner", "viewport.click", {"snapshotId": "snap-1", "nodeId": "0/1"}),
        ("owner", "device.make_call", {"number": "+441234567890"}),
    ]
    assert desktop.calls == []
    # No second, web-side approval: it would pull focus from the app the
    # gesture is aimed at.
    assert registry.pending_for("owner") == []


def test_desktop_local_brain_keeps_existing_open_app_route():
    desktop = _Desktop()
    brain = LocalBrain(None, None, desktop, None)
    reply = asyncio.run(brain.try_task("open Notepad"))
    assert reply
    assert desktop.calls == [("open_app", {"app_name": "Notepad"})]


def test_remote_chat_can_resolve_contact_name_for_call_and_text():
    desktop = _Desktop()
    bridge = _Bridge()
    gate = RemoteToolGate(desktop, RemoteConfirmationRegistry())
    gate.mobile_viewport = bridge
    brain = LocalBrain(None, None, gate, None)

    async def scenario():
        with gate.for_device("owner"):
            await brain.try_task("call Jordan on my phone")
            await brain.try_task("text Jordan saying Running late")

    asyncio.run(scenario())
    assert bridge.calls == [
        ("owner", "device.make_call", {"number": "Jordan"}),
        ("owner", "device.send_sms", {"number": "Jordan", "body": "Running late"}),
    ]
    assert desktop.calls == []


def test_remote_chat_taps_unique_visible_label_from_approved_snapshot():
    desktop = _Desktop()
    bridge = _Bridge()
    gate = RemoteToolGate(desktop, RemoteConfirmationRegistry())
    gate.mobile_viewport = bridge
    brain = LocalBrain(None, None, gate, None)

    async def scenario():
        with gate.for_device("owner"):
            return await brain.try_task("tap Wi-Fi on my phone")

    assert asyncio.run(scenario()) == "Click completed on the phone."
    assert bridge.calls == [
        ("owner", "viewport.click", {"snapshotId": "snap-1", "nodeId": "0/1"})]
    assert desktop.calls == []


def test_generic_my_screen_request_stays_on_paired_phone():
    desktop = _Desktop()
    bridge = _Bridge()
    gate = RemoteToolGate(desktop, RemoteConfirmationRegistry())
    gate.mobile_viewport = bridge
    brain = LocalBrain(None, None, gate, None)

    async def scenario():
        with gate.for_device("owner"):
            return await brain.try_task("What is on my screen?")

    assert "0/1: Wi-Fi" in asyncio.run(scenario())
    assert bridge.calls == [("owner", "viewport.capture_dom", {})]
    assert desktop.calls == []
