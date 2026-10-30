"""The Android viewport is available only through a paired remote turn."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatch_schema import PHONE_DEVICE_DECLARATION, TOOL_DECLARATIONS
from orion_core.remote_capability import (
    RemoteConfirmationRegistry,
    RemoteToolGate,
    Tier,
    classify,
)
from orion_core.remote_mobile_ws import (
    MobileViewportBridge,
    ViewportDom,
    ViewportFrame,
    _valid_command_params,
)


class _Desktop:
    TOOL_DECLARATIONS = TOOL_DECLARATIONS

    def __init__(self) -> None:
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return ToolResult("desktop")


class _Bridge:
    def __init__(self) -> None:
        self.calls = []
        self.frames = {
            "owner": ViewportFrame(b"\xff\xd8frame\xff\xd9", time.monotonic()),
        }
        self.snapshot = {
            "snapshotId": "snapshot-1", "package": "com.android.settings",
            "root": {"id": "0", "children": [{"id": "0/1", "text": "Wi-Fi"}]},
        }

    def latest_frame(self, device_id):
        return self.frames.get(device_id)

    def latest_dom(self, device_id):
        return ViewportDom(self.snapshot, time.monotonic()) if device_id == "owner" else None

    async def request_device_action(self, device_id, method, params):
        self.calls.append((device_id, method, params))
        if method == "viewport.capture_dom":
            return {"authorized": True, "ok": True, "snapshot": self.snapshot}
        return {"authorized": True, "ok": True, "detail": "Performed on the phone"}


def _gate():
    desktop = _Desktop()
    bridge = _Bridge()
    registry = RemoteConfirmationRegistry()
    gate = RemoteToolGate(desktop, registry)
    gate.mobile_viewport = bridge
    return gate, desktop, bridge, registry


def test_declaration_is_remote_only_and_device_actions_use_native_approval():
    gate, _, _, _ = _gate()
    assert PHONE_DEVICE_DECLARATION in gate.TOOL_DECLARATIONS
    assert PHONE_DEVICE_DECLARATION not in TOOL_DECLARATIONS
    assert classify("phone_device", {"action": "latest_frame"}) is Tier.ALLOW
    assert classify("phone_device", {"action": "capture_dom"}) is Tier.ALLOW
    assert classify("phone_device", {"action": "click"}) is Tier.ALLOW
    assert classify("phone_device", {"action": "send_sms"}) is Tier.ALLOW


def test_phone_tool_needs_scoped_device_even_when_id_is_supplied():
    gate, desktop, bridge, registry = _gate()

    async def scenario():
        out = await gate.dispatch("phone_device", {"action": "latest_frame"}, device_id="owner")
        assert not out.ok
        with gate.for_device("owner"):
            out = await gate.dispatch("phone_device", {"action": "latest_frame"}, device_id="other")
            assert not out.ok
            good = await gate.dispatch("phone_device", {"action": "latest_frame"})
            assert good.ok
            assert good.media == {"data": bridge.frames["owner"].jpeg, "mime_type": "image/jpeg"}

    asyncio.run(scenario())
    assert desktop.calls == []
    assert registry.pending_for("owner") == []


def test_uplink_cannot_insert_unapproved_dom_snapshot():
    bridge = MobileViewportBridge(object())

    class Socket:
        def __init__(self):
            self.sent = []

        async def send_json(self, payload):
            self.sent.append(payload)

    socket = Socket()
    asyncio.run(bridge._handle_text(socket, "owner", json.dumps({
        "jsonrpc": "2.0", "id": "x", "method": "viewport.dom",
        "params": {"snapshot": {"snapshotId": "forged"}},
    })))
    assert bridge._doms == {}
    assert socket.sent[0]["error"]["code"] == -32601


def test_capture_dom_invokes_locally_approved_bridge_command():
    gate, desktop, bridge, _ = _gate()

    async def scenario():
        with gate.for_device("owner"):
            return await gate.dispatch("phone_device", {"action": "capture_dom"})

    result = asyncio.run(scenario())
    assert result.ok
    assert json.loads(result.text) == bridge.snapshot
    assert bridge.calls == [("owner", "viewport.capture_dom", {})]
    assert desktop.calls == []


def test_interactive_action_waits_for_native_approval_without_stealing_focus():
    gate, desktop, bridge, registry = _gate()

    async def scenario():
        with gate.for_device("owner"):
            invalid = await gate.dispatch("phone_device", {"action": "send_sms", "number": "123"})
            assert not invalid.ok
            finished = await gate.dispatch("phone_device", {
                "action": "click", "snapshotId": "snapshot-1", "nodeId": "0/1",
            })
        assert finished.ok

    asyncio.run(scenario())
    assert bridge.calls == [("owner", "viewport.click", {
        "snapshotId": "snapshot-1", "nodeId": "0/1",
    })]
    assert desktop.calls == []
    assert registry.pending_for("owner") == []


def test_phone_actions_map_to_fixed_bridge_methods():
    gate, _, bridge, registry = _gate()
    cases = [
        ({"action": "scroll", "snapshotId": "snapshot-1", "nodeId": "0/1", "direction": "down"},
         "viewport.scroll", {"snapshotId": "snapshot-1", "nodeId": "0/1", "direction": "down"}),
        ({"action": "open_app", "package": "com.android.settings"},
         "device.open_app", {"package": "com.android.settings"}),
        ({"action": "make_call", "number": "+441234567890"},
         "device.make_call", {"number": "+441234567890"}),
        ({"action": "send_sms", "number": "+441234567890", "body": "Hello"},
         "device.send_sms", {"number": "+441234567890", "body": "Hello"}),
    ]

    async def scenario():
        for args, _, _ in cases:
            with gate.for_device("owner"):
                assert (await gate.dispatch("phone_device", args)).ok

    asyncio.run(scenario())
    assert bridge.calls == [("owner", method, params) for _, method, params in cases]
    assert registry.pending_for("owner") == []


def test_phone_target_and_sms_validation_matches_native_router():
    assert _valid_command_params("device.make_call", {"number": "+44 (123) 456-789"})
    assert _valid_command_params("device.make_call", {"number": "Alice Smith"})
    assert _valid_command_params("device.send_sms", {
        "number": "Alice Smith", "body": "A" * 180,
    })
    for number in ("+12", "123#", "123.456", "1234567890123456", "Alice 123"):
        assert not _valid_command_params("device.make_call", {"number": number})
    for body in ("", "   ", "A" * 1601):
        assert not _valid_command_params("device.send_sms", {
            "number": "+441234567890", "body": body,
        })
    assert _valid_command_params("device.send_sms", {
        "number": "+441234567890", "body": "😀" * 800,
    })
    assert not _valid_command_params("device.send_sms", {
        "number": "+441234567890", "body": "😀" * 801,
    })


def test_bridge_requires_connected_owner_and_correlated_authorized_result():
    class Auth:
        def verify_access(self, token):
            return "owner" if token == "owner-token" else None

    class Gateway:
        auth = Auth()

    class Socket:
        closed = False

        def __init__(self):
            self.requests = []

        async def send_json(self, request):
            self.requests.append(request)

    async def scenario():
        bridge = MobileViewportBridge(Gateway())
        socket = Socket()
        bridge._sessions["owner"] = socket
        bridge._session_tokens["owner"] = "owner-token"

        pending = asyncio.create_task(bridge.request_device_action(
            "owner", "device.make_call", {"number": "+441234567890"},
            timeout_s=2,
        ))
        await asyncio.sleep(0)
        request = socket.requests[0]
        assert request["requires_local_approval"] is True
        assert request["params"] == {"number": "+441234567890"}
        assert len(bridge._pending) == 1

        # A valid id without an affirmative local result cannot complete work.
        bridge._pending[("owner", request["id"])].set_result({
            "jsonrpc": "2.0", "id": request["id"], "result": {"ok": True},
        })
        try:
            await pending
        except RuntimeError as exc:
            assert "authorize" in str(exc)
        else:
            assert False, "an unapproved result must be rejected"
        assert bridge._pending == {}

        try:
            await bridge.request_device_action(
                "other", "device.make_call", {"number": "+441234567890"},
            )
        except ConnectionError:
            pass
        else:
            assert False, "another device must not inherit the owner's uplink"

    asyncio.run(scenario())
