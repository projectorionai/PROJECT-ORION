"""Regression checks for durable device credentials and live uplink access."""

import asyncio
import time
from types import SimpleNamespace

import pytest

import orion_core.remote as remote
import orion_core.remote_endpoints as endpoints
from orion_core.remote import RemoteAuthManager, RemoteGateway
from orion_core.remote_mobile_ws import MobileViewportBridge, ViewportDom, ViewportFrame


def _paired(tmp_path):
    auth = RemoteAuthManager(tmp_path)
    device_id, refresh = auth.complete_pairing(auth.begin_pairing())
    token, _ = auth.issue_access(device_id, refresh)
    return auth, device_id, refresh, token


def test_signing_key_write_failure_prevents_ephemeral_auth(tmp_path, monkeypatch):
    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(remote, "atomic_write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        RemoteAuthManager(tmp_path)
    assert not (tmp_path / "remote_secret.key").exists()


def test_invalid_existing_signing_key_is_not_replaced(tmp_path):
    key_path = tmp_path / "remote_secret.key"
    key_path.write_text("corrupt-key", encoding="utf-8")
    with pytest.raises(ValueError, match="signing key is invalid"):
        RemoteAuthManager(tmp_path)
    assert key_path.read_text(encoding="utf-8") == "corrupt-key"


def test_failed_pairing_registry_write_never_issues_credentials(tmp_path, monkeypatch):
    auth = RemoteAuthManager(tmp_path)
    code = auth.begin_pairing()

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(remote, "atomic_write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        auth.complete_pairing(code)
    assert not auth.has_devices()
    assert not RemoteAuthManager(tmp_path).has_devices()
    assert auth.complete_pairing(code) is None


def test_failed_revocation_registry_write_is_reported_and_blocks_live_tokens(tmp_path, monkeypatch):
    auth, device_id, refresh, token = _paired(tmp_path)

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(remote, "atomic_write_text", fail_write)
    with pytest.raises(OSError, match="disk full"):
        auth.revoke_device(device_id)
    assert auth.verify_access(token) is None
    assert auth.issue_access(device_id, refresh) is None


def test_revoked_uplink_cannot_serve_cached_screen_or_complete_action(tmp_path):
    async def scenario():
        auth, device_id, _refresh, token = _paired(tmp_path)
        bridge = MobileViewportBridge(SimpleNamespace(auth=auth))

        class Socket:
            closed = False

            async def send_json(self, _message):
                sent.set()

        sent = asyncio.Event()
        socket = Socket()
        bridge._sessions[device_id] = socket
        bridge._session_tokens[device_id] = token
        bridge._frames[device_id] = ViewportFrame(b"private frame", time.monotonic())
        bridge._doms[device_id] = ViewportDom({"snapshotId": "private"}, time.monotonic())
        assert bridge.latest_frame(device_id) is not None
        assert bridge.latest_dom(device_id) is not None

        action = asyncio.create_task(bridge.request_device_action(device_id, "viewport.capture_dom", {}))
        await asyncio.wait_for(sent.wait(), timeout=1)
        assert auth.revoke_device(device_id)
        assert bridge.latest_frame(device_id) is None
        assert bridge.latest_dom(device_id) is None
        ((pending_device, _request_id), future), = bridge._pending.items()
        assert pending_device == device_id
        future.set_result({"result": {"authorized": True, "ok": True,
                                      "snapshot": {"snapshotId": "after-revoke"}}})
        with pytest.raises(ConnectionError, match="no longer authorised"):
            await action
        assert bridge._doms[device_id].snapshot["snapshotId"] == "private"

    asyncio.run(scenario())


def test_lan_discovery_excludes_tailscale_and_non_lan_addresses(monkeypatch):
    class Probe:
        def connect(self, _target):
            pass

        def getsockname(self):
            return ("100.101.102.103", 40000)

        def close(self):
            pass

    monkeypatch.setattr(endpoints.socket, "socket", lambda *_args: Probe())
    monkeypatch.setattr(endpoints.socket, "gethostname", lambda: "orion-host")
    monkeypatch.setattr(endpoints.socket, "getaddrinfo", lambda *_args: [
        (None, None, None, None, ("100.101.102.103", 0)),
        (None, None, None, None, ("203.0.113.10", 0)),
        (None, None, None, None, ("192.168.1.20", 0)),
    ])
    assert endpoints.lan_ipv4_addresses() == ["192.168.1.20"]

    discovered = endpoints.detect_endpoints(
        8765, tailscale={"dns": "", "ips": ["100.101.102.103"]},
        lan=["100.101.102.103", "203.0.113.10", "192.168.1.20"])
    assert [(item["kind"], item["host"]) for item in discovered] == [
        ("tailscale-ip", "100.101.102.103"), ("lan", "192.168.1.20")]


def test_lan_pairing_hint_never_uses_tailscale_address(monkeypatch):
    monkeypatch.setattr(remote, "lan_ipv4_addresses", lambda: ["192.168.1.20"])
    assert RemoteGateway._lan_ip() == "192.168.1.20"
    monkeypatch.setattr(remote, "lan_ipv4_addresses", lambda: [])
    assert RemoteGateway._lan_ip() == "127.0.0.1"
