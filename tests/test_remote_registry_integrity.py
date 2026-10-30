"""Uncertain pairing state must be retained, never replaced as a new install."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import remote


def test_saved_devices_and_revocation_survive_reloading(tmp_path):
    auth = remote.RemoteAuthManager(tmp_path)
    active, active_refresh = auth.complete_pairing(auth.begin_pairing(), "synthetic active")
    revoked, revoked_refresh = auth.complete_pairing(auth.begin_pairing(), "synthetic revoked")
    assert auth.revoke_device(revoked)
    restarted = remote.RemoteAuthManager(tmp_path)
    assert restarted.issue_access(active, active_refresh) is not None
    assert restarted.issue_access(revoked, revoked_refresh) is None
    assert len(restarted.list_devices()) == 2


@pytest.mark.parametrize("contents", [
    b'{"retained_device":', b"[]", b"null", b'{"retained_device":null}', b"\xff\xfe",
])
def test_corrupt_pairing_registry_stays_unchanged_and_auth_is_unavailable(tmp_path, contents):
    remote.RemoteAuthManager(tmp_path)
    registry = tmp_path / "remote_devices.json"
    registry.write_bytes(contents)
    key_before = (tmp_path / "remote_secret.key").read_bytes()
    with pytest.raises(ValueError, match="Remote device registry"):
        remote.RemoteAuthManager(tmp_path)
    assert registry.read_bytes() == contents
    assert (tmp_path / "remote_secret.key").read_bytes() == key_before


def test_temporarily_unreadable_registry_does_not_discard_the_saved_devices(tmp_path, monkeypatch):
    auth = remote.RemoteAuthManager(tmp_path)
    auth.complete_pairing(auth.begin_pairing(), "synthetic")
    registry = tmp_path / "remote_devices.json"
    original = registry.read_bytes()
    read_text = Path.read_text

    def denied(path, *args, **kwargs):
        if path == registry:
            raise PermissionError("synthetic private diagnostic contents")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "read_text", denied)
        with pytest.raises(ValueError, match="Restore or repair") as caught:
            remote.RemoteAuthManager(tmp_path)
    assert "private diagnostic" not in str(caught.value)
    assert registry.read_bytes() == original
    assert remote.RemoteAuthManager(tmp_path).has_devices()


def test_gateway_with_invalid_pairing_state_starts_no_network_or_tasks(tmp_path, monkeypatch):
    remote.RemoteAuthManager(tmp_path)
    registry = tmp_path / "remote_devices.json"
    registry.write_text(json.dumps({"retained": "invalid-device-record"}), encoding="utf-8")
    logs = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append))
    calls = []
    monkeypatch.setattr(remote.asyncio, "create_task", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(ValueError, match="Remote device registry"):
        remote.RemoteGateway(None, None, bus, config_dir=tmp_path)
    assert calls == []
    assert logs == []


def test_optional_uplink_setup_failure_keeps_desktop_startup_available_without_logging_secrets():
    from orion_core.app import _create_optional_remote_gateway

    logs = []
    bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append))

    def broken(**_kwargs):
        raise ValueError("synthetic-private-credential")

    gateway = _create_optional_remote_gateway(broken, bus, router=None, memory=None)
    assert gateway is None
    assert len(logs) == 1 and "ValueError" in logs[0]
    assert "remote_devices.json" in logs[0]
    assert "Desktop startup will continue" in logs[0]
    assert "private-credential" not in logs[0]


def test_optional_uplink_setup_returns_the_constructed_gateway():
    from orion_core.app import _create_optional_remote_gateway

    value = object()
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    observed = []

    def construct(**kwargs):
        observed.append(kwargs)
        return value

    assert _create_optional_remote_gateway(construct, bus, router="stub") is value
    assert observed == [{"bus": bus, "router": "stub"}]
