"""Authentication-critical saved fields require an unambiguous known shape."""
from __future__ import annotations

import json

import pytest

from orion_core.remote import RemoteAuthManager


@pytest.fixture
def paired(tmp_path):
    auth = RemoteAuthManager(tmp_path)
    device, refresh = auth.complete_pairing(auth.begin_pairing(), "synthetic device")
    token, _ = auth.issue_access(device, refresh)
    return tmp_path, device, refresh, token


@pytest.mark.parametrize("field,value", [
    ("revoked", None), ("revoked", 0), ("revoked", 1),
    ("revoked", []), ("revoked", {}), ("revoked", "false"),
    ("refresh_sha256", None), ("refresh_sha256", 123),
    ("refresh_sha256", "\u00e9" * 64), ("refresh_sha256", "0" * 63),
    ("refresh_sha256", "z" * 64),
])
def test_ambiguous_authentication_fields_hold_registry_without_rewriting(paired, field, value):
    root, device, _refresh, _token = paired
    registry = root / "remote_devices.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    data[device][field] = value
    original = json.dumps(data, ensure_ascii=False).encode("utf-8")
    registry.write_bytes(original)
    key = (root / "remote_secret.key").read_bytes()
    with pytest.raises(ValueError, match="Remote device registry"):
        RemoteAuthManager(root)
    assert registry.read_bytes() == original
    assert (root / "remote_secret.key").read_bytes() == key


@pytest.mark.parametrize("field", ["revoked", "refresh_sha256"])
def test_missing_authentication_field_cannot_default_to_active(paired, field):
    root, device, _refresh, _token = paired
    registry = root / "remote_devices.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    del data[device][field]
    original = json.dumps(data).encode("utf-8")
    registry.write_bytes(original)
    with pytest.raises(ValueError, match="Remote device registry"):
        RemoteAuthManager(root)
    assert registry.read_bytes() == original


def test_duplicate_revocation_keys_are_not_silently_resolved_to_active(paired):
    root, device, refresh, _token = paired
    registry = root / "remote_devices.json"
    original = ('{"' + device + '":{"refresh_sha256":"' + RemoteAuthManager._hash(refresh)
                + '","revoked":true,"revoked":false}}').encode("utf-8")
    registry.write_bytes(original)
    with pytest.raises(ValueError, match="Remote device registry"):
        RemoteAuthManager(root)
    assert registry.read_bytes() == original


@pytest.mark.parametrize("contents", [b'{"retained":null}', b'{"version":999,"devices":{}}'])
def test_invalid_or_unknown_shape_is_preserved_without_generating_a_new_key(tmp_path, contents):
    registry = tmp_path / "remote_devices.json"
    registry.write_bytes(contents)
    with pytest.raises(ValueError, match="Remote device registry"):
        RemoteAuthManager(tmp_path)
    assert registry.read_bytes() == contents
    assert not (tmp_path / "remote_secret.key").exists()


def test_valid_legacy_flat_records_and_extra_metadata_keep_their_identity(paired):
    root, device, refresh, token = paired
    registry = root / "remote_devices.json"
    data = json.loads(registry.read_text(encoding="utf-8"))
    data[device]["additional_metadata"] = {"version": 999, "note": "synthetic retained"}
    original = json.dumps(data).encode("utf-8")
    registry.write_bytes(original)
    auth = RemoteAuthManager(root)
    assert auth.issue_access(device, refresh) is not None
    assert auth.verify_access(token) == device
    assert registry.read_bytes() == original
    assert auth.revoke_device(device)
    saved = json.loads(registry.read_text(encoding="utf-8"))
    assert saved[device]["additional_metadata"] == data[device]["additional_metadata"]
    assert RemoteAuthManager(root).issue_access(device, refresh) is None


def test_empty_legacy_registry_remains_a_supported_first_run(tmp_path):
    registry = tmp_path / "remote_devices.json"
    registry.write_text("{}", encoding="utf-8")
    auth = RemoteAuthManager(tmp_path)
    assert not auth.has_devices()
    assert auth.complete_pairing(auth.begin_pairing(), "synthetic")
