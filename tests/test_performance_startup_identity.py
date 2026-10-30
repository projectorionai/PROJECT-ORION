"""Startup identity I/O preserves existing preferences and skips no-op writes."""
import json
from types import SimpleNamespace

import pytest

from orion_core import identity


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(identity, "IDENTITY_PATH", tmp_path / "identity.json")
    lines, beats = [], []
    bus = SimpleNamespace(log=SimpleNamespace(emit=lines.append))
    telemetry = SimpleNamespace(health=SimpleNamespace(register=lambda *_: None,
        beat=lambda *args: beats.append(args)))
    return SimpleNamespace(path=identity.IDENTITY_PATH, bus=bus, telemetry=telemetry,
                           lines=lines, beats=beats)


@pytest.mark.parametrize("raw", [b'{"user_name":', b'[]', b'null', b'\xff\xfe'])
def test_unreadable_retained_identity_is_not_replaced_at_startup_or_later(setup, raw):
    setup.path.write_bytes(raw)
    manager = identity.IdentityManager(setup.bus, setup.telemetry)
    assert setup.path.read_bytes() == raw
    assert manager.storage_error and any("repair" in line.lower() for line in setup.lines)
    assert not manager.set_user_name("Synthetic name")
    assert manager.user_name() == "" and setup.path.read_bytes() == raw
    assert any(beat[1] == "DEGRADED" for beat in setup.beats)


def test_temporary_read_denial_never_replaces_retained_identity(setup, monkeypatch):
    saved = b'{"user_name":"Retained person","unknown":{"preserve":true}}'
    setup.path.write_bytes(saved)
    original = type(setup.path).read_text

    def denied(path, *args, **kwargs):
        if path == setup.path:
            raise PermissionError("synthetic private error detail")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(setup.path), "read_text", denied)
    manager = identity.IdentityManager(setup.bus, setup.telemetry)
    assert setup.path.read_bytes() == saved
    assert not manager.set_user_name("Unwanted replacement")
    assert setup.path.read_bytes() == saved
    assert "synthetic private error detail" not in " ".join(setup.lines)


def test_unchanged_preferences_do_not_perform_a_startup_write(setup, monkeypatch):
    first = identity.IdentityManager(setup.bus)
    original = setup.path.read_bytes()
    writes = []
    monkeypatch.setattr(identity, "atomic_write_text", lambda *_args, **_kwargs: writes.append(True))
    second = identity.IdentityManager(setup.bus)
    assert second.preferences == first.preferences and writes == []
    assert setup.path.read_bytes() == original


def test_missing_defaults_are_migrated_once_with_unknown_fields_retained(setup, monkeypatch):
    setup.path.write_text(json.dumps({"user_name": "Retained", "unknown": {"preserve": True},
                                      "terminology": {"custom": "Exact value"}}))
    first = identity.IdentityManager(setup.bus)
    saved = json.loads(setup.path.read_text())
    assert saved["unknown"] == {"preserve": True} and saved["user_name"] == "Retained"
    assert saved["terminology"]["custom"] == "Exact value"
    assert "honorific_frequency" in saved
    writes = []
    monkeypatch.setattr(identity, "atomic_write_text", lambda *_args, **_kwargs: writes.append(True))
    assert identity.IdentityManager(setup.bus).preferences == first.preferences and not writes


def test_failed_name_persistence_does_not_claim_a_changed_name(setup, monkeypatch):
    manager = identity.IdentityManager(setup.bus)
    assert manager.set_user_name("Original name")
    before = setup.path.read_bytes()

    def denied(*_args, **_kwargs):
        raise PermissionError("synthetic write denial")

    monkeypatch.setattr(identity, "atomic_write_text", denied)
    assert not manager.set_user_name("Unwritten name")
    assert manager.user_name() == "Original name" and setup.path.read_bytes() == before
