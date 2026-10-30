"""Mark XXXX.3: ORION's keys are sealed to the owner's Windows account."""

from __future__ import annotations

import json

import pytest

from orion_core import key_seal

pytestmark = pytest.mark.skipif(not key_seal.available(), reason="DPAPI is Windows-only")


def test_a_value_round_trips_and_is_not_readable_on_disk():
    sealed = key_seal.seal("gsk_abcdefghijklmnop")
    assert sealed.startswith(key_seal.PREFIX)
    assert "gsk_" not in sealed
    key_seal._cache.clear()
    assert key_seal.open_value(sealed) == "gsk_abcdefghijklmnop"


def test_placeholders_and_sealed_values_are_left_alone():
    assert key_seal.seal("") == ""
    assert key_seal.seal("local") == "local"
    once = key_seal.seal("abc123secret")
    assert key_seal.seal(once) == once


def test_a_value_sealed_elsewhere_reads_as_empty():
    assert key_seal.open_value("dpapi:AAAAAAAA") == ""
    assert key_seal.unopenable({"providers": {"x": {"api_key": "dpapi:AAAA"}}}) == 1


def test_only_credential_fields_are_sealed():
    data = {"providers": {"groq": {"api_key": "gsk_x", "api_keys": ["a1", "b2"],
                                   "model": "llama", "strengths": ["fast"]}},
            "integrations": {"notion": {"token": "ntn_x", "database": "db1"}}}
    sealed = key_seal.seal_tree(data)
    groq = sealed["providers"]["groq"]
    assert key_seal.is_sealed(groq["api_key"]) and all(map(key_seal.is_sealed, groq["api_keys"]))
    assert groq["model"] == "llama" and groq["strengths"] == ["fast"]
    assert sealed["integrations"]["notion"]["database"] == "db1"
    assert key_seal.open_tree(sealed) == data
    assert key_seal.needs_sealing(data) and not key_seal.needs_sealing(sealed)


def test_a_plain_key_file_is_sealed_in_place(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"providers": {"g": {"api_key": "AIzaPLAIN"}}}), encoding="utf-8")
    assert key_seal.seal_file_in_place(path) is True
    assert "AIzaPLAIN" not in path.read_text(encoding="utf-8")
    assert key_seal.load_json(path)["providers"]["g"]["api_key"] == "AIzaPLAIN"
    assert key_seal.seal_file_in_place(path) is False


def test_a_file_with_duplicate_fields_is_not_rewritten(tmp_path):
    path = tmp_path / "api_keys.json"
    text = '{"providers": {"g": {"api_key": "one", "api_key": "two"}}}'
    path.write_text(text, encoding="utf-8")
    assert key_seal.seal_file_in_place(path) is False
    assert path.read_text(encoding="utf-8") == text


def test_provider_settings_seal_a_pasted_key_and_still_read_it(tmp_path, monkeypatch):
    from orion_core import providers as mod

    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps({"providers": {"groq": {
        "kind": "openai_compatible", "api_key": "gsk_pasted_in_plain", "model": "m",
        "enabled": True}}}), encoding="utf-8")
    monkeypatch.setattr(mod, "API_CONFIG_PATH", path)
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    settings = mod.read_provider_settings()
    assert "gsk_pasted_in_plain" not in path.read_text(encoding="utf-8")
    assert settings.providers["groq"].api_key == "gsk_pasted_in_plain"


def test_the_phone_signing_key_is_sealed(tmp_path):
    from orion_core.remote import RemoteAuthManager as RemoteAuth

    (tmp_path / "remote_secret.key").write_text("ab" * 32, encoding="utf-8")
    auth = RemoteAuth(tmp_path)
    assert auth._secret == bytes.fromhex("ab" * 32)
    on_disk = (tmp_path / "remote_secret.key").read_text(encoding="utf-8")
    assert key_seal.is_sealed(on_disk)
    assert RemoteAuth(tmp_path)._secret == bytes.fromhex("ab" * 32)


def test_a_key_sealed_by_another_account_survives_a_save(tmp_path):
    """Mark XXXX.4: it opened as "" and the next save wrote "" over it."""
    path = tmp_path / "api_keys.json"
    foreign = "dpapi:AAAAAAAA"
    path.write_text(json.dumps({"providers": {"a": {"api_key": foreign},
                                              "b": {"api_key": "plain-b"}}}), encoding="utf-8")
    data = key_seal.load_json(path)
    assert data["providers"]["a"]["api_key"] == ""
    data["providers"]["b"]["model"] = "new"
    key_seal.save_json(path, data)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["providers"]["a"]["api_key"] == foreign
    assert key_seal.load_json(path)["providers"]["b"] == {"api_key": "plain-b", "model": "new"}


def test_more_credential_names_are_sealed():
    sealed = key_seal.seal_tree({"email": {"app_password": "pw123456"}, "ifttt": {"key": "k123456"}})
    assert key_seal.is_sealed(sealed["email"]["app_password"]) and key_seal.is_sealed(sealed["ifttt"]["key"])
