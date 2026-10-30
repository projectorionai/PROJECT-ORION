"""Synthetic secrets only: vault failures must not silently weaken storage."""
import json

import pytest

from orion_core import plugin_vault as module


@pytest.mark.parametrize("data", [[], {"plugins": []}, {"plugins": {"demo": []}},
                                 {"plugins": {"demo": {"KEY": {"bad": "value"}}}}])
def test_invalid_vault_structure_uses_the_domain_failure(tmp_path, data):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(module.VaultError):
        module.PluginVault(path)
    assert json.loads(path.read_text(encoding="utf-8")) == data


def test_crypto_failure_cannot_silently_save_plaintext(tmp_path, monkeypatch):
    vault = module.PluginVault(tmp_path / "vault.json")
    monkeypatch.setattr(vault, "_choose_method", lambda: "fernet")

    def unavailable(_):
        raise OSError("synthetic key storage lock")

    monkeypatch.setattr(module, "_fernet", unavailable)
    with pytest.raises(module.VaultError):
        vault.set_secret("demo", "KEY", "synthetic-sensitive-value")
    assert vault.secrets_for("demo", ["KEY"]) == {}
    assert not vault.path.exists()


def test_bad_reload_preserves_the_previous_usable_vault(tmp_path):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(module.VaultError):
        vault.reload()
    assert vault.secrets_for("demo", ["KEY"]) == {"KEY": "synthetic-value"}
    assert path.read_text(encoding="utf-8") == "{bad"


def test_a_plaintext_import_is_sealed_on_its_next_save(tmp_path, monkeypatch):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    monkeypatch.setattr(vault, "_choose_method", lambda: "dpapi")
    monkeypatch.setattr(module, "_dpapi_seal", lambda _: "dpapi:synthetic-ciphertext")
    vault.save()
    assert "synthetic-value" not in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["plugins"]["demo"]["KEY"].startswith("dpapi:")


def test_decryption_does_not_generate_a_replacement_for_a_missing_key(tmp_path):
    pytest.importorskip("cryptography")
    path, key = tmp_path / "vault.json", tmp_path / "missing.key"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "fernet:synthetic-token"}}}), encoding="utf-8")
    vault = module.PluginVault(path, key)
    with pytest.raises(module.VaultError):
        vault.secrets_for("demo", ["KEY"])
    assert not key.exists()


def test_failed_reload_cannot_be_overwritten_by_a_later_save(tmp_path):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(module.VaultError):
        vault.reload()
    with pytest.raises(module.VaultError):
        vault.save()
    assert path.read_text(encoding="utf-8") == "{bad"


def test_vault_status_explains_plaintext_until_successful_upgrade(tmp_path, monkeypatch):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    assert not vault.status().encrypted and vault.status().plaintext_entries == 1
    monkeypatch.setattr(vault, "_choose_method", lambda: "dpapi")
    monkeypatch.setattr(module, "_dpapi_seal", lambda _: "dpapi:synthetic-ciphertext")
    vault.save()
    assert vault.status().encrypted and vault.status().plaintext_entries == 0


def test_vault_save_failure_preserves_file_and_does_not_publish_candidate(tmp_path, monkeypatch):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    before = path.read_bytes()
    monkeypatch.setattr(vault, "_choose_method", lambda: "dpapi")
    monkeypatch.setattr(module, "_dpapi_seal", lambda _: "dpapi:synthetic-ciphertext")
    def locked(*_, **__):
        raise PermissionError("synthetic save lock")
    monkeypatch.setattr(module, "atomic_write_text", locked)
    with pytest.raises(module.VaultError):
        vault.save()
    assert path.read_bytes() == before
    assert vault.secrets_for("demo", ["KEY"]) == {"KEY": "synthetic-value"}
    assert not vault.status().encrypted


def test_a_repaired_vault_can_reload_then_save_normally(tmp_path):
    path = tmp_path / "vault.json"
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-value"}}}), encoding="utf-8")
    vault = module.PluginVault(path)
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(module.VaultError):
        vault.reload()
    path.write_text(json.dumps({"plugins": {"demo": {"KEY": "synthetic-repaired"}}}), encoding="utf-8")
    vault.reload()
    vault.save()
    assert vault.secrets_for("demo", ["KEY"]) == {"KEY": "synthetic-repaired"}
