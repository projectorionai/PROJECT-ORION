"""Saved provider policy must not become enabled defaults or lose retained data."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import providers as p


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(p, "API_CONFIG_PATH", tmp_path / "api_keys.json")
    monkeypatch.setattr(p, "_default_provider_payload", lambda *_: {
        "providers": {"synthetic": {"kind": "openai_compatible", "model": "safe-model",
            "api_key": "synthetic-secret", "enabled": True}},
        "provider_order": ["synthetic"], "active_provider": "synthetic"})
    return p.API_CONFIG_PATH


@pytest.mark.parametrize("raw", [b'{"providers":', b'[]', b'null',
    b'{"schema":"orion.future.v999","providers":{}}',
    b'{"providers":{},"providers":{"synthetic":{"enabled":true}}}',
    b'{"providers":{"synthetic":{"enabled":"false","model":"m"}}}',
    b'\xff\xfe'])
def test_invalid_saved_policy_is_preserved_and_cannot_be_saved_as_defaults(config, raw):
    config.write_bytes(raw)
    settings = p.read_provider_settings()
    assert getattr(settings, "configuration_error", "")
    assert not settings.providers and not settings.integrations
    with pytest.raises(RuntimeError, match="[Rr]estore or repair"):
        p.write_provider_settings(settings)
    assert config.read_bytes() == raw


def test_unreadable_saved_policy_does_not_enable_environment_defaults(config, monkeypatch):
    config.write_text('{"providers":{}}', encoding="utf-8")
    original = Path.open

    def denied(path, *args, **kwargs):
        if path == config:
            raise PermissionError("synthetic-private-secret")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as blocked:
        blocked.setattr(Path, "open", denied)
        settings = p.read_provider_settings()
    assert not settings.providers
    assert "Restore or repair" in settings.configuration_error
    assert "private-secret" not in settings.configuration_error


def test_desktop_damaged_policy_keeps_degraded_startup_without_save_or_dialog(config):
    from orion_core.app import _configured_provider_settings
    config.write_bytes(b'{"providers":')
    settings = _configured_provider_settings()
    assert settings is not None and settings.configuration_error and not settings.providers
    assert config.read_bytes() == b'{"providers":'


@pytest.mark.parametrize("change", ["edit", "delete"])
def test_stale_settings_cannot_overwrite_changed_or_deleted_policy(config, change):
    config.write_text(json.dumps({"providers": {}}), encoding="utf-8")
    settings = p.read_provider_settings()
    if change == "delete":
        config.unlink()
    else:
        config.write_text(json.dumps({"providers": {}, "private_marker": "external-edit"}), encoding="utf-8")
    before = config.read_bytes() if config.exists() else None
    with pytest.raises(RuntimeError, match="changed"):
        p.write_provider_settings(settings)
    assert (config.read_bytes() if config.exists() else None) == before


def test_valid_save_retains_unknown_top_profile_and_integration_fields(config):
    saved = {"schema": "orion.mark_viii.providers.v1", "private_extension": {"keep": "exact"},
        "providers": {"synthetic": {"model": "saved", "custom_nested": {"keep": [1, 2]}}},
        "integrations": {"private_service": {"future_field": "retain"}}}
    config.write_text(json.dumps(saved), encoding="utf-8")
    settings = p.read_provider_settings()
    settings.providers["synthetic"].model = "new-model"
    p.write_provider_settings(settings)
    restored = json.loads(config.read_text())
    assert restored["private_extension"] == saved["private_extension"]
    assert restored["providers"]["synthetic"]["custom_nested"] == {"keep": [1, 2]}
    assert restored["integrations"]["private_service"] == {"future_field": "retain"}
    assert restored["providers"]["synthetic"]["model"] == "new-model"
    # An object's own successful save refreshes its observed version.
    settings.providers["synthetic"].model = "next-model"
    p.write_provider_settings(settings)


def test_model_heal_does_not_rewrite_duplicate_saved_policy(config):
    raw = b'{"providers":{},"providers":{"synthetic":{"model":"old"}}}'
    config.write_bytes(raw)
    router = object.__new__(p.ProviderRouter)
    router._persist_model(SimpleNamespace(name="synthetic", model="replacement"))
    assert config.read_bytes() == raw


def test_failed_atomic_save_does_not_refresh_observed_version(config, monkeypatch):
    config.write_text('{"providers":{}}', encoding="utf-8")
    settings = p.read_provider_settings()
    before = config.read_bytes()
    actual = p.atomic_write_text
    monkeypatch.setattr(p, "atomic_write_text", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("synthetic")))
    with pytest.raises(PermissionError):
        p.write_provider_settings(settings)
    assert config.read_bytes() == before
    monkeypatch.setattr(p, "atomic_write_text", actual)
    p.write_provider_settings(settings)


def test_first_run_and_legacy_key_migration_remain_available(config):
    settings = p.read_provider_settings()
    assert settings.providers["synthetic"].enabled
    p.write_provider_settings(settings)
    assert config.is_file()
    config.write_text('{"gemini_api_key":"legacy-synthetic","unrelated":"keep"}', encoding="utf-8")
    migrated = p.read_provider_settings()
    assert not getattr(migrated, "configuration_error", "")
    p.write_provider_settings(migrated)
    assert json.loads(config.read_text())["unrelated"] == "keep"


def test_new_settings_or_legacy_key_writer_cannot_replace_corrupt_saved_bytes(config):
    raw = b'{"providers":'
    config.write_bytes(raw)
    fresh = p._settings_from_payload(p._default_provider_payload())
    for write in (lambda: p.write_provider_settings(fresh), lambda: p.write_api_key("new-synthetic")):
        with pytest.raises(p.ProviderConfigurationError, match="Restore or repair"):
            write()
        assert config.read_bytes() == raw


def test_oversized_policy_read_is_bounded_and_preserves_bytes(config, monkeypatch):
    monkeypatch.setattr(p, "_PROVIDER_CONFIG_MAX_BYTES", 64)
    raw = json.dumps({"providers": {}, "retained": "x" * 100}).encode()
    config.write_bytes(raw)
    settings = p.read_provider_settings()
    assert settings.configuration_error and not settings.providers
    assert config.read_bytes() == raw


def test_two_readers_cannot_silently_overwrite_each_others_saves(config):
    config.write_text('{"providers":{}}', encoding="utf-8")
    readers = [p.read_provider_settings(), p.read_provider_settings()]
    readers[0].providers["synthetic"].model = "writer-a"
    readers[1].providers["synthetic"].model = "writer-b"

    def save(settings):
        try:
            p.write_provider_settings(settings)
            return "saved"
        except p.ProviderConfigurationError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, readers)) == ["conflict", "saved"]
    assert json.loads(config.read_text())["providers"]["synthetic"]["model"] in {"writer-a", "writer-b"}


def test_model_heal_respects_external_model_change_and_retains_extensions(config):
    config.write_text(json.dumps({"providers": {"synthetic": {"model": "external-choice",
        "custom": {"retained": True}}}}), encoding="utf-8")
    before = config.read_bytes()
    router = object.__new__(p.ProviderRouter)
    profile = SimpleNamespace(name="synthetic", model="healed")
    assert router._persist_model(profile, expected_model="old") is False
    assert config.read_bytes() == before
    assert router._persist_model(profile, expected_model="external-choice") is True
    assert json.loads(config.read_text())["providers"]["synthetic"]["custom"] == {"retained": True}


def test_model_heal_refreshes_only_its_matching_settings_snapshot(config):
    config.write_text('{"providers":{"synthetic":{"model":"old"}}}', encoding="utf-8")
    settings = p.read_provider_settings()
    router = object.__new__(p.ProviderRouter)
    router.settings = settings
    settings.providers["synthetic"].model = "healed"
    assert router._persist_model(settings.providers["synthetic"], expected_model="old") is True
    p.write_provider_settings(settings)
    assert json.loads(config.read_text())["providers"]["synthetic"]["model"] == "healed"


def test_unchanged_valid_save_keeps_exact_disk_bytes(config):
    p.write_provider_settings(p._settings_from_payload(p._default_provider_payload()))
    settings = p.read_provider_settings()
    before = config.read_bytes()
    p.write_provider_settings(settings)
    assert config.read_bytes() == before


def test_router_reports_configuration_repair_without_private_data(config):
    config.write_bytes(b'{"synthetic-private-key":')
    settings = p.read_provider_settings()
    logs = []
    router = p.ProviderRouter(settings, SimpleNamespace(log=SimpleNamespace(emit=logs.append)), object())
    assert any("Restore or repair" in line for line in logs)
    assert all("private-key" not in line for line in logs)
    assert router.settings.providers == {}


def test_adding_live_key_preserves_custom_providers_and_disabled_shared_profiles(config, monkeypatch):
    defaults = p._default_provider_payload

    def with_live(key=""):
        payload = defaults()
        payload["providers"].update({
            "gemini": {"kind": "gemini_live", "model": "live", "api_key": key, "enabled": bool(key)},
            "gemini_text": {"kind": "openai_compatible", "model": "text", "api_key": key, "enabled": bool(key)}})
        return payload

    monkeypatch.setattr(p, "_default_provider_payload", with_live)
    saved = {"providers": {
        "gemini": {"kind": "gemini_live", "model": "live", "api_key": "old-key", "enabled": False},
        "gemini_text": {"kind": "openai_compatible", "model": "chosen-text", "api_key": "old-key", "enabled": False},
        "custom": {"kind": "openai_compatible", "model": "chosen", "api_key": "custom-key", "enabled": False}},
        "integrations": {"retained": {"private": "keep"}}, "extension": "keep"}
    config.write_text(json.dumps(saved), encoding="utf-8")
    from orion_core.app import _settings_from_key
    settings = _settings_from_key("new-live-key")
    assert settings.providers["gemini"].api_key == "new-live-key" and settings.providers["gemini"].enabled
    assert settings.providers["gemini_text"].api_key == "new-live-key"
    assert settings.providers["gemini_text"].model == "chosen-text" and not settings.providers["gemini_text"].enabled
    assert settings.providers["custom"].api_key == "custom-key" and not settings.providers["custom"].enabled
    assert json.loads(config.read_text())["extension"] == "keep"
    assert settings.integrations["retained"] == {"private": "keep"}


@pytest.mark.parametrize("timeout", ["nan", "inf", "-inf"])
def test_nonfinite_numeric_policy_is_not_admitted(config, timeout):
    raw = json.dumps({"providers": {"synthetic": {"model": "m", "timeout_s": timeout}}}).encode()
    config.write_bytes(raw)
    settings = p.read_provider_settings()
    assert settings.configuration_error and not settings.providers
    assert config.read_bytes() == raw
