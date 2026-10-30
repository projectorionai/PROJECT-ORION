"""Keep Gemini text quotas separate without overriding custom model lists."""

from orion_core.providers import (
    ProviderRouter, _default_provider_payload, _merge_provider_payload,
    _settings_from_payload,
)
from orion_core.provider_capabilities import Capability, provider_capabilities
from orion_core.model_registry import context_window
from types import SimpleNamespace


def test_shipped_gemini_text_models_include_verified_stream_fallback():
    profiles = _default_provider_payload()["providers"]
    routine = profiles["gemini_text"]["models"]
    demanding = profiles["gemini_reasoning"]["models"]
    assert routine[0] == "gemini-3.5-flash-lite"
    assert demanding[0] == "gemini-3.8-flash"
    assert "gemini-3.1-flash-lite" in routine
    assert all("tts" not in model and "live" not in model and "image" not in model
               for model in (*routine, *demanding))
    assert "fast" in profiles["gemini_text"]["strengths"]
    assert "coding" in profiles["gemini_reasoning"]["strengths"]


def test_previous_shipped_cascade_migrates_to_current_routing():
    old_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
    merged = _merge_provider_payload(_default_provider_payload(), {
        "providers": {"gemini_text": {"model": "gemini-2.5-flash", "models": old_models}}
    })
    assert merged["providers"]["gemini_text"]["model"] == "gemini-3.5-flash-lite"
    assert merged["providers"]["gemini_text"]["models"][0] == "gemini-3.5-flash-lite"


def test_custom_gemini_text_cascade_is_preserved():
    custom = ["gemini-2.5-pro", "gemini-2.5-flash"]
    merged = _merge_provider_payload(_default_provider_payload(), {
        "providers": {"gemini_text": {"model": custom[0], "models": custom}}
    })
    assert merged["providers"]["gemini_text"]["models"] == custom
    assert merged["providers"]["gemini_text"]["model"] == custom[0]
    assert not merged["providers"]["gemini_reasoning"]["enabled"]


def test_new_reasoning_profile_inherits_key_and_respects_disabled_text():
    defaults = _default_provider_payload()
    with_key = _merge_provider_payload(defaults, {"providers": {
        "gemini": {"api_key": "test-key", "enabled": True},
        "gemini_text": {"enabled": True, "model": "gemini-2.5-flash",
                        "models": list(("gemini-2.5-flash", "gemini-2.5-flash-lite",
                                        "gemini-flash-latest"))},
    }})
    assert with_key["providers"]["gemini_reasoning"]["enabled"]
    assert with_key["providers"]["gemini_reasoning"]["api_key"] == "test-key"
    disabled = _merge_provider_payload(defaults, {"providers": {
        "gemini": {"api_key": "test-key", "enabled": True},
        "gemini_text": {"enabled": False},
    }})
    assert not disabled["providers"]["gemini_reasoning"]["enabled"]


def test_workload_routes_to_suitable_gemini_profile(monkeypatch):
    monkeypatch.setenv("ORION_MODE", "cloud")
    monkeypatch.delenv("ORION_PREFER", raising=False)
    settings = _settings_from_payload(_default_provider_payload("test-key"))
    router = ProviderRouter(settings, bus=SimpleNamespace(), memory=object())
    assert router.select_text_profiles("Hello ORION")[0].name == "gemini_text"
    assert router.select_text_profiles("Design a reliable network architecture")[0].name == "gemini_reasoning"


def test_specialised_image_model_is_not_misclassified_as_chat():
    def caps(model):
        return provider_capabilities(SimpleNamespace(
            kind="openai_compatible", model=model, enabled=True,
            api_key="configured", is_local=False))

    assert caps("gemini-3.1-flash-lite-image") == {Capability.IMAGE_GENERATION}
    assert Capability.VISION in caps("gemini-3.5-flash-lite")
    assert Capability.TEXT_GENERATION not in caps("gemini-3.8-flash-tts")
    assert context_window("gemini-3.8-flash") == 1_048_576
