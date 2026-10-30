"""Hosted provider profiles remain opt-in and preserve saved choices."""

from __future__ import annotations

import pytest

from orion_core.providers import (
    _default_provider_payload,
    _merge_provider_payload,
    _settings_from_payload,
)


@pytest.mark.parametrize("name,environment,base_url,model", [
    ("mistral", "ORION_MISTRAL_API_KEY", "https://api.mistral.ai/v1", "mistral-small-latest"),
    ("huggingface", "ORION_HUGGINGFACE_API_KEY", "https://router.huggingface.co/v1", "openai/gpt-oss-120b:cheapest"),
    ("nvidia", "ORION_NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct"),
    ("deepseek", "ORION_DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-flash"),
    ("together", "ORION_TOGETHER_API_KEY", "https://api.together.ai/v1", "Qwen/Qwen3.5-9B"),
    ("cerebras", "ORION_CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "gpt-oss-120b"),
])
def test_hosted_profile_needs_its_own_key(monkeypatch, name, environment, base_url, model):
    monkeypatch.delenv(environment, raising=False)
    defaults = _default_provider_payload()
    profile = defaults["providers"][name]
    assert name in defaults["provider_order"]
    assert profile["base_url"] == base_url
    assert profile["model"] == model
    assert profile["enabled"] is False
    assert _settings_from_payload(defaults).providers[name].supports_text_generation is False

    monkeypatch.setenv(environment, "test-key")
    configured = _settings_from_payload(_default_provider_payload()).providers[name]
    assert configured.api_key == "test-key"
    assert configured.supports_text_generation is True


def test_new_profiles_merge_into_saved_order_without_overwriting_user_choice(monkeypatch):
    for name in ("ORION_MISTRAL_API_KEY", "ORION_HUGGINGFACE_API_KEY",
                 "ORION_NVIDIA_API_KEY", "ORION_DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    defaults = _default_provider_payload()
    existing = {
        "provider_order": ["gemini_text", "groq", "local_ollama"],
        "providers": {
            "mistral": {"enabled": True, "api_key": "saved-key", "model": "my-model"},
        },
    }
    merged = _merge_provider_payload(defaults, existing)
    assert merged["providers"]["mistral"]["model"] == "my-model"
    assert merged["providers"]["mistral"]["api_key"] == "saved-key"
    assert merged["provider_order"].index("mistral") > merged["provider_order"].index("groq")
    assert merged["provider_order"].index("mistral") < merged["provider_order"].index("local_ollama")


def test_cloudflare_requires_valid_account_and_token(monkeypatch):
    monkeypatch.delenv("ORION_CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("ORION_CLOUDFLARE_ACCOUNT_ID", raising=False)
    profile = _default_provider_payload()["providers"]["cloudflare"]
    assert not profile["enabled"]
    assert profile["base_url"] == ""
    assert profile["model"] == "@cf/google/gemma-4-26b-a4b-it"

    monkeypatch.setenv("ORION_CLOUDFLARE_API_TOKEN", "test-token")
    monkeypatch.setenv("ORION_CLOUDFLARE_ACCOUNT_ID", "invalid/account")
    assert not _default_provider_payload()["providers"]["cloudflare"]["enabled"]

    account = "a" * 32
    monkeypatch.setenv("ORION_CLOUDFLARE_ACCOUNT_ID", account)
    configured = _settings_from_payload(_default_provider_payload()).providers["cloudflare"]
    assert configured.supports_text_generation
    assert configured.base_url == (
        f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
    )


def test_new_environment_key_activates_previously_persisted_blank_profile(monkeypatch):
    monkeypatch.delenv("ORION_MISTRAL_API_KEY", raising=False)
    previously_saved = _default_provider_payload()
    assert previously_saved["providers"]["mistral"]["api_key"] == ""
    monkeypatch.setenv("ORION_MISTRAL_API_KEY", "new-key")
    merged = _merge_provider_payload(_default_provider_payload(), previously_saved)
    assert merged["providers"]["mistral"]["enabled"] is True
    assert merged["providers"]["mistral"]["api_key"] == "new-key"

    previously_saved["providers"]["mistral"].update(
        {"api_key": "saved-key", "enabled": False}
    )
    preserved = _merge_provider_payload(_default_provider_payload(), previously_saved)
    assert preserved["providers"]["mistral"]["enabled"] is False
    assert preserved["providers"]["mistral"]["api_key"] == "saved-key"


@pytest.mark.parametrize("name,environment", [
    ("gemini", "ORION_GEMINI_API_KEY"),
    ("gemini_text", "ORION_GEMINI_API_KEY"),
    ("gemini_reasoning", "ORION_GEMINI_API_KEY"),
    ("gemma_4_31b", "ORION_GEMINI_API_KEY"),
    ("gemma_4_26b", "ORION_GEMINI_API_KEY"),
    ("openai", "ORION_OPENAI_API_KEY"),
    ("openrouter", "ORION_OPENROUTER_API_KEY"),
    ("groq", "ORION_GROQ_API_KEY"),
    ("cerebras", "ORION_CEREBRAS_API_KEY"),
    ("sambanova", "ORION_SAMBANOVA_API_KEY"),
    ("together", "ORION_TOGETHER_API_KEY"),
    ("xai_grok", "ORION_XAI_API_KEY"),
    ("thoughts", "ORION_THOUGHTS_API_KEY"),
])
def test_environment_key_added_after_first_launch_activates_hosted_profile(
    monkeypatch, name, environment
):
    monkeypatch.delenv(environment, raising=False)
    saved = _default_provider_payload()
    assert saved["providers"][name]["api_key"] == ""
    monkeypatch.setenv(environment, "later-key")

    merged = _merge_provider_payload(_default_provider_payload(), saved)
    assert merged["providers"][name]["api_key"] == "later-key"
    assert merged["providers"][name]["enabled"] is True


def test_saved_disabled_gemini_key_is_not_reenabled_by_environment(monkeypatch):
    monkeypatch.delenv("ORION_GEMINI_API_KEY", raising=False)
    saved = _default_provider_payload()
    saved["providers"]["gemini_text"].update(
        {"api_key": "saved-key", "enabled": False}
    )
    monkeypatch.setenv("ORION_GEMINI_API_KEY", "later-key")

    merged = _merge_provider_payload(_default_provider_payload(), saved)
    assert merged["providers"]["gemini_text"]["api_key"] == "saved-key"
    assert merged["providers"]["gemini_text"]["enabled"] is False


def test_cloudflare_environment_added_after_first_launch(monkeypatch):
    monkeypatch.delenv("ORION_CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("ORION_CLOUDFLARE_API_TOKEN", raising=False)
    previously_saved = _default_provider_payload()
    account = "b" * 32
    monkeypatch.setenv("ORION_CLOUDFLARE_ACCOUNT_ID", account)
    monkeypatch.setenv("ORION_CLOUDFLARE_API_TOKEN", "new-token")
    merged = _merge_provider_payload(_default_provider_payload(), previously_saved)
    profile = merged["providers"]["cloudflare"]
    assert profile["enabled"] is True
    assert profile["api_key"] == "new-token"
    assert profile["base_url"].endswith(f"/{account}/ai/v1")
