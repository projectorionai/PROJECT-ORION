"""
Tests for multi-key / multi-model provider rotation and config self-rescue.

* Backup API keys rotate on auth and quota failures with a near-zero cooldown,
  instead of benching the provider for minutes while working keys sit idle.
* Alternate models rotate on retired-model errors.
* A wrapped rotation (every key tried) falls back to the normal long cooldown.
* api_keys.json is restored from a rescue copy when the canonical file is
  missing (the config/config/ stray-copy incident).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (
    AIProviderProfile,
    OrionProviderSettings,
    ProviderRouter,
)


class _Signal:
    def __init__(self, sink, name):
        self._sink, self._name = sink, name

    def emit(self, *payload):
        self._sink.append((self._name, payload[0] if len(payload) == 1 else payload))

    def connect(self, *_a, **_k):
        pass


class _StubBus:
    def __init__(self):
        self.events = []
        self.log = _Signal(self.events, "log")

    def __getattr__(self, name):
        sig = _Signal(self.events, name)
        object.__setattr__(self, name, sig)
        return sig


def _router(profile):
    settings = OrionProviderSettings(
        active_provider=profile.name,
        provider_order=[profile.name],
        providers={profile.name: profile},
    )
    bus = _StubBus()
    return ProviderRouter(settings, bus, memory=object()), bus


def _profile(**overrides):
    base = dict(
        name="p", kind="openai_compatible", model="model-a",
        api_key="key-1", base_url="https://api.example.com/v1", enabled=True,
    )
    base.update(overrides)
    return AIProviderProfile(**base)


def _skills_router(tmp_path):
    from orion_core.skills import SkillManager
    router, bus = _router(_profile())
    router.memory = SimpleNamespace(prompt_context=lambda **kwargs: "OWNER MEMORY",
                                    recent_replies=lambda *args: [])
    router.skills = SkillManager(bus, root=tmp_path / "skills")
    return router


@pytest.mark.parametrize("method", ["system_instruction", "system_instruction_lean"])
def test_enabled_skill_instructions_reach_provider_prompts(tmp_path, method, monkeypatch):
    router = _skills_router(tmp_path)
    prompt = getattr(router, method)()
    for skill in router.skills.registry.enabled():
        assert f"[SKILL {skill.name}]" in prompt
    assert "OWNER MEMORY" in prompt
    assert router.skills.set_enabled("research-method", False).ok
    assert "[SKILL research-method]" not in getattr(router, method)()
    assert router.skills.set_enabled("research-method", True).ok
    # Turning skills into instructions must not introduce a disk scan per turn.
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("prompt read disk"))
    assert "[SKILL research-method]" in getattr(router, method)()


def test_enabled_skill_instructions_reach_actual_text_attempt(tmp_path, monkeypatch):
    import asyncio
    router = _skills_router(tmp_path)
    requests = []

    async def send(profile, transport, request, scope, max_tokens, trace):
        requests.append(request)
        return SimpleNamespace(text="Finished", done_reason="stop", usage={},
                               response_id="synthetic", model="model-a", streamed=False,
                               total_ms=1, tool_calls=[], first_token_ms=None, load_ms=None)

    monkeypatch.setattr(router, "_send_attempt", send)

    async def scenario():
        profile = _profile()
        assert await router._openai_compatible_chat(profile, "Research a topic") == "Finished"
        assert "[SKILL research-method]" in requests[-1].system
        assert router.skills.set_enabled("research-method", False).ok
        await router._openai_compatible_chat(profile, "Research again")
        assert "[SKILL research-method]" not in requests[-1].system
        await router._openai_compatible_chat(profile, "Return JSON", instruction="JSON ONLY")
        assert requests[-1].system == "JSON ONLY"
        await router.aclose()

    asyncio.run(scenario())


def test_optional_skill_prompt_failure_preserves_core_instructions(tmp_path):
    router = _skills_router(tmp_path)
    router.skills = SimpleNamespace(prompt_context=lambda **kwargs: (_ for _ in ()).throw(ValueError("bad cache")))
    assert "OWNER MEMORY" in router.system_instruction()
    assert "OWNER MEMORY" in router.system_instruction_lean()


def _cooldown_remaining(router, name):
    return router._cooldowns.get(name, 0.0) - time.monotonic()


def test_auth_failure_rotates_to_backup_key_with_short_cooldown():
    profile = _profile(api_keys=("key-2", "key-3"))
    router, bus = _router(profile)
    assert router.active_key(profile) == "key-1"
    router.mark_failure(profile, "HTTP 401: unauthorised")
    assert router.active_key(profile) == "key-2"
    assert _cooldown_remaining(router, "p") <= 2.0
    assert any("rotating to backup key" in str(p) for (n, p) in bus.events if n == "log")


def test_quota_failure_also_rotates_keys():
    profile = _profile(api_keys=("key-2",))
    router, _bus = _router(profile)
    router.mark_failure(profile, "HTTP 429: rate limit exceeded")
    assert router.active_key(profile) == "key-2"
    assert _cooldown_remaining(router, "p") <= 2.0


def test_unknown_rate_limit_uses_bounded_exponential_backoff():
    profile = _profile()
    router, _bus = _router(profile)
    for expected in (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0):
        router.mark_failure(profile, "HTTP 429: rate limit exceeded")
        assert expected - 1.0 < _cooldown_remaining(router, "p") <= expected + 1e-6
    router._record_success(profile)
    router.mark_failure(profile, "HTTP 429: rate limit exceeded")
    assert 4.0 < _cooldown_remaining(router, "p") <= 5.0


def test_rate_limit_uses_provider_retry_delay_when_available():
    profile = _profile()
    router, _bus = _router(profile)
    router.mark_failure(profile, 'HTTP 429: {"retryDelay": "37s"}')
    assert 37.0 < _cooldown_remaining(router, "p") <= 38.0


def test_exhausted_key_cycle_falls_back_to_long_cooldown():
    profile = _profile(api_keys=("key-2",))
    router, _bus = _router(profile)
    router.mark_failure(profile, "HTTP 401: unauthorised")   # key-1 -> key-2
    router.mark_failure(profile, "HTTP 401: unauthorised")   # wraps to key-1
    assert router.active_key(profile) == "key-1"
    # Every key failed this cycle: the provider takes the real auth cooldown.
    assert _cooldown_remaining(router, "p") > 60.0


def test_single_key_provider_never_short_circuits_cooldown():
    profile = _profile()
    router, _bus = _router(profile)
    router.mark_failure(profile, "HTTP 401: unauthorised")
    assert _cooldown_remaining(router, "p") > 60.0


def test_model_error_rotates_to_alternate_model():
    profile = _profile(models=("model-b",))
    router, bus = _router(profile)
    assert router.active_model(profile) == "model-a"
    router.mark_failure(profile, "The model `model-a` has been decommissioned")
    assert router.active_model(profile) == "model-b"
    assert _cooldown_remaining(router, "p") <= 2.0
    assert any("switching to" in str(p) for (n, p) in bus.events if n == "log")


def test_daily_gemini_quota_uses_next_unspent_text_model():
    profile = _profile(name="gemini_text", model="gemini-2.5-flash",
                       models=("gemini-3.5-flash-lite", "gemini-2.5-flash-lite"))
    router, _bus = _router(profile)
    router.mark_failure(
        profile,
        "HTTP 429: GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded",
    )
    assert router.active_model(profile) == "gemini-3.5-flash-lite"
    assert _cooldown_remaining(router, "gemini_text") <= 2.0


def test_all_keys_deduplicates_and_drops_blanks():
    profile = _profile(api_key=" key-1 ", api_keys=("key-1", "", "key-2"))
    assert profile.all_keys() == ("key-1", "key-2")


def test_backup_keys_enable_a_provider_with_no_primary_key():
    profile = _profile(api_key="", api_keys=("key-2",))
    assert profile.supports_text_generation


def test_rescue_restores_missing_api_config(tmp_path, monkeypatch):
    from orion_core import providers as mod
    canonical = tmp_path / "api_keys.json"
    stray_dir = tmp_path / "config"
    stray_dir.mkdir()
    stray = stray_dir / "api_keys.json"
    stray.write_text(json.dumps({
        "providers": {"groq": {"kind": "openai_compatible", "api_key": "gsk_x",
                               "model": "m", "enabled": True}},
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "API_CONFIG_PATH", canonical)
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    mod._rescue_api_config()
    assert canonical.exists()
    # Mark XXXX.3: the restored key is sealed to the Windows account on disk.
    from orion_core import key_seal
    restored = key_seal.load_json(canonical)
    assert restored["providers"]["groq"]["api_key"] == "gsk_x"
    assert "gsk_x" not in canonical.read_text(encoding="utf-8") or not key_seal.available()


def test_rescue_ignores_keyless_copies(tmp_path, monkeypatch):
    from orion_core import providers as mod
    canonical = tmp_path / "api_keys.json"
    stray_dir = tmp_path / "config"
    stray_dir.mkdir()
    (stray_dir / "api_keys.json").write_text(json.dumps({
        "providers": {"groq": {"api_key": "", "enabled": False}},
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "API_CONFIG_PATH", canonical)
    monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
    mod._rescue_api_config()
    assert not canonical.exists()


def test_profile_config_roundtrips_backup_keys_and_models():
    from orion_core.providers import _profile_from_config, _profile_to_config
    profile = _profile(api_keys=("key-2",), models=("model-b",))
    raw = _profile_to_config(profile)
    restored = _profile_from_config("p", raw)
    assert restored.api_keys == ("key-2",)
    assert restored.models == ("model-b",)
