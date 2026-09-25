"""
Tests for the 2026-07-17 intelligence/efficiency pass.

* HTTP 413 / context-length failures are payload-class: a short cooldown and a
  tightened budget — never a minutes-long bench (the Groq 413 incident).
* _fit_to_budget swaps in the lean instruction and trims oversized prompts so
  small models are never sent the full instruction stack.
* Tier-aware routing: LARGE-tier work prefers reasoning-strength providers;
  small talk prefers fast ones.
* Forge JSON robustness: mechanical repair of raw newlines and trailing
  commas; the sandbox stages the module under BOTH names; the forge never
  pip-installs the tool's own module name.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.forge import LlmForgeBrain, strict_json_turn
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


class _StubMemory:
    def prompt_context(self, limit=16):
        return "MEMORY CONTEXT"


def _profile(name="groq", model="llama-3.1-8b-instant", **kw):
    defaults = dict(kind="openai_compatible", api_key="key",
                    base_url="https://example.invalid/v1", enabled=True)
    defaults.update(kw)
    return AIProviderProfile(name=name, model=model, **defaults)


def _router(profiles):
    settings = OrionProviderSettings(
        active_provider=profiles[0].name,
        provider_order=[p.name for p in profiles],
        providers={p.name: p for p in profiles},
    )
    return ProviderRouter(settings, _StubBus(), memory=_StubMemory())


# ── payload-class failures ────────────────────────────────────────────────────

def test_413_is_short_cooldown_not_a_bench():
    profile = _profile()
    router = _router([profile])
    router.mark_failure(profile, "HTTP 413: Request too large for model "
                                 "`llama-3.1-8b-instant` ... tokens per minute")
    remaining = router._cooldowns[profile.name] - time.monotonic()
    assert remaining < 10.0          # previously 300 s via the quota class
    assert router._budget_shrink[profile.name] < router.payload_budget(_profile("other"))


def test_payload_budget_hints():
    router = _router([_profile()])
    assert router.payload_budget(_profile(model="llama-3.1-8b-instant")) == 5000
    assert router.payload_budget(_profile(model="gpt-4o-mini")) == 60000
    assert router.payload_budget(_profile(model="claude-sonnet-5")) == 32000
    assert router.payload_budget(_profile(model="x", budget_tokens=1234)) == 1234


def test_fit_to_budget_trims_oversized_prompt():
    profile = _profile()               # 5000-token budget
    router = _router([profile])
    huge = "x" * 120000
    system, prompt = router._fit_to_budget(profile, "SYSTEM" * 4000, huge, "", None)
    assert len(prompt) < len(huge)
    assert "trimmed" in prompt
    # the lean instruction replaced the full stack
    assert "MEMORY CONTEXT" in system


def test_fit_to_budget_leaves_small_payloads_alone():
    profile = _profile(model="claude-sonnet-5")
    router = _router([profile])
    system, prompt = router._fit_to_budget(profile, "sys", "hello", "", None)
    assert (system, prompt) == ("sys", "hello")


# ── tier-aware routing ────────────────────────────────────────────────────────

def test_large_tier_prefers_reasoning_provider():
    fast = _profile("fastp", "llama-3.1-8b-instant", strengths=("fast",))
    strong = _profile("strongp", "big-model", strengths=("reasoning",))
    router = _router([fast, strong])
    ordered = router.select_text_profiles(
        "Analyse the architecture and design a detailed migration strategy "
        "step by step for the whole platform.")
    assert ordered[0].name == "strongp"
    ordered_small = router.select_text_profiles("hello there")
    assert ordered_small[0].name == "fastp"


# ── forge JSON robustness ─────────────────────────────────────────────────────

def test_repair_json_raw_newlines_and_trailing_commas():
    raw = '{"tool_code": "def run():\n    return \'ok\'",\n "requirements": [],}'
    data = LlmForgeBrain._parse_json(raw)
    assert "def run():" in data["tool_code"]
    assert data["requirements"] == []


def test_strict_json_turn_retries_with_parse_error():
    calls = []

    class _Router:
        async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
            calls.append(prompt)
            if len(calls) == 1:
                return None, "this is not json at all"
            return None, '{"tool_code": "print(1)"}'

    data = asyncio.run(strict_json_turn(_Router(), "build it"))
    assert data == {"tool_code": "print(1)"}
    assert len(calls) == 2
    assert "NOT VALID JSON" in calls[1]


def test_strict_json_turn_supports_legacy_router_signature():
    class _LegacyRouter:
        async def generate_text(self, prompt, system_extra=""):
            return None, '{"ok": true}'

    data = asyncio.run(strict_json_turn(_LegacyRouter(), "build it"))
    assert data == {"ok": True}


# ── sandbox stages both module names ──────────────────────────────────────────

def test_sandbox_stages_bare_and_tool_names(tmp_path):
    from orion_core.sandbox import SandboxVerificationHarness
    harness = SandboxVerificationHarness(_StubBus())
    harness.staging_dir = tmp_path
    harness._write_artefacts("dependency_resolver", "def run():\n    return 'ok'\n",
                             "import dependency_resolver\n")
    assert (tmp_path / "dependency_resolver_tool.py").exists()
    assert (tmp_path / "dependency_resolver.py").exists()
    assert (tmp_path / "dependency_resolver_test.py").exists()
