"""
Mark XXII router expansion — ultra-fast inference clouds, rented cloud GPUs, and
the strict Tier-2 local hardware constraints.

  · Cerebras / SambaNova — OpenAI-compatible ultra-fast inference.
  · RunPod / Vast.ai / Modal — Tier 1 fallback, rented GPUs.
  · local_ollama (qwen2.5:7b) — Tier 2, held to num_ctx<=2048 + keep_alive=0
    so a 16GB-RAM host never swaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.providers import (  # noqa: E402
    ProviderRouter, _default_provider_payload, _merge_provider_payload,
)


PAYLOAD = _default_provider_payload()
PROVIDERS = PAYLOAD["providers"]
ORDER = PAYLOAD["provider_order"]

NEW_CLOUD = ("cerebras", "sambanova")
NEW_GPU = ("runpod", "vast_ai", "modal")


# ── the new providers exist and are well-formed ──────────────────────────────

def test_all_five_new_providers_are_present():
    for name in NEW_CLOUD + NEW_GPU:
        assert name in PROVIDERS, f"{name} missing from provider config"
        assert name in ORDER, f"{name} missing from cascade order"


def test_new_providers_are_openai_compatible():
    for name in NEW_CLOUD + NEW_GPU:
        assert PROVIDERS[name]["kind"] == "openai_compatible"


def test_new_providers_are_inert_until_configured():
    # With no API keys / endpoint URLs in the environment they must be disabled,
    # so merely adding them changes nothing about routing.
    for name in NEW_CLOUD + NEW_GPU:
        assert PROVIDERS[name]["enabled"] is False


def test_cerebras_and_sambanova_point_at_their_real_endpoints():
    assert PROVIDERS["cerebras"]["base_url"] == "https://api.cerebras.ai/v1"
    assert PROVIDERS["sambanova"]["base_url"] == "https://api.sambanova.ai/v1"


# ── the multi-tier cascade order ─────────────────────────────────────────────

def test_cloud_gpus_fall_after_commercial_and_before_local():
    idx = {name: ORDER.index(name) for name in ORDER}
    # commercial/inference clouds precede the rented GPU tier
    assert idx["together"] < idx["runpod"]
    assert idx["cerebras"] < idx["runpod"]
    # the rented GPU tier precedes local hardware
    for gpu in NEW_GPU:
        assert idx[gpu] < idx["local_lm_studio"]
        assert idx[gpu] < idx["local_ollama"]


def test_local_ollama_is_the_last_resort():
    assert ORDER[-1] == "local_ollama"


# ── Tier-2 hardware constraints (the 16GB-RAM box) ───────────────────────────

def test_local_context_is_capped_at_2048():
    assert ProviderRouter.LOCAL_NUM_CTX <= 2048


def test_ollama_targets_qwen25_7b_by_default():
    assert PROVIDERS["local_ollama"]["model"] == "qwen2.5:7b"


# ── merge keeps user order but appends the new tiers ─────────────────────────

def test_merge_appends_new_providers_to_an_old_order():
    # a saved config from before the new providers existed
    old = {
        "active_provider": "gemini",
        "provider_order": ["gemini", "groq", "local_ollama"],
        "providers": {"groq": {"enabled": True, "api_key": "x"}},
    }
    merged = _merge_provider_payload(_default_provider_payload(), old)
    order = merged["provider_order"]
    # the user's order is preserved (newcomers slot in where the defaults
    # put them, so a new provider is not stuck behind every tie in routing)
    assert [n for n in order if n in old["provider_order"]] == [
        "gemini", "groq", "local_ollama"]
    # and the new providers are appended so they're actually reachable
    for name in NEW_CLOUD + NEW_GPU:
        assert name in order
    # the user's own provider settings still win
    assert merged["providers"]["groq"]["enabled"] is True


def test_merge_without_saved_order_uses_the_full_default():
    merged = _merge_provider_payload(_default_provider_payload(),
                                     {"providers": {}})
    for name in NEW_CLOUD + NEW_GPU:
        assert name in merged["provider_order"]
