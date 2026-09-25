"""
Which model to use — free first, then cheap.

  "I want mostly all the free API models but also cheap ones such as DeepSeek,
   the ones which cost less $ per million tokens ... that also do not lag out my
   computer."

A curated shortlist: free tiers ranked ahead of cheap paid ones, DeepSeek as the
value leader, and an honest note that cloud APIs use none of the machine while
local models lean on it. Prices are indicative and labelled as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import model_advisor as ma  # noqa: E402
from orion_core.model_advisor import Host, Load  # noqa: E402


# ── the catalogue is coherent ────────────────────────────────────────────────

def test_the_free_models_the_user_wanted_are_present():
    names = " ".join(m.name.lower() for m in ma.CATALOGUE)
    for expected in ("llama", "gemini", "qwen", "cerebras"):
        assert expected in names


def test_deepseek_is_in_the_cheap_tier():
    names = [m.name.lower() for m in ma.CATALOGUE]
    assert any("deepseek" in n for n in names)


def test_every_model_has_a_context_and_a_host():
    for m in ma.CATALOGUE:
        assert m.context > 0
        assert m.host in (Host.CLOUD, Host.LOCAL)


def test_free_tier_models_cost_zero():
    for m in ma.CATALOGUE:
        if m.free_tier and m.input_per_mtok == 0:
            assert m.blended_per_mtok == 0.0


def test_local_models_declare_their_load():
    for m in ma.CATALOGUE:
        if m.host is Host.LOCAL:
            assert m.local_load in (Load.LIGHT, Load.HEAVY)
        else:
            assert m.local_load is Load.NONE


# ── ranking ──────────────────────────────────────────────────────────────────

def test_cheapest_puts_free_tiers_first():
    ranked = ma.cheapest(6)
    # The first entries should be free.
    assert ranked[0].free_tier
    # And the list is non-decreasing in blended cost within paid tiers.
    paid = [m for m in ranked if not m.free_tier]
    costs = [m.blended_per_mtok for m in paid]
    assert costs == sorted(costs)


def test_cheapest_excludes_local_by_default():
    """A local model is 'free' but uses the machine — not what 'cheapest cloud'
    means. It only appears when explicitly asked for."""
    assert all(m.host is Host.CLOUD for m in ma.cheapest(20))
    assert any(m.host is Host.LOCAL for m in ma.cheapest(20, include_local=True))


def test_deepseek_is_the_cheapest_paid_option():
    paid = [m for m in ma.cheapest(20) if not m.free_tier]
    assert paid, "no paid options ranked"
    assert "deepseek" in paid[0].name.lower()


# ── recommendations by task ──────────────────────────────────────────────────

@pytest.mark.parametrize("task,expect", [
    ("coding", "deepseek"),
    ("reasoning", "deepseek-r1"),
    ("vision", "gemini"),
    ("offline", "local"),
    ("fastest", "cerebras"),
    ("free", "gemini"),
])
def test_recommendations_are_sensible(task, expect):
    picks = ma.recommend(task)
    assert picks, f"no recommendation for {task}"
    joined = " ".join(m.name.lower() for m in picks)
    assert expect in joined, f"{task} -> {joined}"


@pytest.mark.parametrize("alias,canon", [
    ("cheap", "cheapest"), ("code", "coding"), ("python", "coding"),
    ("maths", "reasoning"), ("image", "vision"), ("private", "offline"),
    ("speed", "fastest"),
])
def test_task_aliases_resolve(alias, canon):
    assert ma.recommend(alias) == ma.recommend(canon)


def test_an_unknown_task_falls_back_to_value():
    assert ma.recommend("something odd") == ma.recommend("cheapest")


# ── the advice text ──────────────────────────────────────────────────────────

def test_advice_names_the_price_date():
    assert ma.PRICES_AS_OF in ma.advise("cheapest")


def test_advice_answers_the_lag_question_for_cloud():
    text = ma.advise("cheapest")
    assert "CPU" in text or "GPU" in text
    assert "won't lag" in text or "use neither" in text.lower() or "none" in text.lower()


def test_advice_warns_about_local_load_for_offline():
    text = ma.advise("offline")
    assert "lag" in text.lower()


def test_advice_recommends_deepseek_for_value():
    assert "deepseek" in ma.advise("coding").lower()


# ── the tool ─────────────────────────────────────────────────────────────────

def test_the_ai_mode_tool_can_recommend():
    import inspect

    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    source = inspect.getsource(ProductivityDispatchMixin.ai_mode)
    assert "model_advisor" in source
    assert "recommend" in source


def test_the_schema_advertises_recommendation():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "ai_mode")
    assert "recommend" in tool["description"].lower()
    assert "deepseek" in tool["description"].lower()
    assert "task" in tool["parameters"]["properties"]
