"""
Regression tests for AgentManager routing (Improvement Pass, Priority 1.1).

Mark X.6 fixed trigger-happy specialist routing by introducing weighted
scoring plus ROUTE_THRESHOLD; these tests pin that behaviour so it cannot
silently regress: primary signals dominate, weak lone keywords fall through
to the general ORION capacity, and ties resolve in registration order.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.agents import AgentManager, BaseAgent


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubRouter:
    """No live provider: agents fall back to returning the specialist brief."""

    def has_text_fallback(self):
        return False


def _manager():
    return AgentManager(_StubRouter(), _StubBus())


class _NicheAgent(BaseAgent):
    name = "niche"
    title = "Niche Test Agent"
    expertise = "test fixture"
    primary = (r"\bzorbulator\b",)
    keywords = (r"\bzorb\b", r"\bflibber\b")


class _RivalAgent(BaseAgent):
    name = "rival"
    title = "Rival Test Agent"
    expertise = "test fixture"
    primary = (r"\bzorbulator\b",)
    keywords = ()


# ── weighted scoring mechanics ────────────────────────────────────────────────

def test_route_threshold_value_pinned():
    # Mark X.6 contract: a lone supporting keyword (weight 1) must never route.
    assert AgentManager.ROUTE_THRESHOLD == 3
    assert BaseAgent._PRIMARY_WEIGHT == 3
    assert BaseAgent._SECONDARY_WEIGHT == 1


def test_primary_signal_meets_threshold_and_routes():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    agent, score = mgr.route_with_confidence("engage the zorbulator")
    assert agent is not None and agent.name == "niche" and score == 3
    assert mgr.route("engage the zorbulator") is mgr._agents["niche"]


def test_lone_keyword_scores_but_does_not_route():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    agent, score = mgr.route_with_confidence("pass me the zorb")
    assert agent is not None and score == 1          # signal exists …
    assert mgr.route("pass me the zorb") is None     # … but below threshold


def test_two_keywords_still_below_threshold():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    _agent, score = mgr.route_with_confidence("the zorb and the flibber")
    assert score == 2
    assert mgr.route("the zorb and the flibber") is None


def test_primary_plus_keywords_accumulate():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    _agent, score = mgr.route_with_confidence("zorbulator with zorb and flibber")
    assert score == 5


def test_no_signal_returns_none_and_zero():
    mgr = _manager()
    agent, score = mgr.route_with_confidence("qqq zzz xxyzzy")
    assert agent is None and score == 0
    assert mgr.route("qqq zzz xxyzzy") is None


def test_ties_resolve_in_registration_order():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))   # registered first
    mgr.register(_RivalAgent(mgr.router, mgr.bus))   # same primary pattern
    agent, score = mgr.route_with_confidence("engage the zorbulator")
    assert score == 3 and agent.name == "niche"


# ── real specialists: representative routing smoke ────────────────────────────

def test_coding_request_routes_to_coding_agent():
    agent = _manager().route("debug this python traceback for me")
    assert agent is not None and agent.name == "coding"


def test_marketing_request_routes_to_marketing_agent():
    agent = _manager().route("plan a marketing campaign for the newsletter")
    assert agent is not None and agent.name == "marketing"


def test_weak_lone_email_mention_falls_through():
    # The documented Mark X.6 case: "email" alone is a weight-1 keyword and
    # must not commandeer the turn away from the general persona.
    assert _manager().route("check my email please") is None


# ── dispatch entry point ──────────────────────────────────────────────────────

def test_dispatch_empty_request_refused():
    result = asyncio.run(_manager().dispatch(""))
    assert not result.ok


def test_dispatch_unknown_agent_lists_registry():
    result = asyncio.run(_manager().dispatch("do something", agent_name="nonexistent"))
    assert not result.ok
    assert "coding" in result.text and "marketing" in result.text


def test_dispatch_auto_with_no_signal_falls_back_to_general():
    result = asyncio.run(_manager().dispatch("qqq zzz xxyzzy", agent_name="auto"))
    assert result.ok
    assert "general" in result.text.lower()


def test_dispatch_named_agent_bypasses_threshold():
    # Explicitly naming a specialist must work even with zero keyword signal.
    result = asyncio.run(_manager().dispatch("qqq zzz xxyzzy", agent_name="coding"))
    assert result.ok
    assert "Specialist brief" in result.text        # provider-less fallback path


def test_dispatch_blocks_destructive_request():
    import pytest
    from orion_core.security import SecurityViolation
    with pytest.raises(SecurityViolation):
        asyncio.run(_manager().dispatch("run rm -rf / now", agent_name="coding"))


def test_describe_lists_all_registered_specialists():
    rows = _manager().describe()
    names = {row["name"] for row in rows}
    assert {"marketing", "coding", "research"} <= names
    # Retired in Mark XXXI: chat personas with no instruments behind them.
    assert not names & {"design", "fashion", "entertainment"}


# ── Mark XXII: three new specialists fitting ORION's purpose ─────────────────

def test_new_specialists_are_registered():
    mgr = _manager()
    for name in ("neuroscience", "aiml", "security"):
        assert mgr.get(name) is not None, name


def test_neuroscience_question_routes_to_the_neuroscientist():
    mgr = _manager()
    pick = mgr.route("explain synaptic plasticity and how dopamine shapes the hippocampus")
    assert pick is not None and pick.name == "neuroscience"


def test_ml_question_routes_to_the_ml_agent():
    mgr = _manager()
    pick = mgr.route("how do I fine-tune a transformer in pytorch without overfitting")
    assert pick is not None and pick.name == "aiml"


def test_security_question_routes_to_the_security_agent():
    mgr = _manager()
    pick = mgr.route("how do I recognise a keylogger or rootkit and harden against ransomware")
    assert pick is not None and pick.name == "security"


def test_security_agent_is_defensive_and_educational():
    mgr = _manager()
    persona = mgr.get("security").persona.lower()
    assert "recognition" in persona or "defend" in persona
    assert "educational" in persona or "lawful" in persona
    assert "never weaponise" in persona or "no working attack" in persona


def test_neuroscience_persona_matches_the_users_field():
    mgr = _manager()
    persona = mgr.get("neuroscience").persona.lower()
    assert "neural engineering" in persona and "psychology" in persona
