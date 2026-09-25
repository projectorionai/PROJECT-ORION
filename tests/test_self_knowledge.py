"""
Runtime self-knowledge — what ORION is running on, and what he cannot do.

ORION's persona described at length who he was and never once said what he
could not do. That matters more than it sounds: a model with no stated limits
does not decline gracefully, it improvises, and the failure mode is a confident
description of having done something it never did.

The other half is that anything depending on the machine or the build has to be
ASSEMBLED, not written. A sentence naming the OS is wrong on a different one; a
list of abilities goes stale the moment a plugin is added.

These tests pin both, plus the boundary that keeps them from corrupting
IdentityManager: identity is who ORION is and is hashed for drift detection, so
it must be byte-identical on every machine. Session facts belong elsewhere.

Offline: no model, no Qt, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.self_knowledge import capability_block, machine_summary, tool_names


# ── the machine ──────────────────────────────────────────────────────────────

def test_the_machine_is_read_from_the_live_system():
    import platform

    summary = machine_summary()
    assert platform.system() in summary
    assert "Python" in summary


def test_the_machine_summary_never_raises(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: (_ for _ in ()).throw(OSError()))
    assert isinstance(machine_summary(), str)
    assert machine_summary()


# ── the tools ────────────────────────────────────────────────────────────────

def test_tools_come_from_the_live_registry():
    """Add a tool and ORION knows he gained it; remove one and he stops
    claiming it. A hardcoded list could do neither."""
    names = tool_names()
    assert names, "no tools discovered at all"
    assert "file_controller" in names
    assert names == sorted(set(names)), "duplicated or unsorted"


def test_the_router_wins_over_the_declarations():
    """The declarations are what the model is TOLD about; the router is what
    can actually be CALLED. Where they disagree, claiming the declaration would
    be claiming an ability that does not exist."""
    class _Dispatcher:
        _routes = {"only_this_one": None}

    assert tool_names(_Dispatcher()) == ["only_this_one"]


def test_explicit_declarations_are_used_when_there_is_no_router():
    got = tool_names(None, declarations=[{"name": "alpha"}, {"name": "beta"}])
    assert got == ["alpha", "beta"]


def test_a_broken_registry_falls_back_rather_than_raising():
    class _Dispatcher:
        _routes = "not a dict"

    assert isinstance(tool_names(_Dispatcher()), list)


# ── the block ────────────────────────────────────────────────────────────────

def test_the_block_states_the_limits_plainly():
    block = capability_block()
    assert "WHAT YOU CANNOT DO" in block
    assert "this machine" in block.lower()
    assert "never describe having done something you did not do" in block


def test_the_block_tells_the_model_to_say_so_rather_than_improvise():
    """The whole point. Without this the model fills the gap with prose."""
    block = capability_block().lower()
    assert "say so plainly" in block
    assert "nearest thing you can actually do" in block


def test_sight_is_described_as_a_frame_not_a_feed():
    """ORION captures a frame when asked. Left unsaid, a model will happily
    talk as though it has been watching."""
    block = capability_block(None, declarations=[{"name": "vision_analyse"}])
    assert "not a live feed" in block


def test_a_limit_is_only_mentioned_when_the_tool_exists():
    """A limit about a tool ORION does not have is wasted prompt, and worse,
    tells him about an ability he was not going to claim."""
    without = capability_block(None, declarations=[{"name": "file_controller"}])
    assert "live feed" not in without


def test_the_block_reports_the_real_tool_count():
    block = capability_block(None, declarations=[{"name": f"t{i}"} for i in range(7)])
    assert "7 tools" in block


def test_a_session_with_no_tools_says_it_can_only_converse():
    """Claiming 0 tools while listing abilities would be the exact failure this
    module exists to prevent."""
    block = capability_block(None, declarations=[])
    assert "only converse" in block


def test_the_block_is_short_enough_to_afford(tmp_path, monkeypatch):
    """A long list of prohibitions reads as distrust and crowds out the rest of
    the prompt. Three honest sentences do the work.

    Measured against an isolated language store. This used to read the real
    one, so it passed until ORION actually learned a language from being
    used — and then failed on the developer's machine and nowhere else, over
    an accumulated fact rather than a change to the code.
    """
    import orion_core.language_memory as language_memory

    monkeypatch.setattr(language_memory, "LANGUAGE_PATH",
                        tmp_path / "language.json")
    assert len(capability_block()) < 1450


def test_the_block_is_still_affordable_once_he_knows_your_language(
        tmp_path, monkeypatch):
    """The size that is actually sent in production.

    The language line is ~284 characters and appears on every turn once ORION
    has heard enough to be sure — so the budget that matters is this one, not
    the one measured on a fresh install. About 420 tokens per turn.
    """
    import orion_core.language_memory as language_memory

    path = tmp_path / "language.json"
    monkeypatch.setattr(language_memory, "LANGUAGE_PATH", path)
    memory = language_memory.LanguageMemory(path)
    for _ in range(language_memory.SWITCH_AFTER + 1):
        memory.observe("this is a sentence written in plain English")
    assert len(capability_block()) < 1800


# ── the boundary with identity ───────────────────────────────────────────────

def test_identity_does_not_carry_session_facts():
    """IdentityManager is hashed for drift detection, so persona_text must be
    the same string on every machine. Putting the OS in it would make the
    signature differ per machine AND still go stale on a plugin install."""
    from orion_core.bus import OrionBus
    from orion_core.identity import IdentityManager

    identity = IdentityManager(OrionBus())
    persona = identity.persona_text()
    assert "WHAT YOU CANNOT DO" not in persona
    assert "Python 3." not in persona
    assert "tools registered" not in persona


def test_the_provider_appends_the_block_to_the_persona():
    from orion_core.providers import ProviderRouter

    router = ProviderRouter.__new__(ProviderRouter)
    router._identity = None
    combined = router._with_self_knowledge("PERSONA")
    assert combined.startswith("PERSONA")
    assert "WHAT YOU CANNOT DO" in combined


def test_a_failure_assembling_the_block_never_costs_the_persona(monkeypatch):
    """An extra must not be able to take the identity down with it."""
    from orion_core.providers import ProviderRouter

    monkeypatch.setattr(
        "orion_core.self_knowledge.capability_block",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    router = ProviderRouter.__new__(ProviderRouter)
    router._identity = None
    assert router._with_self_knowledge("PERSONA") == "PERSONA"


# ── the silence rule ─────────────────────────────────────────────────────────

def test_the_gap_rule_is_about_silence_not_about_slow_tools():
    """Naming which tools 'take a while' does not work: the list is wrong the
    moment anything changes, and it misses the case where nothing is slow at
    all and ORION is simply composing a long answer. From the user's side both
    are the same thing — an assistant that stopped responding."""
    from orion_core.self_knowledge import SILENCE_RULE

    assert "GAP WOULD FORM" in SILENCE_RULE
    assert "long to write" in SILENCE_RULE, "the composing case is not covered"
    assert "one short sentence" in SILENCE_RULE.lower()


def test_the_gap_rule_guards_against_a_running_commentary():
    """An acknowledgement before every trivial answer is worse than the pause
    it replaces."""
    from orion_core.self_knowledge import SILENCE_RULE

    assert "not do this for quick answers" in SILENCE_RULE.lower()


def test_the_gap_rule_stays_in_the_users_language():
    from orion_core.self_knowledge import SILENCE_RULE

    assert "own language" in SILENCE_RULE


def test_the_gap_rule_reaches_the_prompt():
    assert "GAP WOULD FORM" in capability_block()
