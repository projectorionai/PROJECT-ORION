"""
Reflex expansion (Mark XXVI) — the commands that no longer wait for the model.

A reflex turns a 1-3 second model round-trip into a ~3 microsecond local
dispatch. That is the single largest perceived-latency win available in ORION,
because it is measured in seconds rather than milliseconds.

It is also the most dangerous optimisation in the system, for one reason: a
false positive makes ORION act *without thinking*. Missing a reflex costs a
second; firing the wrong one costs trust. So the discipline here is asymmetric
and deliberate — every rule is anchored end-to-end, and this file spends far
more assertions on what must NOT match than on what must.

The schema check at the bottom is the other half: a reflex that names a tool or
an action that does not exist would dispatch into nothing, and would only be
discovered by a user saying the phrase out loud.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatch_schema import TOOL_DECLARATIONS  # noqa: E402
from orion_core.reflex import _RULES, match_reflex  # noqa: E402


# ── what the expansion added ─────────────────────────────────────────────────

NEW_COMMANDS = [
    # wellbeing
    ("how am I sleeping lately", "wellbeing", {"action": "trend"}),
    ("how is my sleep", "wellbeing", {"action": "trend"}),
    ("how's my mood", "wellbeing", {"action": "trend"}),
    ("how is my wellbeing", "wellbeing", {"action": "trend"}),
    ("wellbeing trend", "wellbeing", {"action": "trend"}),
    ("wellbeing report", "wellbeing", {"action": "trend"}),
    ("wellbeing checkin", "wellbeing", {"action": "checkin"}),
    ("wellbeing check in", "wellbeing", {"action": "checkin"}),
    ("log my mood", "wellbeing", {"action": "checkin"}),
    ("record a checkin", "wellbeing", {"action": "checkin"}),
    # decisions
    ("what decisions are due", "decision", {"action": "due"}),
    ("decisions due", "decision", {"action": "due"}),
    ("any decisions to resolve", "decision", {"action": "due"}),
    ("my calibration", "decision", {"action": "calibration"}),
    ("calibration score", "decision", {"action": "calibration"}),
    ("how calibrated am I", "decision", {"action": "calibration"}),
    ("my brier score", "decision", {"action": "calibration"}),
    # plugins
    ("list my plugins", "plugin", {"action": "list"}),
    ("show plugins", "plugin", {"action": "list"}),
    ("what plugins are installed", "plugin", {"action": "list"}),
    ("plugin doctor", "plugin", {"action": "doctor"}),
    ("plugin health", "plugin", {"action": "doctor"}),
    ("are my plugins healthy", "plugin", {"action": "doctor"}),
    # responsiveness
    ("why were you so slow", "diagnostics", {"action": "latency"}),
    ("why are you laggy", "diagnostics", {"action": "latency"}),
    ("were you lagging", "diagnostics", {"action": "latency"}),
    ("latency report", "diagnostics", {"action": "latency"}),
    ("show me your performance stats", "diagnostics", {"action": "latency"}),
    # self-diagnostic
    ("run a self diagnostic", "diagnostics", {"action": "full"}),
    ("run diagnostics", "diagnostics", {"action": "full"}),
    ("system check", "diagnostics", {"action": "full"}),
    ("check yourself", "diagnostics", {"action": "full"}),
]


@pytest.mark.parametrize("text,tool,args", NEW_COMMANDS)
def test_the_new_commands_dispatch_locally(text, tool, args):
    match = match_reflex(text)
    assert match is not None, "'%s' still costs a model round-trip" % text
    assert match.tool == tool
    for key, value in args.items():
        assert match.args.get(key) == value, (text, match.args)


@pytest.mark.parametrize("text,tool,args", NEW_COMMANDS)
def test_case_and_trailing_punctuation_do_not_matter(text, tool, args):
    for variant in (text.upper(), text.capitalize(), text + "?", text + ".",
                    "  " + text + "  "):
        match = match_reflex(variant)
        assert match is not None and match.tool == tool, variant


# ── precision: the half that actually matters ────────────────────────────────

MUST_NOT_REFLEX = [
    # a command with a second clause is a conversation
    "can you list my plugins and then disable the discord one",
    "run a self diagnostic and tell me if the audio is broken",
    "what decisions are due for the business this quarter",
    "why were you so slow to reply about the finance thing",
    "plugin doctor said something odd, what does it mean",
    # reflection and advice-seeking, not a lookup
    "how am I sleeping compared to last month and why",
    "I think my sleep has been bad, what should I do",
    "should I log my mood every day or is that excessive",
    "explain my calibration score to me",
    "what does my brier score actually mean",
    # near-misses that change the meaning
    "how is your sleep",
    "list the plugins you would recommend building",
    "why was the network so slow",
    "is a system check something I should schedule",
    "how am I doing",
    # not commands at all
    "the plugin doctor is a good idea",
    "latency",
    "diagnostics are useful",
]


@pytest.mark.parametrize("text", MUST_NOT_REFLEX)
def test_conversation_still_reaches_the_model(text):
    assert match_reflex(text) is None, (
        "'%s' fired a reflex — ORION would act without thinking" % text)


def test_a_command_with_a_trailing_clause_never_reflexes():
    """Anchoring is the whole safety property; prove it holds generally."""
    for text, _tool, _args in NEW_COMMANDS:
        assert match_reflex(text + " and also open the browser") is None, text


def test_a_question_about_a_command_is_not_the_command():
    for text, _tool, _args in NEW_COMMANDS:
        assert match_reflex("what happens if I say " + text) is None, text


# ── the rules must be well formed ────────────────────────────────────────────

def test_every_reflex_names_a_real_registered_tool():
    """A reflex pointing at a tool that does not exist would dispatch into
    nothing, and only a user saying the phrase aloud would ever find out."""
    registered = {t["name"] for t in TOOL_DECLARATIONS}
    for text, _tool, _args in NEW_COMMANDS:
        match = match_reflex(text)
        assert match.tool in registered, (
            "reflex '%s' targets unregistered tool '%s'" % (text, match.tool))


def test_every_reflex_action_is_one_the_tool_advertises():
    schema = {t["name"]: t for t in TOOL_DECLARATIONS}
    for text, _tool, _args in NEW_COMMANDS:
        match = match_reflex(text)
        action = match.args.get("action")
        if not action:
            continue
        described = (schema[match.tool]["parameters"]["properties"]
                     .get("action", {}).get("description", ""))
        assert action in described.lower(), (
            "reflex '%s' uses action '%s', which %s does not advertise: %r"
            % (text, action, match.tool, described))


def test_no_two_rules_claim_the_same_phrase_differently():
    """Rule order decides ties; a phrase matching two rules is ambiguous and a
    later edit could silently swap which one wins."""
    for text, tool, args in NEW_COMMANDS:
        hits = []
        for pattern, build in _RULES:
            m = pattern.match(text.strip())
            if m:
                hits.append(build(m)[0])
        assert len(set(hits)) <= 1, (
            "'%s' matches rules for different tools: %s" % (text, set(hits)))


def test_the_rule_table_grew_but_stayed_small_enough_to_scan():
    assert len(_RULES) >= 19
    assert len(_RULES) < 60, (
        "the reflex table is becoming a rules engine — at this size, precision "
        "should be re-verified against a held-out phrase set rather than grown")


# ── it has to stay instant to be worth the risk ──────────────────────────────

def test_matching_stays_in_the_microseconds():
    phrases = [text for text, _, _ in NEW_COMMANDS] + MUST_NOT_REFLEX
    for p in phrases:
        match_reflex(p)                       # warm the regex cache
    start = time.perf_counter()
    for i in range(2000):
        match_reflex(phrases[i % len(phrases)])
    per_call_us = (time.perf_counter() - start) * 1e6 / 2000
    assert per_call_us < 500.0, (
        "a reflex takes %.0f us — it exists to be instant" % per_call_us)


def test_a_miss_is_as_cheap_as_a_hit():
    """The common case is a miss (most speech is conversation); it walks the
    whole table, so it must not be the slow path."""
    start = time.perf_counter()
    for _ in range(2000):
        match_reflex("tell me something interesting about the brain")
    per_call_us = (time.perf_counter() - start) * 1e6 / 2000
    assert per_call_us < 500.0, "a reflex miss costs %.0f us" % per_call_us
