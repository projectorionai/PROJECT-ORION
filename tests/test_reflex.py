"""
Reflex fast-path (Mark XXVI) — instant local dispatch of unambiguous commands.

The whole value rests on PRECISION: it must fire on clear imperatives and stay
silent on anything conversational (which flows to the model). Both directions are
locked here, plus argument extraction and the worker wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.reflex import match_reflex, reflex_enabled  # noqa: E402


# ── recall: clear commands must map to the right tool + args ──────────────────

RECALL = [
    ("start a focus block",                 "focus", {"action": "start"}),
    ("start a pomodoro",                    "focus", {"preset": "pomodoro"}),
    ("start a 25 minute focus block on calculus", "focus", {"minutes": 25, "label": "calculus"}),
    ("begin a deep work session",           "focus", {"action": "start"}),
    ("how long is left on my focus",        "focus", {"action": "status"}),
    ("stop my focus block",                 "focus", {"action": "done"}),
    ("quiz me",                             "study", {"action": "review"}),
    ("review my flashcards",                "study", {"action": "review"}),
    ("what's due for review",               "study", {"action": "stats"}),
    ("catch me up",                         "catch_up", {}),
    ("situation report",                    "catch_up", {}),
    ("what's my runway",                    "finance", {"action": "runway"}),
    ("how much runway",                     "finance", {"action": "runway"}),
    ("what's my balance",                   "finance", {"action": "balance"}),
    ("what can you do",                     "diagnostics", {"action": "capabilities"}),
    ("token usage",                         "token_usage", {}),
]


@pytest.mark.parametrize("text,tool,expected_args", RECALL)
def test_clear_commands_reflex_to_the_right_tool(text, tool, expected_args):
    m = match_reflex(text)
    assert m is not None, f"{text!r} should reflex, not go to the model"
    assert m.tool == tool
    for k, v in expected_args.items():
        assert m.args.get(k) == v, f"{text!r}: {k}={m.args.get(k)!r} != {v!r}"


# ── precision: conversation MUST fall through to the model ────────────────────

CONVERSATIONAL = [
    "what do you think about chess",
    "tell me about my finances",
    "can we talk about focus techniques",
    "how are you today",
    "i'm not sure what to work on",
    "why is my runway important",
    "i was reading about the pomodoro technique earlier",
    "do you have any study tips",
    "what's the capital of France",
    "remind me why we chose this approach",
    "i think i should focus more",
    "let's discuss my finances tomorrow",
]


@pytest.mark.parametrize("text", CONVERSATIONAL)
def test_conversation_never_reflexes(text):
    assert match_reflex(text) is None, f"{text!r} wrongly matched a reflex"


def test_a_long_utterance_is_treated_as_conversation():
    long = "start a focus block and then also remind me about the thing we " \
           "discussed regarding my dissertation and the deadline next week"
    assert match_reflex(long) is None


def test_empty_or_blank_is_none():
    assert match_reflex("") is None
    assert match_reflex("   ") is None


# ── flag ──────────────────────────────────────────────────────────────────────

def test_reflex_is_on_by_default_and_disableable(monkeypatch):
    monkeypatch.delenv("ORION_REFLEX", raising=False)
    assert reflex_enabled() is True
    monkeypatch.setenv("ORION_REFLEX", "0")
    assert reflex_enabled() is False


# ── worker wiring (source-checked: importing live_worker is heavy/Qt) ─────────

def test_the_worker_runs_the_reflex_before_the_model():
    src = (Path(__file__).resolve().parents[1] / "orion_core" / "live_worker.py"
           ).read_text(encoding="utf-8")
    assert "async def _handle_reflex" in src
    # It must sit BEFORE the model turn, and fall through on failure.
    assert "if await self._handle_reflex(lowered, text):" in src
    reflex_at = src.index("if await self._handle_reflex")
    model_at = src.index("_send_text_turn(text)")
    assert reflex_at < model_at, "reflex must be checked before the model turn"
    assert "dispatch_chain(m.tool" in src


# ── "what's your Elo?" is a chess question ────────────────────────────────────

import pytest as _pytest


@_pytest.mark.parametrize("text", [
    "what's your elo?", "whats your elo", "what is your chess rating",
    "orion, what's your elo?", "how good are you at chess", "what elo are you",
    "what is your e.l.o.", "your elo",
])
def test_his_elo_is_his_chess_rating(text):
    """Asked for his Elo he used to answer as if the word meant nothing; the
    only Elo he has is his chess rating, and the chess tool holds it."""
    from orion_core.reflex import match_reflex
    m = match_reflex(text)
    assert m is not None and (m.tool, m.args) == ("chess", {"action": "rating"})


@_pytest.mark.parametrize("text", [
    "what's the elo system", "rate this film", "what's your rating of this song",
    "what is your favourite chess opening",
])
def test_other_rating_questions_stay_conversation(text):
    from orion_core.reflex import match_reflex
    m = match_reflex(text)
    assert m is None or m.tool != "chess"


# ── typing slips (spelling.py) ───────────────────────────────────────────────

@pytest.fixture
def speller(monkeypatch):
    """A tiny dictionary standing in for the sentence model's, so a command's
    slips can be put right offline."""
    from orion_core import spelling
    english = {"the": 1, "music": 2, "volume": 3, "turn": 4, "play": 5, "on": 6, "up": 7, "adele": 8}
    monkeypatch.setattr(spelling, "_english_words", lambda: english)
    monkeypatch.setattr(spelling, "_ENGLISH", None)
    cached = spelling.CachedSpeller(
        "test", lambda: ({"pause": 5, "music": 5, "volume": 5, "stormy": 3, "spotify": 4, "play": 5}, {}), lambda: 1)
    cached.get()
    monkeypatch.setattr(spelling, "TOOLS", cached)
    return cached


def test_a_slip_in_a_command_word_still_acts_at_once(speller):
    from orion_core.reflex import match_reflex_forgiving
    assert match_reflex("pasue the music") is None
    m = match_reflex_forgiving("pasue the music")
    assert m is not None and m.tool == "media_control" and m.args == {"action": "pause"}
    assert match_reflex_forgiving("turn the volmue up").args == {"action": "volume_up"}


def test_a_corrected_argument_goes_to_the_model_instead(speller):
    # "stormzy" is one edit from a word ORION knows; acting on that guess
    # would play the wrong thing, so no reflex fires.
    from orion_core.reflex import match_reflex_forgiving
    assert speller.correct("plya stormzy on spotify") == "play stormy on spotify"
    assert match_reflex_forgiving("plya stormzy on spotify") is None
    # The command word alone corrected, the name left as typed: that acts.
    m = match_reflex_forgiving("plya adele on spotify")
    assert m is not None and m.args.get("query") == "adele"
    assert match_reflex_forgiving("how was your day") is None
