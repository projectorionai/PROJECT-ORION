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
