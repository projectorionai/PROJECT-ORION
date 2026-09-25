"""Tests for the casual-correction loop (Mark X.12 §2.3)."""

from __future__ import annotations

import asyncio
import types

import pytest

from orion_core.bus import OrionBus
from orion_core.data import ToolResult
from orion_core.local_brain import LocalBrain, detect_correction


# ── the pure detector ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("utterance", [
    "no, that's wrong",
    "that's not right",
    "that is incorrect",
    "you're wrong",
    "no, actually it's the hippocampus",
    "actually, it's blue",
    "you're wrong, it's 1969",
    "no, that's not correct",
])
def test_detect_correction_fires_on_disputes(utterance):
    assert detect_correction(utterance) is not None


@pytest.mark.parametrize("utterance", [
    "no",
    "no thanks",
    "yes, that's right",
    "what's wrong with the printer?",
    "that's great, thank you",
    "",
])
def test_detect_correction_ignores_non_corrections(utterance):
    assert detect_correction(utterance) is None


def test_detect_correction_extracts_the_replacement():
    assert detect_correction("no, actually it's the hippocampus") == "it's the hippocampus"
    assert detect_correction("actually, it's blue") == "it's blue"


# ── integration through LocalBrain ─────────────────────────────────────────────

class _FakeLearning:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def correct(self, topic: str, correction: str) -> ToolResult:
        self.calls.append((topic, correction))
        return ToolResult("Understood, sir — I've corrected that.")


class _FakeKnowledge:
    def is_neuro_query(self, low: str) -> bool:
        return "synapse" in low

    def answer(self, raw: str):
        return "A synapse is a neural junction." if "synapse" in raw.lower() else None


def _brain(learning: _FakeLearning | None):
    bus = OrionBus()
    memory = types.SimpleNamespace(query=lambda *a, **k: [])
    dispatcher = types.SimpleNamespace(learning=learning)
    return LocalBrain(bus, memory, dispatcher, _FakeKnowledge())


def test_correction_after_a_fact_routes_to_learning():
    learning = _FakeLearning()
    brain = _brain(learning)

    async def _drive() -> str:
        fact = await brain.respond("what is a synapse")
        assert "junction" in fact                    # a correctable fact was given
        return await brain.respond("no, that's wrong, it's a gap")

    ack = asyncio.run(_drive())

    assert learning.calls, "a correction should have been recorded"
    topic, correction = learning.calls[0]
    assert topic == "what is a synapse"
    assert "it's a gap" in correction
    assert "correcting the earlier answer" in correction   # prior answer as context
    assert "corrected" in ack.lower()


def test_bare_no_without_a_prior_fact_is_not_a_correction():
    learning = _FakeLearning()
    brain = _brain(learning)
    # No fact was given first, so "no" must fall through to normal handling.
    asyncio.run(brain.respond("no"))
    assert not learning.calls


def test_correction_state_clears_after_one_use():
    learning = _FakeLearning()
    brain = _brain(learning)

    async def _drive() -> None:
        await brain.respond("what is a synapse")
        await brain.respond("no, that's wrong")     # consumes the correctable answer
        await brain.respond("that's wrong")         # nothing live to correct now

    asyncio.run(_drive())
    assert len(learning.calls) == 1                 # only the first correction fired
