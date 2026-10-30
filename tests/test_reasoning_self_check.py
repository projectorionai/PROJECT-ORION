"""A deliberate answer's written-out arithmetic is re-done exactly, and a
wrong sum is sent back to be fixed (or fixed in place) before he says it."""
from __future__ import annotations

import pytest

from orion_core.agents import AgentManager
from orion_core.reasoning import (ReasoningEngine, ReasoningTier, arithmetic_slips,
                                  correct_arithmetic)

#: Model-style slips, each one wrong: every one must be caught.
WRONG = [
    "17 x 23 = 381", "12 × 12 = 124", "1,250 + 3,400 = 4,560", "96 / 8 = 13",
    "45 - 17 = 32", "3 * 4 + 5 = 27", "100 − 37 = 73", "7.5 * 4 = 31",
    "So the total is 19 + 26 = 46 pens.", "2,048 / 16 = 138",
]
#: Right sums, rounding to the stated precision, and things that are not sums.
RIGHT = [
    "17 x 23 = 391", "22 / 7 = 3.14", "10 / 3 = 3.33", "3 * 4 + 5 = 17", "1,250 + 3,400 = 4,650",
    "pages 10–12 = 3 pages", "2026-10-29", "x = 5", "the ratio 16:9 = widescreen",
    "50% = half", "5 - 8 = -3", "£5 + £3 = £8", "(3 + 4) x 2 = 14", "2^10 = 1024",
]


@pytest.mark.parametrize("text", WRONG)
def test_every_wrong_sum_is_caught(text):
    assert len(arithmetic_slips(text)) == 1, text


@pytest.mark.parametrize("text", RIGHT)
def test_right_sums_and_non_sums_are_left_alone(text):
    assert arithmetic_slips(text) == [], text


def test_a_slip_is_corrected_in_place_keeping_its_format():
    text = "Each box holds 17 x 23 = 381 parts, and 1,250 + 3,400 = 4,560 in all."
    slips = arithmetic_slips(text)
    assert [s.correct for s in slips] == ["391", "4,650"]
    assert correct_arithmetic(text, slips) == (
        "Each box holds 17 x 23 = 391 parts, and 1,250 + 3,400 = 4,650 in all.")


class _Signal:
    def emit(self, *_a):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Profile:
    name = "stub"


class _Router:
    def __init__(self, answer, fix):
        self.answer, self.fix, self.tasks = answer, fix, []

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
        self.tasks.append(task)
        if task == "reason.self_check":
            assert "17 x 23 = 391, not 381" in prompt
            return _Profile(), self.fix
        if task == "reason.critic":
            return _Profile(), "No issues found."
        return _Profile(), self.answer


def _engine(router):
    bus = _Bus()
    return ReasoningEngine(bus, router, AgentManager(router, bus))


ANSWER = ("You need 17 x 23 = 381 bricks for the wall, so order 400 to allow for breakage "
          "and keep the remainder for repairs later on.")


async def test_a_deliberate_answer_with_a_slip_is_rewritten_by_the_model():
    fixed = ANSWER.replace("381", "391")
    router = _Router(ANSWER, fixed)
    outcome = await _engine(router).reason("How many bricks do I need?",
                                           tier=ReasoningTier.DELIBERATE)
    assert "reason.self_check" in router.tasks
    assert "17 x 23 = 391" in outcome.answer and "381" not in outcome.answer
    assert "self-check: 1 arithmetic slip corrected" in outcome.footer()


async def test_a_rewrite_that_is_still_wrong_falls_back_to_the_exact_fix():
    router = _Router(ANSWER, "Still 17 x 23 = 380, honestly.")
    outcome = await _engine(router).reason("How many bricks do I need?",
                                           tier=ReasoningTier.DELIBERATE)
    assert outcome.answer.startswith("You need 17 x 23 = 391 bricks")
    assert "corrected in place" in outcome.self_check


async def test_a_clean_answer_costs_no_extra_call_and_a_standard_one_is_not_checked():
    router = _Router(ANSWER.replace("381", "391"), "unused")
    await _engine(router).reason("How many bricks do I need?", tier=ReasoningTier.DELIBERATE)
    assert "reason.self_check" not in router.tasks
    router = _Router(ANSWER, "unused")
    await _engine(router).reason("How many bricks?", tier=ReasoningTier.STANDARD)
    assert "reason.self_check" not in router.tasks
