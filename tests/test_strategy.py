"""
Tests for Track D — the strategy engine.

Asked "what should I do?", a language model returns the three or four options
it thought of first. The options it did not mention are invisible, and so is
their absence. This engine searches the space instead: the model defines the
axes, local arithmetic enumerates and scores thousands of combinations, and the
model is spent only on the handful that survive.

What must be true:

  * a space too large to enumerate is SAMPLED, never truncated — truncation
    silently fixes the first dimension to its opening option;
  * the same question searched twice gives the same answer;
  * the Pareto sweep is EXACT — it is an optimisation over the naive all-pairs
    version, and an optimisation that changes the answer is a bug;
  * the frontier keeps its extremes: the cheapest and the fastest are exactly
    what a person wants to see;
  * the tournament rewards winning on most criteria, not winning enormously on
    one heavily-weighted criterion;
  * a model-authored space is parsed as DATA — a constraint is a bound on a
    trait and can never be a predicate;
  * the search survives the model: no space, no judgement, an unsatisfiable
    constraint set — each degrades to something honest.
"""

from __future__ import annotations

import asyncio
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.concurrency import ToolClass, classify
from orion_core.strategy import (
    Candidate,
    Constraint,
    Dimension,
    Objective,
    Option,
    StrategyEngine,
    StrategySpace,
    aggregate_traits,
    crowding_distance,
    dominates,
    pareto_frontier,
    parse_space,
    score_candidate,
    thin,
    tournament,
)


# ──────────────────────────────────────────────────────────────────────────────
# STUBS AND FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _Profile:
    name = "stub-provider"


_SPACE_JSON = """```json
{
  "dimensions": [
    {"name": "channel", "options": [
      {"name": "TikTok", "traits": {"reach": 0.9, "cost": 0.3}},
      {"name": "Email",  "traits": {"reach": 0.4, "cost": 0.1}},
      {"name": "Paid",   "traits": {"reach": 0.95, "cost": 0.9}}
    ]},
    {"name": "price", "options": [
      {"name": "9",  "traits": {"reach": 0.9, "cost": 0.2}},
      {"name": "49", "traits": {"reach": 0.5, "cost": 0.5}}
    ]}
  ],
  "objectives": [
    {"name": "reach", "trait": "reach", "weight": 1.0, "maximise": true},
    {"name": "cheapness", "trait": "cost", "weight": 1.0, "maximise": false}
  ],
  "constraints": [{"name": "affordable", "trait": "cost", "maximum": 0.6}]
}
```"""


class _Router:
    def __init__(self, space=_SPACE_JSON, judgement="Go with the first one.",
                 fail=()):
        self.space = space
        self.judgement = judgement
        self.fail = set(fail)
        self.calls = []

    def has_text_fallback(self):
        return True

    async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
        self.calls.append({"task": task, "prompt": prompt})
        if task in self.fail:
            raise RuntimeError(f"{task} provider down")
        if task == "strategy.space":
            return _Profile(), self.space
        return _Profile(), self.judgement

    def tasks(self):
        return [c["task"] for c in self.calls]


def _option(dimension, name, **traits):
    return Option(dimension=dimension, name=name, traits=dict(traits))


def _space(dimension_count=3, option_count=4, seed=1):
    rng = random.Random(seed)
    dimensions = tuple(
        Dimension(f"d{i}", tuple(
            _option(f"d{i}", f"o{j}", a=rng.random(), b=rng.random(), c=rng.random())
            for j in range(option_count)))
        for i in range(dimension_count))
    objectives = (Objective("A", "a"), Objective("B", "b"),
                  Objective("C", "c", maximise=False))
    return StrategySpace(dimensions, objectives, ())


def _scored(space, cap=10_000):
    return [score_candidate(c, space.objectives) for c in space.generate(cap)]


# ──────────────────────────────────────────────────────────────────────────────
# THE SPACE
# ──────────────────────────────────────────────────────────────────────────────

def test_the_space_is_the_cartesian_product():
    space = _space(dimension_count=3, option_count=4)
    assert space.total_combinations == 64


def test_a_small_space_is_enumerated_completely_and_without_repeats():
    space = _space(dimension_count=3, option_count=4)
    candidates = list(space.generate())
    assert len(candidates) == 64
    labels = {c.label for c in candidates}
    assert len(labels) == 64, "the mixed-radix decode must be bijective"


def test_every_option_of_every_dimension_appears():
    space = _space(dimension_count=3, option_count=4)
    seen = {(o.dimension, o.name) for c in space.generate() for o in c.choices}
    assert len(seen) == 12


def test_an_oversized_space_is_sampled_not_truncated():
    """Truncation would fix the first dimension to its opening option, because
    the product varies its LAST dimension fastest."""
    space = _space(dimension_count=5, option_count=6)     # 7,776
    candidates = list(space.generate(cap=300))
    assert len(candidates) == 300
    first_axis = {c.choices[0].name for c in candidates}
    assert len(first_axis) == 6, "the sample must span the first dimension"
    assert len({c.label for c in candidates}) == 300, "samples must be distinct"


def test_the_same_search_twice_gives_the_same_candidates():
    space = _space(dimension_count=5, option_count=6)
    first = [c.label for c in space.generate(cap=200)]
    second = [c.label for c in space.generate(cap=200)]
    assert first == second


def test_an_empty_space_generates_nothing():
    assert list(StrategySpace().generate()) == []
    assert StrategySpace().total_combinations == 0
    assert not StrategySpace().valid()


# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────────────────────────

def test_traits_average_across_the_chosen_options():
    candidate = Candidate(choices=(_option("x", "a", cost=0.2),
                                   _option("y", "b", cost=0.8)))
    assert aggregate_traits(candidate)["cost"] == 0.5


def test_an_option_silent_on_a_trait_abstains_rather_than_scoring_zero():
    candidate = Candidate(choices=(_option("x", "a", cost=0.6),
                                   _option("y", "b", speed=0.9)))
    traits = aggregate_traits(candidate)
    assert traits["cost"] == 0.6 and traits["speed"] == 0.9


def test_a_minimised_objective_inverts_its_trait():
    candidate = Candidate(choices=(_option("x", "a", cost=0.25),))
    score_candidate(candidate, (Objective("cheapness", "cost", maximise=False),))
    assert candidate.scores["cheapness"] == 0.75


def test_the_weighted_total_respects_the_weights():
    candidate = Candidate(choices=(_option("x", "a", a=1.0, b=0.0),))
    score_candidate(candidate, (Objective("A", "a", weight=3.0),
                                Objective("B", "b", weight=1.0)))
    assert abs(candidate.total - 0.75) < 1e-9


def test_an_absent_trait_scores_zero_rather_than_raising():
    candidate = Candidate(choices=(_option("x", "a", cost=0.5),))
    score_candidate(candidate, (Objective("Missing", "nonexistent"),))
    assert candidate.scores["Missing"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DOMINANCE AND THE PARETO FRONTIER
# ──────────────────────────────────────────────────────────────────────────────

def _fixed(**scores):
    candidate = Candidate(choices=())
    candidate.scores = dict(scores)
    candidate.total = sum(scores.values()) / max(1, len(scores))
    return candidate


_TWO = (Objective("A", "a"), Objective("B", "b"))


def test_dominance_needs_better_somewhere_and_worse_nowhere():
    strong, weak = _fixed(A=0.8, B=0.6), _fixed(A=0.5, B=0.4)
    assert dominates(strong, weak, _TWO)
    assert not dominates(weak, strong, _TWO)


def test_an_identical_candidate_does_not_dominate():
    a, b = _fixed(A=0.5, B=0.5), _fixed(A=0.5, B=0.5)
    assert not dominates(a, b, _TWO) and not dominates(b, a, _TWO)


def test_a_trade_off_is_not_dominance():
    cheap, fast = _fixed(A=0.9, B=0.1), _fixed(A=0.1, B=0.9)
    assert not dominates(cheap, fast, _TWO) and not dominates(fast, cheap, _TWO)


def test_the_frontier_keeps_both_sides_of_a_trade_off():
    """Both extremes survive; only something beaten outright is dropped.

    Note ``middling`` here is (0.05, 0.05) and not (0.2, 0.2): the latter is
    NOT dominated by either extreme, because it beats each of them on the axis
    that extreme sacrifices. That is the whole point of a frontier, and it is
    easy to get wrong by eye.
    """
    cheap, fast = _fixed(A=0.9, B=0.1), _fixed(A=0.1, B=0.9)
    middling, beaten = _fixed(A=0.2, B=0.2), _fixed(A=0.05, B=0.05)
    frontier = pareto_frontier([cheap, fast, middling, beaten], _TWO)
    assert cheap in frontier and fast in frontier and middling in frontier
    assert beaten not in frontier, "dominated on every objective"


def test_the_fast_sweep_matches_brute_force_exactly():
    """The sweep is an optimisation over all-pairs; one that changes the answer
    is a bug, so it is checked against the definition on real spaces."""
    for seed in range(6):
        space = _space(dimension_count=3, option_count=5, seed=seed)
        candidates = _scored(space)
        fast = {id(c) for c in pareto_frontier(candidates, space.objectives)}
        brute = {
            id(c) for c in candidates
            if not any(dominates(other, c, space.objectives) for other in candidates)
        }
        assert fast == brute, f"sweep and definition disagreed at seed {seed}"


def test_the_frontier_is_never_empty_for_a_non_empty_input():
    for seed in range(4):
        space = _space(dimension_count=3, option_count=4, seed=seed)
        assert pareto_frontier(_scored(space), space.objectives)


def test_an_empty_input_yields_an_empty_frontier():
    assert pareto_frontier([], _TWO) == []


# ──────────────────────────────────────────────────────────────────────────────
# CROWDING AND THINNING
# ──────────────────────────────────────────────────────────────────────────────

def test_the_extremes_are_infinitely_uncrowded():
    frontier = [_fixed(A=0.9, B=0.1), _fixed(A=0.5, B=0.5), _fixed(A=0.1, B=0.9)]
    distance = crowding_distance(frontier, _TWO)
    assert distance[0] == float("inf") and distance[2] == float("inf")
    assert distance[1] < float("inf")


def test_thinning_never_discards_an_extreme():
    frontier = [_fixed(A=x / 20, B=1 - x / 20) for x in range(21)]
    kept = thin(frontier, _TWO, keep=5)
    assert len(kept) == 5
    scores = [c.scores["A"] for c in kept]
    assert min(scores) == 0.0 and max(scores) == 1.0


def test_thinning_a_small_frontier_changes_nothing():
    frontier = [_fixed(A=0.9, B=0.1), _fixed(A=0.1, B=0.9)]
    assert thin(frontier, _TWO, keep=10) == frontier


# ──────────────────────────────────────────────────────────────────────────────
# THE TOURNAMENT
# ──────────────────────────────────────────────────────────────────────────────

_THREE = (Objective("A", "a", weight=10.0), Objective("B", "b"), Objective("C", "c"))


def _weighted(**traits):
    """A real candidate, scored through the weighted machinery under test."""
    return score_candidate(
        Candidate(choices=(_option("x", "only", **traits),)), _THREE)


def test_winning_most_criteria_beats_winning_hugely_on_one():
    """The reason this is a tournament and not a weighted sum.

    ``specialist`` wins the heavily-weighted A outright and loses B and C;
    ``allrounder`` takes two of the three. The weighted sum genuinely prefers
    the specialist — that is asserted here, not assumed — and the tournament
    still crowns the all-rounder, which is what a person usually means by
    'better overall'.
    """
    specialist = _weighted(a=1.0, b=0.1, c=0.1)
    allrounder = _weighted(a=0.6, b=0.9, c=0.9)
    assert specialist.total > allrounder.total, \
        "premise: the weighted sum favours the specialist"
    ranked = tournament([specialist, allrounder], _THREE)
    assert ranked[0][0] is allrounder


def test_the_winner_takes_the_most_pairwise_wins():
    best = _fixed(A=0.9, B=0.9, C=0.9)
    middle = _fixed(A=0.5, B=0.5, C=0.5)
    worst = _fixed(A=0.1, B=0.1, C=0.1)
    ranked = tournament([middle, worst, best], _THREE)
    assert [wins for _c, wins in ranked] == [2, 1, 0]
    assert ranked[0][0] is best


def test_a_tie_falls_back_to_the_weighted_total_so_ranking_is_strict():
    """Each wins one objective and ties the third, so Copeland cannot separate
    them; the weighting must, or the order would be an accident of input."""
    heavy = _weighted(a=0.9, b=0.1, c=0.5)
    light = _weighted(a=0.1, b=0.9, c=0.5)
    ranked = tournament([light, heavy], _THREE)      # deliberately worst-first
    assert [wins for _c, wins in ranked] == [0, 0], "premise: Copeland ties"
    assert ranked[0][0] is heavy, "the heavier-weighted A must break the tie"


def test_a_single_candidate_tournament_is_well_defined():
    assert tournament([_fixed(A=0.5, B=0.5, C=0.5)], _THREE)[0][1] == 0


# ──────────────────────────────────────────────────────────────────────────────
# CONSTRAINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_a_constraint_is_a_bound_and_holds_both_ways():
    constraint = Constraint("budget", "cost", maximum=0.5)
    assert constraint.holds({"cost": 0.4}) and not constraint.holds({"cost": 0.6})
    floor = Constraint("quality", "quality", minimum=0.7)
    assert floor.holds({"quality": 0.8}) and not floor.holds({"quality": 0.6})


def test_a_missing_trait_reads_as_zero_against_a_constraint():
    assert Constraint("budget", "cost", maximum=0.5).holds({})
    assert not Constraint("quality", "quality", minimum=0.1).holds({})


# ──────────────────────────────────────────────────────────────────────────────
# PARSING A MODEL-AUTHORED SPACE
# ──────────────────────────────────────────────────────────────────────────────

def test_a_fenced_space_is_parsed():
    space = parse_space(_SPACE_JSON)
    assert space.valid()
    assert [d.name for d in space.dimensions] == ["channel", "price"]
    assert space.total_combinations == 6
    assert [o.name for o in space.objectives] == ["reach", "cheapness"]
    assert space.constraints[0].maximum == 0.6


def test_bare_json_without_a_fence_is_parsed():
    space = parse_space(_SPACE_JSON.replace("```json", "").replace("```", ""))
    assert space.valid()


def test_an_objective_naming_an_unknown_trait_is_dropped():
    payload = _SPACE_JSON.replace('"trait": "reach"', '"trait": "vibes"')
    space = parse_space(payload)
    assert [o.name for o in space.objectives] == ["cheapness"]


def test_a_constraint_with_no_bounds_is_dropped():
    payload = _SPACE_JSON.replace('"maximum": 0.6', '"note": "be sensible"')
    assert parse_space(payload).constraints == ()


def test_out_of_range_traits_are_clamped():
    space = parse_space(_SPACE_JSON.replace('"reach": 0.9', '"reach": 4.2'))
    values = [o.traits["reach"] for d in space.dimensions for o in d.options]
    assert max(values) <= 1.0


def test_unparseable_output_yields_an_invalid_space_not_an_exception():
    for payload in ("", "I'd rather not.", "```json\n{oh dear\n```", "[]", "null"):
        assert not parse_space(payload).valid()


def test_a_dimension_with_no_usable_options_is_dropped():
    space = parse_space('{"dimensions": [{"name": "x", "options": [{}]}], '
                        '"objectives": []}')
    assert not space.valid()


# ──────────────────────────────────────────────────────────────────────────────
# THE ENGINE END TO END
# ──────────────────────────────────────────────────────────────────────────────

def _engine(router, **kwargs):
    return StrategyEngine(_Bus(), router, **kwargs)


async def test_a_decision_is_mapped_searched_and_recommended():
    router = _Router()
    report = await _engine(router).strategise("How should I launch the course?")
    assert report.ok
    assert router.tasks() == ["strategy.space", "strategy.judge"], \
        "exactly two model calls: define the space, judge the finalists"
    assert report.considered == 6
    assert report.ranked and report.recommendation == "Go with the first one."


async def test_the_constraint_tally_is_reported():
    router = _Router()
    report = await _engine(router).strategise("How should I launch the course?")
    assert report.eliminated["affordable"] >= 1
    assert report.survivors == report.considered - report.eliminated["affordable"]
    assert "eliminated by: affordable" in report.search_summary()


async def test_the_search_summary_names_what_was_actually_searched():
    report = await _engine(_Router()).strategise("How should I launch the course?")
    summary = report.search_summary()
    assert "Searched 6 candidate strategies" in summary
    assert "Pareto frontier" in summary


async def test_an_unusable_space_fails_honestly():
    report = await _engine(_Router(space="no thanks")).strategise("Something vague")
    assert not report.ok and "searchable decision space" in report.note


async def test_a_failed_space_call_does_not_raise():
    report = await _engine(_Router(fail={"strategy.space"})).strategise("A decision")
    assert not report.ok


async def test_losing_the_judgement_still_returns_the_shortlist():
    """The search is the valuable part and it is already computed."""
    report = await _engine(_Router(fail={"strategy.judge"})).strategise(
        "How should I launch the course?")
    assert report.ok and report.ranked and report.recommendation == ""
    assert "OPTIONS CONSIDERED" in report.to_tool_result().text


async def test_unsatisfiable_constraints_are_reported_as_such():
    impossible = _SPACE_JSON.replace('"maximum": 0.6', '"maximum": -1')
    report = await _engine(_Router(space=impossible)).strategise("A decision")
    assert report.survivors == 0
    assert "constraint" in report.note


async def test_judging_can_be_skipped_entirely():
    router = _Router()
    report = await _engine(router).strategise("How should I launch the course?",
                                              judge=False)
    assert router.tasks() == ["strategy.space"]
    assert report.ranked


async def test_a_supplied_space_skips_the_mapping_call():
    router = _Router()
    report = await _engine(router).strategise(
        "How should I launch the course?", space=parse_space(_SPACE_JSON))
    assert router.tasks() == ["strategy.judge"]
    assert report.ok


async def test_an_empty_decision_is_refused():
    report = await _engine(_Router()).strategise("   ")
    assert not report.ok


async def test_the_reasoning_engine_judges_the_finalists_when_present():
    """Track C and Track D compose: the one call that matters gets a panel."""
    class _Reasoning:
        def __init__(self):
            self.prompts = []

        async def reason(self, prompt, **_kwargs):
            self.prompts.append(prompt)

            class _Outcome:
                ok = True
                answer = "Deliberated recommendation."
            return _Outcome()

    reasoning = _Reasoning()
    router = _Router()
    report = await _engine(router, reasoning=reasoning).strategise(
        "How should I launch the course?")
    assert report.recommendation == "Deliberated recommendation."
    assert "strategy.judge" not in router.tasks(), "the panel replaces the plain call"
    assert "SHORTLIST" in reasoning.prompts[0]


async def test_a_broken_reasoning_engine_falls_back_to_the_plain_judgement():
    class _Broken:
        async def reason(self, *_a, **_k):
            raise RuntimeError("panel unavailable")

    report = await _engine(_Router(), reasoning=_Broken()).strategise(
        "How should I launch the course?")
    assert report.recommendation == "Go with the first one."


async def test_the_snapshot_explains_the_last_search():
    engine = _engine(_Router())
    assert engine.snapshot() == {"ran": False}
    await engine.strategise("How should I launch the course?")
    snapshot = engine.snapshot()
    assert snapshot["ran"] and snapshot["dimensions"] == ["channel", "price"]


async def test_a_large_space_completes_quickly():
    """Thousands of candidates is the design point, not an aspiration."""
    import time
    space = _space(dimension_count=6, option_count=6)      # 46,656
    engine = _engine(_Router())
    started = time.perf_counter()
    report = await engine.strategise("A big decision", space=space,
                                     max_candidates=8000, judge=False)
    assert report.considered == 8000
    assert time.perf_counter() - started < 5.0


# ──────────────────────────────────────────────────────────────────────────────
# DISPATCHER WIRING
# ──────────────────────────────────────────────────────────────────────────────

def _dispatcher():
    from orion_core.bus import OrionBus
    from orion_core.dispatcher import OrionDispatcher
    return OrionDispatcher(
        bus=OrionBus(), memory=None, grabber=None, file_intel=None,
        desktop=None, vision=None, outlook=None, notion=None,
        agent_manager=None, briefing=None,
    )


def test_the_strategy_tool_is_declared_and_routed():
    from orion_core.dispatcher import TOOL_DECLARATIONS
    assert "strategy" in _dispatcher().handler_table()
    assert any(t["name"] == "strategy" for t in TOOL_DECLARATIONS)


def test_strategy_is_not_batched_with_other_tools():
    assert classify("strategy") is ToolClass.SERIAL


async def test_the_strategy_tool_degrades_when_the_engine_is_absent():
    result = await _dispatcher().dispatch("strategy", {"objective": "Anything?"})
    assert not result.ok and "not available" in result.text


async def test_the_strategy_tool_returns_the_shortlist_and_the_search():
    dispatcher = _dispatcher()
    dispatcher.strategy = _engine(_Router())
    result = await dispatcher.dispatch(
        "strategy", {"objective": "How should I launch the course?"})
    assert result.ok
    assert "OPTIONS CONSIDERED" in result.text
    assert "Searched 6 candidate strategies" in result.text


async def test_the_strategy_tool_reports_the_previous_search():
    dispatcher = _dispatcher()
    dispatcher.strategy = _engine(_Router())
    empty = await dispatcher.dispatch("strategy", {"action": "last"})
    assert "not searched a decision space yet" in empty.text
    await dispatcher.dispatch("strategy", {"objective": "How should I launch?"})
    report = await dispatcher.dispatch("strategy", {"action": "last"})
    assert "channel" in report.text


async def test_out_of_range_tool_arguments_are_clamped_not_obeyed():
    dispatcher = _dispatcher()
    dispatcher.strategy = _engine(_Router())
    result = await dispatcher.dispatch("strategy", {
        "objective": "How should I launch the course?",
        "top": 9999, "max_candidates": "not a number"})
    assert result.ok
