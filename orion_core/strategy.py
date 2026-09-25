"""
The strategy engine — search a decision space instead of guessing at it.

Ask a language model "what should I do?" and you get the three or four options
it thought of first, argued for fluently. That is not strategy; it is the
first plausible framing, dressed up. The options it did not happen to mention
are invisible, and their absence is invisible too — nothing in the answer tells
you that the space was never searched.

This module searches it. The division of labour is the whole idea:

    the model defines the space      One call. What are the real dimensions of
                                     this decision, what options exist on each,
                                     what are we optimising, what is forbidden?
                                     Framing is a language problem.

    local compute explores it        Thousands of candidates enumerated from the
                                     Cartesian product, every one scored against
                                     every objective. No model, no network, no
                                     rate limit — microseconds each. Search is
                                     an arithmetic problem.

    the model judges the survivors   One call, on the handful that survived.
                                     Brand fit, the user's appetite for risk,
                                     what the numbers cannot see. Judgement is a
                                     language problem again.

Two model calls buy exhaustive coverage of a space that would take dozens of
calls to enumerate conversationally, and hundreds to score.

Between the two, three filters do the work:

    Constraints   Hard, declarative bounds. A candidate that breaks one is
                  discarded, and the count of what each constraint killed is
                  reported — "1,842 of 2,400 failed the budget" is itself a
                  finding, and often the most useful one.

    Pareto        Non-dominated sorting. A candidate survives unless something
                  else is at least as good on every objective and better on
                  one. This is the step that refuses to collapse a
                  multi-objective decision into a single number too early: the
                  cheapest option and the fastest option both survive, and the
                  user gets to see the trade-off rather than have it averaged
                  away by a weighting nobody agreed to.

    Tournament    Copeland scoring over the frontier — every survivor plays
                  every other, and a match is won by taking a majority of the
                  objectives. Deliberately not a weighted sum: a candidate that
                  wins on most criteria beats one that wins enormously on a
                  single heavily-weighted criterion, which is usually what a
                  person means by "better overall".

Nothing here executes anything. The engine proposes; the user decides. And the
constraints are parsed as *data* — bounds on named traits, never predicates —
so a model-authored space can never become model-authored code.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from .bus import OrionBus
from .data import ToolResult
from .security import SecuritySanitiser
from .utils import first_line


# Ceiling on candidates actually materialised. Above this the space is sampled
# rather than enumerated — see ``StrategySpace.generate``. The limit is memory
# and latency, not principle: 20k candidates score in well under a second.
DEFAULT_MAX_CANDIDATES = 20_000

# The frontier is thinned to this before the O(n²) tournament runs.
MAX_TOURNAMENT = 64

# How many finalists the model is asked to judge.
DEFAULT_TOP = 5


# ──────────────────────────────────────────────────────────────────────────────
# THE SPACE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Option:
    """One choice on one dimension, described by named traits in 0..1."""

    dimension: str
    name: str
    traits: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.dimension}={self.name}"


@dataclass(frozen=True)
class Dimension:
    """An axis of the decision, and the options available on it."""

    name: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Objective:
    """Something being optimised, read off one trait."""

    name: str
    trait: str
    weight: float = 1.0
    maximise: bool = True

    def score(self, traits: dict[str, float]) -> float:
        value = float(traits.get(self.trait, 0.0))
        return value if self.maximise else 1.0 - value


@dataclass(frozen=True)
class Constraint:
    """A hard bound on a trait. Declarative on purpose — never a predicate.

    A model proposing the space also proposes the constraints, so they must be
    data that can only ever be compared against, not logic that could be run.
    """

    name: str
    trait: str
    minimum: float | None = None
    maximum: float | None = None

    def holds(self, traits: dict[str, float]) -> bool:
        value = float(traits.get(self.trait, 0.0))
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def describe(self) -> str:
        bounds = []
        if self.minimum is not None:
            bounds.append(f"≥ {self.minimum:g}")
        if self.maximum is not None:
            bounds.append(f"≤ {self.maximum:g}")
        return f"{self.name} ({self.trait} {' and '.join(bounds) or 'unbounded'})"


@dataclass
class Candidate:
    """One complete strategy: a choice on every dimension, and how it scores."""

    choices: tuple[Option, ...]
    traits: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0

    @property
    def label(self) -> str:
        return " · ".join(f"{o.dimension}: {o.name}" for o in self.choices)

    def vector(self, objectives: Sequence[Objective]) -> tuple[float, ...]:
        return tuple(self.scores.get(o.name, 0.0) for o in objectives)

    def explain(self, objectives: Sequence[Objective]) -> str:
        parts = ", ".join(f"{o.name} {self.scores.get(o.name, 0.0):.2f}"
                          for o in objectives)
        return f"{self.label}  [{parts}]"


@dataclass
class StrategySpace:
    """Everything needed to search a decision: axes, aims and hard limits."""

    dimensions: tuple[Dimension, ...] = ()
    objectives: tuple[Objective, ...] = ()
    constraints: tuple[Constraint, ...] = ()

    @property
    def total_combinations(self) -> int:
        if not self.dimensions:
            return 0
        total = 1
        for dimension in self.dimensions:
            total *= max(1, len(dimension.options))
        return total

    def valid(self) -> bool:
        return bool(self.dimensions) and bool(self.objectives) and all(
            d.options for d in self.dimensions)

    def generate(self, cap: int = DEFAULT_MAX_CANDIDATES,
                 seed: int = 20260727) -> Iterator[Candidate]:
        """Every combination, or a fair sample when there are too many.

        Truncating the Cartesian product would be a silent bias: the product
        varies its last dimension fastest, so the first *cap* combinations all
        share the first dimension's opening option. When the space overflows,
        a seeded sample of distinct indices is decoded from mixed radix
        instead — unbiased across every dimension, and reproducible, because a
        strategy that changes each time you ask is not a strategy.
        """
        radices = [max(1, len(d.options)) for d in self.dimensions]
        total = self.total_combinations
        if not total:
            return
        cap = max(1, int(cap))
        if total <= cap:
            indices: Iterator[int] | Sequence[int] = range(total)
        else:
            indices = random.Random(seed).sample(range(total), cap)
        for index in indices:
            yield Candidate(choices=self._decode(index, radices))

    def _decode(self, index: int, radices: Sequence[int]) -> tuple[Option, ...]:
        """Mixed-radix decode: one integer ↔ one combination, bijectively."""
        picks: list[Option] = []
        for dimension, radix in zip(reversed(self.dimensions), reversed(radices)):
            picks.append(dimension.options[index % radix])
            index //= radix
        return tuple(reversed(picks))


# ──────────────────────────────────────────────────────────────────────────────
# SCORING, PARETO, TOURNAMENT
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_traits(candidate: Candidate) -> dict[str, float]:
    """A candidate's traits: the mean of its options', per named trait.

    Mean rather than sum so values stay in 0..1 however many dimensions the
    space has — otherwise a six-axis decision would make every constraint
    trivially satisfied and every objective saturate. Options that say nothing
    about a trait do not drag it down; they simply abstain.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for option in candidate.choices:
        for key, value in option.traits.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def score_candidate(candidate: Candidate, objectives: Sequence[Objective]) -> Candidate:
    candidate.traits = aggregate_traits(candidate)
    candidate.scores = {o.name: o.score(candidate.traits) for o in objectives}
    weight = sum(abs(o.weight) for o in objectives) or 1.0
    candidate.total = sum(
        candidate.scores[o.name] * o.weight for o in objectives) / weight
    return candidate


def dominates(a: Candidate, b: Candidate, objectives: Sequence[Objective]) -> bool:
    """True when *a* is at least as good everywhere and better somewhere."""
    better_somewhere = False
    for objective in objectives:
        left = a.scores.get(objective.name, 0.0)
        right = b.scores.get(objective.name, 0.0)
        if left < right - 1e-9:
            return False
        if left > right + 1e-9:
            better_somewhere = True
    return better_somewhere


def pareto_frontier(candidates: Sequence[Candidate],
                    objectives: Sequence[Objective]) -> list[Candidate]:
    """Exact non-dominated set, by sort-and-sweep.

    Sorting by the first objective (then by weighted total) descending means
    anything that dominates candidate *i* must already have been seen: a
    dominator is ≥ on every objective, so it is ≥ on the sort key, and ties on
    both keys cannot dominate strictly. Each candidate is therefore compared
    only against the frontier so far, which stays small — the naive all-pairs
    version is O(n²) and would not survive ten thousand candidates.
    """
    if not candidates or not objectives:
        return list(candidates)
    lead = objectives[0].name
    ordered = sorted(candidates,
                     key=lambda c: (c.scores.get(lead, 0.0), c.total),
                     reverse=True)
    frontier: list[Candidate] = []
    for candidate in ordered:
        if any(dominates(kept, candidate, objectives) for kept in frontier):
            continue
        frontier = [kept for kept in frontier
                    if not dominates(candidate, kept, objectives)]
        frontier.append(candidate)
    return frontier


def crowding_distance(frontier: Sequence[Candidate],
                      objectives: Sequence[Objective]) -> dict[int, float]:
    """NSGA-II crowding: how much empty space surrounds each candidate.

    Used to thin an over-large frontier without collapsing it onto one region.
    Extremes score infinity and are never thinned away, because the cheapest
    and the fastest option are precisely the ones a person wants to see.
    """
    distance = {index: 0.0 for index in range(len(frontier))}
    if len(frontier) <= 2:
        return {index: math.inf for index in distance}
    for objective in objectives:
        order = sorted(range(len(frontier)),
                       key=lambda i: frontier[i].scores.get(objective.name, 0.0))
        low = frontier[order[0]].scores.get(objective.name, 0.0)
        high = frontier[order[-1]].scores.get(objective.name, 0.0)
        span = high - low
        distance[order[0]] = math.inf
        distance[order[-1]] = math.inf
        if span <= 1e-9:
            continue
        for position in range(1, len(order) - 1):
            index = order[position]
            if distance[index] == math.inf:
                continue
            ahead = frontier[order[position + 1]].scores.get(objective.name, 0.0)
            behind = frontier[order[position - 1]].scores.get(objective.name, 0.0)
            distance[index] += (ahead - behind) / span
    return distance


def thin(frontier: Sequence[Candidate], objectives: Sequence[Objective],
         keep: int) -> list[Candidate]:
    """Reduce a frontier to *keep*, preferring the least crowded."""
    if len(frontier) <= keep:
        return list(frontier)
    distance = crowding_distance(frontier, objectives)
    ranked = sorted(range(len(frontier)),
                    key=lambda i: (distance[i], frontier[i].total), reverse=True)
    return [frontier[i] for i in ranked[:keep]]


def tournament(frontier: Sequence[Candidate],
               objectives: Sequence[Objective]) -> list[tuple[Candidate, int]]:
    """Copeland ranking: each candidate's pairwise wins over all the others.

    A match goes to whoever takes more objectives, not to whoever has the
    higher weighted sum. The difference matters: a weighted sum lets one
    heavily-weighted criterion carry a candidate that loses on everything else,
    which is rarely what a person means by "better overall". Ties in Copeland
    score fall back to the weighted total, so the ranking is always strict.
    """
    scores: list[tuple[Candidate, int]] = []
    for candidate in frontier:
        wins = 0
        for rival in frontier:
            if rival is candidate:
                continue
            mine = sum(1 for o in objectives
                       if candidate.scores.get(o.name, 0.0) > rival.scores.get(o.name, 0.0) + 1e-9)
            theirs = sum(1 for o in objectives
                         if rival.scores.get(o.name, 0.0) > candidate.scores.get(o.name, 0.0) + 1e-9)
            if mine > theirs:
                wins += 1
        scores.append((candidate, wins))
    scores.sort(key=lambda pair: (pair[1], pair[0].total), reverse=True)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# THE REPORT
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyReport:
    """What the search found, and what it cost to find it."""

    objective: str
    space: StrategySpace = field(default_factory=StrategySpace)
    considered: int = 0
    eliminated: dict[str, int] = field(default_factory=dict)
    survivors: int = 0
    frontier: list[Candidate] = field(default_factory=list)
    ranked: list[tuple[Candidate, int]] = field(default_factory=list)
    recommendation: str = ""
    elapsed_s: float = 0.0
    ok: bool = True
    note: str = ""

    def top(self, count: int = DEFAULT_TOP) -> list[Candidate]:
        return [candidate for candidate, _wins in self.ranked[:max(1, count)]]

    def search_summary(self) -> str:
        exhaustive = self.space.total_combinations
        sampled = "" if self.considered >= exhaustive else \
            f" (sampled from {exhaustive:,})"
        lines = [f"Searched {self.considered:,} candidate strategies{sampled} "
                 f"across {len(self.space.dimensions)} dimension(s)."]
        for name, count in sorted(self.eliminated.items(), key=lambda p: -p[1]):
            if count:
                lines.append(f"  {count:,} eliminated by: {name}")
        lines.append(f"  {self.survivors:,} feasible → {len(self.frontier)} on the "
                     f"Pareto frontier → {len(self.ranked[:DEFAULT_TOP])} finalists.")
        return "\n".join(lines)

    def shortlist(self, count: int = DEFAULT_TOP) -> str:
        lines = []
        for position, (candidate, wins) in enumerate(self.ranked[:max(1, count)], 1):
            lines.append(f"{position}. {candidate.explain(self.space.objectives)} "
                         f"— {wins} pairwise win(s)")
        return "\n".join(lines) or "(no candidate survived the constraints)"

    def to_tool_result(self) -> ToolResult:
        if not self.ok:
            return ToolResult(self.note or "The strategy search failed.", ok=False)
        blocks = [self.recommendation.strip()] if self.recommendation.strip() else []
        blocks.append("OPTIONS CONSIDERED\n" + self.shortlist())
        blocks.append(self.search_summary())
        if self.note:
            blocks.append(self.note)
        blocks.append(f"— strategy search: {self.elapsed_s:.1f}s")
        return ToolResult("\n\n".join(blocks))


# ──────────────────────────────────────────────────────────────────────────────
# MODEL-FACING PASSES
# ──────────────────────────────────────────────────────────────────────────────

SPACE_INSTRUCTION = (
    "You are a decision architect. Given a decision, describe its SPACE as "
    "JSON — do not solve it, do not recommend anything, do not write prose.\n"
    "Return one JSON object in a ```json fenced block:\n"
    "{\n"
    '  "dimensions": [ {"name": "channel", "options": [\n'
    '      {"name": "TikTok", "traits": {"reach": 0.9, "cost": 0.3, "speed": 0.8}} ]} ],\n'
    '  "objectives": [ {"name": "reach", "trait": "reach", "weight": 1.0, "maximise": true} ],\n'
    '  "constraints": [ {"name": "affordable", "trait": "cost", "maximum": 0.5} ]\n'
    "}\n"
    "Rules. 3 to 6 dimensions, each a genuinely independent axis of the "
    "decision, each with 3 to 8 concrete named options — real ones, specific to "
    "this decision, not 'option A'. Every option carries the SAME trait keys, "
    "each a number from 0 to 1 estimating that option on that trait. Choose 2 "
    "to 5 objectives that genuinely conflict — a space where one candidate wins "
    "everything was described wrongly. Constraints are optional and are bounds "
    "on a trait, never rules or conditions. Traits named in objectives and "
    "constraints must exist on the options. JSON only."
)

JUDGE_INSTRUCTION = (
    "You are advising on a decision that has already been searched "
    "exhaustively. You are given the objective, how the search went, and the "
    "few candidates that survived every constraint and are not beaten outright "
    "by any other candidate.\n"
    "Recommend ONE, and justify it on what the arithmetic cannot see: fit with "
    "this person's situation, second-order effects, execution risk, what fails "
    "first and what the early signal would be. Name the strongest runner-up and "
    "the single condition under which you would switch to it. If the shortlist "
    "shows a real trade-off, say plainly what is being given up.\n"
    "Do not re-derive the scores or praise the process. Do not invent options "
    "that are not on the shortlist. Write as ORION — calm, precise British "
    "English — and be brief enough to act on."
)

_FENCED = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def parse_space(payload: str) -> StrategySpace:
    """Build a space from model JSON, keeping only what is structurally sound.

    Tolerant by design: a space missing one malformed option is still a usable
    space, and refusing the whole search over a stray key would waste the call.
    Anything that cannot be read as a number, or that names a trait no option
    carries, is dropped rather than guessed at.
    """
    data = _extract_json(payload)
    dimensions: list[Dimension] = []
    for raw_dimension in _as_list(data.get("dimensions")):
        name = str(raw_dimension.get("name") or "").strip()
        if not name:
            continue
        options: list[Option] = []
        for raw_option in _as_list(raw_dimension.get("options")):
            option_name = str(raw_option.get("name") or "").strip()
            if not option_name:
                continue
            options.append(Option(dimension=name, name=option_name[:80],
                                  traits=_as_traits(raw_option.get("traits"))))
        if options:
            dimensions.append(Dimension(name=name[:60], options=tuple(options)))

    known = {trait for d in dimensions for o in d.options for trait in o.traits}
    objectives: list[Objective] = []
    for raw in _as_list(data.get("objectives")):
        trait = str(raw.get("trait") or raw.get("name") or "").strip()
        if trait not in known:
            continue
        objectives.append(Objective(
            name=str(raw.get("name") or trait).strip()[:60],
            trait=trait,
            weight=_as_float(raw.get("weight"), 1.0),
            maximise=bool(raw.get("maximise", raw.get("maximize", True))),
        ))
    constraints: list[Constraint] = []
    for raw in _as_list(data.get("constraints")):
        trait = str(raw.get("trait") or "").strip()
        if trait not in known:
            continue
        minimum = _as_optional_float(raw.get("minimum", raw.get("min")))
        maximum = _as_optional_float(raw.get("maximum", raw.get("max")))
        if minimum is None and maximum is None:
            continue
        constraints.append(Constraint(
            name=str(raw.get("name") or trait).strip()[:60],
            trait=trait, minimum=minimum, maximum=maximum))
    return StrategySpace(tuple(dimensions), tuple(objectives), tuple(constraints))


def _extract_json(payload: str) -> dict[str, Any]:
    raw = str(payload or "")
    match = _FENCED.search(raw)
    blob = match.group(1) if match else raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        data = json.loads(blob)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_traits(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    traits: dict[str, float] = {}
    for key, raw in value.items():
        number = _as_optional_float(raw)
        if number is not None:
            traits[str(key).strip()[:40]] = max(0.0, min(1.0, number))
    return traits


def _as_float(value: Any, fallback: float) -> float:
    number = _as_optional_float(value)
    return fallback if number is None else number


def _as_optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


# ──────────────────────────────────────────────────────────────────────────────
# THE ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """Generate thousands, score locally, rank, and spend the model on five."""

    def __init__(self, bus: OrionBus, router: Any, *, reasoning: Any | None = None,
                 telemetry: Any | None = None) -> None:
        self.bus = bus
        self.router = router
        self.reasoning = reasoning
        self.telemetry = telemetry
        self.last: StrategyReport | None = None

    async def strategise(
        self,
        objective: str,
        *,
        space: StrategySpace | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        top: int = DEFAULT_TOP,
        judge: bool = True,
        context: str = "",
    ) -> StrategyReport:
        """Search the decision *objective* describes and recommend one option."""
        objective = SecuritySanitiser.guard_text(
            str(objective or "").strip(), "strategy.objective")
        started = time.perf_counter()
        if not objective:
            return StrategyReport("", ok=False, note="No decision was supplied.")

        if space is None:
            space = await self._propose_space(objective, context)
        if space is None or not space.valid():
            return StrategyReport(
                objective, ok=False,
                note="I could not map that into a searchable decision space — "
                     "it may need to be stated as a choice between options, "
                     "with what you are optimising for.")

        self.bus.log.emit(
            f"STRATEGY: {space.total_combinations:,} combination(s) across "
            f"{len(space.dimensions)} dimension(s); "
            f"{len(space.objectives)} objective(s).")

        report = await asyncio.to_thread(
            self._search, objective, space, max_candidates)
        if not report.ranked:
            report.elapsed_s = time.perf_counter() - started
            report.note = (report.note or
                           "Every candidate failed a constraint — the constraints "
                           "may be mutually unsatisfiable.")
            self.last = report
            return report

        if judge:
            report.recommendation = await self._judge(report, top, context)
        report.elapsed_s = time.perf_counter() - started
        self.last = report
        self._publish(report)
        return report

    def snapshot(self) -> dict[str, Any]:
        if self.last is None:
            return {"ran": False}
        return {
            "ran": True,
            "objective": first_line(self.last.objective, 120),
            "considered": self.last.considered,
            "survivors": self.last.survivors,
            "frontier": len(self.last.frontier),
            "dimensions": [d.name for d in self.last.space.dimensions],
            "objectives": [o.name for o in self.last.space.objectives],
            "shortlist": self.last.shortlist(),
            "elapsed_s": round(self.last.elapsed_s, 2),
        }

    # ── the local search (no model, no network) ───────────────────────────────

    def _search(self, objective: str, space: StrategySpace,
                max_candidates: int) -> StrategyReport:
        """Generate, constrain, score and rank. Pure arithmetic, off-thread.

        Runs in a worker thread because a large space is the one genuinely
        CPU-bound thing ORION does, and the event loop is also carrying a live
        audio session.
        """
        report = StrategyReport(objective=objective, space=space)
        report.eliminated = {c.name: 0 for c in space.constraints}
        feasible: list[Candidate] = []
        for candidate in space.generate(max_candidates):
            report.considered += 1
            candidate.traits = aggregate_traits(candidate)
            broken = next((c for c in space.constraints
                           if not c.holds(candidate.traits)), None)
            if broken is not None:
                report.eliminated[broken.name] += 1
                continue
            feasible.append(score_candidate(candidate, space.objectives))
        report.survivors = len(feasible)
        if not feasible:
            return report
        report.frontier = pareto_frontier(feasible, space.objectives)
        contenders = thin(report.frontier, space.objectives, MAX_TOURNAMENT)
        report.ranked = tournament(contenders, space.objectives)
        return report

    # ── the two model passes ──────────────────────────────────────────────────

    async def _propose_space(self, objective: str, context: str) -> StrategySpace | None:
        prompt = f"DECISION:\n{objective}"
        if context.strip():
            prompt += f"\n\nCONTEXT:\n{context.strip()[:1500]}"
        try:
            _profile, text = await self.router.generate_text(
                prompt, instruction=SPACE_INSTRUCTION, task="strategy.space")
        except Exception as exc:
            self.bus.log.emit(f"STRATEGY: could not map the space - {first_line(exc, 120)}")
            return None
        space = parse_space(text)
        if not space.valid():
            self.bus.log.emit("STRATEGY: the proposed space was not usable.")
            return None
        return space

    async def _judge(self, report: StrategyReport, top: int, context: str) -> str:
        finalists = report.top(top)
        prompt = (
            f"DECISION:\n{report.objective}\n\n"
            + (f"CONTEXT:\n{context.strip()[:1500]}\n\n" if context.strip() else "")
            + f"HOW THE SEARCH WENT:\n{report.search_summary()}\n\n"
            + "OBJECTIVES: "
            + ", ".join(f"{o.name} ({'maximise' if o.maximise else 'minimise'} "
                        f"{o.trait}, weight {o.weight:g})" for o in report.space.objectives)
            + "\n\nSHORTLIST (each already survived every constraint and is beaten "
              "outright by nothing):\n"
            + "\n".join(f"{i}. {c.explain(report.space.objectives)}"
                        for i, c in enumerate(finalists, 1))
        )
        # The reasoning engine gives the final judgement a panel, a red team and
        # an evidence check — worth it here, because this is the one step where
        # a confident wrong call costs the user something real.
        if self.reasoning is not None:
            try:
                outcome = await self.reasoning.reason(prompt, tier="deliberate")
                if outcome.ok and outcome.answer.strip():
                    return outcome.answer.strip()
            except Exception as exc:
                self.bus.log.emit(f"STRATEGY: deliberation unavailable - {first_line(exc, 120)}")
        try:
            _profile, text = await self.router.generate_text(
                prompt, instruction=JUDGE_INSTRUCTION, task="strategy.judge")
            return text.strip()
        except Exception as exc:
            self.bus.log.emit(f"STRATEGY: judgement unavailable - {first_line(exc, 120)}")
            # The shortlist is the valuable part and it is already computed;
            # losing the commentary must not lose the search.
            return ""

    # ── reporting ─────────────────────────────────────────────────────────────

    def _publish(self, report: StrategyReport) -> None:
        self.bus.log.emit(
            f"STRATEGY: {report.considered:,} considered → {report.survivors:,} "
            f"feasible → {len(report.frontier)} on the frontier "
            f"({report.elapsed_s:.1f}s).")
        try:
            self.bus.dashboard_event.emit("strategy", self.snapshot())
        except Exception:
            pass
        if self.telemetry is not None:
            try:
                self.telemetry.metrics.incr("strategy.runs")
                self.telemetry.metrics.observe("strategy.candidates", report.considered)
                self.telemetry.metrics.observe("strategy.ms", report.elapsed_s * 1000.0)
            except Exception:
                pass


__all__ = [
    "Candidate",
    "Constraint",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_TOP",
    "Dimension",
    "JUDGE_INSTRUCTION",
    "MAX_TOURNAMENT",
    "Objective",
    "Option",
    "SPACE_INSTRUCTION",
    "StrategyEngine",
    "StrategyReport",
    "StrategySpace",
    "aggregate_traits",
    "crowding_distance",
    "dominates",
    "pareto_frontier",
    "parse_space",
    "score_candidate",
    "thin",
    "tournament",
]
