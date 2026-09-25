"""
Tests for Track C — the reasoning upgrade.

Before this, every question took the same path: one regex picked a persona, one
provider call answered it, and nothing checked the result. A specialist could
not look anything up, no second opinion was ever taken, and a confident
invention reached the user unexamined.

What must be true after the change:

  * how much thought a question gets is an explicit, deterministic decision —
    trivia is answered at REFLEX, a consequential decision convenes a panel,
    and an environment ceiling can hold the whole system cheaper;
  * a specialist can take readings mid-thought, and those readings are
    READ-ONLY by construction — the allowlist and the concurrency class must
    BOTH agree before an instrument fires, so reasoning can never act on the
    machine;
  * panel members draft independently and concurrently, then a red team
    attacks the drafts and a chair keeps only what survived;
  * the answer's claims are put to the evidence store, and a contradiction is
    surfaced rather than averaged away;
  * only instrument-grounded claims are written back — the model's own prose
    never becomes tomorrow's corroborating evidence;
  * every stage degrades instead of failing: no provider, no evidence store, no
    research agent, a critic that returns nonsense — each costs depth, not the
    answer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.agents import (
    AgentManager,
    AgentToolbelt,
    BaseAgent,
    CodingAgent,
    parse_tool_calls,
)
from orion_core.concurrency import ToolClass, classify
from orion_core.data import ToolResult
from orion_core.evidence import EvidenceEngine
from orion_core.reasoning import (
    BUDGETS,
    ReasoningEngine,
    ReasoningTier,
    assess_complexity,
    cap_tier,
    extract_claims,
    resolve_tier,
)


# ──────────────────────────────────────────────────────────────────────────────
# STUBS
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


class _Router:
    """Scripted provider. Replies are keyed by the ``task`` tag the caller uses,
    so a test can script the critic and the chair independently of the panel."""

    #: Tasks that are not a panel member drafting.
    _OVERHEAD = {"reason.critic", "reason.chair", "reason.audit"}

    def __init__(self, default="A specialist answer that is quite specific.",
                 scripts=None, delay=0.0, fallback=True, delay_drafts_only=False):
        self.default = default
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.delay = delay
        self.delay_drafts_only = delay_drafts_only
        self.calls = []
        self._fallback = fallback

    def has_text_fallback(self):
        return self._fallback

    async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
        self.calls.append({"prompt": prompt, "system": system_extra,
                           "instruction": instruction, "task": task})
        if self.delay and not (self.delay_drafts_only and task in self._OVERHEAD):
            await asyncio.sleep(self.delay)
        for key, queue in self.scripts.items():
            if task.startswith(key) and queue:
                return _Profile(), queue.pop(0)
        return _Profile(), self.default

    def tasks(self):
        return [c["task"] for c in self.calls]


def _agents(router=None, bus=None):
    return AgentManager(router or _Router(), bus or _Bus())


# ──────────────────────────────────────────────────────────────────────────────
# BUDGET TIERS
# ──────────────────────────────────────────────────────────────────────────────

def test_trivial_questions_are_answered_at_reflex():
    for question in ("hello", "thanks", "what time is it", "yes"):
        tier, _why = assess_complexity(question)
        assert tier is ReasoningTier.REFLEX, question


def test_a_consequential_decision_convenes_a_panel():
    tier, why = assess_complexity(
        "Should I use Postgres or SQLite for this, and why does it matter?")
    assert tier is ReasoningTier.DELIBERATE
    assert why, "the tier decision must be explainable"


def test_asking_to_think_hard_reaches_the_top_tier():
    tier, _why = assess_complexity(
        "Think hard about whether to rewrite the worker in Rust. The budget is "
        "limited and the audio path must not regress — what are the trade-offs?")
    assert tier is ReasoningTier.EXHAUSTIVE


def test_asking_to_double_check_escalates():
    tier, _why = assess_complexity("Are you sure that migration is safe to run?")
    assert tier in {ReasoningTier.DELIBERATE, ReasoningTier.EXHAUSTIVE}


def test_the_rationale_names_the_signal_that_escalated_it():
    _tier, why = assess_complexity("Should we compare the two vendors on cost?")
    assert "decision" in why or "comparison" in why


def test_environment_ceiling_holds_the_whole_system_cheaper():
    os.environ["ORION_MAX_REASONING_TIER"] = "standard"
    try:
        assert cap_tier(ReasoningTier.EXHAUSTIVE) is ReasoningTier.STANDARD
        tier, why = assess_complexity(
            "Think hard about this irreversible migration and its risks.")
        assert tier is ReasoningTier.STANDARD
        assert "policy" in why
    finally:
        os.environ.pop("ORION_MAX_REASONING_TIER", None)


def test_a_nonsense_ceiling_is_ignored_rather_than_crashing():
    os.environ["ORION_MAX_REASONING_TIER"] = "turbo"
    try:
        assert cap_tier(ReasoningTier.EXHAUSTIVE) is ReasoningTier.EXHAUSTIVE
    finally:
        os.environ.pop("ORION_MAX_REASONING_TIER", None)


def test_an_explicit_tier_overrides_the_assessment():
    budget, why = resolve_tier("hello", "exhaustive")
    assert budget.tier is ReasoningTier.EXHAUSTIVE
    assert why == "requested"


def test_tier_aliases_are_accepted():
    assert resolve_tier("x", "deep")[0].tier is ReasoningTier.DELIBERATE
    assert resolve_tier("x", "max")[0].tier is ReasoningTier.EXHAUSTIVE
    # "auto" means "you decide", not a tier.
    assert resolve_tier("hello", "auto")[0].tier is ReasoningTier.REFLEX


def test_an_unknown_tier_falls_back_to_judgement():
    budget, _why = resolve_tier("Should I do this, and why?", "ludicrous")
    assert budget.tier is ReasoningTier.DELIBERATE


def test_budgets_escalate_monotonically():
    tiers = [BUDGETS[t] for t in
             (ReasoningTier.REFLEX, ReasoningTier.STANDARD,
              ReasoningTier.DELIBERATE, ReasoningTier.EXHAUSTIVE)]
    for cheaper, dearer in zip(tiers, tiers[1:]):
        assert dearer.panel >= cheaper.panel
        assert dearer.lookups >= cheaper.lookups
        assert dearer.critiques >= cheaper.critiques


def test_reflex_buys_nothing_beyond_a_single_answer():
    budget = BUDGETS[ReasoningTier.REFLEX]
    assert (budget.panel, budget.lookups, budget.critiques) == (1, 0, 0)
    assert not budget.verify and not budget.record


# ──────────────────────────────────────────────────────────────────────────────
# TOOL-CALL PARSING
# ──────────────────────────────────────────────────────────────────────────────

def test_line_protocol_is_parsed():
    calls, prose = parse_tool_calls('TOOL: web_lookup\nARGS: {"query": "orion"}')
    assert calls == [("web_lookup", {"query": "orion"})]
    assert prose == ""


def test_several_calls_in_one_reply_are_all_parsed():
    calls, _prose = parse_tool_calls(
        'TOOL: query_intelligence\nARGS: {"query": "a"}\n'
        'TOOL: recall_conversation\nARGS: {"query": "b"}')
    assert [name for name, _ in calls] == ["query_intelligence", "recall_conversation"]


def test_a_fenced_json_call_is_understood():
    calls, _prose = parse_tool_calls(
        'I need a lookup.\n```tool\n{"tool": "find_files", "args": {"query": "orion"}}\n```')
    assert calls == [("find_files", {"query": "orion"})]


def test_a_call_with_no_arguments_is_understood():
    calls, _prose = parse_tool_calls("TOOL: situation")
    assert calls == [("situation", {})]


def test_prose_with_no_call_is_the_final_answer():
    calls, prose = parse_tool_calls("The answer is that SQLite is enough here.")
    assert calls == []
    assert prose.startswith("The answer")


def test_malformed_arguments_do_not_lose_the_call():
    calls, _prose = parse_tool_calls("TOOL: situation\nARGS: {not json}")
    assert calls == [("situation", {})]


# ──────────────────────────────────────────────────────────────────────────────
# THE TOOLBELT — READ-ONLY BY CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

async def _dispatch_ok(name, args):
    return ToolResult(f"{name} says: {args}")


async def test_an_allowed_read_only_tool_runs():
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=2)
    reading = await belt.run("query_intelligence", {"query": "orion"})
    assert reading.ok and "query_intelligence says" in reading.observation


async def test_an_unlisted_tool_is_refused_with_the_catalogue():
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=2)
    reading = await belt.run("open_app", {"app_name": "notepad"})
    assert not reading.ok
    assert "query_intelligence" in reading.observation


async def test_a_side_effecting_tool_is_refused_even_when_allow_listed():
    """The class gate is the real guarantee, not the allowlist.

    ``open_app`` here is deliberately allow-listed by mistake, exactly as a
    careless future edit might do it. It must still not fire, because
    ``classify`` does not call it PARALLEL.
    """
    assert classify("open_app") is not ToolClass.PARALLEL
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"open_app": "launch an app"}, budget=2)
    reading = await belt.run("open_app", {"app_name": "notepad"})
    assert not reading.ok
    assert "read-only" in reading.observation


async def test_every_direct_instrument_is_classified_read_only():
    for name in ReasoningEngine.DIRECT_TOOLS:
        assert classify(name) is ToolClass.PARALLEL, name


async def test_the_budget_is_a_hard_ceiling():
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=2)
    for _ in range(2):
        assert (await belt.run("query_intelligence", {"query": "x"})).ok
    spent = await belt.run("query_intelligence", {"query": "x"})
    assert not spent.ok and "budget" in spent.observation.lower()


async def test_readings_beyond_the_budget_are_dropped_not_queued():
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=2)
    readings = await belt.run_many([("query_intelligence", {"query": str(i)})
                                    for i in range(5)])
    assert len(readings) == 2


async def test_readings_are_taken_concurrently():
    async def _slow(_name, _args):
        await asyncio.sleep(0.05)
        return ToolResult("done")

    belt = AgentToolbelt(dispatch=_slow, allowed={"query_intelligence": "m",
                                                  "recall_conversation": "c"},
                         budget=4)
    started = time.perf_counter()
    readings = await belt.run_many([("query_intelligence", {"query": "a"}),
                                    ("recall_conversation", {"query": "b"})])
    elapsed = time.perf_counter() - started
    assert len(readings) == 2 and all(r.ok for r in readings)
    assert elapsed < 0.09, f"readings ran sequentially ({elapsed:.3f}s)"


async def test_a_failing_instrument_reports_rather_than_raises():
    async def _boom(_name, _args):
        raise RuntimeError("the store is locked")

    belt = AgentToolbelt(dispatch=_boom, allowed={"query_intelligence": "m"}, budget=2)
    reading = await belt.run("query_intelligence", {"query": "x"})
    assert not reading.ok and "locked" in reading.observation


async def test_an_extra_instrument_runs_its_own_callable():
    async def _lookup(args):
        return f"looked up {args['query']}"

    belt = AgentToolbelt(extras={"web_lookup": ("live web", _lookup)}, budget=2)
    reading = await belt.run("web_lookup", {"query": "orion"})
    assert reading.ok and reading.observation == "looked up orion"


def test_an_empty_toolbelt_is_not_offered_to_the_specialist():
    assert not AgentToolbelt(budget=0).available()
    assert not AgentToolbelt(dispatch=_dispatch_ok, allowed={}, budget=3).available()


# ──────────────────────────────────────────────────────────────────────────────
# TOOL-USING SPECIALISTS
# ──────────────────────────────────────────────────────────────────────────────

async def test_a_specialist_looks_something_up_then_answers():
    router = _Router(scripts={"reason.coding": [
        'TOOL: query_intelligence\nARGS: {"query": "the build"}',
        "The build is configured in pyproject.toml.",
    ]})
    agent = CodingAgent(router, _Bus())
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=3)
    finding = await agent.investigate("How is the build configured?", toolbelt=belt)
    assert finding.answer == "The build is configured in pyproject.toml."
    assert finding.grounded and finding.tools[0].tool == "query_intelligence"
    assert len(router.calls) == 2, "the reading must be fed back for a second pass"


async def test_the_reading_is_visible_in_the_follow_up_prompt():
    router = _Router(scripts={"reason.coding": [
        'TOOL: query_intelligence\nARGS: {"query": "x"}', "Answered.",
    ]})
    agent = CodingAgent(router, _Bus())
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=3)
    await agent.investigate("Anything?", toolbelt=belt)
    assert "INSTRUMENT READINGS" in router.calls[1]["prompt"]


async def test_a_specialist_that_wants_nothing_costs_one_call():
    router = _Router(default="No lookup needed; the answer is 42.")
    agent = CodingAgent(router, _Bus())
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=3)
    finding = await agent.investigate("What is 6 times 7?", toolbelt=belt)
    assert len(router.calls) == 1
    assert not finding.grounded and finding.answer.endswith("42.")


async def test_the_instrument_catalogue_is_only_shown_when_there_are_instruments():
    router = _Router()
    agent = CodingAgent(router, _Bus())
    await agent.investigate("Question?", toolbelt=AgentToolbelt(budget=0))
    assert "AVAILABLE INSTRUMENTS" not in router.calls[0]["system"]


async def test_a_lookup_spiral_is_forced_to_close():
    """A model that only ever asks for more readings must still produce prose."""
    router = _Router(default='TOOL: query_intelligence\nARGS: {"query": "again"}',
                     scripts={})
    agent = CodingAgent(router, _Bus())
    belt = AgentToolbelt(dispatch=_dispatch_ok,
                         allowed={"query_intelligence": "memory"}, budget=99)
    finding = await agent.investigate("Loop forever?", toolbelt=belt)
    assert len(router.calls) <= agent.MAX_INVESTIGATION_STEPS + 1
    assert finding.answer, "the forced close must still yield an answer"
    assert "TOOL:" not in finding.answer, "protocol lines must never reach the user"
    assert "ARGS:" not in finding.answer


async def test_no_provider_returns_the_specialist_brief():
    agent = CodingAgent(_Router(fallback=False), _Bus())
    finding = await agent.investigate("How do I debug this?")
    assert "Specialist brief" in finding.answer
    assert "SPECIALIST MODE" in finding.answer


async def test_a_provider_failure_is_reported_not_raised():
    class _Broken(_Router):
        async def generate_text(self, *a, **k):
            raise RuntimeError("all providers failed")

    finding = await CodingAgent(_Broken(), _Bus()).investigate("Anything?")
    assert not finding.ok and "could not reach a provider" in finding.answer


# ──────────────────────────────────────────────────────────────────────────────
# PANEL SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def test_the_analyst_is_always_seated():
    panel = _agents().panel("Refactor this Python code and debug the exception", size=3)
    assert "research" in {a.name for a in panel}


def test_the_panel_leads_with_the_matching_specialist():
    panel = _agents().panel("Debug this Python traceback in my code", size=3)
    assert panel[0].name == "coding"


def test_a_panel_of_one_is_the_best_specialist_not_the_anchor():
    panel = _agents().panel("Debug this Python traceback in my code", size=1)
    assert [a.name for a in panel] == ["coding"]


def test_the_panel_never_exceeds_its_size():
    panel = _agents().panel(
        "Compare marketing funnels, code architecture, visual design and outfit "
        "styling for this launch — analyse the trade-offs", size=2)
    assert len(panel) == 2


def test_an_unmatched_question_still_seats_the_analyst():
    panel = _agents().panel("qwertyuiop zxcvbnm", size=3)
    assert [a.name for a in panel] == ["research"]


def test_below_threshold_specialists_are_not_seated():
    """A lone supporting keyword is not enough to earn a chair."""
    panel = _agents().panel("Should I send that email, and why?", size=3)
    assert "marketing" not in {a.name for a in panel}


# ──────────────────────────────────────────────────────────────────────────────
# THE ENGINE END TO END
# ──────────────────────────────────────────────────────────────────────────────

def _engine(router, **kwargs):
    bus = _Bus()
    return ReasoningEngine(bus, router, AgentManager(router, bus), **kwargs)


async def test_reflex_costs_exactly_one_provider_call():
    router = _Router()
    outcome = await _engine(router).reason("hello")
    assert outcome.budget.tier is ReasoningTier.REFLEX
    assert len(router.calls) == 1, "reflex must not convene anything"
    assert not outcome.critique.applied
    assert not outcome.verification.ran


async def test_a_deliberate_question_drafts_criticises_and_synthesises():
    router = _Router(scripts={
        "reason.critic": ["DRAFT 1 — X\nWRONG: none found\nSCORE: 8\n"
                          "DISAGREEMENT: the drafts agree"],
        "reason.chair": ["Use SQLite; it is sufficient at this scale."],
    })
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    tasks = router.tasks()
    assert outcome.budget.tier is ReasoningTier.DELIBERATE
    assert len(outcome.findings) >= 2, "a panel, not a single specialist"
    assert "reason.critic" in tasks and "reason.chair" in tasks
    assert outcome.answer == "Use SQLite; it is sufficient at this scale."
    assert outcome.critique.applied


async def test_panel_members_never_see_each_other_drafts():
    router = _Router(scripts={"reason.chair": ["Synthesised."]})
    await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    drafting = [c for c in router.calls if c["task"].startswith("reason.")
                and c["task"] not in {"reason.critic", "reason.chair", "reason.audit"}]
    assert len(drafting) >= 2
    for call in drafting:
        assert "DRAFT" not in call["prompt"], "drafts must be independent"


async def test_panel_members_draft_concurrently():
    # Only the drafting calls are slow, so the measurement is of the panel and
    # not of the critic and chair that follow it sequentially either way.
    router = _Router(delay=0.05, delay_drafts_only=True,
                     scripts={"reason.chair": ["Synthesised."]})
    started = time.perf_counter()
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    elapsed = time.perf_counter() - started
    members = len(outcome.findings)
    assert members >= 2
    assert elapsed < 0.05 * 1.8, (
        f"{members} drafts took {elapsed:.3f}s — they ran sequentially")


async def test_the_parallel_kill_switch_reaches_the_panel():
    """``ORION_PARALLEL_TOOLS=0`` must make reasoning sequential too."""
    os.environ["ORION_PARALLEL_TOOLS"] = "0"
    try:
        router = _Router(delay=0.05, delay_drafts_only=True,
                         scripts={"reason.chair": ["Synthesised."]})
        started = time.perf_counter()
        outcome = await _engine(router).reason(
            "Should I refactor this Python code for performance, or redesign the "
            "database schema? Compare the trade-offs and explain why.")
        elapsed = time.perf_counter() - started
        assert elapsed >= 0.05 * len(outcome.findings) * 0.9
    finally:
        os.environ.pop("ORION_PARALLEL_TOOLS", None)


async def test_the_red_team_scores_are_read_back():
    router = _Router(scripts={
        "reason.critic": ["DRAFT 1 — A\nSCORE: 9\nDRAFT 2 — B\nSCORE: 3\n"
                          "DISAGREEMENT: they differ on cost"],
        "reason.chair": ["Answer."],
    })
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert sorted(outcome.critique.scores.values(), reverse=True)[:2] == [9, 3]
    assert "cost" in outcome.critique.disagreement
    assert outcome.critique.weakest()


async def test_a_broken_red_team_does_not_stop_the_answer():
    class _CriticFails(_Router):
        async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
            if task == "reason.critic":
                raise RuntimeError("critic provider down")
            return await super().generate_text(prompt, system_extra,
                                               instruction=instruction, task=task)

    router = _CriticFails(scripts={"reason.chair": ["Answer stands."]})
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert outcome.ok and not outcome.critique.applied
    assert outcome.answer


async def test_a_broken_chair_falls_back_to_the_best_scored_draft():
    class _ChairFails(_Router):
        async def generate_text(self, prompt, system_extra="", *, instruction=None, task=""):
            if task == "reason.chair":
                raise RuntimeError("chair provider down")
            return await super().generate_text(prompt, system_extra,
                                               instruction=instruction, task=task)

    router = _ChairFails(default="A usable draft answer.",
                         scripts={"reason.critic": ["SCORE: 7\nDISAGREEMENT: none"]})
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert outcome.ok and outcome.answer == "A usable draft answer."


async def test_exhaustive_revises_and_audits():
    router = _Router(scripts={
        "reason.critic": ["SCORE: 4\nDISAGREEMENT: the drafts agree"],
        "reason.chair": ["A synthesised answer of a reasonable length here."],
        "reason.audit": ["A synthesised answer of a reasonable length here, repaired."],
    })
    outcome = await _engine(router).reason(
        "Think hard: should I rewrite the worker in Rust? The budget is limited, "
        "the audio path must not regress, and I need the trade-offs and risks.")
    assert outcome.budget.tier is ReasoningTier.EXHAUSTIVE
    assert "reason.audit" in router.tasks()
    assert outcome.answer.endswith("repaired.")


async def test_a_truncating_auditor_is_ignored():
    router = _Router(scripts={
        "reason.critic": ["SCORE: 6\nDISAGREEMENT: none"],
        "reason.chair": ["A carefully synthesised answer with real substance in it."],
        "reason.audit": ["Fine."],
    })
    outcome = await _engine(router).reason(
        "Think hard: should I rewrite the worker in Rust? The budget is limited, "
        "the audio path must not regress, and I need the trade-offs and risks.")
    assert outcome.answer.startswith("A carefully synthesised")


async def test_a_named_specialist_replaces_the_panel():
    router = _Router()
    outcome = await _engine(router).reason(
        "Should I refactor this code and why?", agent="marketing")
    assert [f.agent for f in outcome.findings] == ["marketing"]


async def test_an_unknown_specialist_is_refused():
    outcome = await _engine(_Router()).reason("Anything?", agent="astrologer")
    assert not outcome.ok


async def test_the_footer_records_how_the_answer_was_reached():
    router = _Router(scripts={"reason.chair": ["Answer."],
                              "reason.critic": ["SCORE: 8\nDISAGREEMENT: none"]})
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    footer = outcome.to_tool_result().text
    assert "deliberate tier" in footer
    assert "red-team critique applied" in footer


async def test_the_snapshot_explains_the_last_deliberation():
    engine = _engine(_Router())
    assert engine.snapshot() == {"ran": False}
    await engine.reason("hello")
    snapshot = engine.snapshot()
    assert snapshot["ran"] and snapshot["tier"] == "reflex"


# ──────────────────────────────────────────────────────────────────────────────
# CLAIM EXTRACTION AND VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def test_only_assertions_are_extracted():
    claims = extract_claims(
        "# Heading\n"
        "SQLite is a single-file database engine.\n"
        "Consider using it for this project.\n"
        "The build takes 4 minutes on this machine.\n"
        "What do you think?\n"
        "It might be faster.\n"
    )
    joined = " ".join(claims)
    assert "single-file database" in joined
    assert "4 minutes" in joined
    assert "Consider" not in joined      # an imperative asserts nothing
    assert "What do you" not in joined   # nor does a question
    assert "might" not in joined         # nor does a hedge


def test_code_blocks_are_not_claims():
    claims = extract_claims("Here is the fix.\n```python\nx = 1 is not None\n```\n")
    assert not any("x = 1" in c for c in claims)


def test_claim_extraction_is_bounded():
    text = " ".join(f"Fact number {i} is recorded here." for i in range(40))
    assert len(extract_claims(text, limit=5)) == 5


def test_the_evidence_store_recognises_a_supporting_claim(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    store.record_claim("builds", "The build takes 4 minutes on this machine.",
                       source="measured")
    verdict = store.assess_claim("The build takes 4 minutes on this machine.")
    assert verdict["verdict"] == "supported"
    store.close()


def test_the_evidence_store_catches_a_clashing_figure(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    store.record_claim("builds", "The build takes 4 minutes on this machine.",
                       source="measured")
    verdict = store.assess_claim("The build takes 9 minutes on this machine.")
    assert verdict["verdict"] == "contradicted"
    assert verdict["conflict"][0]["source"] == "measured"
    store.close()


def test_an_unknown_claim_is_unsupported_not_false(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    verdict = store.assess_claim("Saturn has eighty-two confirmed moons.")
    assert verdict["verdict"] == "unsupported"
    store.close()


def test_contradiction_outranks_partial_support(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    store.record_claim("builds", "The build takes 4 minutes on this machine.",
                       source="measured")
    store.record_claim("builds", "The build takes 9 minutes on this machine.",
                       source="rumour")
    verdict = store.assess_claim("The build takes 4 minutes on this machine.")
    assert verdict["verdict"] == "contradicted"
    store.close()


async def test_a_contradiction_is_surfaced_to_the_user(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    store.record_claim("builds", "The build takes 4 minutes on this machine.",
                       source="measured")
    router = _Router(scripts={
        "reason.critic": ["SCORE: 7\nDISAGREEMENT: none"],
        "reason.chair": ["The build takes 9 minutes on this machine."],
    })
    outcome = await _engine(router, evidence=store).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert outcome.verification.ran
    assert outcome.verification.contradicted
    text = outcome.to_tool_result().text
    assert "conflicts with what I have on record" in text
    assert "measured" in text
    store.close()


async def test_an_empty_store_does_not_nag(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    router = _Router(scripts={
        "reason.critic": ["SCORE: 7\nDISAGREEMENT: none"],
        "reason.chair": ["Postgres handles concurrent writers better than SQLite."],
    })
    outcome = await _engine(router, evidence=store).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert outcome.verification.ran and outcome.verification.unsupported
    assert outcome.verification.note() == "", "unknown is not a warning"
    store.close()


async def test_verification_is_skipped_without_an_evidence_store():
    router = _Router(scripts={"reason.chair": ["Answer."],
                              "reason.critic": ["SCORE: 8\nDISAGREEMENT: none"]})
    outcome = await _engine(router).reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    assert not outcome.verification.ran
    assert outcome.ok


# ──────────────────────────────────────────────────────────────────────────────
# WRITING EVIDENCE BACK
# ──────────────────────────────────────────────────────────────────────────────

async def test_only_instrument_grounded_claims_are_recorded(tmp_path):
    """The model's own prose must never become tomorrow's corroborating evidence."""
    store = EvidenceEngine(path=tmp_path / "e.db")

    async def _dispatch(name, _args):
        return ToolResult("Postgres supports 100 concurrent connections by default.")

    router = _Router(
        default='TOOL: query_intelligence\nARGS: {"query": "postgres"}',
        scripts={"reason.critic": ["SCORE: 8\nDISAGREEMENT: none"],
                 "reason.chair": ["Elephants are entirely made of cheese."]})
    engine = _engine(router, evidence=store, dispatch=_dispatch)
    await engine.reason(
        "Should I refactor this Python code for performance, or redesign the "
        "database schema? Compare the trade-offs and explain why.")
    recorded = store.search_claims("Postgres supports concurrent connections", limit=5)
    assert recorded, "an instrument reading should have been recorded"
    assert recorded[0]["source"].startswith("instrument:")
    assert not store.search_claims("Elephants are entirely made of cheese", limit=5), \
        "the chair's prose must never be recorded as evidence"
    store.close()


async def test_nothing_is_recorded_at_the_cheaper_tiers(tmp_path):
    store = EvidenceEngine(path=tmp_path / "e.db")
    await _engine(_Router(), evidence=store).reason("hello")
    assert store.stats()["claims"] == 0
    store.close()


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


def test_the_reason_tool_is_declared_and_routed():
    from orion_core.dispatcher import TOOL_DECLARATIONS
    assert "reason" in _dispatcher().handler_table()
    assert any(t["name"] == "reason" for t in TOOL_DECLARATIONS)


def test_reason_is_not_batched_with_other_tools():
    """Read-only, but its own fan-out makes batching it a rate-limit hazard."""
    assert classify("reason") is ToolClass.SERIAL


async def test_the_reason_tool_degrades_when_the_engine_is_absent():
    dispatcher = _dispatcher()
    result = await dispatcher.dispatch("reason", {"question": "Anything?"})
    assert not result.ok and "not available" in result.text


async def test_the_reason_tool_returns_the_answer_and_its_provenance():
    dispatcher = _dispatcher()
    router = _Router()
    dispatcher.reasoning = _engine(router)
    result = await dispatcher.dispatch("reason", {"question": "hello"})
    assert result.ok
    assert "reflex tier" in result.text


async def test_the_reason_tool_reports_the_previous_deliberation():
    dispatcher = _dispatcher()
    engine = _engine(_Router())
    dispatcher.reasoning = engine
    empty = await dispatcher.dispatch("reason", {"action": "last"})
    assert "not reasoned through anything yet" in empty.text
    await dispatcher.dispatch("reason", {"question": "hello"})
    report = await dispatcher.dispatch("reason", {"action": "last"})
    assert "Tier: reflex" in report.text


async def test_an_explicit_tier_reaches_the_engine_through_the_tool():
    dispatcher = _dispatcher()
    router = _Router(scripts={"reason.critic": ["SCORE: 8\nDISAGREEMENT: none"],
                              "reason.chair": ["Answer."]})
    dispatcher.reasoning = _engine(router)
    result = await dispatcher.dispatch(
        "reason", {"question": "hello", "tier": "deliberate"})
    assert "deliberate tier" in result.text
