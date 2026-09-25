"""
Tool routing quality — the evidence that decides whether the pre-filter is safe.

ORION declares 136 tools. The resolver exists to expose only the handful a turn
needs, and it was switched off and wired to nothing because filtering risks
hiding a tool the model needed and nobody could prove it did not.

Shadow evaluation broke that circle, and the first measurement justified the
caution: **76.5% recall**. Enabling the resolver would have silently removed the
needed tool on about one turn in four. The cause was not the scoring algorithm —
the words people say ("weather", "quiz", "summarise", "yesterday") appeared
nowhere in the tool schema at all. ``tool_vocabulary`` supplies them.

**The held-out half is the number that counts.** These cases are split by index:
the development half may be inspected while tuning, the holdout may not. A
scorer tuned until it passes the cases used to tune it has measured only its own
tuning. Recall on the holdout went 72.7% -> 100%.

The threshold below is deliberately absolute. Recall is a safety property: a
miss is a turn where ORION reaches for a capability that is not there. There is
no token saving that buys that back, so the test demands all of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from data.tool_routing_cases import CASES, dev_cases, holdout_cases  # noqa: E402
from orion_core.dispatch_schema import TOOL_DECLARATIONS  # noqa: E402
from orion_core.resolver_shadow import ShadowEvaluator  # noqa: E402
from orion_core.tool_vocabulary import (  # noqa: E402
    VOCABULARY,
    collisions,
    coverage,
    terms_for,
)

TOOL_NAMES = [t["name"] for t in TOOL_DECLARATIONS]


@pytest.fixture
def evaluator(tmp_path):
    made = ShadowEvaluator(tmp_path / "shadow.db", min_sample=1)
    yield made
    made.close()


def _recall(evaluator, cases) -> tuple[float, list[str]]:
    for query, tool in cases:
        evaluator.observe(query, tool)
    stats = evaluator.stats()
    missed = [f"{m['tool']}: {m['query']}" for m in evaluator.misses(50)]
    return stats["recall"], missed


# ── the evaluation set itself ────────────────────────────────────────────────

def test_every_case_names_a_registered_tool():
    """A case naming a tool that does not exist tests nothing and hides a real
    gap behind a skip."""
    unknown = sorted({t for _, t in CASES if t not in set(TOOL_NAMES)})
    assert not unknown, "evaluation cases name unregistered tools: %s" % unknown


def test_the_split_is_deterministic_and_balanced():
    dev, hold = dev_cases(), holdout_cases()
    assert len(dev) + len(hold) == len(CASES)
    assert abs(len(dev) - len(hold)) <= 1
    assert dev_cases() == dev, "the split must not vary between calls"
    assert not (set(dev) & set(hold)), "dev and holdout overlap"


def test_the_set_is_large_and_broad_enough_to_mean_something():
    assert len(CASES) >= 60
    assert len({t for _, t in CASES}) >= 25, "too few distinct tools covered"


# ── recall: the safety property ──────────────────────────────────────────────

def test_recall_on_the_development_half_is_perfect(evaluator):
    recall, missed = _recall(evaluator, dev_cases())
    assert recall == 1.0, "dev recall %.1f%%; lost: %s" % (recall * 100, missed)


def test_recall_on_the_held_out_half_is_perfect(evaluator):
    """The honest score — these cases were never inspected while tuning."""
    recall, missed = _recall(evaluator, holdout_cases())
    assert recall == 1.0, (
        "HOLDOUT recall %.1f%% — the resolver would remove a tool ORION needs "
        "on these turns: %s" % (recall * 100, missed))


def test_the_filter_actually_filters(evaluator):
    """Perfect recall is trivial if nothing is removed; the saving must be real."""
    _recall(evaluator, CASES)
    stats = evaluator.stats()
    assert stats["mean_selected"] < stats["mean_total"] * 0.5, (
        "the resolver exposes %.0f of %.0f tools — it is barely filtering"
        % (stats["mean_selected"], stats["mean_total"]))


def test_the_needed_tool_ranks_near_the_top(evaluator):
    """Recall says the tool survives; rank says the scoring is actually sane
    rather than accidentally scraping in at the cut-off."""
    _recall(evaluator, CASES)
    assert evaluator.stats()["mean_rank"] < 12


# ── the vocabulary ───────────────────────────────────────────────────────────

def test_no_vocabulary_term_is_another_tools_name():
    """The regression that made 'study' rank the research tool first.

    Vocabulary carries a tool's own name weight, so a term that IS another
    tool's name lets one capability outrank the tool being asked for. Twelve of
    these existed on the first draft.
    """
    found = collisions(TOOL_NAMES)
    assert not found, (
        "vocabulary terms that steal another tool's name: "
        + ", ".join("%s contains %r" % (t, w) for t, w in found))


def test_a_literal_tool_name_still_ranks_that_tool_first():
    from orion_core.tool_resolver import lexical_scores
    for name in ("study", "focus", "chess", "finance", "research", "briefing",
                 "decision", "workflow", "protocol", "transcript"):
        if name not in set(TOOL_NAMES):
            continue
        scores = lexical_scores(name, TOOL_DECLARATIONS)
        winner = max(scores, key=lambda k: scores[k])
        assert winner == name, "query %r ranked %r first" % (name, winner)


def test_vocabulary_only_names_real_tools():
    """A stale entry for a renamed tool is dead weight that looks like coverage."""
    unknown = sorted(set(VOCABULARY) - set(TOOL_NAMES))
    assert not unknown, "vocabulary for tools that no longer exist: %s" % unknown


def test_every_tool_has_invoking_vocabulary():
    """Full coverage, and deliberately a forcing function: a tool added without
    vocabulary is a tool the resolver is most likely to filter away, so adding
    one should fail here until someone writes the words people would use."""
    have, total, missing = coverage(TOOL_NAMES)
    assert not missing, (
        "%d of %d tools have no invoking vocabulary: %s"
        % (total - have, total, ", ".join(missing)))


def test_vocabulary_entries_are_plain_lowercase_words():
    for tool, phrase in VOCABULARY.items():
        assert phrase == phrase.lower(), tool
        assert phrase.strip() == phrase and "  " not in phrase, (
            "%s has ragged whitespace — a regex edit probably mangled it" % tool)
        for word in phrase.split():
            assert len(word) < 18, (
                "%s contains %r — two phrases were concatenated without a space"
                % (tool, word))


def test_terms_for_an_unknown_tool_is_empty():
    assert terms_for("no_such_tool") == ""


def test_the_expansion_is_never_sent_to_the_model():
    """Its whole justification is that it costs zero tokens.

    Stated as the invariant that actually matters: building the resolver index
    must not mutate TOOL_DECLARATIONS. (A substring check on the schema JSON is
    the wrong test — plenty of short vocabulary words legitimately occur inside
    longer words in real descriptions.)
    """
    import json

    from orion_core import tool_resolver as tr
    before = json.dumps(TOOL_DECLARATIONS, sort_keys=True)
    tr.reset_index()
    tr.lexical_scores("quiz me on the weather while I revise", TOOL_DECLARATIONS)
    after = json.dumps(TOOL_DECLARATIONS, sort_keys=True)
    assert before == after, "the resolver mutated the schema sent to the provider"


def test_the_vocabulary_adds_terms_the_schema_does_not_have():
    """If every vocabulary word were already in the schema it would be earning
    nothing — the whole diagnosis was that the words were missing."""
    from orion_core.tool_resolver import _tokens
    schema_terms = set(_tokens(" ".join(
        t["name"] + " " + t.get("description", "") for t in TOOL_DECLARATIONS)))
    vocab_terms = {w for phrase in VOCABULARY.values() for w in phrase.split()}
    novel = vocab_terms - schema_terms
    assert len(novel) >= 30, (
        "vocabulary adds only %d terms the schema lacks" % len(novel))
    # the four that started this investigation
    for word in ("weather", "quiz", "summarise", "yesterday"):
        assert word in vocab_terms, word


def test_the_resolver_degrades_if_the_vocabulary_is_missing(monkeypatch):
    """A missing module must cost recall, never routing itself."""
    from orion_core import tool_resolver as tr
    monkeypatch.setattr(tr, "vocabulary_for", lambda _t: "")
    tr.reset_index()
    try:
        scores = tr.lexical_scores("what's the weather", TOOL_DECLARATIONS)
        assert len(scores) == len(TOOL_DECLARATIONS)
    finally:
        tr.reset_index()


# ── the shadow evaluator ─────────────────────────────────────────────────────

def test_an_unknown_tool_is_not_scored(evaluator):
    """Forged, MCP and plugin tools were never shown to the resolver. Counting
    them as misses would punish it for something it cannot see."""
    assert evaluator.observe("do the thing", "a_tool_that_does_not_exist") is None
    assert evaluator.stats()["turns"] == 0


def test_empty_input_is_not_scored(evaluator):
    assert evaluator.observe("", "study") is None
    assert evaluator.observe("what's due", "") is None


def test_an_outcome_records_rank_and_saving(evaluator):
    out = evaluator.observe("quiz me on neuroscience", "study")
    assert out is not None and out.kept
    assert out.rank is not None and out.rank >= 0
    assert 0.0 < out.saving < 1.0


def test_a_miss_records_no_rank():
    from orion_core.resolver_shadow import ShadowOutcome
    out = ShadowOutcome(query="q", tool="t", kept=False, rank=None,
                        selected=25, total=136)
    assert out.rank is None
    assert abs(out.saving - (1 - 25 / 136)) < 1e-9


def test_queries_are_truncated_before_storage(evaluator):
    from orion_core.resolver_shadow import MAX_QUERY_CHARS
    out = evaluator.observe("x " * 200 + "study", "study")
    if out is not None:
        assert len(out.query) <= MAX_QUERY_CHARS, (
            "this is routing telemetry, not a transcript")


def test_a_resolver_fault_never_disturbs_a_turn(evaluator, monkeypatch):
    from orion_core import tool_resolver as tr

    def boom(*_a, **_kw):
        raise RuntimeError("resolver exploded")
    monkeypatch.setattr(tr, "resolve", boom)
    assert evaluator.observe("what's due", "study") is None


def test_the_verdict_refuses_to_judge_a_small_sample(tmp_path):
    ev = ShadowEvaluator(tmp_path / "s.db", min_sample=200)
    try:
        for query, tool in CASES:
            ev.observe(query, tool)
        verdict = ev.verdict()
        assert verdict.startswith("NOT YET"), verdict
        assert "200" in verdict, "the verdict must say what sample is needed"
    finally:
        ev.close()


def test_the_verdict_reports_unsafe_when_recall_is_imperfect(tmp_path):
    ev = ShadowEvaluator(tmp_path / "s.db", min_sample=1)
    try:
        ev.observe("what's due for review", "study")
        # a query with nothing to do with the tool it claims
        ev.observe("aaaa bbbb cccc dddd", "chess")
        verdict = ev.verdict()
        if ev.stats()["missed"]:
            assert verdict.startswith("NOT SAFE"), verdict
            assert "chess" in verdict, "a verdict must name the offender"
    finally:
        ev.close()


def test_the_verdict_reports_safe_only_on_a_clean_sample(tmp_path):
    ev = ShadowEvaluator(tmp_path / "s.db", min_sample=len(CASES))
    try:
        for query, tool in CASES:
            ev.observe(query, tool)
        assert ev.ready() is True
        verdict = ev.verdict()
        assert verdict.startswith("SAFE TO ENABLE"), verdict
        assert "ORION_TOOL_RESOLVER" in verdict, "say how to act on it"
    finally:
        ev.close()


def test_a_recall_figure_never_appears_without_its_sample_size(tmp_path):
    ev = ShadowEvaluator(tmp_path / "s.db", min_sample=1)
    try:
        for query, tool in CASES[:12]:
            ev.observe(query, tool)
        report = ev.report()
        assert "turns observed" in report and "recall" in report
    finally:
        ev.close()


def test_an_empty_evaluator_reports_honestly(tmp_path):
    ev = ShadowEvaluator(tmp_path / "s.db")
    try:
        assert "no turns observed" in ev.report().lower()
        assert ev.ready() is False
    finally:
        ev.close()


def test_evidence_survives_a_restart(tmp_path):
    path = tmp_path / "s.db"
    first = ShadowEvaluator(path, min_sample=1)
    first.observe("quiz me on neuroscience", "study")
    first.close()
    second = ShadowEvaluator(path, min_sample=1)
    try:
        assert second.stats()["turns"] == 1
    finally:
        second.close()


def test_clear_discards_evidence_for_a_changed_algorithm(evaluator):
    evaluator.observe("quiz me on neuroscience", "study")
    evaluator.clear()
    assert evaluator.stats()["turns"] == 0


def test_the_store_uses_the_projects_sqlite_policy(evaluator):
    from orion_core.db import journal_mode
    assert journal_mode(evaluator._db).lower() == "wal"


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_worker_shadow_evaluates_every_tool_used():
    src = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
    assert "observe_resolver_shadow" in src
    assert "for used in names:" in src, (
        "recall is per-tool; a multi-tool turn must score each one")


def test_shadow_observation_never_breaks_a_turn():
    import inspect
    from orion_core.live_worker import GenAILiveWorker
    src = inspect.getsource(GenAILiveWorker.observe_resolver_shadow)
    assert "except Exception" in src


def test_the_diagnostics_tool_reports_routing():
    src = (ROOT / "orion_core" / "dispatch_files.py").read_text(encoding="utf-8")
    assert '"routing"' in src and "ShadowEvaluator" in src


def test_the_schema_advertises_routing():
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "diagnostics")
    assert "routing" in tool["description"]


def test_the_resolver_is_still_off_by_default(monkeypatch):
    """Nothing in this pass may switch it on. The evidence decides that, and
    the evidence has to come from real turns, not from this test file."""
    monkeypatch.delenv("ORION_TOOL_RESOLVER", raising=False)
    from orion_core.tool_resolver import resolver_enabled
    assert resolver_enabled() is False


def test_the_seam_returns_everything_while_the_flag_is_off(monkeypatch):
    monkeypatch.delenv("ORION_TOOL_RESOLVER", raising=False)
    from orion_core.dispatcher import OrionDispatcher
    seam = OrionDispatcher.resolve_tool_declarations

    class _Stub:
        TOOL_DECLARATIONS = list(TOOL_DECLARATIONS)
    assert len(seam(_Stub(), "anything")) == len(TOOL_DECLARATIONS)
