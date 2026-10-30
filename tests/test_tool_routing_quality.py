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
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from data.tool_routing_cases import CASES, dev_cases, holdout_cases  # noqa: E402
from orion_core.dispatch_schema import TOOL_DECLARATIONS  # noqa: E402
from orion_core import intent_brain as _intent_lifecycle_module  # noqa: E402
from orion_core.resolver_shadow import ShadowEvaluator  # noqa: E402
from orion_core.tool_vocabulary import (  # noqa: E402
    VOCABULARY,
    collisions,
    coverage,
    terms_for,
)

TOOL_NAMES = [t["name"] for t in TOOL_DECLARATIONS]
# conftest disables background learning per test; retain the production entry
# point at collection for the explicit, fully fake lifecycle tests below.
_REAL_INTENT_OBSERVE_ASYNC = _intent_lifecycle_module.observe_async


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
    # Mark XXXVII: every call is scored as the tool that RAN (use_tool's inner
    # tool, never find_tool) - behaviour in test_mark37_voice_loop.py.
    assert "for index, (tool, _params) in enumerate(effective):" in src, (
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


# ── words that are not evidence ──────────────────────────────────────────────

def test_filler_words_do_not_decide_the_tool():
    """"how much ... left" is how token_usage's vocabulary describes usage;
    while those words counted, a question about cash ranked token_usage
    first."""
    from orion_core import tool_resolver as tr
    scores = tr.lexical_scores("how much cash have I got left", TOOL_DECLARATIONS)
    assert max(scores, key=scores.get) == "finance"


def test_a_plural_is_its_singular():
    from orion_core import tool_resolver as tr
    assert tr._tokens("emails helicopters") == tr._tokens("email helicopter")
    scores = tr.lexical_scores("any helicopters about", TOOL_DECLARATIONS)
    assert max(scores, key=scores.get) == "aviation"


# ── the offline routing gate: 394 everyday requests (Mark XXXVII) ────────────
#
# tests/data/intent_routing_cases.py was used by no test, so a vocabulary or
# description edit could quietly route worse and nothing failed. This scores
# the words-and-vocabulary ranker alone (tool_resolver.lexical_scores: no
# encoder, no intent network, no spelling model - deterministic and offline,
# exactly what ORION routes by while the encoder is cold) and fails if top-3
# drops below what it achieves today. Measured 23 Oct 2026:
#
#     before this pass   all 82.49%  dev 82.74%  holdout 82.23%   (top-1 65.99%)
#     after              all 82.99%  dev 82.74%  holdout 83.25%   (top-1 66.50%)
#
# The floors sit just under those figures. Raise them when routing improves;
# never lower them to make an edit pass.

ROUTING_TOP3_FLOOR = {"all": 0.825, "dev": 0.825, "holdout": 0.830}
ROUTING_TOP1_FLOOR = 0.66


def _lexical_rank(cases):
    from data.intent_routing_cases import acceptable
    from orion_core.tool_resolver import lexical_scores
    menu = [d for d in TOOL_DECLARATIONS if d["name"] not in {"find_tool", "use_tool"}]
    top1 = top3 = 0
    missed = []
    for case in cases:
        scores = lexical_scores(case[0], menu)
        ranked = sorted(scores, key=lambda name: (-scores[name], name))
        right = acceptable(case)
        if scores[ranked[0]] > 0 and ranked[0] in right:
            top1 += 1
        if any(scores[name] > 0 and name in right for name in ranked[:3]):
            top3 += 1
        else:
            missed.append(f"{case[1]}: {case[0]} (got {ranked[:3]})")
    return top1 / len(cases), top3 / len(cases), missed


@pytest.mark.parametrize("half", ["all", "dev", "holdout"])
def test_offline_routing_does_not_regress_on_everyday_requests(half, monkeypatch):
    from data import intent_routing_cases as everyday
    from orion_core import tool_resolver
    # Encoder-cold by construction: lexical_scores never consults meaning,
    # and this proves it — reaching for the semantic scorer fails the test.
    monkeypatch.setattr(tool_resolver, "semantic_scores",
                        lambda *_a, **_k: pytest.fail("the offline gate used the encoder"))
    cases = {"all": everyday.CASES, "dev": everyday.dev_cases(),
             "holdout": everyday.holdout_cases()}[half]
    assert len(everyday.CASES) == 394, "the evaluation set changed: re-measure the floors"
    top1, top3, missed = _lexical_rank(cases)
    assert top3 >= ROUTING_TOP3_FLOOR[half], (
        f"{half} top-3 {top3:.2%} < {ROUTING_TOP3_FLOOR[half]:.1%}; first misses: "
        + "; ".join(missed[:12]))
    if half == "all":
        assert top1 >= ROUTING_TOP1_FLOOR, f"top-1 {top1:.2%} < {ROUTING_TOP1_FLOOR:.0%}"


@pytest.mark.parametrize("request_text,tool", [
    ("what changed in your last update", "self_changes"),
    ("what are your latest updates", "patch_notes"),
    ("read me the patch notes", "patch_notes"),
    ("sing me a song", "sing"),
    ("any events at my university this week", "web_search"),
    ("close that window", "window_control"),
    ("i've done all those tasks", "awareness"),
])
def test_the_owners_misrouted_requests_now_rank_their_tool_first(request_text, tool):
    """From the owner's sessions, Oct 2026: each of these reached the wrong
    tool (resource_status, notion_workspace) or none."""
    from orion_core.tool_resolver import lexical_scores
    menu = [d for d in TOOL_DECLARATIONS if d["name"] not in {"find_tool", "use_tool"}]
    scores = lexical_scores(request_text, menu)
    ranked = sorted(scores, key=lambda name: (-scores[name], name))
    assert ranked[0] == tool, ranked[:3]


# ── the learned intent network (intent_brain) ────────────────────────────────

_TOY = {
    "geo": "weather forecast rain temperature",
    "finance": "money spend spent budget cash runway",
    "reminder": "remind alarm alert wake me",
    "media_control": "play pause skip next track music",
}
_TOY_TEXT = {
    "geo": "Weather and places. Use for 'will it rain tomorrow'.",
    "finance": "A money ledger. Use for 'I spent 30 on ads'.",
    "reminder": "Set spoken reminders and alarms.",
    "media_control": "Control music playback and volume.",
}


def _toy_brain(folder, epochs=6):
    from orion_core import intent_brain
    made = intent_brain.IntentBrain(directory=folder)
    made.seed(_TOY, _TOY_TEXT, epochs=epochs)
    return made


def test_the_sparse_network_learns_and_round_trips(tmp_path):
    import numpy as np
    from orion_core.neural import SparseMLP
    net = SparseMLP(64, 8, 3, seed=1)
    for _ in range(40):
        net.train_batch([[1, 2, 3], [10, 11], [40]], [0, 1, 2])
    assert [int(net.predict(r).argmax()) for r in ([1, 2, 3], [10, 11], [40])] == [0, 1, 2]
    assert net.save(tmp_path / "n.npz")
    again, _ = SparseMLP.load(tmp_path / "n.npz")
    assert np.allclose(again.predict([10, 11]), net.predict([10, 11]))


def test_the_intent_network_learns_the_tools_own_phrases(tmp_path):
    brain = _toy_brain(tmp_path / "brain")
    assert brain.top("is it going to rain later", 1)[0][0] == "geo"
    assert brain.top("pause the music a sec", 1)[0][0] == "media_control"


def test_letter_trigrams_forgive_a_slip():
    from orion_core.intent_brain import features
    # Words alone would share only "my"; stems and trigrams share the rest.
    assert len(set(features("check my emails")) & set(features("chek my emials"))) >= 4


def test_the_network_saves_into_a_folder_that_does_not_exist_yet(tmp_path):
    """atomic_write_bytes will not create a folder and nothing created
    CONFIG_DIR/brain, so the network never saved: re-seeded on every start,
    forgetting everything it had learned."""
    from orion_core import intent_brain
    folder = tmp_path / "not" / "yet"
    brain = _toy_brain(folder)
    brain.learn_now("what's it doing outside", "geo")
    assert brain.save()
    again = intent_brain.IntentBrain(directory=folder)
    assert again.seeded and again.labels == brain.labels and again.observed == 1
    assert again.top("is it going to rain later", 1)[0][0] == "geo"


def test_a_restart_refills_the_replay_memory(tmp_path):
    """Without the tools' phrases back in the replay, online learning after a
    restart sampled only the user's requests and drifted towards them."""
    from orion_core import intent_brain
    brain = _toy_brain(tmp_path / "b")
    brain.save()
    again = intent_brain.IntentBrain(directory=tmp_path / "b")
    assert len(again.replay) == 0
    again.restore_corpus(_TOY, _TOY_TEXT)
    assert len(again.replay) == len(brain.replay)


def test_restoring_never_hands_a_new_tool_an_untrained_output(tmp_path):
    from orion_core import intent_brain
    _toy_brain(tmp_path / "b").save()
    again = intent_brain.IntentBrain(directory=tmp_path / "b")
    again.restore_corpus({**_TOY, "aviation": "plane overhead aircraft"}, _TOY_TEXT)
    assert "aviation" not in again.labels      # left for ensure_known to teach


def test_learning_since_the_last_save_is_flushed(tmp_path):
    from orion_core import intent_brain
    brain = _toy_brain(tmp_path / "b")
    brain.save()
    brain.learn_now("wake me at seven", "reminder")         # below SAVE_EVERY
    assert intent_brain.IntentBrain(directory=tmp_path / "b").observed == 0
    brain.flush()
    assert intent_brain.IntentBrain(directory=tmp_path / "b").observed == 1


def test_the_previous_networks_history_is_carried_across(tmp_path):
    from orion_core import intent_brain
    from orion_core.neural import MLP
    folder = tmp_path / "b"
    folder.mkdir()
    MLP([4096, 64, 256], output="softmax").save(folder / intent_brain.LEGACY_FILE, {
        "labels": ["geo"], "observed": 3, "seeded": True, "user_counts": {"finance": 3},
        "recent": [["am I skint", "finance"]] * 3})
    brain = intent_brain.IntentBrain(directory=folder)
    assert brain.observed == 3 and brain.user_counts == {"finance": 3} and not brain.seeded
    brain.seed(_TOY, _TOY_TEXT, epochs=6)
    assert brain.top("am I skint", 1)[0][0] == "finance"


def test_a_tool_it_has_never_seen_is_queued_then_learned(tmp_path, monkeypatch):
    # A closed four-tool world: the real situational phrases (intent_situations)
    # would triple these tiny classes and swamp one learned tool. At real scale
    # (15 tools withheld, 3 seeds) they left new-tool top-1 level: 75.3 -> 74.1.
    monkeypatch.setattr("orion_core.intent_situations.SITUATIONS", {})
    brain = _toy_brain(tmp_path / "b")
    jobs = []
    monkeypatch.setattr(brain, "_submit", jobs.append)
    declarations = ([{"name": n, "description": _TOY_TEXT[n]} for n in _TOY]
                    + [{"name": "aviation", "description": "Live aircraft overhead."}])
    brain.ensure_known(declarations)
    brain.ensure_known(declarations)          # unchanged schema: nothing more to queue
    assert [job[0] for job in jobs] == ["tools"]
    assert [d["name"] for d in jobs[0][1]] == ["aviation"]
    assert brain.learn_tools(jobs[0][1], epochs=6) > 0
    assert brain.top("what aircraft is overhead", 1)[0][0] == "aviation"
    assert brain.top("is it going to rain later", 1)[0][0] == "geo"


def test_routing_never_builds_the_network_on_the_callers_thread(monkeypatch):
    from orion_core import intent_brain
    from orion_core import tool_resolver as tr
    monkeypatch.setattr(intent_brain, "_BRAIN", None)

    def _refuse():
        raise AssertionError("routing built the network on the caller's thread")
    monkeypatch.setattr(intent_brain, "brain", _refuse)
    assert (intent_brain.routing_scores("pause the music", TOOL_DECLARATIONS)
            == tr.hybrid_scores("pause the music", TOOL_DECLARATIONS))


@pytest.mark.parametrize("path", ["seed", "new_tools", "online"])
def test_all_intent_learning_paths_wait_for_quiet_before_every_batch(tmp_path, monkeypatch, path):
    """Learning a successful spoken request must not train through the reply.

    Each synthetic batch makes the conversation busy again, so checking only
    once (or after the first batch) cannot satisfy this test.
    """
    import numpy as np

    from orion_core import intent_brain

    brain = intent_brain.IntentBrain(directory=tmp_path / "brain")
    brain._label("geo")
    state = {"quiet": False, "waits": 0, "batches": 0}
    monkeypatch.setattr(intent_brain, "_QUIET", [lambda: state["quiet"]])

    def sleep(seconds):
        if seconds:
            state["waits"] += 1
            state["quiet"] = True

    def train(*args, **kwargs):
        assert state["quiet"], "a batch began while ORION was speaking/listening"
        state["batches"] += 1
        state["quiet"] = False

    monkeypatch.setattr(intent_brain.time, "sleep", sleep)
    monkeypatch.setattr(brain.net, "train_batch", train)
    monkeypatch.setattr(brain, "save", lambda: True)
    tool = "new_tool" if path == "new_tools" else "geo"
    corpus = [(np.array([i + 1], dtype=np.int16), tool) for i in range(80)]
    monkeypatch.setattr(brain, "_corpus_from", lambda *args: corpus)

    if path == "seed":
        brain.seed({"geo": "weather forecast"}, epochs=1)
        assert brain.seeded
    elif path == "new_tools":
        assert brain.learn_tools([{"name": "new_tool", "description": "new capability"}], epochs=1)
        assert "new_tool" in brain.labels
    else:
        brain.learn_now("weather forecast please", "geo", steps=3)
        assert brain.observed == 1
        assert brain.user_counts["geo"] == 1
    assert state["batches"] >= 2
    assert state["waits"] == state["batches"]


def test_routing_uses_the_network_once_it_is_built(tmp_path, monkeypatch):
    from orion_core import intent_brain
    from orion_core import tool_resolver as tr
    brain = _toy_brain(tmp_path / "b")
    monkeypatch.setattr(intent_brain, "_BRAIN", brain)
    monkeypatch.setattr(brain, "_submit", lambda job: None)
    plain = tr.hybrid_scores("will it rain", TOOL_DECLARATIONS)
    learned = intent_brain.routing_scores("will it rain", TOOL_DECLARATIONS)
    assert learned["geo"] > plain["geo"]


def test_waiting_lessons_are_bounded_and_the_oldest_gives_way(tmp_path):
    """Both learner queues were unbounded, and the first build (seconds of
    seeding) is exactly when lessons pile up. A lesson is a replaceable
    sample, so past the bound the OLDEST gives way; the caller — the voice
    loop — is never blocked."""
    import time as _t
    from types import SimpleNamespace

    from orion_core import intent_brain
    brain = intent_brain.IntentBrain(directory=tmp_path / "b")
    brain._worker = SimpleNamespace(is_alive=lambda: True)   # no learner thread
    started = _t.perf_counter()
    for index in range(intent_brain.MAX_PENDING_LESSONS + 40):
        brain.observe(f"lesson {index}", "geo")
    assert _t.perf_counter() - started < 1.0
    assert brain._pending.qsize() == intent_brain.MAX_PENDING_LESSONS
    assert brain.dropped == 40
    assert brain._pending.get_nowait()[1] == "lesson 40", "the oldest went first"


def test_a_dropped_new_tools_job_is_queued_again_later(tmp_path):
    """New tools are not a replaceable sample: if one is pushed out, the next
    ensure_known() must queue it again rather than believe it was learned."""
    from types import SimpleNamespace

    from orion_core import intent_brain
    brain = intent_brain.IntentBrain(directory=tmp_path / "b")
    brain._worker = SimpleNamespace(is_alive=lambda: True)
    declarations = [{"name": "brand_new_tool", "description": "does a new thing"}]
    brain.ensure_known(declarations)
    assert brain._known_key is not None
    for index in range(intent_brain.MAX_PENDING_LESSONS):
        brain.observe(f"lesson {index}", "geo")
    assert brain._known_key is None, "the dropped tools job must be re-queued"
    brain.ensure_known(declarations)
    queued = [brain._pending.get_nowait() for _ in range(brain._pending.qsize())]
    assert queued[-1][0] == "tools"


# ── the text path's menu ─────────────────────────────────────────────────────

def test_the_text_menu_ranks_the_request_not_the_memory_note():
    """With a three-fact memory note on the request, the right tool was on the
    menu for 66.5% of 376 everyday requests against 92.3% without one."""
    from types import SimpleNamespace

    from orion_core import text_tools
    from orion_core.memory import MemoryAgent
    stub = SimpleNamespace(relevant=lambda text, limit=3: [
        {"value": "Sam is allergic to ibuprofen."}, {"value": "The user supports Aston Villa."}])
    ask = "pause the music a sec"
    note = MemoryAgent.relevant_note(stub, ask)
    assert note and text_tools.routing_text(ask + "\n\n" + note) == ask
    assert (text_tools.choose_tools(ask + "\n\n" + note, TOOL_DECLARATIONS)
            == text_tools.choose_tools(ask, TOOL_DECLARATIONS))
    assert text_tools.routing_text("no note here") == "no note here"


def test_a_tool_is_as_close_as_its_nearest_example_request(monkeypatch):
    """An indirect request is far from a tool's whole description but close to
    one of the everyday requests it is known by (intent_situations.py)."""
    import re as _re
    import zlib

    import numpy as np

    from orion_core import semantic
    from orion_core import tool_resolver as tr

    class _Words:                       # bag-of-words stand-in for the encoder
        def ready(self):
            return True

        def _vec(self, text):
            v = np.zeros(64)
            for w in _re.findall(r"[a-z]+", text.lower()):
                v[zlib.crc32(w.encode()) % 64] += 1
            return v / (np.linalg.norm(v) or 1.0)

        def encode(self, texts):
            return np.vstack([self._vec(t) for t in texts])

        def encode_one(self, text):
            return self._vec(text)

    monkeypatch.setattr(semantic, "ENCODER", _Words())
    for cache in ("_TOOL_VECTORS", "_TOOL_MATRIX", "_EXAMPLE_VECTORS", "_EXAMPLE_MATRIX"):
        monkeypatch.setattr(tr, cache, {})
    decls = [{"name": "resource_status", "description": "CPU, memory and disk usage."},
             {"name": "geo", "description": "Weather and places, e.g. 'will it rain tomorrow'."}]
    monkeypatch.setattr(tr, "_warm_in_background", lambda _d: None)
    # Nothing embedded yet, on the GUI thread: words only, nothing embedded here.
    assert tr.semantic_scores("my pc is running slow", decls) == {}
    assert tr._TOOL_VECTORS == {} and tr._EXAMPLE_MATRIX == {}
    tr._tool_matrix(decls)              # warm_semantic's job, on a worker in the app
    tr._example_matrix(decls)
    with_examples = tr.semantic_scores("my pc is running slow", decls)
    assert with_examples["resource_status"] == pytest.approx(1.0)
    assert tr.semantic_scores("will it rain tomorrow", decls)["geo"] == pytest.approx(1.0)
    # On the main thread nothing is embedded: until the worker has built
    # them, the documents alone answer.
    monkeypatch.setattr(tr, "_EXAMPLE_MATRIX", {})
    assert tr.semantic_scores("my pc is running slow", decls)["resource_status"] < 0.9
    assert tr._EXAMPLE_MATRIX == {}


class _IntentImmediateTimeoutQueue(Queue):
    """Exercise the idle-exit branch without a 30/60-second real wait."""

    def get(self, block=True, timeout=None):
        return super().get(block=False)


def _intent_lifecycle_harness(monkeypatch, kind, pending, learn, flush):
    """Use actual lifecycle methods with no constructor, model or checkpoint."""
    module = _intent_lifecycle_module
    if kind == "global":
        fake = SimpleNamespace(learn_now=learn, flush=flush)
        monkeypatch.setattr(module, "_QUEUE", pending)
        monkeypatch.setattr(module, "_BRAIN", fake)
        monkeypatch.setattr(module, "brain", lambda: fake)
        monkeypatch.setattr(module, "_DROPPED", [0])
        monkeypatch.setattr(module, "_FEEDER_LOCK", Lock())
        monkeypatch.setattr(module, "_FEEDER", None)
        return SimpleNamespace(
            target=module._feed,
            submit=lambda text: _REAL_INTENT_OBSERVE_ASYNC(text, "audit_tool"),
            owner=lambda: module._FEEDER,
            assign=lambda worker: setattr(module, "_FEEDER", worker),
            set_lock=lambda lock: setattr(module, "_FEEDER_LOCK", lock),
        )
    brain = module.IntentBrain.__new__(module.IntentBrain)
    brain._pending = pending
    brain._worker = None
    brain._worker_lock = Lock()
    brain.dropped = 0
    brain._known_key = None
    brain.learn_now = learn
    brain.learn_tools = lambda *_args: None
    brain.flush = flush
    return SimpleNamespace(
        target=brain._learn_loop,
        submit=lambda text: brain._submit(("learn", text, "audit_tool")),
        owner=lambda: brain._worker,
        assign=lambda worker: setattr(brain, "_worker", worker),
        set_lock=lambda lock: setattr(brain, "_worker_lock", lock),
    )


def _intent_guarded(errors, work):
    def guarded():
        try:
            work()
        except BaseException as exc:
            errors.append(repr(exc))
    return guarded


@pytest.mark.parametrize("kind", ["global", "instance"])
def test_intent_idle_flush_hands_a_new_lesson_to_a_new_owner_without_clobbering_it(monkeypatch, kind):
    pending = _IntentImmediateTimeoutQueue()
    flushing, allow_exit, learning, allow_learn = Event(), Event(), Event(), Event()
    errors, learned = [], []
    retiring = replacement = None

    def flush():
        if current_thread() is retiring:
            flushing.set()
            if not allow_exit.wait(5):
                raise RuntimeError("retiring intent flush timed out")

    def learn(*lesson):
        learned.append(lesson)
        learning.set()
        if not allow_learn.wait(5):
            raise RuntimeError("replacement intent learning timed out")

    harness = _intent_lifecycle_harness(monkeypatch, kind, pending, learn, flush)
    retiring = Thread(target=_intent_guarded(errors, harness.target), daemon=True)
    harness.assign(retiring)
    retiring.start()
    try:
        assert flushing.wait(5)
        assert retiring.is_alive()
        # Release ownership before slow persistence; the caller must be able
        # to start a consumer while the old thread is still saving.
        assert harness.owner() is None
        harness.submit("lesson arriving during idle flush")
        assert learning.wait(5)
        replacement = harness.owner()
        assert replacement is not None and replacement is not retiring
        assert replacement.is_alive()
        allow_exit.set()
        retiring.join(5)
        assert not retiring.is_alive()
        assert harness.owner() is replacement, "Retiring worker overwrote its successor"
        allow_learn.set()
        replacement.join(5)
        assert not replacement.is_alive() and errors == []
        assert pending.empty()
        assert learned == [("lesson arriving during idle flush", "audit_tool")]
        assert harness.owner() is None
    finally:
        allow_exit.set()
        allow_learn.set()
        retiring.join(5)
        if replacement is not None:
            replacement.join(5)
        remaining = harness.owner()
        if remaining is not None and remaining is not retiring and remaining is not replacement:
            remaining.join(5)


@pytest.mark.parametrize("kind", ["global", "instance"])
def test_intent_idle_timeout_rechecks_a_lesson_queued_before_retirement(monkeypatch, kind):
    timed_out, allow_recheck = Event(), Event()
    errors, learned = [], []

    class TimeoutBoundaryQueue(_IntentImmediateTimeoutQueue):
        armed = True

        def get(self, block=True, timeout=None):
            if self.armed:
                self.armed = False
                timed_out.set()
                if not allow_recheck.wait(5):
                    raise RuntimeError("intent timeout boundary timed out")
                raise Empty
            return super().get(block=block, timeout=timeout)

    pending = TimeoutBoundaryQueue()
    harness = _intent_lifecycle_harness(monkeypatch, kind, pending,
                                        lambda *lesson: learned.append(lesson), lambda: None)
    worker = Thread(target=_intent_guarded(errors, harness.target), daemon=True)
    harness.assign(worker)
    worker.start()
    try:
        assert timed_out.wait(5)
        harness.submit("lesson queued before the empty recheck")
        assert harness.owner() is worker and pending.qsize() == 1
        allow_recheck.set()
        worker.join(5)
        assert not worker.is_alive() and errors == []
        assert learned == [("lesson queued before the empty recheck", "audit_tool")]
        assert pending.empty() and harness.owner() is None
    finally:
        allow_recheck.set()
        worker.join(5)
        remaining = harness.owner()
        if remaining is not None and remaining is not worker:
            remaining.join(5)


@pytest.mark.parametrize("kind", ["global", "instance"])
def test_concurrent_intent_producers_create_one_owner_while_worker_start_is_pending(monkeypatch, kind):
    pending = Queue()
    starting, allow_start, second_at_lock = Event(), Event(), Event()
    errors, created = [], []
    harness = _intent_lifecycle_harness(monkeypatch, kind, pending, lambda *_args: None, lambda: None)

    class ObservedLifecycleLock:
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if current_thread().name == "intent-second-producer":
                second_at_lock.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

    class PendingWorker:
        def __init__(self, *args, **kwargs):
            self.alive = False
            created.append(self)

        def start(self):
            starting.set()
            if not allow_start.wait(5):
                raise RuntimeError("fake intent worker start timed out")
            self.alive = True

        def is_alive(self):
            return self.alive

    harness.set_lock(ObservedLifecycleLock())
    # Replace only this module's threading reference; test orchestration uses
    # the real Thread imported above and no real learner is launched.
    monkeypatch.setattr(_intent_lifecycle_module, "threading", SimpleNamespace(Thread=PendingWorker))
    producers = [Thread(target=_intent_guarded(errors, lambda: harness.submit("first lesson")),
                        name="intent-first-producer", daemon=True),
                 Thread(target=_intent_guarded(errors, lambda: harness.submit("second lesson")),
                        name="intent-second-producer", daemon=True)]
    producers[0].start()
    try:
        assert starting.wait(5)
        producers[1].start()
        assert second_at_lock.wait(5)
        assert len(created) == 1 and pending.qsize() == 1
        allow_start.set()
        for producer in producers:
            producer.join(5)
        assert all(not producer.is_alive() for producer in producers)
        assert errors == []
        assert len(created) == 1
        assert harness.owner() is created[0] and created[0].is_alive()
        assert pending.qsize() == 2
    finally:
        allow_start.set()
        for producer in producers:
            if producer.ident is not None:
                producer.join(5)


# ── requests that ask for more than one thing (29 Oct 2026) ─────────────────

def test_a_request_is_split_into_its_asks_and_a_single_ask_is_not():
    from orion_core.tool_resolver import request_parts
    assert request_parts("find my cv and email it to hannah") == ["find my cv", "email it to hannah"]
    assert request_parts("check the weather and then remind me to take an umbrella") == [
        "check the weather", "remind me to take an umbrella"]
    assert request_parts("Run diagnostics and tell me what changed in the last update.") == [
        "Run diagnostics", "tell me what changed in the last update."]
    for single in ("weigh up the pros and cons", "play some rock and roll",
                   "what's the weather like", "open main.py in the editor"):
        assert request_parts(single) == [], single


def test_each_part_lifts_only_its_best_tool_and_the_first_tool_never_changes():
    from orion_core.tool_resolver import PART_STEP, merge_part_scores
    whole = {"find_files": 9.0, "process_file": 5.0, "web_search": 4.0, "messaging": 1.0}
    parts = [{"find_files": 7.0, "process_file": 6.0}, {"messaging": 3.0, "web_search": 2.9}]
    merged = merge_part_scores(whole, parts)
    assert sorted(merged, key=lambda name: -merged[name])[:2] == ["find_files", "messaging"]
    assert merged["messaging"] == pytest.approx(9.0 * (1 - 2 * PART_STEP))
    assert merged["process_file"] == 5.0 and merged["web_search"] == 4.0
    assert merge_part_scores(whole, []) == whole
    assert merge_part_scores({"a": 0.0}, parts) == {"a": 0.0}


def test_every_part_of_a_multi_part_goal_reaches_the_top_three_more_often(monkeypatch):
    """The planner's seven multi-part goals, words alone (deterministic):
    measured 29 Oct 2026 with the speller on, parts with their tool in the
    top three 9/14 -> 10/14, and the 394 everyday requests' top-1 unchanged."""
    import json
    from orion_core import intent_brain, tool_resolver
    monkeypatch.setattr(intent_brain, "combined_scores",
                        lambda query, declarations, _net=None: tool_resolver.lexical_scores(
                            query, declarations))
    menu = [d for d in TOOL_DECLARATIONS if d["name"] not in {"find_tool", "use_tool"}]
    plan = json.loads((ROOT / "tools" / "planner_eval_set.json").read_text(encoding="utf-8"))
    goals = [c for c in plan["cases"] if c.get("family") == "multi"]

    def covered(scorer):
        hits = 0
        for case in goals:
            scores = scorer(case["goal"], menu)
            top = sorted(scores, key=lambda name: (-scores[name], name))[:3]
            needs = [[tool] for tool in case["expect"].get("tools_all", [])]
            if case["expect"].get("tools_any"):
                needs.append(case["expect"]["tools_any"])
            hits += sum(any(tool in top for tool in need) for need in needs)
        return hits

    assert covered(intent_brain.routing_scores) > covered(tool_resolver.lexical_scores)
    from data import intent_routing_cases as everyday
    for case in everyday.CASES:
        whole = tool_resolver.lexical_scores(case[0], menu)
        split = intent_brain.routing_scores(case[0], menu)
        assert (max(split, key=lambda name: (split[name], name))
                == max(whole, key=lambda name: (whole[name], name))), case[0]


def test_a_plan_that_leaves_a_part_out_is_noticed():
    """Words alone, on the planner's seven multi-part goals, each plan with
    one part's tool removed: 12 of 14 such plans noticed, 2 of 7 complete
    plans wrongly noticed (29 Oct 2026) - so it is a note to check, never a
    rejection."""
    from orion_core.tool_resolver import lexical_scores, uncovered_parts
    menu = [d for d in TOOL_DECLARATIONS if d["name"] not in {"find_tool", "use_tool"}]
    goal = "Check my system resources and how many tokens I have used today."
    assert uncovered_parts(goal, ["resource_status", "token_usage"], menu, lexical_scores) == []
    assert uncovered_parts(goal, ["resource_status"], menu, lexical_scores) == [
        "how many tokens I have used today."]
    assert uncovered_parts("search the web for CV writing tips", [], menu) == []   # one ask


# ── how much he can learn (29 Oct 2026) ──────────────────────────────────────

def test_pinned_experience_is_never_evicted_and_sampling_draws_from_both():
    from orion_core.neural import ReplayMemory
    memory = ReplayMemory(capacity=3, seed=1)
    for index in range(5):
        memory.pin(("tool", index), 0)
    for index in range(10):
        memory.add(("lesson", index), 1, 3.0)
    assert len(memory.pinned) == 5 and len(memory.items) == 3 and len(memory) == 8
    drawn = memory.sample(50)
    assert len(drawn) == 8 and len({row[0] for row in drawn}) == 8
    assert {row[0] for row in drawn} >= {("lesson", 9), ("tool", 0)}
    memory.clear()
    assert len(memory) == 0 and memory.sample(3) == []


def test_lessons_beyond_the_old_400_and_the_tools_phrases_survive_a_restart(tmp_path, monkeypatch):
    """The last 400 TURNS used to be all that survived a restart or a
    re-training, repeats included, and lessons evicted the tools' phrases
    from the replay. Counted on 3,000 simulated lessons (712 distinct):
    225 -> 712 phrasings kept, 9,000 -> 9,915 of 9,915 tool phrases."""
    import threading
    from orion_core import intent_brain
    monkeypatch.setattr(intent_brain, "_STOPPING", threading.Event())
    monkeypatch.setattr(intent_brain, "_QUIET", [None])
    brain = _toy_brain(tmp_path / "b", epochs=1)
    pinned = len(brain.replay.pinned)
    assert pinned and not brain.replay.items
    monkeypatch.setattr(brain.net, "train_batch", lambda *_a, **_k: None)   # counting, not learning
    monkeypatch.setattr(brain, "save", lambda: True)
    for index in range(450):
        brain.learn_now(f"is it going to rain on day {index}", "geo")
    for _ in range(20):
        brain.learn_now("pause the music", "media_control")
    assert len(brain.replay.pinned) == pinned
    assert intent_brain.IntentBrain.save(brain)
    again = intent_brain.IntentBrain(directory=tmp_path / "b")
    assert len(again._recent) == 451
    assert again._recent[-1] == ("pause the music", "media_control")
    assert again._recent.count(("pause the music", "media_control")) == 1
    again.restore_corpus(_TOY, _TOY_TEXT)
    assert len(again.replay.pinned) == pinned and len(again.replay.items) == 451



def test_a_silent_turn_is_summarised_by_every_step_in_plain_words():
    """30 Oct 2026: only the last step was reported, raw - "That did not work:
    [notion_workspace] Unsupported notion action: create. OPERATION
    OBSERVATIONS: {...}" - while the research before it was running."""
    from orion_core.text_tools import ToolStep, summarise_steps
    steps = [ToolStep("research", {}, True, "Very good. I'll research 'sleep' for about 30 minutes."),
             ToolStep("notion_workspace", {}, False,
                      "[notion_workspace] Unsupported notion action: create.\n"
                      "OPERATION OBSERVATIONS: {\"execution\": {}}")]
    said = summarise_steps(steps)
    assert said.startswith("Very good. I'll research 'sleep' for about 30 minutes.")
    assert "One part did not work - notion workspace: Unsupported notion action: create." in said
    assert "OPERATION" not in said
