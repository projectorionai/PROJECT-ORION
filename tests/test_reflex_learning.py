"""
Learned reflexes (Mark XXVI) — ORION deriving his own fast-paths.

A learned reflex skips the model entirely, so the tests that matter are the ones
that prove it CANNOT fire when it should not. The ordering below follows the
design's own priority: safety gates first, then correctness, then the speed the
feature exists for.

The gate reuses ``concurrency.classify`` (PARALLEL == read-only) ANDed with
``remote_capability.classify`` (not FORBID). Those tests assert against the real
classifiers rather than mocks — a mocked gate would pass happily while the live
one had drifted, which is exactly the failure this feature must not have.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.db import journal_mode  # noqa: E402
from orion_core.reflex_learning import (  # noqa: E402
    MAX_PHRASE_LEN,
    MIN_OBSERVATIONS,
    LearnedReflex,
    ReflexLearner,
    normalise,
)

#: A genuinely read-only tool, per orion_core.concurrency.PARALLEL_TOOLS.
READ_ONLY = "web_search"
READ_ONLY_2 = "research"


@pytest.fixture
def learner(tmp_path):
    made = ReflexLearner(tmp_path / "learn.db")
    yield made
    made.close()


def _teach(learner, phrase, tool=READ_ONLY, args=None, times=MIN_OBSERVATIONS):
    for _ in range(times):
        learner.observe(phrase, tool, args if args is not None else {})


# ── normalisation ─────────────────────────────────────────────────────────────

def test_normalise_folds_case_punctuation_and_whitespace():
    assert normalise("  What's   DUE, today? ") == "what's due today"


def test_normalise_keeps_apostrophes_because_they_carry_meaning():
    assert "'" in normalise("what's due")
    assert normalise("whats due") != normalise("what's due")


def test_normalise_does_not_stem():
    """Aggressive normalisation is how a matcher starts firing on things it was
    never taught: 'delete the file' must not collapse into 'deleting files'."""
    assert normalise("delete the file") != normalise("deleting the files")


def test_normalise_of_junk_is_empty():
    assert normalise("   ") == ""
    assert normalise("!!!???") == ""
    assert normalise(None) == ""


# ── safety gate: what may never be learned ───────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("desktop_control", {"action": "click"}),
    ("forge", {}),
    ("security_recon", {"target": "10.0.0.1"}),
    ("process_file", {"path": "x"}),
    ("study", {"action": "add"}),
    ("wellbeing", {"action": "checkin"}),
    ("a_tool_nobody_has_ever_classified", {}),
])
def test_a_tool_that_is_not_read_only_is_never_observed(learner, tool, args):
    assert learner.observe("do the thing", tool, args) is None
    _teach(learner, "do the thing", tool, args, times=10)
    assert learner.candidates() == []


def test_a_read_only_tool_is_observed(learner):
    assert learner.observe("look that up", READ_ONLY, {}) is not None


def test_the_gate_uses_the_live_classifiers_not_a_local_list():
    """If concurrency stops calling a tool PARALLEL, this must stop learning it.
    Asserting against the real classifier is the point of the test."""
    from orion_core.concurrency import PARALLEL_TOOLS
    assert READ_ONLY in PARALLEL_TOOLS
    assert ReflexLearner._is_read_only(READ_ONLY, {}) is True
    assert ReflexLearner._is_read_only("desktop_control", {"action": "click"}) is False


def test_an_over_long_utterance_is_a_sentence_not_a_command(learner):
    assert learner.observe("x" * (MAX_PHRASE_LEN + 1), READ_ONLY, {}) is None


def test_an_empty_utterance_is_ignored(learner):
    assert learner.observe("", READ_ONLY, {}) is None
    assert learner.observe("   ", READ_ONLY, {}) is None


def test_a_missing_tool_name_is_ignored(learner):
    assert learner.observe("something", "", {}) is None


# ── promotion rules ──────────────────────────────────────────────────────────

def test_a_phrase_needs_the_full_threshold(learner):
    for i in range(MIN_OBSERVATIONS - 1):
        learner.observe("check the news", READ_ONLY, {})
        assert learner.candidates() == [], "promoted after only %d times" % (i + 1)
    learner.observe("check the news", READ_ONLY, {})
    assert len(learner.candidates()) == 1


def test_an_ambiguous_phrase_is_disqualified_not_out_voted(learner):
    """Unanimity, not majority: if the model ever chose differently for this
    phrase, the model should keep choosing."""
    for _ in range(20):
        learner.observe("show me the thing", READ_ONLY, {})
    learner.observe("show me the thing", READ_ONLY_2, {})
    assert learner.candidates() == []


def test_a_phrase_that_ever_failed_is_not_promoted(learner):
    """Otherwise ORION learns to fail faster."""
    _teach(learner, "flaky lookup", times=8)
    learner.observe("flaky lookup", READ_ONLY, {}, ok=False)
    assert learner.candidates() == []


def test_a_different_action_counts_as_a_different_target(learner):
    for _ in range(5):
        learner.observe("dig into it", READ_ONLY, {"action": "news"})
        learner.observe("dig into it", READ_ONLY, {"action": "images"})
    assert learner.candidates() == []


def test_nothing_fires_until_it_is_promoted(learner):
    _teach(learner, "check the news")
    assert len(learner.candidates()) == 1
    assert learner.match("check the news") is None, "a candidate must not fire"
    assert learner.promote("check the news") is True
    assert learner.match("check the news") is not None


def test_promoting_an_ineligible_phrase_fails(learner):
    assert learner.promote("never said this") is False
    learner.observe("said once", READ_ONLY, {})
    assert learner.promote("said once") is False


def test_promote_accepts_an_unnormalised_phrase(learner):
    _teach(learner, "check the news")
    assert learner.promote("  Check The News?  ") is True


# ── matching ─────────────────────────────────────────────────────────────────

def test_a_promoted_phrase_matches_regardless_of_case_and_punctuation(learner):
    _teach(learner, "what's the weather doing")
    learner.promote("what's the weather doing")
    for variant in ("What's the weather doing?", "WHAT'S THE WEATHER DOING",
                    "  what's the weather doing.  "):
        assert learner.match(variant) is not None, variant


def test_only_the_exact_phrase_matches(learner):
    """No fuzzy matching, by design — this is the property that makes a learned
    reflex safe without a human reviewing it."""
    _teach(learner, "check the news")
    learner.promote("check the news")
    for near_miss in ("check news", "check the news please", "check the newspaper",
                      "can you check the news", "check the news for me"):
        assert learner.match(near_miss) is None, near_miss


def test_match_returns_the_tool_and_action(learner):
    _teach(learner, "dig into that", args={"action": "news"})
    learner.promote("dig into that")
    found = learner.match("dig into that")
    assert found.tool == READ_ONLY
    assert found.args() == {"action": "news"}
    assert "learned" in found.why()


def test_match_of_empty_or_huge_text_is_none(learner):
    assert learner.match("") is None
    assert learner.match(None) is None
    assert learner.match("y" * (MAX_PHRASE_LEN + 5)) is None


# ── withdrawing ──────────────────────────────────────────────────────────────

def test_a_learned_reflex_can_be_demoted(learner):
    _teach(learner, "check the news")
    learner.promote("check the news")
    assert learner.demote("check the news") is True
    assert learner.match("check the news") is None
    assert len(learner.candidates()) == 1, "demoting returns it to candidacy"


def test_demoting_something_not_promoted_is_false(learner):
    assert learner.demote("never") is False


def test_forget_erases_the_history(learner):
    _teach(learner, "check the news")
    learner.promote("check the news")
    assert learner.forget("check the news") is True
    assert learner.match("check the news") is None
    assert learner.candidates() == []
    assert learner.forget("check the news") is False


# ── persistence and reporting ────────────────────────────────────────────────

def test_learning_survives_a_restart(tmp_path):
    path = tmp_path / "learn.db"
    first = ReflexLearner(path)
    _teach(first, "check the news")
    first.promote("check the news")
    first.close()

    second = ReflexLearner(path)
    try:
        assert second.match("check the news") is not None
    finally:
        second.close()


def test_the_store_uses_the_projects_sqlite_policy(learner):
    """It is written on the turn path, so a full fsync per write would be the
    exact lag this pass removed elsewhere."""
    assert journal_mode(learner._db).lower() == "wal"


def test_an_empty_report_says_so_plainly(learner):
    assert "No learned reflexes yet" in learner.report()


def test_the_report_separates_live_reflexes_from_candidates(learner):
    _teach(learner, "check the news")
    _teach(learner, "look up the docs")
    learner.promote("check the news")
    report = learner.report()
    assert "1 learned reflex" in report
    assert "1 candidate" in report
    assert "check the news" in report and "look up the docs" in report


def test_stats_counts_what_it_says_it_counts(learner):
    _teach(learner, "check the news")
    learner.promote("check the news")
    _teach(learner, "look up the docs", times=1)
    stats = learner.stats()
    assert stats["phrases"] == 2
    assert stats["observations"] == MIN_OBSERVATIONS + 1
    assert stats["learned"] == 1
    assert stats["candidates"] == 0


def test_a_learned_reflex_renders_readably():
    line = LearnedReflex("check the news", READ_ONLY, "", 4, promoted=True).line()
    assert "check the news" in line and READ_ONLY in line and "4 times" in line


# ── speed ────────────────────────────────────────────────────────────────────

def test_matching_stays_flat_as_the_table_grows(learner):
    """A dict lookup on a normalised string: the cost is the normalisation, not
    the search, so 500 learned reflexes must cost the same as one."""
    for i in range(500):
        phrase = "learned phrase number %d" % i
        _teach(learner, phrase)
        learner.promote(phrase)
    assert len(learner.learned()) == 500

    learner.match("learned phrase number 250")
    start = time.perf_counter()
    for _ in range(3000):
        learner.match("learned phrase number 250")
    per_call_us = (time.perf_counter() - start) * 1e6 / 3000
    assert per_call_us < 200.0, "a learned match costs %.1f us" % per_call_us


def test_a_miss_is_as_cheap_as_a_hit(learner):
    for i in range(200):
        phrase = "learned phrase number %d" % i
        _teach(learner, phrase)
        learner.promote(phrase)
    start = time.perf_counter()
    for _ in range(3000):
        learner.match("something entirely unrelated to any of that")
    per_call_us = (time.perf_counter() - start) * 1e6 / 3000
    assert per_call_us < 200.0, "a learned miss costs %.1f us" % per_call_us


# ── wiring ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]


def test_the_worker_consults_learned_reflexes_after_the_written_ones():
    src = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
    written = src.index("m = match_reflex(lowered)")
    learned = src.index("self._learned_reflex(lowered)")
    assert learned > written, "hand-written rules must win over learned ones"


def test_the_worker_only_learns_from_a_single_tool_turn():
    src = (ROOT / "orion_core" / "live_worker.py").read_text(encoding="utf-8")
    assert "if len(calls) == 1:" in src, (
        "a two-tool turn does not say which tool the phrase meant")
    assert "observe_for_reflex" in src


def test_observation_never_breaks_a_turn():
    """It is bookkeeping; bookkeeping must not be able to break a reply."""
    import inspect
    from orion_core.live_worker import GenAILiveWorker
    src = inspect.getsource(GenAILiveWorker.observe_for_reflex)
    assert "except Exception" in src


def test_the_diagnostics_tool_reports_learned_reflexes():
    src = (ROOT / "orion_core" / "dispatch_files.py").read_text(encoding="utf-8")
    assert '"reflexes"' in src and "ReflexLearner" in src


def test_the_schema_advertises_the_reflexes_action():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "diagnostics")
    assert "reflexes" in tool["description"].lower()
