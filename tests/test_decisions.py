"""
Decision journal & calibration (Mark XXVI) — was that judgement any good?

The point of the exercise is that the reasoning is captured BEFORE the outcome is
known, and that the confidence is then scored against reality. So the tests care
about two things above all: the record cannot be silently rewritten after the
fact, and the calibration maths is honest — including telling the user something
unflattering when the data says so.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.decisions import (  # noqa: E402
    OUTCOMES,
    Decision,
    DecisionJournal,
    DecisionStore,
    brier_score,
    calibration_bins,
    describe_calibration,
)

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


@pytest.fixture()
def journal(tmp_path):
    return DecisionJournal(DecisionStore(tmp_path / "decisions.db"))


# ── recording ────────────────────────────────────────────────────────────────

def test_recording_captures_the_reasoning_and_a_review_date(journal):
    decision = journal.record(
        "Ship the standalone build this week?", "ship on Friday",
        confidence=80, prediction="no rollback needed",
        rationale="the suite is green and the freeze smoke-tested", now=NOW)
    assert decision.id is not None
    assert decision.confidence == 80
    assert decision.rationale.startswith("the suite is green")
    assert decision.review_at, "a decision with no review date is never revisited"
    assert not decision.resolved


def test_a_decision_needs_a_question_and_a_choice(journal):
    with pytest.raises(ValueError):
        journal.record("", "something")
    with pytest.raises(ValueError):
        journal.record("something", "")


def test_confidence_is_clamped(journal):
    assert journal.record("q", "c", confidence=140).confidence == 100
    assert journal.record("q", "c", confidence=-5).confidence == 0


def test_the_record_survives_a_reload(tmp_path):
    store = DecisionStore(tmp_path / "d.db")
    first = DecisionJournal(store)
    first.record("q", "c", confidence=65, rationale="because", now=NOW)
    store.close()

    again = DecisionJournal(DecisionStore(tmp_path / "d.db")).store.all()[0]
    assert again.confidence == 65 and again.rationale == "because"


# ── resurfacing ──────────────────────────────────────────────────────────────

def test_a_decision_becomes_due_on_its_review_date(journal):
    journal.record("q", "c", review_days=30, now=NOW)
    assert journal.due(now=NOW) == []
    assert len(journal.due(now=NOW + timedelta(days=31))) == 1


def test_a_resolved_decision_is_never_due_again(journal):
    decision = journal.record("q", "c", review_days=1, now=NOW)
    journal.resolve(decision.id, "right", now=NOW + timedelta(days=2))
    assert journal.due(now=NOW + timedelta(days=30)) == []


def test_defer_pushes_the_review_out_without_losing_it(journal):
    decision = journal.record("q", "c", review_days=1, now=NOW)
    later = NOW + timedelta(days=2)
    assert journal.due(now=later)
    journal.defer(decision.id, days=30, now=later)
    assert journal.due(now=later) == []
    assert len(journal.open_decisions()) == 1


def test_deferring_a_resolved_decision_does_nothing(journal):
    decision = journal.record("q", "c", now=NOW)
    journal.resolve(decision.id, "right")
    assert journal.defer(decision.id) is None


# ── resolving ────────────────────────────────────────────────────────────────

def test_resolving_records_the_outcome_and_the_lesson(journal):
    decision = journal.record("q", "c", now=NOW)
    resolved = journal.resolve(decision.id, "wrong", actual="it slipped a week",
                               lesson="I ignored the dependency risk")
    assert resolved.outcome == "wrong"
    assert resolved.actual == "it slipped a week"
    assert resolved.resolved is True
    assert "dependency risk" in journal.lessons()[0]


def test_an_unknown_outcome_is_refused(journal):
    decision = journal.record("q", "c", now=NOW)
    with pytest.raises(ValueError):
        journal.resolve(decision.id, "sort of")


def test_resolving_a_missing_decision_is_graceful(journal):
    assert journal.resolve(999, "right") is None


def test_the_documented_outcomes_are_the_supported_ones():
    assert OUTCOMES == ("right", "wrong", "mixed", "unknowable")


# ── scoring: the honest mirror ───────────────────────────────────────────────

def test_brier_score_rewards_accuracy():
    assert brier_score([(1.0, 1.0), (0.0, 0.0)]) == 0.0          # perfect
    assert brier_score([(1.0, 0.0)]) == 1.0                      # confidently wrong
    assert brier_score([(0.5, 1.0), (0.5, 0.0)]) == 0.25         # coin flip
    assert brier_score([]) is None


def test_calibration_bins_report_the_real_hit_rate():
    # said 90% five times, right twice → the band should show 40%
    pairs = [(0.9, 1.0), (0.9, 1.0), (0.9, 0.0), (0.9, 0.0), (0.9, 0.0)]
    bins = calibration_bins(pairs)
    band = next(b for b in bins if b["band"].startswith("90"))
    assert band["n"] == 5
    assert band["actual"] == 40
    assert band["gap"] < 0, "an inflated band must report a negative gap"


def test_overconfidence_is_named_plainly():
    pairs = [(0.95, 0.0)] * 4 + [(0.9, 0.0)] * 3
    assert "OVERCONFIDENT" in describe_calibration(calibration_bins(pairs))


def test_underconfidence_is_named_too():
    pairs = [(0.55, 1.0)] * 4 + [(0.6, 1.0)] * 3
    assert "UNDERCONFIDENT" in describe_calibration(calibration_bins(pairs))


def test_good_calibration_is_recognised():
    pairs = [(0.7, 1.0)] * 7 + [(0.7, 0.0)] * 3
    assert "Well calibrated" in describe_calibration(calibration_bins(pairs))


def test_not_enough_data_says_so_rather_than_guessing():
    assert "Not enough" in describe_calibration(calibration_bins([(0.9, 1.0)]))


def test_a_mixed_outcome_counts_as_a_half(journal):
    decision = journal.record("q", "c", confidence=50, now=NOW)
    journal.resolve(decision.id, "mixed")
    assert journal.calibration()["brier"] == 0.0     # 50% said, 0.5 happened


def test_unknowable_outcomes_are_excluded_from_scoring(journal):
    first = journal.record("q", "c", confidence=90, now=NOW)
    journal.resolve(first.id, "unknowable")
    assert journal.calibration()["scored"] == 0, (
        "a decision whose outcome cannot be judged must not distort the score")


def test_calibration_summarises_the_whole_journal(journal):
    for confidence, outcome in ((90, "wrong"), (90, "wrong"), (60, "right")):
        decision = journal.record("q", "c", confidence=confidence, now=NOW)
        journal.resolve(decision.id, outcome)
    calibration = journal.calibration()
    assert calibration["scored"] == 3
    assert calibration["brier"] is not None
    assert calibration["hit_rate"] == 33
    assert calibration["verdict"]


def test_stats_count_the_journal(journal):
    a = journal.record("q1", "c", confidence=80, review_days=1, now=NOW)
    journal.record("q2", "c", confidence=60, now=NOW)
    journal.resolve(a.id, "right")
    stats = journal.stats(now=NOW + timedelta(days=2))
    assert stats["total"] == 2 and stats["resolved"] == 1 and stats["open"] == 1
    assert stats["right"] == 1
    assert stats["average_confidence"] == 70


# ── the tool ─────────────────────────────────────────────────────────────────

class _Bag:
    def __init__(self, dbpath):
        self.decisions = DecisionJournal(DecisionStore(dbpath))


def _tool(stub, args):
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    return ProductivityDispatchMixin.decision_tool(stub, args)


def test_the_tool_runs_the_whole_loop(tmp_path):
    stub = _Bag(tmp_path / "d.db")
    result = _tool(stub, {"action": "record", "question": "Ship Friday?",
                          "choice": "ship", "confidence": 80,
                          "why": "tests are green"})
    assert result.ok and "80% confident" in result.text

    stub.decisions.store.all()[0]
    result = _tool(stub, {"action": "resolve", "id": "1", "outcome": "right",
                          "lesson": "green tests were a good signal"})
    assert result.ok and "RIGHT" in result.text and "Brier" in result.text

    assert "Brier score" in _tool(stub, {"action": "calibration"}).text
    assert "green tests" in _tool(stub, {"action": "lessons"}).text
    assert "1 decision(s) logged" in _tool(stub, {"action": "stats"}).text


def test_the_tool_refuses_an_incomplete_record(tmp_path):
    stub = _Bag(tmp_path / "d.db")
    assert _tool(stub, {"action": "record", "question": "only a question"}).ok is False


def test_the_tool_refuses_a_bad_outcome(tmp_path):
    stub = _Bag(tmp_path / "d.db")
    _tool(stub, {"action": "record", "question": "q", "choice": "c"})
    assert _tool(stub, {"action": "resolve", "id": "1", "outcome": "maybe"}).ok is False


def test_the_tool_is_registered():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "decision")
    for key in ("action", "question", "choice", "confidence", "outcome"):
        assert key in tool["parameters"]["properties"]
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"decision"' in inspect.getsource(OrionDispatcher)


def test_the_daily_brief_surfaces_decisions_due_for_review(tmp_path):
    from types import SimpleNamespace
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin

    bag = SimpleNamespace()
    bag._rewind = SimpleNamespace(reload=lambda: None, decisions=lambda limit=5: [])
    bag._standing_questions = SimpleNamespace(due=lambda: [], list=lambda: [])
    bag.decisions = DecisionJournal(DecisionStore(tmp_path / "d.db"))
    bag.decisions.record("Ship Friday?", "ship", review_days=1,
                         now=datetime.now(timezone.utc) - timedelta(days=5))
    result = KnowledgeDispatchMixin.catch_up_tool(bag, {})
    assert "decision" in result.text.lower()
