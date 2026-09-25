"""
What ORION keeps doing while he is not listening to you.

  "When I tell ORION that I will need him to stop listening to me, he must go
   in standby mode - during this time he cannot listen to me, talk to me and is
   limited to research tasks, or any task that has been given to him (this will
   show in logs) ... He is still free to think at this time and autonomously
   improve himself."

  "if I tell ORION that I need to focus it goes into it's standby mode"

Standby already closed the microphone and stopped the voice. Two things were
missing, and neither was a bug so much as an absence.

Nothing anywhere in the code could answer "is this piece of work allowed to run
right now?". Research and the forge carried on because nothing had told them to
stop — the right behaviour by accident rather than by decision. And nothing
recorded what continued, so "this will show in logs" had nothing behind it.

The organising idea is that standby is about ATTENTION, not activity. Work that
points outward (speaking, listening, volunteering) stops; work that points
inward (research, thinking, finishing what he was already given, improving
himself) continues, because pausing it would waste exactly the quiet stretch
that suits it best.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.standby_work import (  # noqa: E402
    LEDGER, PERMITTED_IN_STANDBY, StandbyLedger, Work, gate, may_run,
)


# ── the permission model ─────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [
    Work.RESEARCH, Work.ASSIGNED_TASK, Work.THINKING,
    Work.SELF_IMPROVEMENT, Work.INGESTION, Work.MAINTENANCE,
])
def test_inward_work_continues_in_standby(kind):
    assert may_run(kind, standby=True) is True


@pytest.mark.parametrize("kind", [
    Work.LISTENING, Work.PROACTIVE_SPEECH, Work.CONVERSATION, Work.BRIEFING,
])
def test_outward_work_stops_in_standby(kind):
    assert may_run(kind, standby=True) is False


def test_safety_is_never_gated():
    """A safety path that can be switched off is not a safety path."""
    assert may_run(Work.SAFETY, standby=True) is True


def test_everything_runs_when_not_in_standby():
    for kind in Work:
        assert may_run(kind, standby=False) is True


def test_a_string_kind_works_as_well_as_the_enum():
    """Call sites should not have to import the enum to ask a question."""
    assert may_run("research", standby=True) is True
    assert may_run("listening", standby=True) is False


def test_an_enum_member_is_not_mangled_by_str():
    """The real bug: Work subclasses (str, Enum), so str(Work.RESEARCH) is
    'Work.RESEARCH', not 'research'. Every lookup raised ValueError and fell
    through to "stop", which made standby look like it halted everything."""
    assert may_run(Work.RESEARCH, standby=True) is True


def test_an_unclassified_kind_is_stopped():
    """Standby is a promise to leave the user alone. Being wrong in the
    permissive direction breaks the promise; being wrong the other way just
    delays some work."""
    assert may_run("something_nobody_classified", standby=True) is False


def test_the_permitted_set_matches_the_predicate():
    for kind in Work:
        assert may_run(kind, standby=True) == (kind in PERMITTED_IN_STANDBY)


# ── the ledger: "this will show in logs" ─────────────────────────────────────

@pytest.fixture
def ledger():
    book = StandbyLedger()
    book.start()
    return book


def test_permitted_work_is_recorded():
    LEDGER.start()
    assert gate(Work.RESEARCH, True, "neural interfaces sweep") is True
    assert LEDGER.counts() == {"research": 1}


def test_blocked_work_is_counted_but_not_recorded_as_done():
    LEDGER.start()
    assert gate(Work.PROACTIVE_SPEECH, True, "a reminder") is False
    assert LEDGER.counts() == {}
    assert LEDGER.blocked == {"proactive_speech": 1}


def test_nothing_is_recorded_when_not_in_standby():
    """The ledger is a record of a specific stretch of time, not a general log."""
    LEDGER.start()
    assert gate(Work.RESEARCH, False, "ordinary work") is True
    assert LEDGER.counts() == {}


def test_the_ledger_does_not_grow_without_bound(ledger):
    for index in range(ledger.LIMIT + 60):
        ledger.record(Work.THINKING, f"thought {index}")
    assert len(ledger.entries) == ledger.LIMIT


def test_starting_a_new_standby_clears_the_last_one(ledger):
    ledger.record(Work.RESEARCH, "old")
    ledger.start()
    assert ledger.counts() == {}


# ── what he says on waking ───────────────────────────────────────────────────

def test_the_wake_summary_reads_like_a_sentence(ledger):
    ledger.record(Work.RESEARCH, "a")
    ledger.record(Work.RESEARCH, "b")
    ledger.record(Work.SELF_IMPROVEMENT, "c")
    summary = ledger.summary()
    assert "2 research tasks" in summary
    assert "1 improvement" in summary
    assert summary.endswith(".")


def test_plurals_are_correct(ledger):
    """A naive +'s' produced '2 researchs' — and this is said out loud on every
    single wake."""
    ledger.record(Work.RESEARCH, "a")
    assert "1 research task" in ledger.summary()
    ledger.record(Work.RESEARCH, "b")
    assert "2 research tasks" in ledger.summary()
    assert "researchs" not in ledger.summary()


def test_doing_nothing_says_nothing(ledger):
    """'While you were away I did nothing' is a sentence no one needs."""
    assert ledger.summary() == ""


def test_the_report_names_what_was_held_back(ledger):
    ledger.record(Work.RESEARCH, "a sweep")
    ledger.note_blocked(Work.PROACTIVE_SPEECH)
    report = ledger.report()
    assert "a sweep" in report
    assert "held back" in report
    assert "proactive_speech" in report


def test_an_empty_report_is_still_a_sentence(ledger):
    assert "Nothing ran" in ledger.report()


# ── the trigger ──────────────────────────────────────────────────────────────

def _worker_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core"
            / "live_worker.py").read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("phrase", [
    "i need to focus",
    "i need to concentrate",
    "let me focus",
    "stop listening to me",
    "go on standby",
    "i need some quiet",
])
def test_needing_to_focus_means_full_standby(phrase):
    from orion_core.live_worker import GenAILiveWorker

    assert GenAILiveWorker._NEED_FOCUS_RE.search(phrase), phrase


@pytest.mark.parametrize("phrase", ["i am busy", "i'm concentrating", "i'm in a meeting"])
def test_merely_being_busy_stays_the_lighter_state(phrase):
    """'I'm busy' means stop volunteering; it does not mean stop listening.
    Collapsing the two would take away a genuinely useful middle state."""
    from orion_core.live_worker import GenAILiveWorker

    assert not GenAILiveWorker._NEED_FOCUS_RE.search(phrase), phrase
    assert GenAILiveWorker._FOCUS_ON_RE.search(phrase), phrase


def test_the_release_phrase_is_not_caught_by_the_trigger():
    """Otherwise the phrase meant to free him would put him back under."""
    from orion_core.live_worker import GenAILiveWorker

    for phrase in ("i'm done", "what did i miss", "you can talk again"):
        assert GenAILiveWorker._FOCUS_OFF_RE.search(phrase), phrase


def test_the_trigger_is_checked_before_the_lighter_one():
    """_FOCUS_ON_RE matches on the word 'focus' and would otherwise swallow
    'I need to focus', leaving him merely quiet — still listening, which is
    exactly what was not wanted."""
    source = _worker_source()
    body = source[source.index("def _handle_focus_command"):]
    body = body[:body.index("\n    #:")]
    assert body.index("_NEED_FOCUS_RE") < body.index("_FOCUS_ON_RE")


def test_entering_standby_closes_the_microphone():
    source = _worker_source()
    body = source[source.index("def _enter_focus_standby"):]
    body = body[:body.index("def _handle_focus_command")]
    assert "enter_standby" in body


def test_he_says_how_to_get_him_back():
    """A state the user cannot see the exit from is a trap."""
    source = _worker_source()
    body = source[source.index("def _enter_focus_standby"):]
    body = body[:body.index("def _handle_focus_command")]
    assert "I need you" in body


def test_waking_reports_what_he_got_done():
    source = _worker_source()
    assert "LEDGER.summary()" in source
    assert "LEDGER.report()" in source
