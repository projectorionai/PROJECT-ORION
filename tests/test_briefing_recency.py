"""
ORION remembering that he already briefed you — and saying so.

  "ORION seems to not remember if he's given a briefing in the day too, I want
   him to explicitly remember and say 'It seems you've had your briefing in the
   past hour' or if there is some new news he should say [that]."

The bookkeeping already existed and was working. Two things were missing.

The state recorded only the DATE, which is enough to avoid re-offering and not
enough to say anything useful — "in the past hour" and "earlier today" are
different sentences and only a timestamp separates them.

And the startup instruction said, in as many words, "Do NOT offer or mention
the briefing". Correct about not re-offering; wrong about staying silent. From
the user's side an assistant who says nothing is indistinguishable from one who
has forgotten — which is exactly how this was reported.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orion_core.briefing as briefing_module  # noqa: E402
from orion_core.briefing import MorningBriefingService  # noqa: E402

NOW = datetime(2026, 8, 9, 14, 0, 0)


@pytest.fixture
def service(monkeypatch, tmp_path):
    monkeypatch.setattr(briefing_module, "BRIEFING_STATE_PATH", tmp_path / "state.json")
    svc = MorningBriefingService.__new__(MorningBriefingService)
    svc._briefing_state = {}
    return svc


def _briefed(service, minutes_ago: float):
    service._briefing_state = {}
    service.mark_briefed(NOW - timedelta(minutes=minutes_ago))
    return service


# ── minute-accurate, as the user asked ───────────────────────────────────────
#
# "he's inaccurate on timings, he says sometimes a briefing was done a few hours
#  ago when it was done like 45 minutes ago ... can this be accurate to the
#  minute?" — so the vague bands ("in the past hour") were replaced with an
#  exact phrase the model is told to say verbatim.

@pytest.mark.parametrize("minutes,expected", [
    (0, "just now"),
    (1, "1 minute ago"),
    (9, "9 minutes ago"),
    (45, "45 minutes ago"),
    (59, "59 minutes ago"),
    (60, "1 hour ago"),
    (75, "1 hour and 15 minutes ago"),
    (125, "2 hours and 5 minutes ago"),
    (200, "3 hours and 20 minutes ago"),
])
def test_recency_is_accurate_to_the_minute(service, minutes, expected):
    assert _briefed(service, minutes).briefing_recency_phrase(NOW) == expected


def test_the_exact_phrase_from_the_request_is_reachable(service):
    """The specific case the user reported: 45 minutes must say 45 minutes."""
    assert _briefed(service, 45).briefing_recency_phrase(NOW) == "45 minutes ago"


def test_a_clock_time_is_never_read_out(service):
    """Minute-accurate, but still speech — "45 minutes ago", never "13:15"."""
    for minutes in (5, 30, 90, 200, 400):
        phrase = _briefed(service, minutes).briefing_recency_phrase(NOW)
        assert ":" not in phrase
        assert "ago" in phrase or phrase == "just now"


def test_plurals_are_correct(service):
    assert _briefed(service, 1).briefing_recency_phrase(NOW) == "1 minute ago"
    assert _briefed(service, 60).briefing_recency_phrase(NOW) == "1 hour ago"
    assert _briefed(service, 61).briefing_recency_phrase(NOW) == "1 hour and 1 minute ago"


# ── honesty when the time is not known ───────────────────────────────────────

def test_never_briefed_says_nothing(service):
    assert service.briefing_recency_phrase(NOW) == ""
    assert service.time_since_briefing(NOW) is None


def test_state_written_before_timestamps_says_nothing(service):
    """Upgrading must not invent a time. Silence beats a confident guess."""
    service._briefing_state = {"last_briefed_date": "2026-08-09"}
    assert service.already_briefed_today(NOW) is True
    assert service.last_briefed_at() is None
    assert service.briefing_recency_phrase(NOW) == ""


def test_a_future_timestamp_says_nothing(service):
    """A clock change or an edited state file; 'in -2 hours' is worse than
    saying nothing."""
    service._briefing_state = {}
    service.mark_briefed(NOW + timedelta(hours=2))
    assert service.briefing_recency_phrase(NOW) == ""


def test_a_corrupt_timestamp_says_nothing(service):
    service._briefing_state = {"last_briefed_at": "not a date"}
    assert service.last_briefed_at() is None
    assert service.briefing_recency_phrase(NOW) == ""


# ── the existing behaviour still holds ───────────────────────────────────────

def test_the_date_is_still_recorded(service):
    """already_briefed_today is what stops the re-offer; it must not regress."""
    _briefed(service, 30)
    assert service.already_briefed_today(NOW) is True


def test_yesterdays_briefing_does_not_count_as_today(service):
    _briefed(service, 60 * 26)
    assert service.already_briefed_today(NOW) is False


def test_the_timestamp_survives_a_restart(monkeypatch, tmp_path):
    path = tmp_path / "state.json"
    monkeypatch.setattr(briefing_module, "BRIEFING_STATE_PATH", path)
    first = MorningBriefingService.__new__(MorningBriefingService)
    first._briefing_state = {}
    first.mark_briefed(NOW - timedelta(minutes=40))

    reloaded = MorningBriefingService.__new__(MorningBriefingService)
    reloaded._briefing_state = MorningBriefingService._load_briefing_state()
    assert reloaded.already_briefed_today(NOW) is True
    assert reloaded.briefing_recency_phrase(NOW) == "40 minutes ago"


# ── he actually says it ──────────────────────────────────────────────────────

def _worker_source() -> str:
    return (Path(__file__).resolve().parents[1] / "orion_core" / "live_worker.py").read_text(
        encoding="utf-8", errors="replace")


def test_he_is_no_longer_told_to_stay_silent_about_it():
    """The defect. This instruction is why he appeared to have forgotten."""
    assert "Do NOT offer or mention the briefing" not in _worker_source()


def test_he_is_told_to_say_he_remembers():
    source = _worker_source()
    assert "briefing_recency_phrase" in source
    assert "already had" in source


def test_he_still_does_not_re_offer_when_nothing_is_new():
    assert "Do NOT offer another briefing" in _worker_source()


def test_new_news_is_announced_as_new_rather_than_re_offered():
    """Having been briefed AND something having broken since is a specific
    situation; asking a generic 'would you like your briefing?' throws away the
    one useful thing ORION knows."""
    source = _worker_source()
    assert "something new has broken" in source
    assert "is NEW material, not a repeat" in source


def test_the_spoken_fallback_works_without_a_live_session():
    """The no-session path speaks directly rather than instructing a model, and
    it must carry the same memory."""
    source = _worker_source()
    assert "You've already had your briefing" in source


# ── the midnight hole ────────────────────────────────────────────────────────
#
# already_briefed_today() is a CALENDAR test, so a briefing at 23:40 stopped
# counting at 00:01 and ORION would offer a whole fresh briefing twenty minutes
# after giving one. Found by running the real flow rather than a fake, at a
# moment that happened to be just past midnight.

MIDNIGHT_ISH = datetime(2026, 8, 10, 0, 10, 0)


def test_a_briefing_from_just_before_midnight_still_counts(service):
    service._briefing_state = {}
    service.mark_briefed(datetime(2026, 8, 9, 23, 40, 0))
    assert service.already_briefed_today(MIDNIGHT_ISH) is False, (
        "the calendar test is supposed to roll over; that is not the bug")
    assert service.briefed_recently(MIDNIGHT_ISH) is True
    assert service.briefing_recency_phrase(MIDNIGHT_ISH) == "30 minutes ago"


def test_recency_does_not_keep_a_stale_briefing_alive_forever(service):
    service._briefing_state = {}
    service.mark_briefed(MIDNIGHT_ISH - timedelta(hours=9))
    assert service.briefed_recently(MIDNIGHT_ISH) is False


def test_never_briefed_is_not_recent(service):
    assert service.briefed_recently(MIDNIGHT_ISH) is False


def test_the_worker_checks_recency_as_well_as_the_date():
    source = _worker_source()
    assert "briefed_recently" in source
    assert "already_briefed_today" in source
