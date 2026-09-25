"""
TimeService — the authoritative clock.

The reported bug: at 00:31 ORION greeted the user with "Good morning" and
described "a summer day".  Both came from `hour < 12` tests scattered across
seven modules with three different evening boundaries.  These tests pin the
band table, the greeting/period split, and the fact that every consumer now
reads the same clock.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion_core.time_service import (
    AFTERNOON,
    EVENING,
    MORNING,
    NIGHT,
    TIME,
    TimeService,
    classify,
    greeting_for,
    ordinal,
)


@pytest.fixture
def service():
    return TimeService()


def at(hour: int, minute: int = 0, month: int = 8, day: int = 8):
    return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)


# ── the exact clock times the brief asks for (§37) ────────────────────────────

@pytest.mark.parametrize("hour,minute,expected", [
    (0, 0, NIGHT),
    (6, 0, MORNING),
    (11, 59, MORNING),
    (12, 0, AFTERNOON),
    (15, 0, AFTERNOON),
    (17, 59, AFTERNOON),
    (18, 0, EVENING),
    (23, 59, NIGHT),
])
def test_the_specified_clock_times_are_categorised_correctly(hour, minute, expected):
    assert classify(at(hour, minute)) == expected


def test_every_hour_of_the_day_lands_in_exactly_one_band():
    bands = {classify(h) for h in range(24)}
    assert bands == {NIGHT, MORNING, AFTERNOON, EVENING}
    # and no hour is unclassified
    assert all(classify(h) in bands for h in range(24))


def test_the_night_band_wraps_midnight():
    assert classify(22) == NIGHT
    assert classify(23) == NIGHT
    assert classify(0) == NIGHT
    assert classify(4) == NIGHT
    assert classify(5) == MORNING


# ── the reported bug, exactly ─────────────────────────────────────────────────

def test_half_past_midnight_in_august_is_night_and_not_daytime(service):
    """The literal report: 12:31 AM described as a summer daytime."""
    service.set_clock(lambda: at(0, 31, month=8))
    assert service.time_of_day() == NIGHT
    assert service.is_night() is True
    assert service.is_morning() is False
    assert service.is_daytime() is False
    assert service.season() == "summer"       # August IS summer — that part was right


def test_half_past_midnight_still_greets_as_evening(service):
    """Period and greeting are deliberately different: English has no
    greeting 'good night' that means hello."""
    service.set_clock(lambda: at(0, 31))
    assert service.time_of_day() == NIGHT
    assert service.greeting_period() == EVENING


def test_greeting_never_returns_night():
    assert all(greeting_for(h) != NIGHT for h in range(24))


# ── seasons ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("month,expected", [
    (1, "winter"), (2, "winter"), (3, "spring"), (4, "spring"),
    (5, "spring"), (6, "summer"), (7, "summer"), (8, "summer"),
    (9, "autumn"), (10, "autumn"), (11, "autumn"), (12, "winter"),
])
def test_season_is_derived_from_the_month_not_assumed(service, month, expected):
    service.set_clock(lambda: at(12, 0, month=month, day=15))
    assert service.season() == expected


def test_southern_hemisphere_seasons_are_inverted(service):
    service.set_clock(lambda: at(12, 0, month=8))
    assert service.season() == "summer"
    service.set_hemisphere("south")
    assert service.season() == "winter"


# ── nothing is cached: midnight, DST and clock changes are picked up ──────────

def test_crossing_midnight_changes_the_date_and_the_period(service):
    moments = iter([at(23, 59, day=8), at(0, 1, day=9)])
    service.set_clock(lambda: next(moments))
    first = service.snapshot()
    second = service.snapshot()
    assert first.date.day == 8 and second.date.day == 9
    assert first.period == NIGHT and second.period == NIGHT
    assert second.moment > first.moment


def test_a_system_clock_correction_is_picked_up_immediately(service):
    moments = iter([at(9, 0), at(20, 0)])
    service.set_clock(lambda: next(moments))
    assert service.time_of_day() == MORNING
    assert service.time_of_day() == EVENING


def test_the_real_clock_is_timezone_aware_with_a_live_offset():
    """UK BST/GMT correctness rides on the datetime being aware."""
    moment = TIME.now()
    assert moment.tzinfo is not None
    assert moment.utcoffset() is not None


def test_utc_offset_is_formatted_as_a_signed_hh_mm(service):
    service.set_clock(lambda: datetime(2026, 8, 8, 12, 0,
                                       tzinfo=timezone(timedelta(hours=1))))
    assert service.utc_offset() == "+01:00"
    service.set_clock(lambda: datetime(2026, 1, 8, 12, 0, tzinfo=timezone.utc))
    assert service.utc_offset() == "+00:00"


def test_a_naive_fake_clock_is_still_accepted(service):
    service.set_clock(lambda: datetime(2026, 8, 8, 3, 0))
    assert service.time_of_day() == NIGHT
    assert service.now().tzinfo is not None


# ── the snapshot is one atomic read ───────────────────────────────────────────

def test_snapshot_fields_all_describe_the_same_instant(service):
    service.set_clock(lambda: at(19, 45, month=11, day=3))
    snap = service.snapshot()
    assert snap.period == EVENING
    assert snap.greeting == EVENING
    assert snap.season == "autumn"
    assert snap.is_daytime is False
    assert snap.date.day == 3
    assert snap.moment.hour == 19


def test_is_daytime_is_true_only_for_morning_and_afternoon(service):
    for hour, expected in [(3, False), (9, True), (14, True), (19, False), (23, False)]:
        service.set_clock(lambda h=hour: at(h))
        assert service.is_daytime() is expected, hour


# ── the prompt line the model actually receives ───────────────────────────────

def test_prompt_line_states_the_period_and_that_it_is_not_daytime(service):
    service.set_clock(lambda: at(0, 31, month=8))
    line = service.prompt_line()
    assert "NIGHT" in line
    assert "NOT daytime" in line
    assert "summer" in line
    assert "2026" in line


def test_prompt_line_says_daytime_in_the_afternoon(service):
    service.set_clock(lambda: at(14, 0))
    line = service.prompt_line()
    assert "AFTERNOON" in line
    assert "NOT daytime" not in line


# ── every consumer reads this clock ───────────────────────────────────────────

def test_provider_router_temporal_line_comes_from_the_service(monkeypatch):
    """The system prompt must not carry its own idea of the time."""
    from orion_core import providers

    TIME.set_clock(lambda: at(0, 31))
    try:
        router = providers.ProviderRouter.__new__(providers.ProviderRouter)
        router._locality = ""
        line = router._temporal_line()
        assert "NIGHT" in line
        assert "NOT daytime" in line
    finally:
        TIME.set_clock(None)


def test_briefing_greeting_period_delegates_to_the_shared_bands():
    from orion_core.briefing import MorningBriefingService

    # 17:30 used to be "evening" here and "afternoon" in local_brain.
    assert MorningBriefingService.greeting_period(at(17, 30)) == AFTERNOON
    assert MorningBriefingService.greeting_period(at(18, 30)) == EVENING
    assert MorningBriefingService.greeting_period(at(0, 31)) == EVENING


def test_briefing_engine_period_uses_the_shared_boundaries():
    from orion_core.briefing_engine import DynamicBriefingEngine

    assert DynamicBriefingEngine.period(at(9, 0)) == "morning"
    assert DynamicBriefingEngine.period(at(13, 0)) == "midday"
    assert DynamicBriefingEngine.period(at(17, 30)) == "midday"   # was "evening"
    assert DynamicBriefingEngine.period(at(19, 0)) == "evening"


def test_local_brain_greeting_uses_the_shared_bands():
    from orion_core.local_brain import LocalBrain

    TIME.set_clock(lambda: at(0, 31))
    try:
        brain = LocalBrain.__new__(LocalBrain)
        reply = brain._intent_greeting("hello", "hello")
        assert reply is not None
        assert "morning" not in reply.lower()
    finally:
        TIME.set_clock(None)


# ── small helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("day,expected", [
    (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
    (21, "21st"), (22, "22nd"), (23, "23rd"), (30, "30th"), (31, "31st"),
])
def test_ordinals(day, expected):
    assert ordinal(day) == expected


def test_spoken_forms_are_words_not_digits(service):
    service.set_clock(lambda: at(22, 27))
    spoken = service.spoken_now()
    assert "22" not in spoken
    assert service.spoken_date() == "Saturday the 8th of August"
