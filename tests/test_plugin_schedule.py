"""
Plugins that run without being asked.

Cron is where the silent bugs live. A schedule that fires an hour early, or
runs half as often as intended, does not raise anything — it just quietly
misbehaves for a month before anyone notices. So the parser is tested against
known-answer dates rather than against itself.

The one that catches people: when both day-of-month and day-of-week are
restricted, cron runs the job when **either** matches. ``0 0 13 * 5`` is
"midnight on the 13th, and also every Friday" — not "Friday the 13th".
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orion_core.plugin_manifest import ManifestError, PluginManifest  # noqa: E402
from orion_core.plugin_runner import MUTE_AFTER, PluginScheduler  # noqa: E402
from orion_core.plugin_schedule import (  # noqa: E402
    ScheduleError,
    describe,
    parse_cron,
    parse_schedule,
)
from orion_core.plugin_vault import PluginVault, VaultError  # noqa: E402

#: A Saturday afternoon, so weekday arithmetic has a known start.
SATURDAY = datetime(2026, 9, 19, 14, 30)


# ── cron ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expression, expected", [
    ("0 6 * * *", datetime(2026, 9, 20, 6, 0)),
    ("*/15 * * * *", datetime(2026, 9, 19, 14, 45)),
    ("0 9 * * 1-5", datetime(2026, 9, 21, 9, 0)),
    ("@daily", datetime(2026, 9, 20, 0, 0)),
    ("@hourly", datetime(2026, 9, 19, 15, 0)),
    ("0 0 1 * *", datetime(2026, 10, 1, 0, 0)),
    ("0 12 * * sat", datetime(2026, 9, 26, 12, 0)),
    ("0 8 * * mon-fri", datetime(2026, 9, 21, 8, 0)),
    ("5,25,45 * * * *", datetime(2026, 9, 19, 14, 45)),
])
def test_when_a_schedule_next_fires(expression, expected):
    assert parse_cron(expression).next_due(SATURDAY) == expected


def test_a_leap_day_schedule_waits_for_a_leap_year():
    """2026 and 2027 are not leap years, so the next 29 February is 2028.

    The search has to look years ahead without giving up or looping.
    """
    assert parse_cron("0 0 29 2 *").next_due(SATURDAY) == datetime(2028, 2, 29, 0, 0)


def test_a_range_may_wrap_round_midnight():
    """"22-2" means late evening through the small hours, which is what
    someone writing it means."""
    schedule = parse_cron("0 22-2 * * *")
    assert schedule.matches(datetime(2026, 9, 19, 23, 0))
    assert schedule.matches(datetime(2026, 9, 19, 1, 0))
    assert not schedule.matches(datetime(2026, 9, 19, 12, 0))


def test_day_and_weekday_together_mean_either():
    """The classic cron bug, and it is silent — the job simply runs less often
    than anyone expected."""
    schedule = parse_cron("0 0 13 * 5")
    assert schedule.day_or_weekday is True
    assert schedule.matches(datetime(2026, 11, 13, 0, 0))   # the 13th
    assert schedule.matches(datetime(2026, 11, 6, 0, 0))    # a Friday
    assert not schedule.matches(datetime(2026, 11, 10, 0, 0))


def test_a_restricted_day_alone_does_not_become_or():
    schedule = parse_cron("0 0 13 * *")
    assert schedule.day_or_weekday is False
    assert not schedule.matches(datetime(2026, 11, 6, 0, 0))


def test_sunday_may_be_written_as_0_or_7():
    assert parse_cron("0 0 * * 0").weekdays == parse_cron("0 0 * * 7").weekdays


@pytest.mark.parametrize("bad", [
    "0 6 * *", "0 6 * * * *", "99 * * * *", "* 25 * * *", "0 0 32 * *",
    "*/0 * * * *", "0 0 * * xyz", "", "   ",
])
def test_a_schedule_that_cannot_mean_anything_is_refused(bad):
    with pytest.raises(ScheduleError):
        parse_cron(bad)


def test_the_error_says_which_field_was_wrong():
    with pytest.raises(ScheduleError) as caught:
        parse_cron("99 * * * *")
    assert "minute" in str(caught.value)


# ── intervals ─────────────────────────────────────────────────────────────────

def test_an_interval_schedule_counts_from_now():
    schedule = parse_schedule(interval_seconds=1800)
    assert schedule.next_due(SATURDAY) == SATURDAY + timedelta(seconds=1800)


def test_an_interval_that_is_really_a_loop_is_refused():
    with pytest.raises(ScheduleError):
        parse_schedule(interval_seconds=5)


def test_declaring_both_is_refused():
    """Two answers to "when does this run?" is a disagreement someone will
    lose an afternoon to."""
    with pytest.raises(ScheduleError):
        parse_schedule("cron(0 6 * * *)", 60)


def test_declaring_neither_means_on_request_only():
    assert parse_schedule() is None


def test_a_schedule_describes_itself_readably():
    assert describe(None) == "on request only"
    assert describe(parse_schedule(interval_seconds=3600)) == "every 1 hour"
    assert "next" in describe(parse_cron("0 6 * * *"))


# ── the manifest ──────────────────────────────────────────────────────────────

def _manifest(**extra) -> dict:
    return {"name": "news", "module": "news_tool.py",
            "description": "d", **extra}


def test_a_manifest_may_declare_a_schedule():
    manifest = PluginManifest.from_dict(
        _manifest(schedule="cron(0 6 * * *)", secrets=["NEWS_KEY"]))
    assert manifest.is_scheduled is True
    assert manifest.secrets == ("NEWS_KEY",)
    assert manifest.parsed_schedule is not None


def test_an_ordinary_plugin_is_unaffected():
    manifest = PluginManifest.from_dict(_manifest())
    assert manifest.is_scheduled is False
    assert manifest.parsed_schedule is None
    assert manifest.secrets == ()


def test_a_bad_schedule_is_caught_at_load_not_at_six_in_the_morning():
    """A plugin that says "cron(0 6 * *)" should be rejected while someone is
    looking at it."""
    with pytest.raises(ManifestError):
        PluginManifest.from_dict(_manifest(schedule="cron(0 6 * *)"))


# ── the vault ─────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path) -> PluginVault:
    store = PluginVault(tmp_path / "plugin_vault.json")
    store.set_secret("weather", "WEATHER_API_KEY", "wk-secret-123")
    store.set_secret("weather", "WEATHER_REGION", "uk")
    store.set_secret("dialler", "TWILIO_AUTH_TOKEN", "tw-secret-999")
    store.save()
    return PluginVault(tmp_path / "plugin_vault.json")


def test_a_plugin_gets_the_secret_it_declared(vault):
    assert vault.secrets_for("weather", ["WEATHER_API_KEY"]) == {
        "WEATHER_API_KEY": "wk-secret-123"}


def test_a_plugin_does_not_get_what_it_did_not_declare(vault):
    """The manifest is the contract. A plugin that starts reading something it
    never declared should fail visibly rather than succeed quietly."""
    got = vault.secrets_for("weather", ["WEATHER_API_KEY"])
    assert "WEATHER_REGION" not in got


def test_a_plugin_cannot_reach_another_plugins_secrets(vault):
    """The property that always holds, whatever the platform's crypto."""
    assert vault.secrets_for("weather", ["TWILIO_AUTH_TOKEN"]) == {}


def test_declaring_nothing_gets_nothing(vault):
    assert vault.secrets_for("weather", []) == {}


def test_secrets_are_not_written_to_disk_in_the_clear(tmp_path):
    store = PluginVault(tmp_path / "plugin_vault.json")
    if not store.status().encrypted:
        pytest.skip("no DPAPI and no cryptography on this machine")
    store.set_secret("x", "K", "a-very-secret-value")
    store.save()
    assert "a-very-secret-value" not in (
        tmp_path / "plugin_vault.json").read_text(encoding="utf-8")


def test_a_hand_written_plaintext_vault_still_works(tmp_path):
    """A vault the user edited by hand should work immediately, not fail
    because it was not written by this module."""
    path = tmp_path / "hand.json"
    path.write_text(json.dumps({"plugins": {"x": {"K": "plain"}}}),
                    encoding="utf-8")
    assert PluginVault(path).secrets_for("x", ["K"]) == {"K": "plain"}


def test_a_corrupt_vault_is_reported_not_ignored(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(VaultError):
        PluginVault(path)


def test_missing_secrets_are_reported_before_they_are_needed(vault):
    """Said now, not at three in the morning when the plugin fires."""
    assert vault.missing("weather", ["WEATHER_API_KEY", "ABSENT"]) == ["ABSENT"]


def test_the_vault_says_how_it_is_protecting_itself(tmp_path):
    """"Encrypted vault" invites more confidence than it earns, so the method
    in force is reportable rather than assumed."""
    status = PluginVault(tmp_path / "v.json").status()
    assert status.method in {"dpapi", "fernet", "plaintext"}
    assert status.detail


# ── the supervisor ────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, ok=True, text="done"):
        self.ok, self.text = ok, text


class _Dispatcher:
    def __init__(self, behaviour=None):
        self.calls = []
        self._behaviour = behaviour or (lambda name: _Result())

    async def dispatch(self, name, args):
        self.calls.append(name)
        return self._behaviour(name)


class _Signal:
    def __init__(self):
        self.lines = []

    def emit(self, line):
        self.lines.append(line)


class _Bus:
    def __init__(self):
        self.log = _Signal()


@pytest.fixture
def clock():
    holder = {"now": datetime(2026, 9, 19, 5, 59)}
    return holder


def _scheduler(dispatcher, clock, **kwargs):
    return PluginScheduler(dispatcher, _Bus(),
                           clock=lambda: clock["now"], **kwargs)


def test_a_scheduled_plugin_runs_when_it_is_due(clock):
    dispatcher = _Dispatcher()
    scheduler = _scheduler(dispatcher, clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="news", schedule="cron(0 6 * * *)")))

    asyncio.run(scheduler._tick())
    assert dispatcher.calls == [], "it ran before it was due"

    clock["now"] = datetime(2026, 9, 19, 6, 0)
    asyncio.run(scheduler._tick())
    assert dispatcher.calls == ["news"]


def test_it_is_rescheduled_after_running(clock):
    dispatcher = _Dispatcher()
    scheduler = _scheduler(dispatcher, clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="news", schedule="cron(0 6 * * *)")))
    clock["now"] = datetime(2026, 9, 19, 6, 0)
    asyncio.run(scheduler._tick())
    assert scheduler.entries["news"].next_due == datetime(2026, 9, 20, 6, 0)


def test_an_unscheduled_plugin_is_not_watched(clock):
    scheduler = _scheduler(_Dispatcher(), clock)
    assert scheduler.add(PluginManifest.from_dict(_manifest())) is None
    assert scheduler.entries == {}


def test_a_plugin_that_keeps_failing_is_muted(clock):
    """A broken plugin on a fifteen-minute interval writes ninety-six
    identical errors a day and buries everything else."""
    scheduler = _scheduler(_Dispatcher(lambda n: _Result(False, "boom")), clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="flaky", interval_seconds=30)))
    for _ in range(MUTE_AFTER + 2):
        clock["now"] += timedelta(seconds=31)
        asyncio.run(scheduler._tick())
    entry = scheduler.entries["flaky"]
    assert entry.muted is True
    assert entry.runs == MUTE_AFTER, "it kept running after being muted"


def test_a_muted_plugin_can_be_put_back(clock):
    scheduler = _scheduler(_Dispatcher(lambda n: _Result(False, "boom")), clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="flaky", interval_seconds=30)))
    for _ in range(MUTE_AFTER):
        clock["now"] += timedelta(seconds=31)
        asyncio.run(scheduler._tick())
    assert scheduler.unmute("flaky") is True
    assert scheduler.entries["flaky"].muted is False


def test_an_exception_in_a_plugin_is_contained(clock):
    """Nothing a plugin does may stop ORION — nobody is at the keyboard."""
    class _Boom:
        async def dispatch(self, name, args):
            raise RuntimeError("exploded")

    scheduler = _scheduler(_Boom(), clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="explodes", interval_seconds=30)))
    clock["now"] += timedelta(seconds=31)
    asyncio.run(scheduler._tick())
    assert "exploded" in scheduler.entries["explodes"].last.detail


def test_no_dispatcher_is_reported_rather_than_crashing(clock):
    scheduler = _scheduler(None, clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="orphan", interval_seconds=30)))
    clock["now"] += timedelta(seconds=31)
    asyncio.run(scheduler._tick())
    assert scheduler.entries["orphan"].last.ok is False


def test_a_missed_cron_window_is_a_missed_run(clock):
    """If the machine was asleep at 06:00 the job does not fire at 09:14 when
    it wakes. A briefing three hours late is worse than one that did not
    arrive."""
    dispatcher = _Dispatcher()
    scheduler = _scheduler(dispatcher, clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="news", schedule="cron(0 6 * * *)")))
    # Sleep through 06:00 and wake at 09:14.
    clock["now"] = datetime(2026, 9, 19, 9, 14)
    asyncio.run(scheduler._tick())
    assert dispatcher.calls == ["news"], (
        "the run that was due IS taken when we wake")
    # ...and exactly once, not once per missed minute.
    asyncio.run(scheduler._tick())
    assert dispatcher.calls == ["news"]


def test_the_status_report_says_what_is_scheduled(clock):
    scheduler = _scheduler(_Dispatcher(), clock)
    scheduler.add(PluginManifest.from_dict(
        _manifest(name="news", schedule="cron(0 6 * * *)")))
    row = scheduler.status()[0]
    assert row["plugin"] == "news"
    assert "cron(0 6 * * *)" in row["schedule"]
    assert row["muted"] is False
