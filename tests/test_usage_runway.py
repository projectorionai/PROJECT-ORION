"""
How much longer ORION can keep going today.

  "there must be an estimated time of usage remaining too in ORION"

The token ledger already recorded every call. What it could not answer is the
only question anyone actually asks about usage: am I going to run out, and
when? That needs a BUDGET (which the provider knows and the ledger does not)
and a BURN RATE measured recently enough to reflect what he is doing now.

The estimate is deliberately conservative and deliberately vague, and both are
design decisions rather than sloppiness. An optimistic runway is worse than no
runway, because it is believed right up until it fails; and "about two hours"
is honest where "1h 47m" claims a precision that does not exist.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.usage_runway import (  # noqa: E402
    DEFAULT_DAILY_TOKENS, RATE_WINDOW_MINUTES, Budget, Runway, UsageRunway,
    _human_duration,
)


class _Ledger:
    def __init__(self, used: int = 0, recent_tokens: int = 0,
                 providers: list[str] | None = None):
        self.used = used
        self.recent = recent_tokens
        self.providers = providers or ["groq"]

    def summary(self, filters=None):
        return {"total_tokens": self.used}

    def timeseries(self, bucket="hour", filters=None):
        return [{"at": time.time() - 600, "tokens": self.recent}]

    def by_dimension(self, dimension, filters=None):
        return [{"provider": name} for name in self.providers]


@pytest.fixture
def state(tmp_path):
    return tmp_path / "budgets.json"


# ── the estimate ─────────────────────────────────────────────────────────────

def test_it_reports_what_is_left(state):
    runway = UsageRunway(_Ledger(120_000, 9_000), state).runway("groq")
    assert runway.remaining == DEFAULT_DAILY_TOKENS["groq"] - 120_000
    assert "left at the current rate" in runway.describe()


def test_an_exhausted_allowance_says_so_plainly(state):
    runway = UsageRunway(_Ledger(500_000, 9_000), state).runway("groq")
    assert runway.remaining == 0
    assert runway.minutes_left == 0.0
    assert "allowance is gone" in runway.describe()


def test_idle_gives_no_estimate_rather_than_an_infinite_one(state):
    """Nothing running means there is no rate to project from. Saying
    'unlimited' would be a lie the moment he starts working again."""
    runway = UsageRunway(_Ledger(50_000, 0), state).runway("groq")
    assert runway.minutes_left is None
    assert "no rate to project from" in runway.describe()


def test_the_rate_uses_a_recent_window_not_the_daily_mean():
    """A quiet morning must not hide a heavy afternoon — that would make the
    estimate optimistic exactly when it matters."""
    assert 10 <= RATE_WINDOW_MINUTES <= 120


def test_old_activity_does_not_count_towards_the_current_rate(state):
    class _Stale(_Ledger):
        def timeseries(self, bucket="hour", filters=None):
            return [{"at": time.time() - 86_400, "tokens": 999_999}]

    assert UsageRunway(_Stale(), state).tokens_per_minute("groq") == 0.0


def test_a_provider_with_no_known_budget_says_so(state):
    runway = UsageRunway(_Ledger(1_000, 100), state).runway("some-new-api")
    assert "no budget known" in runway.describe()
    assert runway.minutes_left is not None or runway.budget == 0


# ── phrasing ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("minutes,expected", [
    (2, "a few minutes"),
    (20, "about 20 minutes"),
    (65, "about an hour"),
    (200, "about 3 hours"),
    (900, "the rest of the day"),
])
def test_durations_are_phrased_as_estimates(minutes, expected):
    assert _human_duration(minutes) == expected


def test_no_duration_reads_as_a_countdown():
    """'1h 47m' claims a precision this cannot have."""
    for minutes in (3, 12, 47, 90, 300, 2000):
        phrase = _human_duration(minutes)
        assert ":" not in phrase
        assert "h " not in phrase


def test_about_is_never_doubled():
    """The helper already says 'about'; a caller repeating it produced
    'about about an hour'."""
    runway = Runway("groq", used=480_000, budget=500_000, tokens_per_minute=300)
    assert "about about" not in runway.describe()


# ── budgets ──────────────────────────────────────────────────────────────────

def test_a_known_provider_gets_a_default_budget(state):
    assert UsageRunway(_Ledger(), state).budget_for("groq").daily_tokens > 0


def test_a_learned_limit_overrides_the_table(state):
    """The provider knows its own tier; the table is a guess that goes stale,
    and a limit wrong in the generous direction is the exact failure this
    exists to prevent."""
    runway = UsageRunway(_Ledger(), state)
    before = runway.budget_for("groq").daily_tokens
    runway.learn_from_limit("groq", 42_000)
    after = runway.budget_for("groq")
    assert after.daily_tokens == 42_000 != before
    assert "learned" in after.source


def test_a_nonsense_limit_is_ignored(state):
    runway = UsageRunway(_Ledger(), state)
    assert runway.learn_from_limit("groq", 0) is None
    assert runway.budget_for("groq").daily_tokens > 0


def test_budgets_survive_a_restart(state):
    UsageRunway(_Ledger(), state).set_budget("groq", 77_000)
    assert UsageRunway(_Ledger(), state).budget_for("groq").daily_tokens == 77_000


def test_the_source_of_the_number_is_recorded(state):
    """A default and a learned limit deserve different confidence, so the
    report has to be able to say which it used."""
    runway = UsageRunway(_Ledger(), state)
    assert runway.budget_for("groq").source == "default table"
    runway.set_budget("groq", 1000, "user")
    assert runway.budget_for("groq").source == "user"


def test_a_corrupt_state_file_is_survivable(state):
    state.write_text("{ not json", encoding="utf-8")
    assert UsageRunway(_Ledger(), state).budget_for("groq").daily_tokens > 0


# ── the report ───────────────────────────────────────────────────────────────

def test_the_report_warns_when_something_is_nearly_out(state):
    report = UsageRunway(_Ledger(495_000, 9_000, ["groq"]), state).report(["groq"])
    assert "Heads up" in report


def test_the_report_is_quiet_when_there_is_plenty(state):
    report = UsageRunway(_Ledger(1_000, 90, ["groq"]), state).report(["groq"])
    assert "Heads up" not in report


def test_no_usage_at_all_says_so(state):
    class _Empty(_Ledger):
        def by_dimension(self, dimension, filters=None):
            return []

    assert "nothing to project from" in UsageRunway(_Empty(), state).report()


def test_a_broken_ledger_does_not_raise(state):
    class _Broken:
        def summary(self, filters=None):
            raise RuntimeError("database is locked")

        def timeseries(self, bucket="hour", filters=None):
            raise RuntimeError("database is locked")

        def by_dimension(self, dimension, filters=None):
            raise RuntimeError("database is locked")

    runway = UsageRunway(_Broken(), state)
    assert runway.used_today("groq") == 0
    assert runway.tokens_per_minute("groq") == 0.0
    assert runway.report()


# ── it is reachable ──────────────────────────────────────────────────────────

def test_the_tool_answers_how_much_is_left():
    import inspect

    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    source = inspect.getsource(ProductivityDispatchMixin.token_usage_tool)
    assert "UsageRunway" in source
    assert "remaining" in source


def test_the_schema_tells_the_model_when_to_use_it():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS

    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "token_usage")
    assert "action" in tool["parameters"]["properties"]
    assert "how long have I got" in tool["description"]
