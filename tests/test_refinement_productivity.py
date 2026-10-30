"""Real ledger/date/input regressions; no personal databases or services."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import sqlite3

import pytest

from orion_core import goals as goals_mod
from orion_core.finance import FinanceEngine, FinanceStore
from orion_core.dispatch_productivity import ProductivityDispatchMixin as Tools

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)


@pytest.fixture
def engine(tmp_path):
    store = FinanceStore(tmp_path / "ledger.db")
    yield FinanceEngine(store)
    store.close()


def test_future_posting_is_excluded_from_trailing_burn(engine):
    engine.add_txn(90, at=NOW - timedelta(days=1))
    engine.add_txn(900, at=NOW + timedelta(days=1))
    assert engine.monthly_net_burn(now=NOW) == 30


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), "invalid"])
def test_invalid_transaction_does_not_create_account(engine, amount):
    with pytest.raises(ValueError):
        engine.add_txn(amount, account="Unwanted")
    assert engine.store.accounts() == []


def test_invalid_csv_creates_no_account(engine):
    assert engine.import_csv("date,amount\n2026-10-01,nan\ngarbage,12", account="Unwanted") == 0
    assert engine.store.accounts() == []


def test_csv_batch_rolls_back_all_rows_on_insert_failure(engine):
    engine.add_account("Main")
    engine.store._db.execute("CREATE TRIGGER reject_bad BEFORE INSERT ON transactions "
                            "WHEN NEW.merchant='bad' BEGIN SELECT RAISE(ABORT,'denied'); END")
    engine.store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        engine.import_csv("2026-10-01,10,good\n2026-10-02,20,bad", account="Main")
    assert engine.store.transactions() == []


def test_csv_uses_one_transaction_for_rows(engine):
    engine.add_account("Main")
    statements = []
    engine.store._db.set_trace_callback(statements.append)
    assert engine.import_csv("\n".join(f"2026-10-01,{i},row" for i in range(1, 51))) == 50
    assert sum(s == "COMMIT" for s in statements) == 1


def test_report_does_not_rescan_ledger_for_runway(engine, monkeypatch):
    engine.add_txn(90, at=NOW - timedelta(days=1))
    calls = []
    original = engine.store.transactions
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(engine.store, "transactions", counted)
    assert engine.report(now=NOW)["monthly_burn"] == 30
    assert len(calls) <= 2


def test_finance_speaks_the_amount_actually_recorded(engine):
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {"action": "spend", "amount": 2.675})
    assert result.ok and "Spent 2.68" in result.text
    assert engine.store.transactions()[0].amount == 2.68


def test_finance_unknown_action_does_not_report_success(engine):
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {"action": "delete_everything"})
    assert not result.ok and "Unsupported" in result.text


def test_finance_invalid_amount_is_actionable(engine):
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {"action": "spend", "amount": "invalid"})
    assert not result.ok and result.text
    assert engine.store.accounts() == []


def test_zero_confidence_is_not_replaced_by_default(tmp_path):
    from orion_core.decisions import DecisionJournal, DecisionStore
    store = DecisionStore(tmp_path / "decisions.db")
    try:
        journal = DecisionJournal(store)
        result = Tools.decision_tool(SimpleNamespace(decisions=journal), {
            "action": "record", "question": "Will it work?", "choice": "Try", "confidence": 0})
        assert result.ok
        assert journal.store.all()[0].confidence == 0
    finally:
        store.close()


def test_zero_hours_sleep_is_recorded(tmp_path):
    from orion_core.wellbeing import WellbeingEngine, WellbeingStore
    store = WellbeingStore(tmp_path / "wellbeing.db")
    try:
        result = Tools.wellbeing_tool(SimpleNamespace(wellbeing=WellbeingEngine(store)), {
            "action": "checkin", "sleep_hours": 0})
        assert result.ok and "slept 0h" in result.text
        assert store.all()[0].sleep_hours == 0
    finally:
        store.close()


def test_duplicate_account_does_not_double_report_cash(engine):
    engine.add_account("Main", opening_balance=100)
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {
        "action": "account", "name": "MAIN", "balance": 100})
    assert not result.ok
    assert len(engine.store.accounts()) == 1 and engine.liquid_cash() == 100


def test_account_kind_normalizes_case(engine):
    engine.add_account("Cash", kind="Current", opening_balance=100)
    assert engine.liquid_cash() == 100


def test_unknown_account_is_not_reported_as_zero(engine):
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {"action": "balance", "account": "Missing"})
    assert not result.ok and "Missing" in result.text


def test_zero_cadence_is_rejected_instead_of_defaulted(engine):
    result = Tools.finance_tool(SimpleNamespace(finance=engine), {
        "action": "subscription", "name": "Invalid", "amount": 5, "cadence_days": 0})
    assert not result.ok and engine.store.subscriptions() == []


def test_csv_failed_new_account_is_rolled_back(engine):
    engine.store._db.execute("CREATE TRIGGER deny_import BEFORE INSERT ON transactions "
                            "BEGIN SELECT RAISE(ABORT,'denied'); END")
    engine.store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        engine.import_csv("2026-10-01,10,first", account="Unwanted")
    assert engine.store.accounts() == [] and engine.store.transactions() == []


@pytest.mark.parametrize("operation", ["transaction", "subscription"])
def test_failed_manual_write_does_not_poison_next_import(engine, operation):
    engine.add_account("Main")
    table = "transactions" if operation == "transaction" else "subscriptions"
    engine.store._db.execute(f"CREATE TRIGGER fail_manual BEFORE INSERT ON {table} "
                            "BEGIN SELECT RAISE(ABORT,'denied'); END")
    engine.store._db.commit()
    with pytest.raises(sqlite3.IntegrityError):
        if operation == "transaction":
            engine.add_txn(10, account="Main")
        else:
            engine.add_subscription("Service", 10)
    assert not engine.store._db.in_transaction
    engine.store._db.execute("DROP TRIGGER fail_manual")
    engine.store._db.commit()
    assert engine.import_csv("2026-10-01,20,valid", account="Main") == 1
    assert engine.balance("Main") == 20


@pytest.mark.parametrize("future_amount", [-5000, 50000])
def test_future_csv_does_not_change_current_cash_or_runway(engine, future_amount):
    engine.add_account("Main", opening_balance=9000)
    engine.add_txn(3000, at=NOW - timedelta(days=1))
    assert engine.import_csv(f"{(NOW + timedelta(days=1)).isoformat()},{future_amount},future") == 1
    report = engine.report(now=NOW)
    assert (report["cash"], report["monthly_burn"], report["runway_months"]) == (6000, 1000, 6)
    assert engine.runway_months(now=NOW) == 6
    assert len(engine.store.transactions()) == 2  # future book entry is retained


@pytest.mark.parametrize("offset,expected", [(-1, 100), (0, 90), (1, 90)])
def test_account_cutoff_includes_transaction_exactly_when_due(engine, offset, expected):
    engine.add_account("Main", opening_balance=100)
    engine.add_txn(10, at=NOW)
    assert engine.balance("Main", now=NOW + timedelta(microseconds=offset)) == expected


def test_default_current_cash_uses_now_and_preserves_full_book(engine, monkeypatch):
    monkeypatch.setattr("orion_core.finance._now", lambda: NOW)
    engine.add_account("Main", opening_balance=100)
    engine.add_txn(50, at=NOW + timedelta(days=1))
    assert engine.balance() == engine.liquid_cash() == 100
    assert len(engine.store.transactions()) == 1


def test_explicit_asof_runway_does_not_use_wall_clock_for_cash(engine, monkeypatch):
    monkeypatch.setattr("orion_core.finance._now", lambda: NOW + timedelta(days=30))
    engine.add_account("Main", opening_balance=9000)
    engine.add_txn(3000, at=NOW - timedelta(days=1))
    engine.add_txn(5000, at=NOW + timedelta(days=1))
    assert engine.runway_months(now=NOW) == engine.report(now=NOW)["runway_months"] == 6


# ── personal goals (goals.py): speech, pace, projection, milestones, nudges ──
#
# The goals core shipped wired but untested beyond a WAL check. These pin the
# arithmetic the briefing, the tool and the nudges all read from, on a private
# goals.db and a fixed clock.

GOAL_START = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)


@pytest.fixture
def goal_clock(monkeypatch):
    clock = {"now": GOAL_START}
    monkeypatch.setattr(goals_mod, "_now", lambda: clock["now"])
    return clock


def _goal_engine(path):
    return goals_mod.GoalsEngine(goals_mod.GoalStore(path), import_legacy=False)


@pytest.fixture
def goals(tmp_path, goal_clock):
    engine = _goal_engine(tmp_path / "goals.db")
    yield engine
    engine.close()


def _at(clock, day):
    clock["now"] = GOAL_START + timedelta(days=day)


@pytest.mark.parametrize("said, amount, unit, absolute", [
    ("I ran 5k today", 5.0, "km", False),
    ("log two hours of revision", 2.0, "h", False),
    ("an hour and a half of piano", 1.5, "h", False),
    ("I weigh 82.5kg", 82.5, "kg", True),
    ("saved £250", 250.0, "gbp", False),
    ("read 30 pages", 30.0, "pages", False),
])
def test_progress_is_read_from_ordinary_speech(said, amount, unit, absolute):
    parsed = goals_mod.parse_progress(said)
    assert (parsed.amount, parsed.unit, parsed.absolute) == (amount, unit, absolute)


def test_spoken_progress_lands_on_the_right_goal(goals):
    run = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    car = goals.create("Save £2,000 for a car", unit="£", target=2000)
    goal, _entry, message = goals.log_progress(text="I ran 5k this morning")
    assert goal.id == run.id and "Logged 5 km" in message
    goal, _entry, _message = goals.log_progress(text="saved 1k this month")
    assert goal.id == car.id and goals.assess(car)["current"] == 1000


def test_a_goal_with_nothing_logged_is_not_started_never_on_track(goals, goal_clock):
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-12-31")
    _at(goal_clock, 1)
    status = goals.assess(goal)
    assert status["pace"] == "not started" and status["projected"] == ""


@pytest.mark.parametrize("logged, day, pace", [
    (60, 30, "ahead"),
    (30, 30, "on track"),
    (20, 30, "behind"),
    (10, 45, "at risk"),
    (10, 95, "overdue"),
])
def test_pace_follows_the_logged_rate_against_the_deadline(goals, goal_clock, logged, day, pace):
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")   # 90 days
    goals.log_progress(goal.id, amount=logged, unit="km")
    _at(goal_clock, day)
    assert goals.assess(goal)["pace"] == pace


def test_an_at_risk_projection_says_when_and_how_much_a_week(goals, goal_clock):
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    goals.log_progress(goal.id, amount=10, unit="km")
    _at(goal_clock, 45)                                   # 16 Oct, half the time gone
    status = goals.assess(goal)
    assert status["pace"] == "at risk" and status["percent"] == 10
    assert status["projected"] > "2026-11-30", "at this rate it lands after the deadline"
    assert round(status["needed_per_week"], 1) == 14.0    # 90 km over 45 days
    assert "about 14 km a week" in goals.describe(goal.id)


def test_one_early_log_is_not_a_projection(goals, goal_clock):
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    goals.log_progress(goal.id, amount=5, unit="km")
    assert goals.assess(goal)["projected"] == "", "one log on day one is no rate"


def test_a_reading_goal_takes_the_latest_reading_not_the_sum(goals):
    goal = goals.create("Lose weight", unit="kg", baseline=90, target=80)
    assert goal.mode == "level"
    goals.log_progress(goal.id, text="I weigh 88kg")
    goals.log_progress(goal.id, text="I weigh 86.5kg")
    status = goals.assess(goal)
    assert status["current"] == 86.5 and status["percent"] == 35


def test_milestones_measure_an_unmeasured_goal_and_fall_due(goals, goal_clock):
    goal = goals.create("Finish the dissertation", deadline="2027-04-30")
    for title, due in (("Literature review", "2026-10-01"), ("Methods", "2026-12-01"),
                       ("Results", "2027-02-01"), ("Write-up", "2027-04-01")):
        goals.add_milestone(goal.id, title, due)
    assert goals.complete_milestone(goal.id, "next").title == "Literature review"
    _at(goal_clock, 100)                                   # 10 Dec: Methods is overdue
    status = goals.assess(goal)
    assert status["percent"] == 25 and status["next_milestone"].title == "Methods"
    assert [m.title for m in status["due_milestones"]] == ["Methods"]
    assert any("Methods" in action for action in goals.attention()["actions"])


def test_at_risk_goals_are_nudged_once_a_day_even_across_a_restart(tmp_path, goal_clock):
    engine = _goal_engine(tmp_path / "goals.db")
    goal = engine.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    engine.log_progress(goal.id, amount=10, unit="km")
    engine.add_milestone(goal.id, "Half way", "2026-10-10")
    _at(goal_clock, 45)
    first = engine.take_nudges()
    assert first[0].startswith("Goal at risk: 'Run 100 km' is at 10%")
    assert "about 14 km a week lands it by 30 Nov" in first[0]
    assert first[1] == "Milestone due since 10 Oct: 'Half way' for 'Run 100 km'."
    assert engine.take_nudges() == [], "the same day says nothing twice"
    engine.close()
    engine = _goal_engine(tmp_path / "goals.db")             # ORION restarted
    try:
        assert engine.take_nudges() == [], "a restart does not raise them again"
        _at(goal_clock, 46)
        assert len(engine.take_nudges()) == 2, "the next day they are raised again"
        assert "at risk" in goals_mod.briefing_section(engine)
    finally:
        engine.close()


def test_a_goal_merely_behind_is_nudged_every_few_days(goals, goal_clock):
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    goals.log_progress(goal.id, amount=20, unit="km")
    seen = []
    for day in range(30, 34):
        _at(goal_clock, day)
        assert goals.assess(goal)["pace"] == "behind"
        seen.append(len(goals.take_nudges()))
    assert seen == [1, 0, 0, 1]


class _Signals:
    """A bus that records every emit, by signal name."""

    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: self.emitted.append((name, args)),
                               connect=lambda *_args: None)


def test_nudges_wait_for_admission_and_a_quiet_conversation(goals):
    gate = {"admitted": False, "quiet": True}
    admission = SimpleNamespace(should_run=lambda _name: gate["admitted"])
    nudger = goals_mod.GoalNudges(goals, _Signals(), admission=admission,
                                  quiet=lambda: gate["quiet"])
    assert not nudger.ready(), "admission deferred background work"
    gate.update(admitted=True, quiet=False)
    assert not nudger.ready(), "ORION is speaking or mid-turn"
    gate["quiet"] = True
    assert nudger.ready()


def test_a_nudge_is_a_banner_never_speech(goals, goal_clock):
    bus = _Signals()
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    goals.log_progress(goal.id, amount=10, unit="km")
    _at(goal_clock, 45)
    nudger = goals_mod.GoalNudges(goals, bus, admission=SimpleNamespace(should_run=lambda _n: True))
    nudger.announce(nudger.check())
    assert goals.attention_cached()["at_risk"][0]["title"] == "Run 100 km", (
        "the pass refreshed the summary callers on the loop read")
    assert {name for name, _args in bus.emitted} == {"banner", "dashboard_event"}
    banner = next(args for name, args in bus.emitted if name == "banner")
    assert banner[0].startswith("Goal at risk") and banner[1] == 2


def test_the_nudge_loop_registers_with_admission_and_runs_only_when_admitted(goals, goal_clock,
                                                                             monkeypatch):
    import asyncio
    goal = goals.create("Run 100 km", unit="km", target=100, deadline="2026-11-30")
    goals.log_progress(goal.id, amount=10, unit="km")
    _at(goal_clock, 45)
    monkeypatch.setattr(goals_mod.GoalNudges, "FIRST_DELAY_S", 0.0)
    monkeypatch.setattr(goals_mod.GoalNudges, "RETRY_S", 0.01)
    monkeypatch.setattr(goals_mod.GoalNudges, "INTERVAL_S", 0.01)
    asked, loops = [], []
    admission = SimpleNamespace(
        should_run=lambda name: asked.append(name) or len(asked) >= 3,
        register_loop=lambda name, cls: loops.append(("register", name, cls)),
        unregister_loop=lambda name: loops.append(("unregister", name)))
    bus = _Signals()
    nudger = goals_mod.GoalNudges(goals, bus, admission=admission, quiet=lambda: True)

    async def scenario():
        task = asyncio.create_task(nudger.run())
        for _ in range(500):
            if any(name == "banner" for name, _args in bus.emitted):
                break
            await asyncio.sleep(0.01)
        nudger.stop()
        await asyncio.wait_for(task, 5)

    asyncio.run(scenario())
    assert asked[:3] == ["goal_nudges"] * 3, "nothing ran before admission allowed it"
    assert loops == [("register", "goal_nudges", "background"), ("unregister", "goal_nudges")]
    assert sum(name == "banner" for name, _args in bus.emitted) == 1
