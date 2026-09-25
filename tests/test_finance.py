"""
Finance (Mark XXVI, Phase 3) — the offline money ledger.

The one number that matters is runway, so it is tested hardest: cash over burn,
None when income covers the outgoings, and honest about the trailing window.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from orion_core.finance import FinanceEngine, FinanceStore  # noqa: E402

T0 = datetime(2026, 8, 12, tzinfo=timezone.utc)


@pytest.fixture()
def engine(tmp_path):
    return FinanceEngine(FinanceStore(tmp_path / "finance.db"))


def test_balance_folds_opening_plus_transactions(engine):
    engine.add_account("Main", kind="current", opening_balance=1000.0)
    engine.add_txn(200.0, "out", account="Main", merchant="ads")
    engine.add_txn(50.0, "in", account="Main")
    assert engine.balance("Main") == 850.0


def test_a_transaction_auto_creates_an_account(engine):
    engine.add_txn(30.0, "out")
    assert len(engine.store.accounts()) == 1
    assert engine.liquid_cash() == -30.0


def test_liquid_cash_excludes_credit_accounts(engine):
    engine.add_account("Cash", kind="cash", opening_balance=500.0)
    engine.add_account("Card", kind="credit", opening_balance=-200.0)
    assert engine.liquid_cash() == 500.0        # the credit card is not cash


def test_runway_is_cash_over_burn(engine):
    # Open with 9000, burn 3000 over the trailing 3 months → cash 6000 now,
    # burn 1000/mo → 6 months of runway.
    engine.add_account("Main", kind="current", opening_balance=9000.0)
    for m in range(3):
        engine.add_txn(1000.0, "out", account="Main",
                       at=T0 - timedelta(days=20 + 30 * m))
    assert engine.liquid_cash() == 6000.0
    assert engine.monthly_net_burn(now=T0) == 1000.0
    assert engine.runway_months(now=T0) == 6.0


def test_no_runway_when_income_covers_the_burn(engine):
    engine.add_account("Main", opening_balance=1000.0)
    engine.add_txn(500.0, "out", account="Main", at=T0 - timedelta(days=10))
    engine.add_txn(900.0, "in", account="Main", at=T0 - timedelta(days=9))
    assert engine.monthly_net_burn(now=T0) < 0          # net positive
    assert engine.runway_months(now=T0) is None


def test_upcoming_subscriptions_within_the_window(engine):
    engine.add_subscription("Adobe", 30.0, cadence_days=30,
                            next_due=T0 + timedelta(days=5))
    engine.add_subscription("Domain", 12.0, cadence_days=365,
                            next_due=T0 + timedelta(days=200))
    soon = engine.upcoming_subscriptions(days=14, now=T0)
    assert [s.name for s in soon] == ["Adobe"]


def test_monthly_subscription_cost_normalises_cadence(engine):
    engine.add_subscription("Monthly", 30.0, cadence_days=30)
    engine.add_subscription("Yearly", 120.0, cadence_days=360)   # ~10/mo
    assert engine.monthly_subscription_cost() == pytest.approx(40.0, abs=1.0)


def test_csv_import_reads_signed_amounts(engine):
    csv_text = (
        "date,amount,description\n"
        "2026-08-01,-45.00,Ad spend\n"
        "2026-08-02,500,Client payment\n"
        "garbage,line,skip\n"
    )
    n = engine.import_csv(csv_text, account="Main")
    assert n == 2
    assert engine.balance("Main") == 455.0        # 500 in − 45 out


def test_report_is_a_complete_picture(engine):
    # 1500 spent across the trailing 3 months → 500/mo average burn.
    engine.add_account("Main", opening_balance=2000.0)
    for m in range(3):
        engine.add_txn(500.0, "out", account="Main", at=T0 - timedelta(days=5 + 30 * m))
    r = engine.report(now=T0)
    assert r["cash"] == 500.0
    assert r["monthly_burn"] == 500.0
    assert set(r) >= {"cash", "monthly_burn", "runway_months", "subscriptions"}


# ── tool wiring ───────────────────────────────────────────────────────────────

def test_the_finance_tool_is_registered():
    import inspect
    from orion_core.dispatcher import OrionDispatcher
    assert '"finance"' in inspect.getsource(OrionDispatcher)
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    tool = next(t for t in TOOL_DECLARATIONS if t["name"] == "finance")
    for key in ("action", "amount", "account", "kind"):
        assert key in tool["parameters"]["properties"]


class _Bag:
    def __init__(self, dbpath):
        self.finance = FinanceEngine(FinanceStore(dbpath))


def _tool(stub, args):
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    return ProductivityDispatchMixin.finance_tool(stub, args)


def test_tool_spend_balance_runway_flow(tmp_path):
    stub = _Bag(tmp_path / "f.db")
    _tool(stub, {"action": "account", "name": "Main", "kind": "current", "balance": 1000})
    r = _tool(stub, {"action": "spend", "amount": 250, "account": "Main", "note": "gear"})
    assert r.ok and "Spent 250" in r.text
    r = _tool(stub, {"action": "balance"})
    assert "750" in r.text
