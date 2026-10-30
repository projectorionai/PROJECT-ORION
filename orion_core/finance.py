"""
Personal & business finance (Mark XXVI, Phase 3) — an offline money brain.

An entrepreneur's shared need with the student: know the runway. This is a
first-class, fully-local ledger — accounts, transactions and subscriptions in
SQLite (``config/finance.db``) — with the one number that matters computed
honestly: how many months of cash are left at the current burn.

Deliberately manual/CSV-first: no bank API, no third-party dependency, nothing
leaves the machine. Bank connectors can layer on later; the ledger stands alone.
Mirrors the study/focus scaffold: dataclass records + a Store + an Engine + a
tool + a lazy dispatcher slot.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .constants import CONFIG_DIR
from .db import apply_pragmas

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# ── money ───────────────────────────────────────────────────────────────────
# Amounts are stored as REAL (a schema migration would buy nothing here), but
# every amount is quantised to whole pence on the way IN, and every total is
# added up in integer pence. Two things that floats got wrong: a model or a CSV
# could store 12.3456 and have fractions of a penny accumulate invisibly, and
# round() rounds halves to EVEN (2.675 -> 2.67), not the half-up people expect
# of money. Integer pence are exact, so totals are exact.


def _pence(amount: object) -> int:
    """*amount* in whole pence, rounded half-up. str() first, so 2.675 is 2.675."""
    try:
        value = Decimal(str(amount).strip() or "0")
    except (InvalidOperation, ValueError):
        raise ValueError(f"not an amount of money: {amount!r}") from None
    if not value.is_finite():
        raise ValueError(f"not an amount of money: {amount!r}")
    try:
        return int((value * 100).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    except (InvalidOperation, OverflowError, ValueError):
        raise ValueError(f"not a supported amount of money: {amount!r}") from None


def _money(pence: int | Decimal) -> float:
    """Whole pence back to pounds, half-up if a division left a fraction."""
    whole = Decimal(pence).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return float(whole / 100)


def _to_pennies(amount: object) -> float:
    return _money(_pence(amount))

#: Account kinds whose balance counts as spendable cash (credit is a liability).
LIQUID_KINDS = frozenset({"current", "savings", "cash"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Account:
    name: str
    kind: str = "current"                 # current | savings | credit | cash
    currency: str = "GBP"
    opening_balance: float = 0.0
    created_at: str = ""
    id: int | None = None


@dataclass
class Transaction:
    account_id: int
    amount: float                         # always positive; direction carries sign
    direction: str = "out"                # in | out
    at: str = ""
    category: str = ""
    merchant: str = ""
    note: str = ""
    source: str = "manual"                # manual | csv | rule
    id: int | None = None


@dataclass
class Subscription:
    name: str
    amount: float
    cadence_days: int = 30
    next_due: str = ""
    category: str = "subscription"
    active: bool = True
    id: int | None = None


def _store_locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return locked


class FinanceStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else (CONFIG_DIR / "finance.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # Mark XXVI: WAL + relaxed fsync - a committed write costs
        # 0.03 ms instead of 2.81 ms, and that time is paid on the
        # qasync/GUI thread. See orion_core/db.py.
        apply_pragmas(self._db)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, kind TEXT DEFAULT 'current',
                currency TEXT DEFAULT 'GBP', opening_balance REAL DEFAULT 0,
                created_at TEXT DEFAULT '');
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL, amount REAL NOT NULL,
                direction TEXT DEFAULT 'out', at TEXT DEFAULT '',
                category TEXT DEFAULT '', merchant TEXT DEFAULT '',
                note TEXT DEFAULT '', source TEXT DEFAULT 'manual');
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, amount REAL NOT NULL,
                cadence_days INTEGER DEFAULT 30, next_due TEXT DEFAULT '',
                category TEXT DEFAULT 'subscription', active INTEGER DEFAULT 1);
            CREATE INDEX IF NOT EXISTS idx_txn_at ON transactions(at);
            """)
        self._db.commit()

    # -- accounts --
    @_store_locked
    def add_account(self, a: Account) -> Account:
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self._db.execute("SELECT 1 FROM accounts WHERE lower(name)=lower(?)", (a.name,)).fetchone():
                raise ValueError(f"Account '{a.name}' already exists; its balance was not changed")
            cur = self._db.execute(
                "INSERT INTO accounts (name, kind, currency, opening_balance, created_at) "
                "VALUES (?,?,?,?,?)",
                (a.name, a.kind, a.currency, a.opening_balance, a.created_at or _iso(_now())))
        a.id = int(cur.lastrowid)
        return a

    @_store_locked
    def accounts(self) -> list[Account]:
        return [Account(id=r["id"], name=r["name"], kind=r["kind"],
                        currency=r["currency"], opening_balance=r["opening_balance"],
                        created_at=r["created_at"] or "")
                for r in self._db.execute("SELECT * FROM accounts ORDER BY id").fetchall()]

    @_store_locked
    def account_by_name(self, name: str) -> Account | None:
        r = self._db.execute("SELECT * FROM accounts WHERE lower(name)=lower(?)",
                             (name,)).fetchone()
        return (Account(id=r["id"], name=r["name"], kind=r["kind"], currency=r["currency"],
                        opening_balance=r["opening_balance"], created_at=r["created_at"] or "")
                if r else None)

    # -- transactions --
    @_store_locked
    def add_txn(self, t: Transaction) -> Transaction:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO transactions (account_id, amount, direction, at, category, "
                "merchant, note, source) VALUES (?,?,?,?,?,?,?,?)",
                (t.account_id, abs(t.amount), t.direction, t.at or _iso(_now()),
                 t.category, t.merchant, t.note, t.source))
        t.id = int(cur.lastrowid)
        return t

    @_store_locked
    def import_transactions(self, rows: list[tuple[str, float, str, str]], account: str = "") -> int:
        """One import is one commit, including any new account; errors roll it back."""
        if not rows:
            return 0
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            found = (self._db.execute("SELECT id FROM accounts WHERE lower(name)=lower(?) ORDER BY id LIMIT 1",
                                      (account,)).fetchone() if account else
                     self._db.execute("SELECT id FROM accounts ORDER BY id LIMIT 1").fetchone())
            if found is None:
                cur = self._db.execute(
                    "INSERT INTO accounts (name,kind,currency,opening_balance,created_at) VALUES (?,?,?,?,?)",
                    (account.strip() or "Main", "current", "GBP", 0, _iso(_now())))
                account_id = int(cur.lastrowid)
            else:
                account_id = int(found["id"])
            self._db.executemany(
                "INSERT INTO transactions (account_id,amount,direction,at,merchant,category,source) "
                "VALUES (?,?,?,?,?,?,?)",
                ((account_id, abs(amount), "in" if amount >= 0 else "out", at, desc, category, "csv")
                 for at, amount, desc, category in rows))
        return len(rows)

    @_store_locked
    def transactions(self, since: datetime | None = None,
                     until: datetime | None = None) -> list[Transaction]:
        clauses, values = [], []
        for operator, moment in ((">=", since), ("<=", until)):
            if moment is not None:
                clauses.append(f"at{operator}?")
                values.append(_iso(moment))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._db.execute("SELECT * FROM transactions" + where + " ORDER BY at", values).fetchall()
        return [Transaction(id=r["id"], account_id=r["account_id"], amount=r["amount"],
                            direction=r["direction"], at=r["at"] or "", category=r["category"] or "",
                            merchant=r["merchant"] or "", note=r["note"] or "",
                            source=r["source"] or "manual") for r in rows]

    # -- subscriptions --
    @_store_locked
    def add_subscription(self, s: Subscription) -> Subscription:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO subscriptions (name, amount, cadence_days, next_due, category, active) "
                "VALUES (?,?,?,?,?,?)",
                (s.name, s.amount, s.cadence_days, s.next_due, s.category, int(s.active)))
        s.id = int(cur.lastrowid)
        return s

    @_store_locked
    def subscriptions(self, active_only: bool = True) -> list[Subscription]:
        q = "SELECT * FROM subscriptions" + (" WHERE active=1" if active_only else "")
        return [Subscription(id=r["id"], name=r["name"], amount=r["amount"],
                             cadence_days=r["cadence_days"], next_due=r["next_due"] or "",
                             category=r["category"] or "", active=bool(r["active"]))
                for r in self._db.execute(q + " ORDER BY next_due").fetchall()]

    @_store_locked
    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


class FinanceEngine:
    def __init__(self, store: FinanceStore | None = None) -> None:
        self.store = store or FinanceStore()

    # -- authoring --
    def add_account(self, name: str, kind: str = "current",
                    opening_balance: float = 0.0, currency: str = "GBP") -> Account:
        kind = str(kind or "current").strip().lower()
        if kind not in LIQUID_KINDS | {"credit"}:
            raise ValueError("account kind must be current, savings, credit or cash")
        return self.store.add_account(Account(
            name=name.strip() or "Main", kind=kind,
            currency=currency, opening_balance=_to_pennies(opening_balance)))

    def _resolve_account(self, name: str = "") -> Account:
        """Find the account by name, else the first account, else create 'Main'."""
        if name:
            existing = self.store.account_by_name(name)
            if existing is not None:
                return existing
            return self.add_account(name)
        accounts = self.store.accounts()
        return accounts[0] if accounts else self.add_account("Main")

    def add_txn(self, amount: float, direction: str = "out", *, account: str = "",
                category: str = "", merchant: str = "", note: str = "",
                source: str = "manual", at: datetime | None = None) -> Transaction:
        amount = abs(_to_pennies(amount))
        at_text = _iso(at or _now())
        acc = self._resolve_account(account)
        return self.store.add_txn(Transaction(
            account_id=int(acc.id), amount=amount,
            direction=("in" if str(direction).lower() in {"in", "income", "credit"} else "out"),
            at=at_text, category=category, merchant=merchant, note=note, source=source))

    def import_csv(self, text: str, *, account: str = "") -> int:
        """Import rows of ``date, amount, description[, category]``. A negative
        amount is an expense (out), positive is income (in). Bad rows are skipped
        rather than aborting the import."""
        rows = []
        reader = csv.reader(io.StringIO(text.strip()))
        for row in reader:
            if len(row) < 2:
                continue
            head = row[0].strip().lower()
            if head in {"date", "when"} or not any(c.isdigit() for c in row[0]):
                continue                                   # header / non-date line
            try:
                when = _parse(row[0].strip())
                amount = _to_pennies(str(row[1]).replace(",", "").replace("£", "").replace("$", "").strip())
            except (ValueError, IndexError):
                continue
            desc = row[2].strip() if len(row) > 2 else ""
            category = row[3].strip() if len(row) > 3 else ""
            rows.append((_iso(when), amount, desc, category))
        return self.store.import_transactions(rows, account=account)

    def add_subscription(self, name: str, amount: float, cadence_days: int = 30,
                         next_due: datetime | None = None, category: str = "subscription") -> Subscription:
        if isinstance(cadence_days, bool) or not isinstance(cadence_days, int) or cadence_days <= 0:
            raise ValueError("subscription cadence_days must be a positive whole number")
        if not name.strip():
            raise ValueError("subscription needs a name")
        return self.store.add_subscription(Subscription(
            name=name.strip(), amount=_to_pennies(amount), cadence_days=int(cadence_days),
            next_due=_iso(next_due or (_now() + timedelta(days=int(cadence_days)))),
            category=category))

    # -- reporting --
    def _balances_pence(self, now: datetime | None = None) -> dict[int, int]:
        """Current balances: stored opening basis plus postings through one cutoff.

        balance() used to re-read every transaction once PER ACCOUNT, and
        report() reached it through liquid_cash() twice — so one report read
        the whole ledger about three times for every account.
        """
        totals = {int(acc.id): _pence(acc.opening_balance) for acc in self.store.accounts()}
        for t in self.store.transactions(until=now or _now()):
            if t.account_id in totals:
                amount = _pence(t.amount)
                totals[t.account_id] += amount if t.direction == "in" else -amount
        return totals

    def balance(self, account: str = "", *, now: datetime | None = None) -> float:
        totals = self._balances_pence(now)
        if not account:
            return _money(sum(totals.values()))
        acc = self.store.account_by_name(account)
        if acc is None:
            raise ValueError(f"No account named '{account}'")
        return _money(totals.get(int(acc.id), 0))

    def liquid_cash(self, *, now: datetime | None = None) -> float:
        totals = self._balances_pence(now)
        return _money(sum(totals.get(int(a.id), 0) for a in self.store.accounts()
                          if a.kind in LIQUID_KINDS))

    def monthly_net_burn(self, months: int = 3, now: datetime | None = None) -> float:
        """Average monthly (out − in) over the trailing window. Positive = burning."""
        now = now or _now()
        if isinstance(months, bool) or not isinstance(months, int) or months <= 0:
            raise ValueError("months must be a positive whole number")
        txns = self.store.transactions(since=now - timedelta(days=months * 30), until=now)
        out = sum(_pence(t.amount) for t in txns if t.direction == "out")
        inc = sum(_pence(t.amount) for t in txns if t.direction == "in")
        return _money(Decimal(out - inc) / max(1, months))

    def runway_months(self, months: int = 3, now: datetime | None = None) -> float | None:
        """Cash ÷ average monthly burn. None when not burning (income covers it)."""
        now = now or _now()
        burn = self.monthly_net_burn(months, now)
        if burn <= 0:
            return None
        return round(self.liquid_cash(now=now) / burn, 1)

    def upcoming_subscriptions(self, days: int = 14, now: datetime | None = None) -> list[Subscription]:
        now = now or _now()
        horizon = now + timedelta(days=days)
        out: list[Subscription] = []
        for s in self.store.subscriptions():
            if not s.next_due:
                continue
            try:
                if _parse(s.next_due) <= horizon:
                    out.append(s)
            except ValueError:
                continue
        return out

    def monthly_subscription_cost(self) -> float:
        return _money(sum(Decimal(_pence(s.amount)) * 30 / max(1, s.cadence_days)
                          for s in self.store.subscriptions()))

    def report(self, now: datetime | None = None) -> dict[str, object]:
        now = now or _now()
        cash = self.liquid_cash(now=now)
        burn = self.monthly_net_burn(now=now)
        return {
            "cash": cash,
            "accounts": len(self.store.accounts()),
            "monthly_burn": burn,
            "runway_months": round(cash / burn, 1) if burn > 0 else None,
            "subscriptions": len(self.store.subscriptions()),
            "monthly_subscriptions": self.monthly_subscription_cost(),
            "upcoming": len(self.upcoming_subscriptions(now=now)),
        }


__all__ = [
    "LIQUID_KINDS", "Account", "Transaction", "Subscription",
    "FinanceStore", "FinanceEngine",
]
