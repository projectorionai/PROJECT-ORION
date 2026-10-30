"""Decision writes must roll back when SQLite refuses the commit."""
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_core.decisions import DecisionJournal, DecisionStore
from orion_core.time_service import TIME


def refuse_commit(connection):
    def authorise(action, operation, *_):
        if action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    connection.set_authorizer(authorise)


def test_failed_add_commit_leaves_no_pending_row_to_leak_into_next_write(tmp_path):
    store = DecisionStore(tmp_path / "decisions.db")
    journal = DecisionJournal(store)
    try:
        refuse_commit(store._db)
        with pytest.raises(sqlite3.DatabaseError):
            journal.record("synthetic rejected question", "rejected choice")
        store._db.set_authorizer(None)
        assert store.all() == []
        journal.record("accepted question", "accepted choice")
        assert [d.question for d in store.all()] == ["accepted question"]
    finally:
        store.close()


def test_failed_resolution_commit_retains_prior_open_decision(tmp_path):
    store = DecisionStore(tmp_path / "decisions.db")
    journal = DecisionJournal(store)
    try:
        decision = journal.record("synthetic question", "choice")
        refuse_commit(store._db)
        with pytest.raises(sqlite3.DatabaseError):
            journal.resolve(decision.id, "right", lesson="not saved")
        store._db.set_authorizer(None)
        assert store.get(decision.id).outcome == ""
        journal.record("second question", "choice")
        assert store.get(decision.id).outcome == ""
    finally:
        store.close()


def test_default_review_horizon_uses_orions_local_calendar_clock(tmp_path, monkeypatch):
    now = datetime(2026, 10, 7, 0, 30, tzinfo=timezone(timedelta(hours=1)))
    monkeypatch.setattr(TIME, "_clock", lambda: now)
    store = DecisionStore(tmp_path / "decisions.db")
    try:
        decision = DecisionJournal(store).record("synthetic question", "choice", review_days=1)
        assert datetime.fromisoformat(decision.made_at) == now
        assert datetime.fromisoformat(decision.review_at) == now + timedelta(days=1)
    finally:
        store.close()
