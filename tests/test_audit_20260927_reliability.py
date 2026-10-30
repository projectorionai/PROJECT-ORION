"""Fault injection for the September 27 scheduling, workflow and sync audit."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
import sqlite3
from types import SimpleNamespace

import pytest

from orion_core.admission import Admission
from orion_core.data import ToolResult
from orion_core.job_store import JobStore
from orion_core.plugin_manifest import ManifestError, PluginManifest
from orion_core.plugin_runner import PluginScheduler
from orion_core.plugin_schedule import Schedule, ScheduleError, parse_cron, parse_schedule
from orion_core.sync.journal import SyncJournal
from orion_core.workflow_engine import WorkflowEngine


@pytest.mark.parametrize("expression", [
    "70-5 * * * *", "0 25-2 * * *", "0 0 32-2 * *", "0 0 * 13-2 *",
    "0 0 * * 8-2", "0 0 * mon *", "jan 0 * * *", "0 0 * * jan",
])
def test_cron_rejects_invalid_range_endpoints_and_names(expression):
    with pytest.raises(ScheduleError):
        parse_cron(expression)


def test_sunday_alias_does_not_collapse_a_full_week_range():
    assert parse_cron("0 0 * * 0-7").weekdays == frozenset(range(7))
    assert parse_cron("0 0 * * 0-7/2").weekdays == frozenset({0, 2, 4, 6})


@pytest.mark.parametrize("interval", [float("nan"), float("inf"), -float("inf"), -30, 1e300, 10**400])
def test_invalid_intervals_are_rejected_at_parse_and_manifest_load(interval):
    with pytest.raises(ScheduleError):
        parse_schedule(interval_seconds=interval)
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"name": "audit", "module": "audit.py", "interval_seconds": interval})


def test_impossible_date_search_has_a_day_scale_work_budget(monkeypatch):
    calls = 0
    original = Schedule.matches

    def counted(self, when):
        nonlocal calls
        calls += 1
        assert calls <= 1500, "next_due still scans millions of individual minutes"
        return original(self, when)

    monkeypatch.setattr(Schedule, "matches", counted)
    assert parse_cron("0 0 31 2 *").next_due(datetime(2026, 1, 1)) is None


@pytest.mark.parametrize("schedule", [parse_cron("* * * * *"), parse_schedule(interval_seconds=30)])
def test_no_future_date_is_reported_cleanly_at_datetime_limit(schedule):
    assert schedule.next_due(datetime.max) is None


@pytest.mark.parametrize("change", ["remove", "mute", "replace", "reschedule"])
def test_scheduled_plugin_changed_while_another_runs_is_not_executed(change):
    calls = []
    clock = [datetime(2026, 9, 27, 9)]

    async def dispatch(name, args):
        calls.append(name)
        if change == "remove":
            scheduler.remove("second")
        elif change == "mute":
            scheduler.entries["second"].muted = True
        elif change == "replace":
            scheduler.add(PluginManifest.from_dict({"name": "second", "module": "second.py", "interval_seconds": 60}))
        else:
            scheduler.entries["second"].next_due = clock[0] + timedelta(hours=1)
        return ToolResult("done")

    scheduler = PluginScheduler(SimpleNamespace(dispatch=dispatch), clock=lambda: clock[0])
    for name in ("first", "second"):
        scheduler.add(PluginManifest.from_dict({"name": name, "module": name + ".py", "interval_seconds": 30}))
    clock[0] += timedelta(seconds=31)
    asyncio.run(scheduler._tick())
    assert calls == ["first"]


@pytest.fixture
def journal(tmp_path):
    value = SyncJournal(tmp_path / "journal.db", node_id="local")
    yield value
    value.close()


def test_rejected_sync_batch_rolls_back_entries_and_logical_clock(journal):
    remote = {"node_id": "peer", "lamport_ts": 50, "tier": "notes", "category": "notes", "key": "k", "payload": "partial"}
    with pytest.raises(KeyError):
        journal.apply_remote([remote, {}])
    assert journal.all_entries() == []
    assert not journal.conn.in_transaction
    assert journal.record("notes", "notes", "local", "kept").lamport_ts == 1
    assert [e.payload for e in journal.all_entries()] == ["kept"]


def test_interrupted_sync_generator_rolls_back(journal):
    def entries():
        yield {"node_id": "peer", "lamport_ts": 50, "payload": "partial"}
        raise RuntimeError("transfer interrupted")

    with pytest.raises(RuntimeError, match="transfer interrupted"):
        journal.apply_remote(entries())
    journal.record("notes", "notes", "local", "kept")
    assert [e.payload for e in journal.all_entries()] == ["kept"]


def test_failed_local_sync_write_does_not_advance_clock(journal):
    journal.conn.execute("CREATE TRIGGER reject_insert BEFORE INSERT ON journal BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
    journal.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        journal.record("notes", "notes", "k", "failed")
    assert not journal.conn.in_transaction
    journal.conn.execute("DROP TRIGGER reject_insert")
    journal.conn.commit()
    assert journal.record("notes", "notes", "k", "kept").lamport_ts == 1


class Signal:
    def emit(self, *args):
        pass


class Dispatcher:
    def __init__(self):
        self.calls = []

    def handler_table(self):
        return {"research": self.dispatch}

    async def dispatch(self, name, args):
        self.calls.append((name, deepcopy(args)))
        return ToolResult("done")


@pytest.fixture
def engine(tmp_path, monkeypatch):
    from orion_core import jobs
    monkeypatch.setattr(jobs, "CONTROLS", {})
    monkeypatch.setattr(jobs, "_RESUMERS", {})
    store = JobStore()
    signal = Signal()
    value = WorkflowEngine(SimpleNamespace(log=signal, dashboard_event=signal, banner=signal),
                           dispatcher=Dispatcher(), path=tmp_path / "workflows.json",
                           store=store, admission=Admission())
    yield value
    store.close()


def fail_save(*args, **kwargs):
    raise OSError("disk full")


@pytest.mark.parametrize("operation", ["create", "replace", "remove"])
def test_workflow_failed_save_is_reported_and_preserves_state(engine, monkeypatch, operation):
    if operation != "create":
        assert engine.define("audit", [{"tool": "research", "args": {"topic": "old"}}]).ok
    before = deepcopy(engine.definitions)
    disk = engine.path.read_bytes() if engine.path.exists() else None
    monkeypatch.setattr("orion_core.workflow_engine.atomic_write_text", fail_save)
    result = (engine.remove("audit") if operation == "remove" else
              engine.define("audit", [{"tool": "research", "args": {"topic": "new"}}]))
    assert not result.ok
    assert engine.definitions == before
    assert (engine.path.read_bytes() if engine.path.exists() else None) == disk


def test_workflow_definition_owns_nested_arguments(engine):
    args = {"nested": {"value": "original"}}
    assert engine.define("audit", [{"tool": "research", "args": args}]).ok
    args["nested"]["value"] = "changed"
    assert engine.definitions["audit"]["steps"][0]["args"]["nested"]["value"] == "original"


def test_running_workflow_owns_its_definition_snapshot(engine):
    engine.define("audit", [{"tool": "research", "args": {"nested": {"value": "original"}}}])

    async def scenario():
        assert engine.start("audit").ok
        run = next(iter(engine.runs.values()))
        engine.definitions["audit"]["steps"][0]["args"]["nested"]["value"] = "changed"
        await run["task"]

    asyncio.run(scenario())
    assert engine.dispatcher.calls[0][1]["nested"]["value"] == "original"


def test_resumption_refuses_changed_arguments_even_with_identical_tool_names(engine):
    engine.define("audit", [{"tool": "research", "args": {"action": "agenda", "topic": "new"}}])
    record = engine.store.create(kind="workflow", name="audit", state="interrupted",
                                 args={"workflow": "audit"},
                                 steps=[{"tool": "research", "args": {"action": "agenda", "topic": "old"}}])

    async def scenario():
        result = engine.resume_job(record)
        for run in engine.runs.values():
            await run["task"]
        return result

    result = asyncio.run(scenario())
    assert not result.ok and "changed" in result.text
    assert engine.dispatcher.calls == []


def test_workflow_job_store_failure_releases_admission(engine, monkeypatch):
    engine.define("audit", [{"tool": "research"}])
    released = []
    ticket = SimpleNamespace(running=True, release=lambda: released.append(True))
    monkeypatch.setattr(engine._admission, "submit", lambda *args, **kwargs: ticket)
    monkeypatch.setattr(engine.store, "create", fail_save)

    async def scenario():
        try:
            result = engine.start("audit")
            assert not result.ok
        except OSError:
            pass
        assert released == [True], "a failed job record leaked its running slot"

    asyncio.run(scenario())


def test_failed_job_creation_rolls_back_parent_and_steps():
    store = JobStore()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            store.create(kind="workflow", name="broken", steps=[
                {"index": 1, "tool": "research"}, {"index": 1, "tool": "research"}])
        assert not store._db.in_transaction
        store.create(kind="workflow", name="good")
        assert [row["name"] for row in store.list()] == ["good"]
        assert store._db.execute("SELECT COUNT(*) FROM job_steps").fetchone()[0] == 0
    finally:
        store.close()


def test_nested_workflow_templates_expand_once_without_mutating_the_template():
    original = {"items": [{"value": "{input}"}, "{prev}", 3], "both": "{input}: {prev}"}
    expected = deepcopy(original)
    resolved = WorkflowEngine._resolve_args(original, "literal {prev}", "previous")
    assert resolved == {"items": [{"value": "literal {prev}"}, "previous", 3],
                        "both": "literal {prev}: previous"}
    resolved["items"][0]["value"] = "mutated by tool"
    assert original == expected


def test_resumption_checks_the_retry_policy_saved_when_the_run_started(engine):
    engine.define("audit", [{"tool": "research", "args": {"action": "agenda"}, "retries": 0}])

    async def scenario():
        assert engine.start("audit").ok
        run = next(iter(engine.runs.values()))
        original = engine.store.get(run["job_id"])
        steps = engine.store.steps(run["job_id"])
        run["task"].cancel()
        await asyncio.gather(run["task"], return_exceptions=True)
        engine.runs.clear()
        record = engine.store.create(kind="workflow", name="audit", state="interrupted",
                                     args=original["args"], steps=steps)
        engine.definitions["audit"]["steps"][0]["retries"] = 3
        result = engine.resume_job(record)
        for restarted in engine.runs.values():
            await restarted["task"]
        return result

    result = asyncio.run(scenario())
    assert not result.ok and "changed" in result.text
    assert engine.dispatcher.calls == []


def test_separate_sync_connections_preserve_all_local_ticks(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    path = tmp_path / "shared.db"
    first, second = SyncJournal(path), SyncJournal(path)
    ready = threading.Barrier(2)

    def write(journal):
        ready.wait(timeout=5)
        return [journal.record("notes", "notes", "k", str(i)).lamport_ts for i in range(20)]

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(write, first), pool.submit(write, second)
            assert sorted(a.result(timeout=10) + b.result(timeout=10)) == list(range(1, 41))
        assert len(first.all_entries()) == 40
    finally:
        first.close()
        second.close()
