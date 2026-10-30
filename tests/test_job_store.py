"""
The durable job store — one owner for every job record.

What it must guarantee:

  * ids are unique and never derived from a clock;
  * only legal state transitions apply, and finished is final;
  * progress is real or explicitly indeterminate, and every change has a
    store-wide sequence number a reconnecting view can resume from;
  * a restart marks unfinished work INTERRUPTED — it never resumes anything
    by itself;
  * an approval is bound to one job, one step and the exact tool + arguments,
    expires, and can be used once.

Offline: temporary SQLite files only.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core import job_store  # noqa: E402
from orion_core.job_store import (  # noqa: E402
    EXCLUSIVE,
    READ_ONLY,
    SIDE_EFFECT,
    JobStore,
    State,
    args_digest,
    side_effect_class,
)


class _Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def _job(store, **kw):
    kw.setdefault("kind", "plan")
    kw.setdefault("name", "demo")
    return store.create(**kw)


def test_result_collection_reads_all_durable_job_kinds_after_restart(tmp_path):
    from orion_core.bus import OrionBus
    from orion_core.jobs import JobManager

    path = tmp_path / "jobs.db"
    store = JobStore(path)
    records = []
    for kind in ("tool", "workflow", "plan", "specialist"):
        record = store.create(kind=kind, name=kind, label=f"saved {kind}")
        store.transition(record["id"], State.RUNNING)
        store.transition(record["id"], State.COMPLETED, ok=True,
                         result=f"{kind}: retained findings")
        records.append((record["id"], kind))
    store.close()
    reopened = JobStore(path)
    try:
        manager = JobManager(OrionBus(), dispatch=None, store=reopened)
        for job_id, kind in records:
            result = manager.collect(job_id)
            assert result.ok, result.text
            assert f"{kind}: retained findings" in result.text
    finally:
        reopened.close()


def test_result_collection_does_not_report_running_or_failed_record_as_success():
    from orion_core.bus import OrionBus
    from orion_core.jobs import JobManager

    store = JobStore(None)
    try:
        manager = JobManager(OrionBus(), dispatch=None, store=store)
        record = store.create(kind="workflow", name="review", label="review")
        store.transition(record["id"], State.RUNNING)
        assert not manager.collect(record["id"]).ok
        store.transition(record["id"], State.FAILED, ok=False, result="partial notes only")
        result = manager.collect(record["id"])
        assert not result.ok
        assert "partial notes only" in result.text
    finally:
        store.close()


def test_ids_are_unique_and_not_time_based():
    store = JobStore(None)
    ids = {_job(store)["id"] for _ in range(500)}
    assert len(ids) == 500
    assert all(i.startswith("job-") and len(i) == 16 for i in ids)


def test_only_legal_transitions_apply_and_finished_is_final():
    store = JobStore(None)
    job = _job(store)
    assert job["state"] == "queued" and job["indeterminate"] is True
    assert store.transition(job["id"], State.RUNNING)["started_at"] is not None
    assert store.transition(job["id"], State.COMPLETED, ok=True, result="done")["ok"]
    assert store.transition(job["id"], State.RUNNING) is None, "completed is final"
    assert store.get(job["id"])["state"] == "completed"
    fresh = _job(store)
    assert store.transition(fresh["id"], State.WAITING_FOR_APPROVAL) is None, \
        "a queued job cannot be waiting on an approval for a step it never ran"


def test_progress_is_real_or_explicitly_indeterminate_and_sequenced():
    store = JobStore(None)
    job = _job(store)
    seen = []
    store.subscribe(seen.append)
    first = store.update(job["id"], fraction=0.25, note="a quarter")
    assert first["fraction"] == 0.25 and first["indeterminate"] is False
    second = store.update(job["id"], fraction=None, note="counting")
    assert second["indeterminate"] is True
    assert second["seq"] > first["seq"] > job["seq"]
    assert [r["seq"] for r in seen] == [first["seq"], second["seq"]]
    delta = store.snapshot(since_seq=first["seq"])
    assert [r["id"] for r in delta["jobs"]] == [job["id"]]
    assert store.snapshot(since_seq=second["seq"])["jobs"] == []


def test_a_restart_marks_unfinished_work_interrupted_and_resumes_nothing(tmp_path):
    path = tmp_path / "jobs.db"
    store = JobStore(path)
    running = _job(store, state=State.RUNNING,
                   steps=[{"index": 1, "tool": "web_search", "args": {"query": "x"}},
                          {"index": 2, "tool": "open_app", "args": {"app_name": "x"}}])
    store.set_step(running["id"], 1, state="completed")
    store.set_step(running["id"], 2, state="running")
    queued = _job(store)
    paused = _job(store, state=State.RUNNING)
    store.transition(paused["id"], State.PAUSED)
    done = _job(store, state=State.RUNNING)
    store.transition(done["id"], State.COMPLETED, ok=True)
    store.close()                                  # the process "dies" here

    reopened = JobStore(path)
    states = {r["id"]: r["state"] for r in reopened.recovered}
    assert states == {running["id"]: "interrupted", queued["id"]: "interrupted",
                      paused["id"]: "interrupted"}
    assert reopened.get(done["id"])["state"] == "completed"
    assert "stopped while this was running" in reopened.get(running["id"])["note"]
    steps = {s["idx"]: s["state"] for s in reopened.steps(running["id"])}
    assert steps == {1: "completed", 2: "interrupted"}
    assert reopened.interrupted_step(running["id"])["tool"] == "open_app"
    reopened.close()


def test_the_file_store_uses_the_projects_wal_policy(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    mode = store._db.execute("PRAGMA journal_mode").fetchone()[0]
    store.close()
    assert mode.lower() == "wal"


def test_steps_record_their_side_effect_class():
    assert side_effect_class("web_search") == READ_ONLY
    assert side_effect_class("open_app") == SIDE_EFFECT
    assert side_effect_class("forge") == EXCLUSIVE
    assert side_effect_class("second_brain", {"action": "ingest"}) == SIDE_EFFECT
    assert side_effect_class("some_forged_tool") == SIDE_EFFECT, "unknown is never safe"
    store = JobStore(None)
    job = _job(store, steps=[{"tool": "web_search", "args": {"query": "a"}},
                             {"tool": "open_app", "args": {"app_name": "b"}}])
    assert [s["side_effect"] for s in store.steps(job["id"])] == [READ_ONLY, SIDE_EFFECT]


# ── approvals ────────────────────────────────────────────────────────────────

def _waiting(store, clock=None):
    job = _job(store, state=State.RUNNING)
    approval = store.request_approval(job["id"], 2, "open_app", {"app_name": "x"},
                                      reason="may already have run", ttl_s=60)
    return job, approval


def test_an_approval_puts_the_job_in_waiting_and_is_single_use():
    store = JobStore(None)
    job, approval = _waiting(store)
    assert store.get(job["id"])["state"] == "waiting_for_approval"
    assert store.consume_approval(job["id"], 2, "open_app", {"app_name": "x"})[0] is False, \
        "nothing runs before a decision"
    assert store.decide_approval(approval["id"], True)["state"] == "approved"
    assert store.consume_approval(job["id"], 2, "open_app", {"app_name": "x"}) == (True, "")
    ok, why = store.consume_approval(job["id"], 2, "open_app", {"app_name": "x"})
    assert not ok and "no approval" in why, "an approval is used once"


def test_an_approval_does_not_carry_to_changed_arguments_or_another_step():
    store = JobStore(None)
    job, approval = _waiting(store)
    store.decide_approval(approval["id"], True)
    assert store.consume_approval(job["id"], 3, "open_app", {"app_name": "x"})[0] is False
    ok, why = store.consume_approval(job["id"], 2, "open_app", {"app_name": "y"})
    assert not ok and "changed" in why
    assert store.consume_approval(job["id"], 2, "open_app", {"app_name": "x"})[0] is False, \
        "a mismatched use invalidates it"


def test_an_approval_expires():
    clock = _Clock()
    store = JobStore(None, clock=clock)
    job, approval = _waiting(store)
    clock.now += 61
    assert store.decide_approval(approval["id"], True) is None, "too late to decide"
    job2, approval2 = _waiting(store)
    store.decide_approval(approval2["id"], True)
    clock.now += 61
    ok, why = store.consume_approval(job2["id"], 2, "open_app", {"app_name": "x"})
    assert not ok and "expired" in why
    assert store.pending_approvals() == []


def test_the_binding_is_the_exact_call():
    assert args_digest("open_app", {"a": 1, "b": 2}) == args_digest("open_app", {"b": 2, "a": 1})
    assert args_digest("open_app", {"a": 1}) != args_digest("close_app", {"a": 1})


def test_finished_jobs_are_pruned_but_live_ones_never_are(monkeypatch):
    monkeypatch.setattr(job_store, "RETAINED_FINISHED", 5)
    store = JobStore(None)
    live = _job(store)
    for _ in range(12):
        record = _job(store, state=State.RUNNING)
        store.transition(record["id"], State.COMPLETED, ok=True)
    _job(store)                                     # creation prunes
    counts = store.snapshot()["counts"]
    assert counts["completed"] == 5
    assert store.get(live["id"]) is not None


def test_installing_the_durable_store_reports_what_the_last_run_left(tmp_path, monkeypatch):
    path = tmp_path / "jobs.db"
    first = JobStore(path)
    _job(first, state=State.RUNNING)
    first.close()
    monkeypatch.setattr(job_store, "_STORE", None)
    recovered = job_store.install_store(JobStore(path))
    assert [r["state"] for r in recovered] == ["interrupted"]
    assert job_store.default_store().path == path
    job_store.default_store().close()


def test_an_in_memory_store_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = JobStore(None)
    _job(store)
    assert list(tmp_path.iterdir()) == []
    assert isinstance(store._db, sqlite3.Connection)


def _synchronous_at_each_write(store):
    """The synchronous level in force for every UPDATE/INSERT the store runs."""
    levels = []

    def trace(statement):
        if statement.lstrip().upper().startswith(("UPDATE", "INSERT")):
            levels.append((statement.split()[0].upper(),
                           store._db.execute("PRAGMA synchronous").fetchone()[0]))
    store._db.set_trace_callback(trace)
    return levels


def test_state_step_and_approval_writes_are_flushed_but_progress_is_not(tmp_path):
    """Audit F12: WAL + synchronous=NORMAL can roll back the last commits on a
    power cut. A lost 'running' mark or a spent approval coming back matters;
    a lost progress fraction does not."""
    store = JobStore(tmp_path / "jobs.db")
    record = store.create(kind="plan", name="send", steps=[{"tool": "send_email", "args": {}}])
    levels = _synchronous_at_each_write(store)
    FULL, NORMAL = 2, 1

    store.update(record["id"], fraction=0.25, note="drafting")
    assert {level for _, level in levels} == {NORMAL}
    levels.clear()
    store.transition(record["id"], State.RUNNING)
    store.set_step(record["id"], 0, state="running")
    approval = store.request_approval(record["id"], 0, "send_email", {"to": "a"})
    store.decide_approval(approval["id"], True)
    assert store.consume_approval(record["id"], 0, "send_email", {"to": "a"}) == (True, "")
    store.mark_interrupted(record["id"], "stopping")
    assert levels and {level for _, level in levels} == {FULL}
    levels.clear()
    store.set_step(record["id"], 0, attempts=2, detail="retrying")
    assert {level for _, level in levels} == {NORMAL}
    assert store._db.execute("PRAGMA synchronous").fetchone()[0] == NORMAL
    assert store.durable_commits >= 7
    store.close()


def test_a_failed_durable_write_restores_the_relaxed_setting(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    record = _job(store)
    try:
        store.transition(record["id"], State.RUNNING, not_a_field="x")
    except KeyError:
        pass
    assert store._db.execute("PRAGMA synchronous").fetchone()[0] == 1
    assert store.get(record["id"])["state"] == "queued"
    store.close()


# ── admission ownership when the durable record fails (audit 23 Oct, P1) ────
#
# A slot reserved before the record exists has no control to give it back if
# the record cannot be written; a terminal write that fails must still retire
# the live control. Each leak was permanent: repeat the request often enough
# and admission refuses ORION's own valid work.

import asyncio  # noqa: E402

import pytest  # noqa: E402


class _Bus:
    """log / dashboard_event signals that keep what they were sent."""

    def __init__(self):
        self.lines: list[tuple] = []
        self.events: list[tuple] = []

    @property
    def log(self):
        from types import SimpleNamespace
        return SimpleNamespace(emit=lambda *a: self.lines.append(a))

    @property
    def dashboard_event(self):
        from types import SimpleNamespace
        return SimpleNamespace(emit=lambda *a: self.events.append(a))

    def __getattr__(self, name):
        from types import SimpleNamespace
        return SimpleNamespace(emit=lambda *a: None)


def _one_slot(cls_name="USER"):
    """An admission with exactly one running slot of *cls_name* and NO queue,
    so a leaked slot makes the very next request fail outright."""
    from orion_core.admission import Admission, ClassPolicy, WorkClass
    cls = getattr(WorkClass, cls_name)
    return Admission({cls: ClassPolicy(running=1, pending=0)}, shared_running=1), cls


def _failing(message="disk I/O error"):
    def broken(*_a, **_k):
        raise sqlite3.OperationalError(message)
    return broken


def test_a_record_that_cannot_be_written_gives_back_its_reserved_slot(monkeypatch):
    from orion_core import jobs
    monkeypatch.setattr(jobs, "CONTROLS", {})
    admission, cls = _one_slot()
    store = JobStore(None)
    try:
        ticket = admission.submit(cls, label="first")
        assert ticket.running and admission.in_use() == 1
        store.create = _failing()
        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            jobs.JobControl.create(kind="tool", name="probe", cls=cls, store=store,
                                   admission=admission, ticket=ticket)
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
        del store.create                                 # the disk recovers
        later = admission.submit(cls, label="later")      # pending=0: Busy if leaked
        assert later.running
        later.release()
    finally:
        admission.close()
        store.close()


async def test_a_detached_job_that_cannot_be_recorded_runs_nothing_and_leaks_nothing(monkeypatch):
    from orion_core import jobs
    from orion_core.data import ToolResult
    monkeypatch.setattr(jobs, "CONTROLS", {})
    admission, _cls = _one_slot()
    store = JobStore(None)
    calls = []

    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult("synthetic status")

    try:
        manager = jobs.JobManager(_Bus(), dispatch, store=store, admission=admission)
        store.create = _failing()
        refused = manager.start("resource_status", {})
        assert isinstance(refused, ToolResult) and not refused.ok
        assert "could not be recorded" in refused.text
        assert admission.in_use() == 0 and jobs.CONTROLS == {} and calls == []
        del store.create
        job = manager.start("resource_status", {})
        assert not isinstance(job, ToolResult), "a later valid request is admitted"
        await asyncio.wait_for(job._task, 3)
        assert calls == [("resource_status", {})] and job.ok
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
    finally:
        admission.close()
        store.close()


def test_finish_retires_the_control_and_releases_once_when_the_record_fails(monkeypatch):
    from orion_core import jobs
    monkeypatch.setattr(jobs, "CONTROLS", {})
    admission, cls = _one_slot()
    store = JobStore(None)
    try:
        ticket = admission.submit(cls, label="work")
        released = []
        original = ticket.release
        ticket.release = lambda: (released.append(1), original())[1]
        control = jobs.JobControl.create(kind="tool", name="probe", cls=cls, store=store,
                                         admission=admission, ticket=ticket)
        assert control.id in jobs.CONTROLS
        store.transition = _failing("terminal write refused")
        with pytest.raises(sqlite3.OperationalError, match="terminal write refused"):
            control.finish(State.COMPLETED, ok=True, result="done")
        assert released == [1] and admission.in_use() == 0
        assert jobs.CONTROLS == {}, "a dead control must not stay registered as live work"
        del store.transition
        assert store.get(control.id)["state"] == "running", \
            "the record does not claim a terminal state that was never committed"
    finally:
        admission.close()
        store.close()


async def test_a_job_whose_outcome_cannot_be_recorded_still_ends_in_the_monitor(monkeypatch):
    from orion_core import jobs
    from orion_core.data import ToolResult
    monkeypatch.setattr(jobs, "CONTROLS", {})
    admission, _cls = _one_slot()
    store = JobStore(None)
    bus = _Bus()
    effects = []

    async def dispatch(tool, args):
        effects.append(tool)
        store.transition = _failing("terminal write refused")   # fails from here on
        return ToolResult("synthetic status")

    try:
        manager = jobs.JobManager(bus, dispatch, store=store, admission=admission)
        job = manager.start("resource_status", {})
        released = []
        ticket = job._control.ticket
        original = ticket.release
        ticket.release = lambda: (released.append(1), original())[1]
        await asyncio.wait_for(job._task, 3)
        assert effects == ["resource_status"], "one effect, never retried"
        assert released == [1] and admission.in_use() == 0 and jobs.CONTROLS == {}
        assert job.state.value == "completed"
        last = [payload for name, payload in bus.events if name == "job"][-1]
        assert last["state"] == "completed", "the monitor must not keep a running badge"
        assert any("could not be recorded" in str(line) for line in bus.lines)
    finally:
        admission.close()
        store.close()
