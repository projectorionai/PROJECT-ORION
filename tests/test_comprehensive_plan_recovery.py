"""An uncertain durable step must not be treated as never executed."""
import asyncio
from types import SimpleNamespace

import pytest

from orion_core import jobs
from orion_core.admission import Admission
from orion_core.data import ToolResult
from orion_core.job_store import JobStore, State
from orion_core.plan_executor import PlanExecutor, ToolCatalogue


class Bus:
    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: None)


@pytest.mark.parametrize("damage", ["missing", "tool", "args", "side_effect", "unknown_state", "extra", "valid"])
def test_plan_recovery_requires_matching_durable_step_evidence(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    observed = []
    async def dispatch(tool, args):
        observed.append((tool, args))
        return ToolResult("inert temporary executor")
    store, admission = JobStore(tmp_path / "jobs.db"), Admission()
    try:
        plan = [{"id": "effect", "tool": "custom_effect", "args": {"target": "original"}}]
        record = store.create(kind="plan", name="recovery", state=State.INTERRUPTED,
            args={"steps": plan, "max_retries": 0}, steps=[{
                "index": 1, "step_id": "effect", "tool": "custom_effect", "args": {"target": "original"}}])
        key = record["id"]
        if damage == "missing":
            store._db.execute("DELETE FROM job_steps WHERE job_id=?", (key,))
        elif damage == "extra":
            store._db.execute("INSERT INTO job_steps (job_id,idx,tool,args_json,side_effect) VALUES (?,2,?,?,?)",
                              (key, "foreign_effect", "{}", "write"))
        elif damage in {"tool", "args", "side_effect"}:
            field, value = {"tool": ("tool", "different_effect"), "args": ("args_json", '{"target":"different"}'),
                            "side_effect": ("side_effect", "read_only")}[damage]
            store._db.execute(f"UPDATE job_steps SET {field}=? WHERE job_id=?", (value, key))
        elif damage == "unknown_state":
            store.set_step(key, 1, state="unrecognized")
        store._db.commit()
        before, steps_before = store.get(key), store.steps(key)
        executor = PlanExecutor(Bus(), dispatch, store=store, admission=admission,
                                catalogue=ToolCatalogue([], ["custom_effect"]))
        result = asyncio.run(executor.resume_job(record))
        if damage == "valid":
            assert result.ok and observed == [("custom_effect", {"target": "original"})]
        else:
            assert not result.ok and "step evidence" in result.text.lower()
            assert not observed and key not in jobs.CONTROLS
            assert store.get(key) == before and store.steps(key) == steps_before
            assert store.pending_approvals(key) == []
    finally:
        admission.close()
        store.close()


# ── admission ownership and the restart-approval boundary (audit 23 Oct) ─────
#
# P1: a plan whose record or progress cannot be written must give back its
# admission slot and live control, keep the original error, and never repeat
# a step that may already have happened. P2: a restart approval is used up at
# the admitted step's own dispatch: a resume that is cancelled while queued,
# or whose approval expires while it waits, spends nothing and runs nothing.

import sqlite3  # noqa: E402

from orion_core.admission import ClassPolicy, WorkClass  # noqa: E402


class _Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def _isolate_controls(monkeypatch):
    """One fresh live-job registry for jobs AND plan_executor, which binds the
    name at import: patching only one would split registration from removal."""
    from orion_core import plan_executor
    fresh: dict = {}
    monkeypatch.setattr(jobs, "CONTROLS", fresh)
    monkeypatch.setattr(plan_executor, "CONTROLS", fresh)
    return fresh


def _interactive_admission(pending=0):
    return Admission({WorkClass.INTERACTIVE: ClassPolicy(running=1, pending=pending)})


def _executor(store, admission, calls):
    async def dispatch(tool, args):
        calls.append((tool, dict(args)))
        return ToolResult("synthetic effect observed")
    return PlanExecutor(Bus(), dispatch, store=store, admission=admission,
                        catalogue=ToolCatalogue([], ["custom_effect"]))


_TWO_EFFECTS = [{"id": "first", "tool": "custom_effect", "args": {"target": "one"}},
                {"id": "second", "tool": "custom_effect", "args": {"target": "two"}}]


def test_a_plan_that_cannot_be_recorded_runs_nothing_and_leaks_no_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    _isolate_controls(monkeypatch)
    calls = []
    store, admission = JobStore(tmp_path / "jobs.db"), _interactive_admission()
    try:
        executor = _executor(store, admission, calls)

        def broken(**_kw):
            raise sqlite3.OperationalError("disk I/O error")
        store.create = broken
        refused = asyncio.run(executor.execute(list(_TWO_EFFECTS), "probe"))
        assert not refused.ok and "nothing was run" in refused.text
        assert calls == [] and admission.in_use() == 0 and jobs.CONTROLS == {}
        del store.create
        result = asyncio.run(executor.execute(list(_TWO_EFFECTS), "probe"))
        assert result.ok, "a later valid request is admitted (pending=0: a leak refuses it)"
        assert [args["target"] for _tool, args in calls] == ["one", "two"]
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
    finally:
        admission.close()
        store.close()


@pytest.mark.parametrize("fault,effects", [("step_running", 0), ("step_completed", 1),
                                           ("progress", 1)])
def test_a_persistence_fault_mid_plan_stops_it_once_and_releases_everything(
        tmp_path, monkeypatch, fault, effects):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    _isolate_controls(monkeypatch)
    calls = []
    store, admission = JobStore(tmp_path / "jobs.db"), _interactive_admission()
    try:
        executor = _executor(store, admission, calls)
        real_step, real_update = store.set_step, store.update

        def flaky_step(job_id, idx, **fields):
            if fault == f"step_{fields.get('state')}":
                raise sqlite3.OperationalError("disk I/O error")
            return real_step(job_id, idx, **fields)

        def flaky_update(job_id, **fields):
            if fault == "progress" and "fraction" in fields:
                raise sqlite3.OperationalError("disk I/O error")
            return real_update(job_id, **fields)
        store.set_step, store.update = flaky_step, flaky_update
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            asyncio.run(executor.execute(list(_TWO_EFFECTS), "probe"))
        assert len(calls) == effects, "zero effects before dispatch, one after; never a retry"
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
        [record] = store.list(kinds=["plan"], limit=5)
        assert record["state"] == "failed" and "may already have happened" in record["result"]
        later = admission.submit(WorkClass.INTERACTIVE, label="next turn")
        assert later.running
        later.release()
    finally:
        admission.close()
        store.close()


def test_a_plan_whose_outcome_cannot_be_recorded_still_reports_what_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    _isolate_controls(monkeypatch)
    calls = []
    store, admission = JobStore(tmp_path / "jobs.db"), _interactive_admission()
    try:
        executor = _executor(store, admission, calls)
        real = store.transition

        def refuse_terminal(job_id, state, **fields):
            if State(state) in {State.COMPLETED, State.FAILED, State.CANCELLED}:
                raise sqlite3.OperationalError("terminal write refused")
            return real(job_id, state, **fields)
        store.transition = refuse_terminal
        result = asyncio.run(executor.execute(list(_TWO_EFFECTS), "probe"))
        assert len(calls) == 2
        assert "2/2 succeeded" in result.text and "could not be saved" in result.text
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
    finally:
        admission.close()
        store.close()


def test_a_resume_that_cannot_be_recorded_leaves_no_live_control(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    controls = _isolate_controls(monkeypatch)
    calls = []
    store, admission = JobStore(tmp_path / "jobs.db"), _interactive_admission()
    try:
        executor = _executor(store, admission, calls)
        plan = [{"id": "effect", "tool": "custom_effect", "args": {"target": "x"}}]
        record = store.create(kind="plan", name="recovery", state=State.INTERRUPTED,
                              args={"steps": plan}, steps=[{
                                  "index": 1, "step_id": "effect", "tool": "custom_effect",
                                  "args": {"target": "x"}}])

        def refuse(*_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")
        store.transition = refuse
        result = asyncio.run(executor.resume_job(record))
        assert not result.ok and "nothing was run" in result.text
        assert calls == [] and controls == {} and admission.in_use() == 0
    finally:
        admission.close()
        store.close()


def _approved_interrupted_plan(store):
    plan = [{"id": "effect", "tool": "custom_effect", "args": {"target": "original"}}]
    record = store.create(kind="plan", name="recovery", state=State.INTERRUPTED,
                          args={"steps": plan, "max_retries": 0}, steps=[{
                              "index": 1, "step_id": "effect", "tool": "custom_effect",
                              "args": {"target": "original"}}])
    store.set_step(record["id"], 1, state="interrupted")
    approval = store.request_approval(record["id"], 1, "custom_effect", {"target": "original"},
                                      reason="it may already have run", ttl_s=60)
    assert store.decide_approval(approval["id"], True)["state"] == "approved"
    return store.get(record["id"]), approval


@pytest.mark.parametrize("outcome", ["cancelled_while_queued", "expired_while_queued",
                                     "admitted"])
def test_a_restart_approval_is_spent_only_at_the_admitted_dispatch(tmp_path, monkeypatch,
                                                                   outcome):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    _isolate_controls(monkeypatch)
    calls, clock = [], _Clock()
    store = JobStore(tmp_path / "jobs.db", clock=clock)
    admission = _interactive_admission(pending=4)
    try:
        executor = _executor(store, admission, calls)
        record, approval = _approved_interrupted_plan(store)
        job_id = record["id"]

        async def scenario():
            blocker = admission.submit(WorkClass.INTERACTIVE, label="the turn")
            task = asyncio.create_task(executor.resume_job(record))
            control = None
            for _ in range(200):
                await asyncio.sleep(0.005)
                control = jobs.CONTROLS.get(job_id)
                if control is not None and control.ticket is not None and control.ticket.queued:
                    break
            assert control is not None and control.ticket.queued, "the resume must be waiting"
            assert calls == [] and store.approval(approval["id"])["state"] == "approved"
            if outcome == "cancelled_while_queued":
                control.cancel()
                result = await asyncio.wait_for(task, 3)
                blocker.release()
                return result
            if outcome == "expired_while_queued":
                clock.now += 61
            blocker.release()
            return await asyncio.wait_for(task, 3)

        result = asyncio.run(scenario())
        state = store.approval(approval["id"])["state"]
        if outcome == "cancelled_while_queued":
            assert not result.ok and calls == []
            assert state == "approved", "a cancelled resume must not spend the approval"
            assert store.get(job_id)["state"] == "cancelled"
        elif outcome == "expired_while_queued":
            assert not result.ok and calls == []
            assert state == "expired" and "expired" in result.text
        else:
            assert result.ok and calls == [("custom_effect", {"target": "original"})]
            assert state == "consumed"
            replay = store.consume_approval(job_id, 1, "custom_effect", {"target": "original"})
            assert replay[0] is False, "the approval is single-use"
            again = asyncio.run(executor.resume_job(store.get(job_id)))
            assert not again.ok and len(calls) == 1, "a finished plan is never replayed"
        assert admission.in_use() == 0 and jobs.CONTROLS == {}
    finally:
        admission.close()
        store.close()
