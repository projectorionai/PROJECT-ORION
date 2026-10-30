"""Incomplete recovery metadata must not replay completed effects."""
import asyncio
from types import SimpleNamespace

import pytest

from orion_core import jobs
from orion_core.admission import Admission
from orion_core.data import ToolResult
from orion_core.job_store import JobStore, State
from orion_core.workflow_engine import WorkflowEngine


class Bus:
    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: None)


@pytest.mark.parametrize("damage", ["unicode_overflow", "missing", "stale", "zero",
                                    "negative", "skip", "boolean", "text", "list",
                                    "completed_gap", "failed_effect", "valid"])
def test_recovery_never_replays_or_skips_a_recorded_step(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    calls = []

    async def dispatch(tool, args):
        calls.append(args["path"])
        return ToolResult("fake file effect observed")

    store = JobStore(tmp_path / "jobs.db")
    admission = Admission()
    try:
        engine = WorkflowEngine(Bus(), SimpleNamespace(dispatch=dispatch,
            handler_table=lambda: {"write_file": None}), path=tmp_path / "workflows.json",
            store=store, admission=admission)
        steps = [{"tool": "write_file", "args": {"path": path}, "retries": 0}
                 for path in ("already-written", "not-yet-written")]
        assert engine.define("saved", steps).ok
        record = store.create(kind="workflow", name="saved", state=State.INTERRUPTED,
            args={"workflow": "saved", "definition_digest": engine._definition_digest(engine.definitions["saved"])},
            steps=[dict(step, index=i) for i, step in enumerate(steps, 1)])
        store.set_step(record["id"], 1, state="completed")
        checkpoint = {"next": 2, "prev": "complete prior output"}
        if damage == "unicode_overflow":
            checkpoint["prev"] = "\U0001f600" * 2000
        elif damage == "missing":
            checkpoint = {}
        elif damage == "list":
            checkpoint = ["broken checkpoint"]
        elif damage == "completed_gap":
            store.set_step(record["id"], 1, state="pending")
            store.set_step(record["id"], 2, state="completed")
            checkpoint["next"] = 1
        elif damage == "failed_effect":
            store.set_step(record["id"], 1, state="failed")
            checkpoint["next"] = 1
        elif damage != "valid":
            checkpoint["next"] = {"stale": 1, "zero": 0, "negative": -1,
                                  "skip": 3, "boolean": True, "text": "2"}[damage]
        store.update(record["id"], checkpoint=checkpoint)
        record = store.get(record["id"])
        if damage == "unicode_overflow":
            assert record["checkpoint"] == {"_truncated": True}
        before = store.get(record["id"])
        before_steps = store.steps(record["id"])

        async def scenario():
            result = engine.resume_job(record)
            await asyncio.gather(*(r["task"] for r in engine.runs.values()))
            return result

        result = asyncio.run(scenario())
        if damage == "valid":
            assert result.ok and calls == ["not-yet-written"]
            assert store.get(record["id"])["state"] == "completed"
        else:
            assert not result.ok and "checkpoint" in result.text.lower()
            assert calls == [] and not engine.runs
            assert store.get(record["id"]) == before
            assert store.steps(record["id"]) == before_steps
    finally:
        admission.close()
        store.close()


@pytest.fixture
def continuity_engine(tmp_path, monkeypatch):
    from orion_core.admission import ClassPolicy, WorkClass
    from orion_core.cognition import CognitiveStateManager
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    monkeypatch.setattr(jobs, "CONTROLS", {})
    store = JobStore(tmp_path / "continuity_jobs.db")
    admission = Admission({WorkClass.USER: ClassPolicy(running=2, pending=2)}, shared_running=2)
    cognition = CognitiveStateManager(Bus(), SimpleNamespace(), path=tmp_path / "cognition.json")
    engine = WorkflowEngine(Bus(), SimpleNamespace(handler_table=lambda: {"web_search": None}),
                            cognition=cognition, path=tmp_path / "workflows.json",
                            store=store, admission=admission)
    assert engine.define("shared", [{"tool": "web_search", "args": {"query": "{input}"},
                                     "retries": 0}]).ok
    yield engine, cognition, store
    admission.close()
    store.close()


@pytest.mark.parametrize("remaining_result", ["running", "failed"])
def test_completing_one_run_keeps_its_unfinished_sibling_in_memory(continuity_engine,
                                                                 remaining_result):
    engine, cognition, _store = continuity_engine

    async def scenario():
        first_release, second_release = asyncio.Event(), asyncio.Event()
        first_entered, second_entered = asyncio.Event(), asyncio.Event()

        async def dispatch(_tool, args):
            first = args["query"] == "first"
            (first_entered if first else second_entered).set()
            await asyncio.wait_for((first_release if first else second_release).wait(), 3)
            return ToolResult("done", ok=first or remaining_result != "failed")

        engine.dispatcher.dispatch = dispatch
        assert engine.start("shared", "first").ok
        assert engine.start("shared", "second").ok
        first, second = list(engine.runs.values())
        try:
            await asyncio.wait_for(asyncio.gather(first_entered.wait(), second_entered.wait()), 3)
            if remaining_result == "failed":
                second_release.set()
                await asyncio.wait_for(second["task"], 3)
            first_release.set()
            await asyncio.wait_for(first["task"], 3)
            remembered = cognition.snapshot()["open_workflows"].get("shared")
            assert remembered is not None, "The other run's work is still unfinished"
            assert remembered["run"] == second["id"]
            assert remembered["run_status"] == second["status"]
        finally:
            first_release.set()
            second_release.set()
            await asyncio.wait_for(asyncio.gather(first["task"], second["task"]), 3)
        if remaining_result == "running":
            assert "shared" not in cognition.snapshot()["open_workflows"]

    asyncio.run(scenario())


def test_cancel_before_execution_updates_persisted_workflow_memory(continuity_engine):
    engine, cognition, store = continuity_engine

    async def scenario():
        engine.dispatcher.dispatch = lambda *_args: pytest.fail("Cancelled work must not dispatch")
        assert engine.start("shared").ok
        [run] = engine.runs.values()
        assert engine.cancel(run["id"]).ok
        await asyncio.wait_for(asyncio.gather(run["task"], return_exceptions=True), 3)
        remembered = cognition.snapshot()["open_workflows"]["shared"]
        assert remembered["run_status"] == "cancelled"
        assert store.get(run["job_id"])["state"] == "cancelled"

    asyncio.run(scenario())


def test_restart_resume_restores_workflow_memory_before_execution(continuity_engine):
    engine, cognition, store = continuity_engine
    definition = engine.definitions["shared"]
    record = store.create(kind="workflow", name="shared", state=State.INTERRUPTED,
        args={"workflow": "shared", "definition_digest": engine._definition_digest(definition)},
        steps=[dict(definition["steps"][0], index=1)])

    async def scenario():
        release = asyncio.Event()

        async def dispatch(*_args):
            await asyncio.wait_for(release.wait(), 3)
            return ToolResult("resumed")

        engine.dispatcher.dispatch = dispatch
        assert engine.resume_job(record).ok
        [run] = engine.runs.values()
        try:
            remembered = cognition.snapshot()["open_workflows"].get("shared")
            assert remembered is not None, "Restart recovery must restore the unfinished plan"
            assert remembered["run"] == run["id"]
            assert remembered["job_id"] == record["id"]
        finally:
            release.set()
            await asyncio.wait_for(run["task"], 3)

    asyncio.run(scenario())


def test_workflow_status_recalls_interrupted_jobs_after_restart(continuity_engine):
    engine, _cognition, store = continuity_engine
    record = store.create(kind="workflow", name="shared", state=State.INTERRUPTED,
                          args={"workflow": "shared"},
                          steps=[{"index": 1, "tool": "web_search", "args": {}}])
    before = store.get(record["id"])
    result = engine.status()
    assert result.ok
    assert "interrupted" in result.text.lower() and "shared" in result.text
    assert record["id"] in result.text and "resume" in result.text.lower()
    assert not engine.runs and store.get(record["id"]) == before


@pytest.mark.parametrize("placeholder", ["input", "prev"])
@pytest.mark.parametrize("action", ["status", "add_task"])
@pytest.mark.parametrize("recorded_effect", ["read_only", "side_effect"])
def test_restart_reclassifies_expanded_arguments_before_replaying_a_step(
        continuity_engine, placeholder, action, recorded_effect):
    from orion_core.job_store import args_digest
    engine, _cognition, store = continuity_engine
    engine.dispatcher.handler_table = lambda: {"web_search": None, "awareness": None}
    steps = [{"tool": "web_search", "args": {}},
             {"tool": "awareness", "args": {"action": "{" + placeholder + "}", "title": "Exact task"},
              "retries": 0}]
    assert engine.define("expanded", steps).ok
    definition = engine.definitions["expanded"]
    record = store.create(kind="workflow", name="expanded", state=State.INTERRUPTED,
        args={"workflow": "expanded", "input": action,
              "definition_digest": engine._definition_digest(definition)},
        steps=[dict(step, index=i, side_effect="read_only" if i == 1 else recorded_effect)
               for i, step in enumerate(steps, 1)])
    store.set_step(record["id"], 1, state="completed")
    store.set_step(record["id"], 2, state="interrupted")
    store.update(record["id"], checkpoint={"next": 2, "prev": action, "prev_complete": True})
    calls = []

    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult("Observed synthetic dispatch")

    async def scenario():
        engine.dispatcher.dispatch = dispatch
        result = engine.resume_job(store.get(record["id"]))
        await asyncio.wait_for(asyncio.gather(*(run["task"] for run in engine.runs.values())), 3)
        return result

    result = asyncio.run(scenario())
    resolved = {"action": action, "title": "Exact task"}
    if action == "add_task" or recorded_effect != "read_only":
        assert not result.ok and "approval" in result.text.lower()
        assert calls == [] and not engine.runs
        [approval] = store.pending_approvals(record["id"])
        assert approval["digest"] == args_digest("awareness", resolved)
    else:
        assert result.ok and calls == [("awareness", resolved)]
        assert store.get(record["id"])["state"] == "completed"


def test_recovery_reconciles_every_completed_step_without_replaying_it(continuity_engine):
    engine, _cognition, store = continuity_engine
    definition = engine.definitions["shared"]
    record = store.create(kind="workflow", name="shared", state=State.INTERRUPTED,
        args={"workflow": "shared", "definition_digest": engine._definition_digest(definition)},
        steps=[dict(definition["steps"][0], index=1)])
    store.set_step(record["id"], 1, state="completed")
    store.update(record["id"], checkpoint={"next": 2, "prev": "done", "prev_complete": True})
    before_steps = store.steps(record["id"])
    engine.dispatcher.dispatch = lambda *_args: pytest.fail("Completed steps must never replay")
    result = engine.resume_job(store.get(record["id"]))
    assert result.ok and "already finished" in result.text
    assert store.get(record["id"])["state"] == "completed"
    assert store.get(record["id"])["fraction"] == 1.0
    assert store.steps(record["id"]) == before_steps and not engine.runs


def test_workflow_can_approve_then_resume_by_name_without_another_restart(continuity_engine):
    from orion_core.workflow_engine import AutomationManager
    engine, _cognition, store = continuity_engine
    engine.dispatcher.handler_table = lambda: {"open_app": None}
    assert engine.define("named", [{"tool": "open_app", "args": {"app_name": "synthetic"}}]).ok
    definition = engine.definitions["named"]
    record = store.create(kind="workflow", name="named", state=State.INTERRUPTED,
        args={"workflow": "named", "definition_digest": engine._definition_digest(definition)},
        steps=[dict(definition["steps"][0], index=1)])
    store.set_step(record["id"], 1, state="interrupted")
    calls = []

    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult("Synthetic approved effect")

    async def scenario():
        engine.dispatcher.dispatch = dispatch
        manager = AutomationManager(engine)
        assert not (await manager.handle({"action": "resume", "name": "named"})).ok
        [approval] = store.pending_approvals(record["id"])
        assert record["id"] in engine.status().text
        assert "waiting_for_approval" in engine.status().text
        assert (await manager.handle({"action": "approve", "approval": approval["id"]})).ok
        assert (await manager.handle({"action": "resume", "name": "named"})).ok
        await asyncio.wait_for(asyncio.gather(*(run["task"] for run in engine.runs.values())), 3)
        assert calls == [("open_app", {"app_name": "synthetic"})]
        assert store.get(record["id"])["state"] == "completed"

    asyncio.run(asyncio.wait_for(scenario(), 5))


# ── what a run found, scheduled runs and the read-only built-ins ─────────────
#
# A finished run kept only the word "complete", so "what did my workflow
# find?" had no answer; nothing could start a workflow at a time of day; and
# the library held four creator workflows. These pin the result kept on the
# run and its job record, the scheduler's claim-before-run (no double runs,
# even across a restart), its quiet-moment gate and the new built-ins.

import json  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

from orion_core.workflow_engine import (  # noqa: E402
    AutomationManager, ScheduleError, WorkflowScheduler, parse_when)

SUNDAY_8AM = datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc)
MONDAY_7AM = datetime(2026, 10, 19, 7, 0, tzinfo=timezone.utc)


class _Recorder:
    """A bus that keeps every emit as (signal, *args)."""

    def __init__(self):
        self.lines = []

    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: self.lines.append((name, *args)),
                               connect=lambda *_args: None)


class _Gate:
    """The engine's own admission, with the deferral reason the test sets."""

    def __init__(self, real):
        self.real, self.reason = real, ""

    def defer_reason(self, _cls):
        return self.reason

    def __getattr__(self, name):
        return getattr(self.real, name)


def _gate(engine):
    gate = _Gate(engine._admission)
    engine._admission = gate
    return gate


def _answer(text="Mostly sunny"):
    async def dispatch(_tool, _args):
        return ToolResult(text)
    return dispatch


def _scheduler(engine, clock, quiet=lambda: True):
    return WorkflowScheduler(engine, clock=lambda: clock["now"], quiet=quiet)


async def _tick(scheduler, engine):
    started = await scheduler.tick()
    await asyncio.gather(*(run["task"] for run in list(engine.runs.values())
                           if not run["task"].done()))
    return started


def test_a_finished_run_keeps_what_it_found(continuity_engine):
    engine, _cognition, store = continuity_engine
    assert engine.define("check", [{"tool": "web_search", "args": {"query": "weather"}},
                                   {"tool": "web_search", "args": {"query": "calendar"}}]).ok
    assert engine.define("chain", [{"tool": "web_search", "args": {"query": "{input}"}},
                                   {"tool": "web_search", "args": {"query": "summary of {prev}"}}]).ok

    async def dispatch(_tool, args):
        return ToolResult(f"found {args['query']}")

    async def scenario():
        engine.dispatcher.dispatch = dispatch
        assert engine.start("check").ok
        assert engine.start("chain", "jobs").ok
        await asyncio.gather(*(run["task"] for run in engine.runs.values()))

    asyncio.run(scenario())
    check, chain = list(engine.runs.values())
    assert check["result"] == "web_search: found weather\n\nweb_search: found calendar"
    assert chain["result"] == "found summary of found jobs", "a chain reports its final step"
    assert store.get(check["job_id"])["result"] == check["result"]
    assert "found: found summary of found jobs" in engine.status().text
    answer = engine.result("check").text
    assert "found weather" in answer and "found calendar" in answer


def test_what_a_run_found_survives_a_restart(continuity_engine, tmp_path):
    engine, _cognition, store = continuity_engine

    async def scenario():
        engine.dispatcher.dispatch = _answer("Mostly sunny, 14C")
        assert engine.start("shared", "weather").ok
        await asyncio.gather(*(run["task"] for run in engine.runs.values()))

    asyncio.run(scenario())
    restarted = WorkflowEngine(Bus(), SimpleNamespace(handler_table=lambda: {"web_search": None}),
                               path=tmp_path / "workflows.json", store=store,
                               admission=engine._admission)
    answer = restarted.result("shared").text
    assert "Mostly sunny, 14C" in answer and "before ORION last restarted" in answer
    assert "last one before ORION restarted was 'shared'" in restarted.status().text


def test_a_long_result_is_shortened_fairly_and_marked():
    steps = [{"tool": tool, "args": {}} for tool in ("a", "b", "c")]
    outputs = [(1, steps[0], "x" * 3000, True), (2, steps[1], "short", True),
               (3, steps[2], "y" * 3000, True)]
    text, shortened = WorkflowEngine._report({"steps": steps}, outputs)
    assert shortened and len(text) <= 4000
    assert "b: short" in text, "a short part is kept whole"
    assert text.count(" [...]") == 2, "every cut is marked"
    record = WorkflowEngine._job_result("failed at step 3 (c)", text)
    assert len(record) <= 4000 and record.startswith("failed at step 3 (c): ")
    assert record.endswith(" [...]")


def test_an_optional_step_may_fail_without_stopping_the_run(continuity_engine, tmp_path):
    engine, _cognition, store = continuity_engine
    steps = [{"tool": "web_search", "args": {"query": "offline"}, "retries": 0, "optional": True},
             {"tool": "web_search", "args": {"query": "online"}, "retries": 0, "optional": "false"}]
    assert engine.define("gather", steps).ok
    assert engine.definitions["gather"]["steps"][0]["optional"] is True
    assert "optional" not in engine.definitions["gather"]["steps"][1], "bool('false') is True"

    async def dispatch(_tool, args):
        if args["query"] == "offline":
            return ToolResult("offline source unavailable", ok=False)
        return ToolResult("online result")

    async def scenario():
        engine.dispatcher.dispatch = dispatch
        assert engine.start("gather").ok
        [run] = engine.runs.values()
        await run["task"]
        return run

    run = asyncio.run(scenario())
    assert run["status"] == "complete" and store.get(run["job_id"])["state"] == "completed"
    assert "web_search (failed): offline source unavailable" in run["result"]
    assert "web_search: online result" in run["result"]
    stored = tmp_path / "flagged.json"
    stored.write_text(json.dumps({"x": {"steps": [{"tool": "web_search", "args": {},
                                                    "optional": "yes"}]}}), encoding="utf-8")
    assert WorkflowEngine(Bus(), path=stored)._load_error, "a stored flag must be a real boolean"


@pytest.mark.parametrize("words, expected", [
    ("every weekday at 7:30", ("weekdays", "07:30", None, "")),
    ("daily at 18:00", ("daily", "18:00", None, "")),
    ("every monday at 9", ("weekly", "09:00", 0, "")),
    ("thursdays at 6pm", ("weekly", "18:00", 3, "")),
    ("tomorrow at 9am", ("once", "09:00", None, "2026-10-19")),
    ("at 7", ("once", "07:00", None, "2026-10-19")),
    ("today at 17:45", ("once", "17:45", None, "2026-10-18")),
    ("friday at noon", ("once", "12:00", None, "2026-10-23")),
    ("in 90 minutes", ("once", "09:30", None, "2026-10-18")),
    ("2026-10-20 09:00", ("once", "09:00", None, "2026-10-20")),
])
def test_a_schedule_is_read_from_ordinary_words(words, expected):
    spec = parse_when(words, SUNDAY_8AM)
    assert (spec["repeat"], spec["time"], spec["weekday"], spec["date"]) == expected


@pytest.mark.parametrize("words", ["whenever you like", "every monday and thursday at 9",
                                   "today at 7am", "weekly at 9"])
def test_a_schedule_that_cannot_run_is_refused_with_a_reason(words):
    with pytest.raises(ScheduleError):
        parse_when(words, SUNDAY_8AM)


def test_a_due_schedule_runs_once_and_never_twice_across_a_restart(continuity_engine):
    engine, _cognition, _store = continuity_engine
    _gate(engine)
    calls = []

    async def dispatch(_tool, args):
        calls.append(args["query"])
        return ToolResult("Mostly sunny")

    engine.dispatcher.dispatch = dispatch
    clock = {"now": MONDAY_7AM}
    scheduler = _scheduler(engine, clock)
    added = scheduler.add("shared", "daily at 7:30", "weather")
    assert added.ok and "next run Mon 19 Oct 07:30" in added.text
    [entry_id] = [entry["id"] for entry in scheduler._entries]
    clock["now"] = MONDAY_7AM + timedelta(minutes=20)
    assert asyncio.run(_tick(scheduler, engine)) == []
    clock["now"] = MONDAY_7AM + timedelta(minutes=31)
    assert asyncio.run(_tick(scheduler, engine)) == [entry_id]
    [run] = engine.runs.values()
    assert run["status"] == "complete" and run["scheduled"] == entry_id and calls == ["weather"]
    assert asyncio.run(_tick(scheduler, engine)) == [], "a claimed occurrence never runs twice"
    restarted = _scheduler(engine, clock)           # a new process reading the same file
    assert asyncio.run(_tick(restarted, engine)) == []
    clock["now"] += timedelta(days=1)
    assert asyncio.run(_tick(restarted, engine)) == [entry_id]
    assert calls == ["weather", "weather"]


def test_a_due_run_waits_for_admission_and_a_quiet_moment(continuity_engine):
    engine, _cognition, _store = continuity_engine
    engine.bus = _Recorder()
    gate = _gate(engine)
    engine.dispatcher.dispatch = _answer()
    quiet = {"now": True}
    clock = {"now": MONDAY_7AM}
    scheduler = _scheduler(engine, clock, quiet=lambda: quiet["now"])
    assert scheduler.add("shared", "daily at 7:30").ok
    clock["now"] = MONDAY_7AM + timedelta(minutes=31)
    gate.reason = "a request is being answered"
    assert asyncio.run(_tick(scheduler, engine)) == []
    assert asyncio.run(_tick(scheduler, engine)) == []
    waits = [line for line in engine.bus.lines if "is due and waiting" in str(line)]
    assert len(waits) == 1, "the wait is logged once, not on every check"
    gate.reason, quiet["now"] = "", False
    assert asyncio.run(_tick(scheduler, engine)) == [], "never while ORION is speaking"
    quiet["now"] = True
    assert len(asyncio.run(_tick(scheduler, engine))) == 1
    assert not any(name == "speak_request" for name, *_rest in engine.bus.lines)


def test_a_busy_processor_only_delays_a_run_for_a_while(continuity_engine):
    engine, _cognition, _store = continuity_engine
    gate = _gate(engine)
    engine.dispatcher.dispatch = _answer()
    clock = {"now": MONDAY_7AM}
    scheduler = _scheduler(engine, clock)
    assert scheduler.add("shared", "daily at 7:30").ok
    gate.reason = "the processor is busy"
    clock["now"] = MONDAY_7AM + timedelta(minutes=35)
    assert asyncio.run(_tick(scheduler, engine)) == []
    clock["now"] = MONDAY_7AM + timedelta(minutes=41)      # past CPU_PATIENCE_S
    assert len(asyncio.run(_tick(scheduler, engine))) == 1
    gate.reason = "memory is under pressure"
    clock["now"] += timedelta(days=1)
    assert asyncio.run(_tick(scheduler, engine)) == [], "memory pressure is never overridden"


def test_a_run_missed_while_orion_was_off_is_caught_up_once_or_skipped(continuity_engine):
    engine, _cognition, _store = continuity_engine
    _gate(engine)
    engine.dispatcher.dispatch = _answer()
    clock = {"now": MONDAY_7AM - timedelta(hours=1)}
    scheduler = _scheduler(engine, clock)
    assert scheduler.add("shared", "daily at 7:30").ok
    clock["now"] = MONDAY_7AM + timedelta(days=3, hours=2)       # Thu 09:00, three missed
    assert len(asyncio.run(_tick(scheduler, engine))) == 1, "the latest only, never a backlog"
    assert len(engine.runs) == 1
    clock["now"] = MONDAY_7AM + timedelta(days=5, hours=13)      # Sat 20:00, 12.5 h late
    assert asyncio.run(_tick(scheduler, engine)) == []
    [entry] = scheduler._entries
    assert "missed the Sat 24 Oct 07:30 run" in entry["last_outcome"]
    clock["now"] = MONDAY_7AM + timedelta(days=6, minutes=31)    # Sun 07:31
    assert len(asyncio.run(_tick(scheduler, engine))) == 1


def test_scheduling_through_the_tool(continuity_engine):
    engine, _cognition, _store = continuity_engine
    _gate(engine)
    engine.dispatcher.dispatch = _answer()
    clock = {"now": MONDAY_7AM}
    manager = AutomationManager(engine, scheduler=_scheduler(engine, clock))

    async def scenario():
        once = await manager.handle({"action": "schedule", "name": "shared",
                                     "when": "today at 7:45", "input": "news"})
        assert once.ok and "once, on Mon 19 Oct at 07:45" in once.text
        assert (await manager.handle({"action": "schedule", "name": "shared",
                                      "when": "every friday at 18:00"})).ok
        again = await manager.handle({"action": "schedule", "name": "shared",
                                      "when": "every friday at 18:00"})
        assert "already scheduled" in again.text
        refused = await manager.handle({"action": "schedule", "name": "nope", "when": "daily at 9"})
        assert not refused.ok
        listing = (await manager.handle({"action": "schedules"})).text
        assert "every Friday at 18:00" in listing and "once, on Mon 19 Oct at 07:45" in listing
        clock["now"] = MONDAY_7AM + timedelta(minutes=46)
        assert len(await manager.scheduler.tick()) == 1
        await asyncio.gather(*(r["task"] for r in engine.runs.values() if not r["task"].done()))
        assert "[finished]" in (await manager.handle({"action": "schedules"})).text
        assert "Mostly sunny" in (await manager.handle({"action": "result", "name": "shared"})).text
        assert "found: Mostly sunny" in (await manager.handle({"action": "status"})).text
        weekly = next(e["id"] for e in manager.scheduler._entries if e["repeat"] == "weekly")
        assert (await manager.handle({"action": "unschedule", "schedule": weekly})).ok
        assert weekly not in (await manager.handle({"action": "schedules"})).text

    asyncio.run(scenario())


def test_an_unreadable_schedules_file_is_kept_and_nothing_runs(continuity_engine, tmp_path):
    engine, _cognition, _store = continuity_engine
    path = tmp_path / "workflow_schedules.json"
    path.write_text("{not json", encoding="utf-8")
    scheduler = WorkflowScheduler(engine, clock=lambda: MONDAY_7AM, quiet=lambda: True)
    assert scheduler.path == path
    assert not scheduler.add("shared", "daily at 7:30").ok
    assert asyncio.run(scheduler.tick()) == []
    assert path.read_text(encoding="utf-8") == "{not json"


def test_the_schedule_loop_starts_due_runs_and_stops(continuity_engine, monkeypatch):
    engine, _cognition, _store = continuity_engine
    _gate(engine)
    engine.dispatcher.dispatch = _answer()
    monkeypatch.setattr(WorkflowScheduler, "FIRST_TICK_S", 0.0)
    monkeypatch.setattr(WorkflowScheduler, "TICK_S", 0.01)
    clock = {"now": MONDAY_7AM}
    scheduler = _scheduler(engine, clock)
    assert scheduler.add("shared", "daily at 7:30").ok
    clock["now"] = MONDAY_7AM + timedelta(minutes=31)

    async def scenario():
        task = scheduler.start()
        assert scheduler.start() is task, "a second start is a no-op"
        for _ in range(300):
            if engine.runs:
                break
            await asyncio.sleep(0.01)
        scheduler.stop()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(r["task"] for r in engine.runs.values()))

    asyncio.run(scenario())
    [run] = engine.runs.values()
    assert run["status"] == "complete"


def test_the_new_builtins_only_read_and_never_move_files(continuity_engine, monkeypatch):
    from orion_core.job_store import READ_ONLY, side_effect_class
    engine, _cognition, _store = continuity_engine
    manager = AutomationManager(engine, scheduler=_scheduler(engine, {"now": MONDAY_7AM}))
    for name in ("morning_check", "research_digest", "downloads_tidy_preview",
                 "weekly_goal_review"):
        for step in engine.definitions[name]["steps"]:
            if step["tool"] == "organise_files":
                assert step["args"] == {"action": "preview", "folder": "Downloads"}
            else:
                assert side_effect_class(step["tool"], step["args"]) == READ_ONLY, (name, step)

    async def dispatch(tool, _args):
        if tool == "geo":
            return ToolResult("weather service unreachable", ok=False)
        return ToolResult(f"{tool} ok")

    async def instant(_seconds):
        return None

    monkeypatch.setattr("orion_core.workflow_engine.asyncio.sleep", instant)

    async def scenario():
        engine.dispatcher.dispatch = dispatch
        assert engine.start("morning_check").ok
        [run] = [r for r in engine.runs.values() if r["workflow"] == "morning_check"]
        await run["task"]
        said_as_input = await manager.handle({"action": "schedule", "name": "morning_check",
                                              "input": "every weekday at 7:30"})
        return run, said_as_input

    run, said_as_input = asyncio.run(scenario())
    assert run["status"] == "complete", "one source offline costs that part, not the check"
    for part in ("awareness situation: awareness ok", "proactive_check: proactive_check ok",
                 "geo weather (failed): weather service unreachable",
                 "goals attention: goals ok"):
        assert part in run["result"]
    assert said_as_input.ok and "on weekdays at 07:30" in said_as_input.text


@pytest.mark.parametrize("first_optional, unicode_output", [(False, False), (True, False),
                                                           (False, True)])
def test_resuming_keeps_independent_findings_and_does_not_replay_optional_failures(
        continuity_engine, first_optional, unicode_output):
    engine, _cognition, store = continuity_engine
    steps = [{"tool": "web_search", "args": {"query": query}, "retries": 0,
              "optional": first_optional and index == 0}
             for index, query in enumerate(("source-a", "source-b", "source-c"))]
    assert engine.define("gather", steps).ok
    calls = []
    first_text = "Source A unavailable" if first_optional else (
        "Source A finding " + "a" * 3400 if unicode_output else "Source A finding")
    second_text = "Source B finding " + ("\U0001f600" * 1000 if unicode_output else "intact")

    async def scenario():
        entered = asyncio.Event()

        async def before_restart(_tool, args):
            calls.append(args["query"])
            if args["query"] == "source-a":
                return ToolResult(first_text, ok=not first_optional)
            if args["query"] == "source-b":
                return ToolResult(second_text)
            entered.set()
            await asyncio.wait_for(asyncio.Event().wait(), 3)
            pytest.fail("The interrupted step should have been cancelled")

        engine.dispatcher.dispatch = before_restart
        assert engine.start("gather").ok
        [run] = engine.runs.values()
        await asyncio.wait_for(entered.wait(), 3)
        await jobs.suspend_all(timeout=1)
        record = store.get(run["job_id"])
        assert record["state"] == "interrupted"
        assert not record["checkpoint"].get("_truncated"), "findings fit the serialized JSON budget"

        async def after_restart(_tool, args):
            calls.append(args["query"])
            return ToolResult("Source C finding")

        restarted = WorkflowEngine(Bus(), SimpleNamespace(dispatch=after_restart,
            handler_table=lambda: {"web_search": None}), path=engine.path,
            store=store, admission=engine._admission)
        assert restarted.resume_job(record).ok
        [resumed] = restarted.runs.values()
        await asyncio.wait_for(resumed["task"], 3)
        assert resumed["status"] == "complete"
        assert resumed["result_truncated"] is unicode_output
        answer = restarted.result("gather").text
        assert "Source A" in answer and "Source B finding" in answer and "Source C finding" in answer
        assert "(failed)" in answer if first_optional else "Source A finding" in answer
        assert "[...]" in answer if unicode_output else first_text in answer
        assert store.get(run["job_id"])["result"] == resumed["result"]

    asyncio.run(asyncio.wait_for(scenario(), 8))
    assert calls == ["source-a", "source-b", "source-c", "source-c"]


def test_reconciling_finished_steps_preserves_their_saved_findings(continuity_engine):
    engine, _cognition, store = continuity_engine
    steps = [{"tool": "web_search", "args": {"query": query}, "retries": 0,
              "optional": True} for query in ("offline", "online")]
    assert engine.define("reconcile", steps).ok
    definition = engine.definitions["reconcile"]
    record = store.create(kind="workflow", name="reconcile", state=State.INTERRUPTED,
        args={"workflow": "reconcile", "definition_digest": engine._definition_digest(definition)},
        steps=[dict(step, index=index) for index, step in enumerate(steps, 1)])
    store.set_step(record["id"], 1, state="failed")
    store.set_step(record["id"], 2, state="completed")
    store.update(record["id"], checkpoint={"next": 3, "prev": "Actual finding", "prev_complete": True,
        "outputs": [{"index": 1, "text": "Offline source", "ok": False},
                    {"index": 2, "text": "Actual finding", "ok": True}]})
    result = engine.resume_job(store.get(record["id"]))
    assert result.ok and not engine.runs
    saved = store.get(record["id"])
    assert saved["state"] == "completed"
    assert "Offline source" in saved["result"] and "Actual finding" in saved["result"]
    assert "Actual finding" in engine.result("reconcile").text


@pytest.mark.parametrize("saved", [[{"index": True, "text": "invented", "ok": True}],
                                   [{"index": 1, "text": "invented", "ok": False}],
                                   [{"index": 2, "text": "not run yet", "ok": True}],
                                   "broken"])
def test_invalid_saved_findings_do_not_resume_or_change_a_job(continuity_engine, saved):
    engine, _cognition, store = continuity_engine
    steps = [{"tool": "web_search", "args": {"query": query}, "retries": 0}
             for query in ("done", "pending")]
    assert engine.define("invalid", steps).ok
    definition = engine.definitions["invalid"]
    record = store.create(kind="workflow", name="invalid", state=State.INTERRUPTED,
        args={"workflow": "invalid", "definition_digest": engine._definition_digest(definition)},
        steps=[dict(step, index=index) for index, step in enumerate(steps, 1)])
    store.set_step(record["id"], 1, state="completed")
    store.update(record["id"], checkpoint={"next": 2, "prev": "real finding", "outputs": saved})
    before = store.get(record["id"])
    result = engine.resume_job(before)
    assert not result.ok and "saved findings" in result.text
    assert store.get(record["id"]) == before and not engine.runs


def test_named_results_search_all_retained_jobs_and_accept_exact_job_ids(continuity_engine):
    engine, _cognition, store = continuity_engine
    oldest = store.create(kind="workflow", name="weekly_goal_review", state=State.COMPLETED)
    store.update(oldest["id"], result="The retained weekly finding")
    for index in range(51):
        later = store.create(kind="workflow", name="morning_check", state=State.COMPLETED)
        store.update(later["id"], result=f"Later morning finding {index}")
    unrelated = store.create(kind="plan", name="private-plan", state=State.COMPLETED)
    store.update(unrelated["id"], result="Must not be returned as workflow output")
    assert "The retained weekly finding" in engine.result("weekly goal review").text
    assert "The retained weekly finding" in engine.result(oldest["id"]).text
    assert "Later morning finding 50" in engine.result().text
    assert "Must not be returned" not in engine.result(unrelated["id"]).text


# ── admission ownership and the restart-approval boundary (audit 23 Oct) ─────
#
# The same lifecycle witnesses as the plan executor's, through the workflow
# owner: a run that cannot be recorded leaks no slot; a terminal write that
# fails still ends the run in the monitor; a fault mid-run never repeats an
# effect; and a restart approval is spent only at the admitted step itself.

import sqlite3  # noqa: E402


class _Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class _Events:
    """A bus that keeps log lines and dashboard events."""

    def __init__(self):
        self.lines, self.events = [], []

    def __getattr__(self, name):
        if name == "dashboard_event":
            return SimpleNamespace(emit=lambda *a: self.events.append(a))
        if name == "log":
            return SimpleNamespace(emit=lambda *a: self.lines.append(a))
        return SimpleNamespace(emit=lambda *a: None)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    from orion_core.admission import ClassPolicy, WorkClass
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    monkeypatch.setattr(jobs, "CONTROLS", {})
    clock = _Clock()
    store = JobStore(tmp_path / "jobs.db", clock=clock)
    admission = Admission({WorkClass.USER: ClassPolicy(running=1, pending=4)}, shared_running=1)
    calls, bus = [], _Events()

    async def dispatch(tool, args):
        calls.append((tool, dict(args)))
        return ToolResult("synthetic effect observed")

    engine = WorkflowEngine(bus, SimpleNamespace(dispatch=dispatch,
                                                 handler_table=lambda: {"open_app": None}),
                            path=tmp_path / "workflows.json", store=store, admission=admission)
    assert engine.define("pair", [{"tool": "open_app", "args": {"app_name": "one"}, "retries": 0},
                                  {"tool": "open_app", "args": {"app_name": "two"},
                                   "retries": 0}]).ok
    yield SimpleNamespace(engine=engine, store=store, admission=admission, calls=calls,
                          bus=bus, clock=clock, cls=WorkClass.USER)
    admission.close()
    store.close()


def test_a_run_that_cannot_be_recorded_leaks_no_slot_and_a_later_one_starts(lifecycle):
    lc = lifecycle

    def broken(**_kw):
        raise sqlite3.OperationalError("disk I/O error")
    lc.store.create = broken
    refused = lc.engine.start("pair")
    assert not refused.ok and "could not be recorded" in refused.text
    assert lc.admission.in_use() == 0 and jobs.CONTROLS == {} and not lc.engine.runs
    del lc.store.create

    async def scenario():
        assert lc.engine.start("pair").ok
        await asyncio.wait_for(asyncio.gather(*(r["task"] for r in lc.engine.runs.values())), 3)

    asyncio.run(scenario())
    assert [args["app_name"] for _tool, args in lc.calls] == ["one", "two"]
    assert lc.admission.in_use() == 0 and jobs.CONTROLS == {}


@pytest.mark.parametrize("fault,effects", [("running", 0), ("completed", 1)])
def test_a_step_record_fault_stops_the_run_once_and_releases_it(lifecycle, fault, effects):
    lc = lifecycle
    real = lc.store.set_step

    def flaky(job_id, idx, **fields):
        if fields.get("state") == fault:
            raise sqlite3.OperationalError("disk I/O error")
        return real(job_id, idx, **fields)
    lc.store.set_step = flaky

    async def scenario():
        assert lc.engine.start("pair").ok
        [run] = lc.engine.runs.values()
        await asyncio.wait_for(run["task"], 3)
        return run

    run = asyncio.run(scenario())
    assert len(lc.calls) == effects, "zero effects before dispatch, one after; never a retry"
    assert run["status"].startswith("fault:") and "disk I/O error" in run["status"]
    assert lc.store.get(run["job_id"])["state"] == "failed"
    assert lc.admission.in_use() == 0 and jobs.CONTROLS == {}


def test_a_terminal_record_fault_still_ends_the_run_in_the_monitor(lifecycle):
    lc = lifecycle
    real = lc.store.transition

    def refuse_terminal(job_id, state, **fields):
        if State(state) in {State.COMPLETED, State.FAILED, State.CANCELLED}:
            raise sqlite3.OperationalError("terminal write refused")
        return real(job_id, state, **fields)
    lc.store.transition = refuse_terminal

    async def scenario():
        assert lc.engine.start("pair").ok
        [run] = lc.engine.runs.values()
        ticket = jobs.CONTROLS[run["job_id"]].ticket
        released, original = [], ticket.release
        ticket.release = lambda: (released.append(1), original())[1]
        await asyncio.wait_for(run["task"], 3)
        return run, released

    run, released = asyncio.run(scenario())
    assert len(lc.calls) == 2 and released == [1]
    assert lc.admission.in_use() == 0 and jobs.CONTROLS == {}
    assert run["status"] == "complete", "the run must not stay 'running' in the deck"
    last = [payload for name, payload in lc.bus.events if name == "workflow"][-1]
    assert last["status"] == "complete"
    assert any("could not be recorded" in str(line) for line in lc.bus.lines)


@pytest.mark.parametrize("outcome", ["cancelled_while_queued", "expired_while_queued",
                                     "admitted"])
def test_a_workflow_restart_approval_is_spent_only_at_the_admitted_step(lifecycle, outcome):
    lc = lifecycle
    engine, store = lc.engine, lc.store
    assert engine.define("named", [{"tool": "open_app", "args": {"app_name": "synthetic"},
                                    "retries": 0}]).ok
    definition = engine.definitions["named"]
    record = store.create(kind="workflow", name="named", state=State.INTERRUPTED,
                          args={"workflow": "named",
                                "definition_digest": engine._definition_digest(definition)},
                          steps=[dict(definition["steps"][0], index=1)])
    job_id = record["id"]
    store.set_step(job_id, 1, state="interrupted")
    approval = store.request_approval(job_id, 1, "open_app", {"app_name": "synthetic"},
                                      reason="it may already have run", ttl_s=60)
    assert store.decide_approval(approval["id"], True)["state"] == "approved"

    async def scenario():
        blocker = lc.admission.submit(lc.cls, label="other work")
        assert engine.resume_job(store.get(job_id)).ok
        [run] = engine.runs.values()
        control = None
        for _ in range(200):
            await asyncio.sleep(0.005)
            control = jobs.CONTROLS.get(job_id)
            if control is not None and control.ticket is not None and control.ticket.queued:
                break
        assert control is not None and control.ticket.queued, "the resume must be waiting"
        assert lc.calls == [] and store.approval(approval["id"])["state"] == "approved"
        if outcome == "cancelled_while_queued":
            assert engine.cancel(job_id).ok
        elif outcome == "expired_while_queued":
            lc.clock.now += 61
        if outcome != "cancelled_while_queued":
            blocker.release()
        await asyncio.wait_for(run["task"], 3)
        blocker.release()
        return run

    run = asyncio.run(scenario())
    state = store.approval(approval["id"])["state"]
    if outcome == "cancelled_while_queued":
        assert lc.calls == [] and run["status"] == "cancelled"
        assert state == "approved", "a cancelled resume must not spend the approval"
    elif outcome == "expired_while_queued":
        assert lc.calls == [] and "expired" in run["status"]
        assert state == "expired"
    else:
        assert lc.calls == [("open_app", {"app_name": "synthetic"})]
        assert run["status"] == "complete" and state == "consumed"
        replay = store.consume_approval(job_id, 1, "open_app", {"app_name": "synthetic"})
        assert replay[0] is False, "the approval is single-use"
        again = engine.resume_job(store.get(job_id))
        assert not again.ok and len(lc.calls) == 1, "a finished run is never replayed"
    assert lc.admission.in_use() == 0 and jobs.CONTROLS == {}
