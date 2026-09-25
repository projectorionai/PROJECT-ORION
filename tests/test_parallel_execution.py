"""
Tests for Track B — the parallel execution core.

ORION was single-threaded by construction. The Live channel returns several
function calls in one turn and the worker awaited each in sequence, so three
independent two-second lookups cost six seconds. Plans ran strictly in order
even when their steps had nothing to do with each other, and any long tool held
the microphone in PROCESSING until it finished.

What must be true after the change:

  * independent read-only calls run CONCURRENTLY;
  * anything that drives the machine still runs alone, in the order asked for —
    an unclassified or forged tool is SERIAL, never parallel, because its side
    effects are unknowable;
  * responses still pair with the call that produced them, whatever order the
    work actually finished in;
  * a plan can express dependencies, and a step whose prerequisite failed is
    skipped rather than run against a precondition that never happened;
  * long read-only work can detach from the turn and be collected later.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.concurrency import (
    EXCLUSIVE_TOOLS,
    PARALLEL_TOOLS,
    ToolClass,
    classify,
    describe_plan,
    plan_batches,
)
from orion_core.data import ToolResult
from orion_core.jobs import Job, JobManager, JobState
from orion_core.live_worker import GenAILiveWorker
from orion_core.plan_executor import PlanExecutor


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)

    def connect(self, *_a, **_k):
        pass


class _Bus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig

    def lines(self):
        return " ".join(
            str(part) for payload in self.log.emitted for part in payload)


# ── classification ───────────────────────────────────────────────────────────


def test_read_only_tools_are_parallel():
    assert classify("web_search") is ToolClass.PARALLEL
    assert classify("resource_status") is ToolClass.PARALLEL
    assert classify("vision_analyse") is ToolClass.PARALLEL


def test_machine_driving_tools_are_serial():
    for tool in ("open_app", "desktop_control", "window_control", "file_controller",
                 "messaging", "clipboard_operate"):
        assert classify(tool) is ToolClass.SERIAL, tool


def test_lifecycle_and_self_modification_are_exclusive():
    for tool in ("shutdown_orion", "restart_orion", "self_repair", "forge"):
        assert classify(tool) is ToolClass.EXCLUSIVE, tool


def test_unknown_and_forged_tools_default_to_serial():
    """A forged tool's side effects are unknowable, so it must never be
    scheduled concurrently on the strength of not being recognised."""
    assert classify("some_tool_orion_forged_last_tuesday") is ToolClass.SERIAL
    assert classify("") is ToolClass.SERIAL


# ── the multi-action problem ─────────────────────────────────────────────────
#
# Classification is per-tool but safety is per-CALL. These lock the demotion of
# a read-only tool steered onto its writing branch.


def test_a_read_only_tool_stays_parallel_on_its_read_branch():
    assert classify("second_brain", {"action": "recall", "query": "x"}) is ToolClass.PARALLEL
    assert classify("awareness", {"action": "situation"}) is ToolClass.PARALLEL
    assert classify("research", {"action": "findings", "topic": "x"}) is ToolClass.PARALLEL
    assert classify("find_files", {"query": "x"}) is ToolClass.PARALLEL


def test_a_write_branch_is_demoted_to_serial():
    """The bug this guards: `awareness` add/complete are read-modify-write over
    one cognitive-state file, so two batched together lose an update."""
    assert classify("awareness", {"action": "add_task", "title": "x"}) is ToolClass.SERIAL
    assert classify("awareness", {"action": "complete_task"}) is ToolClass.SERIAL
    assert classify("second_brain", {"action": "ingest", "query": "x"}) is ToolClass.SERIAL
    assert classify("research", {"action": "start", "topic": "x"}) is ToolClass.SERIAL
    assert classify("executive", {"action": "schedule"}) is ToolClass.SERIAL
    assert classify("creator_intel", {"action": "add_creator"}) is ToolClass.SERIAL


def test_a_flag_that_launches_something_demotes_the_call():
    """find_files with open=true calls os.startfile — a read that becomes an act."""
    assert classify("find_files", {"query": "x", "open": True}) is ToolClass.SERIAL
    assert classify("find_files", {"query": "x", "open_first": True}) is ToolClass.SERIAL
    assert classify("find_files", {"query": "x", "open": False}) is ToolClass.PARALLEL


def test_action_matching_ignores_case_and_surrounding_space():
    assert classify("second_brain", {"action": "  INGEST "}) is ToolClass.SERIAL


def test_omitting_arguments_keeps_the_original_coarse_behaviour():
    """Callers without args are unchanged — the fix adds precision, not risk."""
    assert classify("second_brain") is ToolClass.PARALLEL
    assert classify("awareness") is ToolClass.PARALLEL


def test_a_write_branch_is_never_batched_with_its_neighbours():
    names = ["web_search", "awareness", "web_search"]
    args = [{"query": "a"}, {"action": "add_task", "title": "t"}, {"query": "b"}]
    assert plan_batches(names, args=args) == [[0], [1], [2]]
    # Same three tools, read branch: the middle call joins the batch.
    reads = [{"query": "a"}, {"action": "situation"}, {"query": "b"}]
    assert plan_batches(names, args=reads) == [[0, 1, 2]]


def test_a_write_branch_may_not_be_backgrounded():
    jobs = JobManager(_Bus(), lambda *_a, **_k: None)
    assert jobs.can_background("research", {"action": "findings"})[0]
    allowed, reason = jobs.can_background("research", {"action": "start", "topic": "x"})
    assert not allowed and "read-only" in reason


def test_a_write_branch_is_not_batchable_in_a_plan():
    from orion_core.plan_executor import _Step
    read = _Step(index=0, step_id="a", tool="second_brain",
                 args={"action": "recall"}, on_fail="stop")
    write = _Step(index=1, step_id="b", tool="second_brain",
                  args={"action": "ingest"}, on_fail="stop")
    assert read.batchable and not write.batchable


# ── background-job progress ──────────────────────────────────────────────────


async def test_a_job_reports_progress_while_it_runs():
    """'Works while talking' is only true if you can see how far along he is."""
    from orion_core.jobs import report_progress

    async def _work(_tool, _args):
        for step in range(1, 5):
            report_progress(step / 4, f"step {step} of 4")
            await asyncio.sleep(0)
        return ToolResult("done")

    bus = _Bus()
    jobs = JobManager(bus, _work)
    job = jobs.start("research", {"topic": "x"})
    await asyncio.sleep(0.02)
    assert job.fraction == 1.0 and job.note == "step 4 of 4"
    assert sum(1 for payload in bus.dashboard_event.emitted
               if payload[0] == "job") > 2, "progress must reach the interface"


async def test_progress_appears_in_the_summary_with_an_estimate():
    from orion_core.jobs import report_progress

    async def _work(_tool, _args):
        report_progress(0.5, "halfway")
        await asyncio.sleep(0.02)
        return ToolResult("done")

    jobs = JobManager(_Bus(), _work)
    job = jobs.start("research", {})
    await asyncio.sleep(0.005)
    summary = job.summary()
    assert "50%" in summary and "halfway" in summary
    assert job.eta_s() is not None


def test_no_estimate_is_offered_from_a_sliver_of_progress():
    """Extrapolating from 2% produces confident nonsense."""
    job = Job(id="job-1", tool="research", args={}, label="research")
    job._record_progress(0.02, "just started")
    assert job.eta_s() is None


def test_progress_never_goes_backwards():
    job = Job(id="job-1", tool="research", args={}, label="research")
    job._record_progress(0.6, "most of the way")
    job._record_progress(0.2, "re-scoped, more to do")
    assert job.fraction == 0.6, "a bar that slides backwards reads as a bug"
    assert job.note == "re-scoped, more to do", "...but the note tells the truth"


def test_an_unknown_fraction_is_none_not_zero():
    """Zero renders as 'stuck'; None renders as 'not saying'."""
    job = Job(id="job-1", tool="research", args={}, label="research")
    assert job.fraction is None and job.progress_text() == ""
    job._record_progress(None, "working on it")
    assert job.fraction is None and job.progress_text() == "working on it"


def test_nonsense_progress_is_ignored_rather_than_shown():
    job = Job(id="job-1", tool="research", args={}, label="research")
    job._record_progress("three quarters", "")
    assert job.fraction is None
    job._record_progress(4.0, "")
    assert job.fraction == 1.0, "clamped, not rejected"


async def test_reporting_progress_outside_a_job_is_harmless():
    from orion_core.jobs import report_progress
    assert report_progress(0.5, "from the foreground") is False


async def test_progress_can_be_reported_against_a_named_job():
    async def _work(_tool, _args):
        await asyncio.sleep(0.05)
        return ToolResult("done")

    jobs = JobManager(_Bus(), _work)
    job = jobs.start("research", {})
    assert jobs.set_progress(job.id, 0.3, "a third done")
    assert job.fraction == 0.3
    assert not jobs.set_progress("no-such-job", 0.5)
    assert jobs.report().ok, "the whole-board summary must still be reachable"


def test_no_tool_is_both_parallel_and_exclusive():
    assert not (PARALLEL_TOOLS & EXCLUSIVE_TOOLS)


def test_every_classified_tool_actually_exists():
    """A typo in the table silently downgrades a tool to SERIAL forever."""
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    declared = {t["name"] for t in TOOL_DECLARATIONS}
    assert not (PARALLEL_TOOLS - declared), PARALLEL_TOOLS - declared
    assert not (EXCLUSIVE_TOOLS - declared), EXCLUSIVE_TOOLS - declared


# ── batch planning ───────────────────────────────────────────────────────────


def test_adjacent_read_only_calls_batch_together():
    assert plan_batches(["web_search", "resource_status", "token_usage"]) == [[0, 1, 2]]


def test_side_effecting_calls_break_the_batch_and_keep_their_order():
    names = ["web_search", "resource_status", "open_app", "find_files"]
    assert plan_batches(names) == [[0, 1], [2], [3]]


def test_exclusive_call_always_stands_alone():
    assert plan_batches(["web_search", "forge", "web_search"]) == [[0], [1], [2]]


def test_batches_respect_the_ceiling():
    names = ["web_search"] * 5
    assert plan_batches(names, limit=2) == [[0, 1], [2, 3], [4]]


def test_limit_of_one_reproduces_the_old_sequential_behaviour():
    names = ["web_search", "resource_status", "open_app"]
    assert plan_batches(names, limit=1) == [[0], [1], [2]]


def test_parallelism_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setenv("ORION_PARALLEL_TOOLS", "0")
    assert plan_batches(["web_search", "resource_status"]) == [[0], [1]]


def test_empty_plan_is_empty():
    assert plan_batches([]) == []


def test_describe_plan_reads_clearly():
    names = ["web_search", "resource_status", "open_app"]
    described = describe_plan(names, plan_batches(names))
    assert "web_search ‖ resource_status" in described
    assert described.endswith("open_app")


# ── the live tool-call loop ──────────────────────────────────────────────────


class _Call:
    def __init__(self, name, args=None, call_id=None):
        self.name = name
        self.args = args or {}
        self.id = call_id or f"id-{name}"


class _ToolCall:
    def __init__(self, *calls):
        self.function_calls = list(calls)


class _RecordingDispatcher:
    """Records concurrency: peak > 1 means calls genuinely overlapped."""

    def __init__(self, delay=0.05, fail=(), media_for=()):
        self.delay = delay
        self.fail = set(fail)
        self.media_for = set(media_for)
        self.active = 0
        self.peak = 0
        self.order: list[str] = []

    async def dispatch_chain(self, name, args):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(name)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        if name in self.fail:
            raise RuntimeError(f"{name} exploded")
        media = {"mime_type": "image/jpeg", "data": name.encode()} \
            if name in self.media_for else None
        return ToolResult(f"{name} done", ok=True, media=media)


def _worker(dispatcher):
    # google.genai is imported lazily on first use (~2.5 s). Pay that here, not
    # inside a timed section that is measuring concurrency.
    from orion_core import live_worker as _live_worker
    _live_worker.types.FunctionResponse  # noqa: B018
    w = GenAILiveWorker.__new__(GenAILiveWorker)
    w.bus = _Bus()
    w.dispatcher = dispatcher
    w.mic = None
    w.session = None
    w._send_closed = True          # transport not ready: no response is sent
    w.stop_event = asyncio.Event()
    w._send_lock = asyncio.Lock()
    w.tool_busy = False
    w.connected = False
    w.sent_media: list = []
    w._states: list = []
    w._marks = 0

    async def _send_media(media):
        w.sent_media.append(media)

    w._send_media = _send_media
    w._emit_state = lambda state: w._states.append(state)
    w._mark_turn_progress = lambda: setattr(w, "_marks", w._marks + 1)

    class _Speech:
        def output_active(self):
            return False

    w.speech = _Speech()
    return w


def test_independent_calls_run_concurrently():
    dispatcher = _RecordingDispatcher(delay=0.08)
    worker = _worker(dispatcher)
    call = _ToolCall(_Call("web_search"), _Call("resource_status"), _Call("token_usage"))

    started = time.monotonic()
    asyncio.run(worker._handle_tool_call(call))
    elapsed = time.monotonic() - started

    assert dispatcher.peak == 3, "all three should have overlapped"
    assert elapsed < 0.20, f"sequential would take ~0.24s, took {elapsed:.2f}s"


def test_machine_driving_calls_never_overlap():
    dispatcher = _RecordingDispatcher(delay=0.02)
    worker = _worker(dispatcher)
    call = _ToolCall(_Call("open_app"), _Call("desktop_control"), _Call("window_control"))

    asyncio.run(worker._handle_tool_call(call))

    assert dispatcher.peak == 1, "side-effecting tools must run alone"
    assert dispatcher.order == ["open_app", "desktop_control", "window_control"]


def test_mixed_batch_preserves_the_order_of_side_effects():
    dispatcher = _RecordingDispatcher(delay=0.02)
    worker = _worker(dispatcher)
    call = _ToolCall(
        _Call("web_search"), _Call("resource_status"),
        _Call("open_app"), _Call("find_files"))

    asyncio.run(worker._handle_tool_call(call))

    assert dispatcher.peak == 2                      # only the two lookups
    assert dispatcher.order.index("open_app") > dispatcher.order.index("web_search")
    assert dispatcher.order[-1] == "find_files"      # after open_app, as asked


def test_responses_pair_with_their_own_call_whatever_the_finish_order():
    """The model matches answers to calls by id; a reordered response list
    would pair every answer with the wrong question."""
    class _Staggered(_RecordingDispatcher):
        async def dispatch_chain(self, name, args):
            # Deliberately finish in reverse order of request.
            await asyncio.sleep({"web_search": 0.06,
                                 "resource_status": 0.03,
                                 "token_usage": 0.01}[name])
            return ToolResult(f"{name} done", ok=True)

    worker = _worker(_Staggered())
    captured: list = []
    worker._function_response = lambda call_id, name, payload: captured.append(
        (call_id, name, payload)) or (call_id, name, payload)

    asyncio.run(worker._handle_tool_call(_ToolCall(
        _Call("web_search"), _Call("resource_status"), _Call("token_usage"))))

    assert [name for _cid, name, _p in captured] == [
        "web_search", "resource_status", "token_usage"]
    for call_id, name, payload in captured:
        assert call_id == f"id-{name}"
        assert name in str(payload)


def test_one_failure_in_a_batch_does_not_lose_the_others():
    """An unanswered call id stalls the model's whole turn, so a failure must
    still produce a response payload."""
    dispatcher = _RecordingDispatcher(delay=0.01, fail={"resource_status"})
    worker = _worker(dispatcher)
    captured: list = []
    worker._function_response = lambda call_id, name, payload: captured.append(
        (name, payload)) or (name, payload)

    asyncio.run(worker._handle_tool_call(_ToolCall(
        _Call("web_search"), _Call("resource_status"), _Call("token_usage"))))

    assert len(captured) == 3
    payloads = dict(captured)
    assert payloads["resource_status"]["ok"] is False
    assert "exploded" in payloads["resource_status"]["result"]
    assert payloads["web_search"]["ok"] is True


def test_media_is_streamed_in_call_order_after_the_batch():
    dispatcher = _RecordingDispatcher(delay=0.01,
                                      media_for={"vision_analyse", "capture_screen"})
    worker = _worker(dispatcher)

    asyncio.run(worker._handle_tool_call(_ToolCall(
        _Call("vision_analyse"), _Call("capture_screen"))))

    assert [m["data"] for m in worker.sent_media] == [b"vision_analyse", b"capture_screen"]


def test_a_single_call_still_works():
    dispatcher = _RecordingDispatcher(delay=0.01)
    worker = _worker(dispatcher)
    asyncio.run(worker._handle_tool_call(_ToolCall(_Call("web_search"))))
    assert dispatcher.order == ["web_search"]
    assert worker.tool_busy is False


def test_no_calls_is_a_clean_no_op():
    dispatcher = _RecordingDispatcher()
    worker = _worker(dispatcher)
    asyncio.run(worker._handle_tool_call(_ToolCall()))
    assert dispatcher.order == []


# ── the plan executor ────────────────────────────────────────────────────────


class _PlanDispatch:
    def __init__(self, delay=0.03, fail=()):
        self.delay = delay
        self.fail = set(fail)
        self.active = 0
        self.peak = 0
        self.order: list[str] = []

    async def __call__(self, tool, args):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(tool)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.active -= 1
        ok = tool not in self.fail
        return ToolResult(f"{tool} {'ok' if ok else 'failed'}", ok=ok)


def _executor(dispatch):
    return PlanExecutor(_Bus(), dispatch)


def test_linear_plan_batches_independent_lookups():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "web_search", "args": {}},
        {"tool": "resource_status", "args": {}},
        {"tool": "open_app", "args": {}},
    ]))
    assert result.ok
    assert dispatch.peak == 2
    assert dispatch.order[-1] == "open_app"
    assert "ran in parallel" in result.text


def test_dag_plan_runs_dependency_waves():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {}},
        {"id": "b", "tool": "resource_status", "args": {}},
        {"id": "c", "tool": "open_app", "args": {}, "after": ["a", "b"]},
    ], objective="fan in"))
    assert result.ok, result.text
    assert dispatch.peak == 2                     # a and b together
    assert dispatch.order[-1] == "open_app"       # c strictly after both


def test_step_whose_dependency_failed_is_skipped_not_run():
    """Running a step whose precondition never happened is how a plan does
    damage — it must be skipped and reported."""
    dispatch = _PlanDispatch(fail={"web_search"})
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {}, "on_fail": "continue"},
        {"id": "b", "tool": "open_app", "args": {}, "after": ["a"]},
    ]))
    assert not result.ok
    assert "open_app" not in dispatch.order
    assert "skipped on a failed dependency" in result.text


def test_dependency_cycle_is_detected_not_deadlocked():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {}, "after": ["b"]},
        {"id": "b", "tool": "find_files", "args": {}, "after": ["a"]},
    ]))
    assert not result.ok
    assert "cycle" in result.text.lower()
    assert dispatch.order == []


def test_edge_to_an_unknown_step_is_ignored_not_deadlocked():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {}, "after": ["typo"]},
    ]))
    assert result.ok, result.text
    assert dispatch.order == ["web_search"]


def test_abort_policy_stops_the_remaining_plan():
    dispatch = _PlanDispatch(fail={"open_app"})
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "open_app", "args": {}, "on_fail": "abort"},
        {"tool": "find_files", "args": {}},
    ], max_retries=1))
    assert not result.ok
    assert "aborted" in result.text
    assert "find_files" not in dispatch.order


def test_dangerous_tools_are_still_refused_inside_a_plan():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "shutdown_orion", "args": {}},
        {"tool": "self_repair", "args": {}},
    ]))
    assert dispatch.order == []
    assert "not permitted inside a plan" in result.text


def test_visual_steps_never_batch_even_though_vision_verify_is_parallel():
    """vision_verify's verdict is a before/after pixel diff of one screen, so
    two running at once would each measure the other's changes."""
    from orion_core.plan_executor import _Step
    assert classify("vision_verify") is ToolClass.PARALLEL
    step = _Step(index=1, step_id="s", tool="vision_verify", args={}, on_fail="retry")
    assert step.batchable is False


def test_empty_plan_is_rejected():
    assert not asyncio.run(_executor(_PlanDispatch()).execute([])).ok


# ── background jobs ──────────────────────────────────────────────────────────


def _jobs(delay=0.02, fail=()):
    dispatch = _PlanDispatch(delay=delay, fail=fail)
    return JobManager(_Bus(), dispatch), dispatch


def test_a_read_only_tool_detaches_from_the_turn():
    async def scenario():
        manager, dispatch = _jobs(delay=0.05)
        job = manager.start("research", {"topic": "x"}, label="deep dive")
        # start() returns immediately — the work has not finished yet.
        assert job.state is JobState.RUNNING
        assert dispatch.order == []
        await asyncio.sleep(0.12)
        return manager, job

    manager, job = asyncio.run(scenario())
    assert job.state is JobState.SUCCEEDED
    assert job.ok
    assert manager.collect(job.id).ok


def test_a_machine_driving_tool_cannot_be_backgrounded():
    async def scenario():
        manager, _ = _jobs()
        return manager.start("open_app", {"app_name": "notepad"})

    outcome = asyncio.run(scenario())
    assert isinstance(outcome, ToolResult)
    assert not outcome.ok
    assert "must run in the conversation" in outcome.text


def test_a_failing_job_records_the_failure_rather_than_vanishing():
    async def scenario():
        manager, _ = _jobs(delay=0.01, fail={"research"})
        job = manager.start("research", {})
        await asyncio.sleep(0.08)
        return manager, job

    manager, job = asyncio.run(scenario())
    assert job.state is JobState.FAILED
    assert not manager.collect(job.id).ok


def test_collecting_a_running_job_says_so():
    async def scenario():
        manager, _ = _jobs(delay=0.3)
        job = manager.start("research", {})
        collected = manager.collect(job.id)
        job._task.cancel()
        return collected

    collected = asyncio.run(scenario())
    assert not collected.ok
    assert "still running" in collected.text


def test_a_job_can_be_found_by_label_or_tool_name():
    async def scenario():
        manager, _ = _jobs(delay=0.01)
        manager.start("research", {}, label="competitor sweep")
        await asyncio.sleep(0.06)
        return manager

    manager = asyncio.run(scenario())
    assert manager.get("competitor") is not None
    assert manager.get("research") is not None
    assert manager.get("nothing like this") is None


def test_the_active_job_ceiling_is_enforced():
    async def scenario():
        manager, _ = _jobs(delay=0.4)
        started = [manager.start("research", {"n": i}) for i in range(manager.max_active)]
        refused = manager.start("research", {"n": 99})
        for job in started:
            job._task.cancel()
        return refused

    refused = asyncio.run(scenario())
    assert isinstance(refused, ToolResult) and not refused.ok
    assert "already running" in refused.text


def test_cancelling_a_job_marks_it_cancelled():
    async def scenario():
        manager, _ = _jobs(delay=0.4)
        job = manager.start("research", {})
        cancelled = manager.cancel(job.id)
        await asyncio.sleep(0.05)
        return manager, job, cancelled

    manager, job, cancelled = asyncio.run(scenario())
    assert cancelled.ok
    assert job.state is JobState.CANCELLED


def test_report_is_readable_when_nothing_has_run():
    manager, _ = _jobs()
    assert "No background jobs" in manager.report().text


def test_report_separates_running_from_finished():
    async def scenario():
        manager, _ = _jobs(delay=0.01)
        manager.start("research", {}, label="one")
        await asyncio.sleep(0.06)
        slow = manager.start("web_search", {}, label="two")
        text = manager.report().text
        slow._task.cancel()
        return text

    text = asyncio.run(scenario())
    assert "Running (1)" in text
    assert "Finished (1)" in text


def test_starting_a_job_without_a_loop_is_refused():
    manager, _ = _jobs()
    outcome = manager.start("research", {})
    assert isinstance(outcome, ToolResult) and not outcome.ok
    assert "event loop" in outcome.text
