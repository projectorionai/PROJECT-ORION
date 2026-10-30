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

import pytest

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


@pytest.fixture(autouse=True)
def _isolated_admission_and_store(monkeypatch):
    """Each test gets its own admission policy and in-memory job store, so a
    job left queued by one test can never hold a slot in the next."""
    from orion_core import admission, job_store, jobs
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission())
    monkeypatch.setattr(job_store, "_STORE", job_store.JobStore(None))
    jobs.CONTROLS.clear()
    yield
    jobs.CONTROLS.clear()


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


def test_all_acting_specialist_playbooks_have_serial_selector_rules():
    from orion_core.agents import PLAYBOOKS
    from orion_core.concurrency import ACTING_SPECIALISTS, WRITE_ARG_VALUES

    acting = {name for name, playbook in PLAYBOOKS.items() if playbook.acting}
    assert ACTING_SPECIALISTS == acting
    assert WRITE_ARG_VALUES["agent_dispatch"]["agent"] == acting


@pytest.mark.parametrize("selector", ["files", "FILES", "  FiLeS  "])
def test_named_files_specialist_is_serial_and_cannot_detach(selector):
    from orion_core.job_store import SIDE_EFFECT, side_effect_class
    from orion_core.plan_executor import _Step

    args = {"agent": selector, "request": "Synthetic file job"}
    assert classify("agent_dispatch", args) is ToolClass.SERIAL
    assert side_effect_class("agent_dispatch", args) == SIDE_EFFECT
    assert not _Step(1, "files", "agent_dispatch", args, "retry").batchable
    assert not JobManager(_Bus(), lambda *_args: None).can_background("agent_dispatch", args)[0]
    assert plan_batches(["web_search", "agent_dispatch", "web_search"],
                        args=[{}, args, {}]) == [[0], [1], [2]]


@pytest.mark.parametrize("selector", ["", "auto", "AUTO", "  auto  ",
    "marketing", "coding", "research", "neuroscience", "aiml", "security",
    "reliability", "integrations", "continuity", "triage"])
def test_observational_specialist_routes_keep_existing_parallel_policy(selector):
    from orion_core.job_store import READ_ONLY, side_effect_class

    args = {"agent": selector, "request": "Synthetic observation"}
    assert classify("agent_dispatch", args) is ToolClass.PARALLEL
    assert side_effect_class("agent_dispatch", args) == READ_ONLY
    assert JobManager(_Bus(), lambda *_args: None).can_background("agent_dispatch", args)[0]
    assert classify("agent_dispatch") is ToolClass.PARALLEL


@pytest.mark.parametrize("engine_kind", ["plan", "workflow"])
def test_failed_files_specialist_is_never_retried_after_a_possible_effect(engine_kind, tmp_path,
                                                                         monkeypatch):
    from types import SimpleNamespace
    from orion_core import jobs as jobs_module
    from orion_core.plan_executor import _Step
    from orion_core.workflow_engine import WorkflowEngine

    calls = []

    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult("Synthetic partial file effect", ok=False,
                          receipt={"safe_to_retry": True})

    async def instant(*_args):
        pass

    monkeypatch.setattr(asyncio, "sleep", instant)
    monkeypatch.setattr(jobs_module, "_RESUMERS", dict(jobs_module._RESUMERS))
    args = {"agent": "files", "request": "Synthetic file job"}
    if engine_kind == "plan":
        result = asyncio.run(PlanExecutor(_Bus(), dispatch)._run_step(
            _Step(1, "files", "agent_dispatch", args, "retry"), 3))
        assert not result.ok and result.attempts == 1
        assert "not retried" in result.detail
    else:
        engine = WorkflowEngine(_Bus(), SimpleNamespace(dispatch=dispatch),
                                path=tmp_path / "workflow.json")
        run = {"id": "synthetic", "workflow": "files", "input": "", "step": 0,
               "total": 1, "transcript": []}
        asyncio.run(engine._execute(run, {"steps": [{"tool": "agent_dispatch", "args": args,
                                                    "retries": 2}]}))
        assert run["status"].startswith("failed")
        assert "not retried" in run["transcript"][0]["text"]
    assert calls == [("agent_dispatch", args)]


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


def test_writes_the_11_oct_audit_found_behind_read_only_tools_are_serial():
    """Each of these wrote a setting or a record while classify() said
    PARALLEL - which reflex learning and the Live batch scheduler both trust."""
    writes = [
        ("token_usage", {"action": "set_budget", "provider": "gemini"}),
        ("display_info", {"action": "rename", "device": "1", "label": "Left"}),
        ("second_brain", {"action": "remember", "query": "x"}),
        ("second_brain", {"action": "record", "query": "x"}),
        ("conversation_recall", {"action": "compress"}),
        ("executive", {"action": "meeting_summary"}),
        ("executive", {"action": "plan_workflow"}),
        ("code_changes", {"action": "snapshot"}),
    ]
    for name, args in writes:
        assert classify(name, args) is ToolClass.SERIAL, (name, args)
    # Their read branches stay parallel.
    assert classify("token_usage", {"action": "report"}) is ToolClass.PARALLEL
    assert classify("conversation_recall", {"action": "recall", "query": "x"}) is ToolClass.PARALLEL
    assert classify("code_changes", {"action": "diff"}) is ToolClass.PARALLEL


def test_tools_whose_default_writes_are_read_only_only_by_name():
    """research with no action STARTS a run and grew a dozen write aliases a
    denylist never caught; transcript with no action EXPORTS a file."""
    for args in ({"topic": "x"}, {"action": "conduct", "topic": "x"},
                 {"action": "dossier", "topic": "x"}, {"action": "cancel"},
                 {"action": "deep_research", "topic": "x"}):
        assert classify("research", args) is ToolClass.SERIAL, args
    for action in ("status", "findings", "sources", "gaps", "neuroscience_pilot"):
        assert classify("research", {"action": action}) is ToolClass.PARALLEL, action
    assert classify("transcript", {}) is ToolClass.SERIAL
    assert classify("transcript", {"action": "export"}) is ToolClass.SERIAL
    assert classify("transcript", {"action": "search", "query": "x"}) is ToolClass.PARALLEL


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


@pytest.mark.parametrize("action", ["reflexes", "shortcuts", "learned"])
@pytest.mark.parametrize("flag", ["promote", "forget"])
def test_diagnostic_shortcut_mutations_cannot_be_backgrounded_or_batched(action, flag):
    args = {"action": action, flag: "check my usage"}
    assert classify("diagnostics", args) is ToolClass.SERIAL
    assert plan_batches(["web_search", "diagnostics", "web_search"],
                        args=[{}, args, {}]) == [[0], [1], [2]]
    manager = JobManager(_Bus(), lambda *_a, **_k: None)
    assert not manager.can_background("diagnostics", args)[0]


@pytest.mark.parametrize("args", [
    {}, {"query": "Birmingham"}, {"mode": "driving"},
    *({"action": action} for action in (
        "locate", "find", "fly", "directions", "navigate", "route",
        "show_route", "route_on_globe", "SHOW ROUTE", "route-on-globe")),
])
def test_geo_scene_or_browser_actions_cannot_run_as_background_observations(args):
    assert classify("geo", args) is ToolClass.SERIAL
    manager = JobManager(_Bus(), lambda *_a, **_k: None)
    assert not manager.can_background("geo", args)[0]


@pytest.mark.parametrize("tool,args", [
    ("diagnostics", {"action": "incident"}),
    ("diagnostics", {"action": "latency"}),
    ("diagnostics", {"action": "shortcuts", "promote": "", "forget": ""}),
    *(("geo", {"action": action}) for action in (
        "describe", "nearby", "weather", "forecast", "distance", "how_far", "places")),
])
def test_diagnostics_and_geo_observations_keep_parallel_and_background_routes(tool, args):
    assert classify(tool, args) is ToolClass.PARALLEL
    assert JobManager(_Bus(), lambda *_a, **_k: None).can_background(tool, args)[0]


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
    job = jobs.start("research", {"action": "findings", "topic": "x"})
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
    job = jobs.start("research", {"action": "findings"})
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
    job = jobs.start("research", {"action": "findings"})
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

    worker = _live_worker_with_session(_Staggered())
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
    worker = _live_worker_with_session(dispatcher)
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
    worker = _live_worker_with_session(dispatcher)

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
        {"tool": "web_search", "args": {"query": "orion"}},
        {"tool": "resource_status", "args": {}},
        {"tool": "open_app", "args": {"app_name": "notepad"}},
    ]))
    assert result.ok
    assert dispatch.peak == 2
    assert dispatch.order[-1] == "open_app"
    assert "ran in parallel" in result.text


def test_dag_plan_runs_dependency_waves():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {"query": "orion"}},
        {"id": "b", "tool": "resource_status", "args": {}},
        {"id": "c", "tool": "open_app", "args": {"app_name": "notepad"}, "after": ["a", "b"]},
    ], objective="fan in"))
    assert result.ok, result.text
    assert dispatch.peak == 2                     # a and b together
    assert dispatch.order[-1] == "open_app"       # c strictly after both


def test_step_whose_dependency_failed_is_skipped_not_run():
    """Running a step whose precondition never happened is how a plan does
    damage — it must be skipped and reported."""
    dispatch = _PlanDispatch(fail={"web_search"})
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {"query": "orion"}, "on_fail": "continue"},
        {"id": "b", "tool": "open_app", "args": {"app_name": "notepad"}, "after": ["a"]},
    ]))
    assert not result.ok
    assert "open_app" not in dispatch.order
    assert "skipped on a failed dependency" in result.text


def test_dependency_cycle_is_detected_not_deadlocked():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "web_search", "args": {"query": "orion"}, "after": ["b"]},
        {"id": "b", "tool": "find_files", "args": {}, "after": ["a"]},
    ]))
    assert not result.ok
    assert "cycle" in result.text.lower()
    assert dispatch.order == []


def test_edge_to_an_unknown_step_invalidates_the_whole_plan():
    """INTENTIONAL SEMANTIC CHANGE (2026-09-26). This test used to assert the
    opposite: an edge to an unknown step was logged and DROPPED, and the step
    ran anyway. For a consequential step that relaxes its precondition — a
    misspelt "after": ["backup"] let "delete" run with no backup. A plan that
    names a prerequisite that does not exist is now rejected before anything
    is dispatched, never repaired into a different plan."""
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "backup", "tool": "web_search", "args": {"query": "orion"}},
        {"id": "a", "tool": "find_files", "args": {}, "after": ["bakup"]},
    ]))
    assert not result.ok
    assert "PLAN REJECTED" in result.text and "'bakup'" in result.text
    assert dispatch.order == [], "not even the valid first step may run"


def test_abort_policy_stops_the_remaining_plan():
    dispatch = _PlanDispatch(fail={"open_app"})
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "open_app", "args": {"app_name": "notepad"}, "on_fail": "abort"},
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
        job = manager.start("research", {"action": "findings", "topic": "x"}, label="deep dive")
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
        job = manager.start("research", {"action": "findings"})
        await asyncio.sleep(0.08)
        return manager, job

    manager, job = asyncio.run(scenario())
    assert job.state is JobState.FAILED
    assert not manager.collect(job.id).ok


def test_collecting_a_running_job_says_so():
    async def scenario():
        manager, _ = _jobs(delay=0.3)
        job = manager.start("research", {"action": "findings"})
        collected = manager.collect(job.id)
        job._task.cancel()
        return collected

    collected = asyncio.run(scenario())
    assert not collected.ok
    assert "still running" in collected.text


def test_a_job_can_be_found_by_label_or_tool_name():
    async def scenario():
        manager, _ = _jobs(delay=0.01)
        manager.start("research", {"action": "findings"}, label="competitor sweep")
        await asyncio.sleep(0.06)
        return manager

    manager = asyncio.run(scenario())
    assert manager.get("competitor") is not None
    assert manager.get("research") is not None
    assert manager.get("nothing like this") is None


def test_the_active_job_ceiling_is_enforced(monkeypatch):
    """INTENTIONAL CHANGE (2026-09-26): past the running limit a job is no
    longer refused outright — it QUEUES, visibly, under the machine-wide
    admission policy. Only past the pending limit is it refused, with a
    reason. The ceiling on RUNNING jobs is unchanged."""
    from orion_core import admission

    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(
        {admission.WorkClass.USER: admission.ClassPolicy(running=4, pending=1)}))

    async def scenario():
        manager, dispatch = _jobs(delay=0.4)
        started = [manager.start("research", {"action": "findings", "n": i})
                   for i in range(manager.max_active)]
        queued = manager.start("research", {"action": "findings", "n": 98}, label="fifth")
        refused = manager.start("research", {"action": "findings", "n": 99})
        await asyncio.sleep(0.05)
        running_peak = dispatch.peak
        summary = queued.summary()
        for job in started + [queued]:
            job._task.cancel()
        await asyncio.sleep(0)
        return manager, queued, refused, running_peak, summary

    manager, queued, refused, peak, summary = asyncio.run(scenario())
    assert manager.max_active == 4 and peak == 4, "never more than four at once"
    assert queued.state in {JobState.QUEUED, JobState.CANCELLED}
    assert "queued" in summary and "in line" in summary
    assert isinstance(refused, ToolResult) and not refused.ok
    assert "already waiting" in refused.text


def test_cancelling_a_job_marks_it_cancelled():
    async def scenario():
        manager, _ = _jobs(delay=0.4)
        job = manager.start("research", {"action": "findings"})
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
        manager.start("research", {"action": "findings"}, label="one")
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
    outcome = manager.start("research", {"action": "findings"})
    assert isinstance(outcome, ToolResult) and not outcome.ok
    assert "event loop" in outcome.text


# ── strict plan validation: an invalid plan dispatches NOTHING ───────────────

_VALID = {"tool": "web_search", "args": {"query": "orion"}}

_INVALID_PLANS = {
    "duplicate ids": [{"id": "a", **_VALID}, {"id": "a", "tool": "find_files", "args": {}}],
    "missing prerequisite": [{"id": "a", **_VALID},
                             {"id": "b", "tool": "find_files", "args": {}, "after": ["zz"]}],
    "self dependency": [{"id": "a", **_VALID, "after": ["a"]}],
    "cycle": [{"id": "a", **_VALID, "after": ["c"]},
              {"id": "b", "tool": "find_files", "args": {}, "after": ["a"]},
              {"id": "c", "tool": "resource_status", "args": {}, "after": ["b"]}],
    "unknown tool": [_VALID, {"tool": "launch_rockets", "args": {}}],
    "missing required argument": [_VALID, {"tool": "open_app", "args": {}}],
    "wrong argument type": [_VALID, {"tool": "web_search", "args": {"query": {"x": 1}}}],
    "unreadable boolean": [_VALID, {"tool": "find_files", "args": {"open": "perhaps"}}],
    "not a number": [_VALID, {"tool": "find_files", "args": {"limit": "lots"}}],
    "args are not an object": [_VALID, {"tool": "find_files", "args": ["x"]}],
    "args_json is not json": [_VALID, {"tool": "find_files", "args_json": "{nope"}],
    "step is not an object": [_VALID, "open notepad"],
    "step names no tool": [_VALID, {"args": {}}],
    "unknown failure policy": [{**_VALID, "on_fail": "explode"}],
    "forbidden tool": [_VALID, {"tool": "restart_orion", "args": {}}],
    "nested plan": [_VALID, {"tool": "execute_plan", "args": {"steps": []}}],
    "oversize": [dict(_VALID) for _ in range(25)],
}


@pytest.mark.parametrize("case", sorted(_INVALID_PLANS))
def test_an_invalid_plan_dispatches_nothing(case):
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute(_INVALID_PLANS[case]))
    assert not result.ok
    assert "PLAN REJECTED" in result.text or "not permitted" in result.text, result.text
    assert dispatch.order == [], f"{case}: something ran from an invalid plan"


def test_a_rejected_plan_lists_every_problem_not_just_the_first():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", "tool": "launch_rockets", "args": {}},
        {"id": "a", "tool": "open_app", "args": {}},
    ]))
    assert "unknown tool" in result.text and "share the id" in result.text
    assert "missing required argument" in result.text


def test_a_prerequisite_may_name_a_step_by_its_position():
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        _VALID,
        {"tool": "open_app", "args": {"app_name": "notepad"}, "after": [1]},
    ]))
    assert result.ok, result.text
    assert dispatch.order == ["web_search", "open_app"]


def test_tools_registered_at_runtime_are_known_to_the_validator():
    """A forged or plugin tool has a handler but no static declaration."""
    class _Dispatcher:
        _tool_handlers = {"my_forged_tool": None}

        def __init__(self):
            self.inner = _PlanDispatch()

        def handler_table(self):
            return {"web_search": None}

        async def dispatch(self, tool, args):
            return await self.inner(tool, args)

    dispatcher = _Dispatcher()
    executor = PlanExecutor(_Bus(), dispatcher.dispatch)
    result = asyncio.run(executor.execute([{"tool": "my_forged_tool", "args": {"x": 1}}]))
    assert result.ok, result.text
    assert dispatcher.inner.order == ["my_forged_tool"]


# ── failure, retries and dependants ──────────────────────────────────────────

def test_every_transitive_dependant_of_a_failed_step_is_skipped():
    dispatch = _PlanDispatch(fail={"web_search"})
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "a", **_VALID, "on_fail": "continue"},
        {"id": "b", "tool": "open_app", "args": {"app_name": "x"}, "after": ["a"]},
        {"id": "c", "tool": "close_app", "args": {"app_name": "x"}, "after": ["b"]},
        {"id": "d", "tool": "resource_status", "args": {}},
    ], max_retries=1))
    assert "open_app" not in dispatch.order and "close_app" not in dispatch.order
    assert "resource_status" in dispatch.order, "independent work still runs"
    assert "2 skipped on a failed dependency" in result.text


class _RaisingDispatch(_PlanDispatch):
    async def __call__(self, tool, args):
        self.order.append(tool)
        raise TimeoutError("no answer")


def test_a_side_effecting_step_is_not_retried_when_its_outcome_is_unknown():
    """An exception mid-call leaves the outcome unknown; running a side
    effect again could do it twice. A read-only step may be retried."""
    dispatch = _RaisingDispatch()
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "open_app", "args": {"app_name": "x"}, "on_fail": "continue"},
        {"tool": "web_search", "args": {"query": "x"}},
    ], max_retries=3))
    assert dispatch.order.count("open_app") == 1
    assert dispatch.order.count("web_search") == 3
    assert "may already have taken effect" in result.text


def test_an_explicit_mutation_failure_is_not_replayed():
    dispatch = _PlanDispatch(fail={"open_app"}, delay=0)
    asyncio.run(_executor(dispatch).execute(
        [{"tool": "open_app", "args": {"app_name": "x"}}], max_retries=2))
    assert dispatch.order.count("open_app") == 1


# ── Mark XXXVII: a failed action stops the plan; results flow between steps ──
#
# The owner's plans carried on after a failed step: on_fail defaulted to
# 'retry', which for an action means "carry on", so "open Word, then type X"
# typed X into whatever had focus when Word never opened.

_TYPE = {"tool": "desktop_control", "args": {"action": "type_text", "text": "Dear Sam"}}


def test_a_failed_open_stops_the_typing_that_assumed_it():
    dispatch = _PlanDispatch(fail={"open_app"}, delay=0)
    result = asyncio.run(_executor(dispatch).execute([
        {"tool": "open_app", "args": {"app_name": "word"}}, _TYPE,
        {"tool": "resource_status", "args": {}}], objective="write the letter"))
    assert dispatch.order == ["open_app"], "nothing after the failed open may run"
    assert not result.ok and "aborted early: step 1 (open_app) failed" in result.text
    assert "2 not run" in result.text
    skipped = [e for e in result.evidence if e["skipped"]]
    assert [e["tool"] for e in skipped] == ["desktop_control", "resource_status"]
    assert all("open_app" in e["detail"] for e in skipped)


def test_a_failed_lookup_is_retried_and_the_plan_carries_on():
    dispatch = _PlanDispatch(fail={"web_search"}, delay=0)
    result = asyncio.run(_executor(dispatch).execute([
        _VALID, {"tool": "open_app", "args": {"app_name": "notepad"}}], max_retries=2))
    assert dispatch.order == ["web_search", "web_search", "open_app"]
    assert "1/2 succeeded" in result.text and "aborted" not in result.text


def test_continue_lets_a_failed_action_carry_on_when_asked():
    dispatch = _PlanDispatch(fail={"open_app"}, delay=0)
    asyncio.run(_executor(dispatch).execute([
        {"tool": "open_app", "args": {"app_name": "word"}, "on_fail": "continue"}, _TYPE]))
    assert dispatch.order == ["open_app", "desktop_control"]


def test_in_a_graph_a_failed_action_skips_its_dependants_not_independent_work():
    dispatch = _PlanDispatch(fail={"open_app"}, delay=0)
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "word", "tool": "open_app", "args": {"app_name": "word"}},
        {"id": "type", **_TYPE, "after": ["word"]},
        {"id": "aside", "tool": "resource_status", "args": {}}]))
    assert "desktop_control" not in dispatch.order and "resource_status" in dispatch.order
    assert "1 skipped on a failed dependency" in result.text


class _TextDispatch(_PlanDispatch):
    """Returns a scripted text per tool and keeps the arguments it was sent."""

    def __init__(self, texts, fail=()):
        super().__init__(delay=0.01, fail=fail)
        self.texts = texts
        self.sent: list[tuple[str, dict]] = []

    async def __call__(self, tool, args):
        self.sent.append((tool, dict(args)))
        outcome = await super().__call__(tool, args)
        return ToolResult(self.texts.get(tool, outcome.text), ok=outcome.ok)


def test_prev_passes_the_previous_steps_result_into_the_next():
    dispatch = _TextDispatch({"web_search": "1. Synthetic headline\nhttps://example.invalid"})
    result = asyncio.run(_executor(dispatch).execute([
        _VALID, {"tool": "desktop_control",
                 "args": {"action": "type_text", "text": "Notes:\n{prev}"}}]))
    assert result.ok, result.text
    assert dispatch.sent[1] == ("desktop_control", {
        "action": "type_text", "text": "Notes:\n1. Synthetic headline\nhttps://example.invalid"})


def test_a_named_step_output_reaches_a_later_step_and_never_shares_its_batch():
    dispatch = _TextDispatch({"web_search": "first finding",
                              "resource_status": "CPU 12%"})
    result = asyncio.run(_executor(dispatch).execute([
        {"id": "news", **_VALID},
        {"id": "load", "tool": "resource_status", "args": {}},
        {"tool": "web_search", "args": {"query": "about {step:news} and {step:2}"}}]))
    assert result.ok, result.text
    assert dispatch.sent[-1] == ("web_search", {"query": "about first finding and CPU 12%"})
    assert dispatch.order[-1] == "web_search" and dispatch.peak == 2, \
        "the two sources may batch together; their consumer waits for both"


def test_a_long_result_is_passed_on_bounded_and_marked():
    from orion_core.plan_executor import MAX_PASSED_CHARS
    long_text = "\n".join(f"line {n} " + "x" * 60 for n in range(400))
    dispatch = _TextDispatch({"web_search": long_text})
    result = asyncio.run(_executor(dispatch).execute([
        _VALID, {"tool": "desktop_control", "args": {"action": "type_text", "text": "{prev}"}}]))
    typed = dispatch.sent[1][1]["text"]
    assert len(typed) <= MAX_PASSED_CHARS + 4 and typed.endswith(" […]")
    assert typed.split("\n")[-1].startswith("line "), "cut at a line boundary"
    assert "passed on shortened" in result.text


def test_a_step_whose_input_failed_is_skipped_not_run_with_nothing():
    dispatch = _TextDispatch({}, fail={"web_search"})
    result = asyncio.run(_executor(dispatch).execute([
        _VALID, {"tool": "desktop_control", "args": {"action": "type_text", "text": "{prev}"}}],
        max_retries=1))
    assert [tool for tool, _args in dispatch.sent] == ["web_search"]
    assert "is not available" in result.text and not result.ok


@pytest.mark.parametrize("steps,problem", [
    ([{"tool": "web_search", "args": {"query": "{prev}"}}], "no step comes before it"),
    ([_VALID, {"tool": "web_search", "args": {"query": "{step:zz}"}}], "not a step in this plan"),
    ([{"id": "a", "tool": "web_search", "args": {"query": "{step:b}"}},
      {"id": "b", **_VALID}], "earlier step"),
])
def test_a_placeholder_must_name_an_earlier_step(steps, problem):
    dispatch = _PlanDispatch()
    result = asyncio.run(_executor(dispatch).execute(steps))
    assert "PLAN REJECTED" in result.text and problem in result.text
    assert dispatch.order == []


def test_a_step_consuming_output_counts_as_changing_something():
    """An action could be "{prev}"; its template class proves nothing."""
    from orion_core.job_store import SIDE_EFFECT
    from orion_core.plan_executor import ToolCatalogue, validate_plan
    parsed, problems = validate_plan(
        [_VALID, {"tool": "awareness", "args": {"action": "{prev}"}}],
        ToolCatalogue.from_dispatch(_PlanDispatch()))
    assert not problems
    assert parsed[1].side_effect == SIDE_EFFECT and not parsed[1].batchable


def test_the_plan_schemas_offer_ids_dependencies_and_a_failure_policy():
    from orion_core.dispatch_schema import TOOL_DECLARATIONS
    by_name = {d["name"]: d for d in TOOL_DECLARATIONS}
    for name in ("execute_plan", "autoplan"):
        item = by_name[name]["parameters"]["properties"]["steps"]["items"]["properties"]
        assert {"tool", "args_json", "id", "after", "on_fail"} <= set(item), name
        assert item["on_fail"]["enum"] == ["stop", "continue", "retry"]
    assert "{prev}" in by_name["execute_plan"]["description"]


@pytest.mark.parametrize("alias", ["clear_completed", "clear done", "remove-completed",
                                   "mark_all_done"])
def test_an_awareness_alias_that_writes_is_classified_as_a_write(alias):
    """The classifier reads the raw action; an alias the handler resolves to
    complete_tasks must not be batched, retried or backgrounded as a read."""
    assert classify("awareness", {"action": alias}) is ToolClass.SERIAL
    assert classify("awareness", {"action": "situation"}) is ToolClass.PARALLEL


@pytest.mark.parametrize("args,expected", [
    ({"action": "clear_completed"}, {"action": "complete_tasks", "all_listed": True}),
    ({"action": "clear_completed", "titles": ["Gym"]}, {"action": "complete_tasks",
                                                        "titles": ["Gym"]}),
    ({"action": "complete_tasks", "titles": ["Gym"]}, {"action": "complete_tasks",
                                                       "titles": ["Gym"]}),
])
def test_ive_done_all_those_tasks_reaches_complete_tasks(args, expected):
    from types import SimpleNamespace
    from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
    seen = []

    async def complete_owner_tasks(titles, *, all_listed=False):
        seen.append((titles, all_listed))
        return ToolResult("Marked done.")

    host = SimpleNamespace(cognition=object(), cognitive_loop=None,
                           complete_owner_tasks=complete_owner_tasks)
    result = asyncio.run(KnowledgeDispatchMixin.awareness_tool(host, dict(args)))
    assert result.ok and len(seen) == 1
    titles, all_listed = seen[0]
    assert all_listed is expected.get("all_listed", False)
    assert titles[0] == expected.get("titles")


@pytest.mark.parametrize("alias,canonical", [
    ("close_window", "close"), ("Close Window", "close"), ("exit", "close"),
    ("minimize", "minimise"), ("bring_to_front", "focus"), ("maximize_window", "maximise"),
    ("close", "close"), ("kill", "kill"),
])
def test_window_action_aliases_are_resolved_but_ambiguous_words_are_not(alias, canonical):
    from orion_core.action_aliases import normalise_action
    assert normalise_action("window_control", alias) == canonical


@pytest.mark.skipif(sys.platform != "win32", reason="native window enumeration")
def test_window_control_accepts_close_window():
    """'Unsupported window action: close_window' (owner's session). A title
    nothing has means the call only reads the desktop."""
    from orion_core.agents import DesktopAgent
    agent = DesktopAgent.__new__(DesktopAgent)
    result = agent.window_control("close_window", "zz synthetic no such window zz")
    assert "Unsupported" not in result.text and "No visible window matches" in result.text


def test_the_inline_fallback_also_stops_after_a_failed_action():
    from orion_core.dispatch_productivity import ProductivityDispatchMixin
    dispatch = _PlanDispatch(fail={"open_app"}, delay=0)
    host = ProductivityDispatchMixin.__new__(ProductivityDispatchMixin)
    host.dispatch = dispatch
    result = asyncio.run(host._execute_plan_inline(
        [{"tool": "open_app", "args": {"app_name": "word"}}, _TYPE], "letter"))
    assert dispatch.order == ["open_app"] and "STOPPED" in result.text and not result.ok


# ── plans are durable jobs ───────────────────────────────────────────────────

def test_a_plan_is_recorded_step_by_step():
    from orion_core import job_store
    dispatch = _PlanDispatch(delay=0)
    result = asyncio.run(_executor(dispatch).execute(
        [_VALID, {"tool": "open_app", "args": {"app_name": "x"}}], objective="demo"))
    job_id = result.text.rsplit("JOB: ", 1)[1].strip()
    record = job_store.default_store().get(job_id)
    assert record["kind"] == "plan" and record["state"] == "completed"
    assert record["fraction"] == 1.0
    assert [s["state"] for s in job_store.default_store().steps(job_id)] == \
        ["completed", "completed"]


class _GatedDispatch(_PlanDispatch):
    """Blocks a named tool until released, so a test can act mid-plan."""

    def __init__(self, gate_tool):
        super().__init__(delay=0)
        self.gate_tool = gate_tool
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, tool, args):
        self.order.append(tool)
        if tool == self.gate_tool:
            self.entered.set()
            await self.release.wait()
        return ToolResult(f"{tool} ok")


def _plan_job_id():
    from orion_core import jobs
    return next(c.id for c in jobs.CONTROLS.values() if c.kind == "plan")


def test_cancelling_a_plan_stops_it_at_the_next_step():
    async def scenario():
        dispatch = _GatedDispatch("web_search")
        manager = JobManager(_Bus(), dispatch)
        plan = asyncio.create_task(_executor(dispatch).execute([
            _VALID, {"tool": "open_app", "args": {"app_name": "x"}},
            {"tool": "close_app", "args": {"app_name": "x"}}]))
        await dispatch.entered.wait()
        job_id = _plan_job_id()
        assert manager.cancel(job_id).ok
        dispatch.release.set()
        return await plan, dispatch, job_id

    result, dispatch, job_id = asyncio.run(scenario())
    from orion_core import job_store
    assert dispatch.order == ["web_search"], "nothing after the cancel"
    assert "cancelled" in result.text and not result.ok
    assert job_store.default_store().get(job_id)["state"] == "cancelled"


def test_a_plan_pauses_between_steps_and_resumes():
    async def scenario():
        from orion_core import job_store
        dispatch = _GatedDispatch("web_search")
        manager = JobManager(_Bus(), dispatch)
        plan = asyncio.create_task(_executor(dispatch).execute(
            [_VALID, {"tool": "open_app", "args": {"app_name": "x"}}]))
        await dispatch.entered.wait()
        job_id = _plan_job_id()
        assert manager.pause(job_id).ok
        dispatch.release.set()
        await asyncio.sleep(0.15)
        paused_state = job_store.default_store().get(job_id)["state"]
        order_while_paused = list(dispatch.order)
        assert (await manager.resume(job_id)).ok
        return await plan, paused_state, order_while_paused, dispatch

    result, paused_state, order_while_paused, dispatch = asyncio.run(scenario())
    assert paused_state == "paused"
    assert order_while_paused == ["web_search"]
    assert result.ok and dispatch.order == ["web_search", "open_app"]


# ── crash, restart and resume without duplicating side effects ───────────────

def _crash_mid_plan(tmp_path, steps, gate_tool):
    """Run a plan until *gate_tool* is in flight, copy the database as a crash
    would leave it, then abandon the run. Returns the reopened copy."""
    import sqlite3

    from orion_core import job_store

    live = job_store.JobStore(tmp_path / "live.db")

    async def scenario():
        dispatch = _GatedDispatch(gate_tool)
        executor = PlanExecutor(_Bus(), dispatch, store=live)
        plan = asyncio.create_task(executor.execute(steps))
        await dispatch.entered.wait()
        crashed = sqlite3.connect(tmp_path / "crashed.db")
        live._db.backup(crashed)                  # the file as the crash left it
        crashed.close()
        plan.cancel()
        try:
            await plan
        except asyncio.CancelledError:
            pass
        return dispatch

    first_run = asyncio.run(scenario())
    live.close()
    return job_store.JobStore(tmp_path / "crashed.db"), first_run


_CRASH_PLAN = [
    {"id": "look", "tool": "web_search", "args": {"query": "flights"}},
    {"id": "act", "tool": "open_app", "args": {"app_name": "booking"}, "after": ["look"]},
    {"id": "then", "tool": "close_app", "args": {"app_name": "booking"}, "after": ["act"]},
    # Read-only, so it batches with nothing ahead of it and has FINISHED by
    # the time the crash lands during "act".
    {"id": "aside", "tool": "resource_status", "args": {}, "after": ["look"]},
    # Changes something (opens what it finds), independent of "act", and
    # still pending at the crash.
    {"id": "later", "tool": "find_files", "args": {"query": "x", "open": True},
     "after": ["look"]},
]


def test_after_a_crash_a_side_effect_in_flight_is_never_replayed_blindly(tmp_path):
    store, first = _crash_mid_plan(tmp_path, _CRASH_PLAN, "open_app")
    [record] = store.recovered
    assert record["state"] == "interrupted"
    assert first.order == ["web_search", "resource_status", "open_app"]

    dispatch = _PlanDispatch(delay=0)
    executor = PlanExecutor(_Bus(), dispatch, store=store)
    asked = asyncio.run(executor.resume_job(record))
    assert not asked.ok and "approval" in asked.text
    assert dispatch.order == [], "nothing may run until the user decides"
    assert store.get(record["id"])["state"] == "waiting_for_approval"

    approval = store.pending_approvals(record["id"])[0]
    store.decide_approval(approval["id"], True)
    store.transition(record["id"], "interrupted")   # as a later restart leaves it
    result = asyncio.run(executor.resume_job(store.get(record["id"])))
    assert result.ok, result.text
    assert dispatch.order.count("open_app") == 1, "re-run exactly once, by approval"
    assert dispatch.order == ["open_app", "close_app", "find_files"],         "completed steps (web_search, resource_status) are not repeated"
    store.close()


def test_declining_the_replay_skips_the_step_and_its_dependants(tmp_path):
    store, _ = _crash_mid_plan(tmp_path, _CRASH_PLAN, "open_app")
    [record] = store.recovered
    dispatch = _PlanDispatch(delay=0)
    executor = PlanExecutor(_Bus(), dispatch, store=store)
    asyncio.run(executor.resume_job(record))
    approval = store.pending_approvals(record["id"])[0]
    store.decide_approval(approval["id"], False)
    store.transition(record["id"], "interrupted")
    result = asyncio.run(executor.resume_job(store.get(record["id"])))
    assert dispatch.order == ["find_files"],         "the declined step and its dependant are skipped; independent work runs"
    assert "not re-run" in result.text
    store.close()


def test_a_read_only_step_in_flight_resumes_without_asking(tmp_path):
    plan = [{"id": "a", "tool": "resource_status", "args": {}},
            {"id": "b", "tool": "web_search", "args": {"query": "x"}, "after": ["a"]},
            {"id": "c", "tool": "open_app", "args": {"app_name": "x"}, "after": ["b"]}]
    store, _ = _crash_mid_plan(tmp_path, plan, "web_search")
    [record] = store.recovered
    dispatch = _PlanDispatch(delay=0)
    result = asyncio.run(PlanExecutor(_Bus(), dispatch, store=store).resume_job(record))
    assert result.ok, result.text
    assert dispatch.order == ["web_search", "open_app"]
    store.close()


@pytest.mark.parametrize("kind", ["plan", "workflow"])
@pytest.mark.parametrize("decision", ["approve", "decline", "pending", "expired"])
def test_manager_resumes_waiting_recovery_after_an_actual_approval_decision(tmp_path, monkeypatch,
                                                                          kind, decision):
    from types import SimpleNamespace

    from orion_core import jobs as jobs_module
    from orion_core.job_store import JobStore, State
    from orion_core.workflow_engine import WorkflowEngine

    monkeypatch.setattr(jobs_module, "_RESUMERS", dict(jobs_module._RESUMERS))
    now = [1000.0]
    store = JobStore(tmp_path / "approval_recovery.db", clock=lambda: now[0])
    dispatch = _PlanDispatch(delay=0)
    manager = JobManager(_Bus(), dispatch, store=store)
    steps = [{"tool": "open_app", "args": {"app_name": "synthetic"}}]
    engine = None
    if kind == "plan":
        PlanExecutor(_Bus(), dispatch, store=store)
        record = store.create(kind="plan", name="recovery", state=State.INTERRUPTED,
            args={"steps": steps, "objective": "synthetic recovery", "max_retries": 1},
            steps=[dict(steps[0], index=1, step_id="step_1")])
    else:
        engine = WorkflowEngine(_Bus(), SimpleNamespace(dispatch=dispatch,
            handler_table=lambda: {"open_app": None}), path=tmp_path / "workflows.json", store=store)
        assert engine.define("recovery", steps).ok
        record = store.create(kind="workflow", name="recovery", state=State.INTERRUPTED,
            args={"workflow": "recovery", "definition_digest": engine._definition_digest(engine.definitions["recovery"])},
            steps=[dict(steps[0], index=1)])
    store.set_step(record["id"], 1, state="interrupted")

    async def scenario():
        asked = await manager.resume(record["id"])
        assert not asked.ok and "approval" in asked.text.lower() and dispatch.order == []
        assert store.get(record["id"])["state"] == "waiting_for_approval"
        [approval] = store.pending_approvals(record["id"])
        if decision in {"approve", "decline", "expired"}:
            assert manager.approve(approval["id"], decision != "decline").ok
        if decision == "expired":
            now[0] += 601.0
        # No private state reset or simulated restart: the public manager
        # must invoke the waiting job's resumer and retain its approval gates.
        result = await manager.resume(record["id"])
        if engine is not None:
            await asyncio.wait_for(asyncio.gather(*(run["task"] for run in engine.runs.values())), 3)
        if decision == "approve":
            assert result.ok and dispatch.order == ["open_app"]
            assert store.get(record["id"])["state"] == "completed"
            assert store.approval(approval["id"])["state"] == "consumed"
        elif decision == "decline":
            assert dispatch.order == []
            assert store.get(record["id"])["state"] in {"cancelled", "failed"}
        else:
            assert not result.ok and "approval" in result.text.lower()
            assert dispatch.order == [] and store.get(record["id"])["state"] == "waiting_for_approval"

    try:
        asyncio.run(asyncio.wait_for(scenario(), 5))
    finally:
        store.close()


# ── jobs: queueing, pause, priority, restart ─────────────────────────────────

def test_a_queued_job_can_be_paused_reprioritised_and_resumed(monkeypatch):
    from orion_core import admission

    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(
        {admission.WorkClass.USER: admission.ClassPolicy(running=1, pending=4)}))

    async def scenario():
        manager, dispatch = _jobs(delay=0.15)
        first = manager.start("research", {"action": "findings", "n": 1}, label="first")
        second = manager.start("web_search", {"query": "x"}, label="second")
        assert second.state is JobState.QUEUED
        assert manager.pause(second.id).ok
        assert second.state is JobState.PAUSED
        assert manager.reprioritise(second.id, "background").ok
        assert (await manager.resume(second.id)).ok
        await asyncio.sleep(0.5)
        return first, second, dispatch

    first, second, dispatch = asyncio.run(scenario())
    assert first.state is JobState.COMPLETED and second.state is JobState.COMPLETED
    assert dispatch.order == ["research", "web_search"]
    assert second.cls == "background"


def test_cancel_lands_at_a_checkpoint_inside_threaded_work():
    """Cancelling a task cannot stop code already running on a thread; the
    checkpoint is where that code learns it should stop."""
    import threading

    from orion_core import jobs as jobs_mod

    progressed = []
    started = threading.Event()

    def long_scan():
        for index in range(200):
            started.set()
            jobs_mod.checkpoint_sync(index / 200)
            progressed.append(index)
            time.sleep(0.005)
        return ToolResult("scanned")

    async def dispatch(_tool, _args):
        return await asyncio.to_thread(long_scan)

    async def scenario():
        manager = JobManager(_Bus(), dispatch)
        job = manager.start("research", {"action": "findings"})
        await asyncio.to_thread(started.wait, 2.0)
        job._control.cancel_requested = True     # the flag cancel() sets
        await asyncio.sleep(0.2)
        return job

    asyncio.run(scenario())
    assert len(progressed) < 200, "the thread stopped at a checkpoint"


def test_an_interrupted_background_job_can_be_collected_and_resumed(tmp_path):
    from orion_core import job_store

    path = tmp_path / "jobs.db"
    before = job_store.JobStore(path)
    record = before.create(kind="tool", name="research", label="deep dive",
                           state="running",
                           args={"tool": "research", "args": {"action": "findings", "topic": "x"}},
                           steps=[{"tool": "research", "args": {"action": "findings", "topic": "x"}}])
    before.close()
    after = job_store.JobStore(path)

    async def scenario():
        manager, dispatch = _jobs(delay=0)
        manager._store = after
        collected = manager.collect(record["id"])
        resumed = await manager.resume(record["id"])
        await asyncio.sleep(0.05)
        return collected, resumed, dispatch

    collected, resumed, dispatch = asyncio.run(scenario())
    assert "interrupted" in collected.text and not collected.ok
    assert resumed.ok and dispatch.order == ["research"], \
        "read-only by construction, so re-running cannot duplicate a side effect"
    assert after.get(record["id"])["state"] == "cancelled", "superseded, not left dangling"
    after.close()


def test_the_snapshot_serves_a_task_view():
    async def scenario():
        manager, _ = _jobs(delay=0.1)
        manager.start("research", {"action": "findings"}, label="deep dive")
        view = manager.snapshot()
        for job in manager.active():
            job._task.cancel()
        await asyncio.sleep(0)
        return view

    view = asyncio.run(scenario())
    assert {"seq", "jobs", "counts", "approvals", "admission"} <= set(view)
    [job] = view["jobs"]
    assert {"id", "kind", "state", "cls", "fraction", "indeterminate", "note", "step",
            "total", "created_at", "updated_at", "live", "position",
            "pause_requested"} <= set(job)
    assert job["kind"] == "tool" and job["live"] is True
    assert view["admission"]["classes"]["user"]["running"] == 1


def test_an_orderly_shutdown_leaves_jobs_resumable_and_nothing_held(monkeypatch):
    """drain() used to cancel live jobs, which recorded them CANCELLED — so a
    research run cut off by closing ORION could never be resumed. suspend_all
    records them INTERRUPTED, then unwinds their tasks and frees every slot."""
    from orion_core import admission, job_store
    from orion_core import jobs as jobs_mod

    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(
        {admission.WorkClass.USER: admission.ClassPolicy(running=1, pending=4)}))

    async def scenario():
        manager, _ = _jobs(delay=5.0)
        running = manager.start("research", {"action": "findings", "topic": "a"}, label="running one")
        queued = manager.start("web_search", {"query": "b"}, label="queued one")
        await asyncio.sleep(0.02)
        suspended = await jobs_mod.suspend_all(timeout=2.0)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return running, queued, suspended, pending

    running, queued, suspended, pending = asyncio.run(scenario())
    store = job_store.default_store()
    assert suspended == 2 and pending == [], "no task left destroyed-but-pending"
    assert store.get(running.id)["state"] == "interrupted"
    assert store.get(queued.id)["state"] == "interrupted"
    assert store.steps(running.id)[0]["state"] == "interrupted"
    assert admission.ADMISSION.in_use() == 0 and jobs_mod.CONTROLS == {}


def test_the_job_tool_can_pause_resume_and_reprioritise_but_never_approve():
    """Voice can change WHEN work runs. It cannot approve a step: a tool call
    can come from the model or from text it read, and an approval must come
    from the user."""
    import asyncio

    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    calls = []

    class _Jobs:
        def pause(self, job_id):
            calls.append(("pause", job_id))
            return ToolResult("paused")

        async def resume(self, job_id):
            calls.append(("resume", job_id))
            return ToolResult("resumed")

        def reprioritise(self, job_id, priority):
            calls.append(("priority", job_id, priority))
            return ToolResult("moved")

        def approve(self, *a, **k):          # pragma: no cover - must never be reached
            calls.append(("approve",))
            return ToolResult("approved")

    holder = type("D", (), {"jobs": _Jobs(), "job_tool": ProductivityDispatchMixin.job_tool})()
    run = lambda a: asyncio.run(holder.job_tool(a))  # noqa: E731
    assert run({"action": "pause", "job_id": "sweep"}).ok
    assert run({"action": "resume", "job_id": "sweep"}).ok
    assert run({"action": "priority", "job_id": "sweep", "priority": "background"}).ok
    refused = run({"action": "approve", "job_id": "sweep"})
    assert not refused.ok and "Unsupported" in refused.text
    assert calls == [("pause", "sweep"), ("resume", "sweep"),
                     ("priority", "sweep", "background")]


# ── cancelled tool calls (the mid-session "entity not found") ────────────────
#
# The installed build's session journal showed "1008 Requested entity was not
# found" 150–170 s into sessions. The tool call was awaited inside the receive
# loop, so the server's tool_call_cancellation (sent when the user interrupts)
# sat unread and ORION answered a call the server had already forgotten.


class _Session:
    def __init__(self):
        self.sent: list = []

    async def send_tool_response(self, function_responses):
        self.sent.append(list(function_responses))


def _live_worker_with_session(dispatcher):
    worker = _worker(dispatcher)
    worker.session = _Session()
    worker._send_closed = False
    worker._function_response = lambda call_id, name, payload: (call_id, name, payload)
    return worker


def test_a_call_cancelled_before_it_starts_is_neither_run_nor_answered():
    dispatcher = _RecordingDispatcher(delay=0.01)
    worker = _live_worker_with_session(dispatcher)
    worker._note_cancelled_tool_calls(["id-open_app"])
    asyncio.run(worker._handle_tool_call(_ToolCall(_Call("web_search"), _Call("open_app"))))
    assert dispatcher.order == ["web_search"]
    assert [[name for _i, name, _p in batch] for batch in worker.session.sent] == [["web_search"]]


def test_a_cancellation_that_arrives_while_the_tool_runs_suppresses_its_answer():
    dispatcher = _RecordingDispatcher(delay=0.15)
    worker = _live_worker_with_session(dispatcher)

    async def flow():
        task = asyncio.create_task(worker._handle_tool_call(_ToolCall(_Call("web_search"))))
        await asyncio.sleep(0.03)
        worker._note_cancelled_tool_calls(["id-web_search"])
        await task

    asyncio.run(flow())
    assert dispatcher.order == ["web_search"]          # it ran — side effects are not undone
    assert worker.session.sent == []                   # but the answer is never sent


def test_an_answer_is_never_sent_to_a_session_that_reconnected_meanwhile():
    dispatcher = _RecordingDispatcher(delay=0.1)
    worker = _live_worker_with_session(dispatcher)
    first = worker.session

    async def flow():
        task = asyncio.create_task(worker._handle_tool_call(_ToolCall(_Call("web_search"))))
        await asyncio.sleep(0.03)
        worker.session = _Session()                    # the channel reconnected
        await task
        return worker.session

    second = asyncio.run(flow())
    assert first.sent == [] and second.sent == []


def test_the_receive_loop_keeps_reading_while_a_tool_runs():
    """The cancellation must be seen while the tool is still running — which
    is only possible if the loop is not blocked inside the tool call."""
    dispatcher = _RecordingDispatcher(delay=0.25)
    worker = _live_worker_with_session(dispatcher)

    class _Msg:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    class _Cancellation:
        ids = ["id-web_search"]

    session = worker.session
    calls = {"n": 0}

    async def receive():
        calls["n"] += 1
        if calls["n"] > 1:
            worker.stop_event.set()
            return
        yield _Msg(tool_call=_ToolCall(_Call("web_search")))
        yield _Msg(tool_call_cancellation=_Cancellation())

    session.receive = receive

    async def flow():
        started = time.monotonic()
        await worker._receive_realtime()
        read_by = time.monotonic() - started
        await asyncio.gather(*list(getattr(worker, "_tool_call_tasks", set())))
        return read_by

    read_by = asyncio.run(flow())
    assert read_by < 0.2, "the loop waited for the tool before reading on"
    # Read before the task even started, the cancelled call never ran at all.
    assert worker._tool_call_cancelled(_Call("web_search"))
    assert dispatcher.order == []
    assert session.sent == []


# ── camera actions (audit 6 Oct 2026, A07) ───────────────────────────────────

def test_every_camera_tool_alias_is_serialised():
    """A drift guard: every action the vision dispatcher routes to the camera
    tools must be a write in the concurrency table, spelt as the model may
    send it."""
    from orion_core.concurrency import ToolClass, classify
    from orion_core.dispatch_vision import _CAMERA_TOOL_ACTIONS
    for action in _CAMERA_TOOL_ACTIONS:
        for spelling in (action, action.replace("_", " "), action.upper()):
            assert classify("vision_analyse", {"action": spelling}) is ToolClass.SERIAL, spelling


def test_two_camera_calls_never_share_a_batch():
    from orion_core.concurrency import plan_batches
    names = ["web_search", "vision_analyse", "vision_analyse", "resource_status"]
    args = [{"query": "x"}, {"action": "take photo"}, {"action": "record_video"}, None]
    batches = plan_batches(names, limit=4, args=args)
    assert [1] in batches and [2] in batches
    # A look at the screen still runs alongside other reads.
    assert plan_batches(["web_search", "vision_analyse"], limit=4,
                        args=[{"query": "x"}, {"action": "screen"}]) == [[0, 1]]


@pytest.mark.parametrize('tool,action', [
    ('process_file', ''), ('process_file', 'review'),
    ('vision_analyse', 'forget_camera'), ('vision_analyse', 'watch_camera_past'),
    ('perception', 'forget_memory'), ('perception', 'history_on'), ('perception', 'history_off'),
])
def test_memory_mutations_are_never_batched_as_read_only(tool, action):
    assert classify(tool, {'action': action}) is ToolClass.SERIAL


@pytest.mark.parametrize('tool,action', [
    ('process_file', 'recall'), ('process_file', 'history'),
    ('vision_analyse', 'recall_camera'), ('vision_analyse', 'compare_camera'),
    ('perception', 'recall'), ('perception', 'compare'),
])
def test_camera_and_file_recall_remain_parallel_observations(tool, action):
    assert classify(tool, {'action': action}) is ToolClass.PARALLEL


# ── autonomous specialist runs (agents.SpecialistRun) ───────────────────────
#
# The three playbooks driven end to end with a scripted text model and fake
# instruments: what each run reads first, what reaches the dispatcher, what is
# refused or left for the owner, and the honest outcome it reports.


class _ScriptedModel:
    """A text model answering from a script, one reply per model pass."""

    def __init__(self, *replies, available=True):
        self.replies = list(replies)
        self.prompts = []
        self.available = available

    def has_text_fallback(self):
        return self.available

    async def generate_text(self, prompt, **_kwargs):
        from types import SimpleNamespace
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else "SUMMARY: nothing more to add."
        return SimpleNamespace(name="scripted"), reply


class _FakeDispatch:
    """The dispatcher as a specialist reaches it: a canned answer per tool."""

    def __init__(self, answers=None):
        self.answers = dict(answers or {})
        self.calls = []

    async def __call__(self, tool, args):
        self.calls.append((tool, dict(args)))
        answer = self.answers.get(tool, f"{tool} observed")
        return answer if isinstance(answer, ToolResult) else ToolResult(str(answer))


class _GapLedger:
    def __init__(self):
        self.addressed = []

    def recurring(self, minimum=1, limit=12):
        return [{"key": "biggest-files", "count": 3, "kind": "refused",
                 "label": "find the biggest files",
                 "examples": ["can you find the biggest files on the PC"]},
                {"key": "legal-advice", "count": 2, "kind": "refused", "label": "legal advice"}]

    def mark_addressed(self, key, addressed):
        self.addressed.append((key, addressed))
        return {"key": key}


def test_reliability_run_reads_its_openings_then_concludes_from_the_evidence():
    from orion_core.agents import PLAYBOOKS, AgentToolbelt, ReliabilityAgent, SpecialistRun

    model = _ScriptedModel(
        "TOOL: recent_changes\nARGS: {}",
        "SUMMARY: The 23:41 stall followed the audio update.\nTimeline: see readings.")
    readings = []

    async def incident(args):
        readings.append(dict(args))
        return ToolResult("23:41:07 audio callback stalled for 4.2 s")

    tray = AgentToolbelt(extras={"incident_lookup": ("Incident records. ARGS: {}", incident)},
                         budget=6)
    dispatch = _FakeDispatch({"diagnostics": "No crash reports on record.",
                              "self_changes": "audio.py changed at this start-up"})
    request = "Why did ORION go quiet at 23:41?"
    run = SpecialistRun(ReliabilityAgent(model, _Bus()), PLAYBOOKS["reliability"], request,
                        tray=tray, dispatch=dispatch)
    report = asyncio.run(run.execute())

    assert readings == [{"question": request, "when": "23:41"}]
    assert dispatch.calls == [("diagnostics", {"action": "crashes"}),
                              ("self_changes", {"action": "latest"})]
    assert [s.tool for s in report.steps] == ["incident_lookup", "crash_reports",
                                              "recent_changes"]
    assert report.outcome == "done" and report.model_calls == 2
    assert report.summary == "The 23:41 stall followed the audio update."
    # The opening readings ground the first model pass at no model cost.
    assert "audio callback stalled" in model.prompts[0]
    assert "audio.py changed" in model.prompts[1]


def test_files_run_acts_only_through_the_gates_and_never_approves_for_the_owner():
    from orion_core.agents import PLAYBOOKS, FilesOperatorAgent, SpecialistRun

    model = _ScriptedModel(
        'TOOL: search_files\nARGS: {"sort": "largest", "open": true, "limit": 5}',
        'TOOL: file_controller\nARGS: {"action": "delete", "path": "C:/Downloads/setup.exe", '
        '"confirm": true}\n'
        'TOOL: file_controller\nARGS: {"action": "move", "path": "C:/Downloads/a.iso", "to": "D:/"}',
        "SUMMARY: setup.exe waits for your approval; moving a.iso is yours to do.")
    dispatch = _FakeDispatch({
        "find_files": "C:/Downloads/setup.exe 4.1 GB (installer, already installed)",
        "file_controller": ToolResult(
            "Delete C:/Downloads/setup.exe? Approve the on-screen prompt.", ok=False),
    })
    run = SpecialistRun(FilesOperatorAgent(model, _Bus()), PLAYBOOKS["files"],
                        "Find what is filling my disk and clear the junk", dispatch=dispatch)
    report = asyncio.run(run.execute())

    # A pinned argument always wins: the search can never open what it finds.
    assert dispatch.calls[0] == ("find_files", {"sort": "largest", "limit": 5, "open": False})
    # The approval flag never reaches the dispatcher, and the move never runs.
    assert dispatch.calls[1:] == [("file_controller", {"action": "delete",
                                                       "path": "C:/Downloads/setup.exe"})]
    assert [(s.tool, s.kind, s.state()) for s in report.steps] == [
        ("search_files", "observe", "OK"),
        ("file_controller", "act", "WAITING ON THE OWNER"),
        ("file_controller", "proposal", "NOT RUN")]
    assert "removed confirm" in report.steps[1].detail
    assert report.outcome == "partial" and "wait for you" in report.why


def test_a_files_run_is_capped_per_run_and_only_proposes_when_detached():
    from orion_core.agents import PLAYBOOKS, FilesOperatorAgent, SpecialistRun

    deletes = "\n".join(
        f'TOOL: file_controller\nARGS: {{"action": "delete", "path": "C:/junk/{n}.tmp"}}'
        for n in range(6))
    dispatch = _FakeDispatch({"file_controller": "Moved to the Recycle Bin."})
    run = SpecialistRun(FilesOperatorAgent(_ScriptedModel(deletes, "SUMMARY: done."), _Bus()),
                        PLAYBOOKS["files"], "clear the junk folder", dispatch=dispatch)
    report = asyncio.run(run.execute())
    assert len(dispatch.calls) == PLAYBOOKS["files"].acts["file_controller"].per_run == 5
    assert report.steps[-1].kind == "proposal" and "limit" in report.steps[-1].detail

    detached = _FakeDispatch()
    run = SpecialistRun(
        FilesOperatorAgent(_ScriptedModel(
            'TOOL: file_controller\nARGS: {"action": "mkdir", "path": "C:/Users/x/CVs"}',
            "SUMMARY: proposed only."), _Bus()),
        PLAYBOOKS["files"], "gather my CVs", dispatch=detached, allow_acts=False)
    report = asyncio.run(run.execute())
    assert detached.calls == []
    assert report.steps[0].kind == "proposal" and "background" in report.steps[0].detail


def test_triage_retires_only_what_a_search_in_this_run_returned():
    from orion_core.agents import PLAYBOOKS, ImprovementTriageAgent, SpecialistRun

    model = _ScriptedModel(
        'TOOL: retire_gap\nARGS: {"id": "G1", "verdict": "existing", "tool": "find_files"}\n'
        'TOOL: tool_search\nARGS: {"need": "find the biggest files"}',
        'TOOL: retire_gap\nARGS: {"id": "G1", "verdict": "existing", "tool": "find_files"}\n'
        'TOOL: retire_gap\nARGS: {"id": "G2", "verdict": "policy", "reason": "legal"}\n'
        'TOOL: retire_gap\nARGS: {"id": "G2", "verdict": "policy", '
        '"reason": "declined: legal advice is deliberately out of scope"}',
        "SUMMARY: G1 is find_files(sort='largest'); G2 is a deliberate refusal.")
    dispatch = _FakeDispatch({"find_tool": "find_files - ranks the PC by size, sort='largest'"})
    ledger = _GapLedger()
    run = SpecialistRun(ImprovementTriageAgent(model, _Bus()), PLAYBOOKS["triage"],
                        "what can't you do yet?", dispatch=dispatch, ledger=ledger,
                        known_tools=lambda: {"find_files", "capabilities"})
    report = asyncio.run(run.execute())

    assert [(s.tool, s.ok) for s in report.steps] == [
        ("unmet_requests", True), ("retire_gap", False), ("tool_search", True),
        ("retire_gap", True), ("retire_gap", False), ("retire_gap", True)]
    assert "Search first" in report.steps[1].detail          # belief is not evidence
    assert "needs the reason" in report.steps[4].detail
    assert ledger.addressed == [("biggest-files", "existing tool find_files"),
                                ("legal-advice", "declined on purpose (policy)")]
    assert dispatch.calls == [("find_tool", {"need": "find the biggest files"})]


@pytest.mark.parametrize("search_result, accepted", [
    ("find_files_by_hash - finds equal file content", False),
    ("my_find_files - another search tool", False),
    ("find_files_extra - extended search", False),
    ("find_files(sort='largest') ranks the whole PC", True),
    ("`find_files`: ranks files by size", True),
])
def test_triage_retirement_needs_the_complete_tool_name(search_result, accepted):
    from orion_core.agents import PLAYBOOKS, ImprovementTriageAgent, SpecialistRun

    model = _ScriptedModel(
        'TOOL: tool_search\nARGS: {"need": "find the biggest files"}',
        'TOOL: retire_gap\nARGS: {"id": "G1", "verdict": "existing", "tool": "find_files"}',
        "SUMMARY: checked the available file tool.")
    ledger = _GapLedger()
    run = SpecialistRun(
        ImprovementTriageAgent(model, _Bus()), PLAYBOOKS["triage"],
        "what can't you do yet?", dispatch=_FakeDispatch({"find_tool": search_result}),
        ledger=ledger, known_tools=lambda: {"find_files", "find_files_by_hash", "my_find_files"})

    report = asyncio.run(run.execute())
    retired = report.steps[-1]
    assert retired.tool == "retire_gap" and retired.ok is accepted
    assert ledger.addressed == ([("biggest-files", "existing tool find_files")] if accepted else [])
    if not accepted:
        assert "Search first" in retired.detail


def test_a_run_that_never_concludes_stops_at_its_budget():
    from orion_core.agents import PLAYBOOKS, ReliabilityAgent, SpecialistRun

    playbook = PLAYBOOKS["reliability"]
    model = _ScriptedModel(*["TOOL: crash_reports\nARGS: {}"] * 10)
    run = SpecialistRun(ReliabilityAgent(model, _Bus()), playbook, "why so slow?",
                        dispatch=_FakeDispatch({"diagnostics": "one report"}))
    report = asyncio.run(run.execute())
    assert report.model_calls == playbook.max_model_calls
    assert len(report.steps) <= playbook.max_steps
    assert report.outcome == "partial" and "without concluding" in report.why


def test_with_no_text_model_a_run_returns_its_brief_and_readings():
    from orion_core.agents import PLAYBOOKS, ReliabilityAgent, SpecialistRun

    model = _ScriptedModel(available=False)
    run = SpecialistRun(ReliabilityAgent(model, _Bus()), PLAYBOOKS["reliability"],
                        "why did you crash?",
                        dispatch=_FakeDispatch({"diagnostics": "access violation, render thread"}))
    report = asyncio.run(run.execute())
    assert model.prompts == []
    assert report.briefed and report.outcome == "partial"
    assert [s.tool for s in report.steps] == ["crash_reports"]
    assert "Specialist brief" in report.answer


def test_attached_dispatcher_and_sized_tray_reach_every_playbook():
    from orion_core.agents import PLAYBOOKS, AgentManager, AgentToolbelt

    manager = AgentManager(_ScriptedModel(), _Bus())

    class _Dispatcher:
        TOOL_DECLARATIONS = [{"name": "capabilities"}]
        _tool_handlers = {"forged_tool": object()}

        def handler_table(self):
            return {"find_files": None, "file_controller": None}

        async def dispatch(self, tool, args):
            return ToolResult("ok")

    manager.attach_dispatch(_Dispatcher().dispatch)
    assert manager._known_tools() == {"find_files", "file_controller", "capabilities",
                                      "forged_tool"}
    sizes = []
    manager.attach_specialist_tray(lambda n: sizes.append(n) or AgentToolbelt(budget=n))
    assert manager._run_tray(PLAYBOOKS["triage"]).budget == PLAYBOOKS["triage"].max_steps
    assert sizes == [PLAYBOOKS["triage"].max_steps]


def test_agent_dispatch_handler_passes_background_and_jobs():
    from types import SimpleNamespace
    from orion_core import jobs as jobs_module
    from orion_core.dispatch_productivity import ProductivityDispatchMixin

    captured = []

    class _Manager:
        async def dispatch(self, **kwargs):
            captured.append(kwargs)
            return ToolResult("ok")

    class _Dispatch(ProductivityDispatchMixin):
        agent_manager = _Manager()

    handler = _Dispatch()
    handler.jobs = jobs = object()
    asyncio.run(handler.agent_dispatch({"agent": "reliability", "request": "why?",
                                        "background": True}))
    asyncio.run(handler.agent_dispatch({"agent": "reliability", "request": "why?",
                                        "background": "false"}))
    token = jobs_module._CURRENT_CONTROL.set(SimpleNamespace(kind="tool"))
    try:
        # The detached run itself never detaches again.
        asyncio.run(handler.agent_dispatch({"agent": "reliability", "request": "why?",
                                            "background": True}))
    finally:
        jobs_module._CURRENT_CONTROL.reset(token)
    assert [c["background"] for c in captured] == [True, False, False]
    assert all(c["jobs"] is jobs for c in captured)


def test_a_background_specialist_run_becomes_a_job():
    from orion_core.agents import AgentManager

    manager = AgentManager(_ScriptedModel(), _Bus())
    calls = []

    async def dispatch_chain(tool, args):
        calls.append((tool, dict(args)))
        return ToolResult("[Reliability Investigator] DONE — the audio update")

    async def scenario():
        jobs = JobManager(_Bus(), dispatch_chain)
        result = await manager.dispatch("why did you go quiet at 23:41?", "reliability",
                                        background=True, jobs=jobs)
        job = jobs.jobs()[0]
        await job._task
        return result, job

    result, job = asyncio.run(scenario())
    assert "in the background" in result.text and job.id in result.text
    assert calls == [("agent_dispatch", {"agent": "reliability", "context": "",
                                         "request": "why did you go quiet at 23:41?"})]
    assert job.state is JobState.COMPLETED


def test_the_files_specialist_never_detaches_even_when_asked():
    from orion_core.agents import AgentManager

    manager = AgentManager(_ScriptedModel("SUMMARY: nothing needed clearing."), _Bus())

    async def scenario():
        jobs = JobManager(_Bus(), lambda *_a: None)
        result = await manager.dispatch("tidy my downloads", "files", background=True, jobs=jobs)
        return result, jobs.jobs()

    result, detached = asyncio.run(scenario())
    assert detached == []
    assert "worked here in the conversation" in result.text


class _TriggerLoop:
    """Records what the crash trigger schedules instead of waiting for it."""

    def __init__(self):
        self.later = []

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)

    def call_later(self, delay, fn, *args):
        self.later.append((delay, fn, args))

    def fire(self):
        delay, fn, args = self.later.pop(0)
        fn(*args)
        return delay


class _TriggerJobs:
    def __init__(self, rows=()):
        from types import SimpleNamespace
        self.started = []
        self.store = SimpleNamespace(list=lambda **_kw: list(rows))

    def start(self, tool, args, label="", priority="user"):
        from types import SimpleNamespace
        self.started.append((tool, args, label, priority))
        return SimpleNamespace(id="job-crash")


def test_crash_trigger_waits_for_start_up_and_runs_one_read_only_check():
    from orion_core.agents import PLAYBOOKS, SpecialistTriggers

    booted = {"done": False}
    jobs = _TriggerJobs()
    triggers = SpecialistTriggers(_Bus(), None, jobs, delay_s=45, enabled=True,
                                  ready=lambda: booted["done"])
    triggers._loop = loop = _TriggerLoop()
    triggers._on_event("crash", {"kind": "qt-gpu", "summary": "render process gone"})
    triggers._on_event("crash", {"kind": "python", "summary": "a second report"})
    assert len(loop.later) == 1                       # once per session
    assert loop.fire() == 45 and jobs.started == []   # start-up not finished: waits again
    booted["done"] = True
    loop.fire()
    assert len(jobs.started) == 1
    tool, args, label, priority = jobs.started[0]
    assert (tool, args["agent"], label, priority) == (
        "agent_dispatch", "reliability", SpecialistTriggers.CRASH_LABEL, "background")
    assert "render process gone" in args["request"]
    assert not PLAYBOOKS["reliability"].acting
    assert JobManager(_Bus(), lambda *_a: None).can_background(tool, args)[0]


def test_crash_trigger_gives_up_when_start_up_never_finishes():
    from orion_core.agents import SpecialistTriggers

    bus, jobs = _Bus(), _TriggerJobs()
    triggers = SpecialistTriggers(bus, None, jobs, delay_s=45, enabled=True, ready=lambda: False)
    triggers._loop = loop = _TriggerLoop()
    triggers._on_event("crash", {"kind": "python", "summary": "boom"})
    for _ in range(SpecialistTriggers.MAX_READY_WAITS + 1):
        loop.fire()
    assert loop.later == [] and jobs.started == []
    assert "start-up had not finished" in bus.lines()


@pytest.mark.parametrize("hours_ago,runs", [(1.0, False), (7.0, True)])
def test_crash_trigger_runs_at_most_once_per_interval_across_restarts(hours_ago, runs):
    from orion_core.agents import SpecialistTriggers

    now = 1_800_000_000.0
    rows = [{"name": "agent_dispatch", "label": SpecialistTriggers.CRASH_LABEL,
             "created_at": now - hours_ago * 3600}]
    bus, jobs = _Bus(), _TriggerJobs(rows)
    triggers = SpecialistTriggers(bus, None, jobs, delay_s=45, enabled=True,
                                  clock=lambda: now)
    triggers._loop = loop = _TriggerLoop()
    triggers._on_event("crash", {"kind": "python", "summary": "boom"})
    loop.fire()
    assert bool(jobs.started) is runs
    assert ("crash check skipped" in bus.lines()) is not runs


def test_app_wires_the_specialist_dispatcher_tray_and_crash_trigger():
    source = (Path(__file__).resolve().parents[1] / "orion_core" / "app.py").read_text(
        encoding="utf-8")
    for needle in ("agents.attach_dispatch(dispatcher.dispatch)",
                   "agents.attach_specialist_tray(", "SpecialistTriggers(",
                   "specialist_triggers.install()"):
        assert needle in source, needle
    # The start-up gate names a boot phase that is really marked.
    assert 'phase.name == "services + background tasks"' in source
    assert 'budget.mark("services + background tasks")' in source


def test_a_specialist_may_read_the_security_checks_but_not_change_them():
    """Mark XXXX.4: the Cybersecurity Agent was refused security_watch ("not
    read-only") and answered from generic advice while ORION said an audit
    was running."""
    for action in ("audit", "status", "network", "phone_activity"):
        assert classify("security_watch", {"action": action}) is ToolClass.PARALLEL, action
    for action in ("disable", "phone_off", "private_log", ""):
        assert classify("security_watch", {"action": action}) is ToolClass.SERIAL, action
    assert classify("antivirus", {"action": "status"}) is ToolClass.PARALLEL
    assert classify("antivirus", {"action": "scan"}) is ToolClass.SERIAL
