"""
Shutdown robustness (Mark XXII):
  · a watchdog force-exits so ORION always FULLY shuts down, even if a
    background thread (audio/GL/torch) refuses to die;
  · the goodbye gets a tail so it is never clipped mid-sentence.

app.py imports QtWebEngine at module scope (crashes the offscreen Qt platform
for later tests), so its wiring is checked by reading the source, not importing.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "orion_core"


def test_shutdown_watchdog_exists_and_is_armed():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert "def _arm_shutdown_watchdog" in src
    assert "_terminate_process(0)" in src             # the last-resort kill
    assert "TerminateProcess" in src                  # no DLL unload code (Mark 38)
    assert "daemon=True" in src
    # armed the moment shutdown is requested
    assert "_arm_shutdown_watchdog()" in src
    idx_def = src.index("def request_shutdown")
    # the arm call appears inside/after the handler, not only in the definition
    assert "_arm_shutdown_watchdog()" in src[idx_def:]


def test_await_farewell_has_a_tail():
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    # the farewell no longer returns the instant the pipeline reports idle
    assert "asyncio.sleep(1.3)" in src


def test_live_farewell_waits_longer_and_tails():
    """The flat 2.4s sleep cut off the longer sign-offs.

    Asserted against the wait's real defaults rather than a literal in the
    source: the ceiling moved into _await_speech when the farewell started
    coming from the MODEL, whose words arrive a second or three after the turn
    is sent. What matters is that the ceiling is generous and the tail exists,
    not which function holds the number.
    """
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    parameters = inspect.signature(GenAILiveWorker._await_speech).parameters
    assert parameters["end_limit"].default >= 10.0, "too short for a real goodbye"
    assert parameters["tail"].default >= 1.0, "no tail — the last word is clipped"
    # And he is waited for at all, from the shutdown path.
    assert "_await_speech" in inspect.getsource(
        GenAILiveWorker._shutdown_after_farewell)


def test_the_wait_covers_him_starting_as_well_as_finishing():
    """A live turn is silent for a second or three after it is sent. Polling
    only for "not busy" finds that silence, decides he has finished, and fires
    the shutdown before he has said a word."""
    import inspect

    from orion_core.live_worker import GenAILiveWorker

    source = inspect.getsource(GenAILiveWorker._await_speech)
    assert "start_limit" in source
    assert source.index("start_limit") < source.index("end_limit")


# ── the shutdown report (orion_core/shutdown_trace.py) ──────────────────────

def _trace_with_clock():
    from orion_core.shutdown_trace import ShutdownTrace
    now = [100.0]
    return ShutdownTrace(clock=lambda: now[0]), now


def test_a_forced_report_names_the_step_it_interrupted():
    """The whole point: a quit that ran past the watchdog used to leave no
    trace of which of ~58 steps was holding it."""
    trace, now = _trace_with_clock()
    trace.begin()
    with trace.phase("farewell"):
        now[0] += 3.0
    with trace.phase("worker"):
        with trace.phase("live session close"):
            now[0] += 22.0
            report = trace.report(forced=True)
    assert report.startswith("FORCED by the shutdown watchdog after 25.0 s.")
    assert "Still inside: worker > live session close" in report
    assert trace.current is None                  # the stack unwound


def test_a_clean_report_lists_the_slowest_steps_first():
    trace, now = _trace_with_clock()
    trace.begin()
    for label, seconds in (("memory", 0.1), ("farewell", 3.8), ("worker", 0.2)):
        with trace.phase(label):
            now[0] += seconds
    lines = trace.report().splitlines()
    assert lines[0] == "Clean shutdown after 4.1 s."
    assert [line.split("s  ", 1)[1] for line in lines[2:5]] == ["farewell", "worker", "memory"]


def test_threads_the_process_must_still_wait_for_are_named():
    import threading

    from orion_core.shutdown_trace import lingering_threads
    release = threading.Event()
    busy = threading.Thread(target=release.wait, name="stuck-job")
    idle = threading.Thread(target=release.wait, name="background-daemon", daemon=True)
    busy.start()
    idle.start()
    try:
        found = lingering_threads()
    finally:
        release.set()
        busy.join()
        idle.join()
    assert any(entry.startswith("stuck-job  at threading.py") for entry in found), found
    assert not any(entry.startswith("background-daemon") for entry in found)


def test_writing_the_report_never_raises(tmp_path):
    trace, _now = _trace_with_clock()
    trace.begin()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    assert trace.write(blocker / "last_shutdown.txt") is False


def test_he_leaves_the_screen_once_he_has_said_goodbye():
    """Nothing ever closed the windows: the last frame stayed up, frozen,
    until the interpreter finished exiting."""
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    farewell = src.index("await _await_farewell(worker.speech)")
    off_screen = src.index('_safe("off screen"')
    first_service_stop = src.index('_safe("proactive", proactive.stop)')
    assert farewell < off_screen < first_service_stop


def test_live_goodbye_generation_is_bounded_and_local_playback_finishes(monkeypatch):
    import asyncio
    import time
    from types import SimpleNamespace
    from orion_core.live_worker import GenAILiveWorker

    async def scenario():
        worker = object.__new__(GenAILiveWorker)
        worker.session = object()
        worker.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: None))
        worker.stop_event = asyncio.Event()
        worker._drop_live_output = False
        spoken = []
        cancelled = []
        async def blocked_send(text, **kwargs):
            assert kwargs == {"allow_fallback": False}
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        worker._send_text_turn = blocked_send
        def say(text):
            spoken.append(text)
            worker.speech = SimpleNamespace(is_busy=lambda: time.monotonic() < finish[0])
            finish[0] = time.monotonic() + 0.3
        finish = [0.0]
        worker._say = say
        began = time.monotonic()
        await worker.speak_farewell("Goodbye.", generation_limit=0.15)
        generation = time.monotonic() - began
        assert 0.12 <= generation < 0.5
        assert spoken == ["Goodbye."] and cancelled
        assert worker._drop_live_output, "late Live audio must not overlap the local goodbye"
        await worker._await_speech(start_limit=0.1, end_limit=2, tail=0.1)
        assert time.monotonic() >= finish[0] + 0.09
    asyncio.run(scenario())


def test_fast_live_goodbye_keeps_its_voice_and_records_generation(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from orion_core.live_worker import GenAILiveWorker
    from orion_core import shutdown_trace

    trace, _now = _trace_with_clock()
    monkeypatch.setattr(shutdown_trace, "TRACE", trace)
    async def scenario():
        worker = object.__new__(GenAILiveWorker)
        worker.session = object()
        worker._drop_live_output = False
        worker._say = lambda text: (_ for _ in ()).throw(AssertionError("local fallback"))
        async def send(text, **kwargs):
            await asyncio.sleep(0.02)
            worker._farewell_audio_started.set()
        worker._send_text_turn = send
        worker.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: None))
        await worker.speak_farewell("Goodbye.", generation_limit=0.15)
        assert not worker._drop_live_output
        assert worker._farewell_audio_started is None
    asyncio.run(scenario())
    assert any(label == "farewell generation" for label, _seconds in trace.phases)


def test_disconnected_farewell_does_not_start_a_second_provider_turn():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from orion_core.live_worker import GenAILiveWorker

    worker = object.__new__(GenAILiveWorker)
    worker.session = object()
    worker._transport_ready = lambda: False
    worker._submit_text_fallback = AsyncMock()
    spoken = []
    worker._say = spoken.append
    worker.bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *args: None))
    asyncio.run(worker.speak_farewell("Goodbye."))
    assert spoken == ["Goodbye."]
    worker._submit_text_fallback.assert_not_awaited()


def _shutdown_phase_only():
    """Execute the real shutdown helper without booting Qt or owner services."""
    import ast
    import asyncio
    from contextlib import nullcontext
    from types import SimpleNamespace

    source = (_ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    named = {node.name: node for node in ast.walk(tree)
             if isinstance(node, ast.AsyncFunctionDef)}
    logs = []
    namespace = {"asyncio": asyncio, "Any": object, "_SHUTDOWN_TASKS": set(),
                 "_trace": SimpleNamespace(phase=lambda label: nullcontext()),
                 "bus": SimpleNamespace(log=SimpleNamespace(emit=logs.append))}
    name = "_await_shutdown_phase" if "_await_shutdown_phase" in named else "_safe_async"
    exec(compile(ast.Module(body=[named[name]], type_ignores=[]), "<shutdown helper>", "exec"),
         namespace)
    if name == "_safe_async":
        phase = lambda label, coro: namespace[name](label, coro)
    else:
        phase = lambda label, coro: namespace[name](label, coro, logs.append, timeout=0.025)
    return phase, namespace, logs


def test_shutdown_phase_returns_at_deadline_even_if_child_refuses_cancellation():
    import asyncio

    phase, namespace, logs = _shutdown_phase_only()
    async def scenario():
        release = asyncio.Event()
        async def blocked():
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
        stopping = asyncio.create_task(phase("synthetic service", blocked()))
        done, _pending = await asyncio.wait({stopping}, timeout=0.18)
        owned = list(namespace["_SHUTDOWN_TASKS"])
        release.set()
        await stopping
        await asyncio.gather(*owned, return_exceptions=True)
        await asyncio.sleep(0)
        assert stopping in done, "one service still holds the rest of shutdown"
        assert owned, "cancel-resistant tasks must retain an owner"
        assert any("synthetic service" in line and "still running" in line for line in logs)
        assert not namespace["_SHUTDOWN_TASKS"]
    asyncio.run(scenario())


def test_shutdown_phase_caller_cancellation_propagates_and_child_is_observed():
    import asyncio
    phase, namespace, _logs = _shutdown_phase_only()
    async def scenario():
        started, ended = asyncio.Event(), asyncio.Event()
        async def work():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()
        owner = asyncio.create_task(phase("cancelled owner", work()))
        await started.wait()
        owner.cancel()
        try:
            await owner
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("shutdown swallowed its caller's cancellation")
        await ended.wait()
        await asyncio.sleep(0)
        assert not namespace["_SHUTDOWN_TASKS"]
    asyncio.run(scenario())


def test_shutdown_phase_reports_a_failure_after_its_deadline():
    import asyncio
    phase, namespace, logs = _shutdown_phase_only()
    async def scenario():
        released = asyncio.Event()
        async def late_failure():
            while not released.is_set():
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    continue
            raise ValueError("synthetic late close")
        await phase("late service", late_failure())
        owned = list(namespace["_SHUTDOWN_TASKS"])
        released.set()
        await asyncio.gather(*owned, return_exceptions=True)
        await asyncio.sleep(0)
        assert sum("late service" in line and "ValueError" in line for line in logs) == 1
    asyncio.run(scenario())


def test_twenty_synthetic_shutdowns_finish_with_goodbye_and_all_cleanup(tmp_path):
    """Execute the real teardown sequence; services, speech and sockets are fake.

    This is deliberately not the installed-app acceptance probe: no app, real
    device, owner configuration, process host or network listener is started.
    """
    import ast
    import asyncio
    import builtins
    import json
    import time
    from types import SimpleNamespace
    from orion_core.remote import RemoteGateway
    from orion_core.shutdown_trace import ShutdownTrace

    source = (_ROOT / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    application = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == "run_application")
    teardown = next(n.finalbody for n in application.body if isinstance(n, ast.Try)
                    and any(isinstance(p, ast.FunctionDef) and p.name == "_safe"
                            for p in n.finalbody))

    class NoImports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.Pass()
        def visit_ImportFrom(self, node):
            return ast.Pass()

    wrapper = ast.parse("async def teardown():\n    pass\n").body[0]
    wrapper.body = teardown
    wrapper = NoImports().visit(wrapper)
    code = compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                   "<real application teardown with fake dependencies>", "exec")
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in {"_await_shutdown_phase", "_await_farewell", "_close_shutdown_stores"}]
    helper_code = compile(ast.Module(body=functions, type_ignores=[]), "<shutdown helpers>", "exec")
    receipts = []

    async def scenario():
        for cycle in range(20):
            calls, logs, written = [], [], []
            class Fake:
                def __init__(self, name):
                    self.name = name
                def __getattr__(self, name):
                    return Fake(f"{self.name}.{name}")
                def __call__(self, *args, **kwargs):
                    calls.append(self.name)
                    return self
                def __await__(self):
                    yield from ()
                    return None
                def __bool__(self):
                    return False
            namespace = {n.id: Fake(n.id) for n in ast.walk(wrapper)
                         if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                         and not hasattr(builtins, n.id)}
            trace = ShutdownTrace()
            trace.write = lambda path: written.append(trace.report())
            trace.begin()
            bus = SimpleNamespace(state=SimpleNamespace(emit=lambda text: calls.append("state")),
                                  log=SimpleNamespace(emit=logs.append))
            gateway = object.__new__(RemoteGateway)
            gateway.bus = bus
            gateway._closing = False
            gateway._stop_task = None
            gateway._shutdown_tasks = set()
            gateway._event_subs = set()
            gateway._chat_streams = {}
            async def remote_closed():
                calls.append("remote runner cleaned")
            async def socket_closed(**kwargs):
                calls.append("remote socket cleaned")
            gateway._runner = SimpleNamespace(cleanup=remote_closed)
            gateway._audio_sessions = {"phone": SimpleNamespace(close=socket_closed)}
            gateway._amp_task = None
            names = ["wal_task", "audio_task", "worker_task", "telemetry_task",
                     "diagnostics_task", "briefing_task", "proactive_task",
                     "connectivity_task", "reminders_task", "sentinel_task",
                     "presence_task", "audio_studio_task", "pipeline_task", "security_task",
                     "cognitive_loop_task", "thoughts_task", "reporting_task",
                     "housekeeping_task", "improvement_task"]
            tasks = {name: asyncio.create_task(asyncio.Event().wait(), name=name) for name in names}
            namespace.update(tasks)
            namespace.update(asyncio=asyncio, time=time, Any=object, _SHUTDOWN_TASKS=set(),
                             _trace=trace, bus=bus, CONFIG_DIR=tmp_path,
                             gateway=gateway, dispatcher=SimpleNamespace(gateway=None),
                             restart_requested=False, neural_planner=None, gestures=None,
                             face_tracker=None, proactivity_engine_task=None, globe_views=[])
            exec(helper_code, namespace)
            exec(code, namespace)
            began = time.monotonic()
            await namespace["teardown"]()
            elapsed = time.monotonic() - began
            await asyncio.sleep(0)
            assert elapsed < 8.0, f"synthetic cycle {cycle}: {elapsed:.3f}s"
            assert not logs, logs
            assert not namespace["_SHUTDOWN_TASKS"] and not gateway._shutdown_tasks
            assert all(task.done() for task in tasks.values())
            # The goodbye goes through say_goodbye (drops stale queued speech,
            # never waits on the network - tests/test_parting_voice.py).
            assert {"worker.speech.say_goodbye", "remote runner cleaned", "remote socket cleaned",
                    "memory.close", "_job_store.default_store.close"}.issubset(calls), calls
            assert len(trace.phases) >= 40 and written[-1].startswith("Clean shutdown")
            assert not trace.lingering
            receipts.append({"cycle": cycle + 1, "seconds": elapsed,
                             "traced_phases": len(trace.phases), "pending_tasks": 0,
                             "goodbye": "fake playback through real farewell wait",
                             "report": written[-1]})
    asyncio.run(scenario())
    (tmp_path / "twenty_synthetic_shutdowns.json").write_text(
        json.dumps({"scope": "synthetic source teardown; installed acceptance still pending",
                    "cycles": receipts}, indent=2), encoding="utf-8")


# ── W1.3: the report says what was slow, and teardown fits the watchdog ────
# The installed app was FORCED "Still inside: remote gateway" on 8 and 9 Oct
# although that build's gateway stop was already capped, and no phone
# reproduction made it take longer than 4 s. "Still inside" alone cannot tell
# a step that is waiting from one during which the loop itself was blocked.

def _phase_helper():
    """The real _await_shutdown_phase, without importing app.py (Qt)."""
    import ast
    import asyncio

    tree = ast.parse((_ROOT / "app.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                and n.name == "_await_shutdown_phase")
    namespace = {"asyncio": asyncio, "Any": object, "_SHUTDOWN_TASKS": set()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<shutdown helper>", "exec"),
         namespace)
    return namespace["_await_shutdown_phase"], namespace


def test_a_step_that_hits_its_budget_is_named_gave_up_in_the_report():
    import asyncio

    from orion_core.shutdown_trace import ShutdownTrace
    helper, namespace = _phase_helper()
    trace, logs = ShutdownTrace(), []
    trace.begin()

    async def stuck_close():
        await asyncio.Event().wait()            # a phone that never answers

    async def scenario():
        with trace.phase("remote gateway"):
            await helper("remote gateway", stuck_close(), logs.append,
                         timeout=0.2, trace=trace)
        assert trace.awaiting is None
        await asyncio.gather(*namespace["_SHUTDOWN_TASKS"], return_exceptions=True)
    asyncio.run(scenario())
    report = trace.report()
    first = report.splitlines()[0]
    assert first.startswith("Clean shutdown after") and "1 step(s) gave up" in first
    assert "remote gateway  (gave up after 0.2 s)" in report
    assert "remote gateway: gave up after 0.2 s; cancelled; waiting at" in report
    assert "in stuck_close()" in report          # where it was suspended
    assert any("remote gateway gave up after 0.2 s" in line for line in logs)


def test_a_step_that_finishes_in_time_is_not_reported_as_given_up():
    import asyncio

    from orion_core.shutdown_trace import ShutdownTrace
    helper, _namespace = _phase_helper()
    trace, logs = ShutdownTrace(), []
    trace.begin()

    async def quick():
        await asyncio.sleep(0)
        return "closed"

    async def scenario():
        with trace.phase("provider router"):
            return await helper("provider router", quick(), logs.append,
                                timeout=1.0, trace=trace)
    assert asyncio.run(scenario()) == "closed"
    assert not trace.gave_up and not logs and trace.awaiting is None
    assert trace.report().splitlines()[0].endswith(" s.")    # the plain clean line


def test_the_teardown_budget_cuts_each_step_to_the_shared_deadline():
    trace, now = _trace_with_clock()
    assert trace.budget(3.0) == 3.0                  # nothing requested yet
    trace.begin()
    assert trace.budget(3.0) == 3.0
    now[0] += trace.TEARDOWN_BUDGET_S - 1.0
    assert abs(trace.budget(3.0) - 1.0) < 1e-9       # only what is left
    now[0] += 5.0
    assert trace.budget(3.0) == trace.MIN_STEP_S     # late, but still a moment
    assert trace.TEARDOWN_BUDGET_S <= 21.0, "teardown must end well inside the 25 s watchdog"


def test_the_heartbeat_names_a_blocked_event_loop():
    import asyncio
    import time

    from orion_core.shutdown_trace import ShutdownTrace
    trace = ShutdownTrace()
    trace.begin()

    async def scenario():
        beat = asyncio.create_task(trace.watch_loop(period=0.02))
        await asyncio.sleep(0.06)
        with trace.phase("remote gateway"):
            time.sleep(0.4)                  # synchronous code holding the loop
            await asyncio.sleep(0.06)
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
    asyncio.run(scenario())
    assert trace.stalls, "a 0.4 s block of the loop went unrecorded"
    where, seconds = max(trace.stalls, key=lambda s: s[1])
    assert seconds >= 0.3 and "remote gateway" in where
    assert "Event loop blocked during teardown:" in trace.report()


def test_a_forced_report_says_whether_the_loop_was_blocked_or_waiting():
    import asyncio

    trace, now = _trace_with_clock()
    trace.begin()

    async def stuck_close():
        await asyncio.Event().wait()

    async def scenario():
        task = asyncio.create_task(stuck_close())
        await asyncio.sleep(0)
        trace.awaiting = task
        with trace.phase("remote gateway"):
            now[0] += 25.0
            trace.last_beat = now[0] - 18.0
            blocked = trace.report(forced=True)
            trace.last_beat = now[0] - 0.05
            waiting = trace.report(forced=True)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return blocked, waiting
    blocked, waiting = asyncio.run(scenario())
    lines = blocked.splitlines()
    assert lines[0] == "FORCED by the shutdown watchdog after 25.0 s."
    assert lines[1] == "Still inside: remote gateway"     # where diagnosis reads it
    assert "  event loop: BLOCKED - it last ran 18.0 s ago" in lines
    assert any(line.startswith("  awaiting: ") and "in stuck_close()" in line for line in lines)
    assert any(line.startswith("  main thread: ") for line in lines)
    assert "the step is waiting, not blocking it" in waiting


def test_a_slow_synchronous_step_is_named_with_its_budget():
    trace, now = _trace_with_clock()
    trace.begin()
    assert not trace.note_slow("memory", 0.4, 1.0)
    assert trace.note_slow("wal checkpoint", 2.4, 1.0)
    assert "  wal checkpoint: 2.40 s, budget 1.0 s" in trace.report().splitlines()


def test_a_drain_that_runs_out_of_time_names_what_it_left_running():
    """background.drain() and selfrepair.drain() bound themselves and return
    what they could not finish; the teardown used to throw that away, so a
    2.0 s drain said nothing about which task resisted cancellation."""
    import ast

    tree = ast.parse((_ROOT / "app.py").read_text(encoding="utf-8"))
    application = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == "run_application")
    teardown = next(n.finalbody for n in application.body if isinstance(n, ast.Try)
                    and any(isinstance(p, ast.FunctionDef) and p.name == "_safe"
                            for p in n.finalbody))
    helper = next(p for p in teardown if isinstance(p, ast.FunctionDef)
                  and p.name == "_left_behind")
    trace, _now = _trace_with_clock()
    trace.begin()
    logs = []
    from types import SimpleNamespace
    namespace = {"_trace": trace, "Any": object,
                 "bus": SimpleNamespace(log=SimpleNamespace(emit=logs.append))}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<teardown helper>", "exec"),
         namespace)
    left_behind = namespace["_left_behind"]
    left_behind("background drain", 2.0, ["orion-seed-corpora", "orion-change-awareness"])
    left_behind("self-repair drain", 3.0, 2)
    for nothing in ([], 0, None, False):
        left_behind("quiet drain", 1.0, nothing)
    report = trace.report()
    assert ("background drain: gave up after 2.0 s; still running: "
            "orion-change-awareness, orion-seed-corpora") in report
    assert "self-repair drain: gave up after 3.0 s; 2 still running" in report
    assert "quiet drain" not in report and len(logs) == 2
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    assert '_left_behind("background drain", 2.0' in src
    assert '_left_behind("self-repair drain", 3.0' in src


def test_the_last_wal_checkpoint_runs_after_the_stores_close_and_off_the_loop():
    """It ran synchronously mid-teardown: one connection per store, each able
    to wait 5 s on a busy lock with the event loop frozen for all of it."""
    src = (_ROOT / "app.py").read_text(encoding="utf-8")
    teardown = src[src.index("_safe(\"self-repair\", selfrepair.shutdown)"):]
    checkpoint = teardown.index('"wal checkpoint"')
    assert teardown.index('("memory", memory.close)') < checkpoint
    assert "asyncio.to_thread(\n            _close_shutdown_stores" in teardown
    assert "if stores_closed:" in teardown
    assert checkpoint < teardown.index('_trace.write(CONFIG_DIR / "diagnostics"')
    assert "asyncio.to_thread(housekeeper.checkpoint_all)" in teardown
    assert '_safe("wal checkpoint"' not in src


def test_every_awaited_teardown_step_is_bounded_and_traced():
    """No awaited step may sit outside the budgeted helper: a bare await in
    the teardown is exactly how one stuck close used to hold everything."""
    import ast

    tree = ast.parse((_ROOT / "app.py").read_text(encoding="utf-8"))
    application = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)
                       and n.name == "run_application")
    teardown = next(n.finalbody for n in application.body if isinstance(n, ast.Try)
                    and any(isinstance(p, ast.FunctionDef) and p.name == "_safe"
                            for p in n.finalbody))
    allowed = {"_safe_async", "_await_farewell", "gather"}
    bare = []
    for statement in teardown:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Await):
                call = node.value
                name = (getattr(call.func, "attr", None) or getattr(call.func, "id", None)
                        if isinstance(call, ast.Call) else None)
                if name not in allowed:
                    bare.append(ast.unparse(node)[:80])
    assert not bare, f"unbounded awaits in teardown: {bare}"


def _shutdown_function(name):
    import ast
    import asyncio
    import time
    from typing import Any

    tree = ast.parse((_ROOT / "app.py").read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    namespace = {"asyncio": asyncio, "time": time, "Any": Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<shutdown>", "exec"),
         namespace)
    return namespace[name]


def test_muted_goodbye_has_no_generation_grace_or_playback_tail():
    import asyncio
    import time
    from types import SimpleNamespace

    wait = _shutdown_function("_await_farewell")
    began = time.monotonic()
    asyncio.run(wait(SimpleNamespace(muted=True, is_busy=lambda: False), limit=0.1))
    assert time.monotonic() - began < 0.05


def test_store_close_worker_keeps_order_and_attempts_every_store():
    import asyncio
    import threading
    import time

    close = _shutdown_function("_close_shutdown_stores")
    main_thread = threading.get_ident()
    seen, errors, beats = [], [], []

    def slow():
        seen.append(("session", threading.get_ident()))
        time.sleep(0.08)
        raise RuntimeError("synthetic store failure")

    def memory():
        seen.append(("memory", threading.get_ident()))

    async def scenario():
        task = asyncio.create_task(asyncio.to_thread(
            close, (("session", slow), ("memory", memory)), errors.append))
        while not task.done():
            beats.append(time.monotonic())
            await asyncio.sleep(0.01)
        assert not await task

    asyncio.run(scenario())
    assert [label for label, _thread in seen] == ["session", "memory"]
    assert all(thread != main_thread for _label, thread in seen)
    assert len(beats) >= 4 and len(errors) == 1
    assert "session" in errors[0] and "synthetic store failure" in errors[0]


def test_worker_audio_joins_allow_the_event_loop_to_keep_running():
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    from orion_core.live_worker import GenAILiveWorker

    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.stop_event = asyncio.Event()
    worker.conn_state = SimpleNamespace(set=lambda *args: None)
    worker.session = None
    seen, beats = [], []
    main_thread = threading.get_ident()

    def stop(label):
        seen.append((label, threading.get_ident()))
        time.sleep(0.06)

    worker.mic = SimpleNamespace(stop=lambda: stop("microphone"))
    worker.speech = SimpleNamespace(stop=lambda: stop("speech"))

    async def scenario():
        task = asyncio.create_task(worker.stop())
        while not task.done():
            beats.append(time.monotonic())
            await asyncio.sleep(0.01)
        await task
        await worker.stop()

    asyncio.run(scenario())
    assert [label for label, _thread in seen] == ["microphone", "speech"]
    assert all(thread != main_thread for _label, thread in seen)
    assert len(beats) >= 5


def _live_run_tail():
    """Execute the real statements after run's session loop, without booting it."""
    import ast
    import asyncio

    tree = ast.parse((_ROOT / "live_worker.py").read_text(encoding="utf-8"))
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "GenAILiveWorker")
    run = next(node for node in worker.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name == "run")
    loop_index = max(index for index, node in enumerate(run.body) if isinstance(node, ast.While))
    tail = ast.parse("async def tail(self):\n    pass\n").body[0]
    tail.body = run.body[loop_index + 1:]
    namespace = {"asyncio": asyncio}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tail], type_ignores=[])),
                 "<real Live worker run tail>", "exec"), namespace)
    return namespace["tail"]


def test_live_run_tail_unmanaged_speech_close_keeps_the_loop_running():
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    tail = _live_run_tail()
    main_thread = threading.get_ident()

    async def scenario(raises):
        seen, beats = [], []

        def stop_speech():
            seen.append(threading.get_ident())
            time.sleep(0.08)
            if raises:
                raise RuntimeError("synthetic unmanaged speech failure")

        worker = SimpleNamespace(speech=SimpleNamespace(stop=stop_speech))
        task = asyncio.create_task(tail(worker))
        while not task.done():
            beats.append(time.monotonic())
            await asyncio.sleep(0.01)
        error = None
        try:
            await task
        except RuntimeError as exc:
            error = str(exc)
        assert error == ("synthetic unmanaged speech failure" if raises else None)
        assert len(seen) == 1 and seen[0] != main_thread
        assert len(beats) >= 4

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_live_run_tail_never_repeats_an_owned_pending_or_completed_audio_close():
    import asyncio
    import threading
    import time
    from types import SimpleNamespace

    tail = _live_run_tail()

    async def scenario(complete):
        entered, release = threading.Event(), threading.Event()
        seen = []

        def owned_close():
            entered.set()
            assert release.wait(1.0)

        def duplicate_close():
            seen.append(threading.get_ident())
            time.sleep(0.08)

        owned = asyncio.create_task(asyncio.to_thread(owned_close))
        worker = SimpleNamespace(speech=SimpleNamespace(stop=duplicate_close),
                                 _audio_stop_task=owned)
        try:
            while not entered.is_set():
                await asyncio.sleep(0.005)
            if complete:
                release.set()
                await owned
            task = asyncio.create_task(tail(worker))
            await asyncio.sleep(0.01)
            assert task.done()
            await task
            assert seen == []
            if not complete:
                assert not owned.done()
        finally:
            release.set()
            await owned

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))


def test_muted_live_goodbye_does_not_wait_for_a_model_or_queue_speech():
    import asyncio
    from types import SimpleNamespace

    from orion_core.live_worker import GenAILiveWorker

    worker = GenAILiveWorker.__new__(GenAILiveWorker)
    worker.speech = SimpleNamespace(muted=True, is_busy=lambda: True)
    worker.session = object()
    worker._say = lambda _text: (_ for _ in ()).throw(AssertionError("muted speech"))
    worker._send_text_turn = worker._say

    async def scenario():
        await asyncio.wait_for(worker.speak_farewell("goodbye"), timeout=0.05)
        await asyncio.wait_for(worker._await_speech(), timeout=0.05)

    asyncio.run(scenario())


def test_failed_or_cancelled_audio_stop_still_retires_the_live_session():
    import asyncio
    import threading
    from types import SimpleNamespace

    from orion_core.live_worker import GenAILiveWorker

    async def scenario(hang):
        worker = GenAILiveWorker.__new__(GenAILiveWorker)
        worker.stop_event = asyncio.Event()
        worker.conn_state = SimpleNamespace(set=lambda *args: None)
        logs, closed = [], []
        worker.bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append))
        entered, release = threading.Event(), threading.Event()

        async def close():
            closed.append(True)

        def stop_mic():
            entered.set()
            if hang:
                assert release.wait(2)

        def stop_speech():
            raise RuntimeError("synthetic speech stop")

        worker.mic = SimpleNamespace(stop=stop_mic)
        worker.speech = SimpleNamespace(stop=stop_speech)
        worker.session = SimpleNamespace(close=close)
        task = asyncio.create_task(worker.stop())
        try:
            while not entered.is_set():
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.005)
            if hang:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                await task
            assert closed == [True] and worker.session is None
        finally:
            release.set()
            await asyncio.gather(task, worker._audio_stop_task, return_exceptions=True)
        assert any("synthetic speech stop" in line for line in logs)

    asyncio.run(scenario(False))
    asyncio.run(scenario(True))
