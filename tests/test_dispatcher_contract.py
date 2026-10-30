"""
Dispatcher contract tests (Improvement Pass, Priority 1.1).

The three invariants that keep the 90-tool routing table safe to evolve:

  1. every tool declared to the model has a live handler;
  2. every handler is reachable through dispatch() — including with malformed
     or empty arguments — without ever raising out of the tool layer
     (SecurityViolation is the one deliberate exception);
  3. every declaration is schema-consistent (unique snake_case name,
     description, OBJECT parameters whose required keys exist).

The sweep runs against a bare dispatcher whose services are absent, with the
browser/launcher side-effect seams patched out, so no tool can touch the host.
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.data import ToolResult
from orion_core.dispatcher import TOOL_DECLARATIONS, OrionDispatcher
from orion_core.security import SecurityViolation


class _Signal:
    def __init__(self, sink, name):
        self._sink = sink
        self._name = name

    def emit(self, *payload):
        self._sink.append((self._name, payload))


class _StubBus:
    def __init__(self):
        self.emitted = []

    def __getattr__(self, name):
        sig = _Signal(self.emitted, name)
        object.__setattr__(self, name, sig)
        return sig


def _bare_dispatcher():
    d = OrionDispatcher.__new__(OrionDispatcher)
    d.bus = _StubBus()
    d.telemetry = None
    d.active_tools = 0
    d.recent_tools = deque(maxlen=60)
    return d


@pytest.fixture(autouse=True)
def _no_host_side_effects(monkeypatch):
    """No test may open a browser, launch a process or touch the host."""
    import subprocess
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: True)
    # ORION opens links through orion_core.browser.open_url now, so stubbing
    # webbrowser alone no longer covers this. (open_url uses subprocess.Popen,
    # which _forbidden below would catch — but loudly, and this fixture's job
    # is to make the host untouchable, not to fail a test for reaching it.)
    from orion_core import browser as _browser
    monkeypatch.setattr(_browser, "open_url", lambda *a, **k: True)
    monkeypatch.setattr(subprocess, "Popen", _forbidden, raising=True)
    monkeypatch.setattr(subprocess, "run", _forbidden, raising=True)
    if hasattr(sys.modules.get("os"), "startfile"):
        import os
        monkeypatch.setattr(os, "startfile", lambda *a, **k: None)


def _forbidden(*args, **kwargs):
    raise AssertionError(f"tool sweep tried to spawn a process: {args!r}")


# ── declaration ⇄ handler consistency ────────────────────────────────────────

def test_every_declared_tool_has_a_live_handler():
    table = _bare_dispatcher().handler_table()
    declared = [decl["name"] for decl in TOOL_DECLARATIONS]
    missing = [name for name in declared if name not in table]
    assert not missing, f"declared tools without a handler: {missing}"


def test_every_handler_is_callable():
    table = _bare_dispatcher().handler_table()
    not_callable = [name for name, fn in table.items() if not callable(fn)]
    assert not_callable == []


def test_declaration_names_are_unique():
    names = [decl["name"] for decl in TOOL_DECLARATIONS]
    assert len(names) == len(set(names)), "duplicate tool declarations"


def test_handler_names_without_declaration_are_known_aliases():
    # Handlers may exceed declarations only for deliberate aliases / internal
    # tools; anything new must be added here consciously, not by accident.
    table = _bare_dispatcher().handler_table()
    declared = {decl["name"] for decl in TOOL_DECLARATIONS}
    undeclared = set(table) - declared
    known_internal = {"image_processor"}         # alias of process_file
    unexpected = undeclared - known_internal
    assert not unexpected, (
        f"handlers with no model-facing declaration: {sorted(unexpected)} — "
        "declare them in TOOL_DECLARATIONS or add to known_internal deliberately"
    )


# ── declaration schema consistency ────────────────────────────────────────────

def test_declaration_schema_shape():
    for decl in TOOL_DECLARATIONS:
        name = decl.get("name", "<missing>")
        assert isinstance(decl.get("name"), str) and decl["name"], decl
        assert decl["name"].replace("_", "").isalnum(), f"bad tool name {name}"
        assert decl["name"] == decl["name"].lower(), f"tool name not snake_case: {name}"
        assert isinstance(decl.get("description"), str) and decl["description"].strip(), \
            f"{name}: missing description"
        params = decl.get("parameters")
        assert isinstance(params, dict), f"{name}: parameters must be a dict"
        assert str(params.get("type", "")).upper() == "OBJECT", \
            f"{name}: parameters.type must be OBJECT"
        properties = params.get("properties", {})
        assert isinstance(properties, dict), f"{name}: properties must be a dict"
        for required_key in params.get("required", []):
            assert required_key in properties, \
                f"{name}: required key '{required_key}' not among properties"


def test_declared_property_types_are_valid():
    valid = {"STRING", "INTEGER", "NUMBER", "BOOLEAN", "ARRAY", "OBJECT"}
    for decl in TOOL_DECLARATIONS:
        for prop_name, prop in decl["parameters"].get("properties", {}).items():
            ptype = str(prop.get("type", "")).upper()
            assert ptype in valid, \
                f"{decl['name']}.{prop_name}: invalid schema type {ptype!r}"


# ── malformed-argument sweep: nothing raises out of dispatch() ────────────────

@pytest.mark.parametrize("tool_name", [decl["name"] for decl in TOOL_DECLARATIONS])
def test_dispatch_with_empty_args_never_raises(tool_name):
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch(tool_name, {}))
    assert isinstance(result, ToolResult)


@pytest.mark.parametrize("bad_args", [
    None,
    {},
    {"unexpected": object()},
    {"action": 42, "query": ["not", "a", "string"]},
])
def test_dispatch_malformed_args_return_toolresult(bad_args):
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch("window_control", bad_args))
    assert isinstance(result, ToolResult)


def test_unknown_tool_reports_cleanly():
    d = _bare_dispatcher()
    result = asyncio.run(d.dispatch("definitely_not_a_tool", {}))
    assert not result.ok and "Unknown dispatch target" in result.text


def test_dispatch_records_recent_tools_metadata():
    d = _bare_dispatcher()
    asyncio.run(d.dispatch("system_notify", {"message": "test ping"}))
    assert d.recent_tools and d.recent_tools[-1]["tool"] == "system_notify"
    assert d.active_tools == 0                      # always released


# ── the security firewall guards the tool boundary ────────────────────────────

def test_destructive_args_raise_security_violation():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("open_app", {"app_name": "cmd /c del *.*"}))


def test_destructive_tool_name_raises_security_violation():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("del orion_core", {}))


def test_nested_destructive_payload_caught():
    d = _bare_dispatcher()
    with pytest.raises(SecurityViolation):
        asyncio.run(d.dispatch("dev_workbench", {
            "action": "run", "options": {"cmd": "rm -rf /"},
        }))


# ── forged-tool fallthrough ───────────────────────────────────────────────────

def test_forged_tool_route():
    d = _bare_dispatcher()
    d._tool_handlers = {"custom_forged": lambda **kw: f"forged ran with {kw}"}
    result = asyncio.run(d.dispatch("custom_forged", {"a": 1}))
    assert result.ok and "forged ran" in result.text


def test_forged_tool_failure_contained():
    def _boom(**kw):
        raise RuntimeError("forge fault")
    d = _bare_dispatcher()
    d._tool_handlers = {"bad_forged": _boom}
    result = asyncio.run(d.dispatch("bad_forged", {}))
    assert not result.ok and "forge fault" in result.text


# ── file work off the shared loop (audit 6 Oct 2026, A02) ────────────────────

def _file_host(tmp_path, monkeypatch, files):
    from collections import deque
    from types import SimpleNamespace

    import orion_core.dispatcher as dispatcher_module
    logged = []
    monkeypatch.setattr(dispatcher_module, "_gap_ledger",
                        lambda: SimpleNamespace(note_tool_call=lambda: None))
    return SimpleNamespace(
        handler_table=lambda: {"file_controller": files.file_controller},
        _coerce_to_schema=lambda _name, args: (args, None), _swarm_launch=lambda *_a: None,
        active_tools=0, recent_tools=deque(maxlen=60), telemetry=None,
        bus=SimpleNamespace(log=SimpleNamespace(emit=logged.append))), logged


class _SlowFiles:
    """file_controller with a controlled stand-in for a slow drive."""

    def __init__(self, delay):
        self.delay = delay

    def file_controller(self, args):
        import time as _time
        from orion_core.data import ToolResult
        _time.sleep(self.delay)
        return ToolResult(f"listed {args.get('path')}")


def test_file_work_leaves_the_loop_free(tmp_path, monkeypatch):
    import asyncio

    from orion_core.dispatcher import OrionDispatcher
    host, _logged = _file_host(tmp_path, monkeypatch, _SlowFiles(0.15))

    async def run():
        beats = []

        async def heartbeat():
            for _ in range(5):
                await asyncio.sleep(0.01)
                beats.append(asyncio.get_running_loop().time())
        monitor = asyncio.create_task(heartbeat())
        result = await OrionDispatcher.dispatch(host, "file_controller", {"path": "x"})
        await monitor
        return result, beats
    result, beats = asyncio.run(run())
    assert result.ok and len(beats) == 5


def test_a_cancelled_file_operation_reports_no_success(tmp_path, monkeypatch):
    import asyncio

    from orion_core.dispatcher import OrionDispatcher
    host, logged = _file_host(tmp_path, monkeypatch, _SlowFiles(0.2))

    async def run():
        task = asyncio.create_task(
            OrionDispatcher.dispatch(host, "file_controller", {"path": "x"}))
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return task.result()
    assert asyncio.run(run()) == "cancelled"
    assert any("may still finish" in line for line in logged)


# Mark XXXV: the chain boundary is used by Live, the gateway and background jobs.
def _xxxv_chain(results):
    host = _bare_dispatcher()
    calls = []

    async def dispatch(name, args):
        calls.append((name, dict(args)))
        return results[name]

    host.dispatch = dispatch
    host._derive_chain = lambda *_args: []
    return host, calls


def _xxxv_receipts(value):
    """Locate original typed receipts without prescribing a chain envelope."""
    if isinstance(value, dict):
        if value.get("type") == "mcp":
            return [value]
        return [receipt for child in value.values() for receipt in _xxxv_receipts(child)]
    if isinstance(value, list):
        return [receipt for child in value for receipt in _xxxv_receipts(child)]
    return []


def test_xxxv_chain_preserves_two_typed_sources_and_human_evidence():
    receipts = [{"type": "mcp", "structuredContent": {"rows": [n]},
                 "provenance": {"server": f"fixture{n}", "tool": "read"},
                 "untrusted": True} for n in (1, 2)]
    evidence = [{"source": f"fixture{n}", "observations": [n]} for n in (1, 2)]
    media = {"data": b"fixture", "mime_type": "image/png"}
    first = ToolResult("one", receipt=receipts[0], evidence=[evidence[0]],
                       media=media, chain=[("second", {})])
    second = ToolResult("two", receipt=receipts[1], evidence=[evidence[1]])
    host, calls = _xxxv_chain({"first": first, "second": second})
    result = asyncio.run(host.dispatch_chain("first", {}))
    assert result.ok and calls == [("first", {}), ("second", {})]
    assert _xxxv_receipts(result.response_payload()) == receipts
    assert result.evidence == evidence and result.media is media
    first.receipt["structuredContent"]["rows"].append(99)
    first.evidence[0]["observations"].append(99)
    assert _xxxv_receipts(result.response_payload()) == receipts
    assert result.evidence[0]["observations"] == [1]
    payload = result.response_payload()
    _xxxv_receipts(payload)[0]["provenance"]["server"] = "mutated"
    assert _xxxv_receipts(result.response_payload())[0]["provenance"]["server"] == "fixture1"


def test_xxxv_chain_depth_limit_reports_unfinished_work():
    results = {f"step{n}": ToolResult(str(n), chain=[(f"step{n + 1}", {})])
               for n in range(5)}
    results["step5"] = ToolResult("last")
    host, calls = _xxxv_chain(results)
    result = asyncio.run(host.dispatch_chain("step0", {}, max_depth=4))
    assert len(calls) == 4
    assert not result.ok
    assert "unfinished" in result.text.lower() or "not run" in result.text.lower()
    assert "step4" in result.text


def test_xxxv_chain_duplicate_reports_skipped_work_without_repeating_it():
    host, calls = _xxxv_chain({"repeat": ToolResult("completed once", chain=[("repeat", {})])})
    result = asyncio.run(host.dispatch_chain("repeat", {}))
    assert calls == [("repeat", {})]
    assert not result.ok
    assert "repeat" in result.text and ("skipped" in result.text.lower()
                                        or "not run" in result.text.lower())


def test_xxxv_chain_exact_bound_with_no_pending_work_remains_successful():
    results = {f"step{n}": ToolResult(str(n), chain=[(f"step{n + 1}", {})])
               for n in range(3)}
    results["step3"] = ToolResult("last")
    host, calls = _xxxv_chain(results)
    result = asyncio.run(host.dispatch_chain("step0", {}, max_depth=4))
    assert result.ok and len(calls) == 4
    assert result.receipt is None and result.evidence is None


@pytest.mark.parametrize("mode", ["coroutine", "offloop"])
def test_xxxv_cancelled_dispatch_is_not_a_success_in_learning_or_telemetry(monkeypatch, mode):
    from types import SimpleNamespace
    import threading
    import orion_core.dispatcher as dispatcher_module
    entered, release = threading.Event(), threading.Event()
    pattern, telemetry, latency = [], [], []
    monkeypatch.setattr(dispatcher_module, "_gap_ledger",
                        lambda: SimpleNamespace(note_tool_call=lambda: None))
    monkeypatch.setattr(dispatcher_module, "_action_guard_on", lambda: False)
    from orion_core import latency as latency_module
    monkeypatch.setattr(latency_module, "TRACKER", SimpleNamespace(
        record=lambda *args: latency.append(args)))
    host = _bare_dispatcher()
    host.pattern_detector = SimpleNamespace(record=lambda *args: pattern.append(args))
    host.telemetry = SimpleNamespace(
        metrics=SimpleNamespace(observe=lambda *_args: None),
        record_tool_call=lambda *args: telemetry.append(args))

    async def coroutine(_args):
        entered.set()
        await asyncio.Event().wait()

    def offloop(_args):
        entered.set()
        assert release.wait(2)
        return ToolResult("the worker completed")

    name = "file_controller" if mode == "offloop" else "test_wait"
    host.handler_table = lambda: {name: offloop if mode == "offloop" else coroutine}

    async def run():
        task = asyncio.create_task(host.dispatch(name, {}))
        try:
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.002)
            assert entered.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert host.active_tools == 0
    assert host.recent_tools[-1]["ok"] is False
    assert pattern == [(name, {}, False)] and telemetry == [(name, False)]
    assert latency[-1][-1] == "cancelled"


def test_xxxv_slow_skill_lifecycle_leaves_the_loop_free(monkeypatch):
    from types import SimpleNamespace
    import threading
    import time
    import orion_core.dispatcher as dispatcher_module
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(dispatcher_module, "_gap_ledger",
                        lambda: SimpleNamespace(note_tool_call=lambda: None))
    monkeypatch.setattr(dispatcher_module, "_action_guard_on", lambda: False)
    host = _bare_dispatcher()
    loop_thread = threading.get_ident()
    worker_threads = []

    def slow_install(source):
        worker_threads.append(threading.get_ident())
        entered.set()
        # In the broken path this executes on the loop, so the release timeout
        # is finite and the regression fails without trapping the test process.
        assert release.wait(0.25) or threading.get_ident() == loop_thread
        return ToolResult(f"installed {source}")

    host.skills = SimpleNamespace(install=slow_install)
    host.handler_table = lambda: {"skill": host.skill_tool}

    async def run():
        beats = []

        async def heartbeat():
            for _ in range(100):
                await asyncio.sleep(0.002)
                if entered.is_set():
                    beats.append(time.monotonic())
                    release.set()
                    return
            raise AssertionError("skill installer did not start")

        monitor = asyncio.create_task(heartbeat())
        try:
            result = await host.dispatch("skill", {"action": "install", "source": "fixture"})
            await monitor
            return result, beats
        finally:
            release.set()
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    result, beats = asyncio.run(run())
    assert result.ok and beats
    assert worker_threads and worker_threads[0] != loop_thread



@pytest.mark.parametrize("said, answer", [
    ("= 2x to the power of 3 - 2", "2x^3 - 2 is already as simple as it goes; factorised, "
                                    "2(x - 1)(x^2 + x + 1). It is zero when x = 1 (plus 2 complex solutions)."),
    ("solve 2x + 3 = 7", "x = 2."),
    ("x^2 - 5x + 6 = 0", "x = 2 or x = 3."),
    ("factorise x^2 - 4", "x^2 - 4 factorises to (x - 2)(x + 2)."),
    ("differentiate x^3 + 2x", "d/dx of x^3 + 2x = 3x^2 + 2"),
    ("x^3 - 2 = 2x^3 - 2", "x = 0."),
])
def test_the_calculator_works_algebra_exactly(said, answer):
    """27 Oct 2026: "2x to the power of 3 - 2" was refused as not a calculation."""
    from orion_core.calculator import calculate
    result = calculate(said)
    assert result.ok and result.text == answer


def test_algebra_is_never_evaluated_as_code_and_stays_bounded():
    from orion_core.calculator import algebra, calculate
    assert not calculate("__import__('os').system('dir')").ok
    assert algebra("x^7 + x = 1").startswith("I can solve equations in x up to the power of 6")
    assert calculate("3 x 4").text == "3 x 4 = 12"          # arithmetic is unchanged
