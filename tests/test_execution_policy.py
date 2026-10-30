"""Apply staged sources before running; all OS/provider transports are fake."""
from __future__ import annotations

import asyncio
from collections import deque
from contextvars import copy_context
import threading
from types import SimpleNamespace

import pytest

from orion_core import browser, execution_policy as policy
from orion_core.control import AutonomousControlLayer
from orion_core.data import ToolResult


@pytest.mark.parametrize("effect", ["mouse", "keyboard", "focus", "clipboard", "browser_window", "shell_open"])
def test_background_effect_is_a_typed_deferral(effect):
    with policy.background_scope("test-job", "job-1"):
        result = policy.require_foreground(effect)
        assert result is not None and not result.ok
        assert result.receipt == {"status": "requires_foreground", "effect": effect,
                                  "origin": "test-job", "run_id": "job-1",
                                  "background_scope_active": True, "desktop_mutated": False}


def test_nested_foreground_opt_in_cannot_upgrade_background_job():
    with policy.background_scope("job"):
        with pytest.raises(RuntimeError, match="cannot upgrade"):
            with policy.foreground_scope("untrusted-plugin"):
                pytest.fail("Unreachable foreground override")
        with policy.background_scope("nested-agent"):
            assert not policy.foreground_authorized()


def test_explicit_host_foreground_authorization_is_task_local_and_revocable():
    with policy.foreground_scope("explicit-user-desktop-command"):
        assert policy.foreground_authorized()
        assert policy.require_foreground("keyboard") is None
        copied = copy_context()
    assert not policy.foreground_authorized()
    assert copied.run(policy.foreground_authorized) is False
    assert copied.run(policy.require_foreground, "keyboard").receipt["desktop_mutated"] is False


async def test_cancellation_revokes_the_scope_observed_by_late_physical_workers():
    started, release = threading.Event(), threading.Event()
    observed = []
    def worker():
        observed.append(policy.current_scope())
        started.set()
        assert release.wait(3)
        return policy.require_foreground("clipboard")
    pending_worker = None
    async def job():
        nonlocal pending_worker
        with policy.background_scope("job", "cancel-me"):
            pending_worker = asyncio.create_task(asyncio.to_thread(worker))
            await asyncio.Event().wait()
    task = asyncio.create_task(job())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not observed[0].active
    finally:
        release.set()
        if pending_worker is not None:
            result = await pending_worker
            assert result.receipt["background_scope_active"] is False
            assert result.receipt["desktop_mutated"] is False


async def test_to_thread_restriction_does_not_leak_to_an_independent_foreground_task():
    async def quiet():
        with policy.background_scope("plugin_schedule"):
            return await asyncio.to_thread(policy.require_foreground, "focus")
    async def interactive():
        with policy.foreground_scope("explicit-user-control"):
            await asyncio.sleep(0)
            return policy.require_foreground("focus")
    deferred, direct = await asyncio.gather(quiet(), interactive())
    assert deferred.receipt["status"] == "requires_foreground" and direct is None


@pytest.mark.parametrize("method,args", [("send_hotkeys", ("ctrl+v",)), ("focus_window", ("Synthetic",)),
                                         ("edit_text", ("Synthetic text",)), ("open_application", ("Synthetic",))])
def test_control_entrypoints_return_before_touching_any_backend(method, args):
    layer = object.__new__(AutonomousControlLayer)
    with policy.background_scope("ordinary-tool"):
        result = getattr(layer, method)(*args)
    assert not result.ok and result.receipt["status"] == "requires_foreground"


@pytest.mark.parametrize("method,args", [("_send_char", ("x",)), ("_press_backspace", ())])
def test_low_level_keyboard_fallbacks_cannot_bypass_scope(method, args):
    layer = object.__new__(AutonomousControlLayer)
    with policy.background_scope("plugin"):
        with pytest.raises(RuntimeError, match="foreground"):
            getattr(layer, method)(*args)


def test_browser_background_call_never_resolves_or_launches_the_user_browser(monkeypatch):
    monkeypatch.setattr(browser, "browser_path", lambda: pytest.fail("Background browser resolution"))
    monkeypatch.setattr(browser.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Background process launch"))
    monkeypatch.setattr(browser.webbrowser, "open", lambda *_a, **_k: pytest.fail("Background browser fallback"))
    with policy.background_scope("workflow"):
        assert browser.open_url("https://example.invalid") is False


async def test_background_web_search_fetches_structured_results_without_opening_a_browser(monkeypatch):
    from orion_core import dispatch_web, web_search_backends
    calls = []
    async def search(query, *, limit):
        calls.append((query, limit))
        return [SimpleNamespace(title="Synthetic source", url="https://example.invalid/source", snippet="Retained result")]
    monkeypatch.setattr(web_search_backends, "search", search)
    monkeypatch.setattr(dispatch_web, "open_url", lambda *_a: pytest.fail("Background search opened a browser"))
    host = dispatch_web.WebDispatchMixin()
    with policy.background_scope("ordinary-tool"):
        result = await host.web_search({"query": "Synthetic query"})
    assert result.ok and calls == [("Synthetic query", 6)]
    assert result.receipt["browser_opened"] is False
    assert result.receipt["results"][0]["url"] == "https://example.invalid/source"


def _fake_backend(monkeypatch, hits=None, error=None):
    from orion_core import web_search_backends
    calls = []

    async def search(query, *, limit):
        calls.append(query)
        if error is not None:
            raise error
        return hits if hits is not None else [SimpleNamespace(
            title="Synthetic AI PC review", url="https://example.invalid/review",
            snippet="Synthetic benchmark of three laptops, October 2026")]
    monkeypatch.setattr(web_search_backends, "search", search)
    return calls


@pytest.mark.parametrize("asked_to_see", [False, True])
async def test_a_foreground_search_hands_the_model_real_results(monkeypatch, asked_to_see):
    """It used to open Bing and return 'Secure web search opened' - the model
    was told nothing and answered live questions from memory."""
    from orion_core import dispatch_web
    calls = _fake_backend(monkeypatch)
    opened = []
    monkeypatch.setattr(dispatch_web, "open_url", lambda url: opened.append(url) or True)
    args = {"query": "best AI PCs"} | ({"open": True} if asked_to_see else {})
    result = await dispatch_web.WebDispatchMixin().web_search(args)
    assert result.ok and calls == ["best AI PCs"]
    assert "Synthetic AI PC review" in result.text and "https://example.invalid/review" in result.text
    assert result.receipt["results"][0]["url"] == "https://example.invalid/review"
    assert bool(opened) is asked_to_see, "the browser opens only when the user asked to see it"
    assert result.receipt["browser_opened"] is asked_to_see


async def test_a_foreground_search_that_reads_nothing_says_so_and_shows_the_browser(monkeypatch):
    from orion_core import dispatch_web
    _fake_backend(monkeypatch, error=OSError("every engine declined"))
    opened = []
    monkeypatch.setattr(dispatch_web, "open_url", lambda url: opened.append(url) or True)
    result = await dispatch_web.WebDispatchMixin().web_search({"query": "uni events"})
    assert not result.ok and opened and "NOT known to me" in result.text
    assert result.receipt["browser_opened"] is True


async def test_searching_in_the_browser_opens_it_and_still_returns_results(monkeypatch):
    from orion_core import dispatch_web
    _fake_backend(monkeypatch)
    opened = []
    monkeypatch.setattr(dispatch_web, "open_url", lambda url: opened.append(url) or True)
    result = await dispatch_web.WebDispatchMixin().browser_control(
        {"action": "search", "query": "events at my university"})
    assert result.ok and opened and "Synthetic AI PC review" in result.text


async def test_web_control_has_a_search_action_that_returns_results(monkeypatch):
    from orion_core import dispatch_web
    _fake_backend(monkeypatch)
    visited = []

    class _Browser:
        async def open(self, url):
            visited.append(url)
            return ToolResult(f"Opened {url}")

    host = dispatch_web.WebDispatchMixin()
    host.web = _Browser()
    result = await host.web_control({"action": "search", "query": "new AI models"})
    assert result.ok and visited == ["https://www.bing.com/search?q=new+AI+models"]
    assert "Searched for 'new AI models'" in result.text and "Synthetic AI PC review" in result.text
    assert not (await host.web_control({"action": "search"})).ok


async def test_background_search_failure_does_not_fall_back_to_browser(monkeypatch):
    from orion_core import dispatch_web, web_search_backends
    async def search(*_a, **_k):
        raise OSError("Synthetic retrieval failure")
    monkeypatch.setattr(web_search_backends, "search", search)
    monkeypatch.setattr(dispatch_web, "open_url", lambda *_a: pytest.fail("Failure opened a browser"))
    with policy.background_scope("job"):
        result = await dispatch_web.WebDispatchMixin().web_search({"query": "Synthetic"})
    assert not result.ok and result.receipt["browser_opened"] is False


async def test_real_job_manager_dispatch_sets_and_revokes_background_scope(monkeypatch):
    from orion_core import admission, job_store, jobs
    from orion_core.bus import OrionBus
    observed = []
    async def dispatch(name, args):
        assert name == "resource_status"
        observed.append(policy.current_scope())
        assert await asyncio.to_thread(policy.is_background)
        return ToolResult("Synthetic result")
    governor = SimpleNamespace(acting=False)
    monkeypatch.setattr(admission, "ADMISSION", admission.Admission(governor=governor))
    monkeypatch.setattr(jobs, "CONTROLS", {})
    store = job_store.JobStore(None)
    try:
        manager = jobs.JobManager(OrionBus(), dispatch, store=store)
        job = manager.start("resource_status", {})
        await job._task
        assert observed and observed[0].origin == "job" and observed[0].run_id == job.id
        assert not observed[0].active and not policy.is_background()
    finally:
        store.close()


async def test_workflow_wrapper_preserves_resume_checkpoint_arguments():
    from orion_core.workflow_engine import WorkflowEngine
    observed = []
    async def execute(*args):
        observed.append((policy.current_scope(), args))
    owner = SimpleNamespace(_execute_background=execute)
    run, definition, control = {"id": "resume-1"}, {"steps": []}, object()
    previous = [(1, {"tool": "synthetic"}, "output", True)]
    await WorkflowEngine._execute(owner, run, definition, control, 2, "previous", True, previous, False)
    scope, args = observed[0]
    assert scope.origin == "workflow" and scope.run_id == "resume-1" and not scope.active
    assert args == (run, definition, control, 2, "previous", True, previous, False)


@pytest.mark.parametrize("foreground", [False, True])
async def test_actual_dispatcher_defaults_to_background_but_keeps_explicit_host_controls(monkeypatch, foreground):
    from orion_core import dispatcher
    observed = []
    def handler(_args):
        observed.append((policy.current_scope(), policy.foreground_authorized()))
        return policy.require_foreground("keyboard") or ToolResult("Synthetic explicit control")
    monkeypatch.setattr(dispatcher, "_gap_ledger", lambda: SimpleNamespace(note_tool_call=lambda: None))
    monkeypatch.setattr(dispatcher, "_note_gap", lambda *_a: None)
    monkeypatch.setattr(dispatcher.speaker_gate, "refusal", lambda *_a: None)
    host = SimpleNamespace(handler_table=lambda: {"synthetic_control": handler},
                           _coerce_to_schema=lambda _name, args: (args, None),
                           _swarm_launch=lambda *_a: None, active_tools=0,
                           recent_tools=deque(maxlen=60), telemetry=None,
                           bus=SimpleNamespace(log=SimpleNamespace(emit=lambda *_a: None)))
    if foreground:
        with policy.foreground_scope("explicit-user-desktop-command"):
            result = await dispatcher.OrionDispatcher.dispatch(host, "synthetic_control", {})
        assert result.ok and observed == [(None, True)]
    else:
        result = await dispatcher.OrionDispatcher.dispatch(host, "synthetic_control", {"foreground": True})
        assert not result.ok and result.receipt["status"] == "requires_foreground"
        assert observed[0][0].origin == "tool" and not observed[0][0].active
        assert observed[0][1] is False
    assert policy.current_scope() is None


@pytest.mark.parametrize("server,tool", [("orion_desktop", "desktop_click"),
                                         ("custom_desktop", "keyboard_type"),
                                         ("third_party", "apparently_read_only")])
async def test_mcp_broker_blocks_unreviewed_effects_before_wake_or_send(monkeypatch, server, tool):
    from orion_core import mcp_host
    monkeypatch.setattr(mcp_host, "load_config", lambda: {"servers": {server: {
        "command": "synthetic.exe", "args": [], "annotations": {"readOnlyHint": True}}}})
    # Deliberately omit startup/parked/connection state: refusal must happen
    # before waking or sending to any subprocess, even a self-described read.
    host = SimpleNamespace(_closing=False)
    with policy.background_scope("ordinary-tool"):
        reply = await mcp_host.MCPHost.call(host, server, tool, {})
    assert reply.kind == "tool" and reply.structured_content["status"] == "requires_foreground"
    assert reply.structured_content["desktop_mutated"] is False
    assert ToolResult(reply).receipt["structuredContent"]["status"] == "requires_foreground"


@pytest.mark.parametrize("variant", ["renamed_command", "extra_arguments", "custom_env", "inherit_env", "live_mismatch"])
async def test_bundled_mcp_name_cannot_hide_a_different_child_process(monkeypatch, variant):
    from orion_core import mcp_host
    command, args = mcp_host._bundled_spec("orion_desktop")
    trusted = {"command": command, "args": args, "env": {}}
    changed = {**trusted, "args": list(args)}
    if variant in {"renamed_command", "live_mismatch"}:
        changed["command"] = "different-child.exe"
    elif variant == "extra_arguments":
        changed["args"].append("--input")
    elif variant == "custom_env":
        changed["env"] = {"PYTHONPATH": "synthetic-extension"}
    else:
        changed["inherit_env"] = ["PYTHONPATH"]
    saved = trusted if variant == "live_mismatch" else changed
    monkeypatch.setattr(mcp_host, "load_config", lambda: {"servers": {"orion_desktop": saved}})
    conn = SimpleNamespace(spec=changed)
    host = SimpleNamespace(_closing=False, _startup_task=None, parked={}, servers={"orion_desktop": conn})
    with policy.background_scope("workflow"):
        reply = await mcp_host.MCPHost.call(host, "orion_desktop", "desktop_controls", {})
    assert reply.is_error and reply.structured_content["desktop_mutated"] is False


@pytest.mark.parametrize("server,tool", [("orion_desktop", "desktop_controls"),
                                         ("orion_world", "world_feed_status"),
                                         ("orion_network", "phone_gateway_status"),
                                         ("orion_telemetry", "system_snapshot")])
async def test_reviewed_bundled_observation_reaches_only_the_bound_child(monkeypatch, server, tool):
    from orion_core import mcp_host
    command, args = mcp_host._bundled_spec(server)
    spec = {"command": command, "args": args, "env": {}}
    calls = []
    async def call_tool(name, arguments):
        calls.append((name, arguments, policy.is_background()))
        return mcp_host.MCPReply("Synthetic observed state")
    monkeypatch.setattr(mcp_host, "load_config", lambda: {"servers": {server: spec}})
    conn = SimpleNamespace(spec=spec, proc=None, tools=[{"name": tool}], call_tool=call_tool)
    host = SimpleNamespace(_closing=False, _startup_task=None, parked={}, servers={server: conn})
    with policy.background_scope("job"):
        reply = await mcp_host.MCPHost.call(host, server, tool, {})
    assert not reply.is_error and calls == [(tool, {}, True)]


async def test_direct_mcp_connection_call_cannot_bypass_the_host_broker():
    from orion_core import mcp_host
    conn = SimpleNamespace(name="custom_desktop", spec={"command": "synthetic.exe", "args": []})
    with policy.background_scope("plugin"):
        reply = await mcp_host.MCPServerConn.call_tool(conn, "mouse_click", {})
    assert reply.is_error and reply.structured_content["status"] == "requires_foreground"
