"""A plugin failure stays a failure; successful writes must survive reopening."""
import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.plugin_results import plugin_result
from orion_core import local_tool_outcomes as operations


@pytest.mark.parametrize("value", [None, {"ok": "false", "result": "done"},
                                    {"ok": False, "result": "declined"},
                                    "Error: failed", "Failed to save", object()])
def test_unverified_or_explicit_failures_stay_failures(value):
    assert not plugin_result(value).ok


def test_structured_result_preserves_evidence_and_media():
    value = ToolResult("declined", ok=False, evidence=[{"repaired": False}])
    assert plugin_result(value) is value
    assert plugin_result("legacy calculation result").ok


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dispatch_preserves_plugin_failure_and_records_it(asynchronous):
    from orion_core.dispatcher import OrionDispatcher
    expected = ToolResult("write rejected", ok=False, evidence=[{"persisted": False}])
    async def async_tool():
        return expected
    calls = []
    worker = SimpleNamespace(handler_table=lambda: {},
        _tool_handlers={"sample": async_tool if asynchronous else lambda: expected},
        plugins=SimpleNamespace(record_call=lambda name: None,
            mark_loaded=lambda *args: calls.append(args)))
    result = asyncio.run(OrionDispatcher.dispatch(worker, "sample", {}))
    # Dispatch adds a host-owned observation without mutating plugin results.
    assert result is not expected
    assert (result.text, result.ok, result.evidence, result.receipt) == (
        expected.text, False, expected.evidence, expected.receipt)
    assert expected.execution is None
    assert result.execution["state"] == "returned"
    assert result.execution["executor_ok"] is False
    assert result.execution["tool"] == "sample"
    assert calls == [("sample", True, "write rejected")]


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "outcomes.db"
    monkeypatch.setattr(operations, "STORE_PATH", path)
    return path


def test_goal_is_persisted_and_updated_without_duplicate(store):
    assert operations.save_goal("Walk", "2027-01-01", 20).ok
    result = operations.save_goal("Walk", "2027-01-01", 100, "Completed")
    assert result.ok and result.evidence[0]["persisted"]
    with sqlite3.connect(store) as db:
        rows = db.execute("SELECT payload FROM records").fetchall()
    assert len(rows) == 1 and '"progress": 100' in rows[0][0]


@pytest.mark.parametrize("progress", [True, -1, 101, 1.2])
def test_goal_rejects_invalid_progress(store, progress):
    assert not operations.save_goal("Walk", "2027-01-01", progress).ok
    assert not store.exists()


def test_goal_does_not_claim_completion_for_inconsistent_input(store):
    assert not operations.save_goal("Walk", "not a date").ok
    assert not operations.save_goal("Walk", "2027-01-01", 20, "Completed").ok


def test_persistence_failure_does_not_report_success(store):
    store.mkdir()
    assert not operations.save_goal("Walk", "2027-01-01").ok
    assert not operations.record_error("example", "ExampleError").ok
    assert not operations.save_tool_spec("sample", "a plan").ok


def test_records_distinguish_requested_actions_from_performed_actions(store):
    draft = operations.save_tool_spec("sample", "a plan")
    error = operations.record_error("sample error", "ExampleError")
    report = operations.save_progress("example", [{"minutes": 10}], ["tomorrow"], {})
    assert all(result.ok for result in [draft, error, report])
    assert draft.evidence[0]["created_tool"] is False
    assert error.evidence[0]["repaired"] is False
    assert report.evidence[0]["reminders_scheduled"] == 0
    with sqlite3.connect(store) as db:
        assert db.execute("SELECT count(*) FROM records").fetchone()[0] == 3


def test_emotional_reading_depends_on_text_and_has_no_claim_of_certainty():
    positive = operations.analyse_text("Congratulations, excellent work!", 0, 1)
    worried = operations.analyse_text("I am worried, stressed and anxious", 0, 1)
    assert positive.ok and worried.ok
    assert positive.evidence[0]["reading"]["sentiment"] != worried.evidence[0]["reading"]["sentiment"]
    assert not positive.evidence[0]["verified_emotion"]
    assert not operations.analyse_text("hello", float("nan"), 1).ok


def _invocation_host(handler, *, registry=None, parameters=None, vault=None):
    return SimpleNamespace(_tool_handlers={"sample": handler}, plugins=registry,
        plugin_vault=vault, TOOL_DECLARATIONS=[] if parameters is None else [{
            "name": "sample", "description": "Synthetic fixture", "parameters": parameters}])


async def _wait_retired(host):
    async def drain():
        while host._plugin_invocations:
            await asyncio.sleep(0.001)
    await asyncio.wait_for(drain(), 1)


@pytest.mark.parametrize("cancel_caller", [False, True], ids=["queued_timeout", "queued_caller_cancel"])
def test_queued_plugin_never_starts_after_abandonment_and_retires_ownership(monkeypatch, cancel_caller):
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        calls, queue = [], []
        enqueued = asyncio.Event()
        hold = True
        async def executor(function, *args, **kwargs):
            assert function.__name__ == "physical_call"
            if not hold:
                return function(*args, **kwargs)
            future = asyncio.get_running_loop().create_future()
            queue.append((function, args, kwargs, future))
            enqueued.set()
            return await future
        monkeypatch.setattr(asyncio, "to_thread", executor)
        host = _invocation_host(lambda: calls.append(1) or ToolResult("synthetic action"))
        owner = asyncio.create_task(OrionDispatcher.invoke_plugin(host, "sample", {},
            timeout_s=.5 if cancel_caller else .03))
        await asyncio.wait_for(enqueued.wait(), .5)
        if cancel_caller:
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            result = await asyncio.wait_for(owner, .5)
            assert not result.ok and result.receipt["status"] == "uncertain"
            assert result.receipt["attempted"] is False
        assert calls == [] and len(queue) == 1
        pending = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5)
        assert pending.receipt["status"] == "pending" and len(queue) == 1
        function, args, kwargs, future = queue[0]
        assert not future.done()
        # Deterministically release the fake executor only after abandonment.
        future.set_result(function(*args, **kwargs))
        await _wait_retired(host)
        assert calls == []
        hold = False
        explicit = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5)
        assert explicit.ok and calls == [1]
        await _wait_retired(host)
    asyncio.run(scenario())


def test_plugin_executor_submission_failure_releases_unclaimed_ownership(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        calls, fail = [], True
        async def executor(function, *args, **kwargs):
            if fail:
                raise RuntimeError("synthetic closed executor")
            return function(*args, **kwargs)
        monkeypatch.setattr(asyncio, "to_thread", executor)
        host = _invocation_host(lambda: calls.append(1) or ToolResult("synthetic action"))
        result = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5)
        assert not result.ok and result.receipt["attempted"] is False and calls == []
        await _wait_retired(host)
        fail = False
        explicit = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5)
        assert explicit.ok and calls == [1]
        await _wait_retired(host)
    asyncio.run(scenario())


def test_plugin_async_execution_is_inside_the_shared_deadline():
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        cancelled = asyncio.Event()
        calls = []
        async def plugin():
            calls.append(1)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        host = _invocation_host(plugin)
        result = await asyncio.wait_for(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.02), .5)
        assert not result.ok and result.receipt["status"] == "uncertain"
        assert result.receipt["attempted"] is True and calls == [1]
        await asyncio.wait_for(cancelled.wait(), .5)
        await _wait_retired(host)
    asyncio.run(scenario())


def _token_plugin_host(tmp_path, monkeypatch, observer, *, secrets=()):
    """Real metadata, loader identity and human guard; only transport is fake."""
    import json
    import urllib.request
    from orion_core import plugin_registry, tasks
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    from orion_core.system_guard import SystemActionGuard
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "1")
    monkeypatch.delenv("FIXTURE_PLUGIN_TOKEN", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("No real HTTP is allowed"))
    monkeypatch.setattr(plugin_registry, "PLUGIN_STATE_PATH", tmp_path / "plugins.json")
    directory = tmp_path / "custom_tools"
    directory.mkdir()
    schema = {"name": "sample", "description": "Synthetic registered transport.", "parameters": {
        "type": "object", "properties": {key: {"type": "string"} for key in ("action", "to", "message")},
        "required": ["action", "to", "message"], "additionalProperties": False}}
    path = directory / "sample_tool.py"
    path.write_text("def get_tool_schema():\n    return " + repr(schema)
        + "\ndef run(action='', to='', message=''):\n    return _observe(action, to, message)\n", encoding="utf-8")
    manifest = directory / "sample.plugin.json"
    manifest.write_text(json.dumps({**schema, "module": path.name, "tier": "confirm",
        "permissions": ["network"], "requires": [], "secrets": list(secrets)}), encoding="utf-8")
    queued, emitted, logs = [], [], []
    bus = SimpleNamespace(log=SimpleNamespace(emit=logs.append),
                          confirm_action=SimpleNamespace(emit=emitted.append))
    loaded = ReflectiveModuleLoader(bus)._load_module_file(path)
    loaded._observe = observer
    host = OrionDispatcher.__new__(OrionDispatcher)
    host.plugins = plugin_registry.PluginRegistry(bus, directory)
    host._tool_handlers = {"sample": loaded.run}
    host.TOOL_DECLARATIONS = [schema]
    host.system_guard = SystemActionGuard(machine="synthetic host")
    host.bus, host.plugin_vault = bus, None
    def spawn(coro, **_kwargs):
        task = asyncio.create_task(coro)
        queued.append(task)
        return task
    monkeypatch.setattr(tasks, "spawn", spawn)
    return host, queued, emitted, path, manifest


def test_real_token_approved_plugin_runs_exactly_once_with_vault_and_without_input(tmp_path, monkeypatch):
    from orion_core.control import AutonomousControlLayer
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.execution_policy import foreground_authorized, is_background
    from orion_core.plugin_runner import scoped_plugin_secret
    seen, os_calls = [], []
    control = AutonomousControlLayer.__new__(AutonomousControlLayer)
    control.enabled = True
    control._pg = SimpleNamespace(typewrite=lambda *_a, **_k: os_calls.append("typewrite"))
    control._kb = SimpleNamespace(write=lambda *_a, **_k: os_calls.append("write"))
    control._narrate_typing = control._record = lambda *_: None
    def observe(action, to, message):
        refused = control.type_text("synthetic input")
        seen.append((action, to, message, is_background(), foreground_authorized(),
                     scoped_plugin_secret("sample", "FIXTURE_PLUGIN_TOKEN"), refused))
        return ToolResult("Synthetic API accepted", receipt={"kind": "messaging", "status": "submitted",
                                                            "attempted": True, "sent": True})
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch, observe,
                                                    secrets=("FIXTURE_PLUGIN_TOKEN",))
    requested = []
    host.plugin_vault = SimpleNamespace(secrets_for=lambda name, keys:
        requested.append((name, keys)) or {"FIXTURE_PLUGIN_TOKEN": "fixture-only-token"})
    async def scenario():
        body = "  synthetic exact body\nUnicode π  "
        arguments = {"action": "send", "to": "Fixture Recipient", "message": body, "confirm": True}
        pending = await OrionDispatcher.invoke_plugin(host, "sample", arguments, timeout_s=.5)
        assert pending.receipt["status"] == "awaiting_confirmation" and not seen and not requested
        assert emitted[0]["arguments"] == {"action": "send", "to": "Fixture Recipient", "message": body}
        arguments["message"] = "changed after token binding"
        token = emitted[0]["token"]
        release = host.confirm_system_action(token)
        assert release.ok and release.receipt["status"] == "queued"
        result = await asyncio.wait_for(queued[0], 1)
        assert result.ok and result.receipt["status"] == "submitted"
        assert not host.confirm_system_action(token).ok and len(queued) == 1
        assert len(seen) == 1 and seen[0][:6] == (
            "send", "Fixture Recipient", body, True, False, "fixture-only-token")
        assert not seen[0][6].ok and seen[0][6].receipt["status"] == "requires_foreground"
        assert os_calls == [] and host.plugins.get("sample").calls == 1
        assert requested == [("sample", ("FIXTURE_PLUGIN_TOKEN",))]
        await _wait_retired(host)
    asyncio.run(scenario())


@pytest.mark.parametrize("change, status", [("disabled", "disabled"), ("forbidden", "forbidden"),
    ("vault_failure", "unavailable"), ("source_changed", "needs_isolation"),
    ("source_reloaded", "requires_confirmation"),
    ("input_permission", "requires_foreground")])
def test_approved_plugin_still_obeys_live_registry_vault_and_effect_checks(tmp_path, monkeypatch, change, status):
    import json
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    host, queued, emitted, path, manifest = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"), secrets=("FIXTURE_PLUGIN_TOKEN",))
    async def scenario():
        pending = await OrionDispatcher.invoke_plugin(host, "sample",
            {"action": "send", "to": "fixture", "message": "synthetic", "confirm": True}, timeout_s=.5)
        assert pending.receipt["status"] == "awaiting_confirmation"
        if change == "disabled":
            assert host.plugins.set_enabled("sample", False).ok
        elif change in {"forbidden", "input_permission"}:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if change == "forbidden":
                metadata["tier"] = "forbid"
            else:
                metadata["permissions"].append("input")
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
        elif change in {"source_changed", "source_reloaded"}:
            path.write_text(path.read_text(encoding="utf-8") + "\n# different generation\n", encoding="utf-8")
            if change == "source_reloaded":
                from orion_core.dynamic_loader import ReflectiveModuleLoader
                observer = host._tool_handlers["sample"].__globals__["_observe"]
                reloaded = ReflectiveModuleLoader(host.bus)._load_module_file(path)
                reloaded._observe = observer
                host._tool_handlers["sample"] = reloaded.run
        else:
            def unavailable(*_):
                raise RuntimeError("private fixture vault detail")
            host.plugin_vault = SimpleNamespace(secrets_for=unavailable)
        assert host.confirm_system_action(emitted[0]["token"]).ok
        result = await asyncio.wait_for(queued[0], 1)
        assert not result.ok and result.receipt["status"] == status
        assert result.receipt["attempted"] is False and result.receipt["sent"] is False
        assert calls == [] and host.plugins.get("sample").calls == 0
        assert "private fixture vault detail" not in result.text
        await _wait_retired(host)
    asyncio.run(scenario())


def test_approved_plugin_retains_central_deadline_and_has_no_replay(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        calls, cancelled = [], asyncio.Event()
        async def hang(*_):
            calls.append(1)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch, hang)
        pending = await OrionDispatcher.invoke_plugin(host, "sample",
            {"action": "send", "to": "fixture", "message": "synthetic", "confirm": True}, timeout_s=.5)
        assert pending.receipt["status"] == "awaiting_confirmation"
        monkeypatch.setenv("ORION_FORGED_TOOL_TIMEOUT_S", ".05")
        token = emitted[0]["token"]
        assert host.confirm_system_action(token).ok
        result = await asyncio.wait_for(queued[0], 1)
        assert not result.ok and result.receipt["status"] == "uncertain"
        assert result.receipt["attempted"] is True
        await asyncio.wait_for(cancelled.wait(), .5)
        await _wait_retired(host)
        assert calls == [1] and not host.confirm_system_action(token).ok and len(queued) == 1
    asyncio.run(scenario())


def test_model_flags_and_direct_payloads_cannot_mint_a_plugin_approval(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"))
    async def scenario():
        payload = {"plugin": "sample", "arguments": {"action": "send", "to": "fixture", "message": "synthetic"}}
        direct = host._run_confirmed_plugin(payload)
        assert not direct.ok and direct.receipt["status"] == "requires_confirmation"
        missing = await OrionDispatcher.invoke_plugin(host, "sample", payload["arguments"], timeout_s=.5)
        assert missing.receipt == {"kind": "plugin", "status": "requires_confirmation",
                                   "attempted": False, "sent": False}
        invalid = await OrionDispatcher.invoke_plugin(host, "sample",
            {**payload["arguments"], "confirm": True, "_approval": True}, timeout_s=.5)
        assert not invalid.ok and invalid.receipt["status"] == "invalid_arguments"
        assert calls == [] and queued == [] and emitted == [] and host.system_guard.pending_count() == 0
    asyncio.run(scenario())


def test_plugin_confirmation_emit_failure_cancels_the_unshown_real_token(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"))
    def unavailable(_payload):
        raise RuntimeError("synthetic missing confirmation surface")
    host.bus.confirm_action.emit = unavailable
    async def scenario():
        result = await OrionDispatcher.invoke_plugin(host, "sample",
            {"action": "send", "to": "fixture", "message": "synthetic", "confirm": True}, timeout_s=.5)
        assert result.receipt == {"kind": "plugin", "status": "unavailable", "attempted": False, "sent": False}
        assert calls == [] and queued == [] and emitted == [] and host.system_guard.pending_count() == 0
    asyncio.run(scenario())


def test_expired_real_plugin_approval_never_schedules_execution(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    calls, clock = [], [0.0]
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"))
    host.system_guard = SystemActionGuard(machine="synthetic host", ttl_s=1, clock=lambda: clock[0])
    async def scenario():
        pending = await OrionDispatcher.invoke_plugin(host, "sample",
            {"action": "send", "to": "fixture", "message": "synthetic", "confirm": True}, timeout_s=.5)
        assert pending.receipt["status"] == "awaiting_confirmation"
        clock[0] = 2.0
        assert not host.confirm_system_action(emitted[0]["token"]).ok
        assert calls == [] and queued == [] and host.system_guard.pending_count() == 0
    asyncio.run(scenario())


def test_identical_pending_plugin_retries_reuse_one_real_token_and_emit(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"))
    async def scenario():
        arguments = {"action": "send", "to": "fixture", "message": "  same\nexact body  ", "confirm": True}
        for _ in range(3):
            result = await OrionDispatcher.invoke_plugin(host, "sample", dict(arguments), timeout_s=.5)
            assert result.receipt == {"kind": "plugin", "status": "awaiting_confirmation",
                                      "attempted": False, "sent": False}
        assert len(emitted) == host.system_guard.pending_count() == 1
        first = emitted[0]["token"]
        assert host.system_guard.confirm(first).kind.value == "execute"
        again = await OrionDispatcher.invoke_plugin(host, "sample", arguments, timeout_s=.5)
        assert again.receipt["status"] == "awaiting_confirmation"
        assert len(emitted) == 2 and emitted[1]["token"] != first
        assert host.system_guard.pending_count() == 1 and calls == [] and queued == []
        host.system_guard.cancel(emitted[1]["token"])
    asyncio.run(scenario())


def test_different_pending_plugin_payloads_stay_separate_and_expiry_arms_fresh_token(tmp_path, monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    calls, clock = [], [0.0]
    host, queued, emitted, _, _ = _token_plugin_host(tmp_path, monkeypatch,
        lambda *_: calls.append(1) or ToolResult("should not run"))
    host.system_guard = SystemActionGuard(machine="synthetic host", ttl_s=1, clock=lambda: clock[0])
    async def scenario():
        base = {"action": "send", "to": "fixture", "message": "first body", "confirm": True}
        for arguments in (base, {**base, "message": "different body"}):
            result = await OrionDispatcher.invoke_plugin(host, "sample", arguments, timeout_s=.5)
            assert result.receipt["status"] == "awaiting_confirmation"
        assert host.system_guard.pending_count() == len(emitted) == 2
        old = {row["token"] for row in emitted}
        assert len(old) == 2 and calls == [] and queued == []
        clock[0] = 2.0
        result = await OrionDispatcher.invoke_plugin(host, "sample", dict(base), timeout_s=.5)
        assert result.receipt["status"] == "awaiting_confirmation"
        assert host.system_guard.pending_count() == 1 and len(emitted) == 3
        assert emitted[-1]["token"] not in old
        assert not old.intersection(host.system_guard._pending)
        host.system_guard.cancel(emitted[-1]["token"])
    asyncio.run(scenario())


def test_plugin_timeout_does_not_wait_for_cancellation_suppression_or_start_duplicate():
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        release = asyncio.Event()
        calls = []
        async def plugin():
            calls.append(1)
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return ToolResult("late success")
        host = _invocation_host(plugin)
        try:
            first = await asyncio.wait_for(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.02), .5)
            pending = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.02)
            assert not first.ok and not pending.ok
            assert pending.receipt["status"] == "pending" and calls == [1]
        finally:
            release.set()
            await _wait_retired(host)
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_plugin_physical_worker_retains_ownership_until_late_finish(cancel_caller):
    import threading
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        started, release = threading.Event(), threading.Event()
        calls = []
        def plugin():
            calls.append(1)
            started.set()
            assert release.wait(1), "bounded fixture was not released"
            return ToolResult("late success")
        host = _invocation_host(plugin)
        owner = asyncio.create_task(OrionDispatcher.invoke_plugin(host, "sample", {},
                                                                  timeout_s=.5 if cancel_caller else .03))
        try:
            async def wait_started():
                while not started.is_set() and not owner.done():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_started(), .5)
            assert started.is_set()
            if cancel_caller:
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
            else:
                result = await asyncio.wait_for(owner, .5)
                assert not result.ok and result.receipt["status"] == "uncertain"
            pending = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.02)
            assert pending.receipt["status"] == "pending" and calls == [1]
        finally:
            release.set()
            await _wait_retired(host)
        assert calls == [1]  # No automatic retry, including after a late success.
        explicit = await OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5)
        assert explicit.ok and calls == [1, 1]
    asyncio.run(scenario())


def test_plugin_sync_factory_awaitable_is_inside_deadline():
    from orion_core.dispatcher import OrionDispatcher
    async def scenario():
        cancelled = asyncio.Event()
        async def outcome():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        host = _invocation_host(outcome)
        # A synchronous callable may return a coroutine too.
        host._tool_handlers["sample"] = lambda: outcome()
        result = await asyncio.wait_for(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.03), .5)
        assert not result.ok and result.receipt["status"] == "uncertain"
        await asyncio.wait_for(cancelled.wait(), .5)
        await _wait_retired(host)
    asyncio.run(scenario())


def test_plugin_gate_holds_disabled_handler_before_execution():
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    record = SimpleNamespace(tier="allow", permissions=(), undeclared=(), secrets=())
    registry = SimpleNamespace(is_enabled=lambda _: False, get=lambda _: record)
    host = _invocation_host(lambda: calls.append(1), registry=registry)
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert not result.ok and "disabled" in result.text and not calls


def test_plugin_false_confirmation_cannot_become_truthy(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "1")
    record = SimpleNamespace(tier="confirm", permissions=("network",), undeclared=(), secrets=())
    registry = SimpleNamespace(is_enabled=lambda _: True, get=lambda _: record)
    host = _invocation_host(lambda: calls.append(1), registry=registry)
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {"confirm": "false"}, timeout_s=.5))
    assert not result.ok and "Tell the user" in result.text and not calls


@pytest.mark.parametrize("arguments", [{}, {"flag": "maybe"}, {"flag": True, "extra": 1}, {"flag": []}])
def test_plugin_schema_and_signature_reject_before_call(arguments):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    def plugin(flag):
        calls.append(flag)
        return ToolResult("called")
    host = _invocation_host(plugin, parameters={"type": "object", "properties": {
        "flag": {"type": "boolean"}}, "required": ["flag"]})
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", arguments, timeout_s=.5))
    assert not result.ok and not calls


def test_plugin_vault_is_scoped_redacted_and_registry_scan_is_off_loop(monkeypatch):
    import threading
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.plugin_runner import scoped_plugin_secret
    owner = threading.get_ident()
    scans, requested, observed = [], [], []
    record = SimpleNamespace(tier="allow", permissions=(), undeclared=(), secrets=("SAMPLE_TOKEN",))
    def get_record(name):
        scans.append(threading.get_ident())
        return record
    def secrets(name, keys):
        requested.append((name, keys))
        return {"SAMPLE_TOKEN": "fixture-secret", "OTHER_TOKEN": "not-declared"}
    def plugin():
        observed.append((scoped_plugin_secret("sample", "SAMPLE_TOKEN"),
                         scoped_plugin_secret("sample", "OTHER_TOKEN"),
                         scoped_plugin_secret("other", "SAMPLE_TOKEN")))
        return ToolResult("fixture-secret", receipt={"nested": ["fixture-secret"]})
    monkeypatch.delenv("SAMPLE_TOKEN", raising=False)
    registry = SimpleNamespace(is_enabled=lambda _: True, get=get_record,
                               record_call=lambda _: None, mark_loaded=lambda *_: None)
    host = _invocation_host(plugin, registry=registry, vault=SimpleNamespace(secrets_for=secrets))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert result.ok and result.text == "[redacted]" and result.receipt == {"nested": ["[redacted]"]}
    assert scans and all(ident != owner for ident in scans)
    assert requested == [("sample", ("SAMPLE_TOKEN",))]
    assert observed == [("fixture-secret", "", "")]
    assert scoped_plugin_secret("sample", "SAMPLE_TOKEN") is None


def test_plugin_rechecks_disable_after_awaited_vault_read():
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    enabled = [True]
    record = SimpleNamespace(tier="allow", permissions=(), undeclared=(), secrets=("TOKEN",))
    def secrets(*_):
        enabled[0] = False
        return {}
    registry = SimpleNamespace(is_enabled=lambda _: enabled[0], get=lambda _: record)
    host = _invocation_host(lambda: calls.append(1), registry=registry, vault=SimpleNamespace(secrets_for=secrets))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert not result.ok and "disabled before execution" in result.text and not calls


def test_plugin_task_scope_carries_no_foreground_input_permission():
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.execution_policy import is_background, require_foreground
    seen = []
    def plugin():
        seen.append(is_background())
        return require_foreground("keyboard input")
    host = _invocation_host(plugin)
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert seen == [True] and not result.ok
    assert result.receipt["status"] == "requires_foreground"


def test_plugin_forbid_survives_confirmation_prompt_override(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "0")
    calls = []
    record = SimpleNamespace(tier="forbid", permissions=(), undeclared=(), secrets=())
    host = _invocation_host(lambda: calls.append(1), registry=SimpleNamespace(
        is_enabled=lambda _: True, get=lambda _: record))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert not result.ok and "forbid" in result.text and not calls


def test_nested_scheduled_plugin_scope_cannot_fall_back_to_environment(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.plugin_runner import plugin_secret_scope, scoped_plugin_secret
    monkeypatch.setenv("TOKEN", "undeclared-process-fallback")
    record = SimpleNamespace(tier="allow", permissions=(), undeclared=(), secrets=("TOKEN",))
    host = _invocation_host(lambda: ToolResult(scoped_plugin_secret("sample", "TOKEN")),
        registry=SimpleNamespace(is_enabled=lambda _: True, get=lambda _: record))
    with plugin_secret_scope("sample", {}):
        result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert result.ok and result.text == ""


def test_plugin_readiness_uses_only_declared_vault_keys_without_execution(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    monkeypatch.delenv("TOKEN", raising=False)
    calls, keys = [], []
    record = SimpleNamespace(secrets=("TOKEN",))
    def secrets(name, requested):
        keys.append((name, requested))
        return {"TOKEN": "fixture-secret", "OTHER": "not-declared"}
    host = _invocation_host(lambda: calls.append(1), registry=SimpleNamespace(get=lambda _: record),
                            vault=SimpleNamespace(secrets_for=secrets))
    assert asyncio.run(OrionDispatcher.plugin_secret_available(host, "sample")) is True
    assert keys == [("sample", ("TOKEN",))] and not calls


def test_plugin_readiness_error_cannot_masquerade_as_absent_credentials(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    monkeypatch.delenv("TOKEN", raising=False)
    def broken(*_):
        raise ValueError("credential-bearing-internal-detail")
    host = _invocation_host(lambda: None, registry=SimpleNamespace(get=lambda _: SimpleNamespace(secrets=("TOKEN",))),
                            vault=SimpleNamespace(secrets_for=broken))
    with pytest.raises(RuntimeError, match="readiness could not be read") as error:
        asyncio.run(OrionDispatcher.plugin_secret_available(host, "sample"))
    assert "credential-bearing" not in str(error.value)


@pytest.mark.parametrize("effect", ["input", "subprocess"])
def test_plugin_raw_desktop_process_effect_is_deferred_before_handler(effect):
    from orion_core.dispatcher import OrionDispatcher
    calls = []
    record = SimpleNamespace(tier="allow", permissions=(effect,), undeclared=(), secrets=())
    host = _invocation_host(lambda: calls.append(1), registry=SimpleNamespace(
        get=lambda _: record, is_enabled=lambda _: True))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert not result.ok and result.receipt["status"] == "requires_foreground"
    assert not calls


def test_plugin_effect_guard_preserves_explicit_host_foreground_control():
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.execution_policy import foreground_scope
    calls = []
    record = SimpleNamespace(tier="allow", permissions=("input",), undeclared=(), secrets=())
    host = _invocation_host(lambda: calls.append(1) or ToolResult("fixture action"), registry=SimpleNamespace(
        get=lambda _: record, is_enabled=lambda _: True))
    with foreground_scope("explicit-user-control"):
        result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert result.ok and calls == [1]


def test_plugin_effect_metadata_is_bound_to_compiled_source_identity(tmp_path):
    import hashlib
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    source = b'def run():\n    return "fixture-only"\n'
    path = tmp_path / "sample_tool.py"
    path.write_bytes(source)
    loader = ReflectiveModuleLoader(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    module = loader._load_module_file(path)
    assert module.run._orion_plugin_source_sha256 == hashlib.sha256(source).hexdigest()
    record = SimpleNamespace(tier="allow", permissions=(), undeclared=(), secrets=(), source_sha256="different-generation")
    host = _invocation_host(module.run, registry=SimpleNamespace(get=lambda _: record, is_enabled=lambda _: True))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {}, timeout_s=.5))
    assert not result.ok and result.receipt["status"] == "needs_isolation"
    assert result.receipt["attempted"] is False


def test_background_plugin_confirmation_queues_only_a_bound_nonmodal_request(monkeypatch):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.system_guard import SystemActionGuard
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "1")
    emitted = []
    record = SimpleNamespace(tier="confirm", permissions=("network",), undeclared=(), secrets=())
    host = _invocation_host(lambda: pytest.fail("Background modal route reached handler"), registry=SimpleNamespace(
        get=lambda _: record, is_enabled=lambda _: True))
    host.system_guard = SystemActionGuard(machine="synthetic host")
    host.bus = SimpleNamespace(confirm_action=SimpleNamespace(emit=emitted.append))
    result = asyncio.run(OrionDispatcher.invoke_plugin(host, "sample", {"confirm": True}, timeout_s=.5))
    assert not result.ok and result.receipt == {"kind": "plugin", "status": "awaiting_confirmation",
                                                "attempted": False, "sent": False}
    assert len(emitted) == 1 and emitted[0]["background"] is True
    assert emitted[0]["intent"] == "plugin_action" and emitted[0]["arguments"] == {}
    assert host.system_guard.pending_count() == 1
    host.system_guard.cancel(emitted[0]["token"])


async def _wait_game_workers_retired(runner):
    async def drain():
        while runner._game_workers:
            await asyncio.sleep(.001)
    await asyncio.wait_for(drain(), 1)


@pytest.mark.parametrize("cancel_caller", [False, True], ids=["queued_timeout", "queued_caller_cancel"])
def test_reviewed_game_adapter_queued_abandonment_never_starts_late(monkeypatch, cancel_caller):
    import threading
    import orion_core.jv_plugins as package
    from orion_core.jv_plugins import runner
    from orion_core.execution_policy import foreground_scope
    calls = []
    monkeypatch.setattr(package, "legacy_game_updater", SimpleNamespace(
        game_updater=lambda *_a, **_k: calls.append(1) or "Fixture launcher result"), raising=False)
    monkeypatch.setattr(runner, "_game_worker_lock", threading.Lock())
    monkeypatch.setattr(runner, "_game_workers", set())
    monkeypatch.setattr(runner, "_locks", {})
    monkeypatch.setattr(runner, "GAME_TIMEOUT_S", .5 if cancel_caller else .03)
    async def scenario():
        queue, hold = [], True
        enqueued = asyncio.Event()
        async def executor(function, *args, **kwargs):
            if not hold:
                return function(*args, **kwargs)
            future = asyncio.get_running_loop().create_future()
            queue.append((function, args, kwargs, future))
            enqueued.set()
            return await future
        monkeypatch.setattr(asyncio, "to_thread", executor)
        owner = asyncio.create_task(runner.execute("jv_game_updater", {"action": "list"}))
        await asyncio.wait_for(enqueued.wait(), .5)
        # Admission must be visible while the first wrapper is still queued.
        pending = await asyncio.wait_for(runner.execute("jv_game_updater", {"action": "list"}), .5)
        assert pending.receipt["status"] == "pending" and len(queue) == 1
        if cancel_caller:
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            result = await asyncio.wait_for(owner, .5)
            assert not result.ok and result.receipt == {
                "kind": "plugin", "status": "uncertain", "attempted": False}
        assert calls == [] and runner._game_worker_lock.locked()
        pending = await runner.execute("jv_game_updater", {"action": "list"})
        assert pending.receipt["status"] == "pending" and len(queue) == 1
        function, args, kwargs, future = queue[0]
        assert not future.done()  # The public caller never cancelled its worker.
        future.set_result(function(*args, **kwargs))
        await _wait_game_workers_retired(runner)
        assert calls == [] and not runner._game_worker_lock.locked()
        hold = False
        explicit = await runner.execute("jv_game_updater", {"action": "list"})
        assert explicit.ok and calls == [1]
        await _wait_game_workers_retired(runner)
    with foreground_scope("explicit-user-game-control"):
        asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["submission_rejected", "queued_worker_cancelled"])
def test_reviewed_game_adapter_stale_wrapper_cannot_release_a_new_owner(monkeypatch, failure):
    import threading
    import orion_core.jv_plugins as package
    from orion_core.jv_plugins import runner
    from orion_core.execution_policy import foreground_scope
    calls = []
    monkeypatch.setattr(package, "legacy_game_updater", SimpleNamespace(
        game_updater=lambda *_a, **_k: calls.append(1) or "Fixture launcher result"), raising=False)
    monkeypatch.setattr(runner, "_game_worker_lock", threading.Lock())
    monkeypatch.setattr(runner, "_game_workers", set())
    monkeypatch.setattr(runner, "_locks", {})
    monkeypatch.setattr(runner, "GAME_TIMEOUT_S", 1)
    async def scenario():
        queue = []
        enqueued = asyncio.Event()
        async def executor(function, *args, **kwargs):
            future = asyncio.get_running_loop().create_future()
            queue.append((function, args, kwargs, future))
            enqueued.set()
            if failure == "submission_rejected" and len(queue) == 1:
                raise RuntimeError("Synthetic executor rejection")
            return await future
        monkeypatch.setattr(asyncio, "to_thread", executor)
        old_owner = asyncio.create_task(runner.execute("jv_game_updater", {"action": "list"}))
        await asyncio.wait_for(enqueued.wait(), .5)
        if failure == "queued_worker_cancelled":
            worker, = runner._game_workers
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await old_owner
        else:
            assert not (await old_owner).ok
        await _wait_game_workers_retired(runner)
        assert not runner._game_worker_lock.locked() and calls == []
        enqueued.clear()
        new_owner = asyncio.create_task(runner.execute("jv_game_updater", {"action": "list"}))
        await asyncio.wait_for(enqueued.wait(), .5)
        assert len(queue) == 2 and runner._game_worker_lock.locked()
        old_function, args, kwargs, _old_future = queue[0]
        assert old_function(*args, **kwargs) is None
        # Its finally runs after a new owner reserved the same global Lock.
        assert runner._game_worker_lock.locked() and calls == []
        pending = await runner.execute("jv_game_updater", {"action": "list"})
        assert pending.receipt["status"] == "pending" and len(queue) == 2
        function, args, kwargs, future = queue[1]
        future.set_result(function(*args, **kwargs))
        assert (await asyncio.wait_for(new_owner, .5)).ok and calls == [1]
        await _wait_game_workers_retired(runner)
        assert not runner._game_worker_lock.locked()
    with foreground_scope("explicit-user-game-control"):
        asyncio.run(scenario())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_reviewed_game_adapter_holds_physical_worker_after_timeout_or_cancellation(monkeypatch, cancel_caller):
    import threading
    import orion_core.jv_plugins as package
    from orion_core.jv_plugins import runner
    from orion_core.execution_policy import foreground_scope
    started, release = threading.Event(), threading.Event()
    calls = []
    def game(arguments, player=None):
        calls.append(1)
        started.set()
        assert release.wait(1)
        return "Fixture launcher result"
    monkeypatch.setattr(package, "legacy_game_updater", SimpleNamespace(game_updater=game), raising=False)
    monkeypatch.setattr(runner, "_game_worker_lock", threading.Lock())
    monkeypatch.setattr(runner, "_game_workers", set())
    monkeypatch.setattr(runner, "_locks", {})
    monkeypatch.setattr(runner, "GAME_TIMEOUT_S", .5 if cancel_caller else .03)
    async def scenario():
        owner = asyncio.create_task(runner.execute("jv_game_updater", {"action": "list"}))
        try:
            async def wait_started():
                while not started.is_set() and not owner.done():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_started(), .5)
            assert started.is_set()
            if cancel_caller:
                owner.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await owner
            else:
                result = await asyncio.wait_for(owner, .5)
                assert not result.ok and result.receipt["status"] == "uncertain"
            pending = await asyncio.wait_for(runner.execute("jv_game_updater", {"action": "list"}), .5)
            assert not pending.ok and pending.receipt["status"] == "pending"
            assert calls == [1]
        finally:
            release.set()
            async def wait_worker():
                while runner._game_worker_lock.locked():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_worker(), 1)
            await _wait_game_workers_retired(runner)
        assert calls == [1]
    with foreground_scope("explicit-user-game-control"):
        asyncio.run(scenario())


@pytest.mark.parametrize("value", [ToolResult("bad", ok="false"), "x" * 131_073,
    {"ok": True, "result": "x" * 131_073}, ToolResult("x" * 131_073)],
    ids=["invalid_boolean", "oversized_string", "oversized_mapping", "oversized_tool_result"])
def test_plugin_outcome_boolean_and_text_limits_fail_closed(value):
    assert not plugin_result(value).ok


def test_plugin_mapping_receipt_is_preserved_and_bad_receipt_is_refused():
    value = {"ok": False, "result": "not configured", "receipt": {
        "kind": "messaging", "status": "not_configured", "sent": False, "attempted": False}}
    result = plugin_result(value)
    assert not result.ok and result.receipt == value["receipt"]
    value["receipt"]["sent"] = True
    assert result.receipt["sent"] is False
    assert not plugin_result({"ok": True, "result": "done", "receipt": []}).ok


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_real_messaging_dispatch_worker_returns_to_owner_loop_once(monkeypatch, cancel_caller):
    import threading
    from collections import deque
    import orion_core.dispatcher as dispatch_module
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.execution_policy import is_background
    from orion_core.messaging import MessagingGateway
    # These are the real dispatcher/mixin/gateway/plugin routes. Only the
    # registered transport handler, credential lookup and observations are fake.
    monkeypatch.setattr(dispatch_module, "_gap_ledger", lambda: SimpleNamespace(note_tool_call=lambda: None))
    monkeypatch.setattr(dispatch_module, "_note_gap", lambda *_: None)
    monkeypatch.setattr(dispatch_module, "_action_guard_on", lambda: False)
    monkeypatch.setenv("ORION_PLUGIN_CONFIRM", "0")
    async def scenario():
        owner = threading.get_ident()
        started, release = asyncio.Event(), asyncio.Event()
        seen = []
        async def transport(action="", to="", message=""):
            seen.append((threading.get_ident(), is_background(), action, to, message))
            started.set()
            await release.wait()
            return ToolResult("Synthetic transport accepted", receipt={"kind": "messaging",
                "status": "submitted", "attempted": True, "sent": True})
        bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
        host = OrionDispatcher.__new__(OrionDispatcher)
        host.bus, host.telemetry, host.active_tools = bus, None, 0
        host.recent_tools, host.swarm_view, host.control = deque(maxlen=60), None, None
        host.plugins, host.plugin_vault = None, None
        host._tool_handlers = {"discord_message": transport}
        host.TOOL_DECLARATIONS = [{"name": "discord_message", "parameters": {"type": "object",
            "properties": {key: {"type": "string"} for key in ("action", "to", "message")}}}]
        gateway = host.messaging = MessagingGateway(bus)
        monkeypatch.setattr(gateway, "delivery_suppressed", lambda: False)
        monkeypatch.setattr(gateway, "_bot_token", lambda _: "fixture-only")
        monkeypatch.setattr(gateway, "_discord_webhook", lambda: "")
        body = "  synthetic body\nkept exactly  "
        args = {"action": "send", "platform": "discord", "contact": "Fixture Recipient", "message": body}
        turn = asyncio.create_task(host.dispatch("messaging", args))
        try:
            await asyncio.wait_for(started.wait(), 1)
            if cancel_caller:
                turn.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await turn
                pending = await asyncio.wait_for(host.dispatch("messaging", args), .5)
                assert not pending.ok and pending.receipt["status"] == "pending"
                assert len(seen) == 1
            release.set()
            if not cancel_caller:
                result = await asyncio.wait_for(turn, 1)
                assert result.ok and result.receipt["status"] == "submitted"
            async def wait_worker():
                while gateway._delivery_lock.locked():
                    await asyncio.sleep(.001)
            await asyncio.wait_for(wait_worker(), 1)
            assert seen == [(owner, True, "send", "Fixture Recipient", body)]
        finally:
            release.set()
            if not turn.done():
                turn.cancel()
                await asyncio.gather(turn, return_exceptions=True)
            await _wait_retired(host)
    asyncio.run(scenario())
