"""Synthetic delivery and task-scoped secret regressions; no outbound traffic."""
import asyncio
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.messaging import MessagingGateway
from orion_core.plugin_runner import Entry, PluginScheduler


CANARY = "synthetic-vault-canary-xxxv"
KEY = "ORION_SYNTHETIC_PLUGIN_KEY"


class Signal:
    def __init__(self):
        self.lines = []

    def emit(self, value):
        self.lines.append(value)


def gateway(monkeypatch):
    gate = MessagingGateway(SimpleNamespace(log=Signal()))
    monkeypatch.setattr(gate, "delivery_suppressed", lambda: False)
    return gate


def entry():
    return Entry("synthetic", None, secrets=(KEY,))


def scheduler(dispatcher=None, bus=None):
    vault = SimpleNamespace(secrets_for=lambda name, declared: {KEY: CANARY})
    return PluginScheduler(dispatcher=dispatcher, bus=bus, vault=vault)


def registered_bot_send(gate, handler):
    """Exercise the real owner-loop executor; no import or transport bypass."""
    from orion_core.dispatch_web import WebDispatchMixin
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.plugin_registry import PluginRecord
    name = "telegram_message"
    record = PluginRecord(name, tier="allow", permissions=("network",),
                          secrets=(KEY,), module_exists=True)
    admitted = []
    def forbid_fallback(*args):
        pytest.fail("a configured plugin outcome must not retry through another transport")
    gate._desktop_sender = SimpleNamespace(PLATFORMS={"telegram"}, control=None, send=forbid_fallback)
    gate._browser_sender = SimpleNamespace(PLATFORMS={"telegram"}, send=forbid_fallback)
    host = SimpleNamespace(messaging=gate, _tool_handlers={name: handler},
        plugins=SimpleNamespace(get=lambda requested: record if requested == name else None,
            is_enabled=lambda requested: requested == name,
            record_call=lambda requested: admitted.append(requested), mark_loaded=lambda *args: None),
        plugin_vault=SimpleNamespace(secrets_for=lambda requested, declared: {KEY: CANARY}),
        TOOL_DECLARATIONS=[])
    async def invoke(requested, arguments, *, timeout_s=None):
        return await OrionDispatcher.invoke_plugin(host, requested, arguments, timeout_s=timeout_s)
    host.invoke_plugin = invoke
    async def scenario():
        result = await asyncio.wait_for(WebDispatchMixin.messaging_tool(host,
            {"action": "send", "platform": "telegram", "contact": "synthetic-recipient",
             "message": "Synthetic message"}), 2)
        assert admitted == [name]
        return result
    return asyncio.run(scenario())


def test_failed_bot_result_never_logs_sent(monkeypatch):
    gate = gateway(monkeypatch)
    monkeypatch.setattr(gate, "_send_via_bot", lambda *args: ToolResult("Quota exhausted", ok=False))
    result = gate.send_text("telegram", "synthetic-recipient", "Synthetic message")
    assert not result.ok
    assert not any(" sent to " in line for line in gate.bus.log.lines)
    assert any("failed" in line.lower() for line in gate.bus.log.lines)


@pytest.mark.parametrize("async_bot", [False, True])
def test_structured_failed_bot_outcome_survives_gateway(monkeypatch, async_bot):
    gate = gateway(monkeypatch)
    calls = []
    monkeypatch.setattr(gate, "_bot_token", lambda section: "synthetic-token")
    outcome = ToolResult("Quota exhausted", ok=False,
        receipt={"kind": "messaging", "status": "failed", "attempted": True, "sent": False})
    async def async_run(**kwargs):
        calls.append(kwargs)
        return outcome
    def sync_run(**kwargs):
        calls.append(kwargs)
        return outcome
    result = registered_bot_send(gate, async_run if async_bot else sync_run)
    assert not result.ok and result.text == "Configured Telegram bot API: Quota exhausted"
    assert result.receipt["status"] == "failed" and result.receipt["sent"] is False
    assert result.receipt["backend"] == "registered_plugin"
    assert result.receipt["sender_kind"] == "configured_bot"
    assert calls == [{"action": "send", "to": "synthetic-recipient", "message": "Synthetic message"}]
    assert outcome.text == "Quota exhausted" and "sender_kind" not in outcome.receipt
    assert not any(" sent to " in line for line in gate.bus.log.lines)


def test_bot_failure_redacts_configured_token(monkeypatch):
    gate = gateway(monkeypatch)
    calls = []
    monkeypatch.setattr(gate, "_bot_token", lambda section: CANARY)
    def run(**kwargs):
        calls.append(kwargs)
        raise OSError(f"Synthetic transport error for {CANARY}")
    result = registered_bot_send(gate, run)
    assert not result.ok and CANARY not in result.text
    assert len(calls) == 1 and "[redacted]" in result.text
    assert CANARY not in repr(gate.bus.log.lines)


@pytest.mark.parametrize("outcome", ["", "  ", None])
def test_missing_bot_outcome_cannot_claim_delivery(monkeypatch, outcome):
    gate = gateway(monkeypatch)
    calls = []
    monkeypatch.setattr(gate, "_bot_token", lambda section: "synthetic-token")
    def run(**kwargs):
        calls.append(kwargs)
        return outcome
    result = registered_bot_send(gate, run)
    assert not result.ok and result.receipt["sent"] is not True and len(calls) == 1
    assert not any(" sent to " in line for line in gate.bus.log.lines)


def test_async_bot_timeout_is_bounded_without_browser_retry(monkeypatch):
    import orion_core.messaging as module
    gate = gateway(monkeypatch)
    calls = []
    monkeypatch.setattr(gate, "_bot_token", lambda section: "synthetic-token")
    monkeypatch.setattr(module, "BOT_TIMEOUT_S", .05)
    monkeypatch.setattr(module, "open_url", lambda url: pytest.fail("delivery must not retry via browser"))
    async def run(**kwargs):
        calls.append(kwargs)
        await asyncio.sleep(10)
    result = registered_bot_send(gate, run)
    assert not result.ok and "time limit" in result.text and len(calls) == 1
    assert result.receipt["status"] == "uncertain" and result.receipt["attempted"] is True
    assert result.receipt["sent"] is not True and result.receipt["backend"] == "registered_plugin"


def test_unbound_bot_never_imports_or_starts_unregistered_async_work_on_event_loop(monkeypatch):
    gate = gateway(monkeypatch)
    calls = []
    monkeypatch.setattr(gate, "_bot_token", lambda section: "synthetic-token")
    async def run(**kwargs):
        calls.append(kwargs)
        return ToolResult("Synthetic accepted result")
    monkeypatch.setitem(sys.modules, "telegram_message_tool", SimpleNamespace(run=run))
    async def scenario():
        result = gate.send_text("telegram", "synthetic-recipient", "Synthetic message")
        assert not result.ok and calls == []
        assert result.receipt["status"] == "unavailable" and result.receipt["attempted"] is False
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["refused", "exception", "bad_port"])
def test_smtp_failure_never_claims_sent_or_leaks_password(monkeypatch, failure):
    gate = gateway(monkeypatch)
    monkeypatch.setattr(gate, "_email_account", lambda: {
        "address": "synthetic@example.test", "password": CANARY,
        "port": "invalid" if failure == "bad_port" else 587})
    class SMTP:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def starttls(self): pass
        def login(self, *args): pass
        def send_message(self, message):
            if failure == "exception":
                raise OSError(f"Synthetic rejection {CANARY}")
            return {"refused@example.test": (550, b"synthetic refusal")}
    monkeypatch.setattr("smtplib.SMTP", SMTP)
    result = gate.send_email("synthetic-recipient@example.test", "Synthetic", "Message")
    assert not result.ok
    assert "Email sent" not in result.text
    assert not any("email sent" in line for line in gate.bus.log.lines)
    assert CANARY not in result.text + repr(gate.bus.log.lines)


def test_scheduled_secret_never_enters_process_environment_or_other_thread(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    with scheduler()._with_secrets(entry()):
        with ThreadPoolExecutor(max_workers=1) as pool:
            observed = pool.submit(os.getenv, KEY).result(timeout=2)
        assert observed is None
        assert KEY not in os.environ
    assert KEY not in os.environ


def test_scheduled_scope_does_not_modify_existing_environment(monkeypatch):
    monkeypatch.setenv(KEY, "existing-public-value")
    with scheduler()._with_secrets(entry()):
        assert os.getenv(KEY) == "existing-public-value"
    assert os.getenv(KEY) == "existing-public-value"


def test_declared_secret_follows_worker_context_but_never_an_unrelated_task(monkeypatch):
    from orion_core.jv_plugins.runtime import secret
    monkeypatch.setenv(KEY, "different-global-account")
    async def scenario():
        released = asyncio.Event()
        async def unrelated():
            await released.wait()
            return secret("synthetic", KEY)
        other = asyncio.create_task(unrelated())
        with scheduler()._with_secrets(entry()):
            assert await asyncio.to_thread(secret, "synthetic", KEY) == CANARY
            assert secret("other-plugin", KEY) == ""
            assert secret("synthetic", "UNDECLARED") == ""
            released.set()
            assert await other == "different-global-account"
        assert secret("synthetic", KEY) == "different-global-account"
    asyncio.run(scenario())


def test_child_task_cannot_use_revoked_scope_or_fallback_global_secret(monkeypatch):
    from orion_core.jv_plugins.runtime import secret
    monkeypatch.setenv(KEY, "different-global-account")
    async def scenario():
        released = asyncio.Event()
        async def child():
            await released.wait()
            return secret("synthetic", KEY)
        with scheduler()._with_secrets(entry()):
            child_task = asyncio.create_task(child())
            assert secret("synthetic", KEY) == CANARY
        released.set()
        assert await child_task == ""
    asyncio.run(scenario())


def test_cancelled_scheduler_revokes_context_in_spawned_child(monkeypatch):
    from orion_core.jv_plugins.runtime import secret
    monkeypatch.setenv(KEY, "different-global-account")
    async def scenario():
        started, released = asyncio.Event(), asyncio.Event()
        children = []
        async def child():
            await released.wait()
            return secret("synthetic", KEY)
        class Dispatcher:
            async def dispatch(self, name, args):
                assert secret(name, KEY) == CANARY
                children.append(asyncio.create_task(child()))
                started.set()
                await asyncio.Future()
        runner = scheduler(Dispatcher())
        scheduled = entry()
        scheduled.failures = 2
        task = asyncio.create_task(runner._execute(scheduled))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        released.set()
        assert await children[0] == ""
        assert runner.history[-1].detail == "cancelled before a result; completion and side effects are unverified"
        assert runner.history[-1].status == "cancelled" and not runner.history[-1].ok
        assert scheduled.failures == 2 and not scheduled.muted
        assert scheduled.last is runner.history[-1]
        assert os.getenv(KEY) == "different-global-account"
    asyncio.run(scenario())


@pytest.mark.parametrize("fails", [False, True])
def test_scheduler_history_redacts_synthetic_secret_in_result_or_error(fails):
    bus = SimpleNamespace(log=Signal())
    class Dispatcher:
        async def dispatch(self, name, args):
            if fails:
                raise ValueError(CANARY)
            return ToolResult(f"Synthetic result {CANARY}", ok=False)
    runner = scheduler(Dispatcher(), bus)
    run = asyncio.run(runner._execute(entry()))
    assert not run.ok
    assert CANARY not in run.detail + repr(bus.log.lines) + repr(runner.history)
    assert "[redacted]" in run.detail


def test_unreadable_vault_never_dispatches_or_logs_exception_secret():
    bus = SimpleNamespace(log=Signal())
    calls = []
    class Vault:
        def secrets_for(self, *args): raise ValueError(CANARY)
    class Dispatcher:
        async def dispatch(self, *args):
            calls.append(args)
            return ToolResult("unexpected")
    runner = PluginScheduler(dispatcher=Dispatcher(), bus=bus, vault=Vault())
    run = asyncio.run(runner._execute(entry()))
    assert not run.ok and calls == []
    assert CANARY not in run.detail + repr(bus.log.lines)


@pytest.mark.parametrize("raises", [False, True])
def test_extension_redacts_copies_before_registry_health(raises):
    from orion_core.dispatcher import OrionDispatcher
    recorded = []
    original = ToolResult(CANARY, ok=False, receipt={"nested": [CANARY]},
                          evidence=[{"value": CANARY}], media={"label": CANARY, "data": b"synthetic bytes"})
    def plugin(**kwargs):
        if raises:
            raise ValueError(CANARY)
        return original
    host = SimpleNamespace(_tool_handlers={"synthetic": plugin},
        plugins=SimpleNamespace(record_call=lambda name: None,
                                mark_loaded=lambda *args: recorded.append(args)))
    async def scenario():
        with scheduler()._with_secrets(entry()):
            return await OrionDispatcher._dispatch_extension(host, "synthetic", {})
    result = asyncio.run(scenario())
    assert not result.ok and CANARY not in repr(result) + repr(recorded)
    assert original.text == CANARY and original.receipt["nested"] == [CANARY]
    if not raises:
        assert result.media["data"] == b"synthetic bytes"


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("secret", ["credential-canary-crossing-the-output-limit", "credential-canary\nsecond-line"])
def test_extension_redacts_complete_error_before_truncation(rollback, secret):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.plugin_runner import plugin_secret_scope
    recorded = []
    limit = 100 if rollback else 200
    prefix = "x" * (limit - 5)
    def plugin():
        if rollback:
            raise TypeError("synthetic runtime failure")
        raise ValueError(prefix + secret)
    async def restore(*args):
        raise ValueError(prefix + secret)
    host = SimpleNamespace(_tool_handlers={"synthetic": plugin},
        _forge_rollbacks={"synthetic": restore} if rollback else {},
        plugins=SimpleNamespace(record_call=lambda name: None,
                                mark_loaded=lambda *args: recorded.append(args)))
    async def scenario():
        with plugin_secret_scope("ancestor", {KEY: secret}):
            return await OrionDispatcher._dispatch_extension(host, "synthetic", {})
    result = asyncio.run(scenario())
    report = repr(result) + repr(recorded)
    assert not result.ok and recorded
    assert "crede" not in report
    assert prefix in result.text and "[red" in result.text


def test_revoked_nested_empty_scope_keeps_reporting_filters_without_credentials(monkeypatch):
    from orion_core.jv_plugins.runtime import secret
    from orion_core.plugin_runner import plugin_secret_scope, redact_plugin_result
    monkeypatch.setenv(KEY, "unrelated-global-account")
    original = ToolResult(CANARY, receipt={CANARY: [CANARY]}, evidence=[{"value": CANARY}],
                          execution={"detail": CANARY}, chain=[CANARY],
                          media={"label": CANARY, "data": b"synthetic bytes"})
    async def scenario():
        released = asyncio.Event()
        async def child():
            await released.wait()
            assert secret("inner", KEY) == secret("ancestor", KEY) == ""
            return redact_plugin_result(original)
        with plugin_secret_scope("ancestor", {KEY: CANARY}):
            with plugin_secret_scope("inner", {}):
                assert secret("inner", KEY) == secret("ancestor", KEY) == ""
                pending = asyncio.create_task(child())
        released.set()
        return await pending
    result = asyncio.run(scenario())
    assert CANARY not in repr(result) and "[redacted]" in repr(result)
    assert result.media["data"] == b"synthetic bytes"
    assert original.text == CANARY and original.receipt == {CANARY: [CANARY]}
    assert original.execution["detail"] == CANARY and original.chain == [CANARY]


@pytest.mark.parametrize("raises", [False, True])
def test_nested_scheduler_history_redacts_ancestor_without_declared_keys(raises):
    from orion_core.jv_plugins.runtime import secret
    from orion_core.plugin_runner import plugin_secret_scope
    bus = SimpleNamespace(log=Signal())
    class Dispatcher:
        async def dispatch(self, name, args):
            assert secret(name, KEY) == secret("ancestor", KEY) == ""
            if raises:
                raise ValueError(CANARY)
            return ToolResult(CANARY, ok=False)
    runner = PluginScheduler(dispatcher=Dispatcher(), bus=bus)
    async def scenario():
        with plugin_secret_scope("ancestor", {KEY: CANARY}):
            return await runner._execute(Entry("inner", None))
    run = asyncio.run(scenario())
    assert not run.ok and CANARY not in repr(run) + repr(runner.history) + repr(bus.log.lines)
    assert "[redacted]" in run.detail


@pytest.mark.parametrize("asynchronous", [False, True])
def test_late_plugin_error_redacts_revoked_ancestor_without_secret_access(monkeypatch, asynchronous):
    from orion_core.dispatcher import OrionDispatcher
    from orion_core.jv_plugins.runtime import secret
    from orion_core.plugin_runner import plugin_secret_scope, redact_plugin_secrets
    monkeypatch.setenv(KEY, "unrelated-global-account")
    recorded, observed = [], []
    async def scenario():
        started_async, release_async = asyncio.Event(), asyncio.Event()
        started_sync, release_sync = threading.Event(), threading.Event()
        def inspect_scope():
            observed.append((secret("synthetic", KEY), secret("ancestor", KEY),
                             redact_plugin_secrets(CANARY)))
        async def async_plugin():
            inspect_scope()
            started_async.set()
            await release_async.wait()
            inspect_scope()
            raise ValueError(CANARY)
        def sync_plugin():
            inspect_scope()
            started_sync.set()
            assert release_sync.wait(2)
            inspect_scope()
            raise ValueError(CANARY)
        host = SimpleNamespace(_tool_handlers={"synthetic": async_plugin if asynchronous else sync_plugin},
            plugins=SimpleNamespace(record_call=lambda name: None,
                                    mark_loaded=lambda *args: recorded.append(args)))
        pending = None
        try:
            with plugin_secret_scope("ancestor", {KEY: CANARY}):
                pending = asyncio.create_task(OrionDispatcher._dispatch_extension(host, "synthetic", {}))
                if asynchronous:
                    await asyncio.wait_for(started_async.wait(), 1)
                else:
                    assert await asyncio.to_thread(started_sync.wait, 1)
            release_async.set()
            release_sync.set()
            return await asyncio.wait_for(pending, 2)
        finally:
            release_async.set()
            release_sync.set()
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
    result = asyncio.run(scenario())
    assert not result.ok and CANARY not in repr(result) + repr(recorded)
    assert "[redacted]" in result.text and recorded
    assert observed == [("", "", "[redacted]"), ("", "", "[redacted]")]


def test_sync_smtp_dispatch_keeps_event_loop_responsive(monkeypatch):
    from orion_core.dispatch_web import WebDispatchMixin
    from orion_core.dispatcher import OrionDispatcher
    import orion_core.dispatcher as module
    monkeypatch.setattr(module, "_note_gap", lambda *args: None)
    calls = []
    def slow_send(*args):
        calls.append(threading.get_ident())
        time.sleep(.12)
        return ToolResult("Synthetic accepted result")
    host = SimpleNamespace(messaging=SimpleNamespace(send_email=slow_send),
        bus=SimpleNamespace(log=Signal()), active_tools=0, recent_tools=deque(), telemetry=None,
        _coerce_to_schema=lambda name, args: (args, None), _swarm_launch=lambda *args: None)
    host.handler_table = lambda: {"messaging": lambda args: WebDispatchMixin.messaging_tool(host, args)}
    async def scenario():
        ticks = []
        loop_thread = threading.get_ident()
        async def heartbeat():
            for _ in range(5):
                await asyncio.sleep(.01)
                ticks.append(time.monotonic())
        task = asyncio.create_task(heartbeat())
        before = time.monotonic()
        result = await OrionDispatcher.dispatch(host, "messaging", {"action": "email", "to": "synthetic@example.test", "body": "Synthetic"})
        finished = time.monotonic()
        await task
        assert result.ok
        assert calls and calls[0] != loop_thread
        assert len([tick for tick in ticks if before < tick < finished]) >= 3
    asyncio.run(scenario())
