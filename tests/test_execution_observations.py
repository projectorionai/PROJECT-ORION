"""Invocation observations never retain arguments/results or assert effects."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.dispatcher import OrionDispatcher
from orion_core.security import SecurityViolation


SECRET = "synthetic-execution-secret-do-not-record"


def host_for(handler):
    return SimpleNamespace(handler_table=lambda: {"synthetic_execution": handler},
        _coerce_to_schema=lambda name, args: (args, None), _swarm_launch=lambda *args: None,
        active_tools=0, recent_tools=deque(), telemetry=None)


def test_legacy_response_shape_is_unchanged_when_execution_absent():
    assert ToolResult("legacy").response_payload() == {"ok": True, "result": "legacy"}


def test_execution_constructor_and_response_are_defensive_copies():
    supplied = {"state": "returned", "nested": {"safe": True}}
    result = ToolResult("example", execution=supplied)
    supplied["nested"]["safe"] = False
    payload = result.response_payload()
    assert payload["execution"]["nested"]["safe"] is True
    payload["execution"]["nested"]["safe"] = False
    assert result.execution["nested"]["safe"] is True


@pytest.mark.parametrize("ok", [True, False])
def test_dispatch_records_one_invocation_without_effect_completion_or_secrets(ok):
    calls = []
    original = ToolResult(SECRET, ok=ok, receipt={"operation_state": "draft", "nested": [1]},
                          evidence=[{"value": "evidence"}], media={"data": b"bytes"})
    def handler(args):
        calls.append(dict(args))
        return original
    host = host_for(handler)
    result = asyncio.run(OrionDispatcher.dispatch(host, "synthetic_execution", {"body": SECRET}))
    assert result is not original
    assert result.execution["state"] == "returned"
    assert result.execution["executor_ok"] is ok
    assert result.execution["tool"] == "synthetic_execution"
    assert result.execution["call_id"]
    assert "completed" not in repr(result.execution)
    assert result.receipt == original.receipt and result.evidence == original.evidence
    assert result.media == original.media and original.execution is None
    assert calls == [{"body": SECRET}]
    history = OrionDispatcher.recent_executions(host)
    assert len(history) == 1 and SECRET not in repr(history)
    assert history[0]["call_id"] == result.execution["call_id"]
    result.execution["state"] = "tampered"
    result.receipt["nested"].append(2)
    assert OrionDispatcher.recent_executions(host)[0]["state"] == "returned"
    assert original.receipt["nested"] == [1]


def test_running_observation_is_visible_and_then_settles_once():
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def handler(args):
            calls.append(args)
            entered.set()
            await release.wait()
            return ToolResult("returned")
        host = host_for(handler)
        task = asyncio.create_task(OrionDispatcher.dispatch(host, "synthetic_execution", {}))
        await entered.wait()
        running = OrionDispatcher.recent_executions(host)
        assert len(running) == 1 and running[0]["state"] == "running"
        assert running[0]["executor_ok"] is None
        release.set()
        result = await task
        final = OrionDispatcher.recent_executions(host)
        assert len(final) == 1 and final[0]["state"] == "returned"
        assert final[0]["call_id"] == running[0]["call_id"] == result.execution["call_id"]
        assert len(calls) == 1
    asyncio.run(scenario())


def test_cancelled_invocation_is_recorded_then_propagated():
    async def scenario():
        entered = asyncio.Event()
        async def handler(args):
            entered.set()
            await asyncio.Future()
        host = host_for(handler)
        task = asyncio.create_task(OrionDispatcher.dispatch(host, "synthetic_execution", {}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        row = OrionDispatcher.recent_executions(host)[0]
        assert row["state"] == "cancelled" and row["executor_ok"] is None
    asyncio.run(scenario())


def test_security_rejection_propagates_and_does_not_retry():
    calls = []
    def handler(args):
        calls.append(args)
        raise SecurityViolation(SECRET)
    host = host_for(handler)
    with pytest.raises(SecurityViolation, match=SECRET):
        asyncio.run(OrionDispatcher.dispatch(host, "synthetic_execution", {}))
    assert len(calls) == 1
    history = OrionDispatcher.recent_executions(host)
    assert history[0]["state"] == "rejected" and SECRET not in repr(history)


def test_payload_security_rejection_prevents_handler_and_retains_no_payload():
    calls = []
    host = host_for(lambda args: calls.append(args))
    with pytest.raises(SecurityViolation):
        asyncio.run(OrionDispatcher.dispatch(host, "synthetic_execution", {"command": "cmd /c del *.*"}))
    assert calls == []
    row = OrionDispatcher.recent_executions(host)[0]
    assert row["state"] == "rejected" and "cmd" not in repr(row)


def test_escaping_router_fault_is_recorded_without_exception_text_or_retry(monkeypatch):
    calls = []
    async def broken(self, name, args):
        calls.append(name)
        raise RuntimeError(SECRET)
    monkeypatch.setattr(OrionDispatcher, "_dispatch_once", broken)
    host = SimpleNamespace()
    with pytest.raises(RuntimeError, match=SECRET):
        asyncio.run(OrionDispatcher.dispatch(host, "synthetic_execution", {}))
    assert calls == ["synthetic_execution"]
    history = OrionDispatcher.recent_executions(host)
    assert history[0]["state"] == "failed" and SECRET not in repr(history)


def test_observation_store_is_bounded_and_snapshots_cannot_mutate_it():
    from orion_core.execution_receipts import ExecutionObservationStore
    store = ExecutionObservationStore(max_records=3)
    records = [store.begin(f"tool_{n}") for n in range(6)]
    first = store.finish(records[0], "returned", executor_ok=True)
    assert first["state"] == "returned", "an evicted running call can still get its own terminal receipt"
    assert len(store.recent(100)) == 3
    assert [row["tool"] for row in store.recent(100)] == ["tool_5", "tool_4", "tool_3"]
    snapshot = store.recent()
    snapshot[0]["state"] = "tampered"
    assert store.recent()[0]["state"] == "running"
    assert store.recent(0) == [] and store.recent(-1) == []


def test_terminal_observation_cannot_be_rewritten_by_a_second_finish():
    from orion_core.execution_receipts import ExecutionObservationStore
    store = ExecutionObservationStore()
    record = store.begin("synthetic_execution")
    first = store.finish(record, "returned", executor_ok=False)
    second = store.finish(record, "failed")
    assert first == second and second["executor_ok"] is False


def test_store_initialization_and_updates_are_thread_safe():
    from orion_core.execution_receipts import store_for
    host = SimpleNamespace()
    def operate(index):
        store = store_for(host)
        record = store.begin(f"tool_{index}")
        store.finish(record, "returned", executor_ok=True)
        return id(store), record.call_id
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(operate, range(400)))
    assert len({identity for identity, _ in results}) == 1
    assert len({call_id for _, call_id in results}) == 400
    history = store_for(host).recent(1000)
    assert len(history) == 200
    assert all(row["state"] == "returned" and row["executor_ok"] is True for row in history)


def test_malformed_tool_identifier_does_not_store_raw_text():
    from orion_core.execution_receipts import ExecutionObservationStore
    store = ExecutionObservationStore()
    store.begin("bad tool " + SECRET)
    assert SECRET not in repr(store.recent())


def test_chain_keeps_each_execution_and_operation_receipt_separate():
    responses = [ToolResult("one", chain=[("second", {})], receipt={"file_paths": ["synthetic"]},
                            execution={"call_id": "one", "state": "returned", "executor_ok": True}),
                 ToolResult("two", execution={"call_id": "two", "state": "returned", "executor_ok": False}, ok=False)]
    calls = []
    async def dispatch(name, args):
        calls.append(name)
        return responses[len(calls)-1]
    host = SimpleNamespace(dispatch=dispatch, _derive_chain=lambda *args: [])
    result = asyncio.run(OrionDispatcher.dispatch_chain(host, "first", {}))
    assert calls == ["first", "second"] and not result.ok
    assert result.receipt["steps"][0]["receipt"] == {"file_paths": ["synthetic"]}
    assert [step["execution"]["call_id"] for step in result.execution["steps"]] == ["one", "two"]
    assert "execution" not in result.receipt["steps"][0]
