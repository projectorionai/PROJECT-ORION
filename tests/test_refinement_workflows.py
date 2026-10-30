"""A failed result can follow a partial effect; never replay it blindly."""
import asyncio
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.plan_executor import PlanExecutor, _Step
from orion_core.workflow_engine import WorkflowEngine


class Bus:
    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: None)


@pytest.mark.parametrize("tool", ["open_app", "send_message", "custom_plugin"])
def test_partial_failed_plan_effect_is_not_replayed(tool, monkeypatch):
    calls = []
    async def dispatch(name, args):
        calls.append(name)  # represents an effect accepted before the error
        return ToolResult("Partial effect; confirmation failed", ok=False,
                          receipt={"safe_to_retry": True})  # untrusted claim
    async def instant(*args):
        pass
    monkeypatch.setattr(asyncio, "sleep", instant)
    result = asyncio.run(PlanExecutor(Bus(), dispatch)._run_step(
        _Step(1, "first", tool, {}, "retry"), 3))
    assert calls == [tool]
    assert not result.ok and result.attempts == 1
    assert "not retried" in result.detail


@pytest.mark.parametrize("tool,expected", [("send_message", 1), ("custom_plugin", 1), ("web_search", 3)])
def test_workflow_only_retries_read_only_failures(tool, expected, tmp_path, monkeypatch):
    calls = []
    async def dispatch(name, args):
        calls.append(name)
        return ToolResult("Effect may have happened", ok=False)
    async def instant(*args):
        pass
    monkeypatch.setattr(asyncio, "sleep", instant)
    engine = WorkflowEngine(Bus(), dispatcher=SimpleNamespace(dispatch=dispatch),
                            path=tmp_path / "workflows.json")
    run = {"id": "run", "workflow": "test", "input": "", "step": 0,
           "total": 1, "transcript": []}
    asyncio.run(engine._execute(run, {"steps": [{"tool": tool, "args": {}, "retries": 2}]}))
    assert len(calls) == expected
    assert run["status"].startswith("failed")
    if expected == 1:
        assert "not retried" in run["transcript"][0]["text"]


def engine_for(tmp_path):
    return WorkflowEngine(Bus(), dispatcher=SimpleNamespace(handler_table=lambda: {"web_search": None}),
                          path=tmp_path / "workflows.json")


def test_overlong_workflow_is_rejected_as_a_whole(tmp_path):
    engine = engine_for(tmp_path)
    assert not engine.define("oversized", [{"tool": "web_search"}] * 13).ok
    assert "oversized" not in engine.definitions
    assert not engine.path.exists()


@pytest.mark.parametrize("bad", [None, 12, {"args": {}}, {"tool": "web_search", "args": "invalid"},
                                  {"tool": "web_search", "retries": "invalid"}])
def test_invalid_later_step_never_saves_a_shortened_workflow(tmp_path, bad):
    engine = engine_for(tmp_path)
    result = engine.define("broken", [{"tool": "web_search", "args": {}}, bad])
    assert not result.ok and "broken" not in engine.definitions


def test_workflow_preserves_all_json_arguments_and_keys(tmp_path):
    engine = engine_for(tmp_path)
    args = {f"option_{i}": i for i in range(16)}
    args["target_identity_" + "x" * 65] = "exact target"
    assert engine.define("complete", [{"tool": "web_search", "args": args}]).ok
    assert engine.definitions["complete"]["steps"][0]["args"] == args


def test_unrecordable_arguments_are_rejected(tmp_path):
    engine = engine_for(tmp_path)
    result = engine.define("oversized", [{"tool": "web_search", "args": {"query": "x" * 20001}}])
    assert not result.ok and "oversized" not in engine.definitions


@pytest.mark.parametrize("raw", [b'{broken', b'[]', b'{"saved":{"steps":[12]}}',
                                  b'{"saved":{"steps":[]}}', b'{"a":{},"a":{}}'])
def test_unreadable_library_is_retained_and_blocks_mutation(tmp_path, raw):
    path = tmp_path / "workflows.json"
    path.write_bytes(raw)
    engine = engine_for(tmp_path)
    assert not engine.list_workflows().ok
    assert not engine.define("new", [{"tool": "web_search", "args": {}}]).ok
    assert not engine.start("saved").ok
    assert path.read_bytes() == raw


def test_observed_external_library_change_is_not_overwritten(tmp_path):
    engine = engine_for(tmp_path)
    assert engine.define("saved", [{"tool": "web_search", "args": {}}]).ok
    engine.path.write_bytes(b'{externally changed')
    assert not engine.define("new", [{"tool": "web_search", "args": {}}]).ok
    assert engine.path.read_bytes() == b'{externally changed'


def test_overlong_input_does_not_start_a_different_truncated_job(tmp_path):
    engine = engine_for(tmp_path)
    assert engine.define("saved", [{"tool": "web_search", "args": {}}]).ok
    async def scenario():
        result = engine.start("saved", "x" * 2001)
        for run in list(engine.runs.values()):
            await run["task"]
        assert not result.ok and not engine.runs
    asyncio.run(scenario())
