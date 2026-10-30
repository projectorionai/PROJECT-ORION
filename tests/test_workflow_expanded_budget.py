"""Validate complete serialized workflow data before effects or admission."""
import asyncio
import copy
import json
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


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    observed, submitted = [], []
    async def dispatch(tool, args):
        observed.append((tool, args))
        return ToolResult("x" * 2000)
    store, admission = JobStore(tmp_path / "jobs.db"), Admission()
    original = admission.submit
    def submit(*args, **kwargs):
        submitted.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(admission, "submit", submit)
    value = WorkflowEngine(Bus(), SimpleNamespace(dispatch=dispatch,
        handler_table=lambda: {"web_search": None, "write_file": None}),
        path=tmp_path / "workflows.json", store=store, admission=admission)
    yield value, store, observed, submitted
    admission.close()
    store.close()


@pytest.mark.parametrize("case", ["input", "prev", "nested", "emoji", "escapes"])
def test_whole_expansion_budget_includes_nesting_and_json_escaping(case):
    args, text, prev = {"content": "{input}" * 11}, "x" * 2000, ""
    if case == "prev":
        args, text, prev = {"content": "{prev}" * 11}, "", "x" * 2000
    elif case == "nested":
        args = {"a": ["{input}" * 6], "b": {"c": "{input}" * 6}}
    elif case == "emoji":
        args, text = {"content": "{input}"}, "\U0001f600" * 1900
    elif case == "escapes":
        args, text = {"content": "{input}" * 6}, '\\"' * 1000
    original = copy.deepcopy(args)
    with pytest.raises(ValueError, match="Expanded workflow arguments"):
        WorkflowEngine._resolve_args(args, text, prev)
    assert args == original


@pytest.mark.parametrize("size", [20000, 20001])
def test_exact_serialized_boundary_is_inclusive(size):
    padding = size - len(json.dumps({"content": "x" * 19980}))
    args = {"content": "{input}" * 10 + "x" * padding}
    if size == 20001:
        with pytest.raises(ValueError, match="Expanded workflow arguments"):
            WorkflowEngine._resolve_args(args, "x" * 1998, "")
    else:
        resolved = WorkflowEngine._resolve_args(args, "x" * 1998, "")
        assert len(json.dumps(resolved)) == size


def test_inserted_tokens_remain_literal_with_unused_large_prev():
    assert WorkflowEngine._resolve_args({"content": "{input}"}, "literal {prev}", "x" * 5000) == {
        "content": "literal {prev}"}


async def drain(value):
    await asyncio.gather(*(run["task"] for run in value.runs.values()))


@pytest.mark.parametrize("mode", ["input", "prev", "boundary"])
def test_live_expansion_is_complete_or_refused_before_tool_effect(engine, mode):
    value, store, observed, submitted = engine
    if mode == "prev":
        steps = [{"tool": "web_search"},
                 {"tool": "write_file", "args": {"content": "{prev}" * 11}}]
    else:
        steps = [{"tool": "write_file", "args": {"content":
                  "{input}" * (10 if mode == "boundary" else 11) + ("x" * 5 if mode == "boundary" else "")}}]
    assert value.define("chain", steps).ok
    async def scenario():
        assert value.start("chain", "x" * (1998 if mode == "boundary" else 2000)).ok
        await drain(value)
    asyncio.run(scenario())
    run = next(iter(value.runs.values()))
    if mode == "boundary":
        assert run["status"] == "complete" and len(observed) == 1
        assert len(json.dumps(observed[0][1])) == 20000
    else:
        assert "expanded workflow arguments" in run["status"].lower()
        assert [tool for tool, _ in observed] == (["web_search"] if mode == "prev" else [])
        assert store.get(run["job_id"])["state"] == "failed"


@pytest.mark.parametrize("kind", ["ascii", "emoji", "escapes"])
def test_job_input_must_fit_its_record_before_admission(engine, kind):
    value, store, observed, submitted = engine
    text = {"ascii": "x" * 2000, "emoji": "\U0001f600" * 2000, "escapes": '\\"' * 1000}[kind]
    assert value.define("chain", [{"tool": "write_file", "args": {"content": "fixed"}}]).ok
    async def scenario():
        result = value.start("chain", text)
        await drain(value)
        return result
    result = asyncio.run(scenario())
    if kind == "emoji":
        assert not result.ok and "Workflow job metadata" in result.text
        assert not observed and not submitted and not value.runs and not store.list()
    else:
        assert result.ok and store.list()[0]["args_complete"]
        assert store.list()[0]["args"]["input"] == text


@pytest.mark.parametrize("mutating", [False, True])
def test_restart_expansion_refuses_before_approval_or_admission(engine, mutating):
    value, store, observed, submitted = engine
    steps = [{"tool": "write_file" if mutating else "web_search",
              "args": {"content": "{input}" * 11}}]
    assert value.define("chain", steps).ok
    record = store.create(kind="workflow", name="chain", state=State.INTERRUPTED,
        args={"workflow": "chain", "input": "x" * 2000,
              "definition_digest": value._definition_digest(value.definitions["chain"])},
        steps=[dict(steps[0], index=1)])
    store.set_step(record["id"], 1, state="interrupted")
    before = store.get(record["id"])
    async def scenario():
        result = value.resume_job(before)
        await drain(value)
        return result
    result = asyncio.run(scenario())
    assert not result.ok and "Expanded workflow arguments" in result.text
    assert not observed and not submitted and not value.runs
    assert store.get(record["id"]) == before
    assert store.latest_approval(record["id"], 1) is None
