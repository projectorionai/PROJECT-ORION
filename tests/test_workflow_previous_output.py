"""A partial previous result must never become a new tool instruction."""
import asyncio
import copy
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
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    calls = []
    output = ["first result"]

    async def dispatch(tool, args):
        calls.append((tool, args))
        return ToolResult(output[0] if len(calls) == 1 else "done")

    store = JobStore(tmp_path / "jobs.db")
    admission = Admission()
    engine = WorkflowEngine(Bus(), SimpleNamespace(dispatch=dispatch,
        handler_table=lambda: {"web_search": None, "write_file": None}),
        path=tmp_path / "workflows.json", store=store, admission=admission)
    yield engine, store, calls, output
    admission.close()
    store.close()


@pytest.mark.parametrize("length", [1999, 2000, 2001])
@pytest.mark.parametrize("template", [{"content": "{prev}"},
    {"nested": [3, {"content": "prefix {prev}"}]}, {"content": "{prev}/{prev}"}])
def test_previous_output_bound_is_explicit_and_templates_are_immutable(length, template):
    original = copy.deepcopy(template)
    text = "x" * (length - 1) + "Z"
    if length > 2000:
        with pytest.raises(ValueError, match="[Pp]revious workflow output"):
            WorkflowEngine._resolve_args(template, "input", text)
    else:
        resolved = WorkflowEngine._resolve_args(template, "input", text)
        assert text in str(resolved)
    assert template == original


@pytest.mark.parametrize("template", [{"content": "{input}"}, {"literal": "untouched"}])
def test_unused_long_output_and_inserted_tokens_are_data(template):
    assert WorkflowEngine._resolve_args(template, "literal {prev}", "x" * 2001) == {
        key: value.replace("{input}", "literal {prev}") for key, value in template.items()}


@pytest.mark.parametrize("mode", ["overflow", "boundary", "unused"])
def test_live_run_never_dispatches_a_shortened_instruction(setup, mode):
    engine, store, calls, output = setup
    output[0] = "x" * (1999 if mode == "boundary" else 2000) + "Z"
    args = {"content": "{prev}" if mode != "unused" else "independent"}
    assert engine.define("chain", [{"tool": "web_search"},
                                   {"tool": "write_file", "args": args}]).ok

    async def scenario():
        assert engine.start("chain").ok
        run = next(iter(engine.runs.values()))
        await run["task"]
        return run

    run = asyncio.run(scenario())
    record = store.get(run["job_id"])
    if mode == "overflow":
        assert len(calls) == 1 and record["state"] == "failed"
        assert "previous workflow output" in run["status"].lower()
        assert record["checkpoint"]["prev_complete"] is False
        assert output[0][:200] in run["result"], "the undispatched consumer must not hide its source"
    else:
        assert record["state"] == "completed" and len(calls) == 2
        assert calls[1][1]["content"] == (output[0] if mode == "boundary" else "independent")


@pytest.mark.parametrize("marker", ["overflow", "legacy", "complete"])
@pytest.mark.parametrize("mutating", [False, True])
@pytest.mark.parametrize("consumes", [False, True])
def test_restart_rejects_partial_prev_before_execution_or_approval(setup, marker, mutating, consumes):
    engine, store, calls, output = setup
    steps = [{"tool": "web_search"},
             {"tool": "write_file" if mutating else "web_search",
              "args": {"content": "{prev}" if consumes else "independent"}}]
    assert engine.define("chain", steps).ok
    record = store.create(kind="workflow", name="chain", state=State.INTERRUPTED,
        args={"workflow": "chain", "definition_digest": engine._definition_digest(engine.definitions["chain"])},
        steps=[dict(step, index=i) for i, step in enumerate(steps, 1)])
    store.set_step(record["id"], 1, state="completed")
    store.set_step(record["id"], 2, state="interrupted")
    checkpoint = {"next": 2, "prev": "x" * 2000}
    if marker != "legacy":
        checkpoint["prev_complete"] = marker == "complete"
    store.update(record["id"], checkpoint=checkpoint)
    before = store.get(record["id"])

    async def scenario():
        result = engine.resume_job(before)
        await asyncio.gather(*(r["task"] for r in engine.runs.values()))
        return result

    result = asyncio.run(scenario())
    if marker != "complete" and consumes:
        assert not result.ok and "previous workflow output" in result.text.lower()
        assert calls == [] and not engine.runs
        assert store.get(record["id"]) == before
        assert store.latest_approval(record["id"], 2) is None
    elif mutating:
        assert not result.ok and calls == []
        assert store.latest_approval(record["id"], 2)["state"] == "pending"
    else:
        assert result.ok and calls == [("web_search", {"content": "x" * 2000 if consumes else "independent"})]


@pytest.mark.parametrize("consumer", [None, False, True])
def test_a_source_result_is_retained_until_a_consumer_returns_successfully(consumer):
    steps = [{"tool": "web_search", "args": {}},
             {"tool": "write_file", "args": {"content": "{prev}"}}]
    outputs = [(1, steps[0], "Important source findings", True)]
    if consumer is not None:
        outputs.append((2, steps[1], "Saved summary" if consumer else "Could not save", consumer))
    report, _shortened = WorkflowEngine._report({"steps": steps}, outputs)
    assert ("Important source findings" in report) is (consumer is not True)
    if consumer is False:
        assert "Could not save" in report
    elif consumer is True:
        assert report == "Saved summary"
