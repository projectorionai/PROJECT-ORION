"""A new admission must not execute instructions stale on disk."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from orion_core import jobs, workflow_engine as module
from orion_core.admission import Admission
from orion_core.data import ToolResult
from orion_core.job_store import JobStore, State


class Bus:
    def __getattr__(self, name):
        return SimpleNamespace(emit=lambda *args: None)


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_RESUMERS", dict(jobs._RESUMERS))
    observed = []

    async def dispatch(tool, args):
        observed.append((tool, args))
        return ToolResult("temporary executor observed the requested arguments")

    bus = Bus()
    store = JobStore(tmp_path / "jobs.db")
    admission = Admission()
    engine = module.WorkflowEngine(
        bus, SimpleNamespace(dispatch=dispatch, handler_table=lambda: {"web_search": None}),
        path=tmp_path / "workflows.json", store=store, admission=admission)
    assert engine.define("saved", [{"tool": "web_search", "args": {"query": "old target"}}]).ok
    yield engine, store, observed
    admission.close()
    store.close()


async def invoke(engine, operation, record=None):
    result = (engine.start("saved") if operation == "start" else
              engine.resume_job(record) if operation == "restart" else
              engine.list_workflows())
    # Drain even a buggy pre-fix admission: no leaked task or real executor.
    await asyncio.gather(*(r["task"] for r in engine.runs.values()))
    return result


def interrupted(engine, store):
    definition = engine.definitions["saved"]
    return store.create(kind="workflow", name="saved", state=State.INTERRUPTED,
                        args={"workflow": "saved", "definition_digest": engine._definition_digest(definition)},
                        steps=[{"index": 1, "tool": "web_search", "args": {"query": "old target"}}])


@pytest.mark.parametrize("operation", ["start", "restart", "list"])
@pytest.mark.parametrize("change", ["replacement", "deleted", "corrupt", "denied", "oversized"])
def test_changed_library_refuses_new_admission(library, monkeypatch, operation, change):
    engine, store, observed = library
    record = interrupted(engine, store) if operation == "restart" else None
    before_jobs = store.list()
    if change == "replacement":
        data = json.loads(engine.path.read_text())
        data["saved"]["steps"][0]["args"]["query"] = "replacement target"
        engine.path.write_text(json.dumps(data))
    elif change == "deleted":
        engine.path.unlink()
    elif change == "corrupt":
        engine.path.write_bytes(b"{broken external edit")
    elif change == "oversized":
        engine.path.write_text("x" * (module._MAX_LIBRARY_CHARS + 1))
    else:
        def denied(path):
            raise PermissionError("temporary library access denied")
        monkeypatch.setattr(module, "_read_library", denied)
    retained = engine.path.read_bytes() if engine.path.exists() else None
    result = asyncio.run(invoke(engine, operation, record))
    assert not result.ok
    assert not observed and not engine.runs
    assert store.list() == before_jobs
    assert (engine.path.read_bytes() if engine.path.exists() else None) == retained
    assert "retained" in result.text and "reload" in result.text


@pytest.mark.parametrize("operation", ["start", "restart"])
@pytest.mark.parametrize("rewrite", [False, True])
def test_unchanged_library_still_executes_exact_arguments(library, operation, rewrite):
    engine, store, observed = library
    record = interrupted(engine, store) if operation == "restart" else None
    if rewrite:
        engine.path.write_bytes(engine.path.read_bytes())
    assert asyncio.run(invoke(engine, operation, record)).ok
    assert observed == [("web_search", {"query": "old target"})]


def test_already_admitted_run_keeps_its_snapshot(library):
    engine, store, observed = library

    async def scenario():
        assert engine.start("saved").ok
        engine.path.write_text("{}")
        await asyncio.gather(*(r["task"] for r in engine.runs.values()))

    asyncio.run(scenario())
    assert observed == [("web_search", {"query": "old target"})]
