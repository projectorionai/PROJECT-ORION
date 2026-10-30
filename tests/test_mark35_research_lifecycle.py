"""Mark XXXV research features use real task settlement and temporary evidence."""

from __future__ import annotations

import asyncio
import copy
import threading
from types import SimpleNamespace

import pytest

from orion_core.data import ToolResult
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.research import ResearchAgent
from orion_core.research_director import ResearchDirector


class Signal:
    def __init__(self):
        self.sent = []

    def emit(self, *args):
        self.sent.append(args)


class Bus:
    def __getattr__(self, name):
        signal = Signal()
        setattr(self, name, signal)
        return signal


class Cognition:
    def __init__(self):
        self.rows = {}

    def upsert_research_session(self, topic, updates):
        row = self.rows.setdefault(topic, {"topic": topic, "updated_at": str(len(self.rows))})
        row.update(updates)
        return dict(row)

    def snapshot(self):
        return {"research_sessions": copy.deepcopy(self.rows)}


class Dispatcher(KnowledgeDispatchMixin):
    def __init__(self):
        self.bus = Bus()
        self.research = object()
        self.research_director = None


async def _wait_entered(entered: threading.Event, task: asyncio.Task | None = None) -> None:
    for _ in range(3000):
        if entered.is_set():
            return
        if task is not None and task.done():
            task.result()
            raise AssertionError("research task finished before entering target stage")
        await asyncio.sleep(0.001)
    raise TimeoutError("timed out waiting for research stage entry")


def test_real_director_advances_only_after_first_task_settles(tmp_path, monkeypatch):
    async def scenario():
        first_entered, first_release, second_entered = (asyncio.Event() for _ in range(3))
        agent = ResearchAgent(Bus(), None, None)
        cognition = Cognition()
        director = ResearchDirector(agent.bus, agent, cognition, object())
        monkeypatch.setattr(director, "harvest_folder", lambda *_: 0)
        monkeypatch.setattr(agent, "RESEARCH_ROOT", tmp_path)

        async def run(topic, minutes, folder):
            if topic == "First":
                first_entered.set()
                await first_release.wait()
            else:
                second_entered.set()
            # Exercise the actual agent lifecycle wrapper, not an inert done flag.
            return ToolResult("Completed", receipt={"file_paths": [str(folder)], "research_state": "completed"})

        monkeypatch.setattr(agent, "_run", run)
        await director.queue_topic("First", 1)
        await first_entered.wait()
        await director.queue_topic("Second", 1)
        first_task = agent._active["task"]
        first_release.set()
        await first_task
        await asyncio.wait_for(second_entered.wait(), 0.5)
        await asyncio.sleep(0.05)
        assert cognition.rows["First"]["status"] == "complete"
        assert cognition.rows["Second"]["status"] == "complete"

    asyncio.run(scenario())


def test_browse_empty_summary_keeps_notes_but_does_not_complete(tmp_path, monkeypatch):
    from orion_core import deep_research, live_research

    async def investigate(*_a, **_k):
        return SimpleNamespace(notes=[object()], read=["https://example.test"], seconds=1)

    async def summary(*_a, **_k):
        return "  "

    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "note.md").write_text("Useful note", encoding="utf-8")
    monkeypatch.setattr(live_research, "LiveResearcher", lambda **_: SimpleNamespace(
        investigate=investigate, summarise=summary, write_folder=lambda *_: folder))
    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(_ask=object()))
    dispatcher = Dispatcher()
    dispatcher.research = SimpleNamespace(_ask=object())
    result = asyncio.run(dispatcher._browse_research("Topic", {}))
    assert not result.ok and result.receipt["research_state"] == "failed"
    assert str(folder) in result.receipt["file_paths"]
    assert "summary" in result.text.lower() and "incomplete" in result.text.lower()


def test_cancel_during_paper_publication_keeps_late_artifact_in_status(tmp_path, monkeypatch):
    from orion_core import deep_research, research_docx
    entered, release = threading.Event(), threading.Event()
    path = tmp_path / "late-paper.docx"

    def save(_paper):
        entered.set()
        assert release.wait(3)
        path.write_bytes(b"published after cancellation")
        return path, ""

    paper = deep_research.Paper("Topic", deep_research.Style.SYSTEMATIC, deep_research.Depth.BRIEF)

    async def write(*_a, **_k):
        return paper

    monkeypatch.setattr(research_docx, "write_paper", save)
    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()

    async def scenario():
        await dispatcher.research_tool({"action": "paper", "topic": "Topic"})
        task = dispatcher._background_research_runs[-1]["task"]
        await _wait_entered(entered, task)
        await dispatcher.research_tool({"action": "stop", "topic": "Topic"})
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        status = await dispatcher.research_tool({"action": "status", "topic": "Topic"})
        assert not status.ok and status.receipt["research_state"] == "cancelled"
        assert str(path) in status.receipt["file_paths"]
        assert "artifact" in status.text.lower()

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_ledger_restart_marks_unfinished_runs_interrupted_and_preserves_artifacts(tmp_path):
    from orion_core.research_runs import ResearchRunStore
    path = tmp_path / "research-runs.json"
    artifact = tmp_path / "partial.md"
    artifact.write_text("Useful partial findings", encoding="utf-8")
    first = ResearchRunStore(path)
    row = first.start("Topic", "dossier", request_id=17, inputs={"questions": 3, "api_key": "never-store"})
    first.update(row["run_id"], {"research_state": "cancel_requested", "file_paths": [str(artifact)]}, "Publication settling")
    first.close()  # simulate process ownership ending before restart recovery
    restored = ResearchRunStore(path)
    record = restored.list(run_id=row["run_id"])[0]
    assert record["research_state"] == "interrupted"
    assert record["error_category"] == "process_interrupted"
    assert record["file_paths"] == [str(artifact)] and record["request_id"] == 17
    assert record["inputs"] == {"questions": 3}
    assert "never-store" not in path.read_text(encoding="utf-8")


def test_ledger_failed_atomic_save_and_corruption_preserve_existing_state(tmp_path, monkeypatch):
    from orion_core import research_runs
    path = tmp_path / "runs.json"
    store = research_runs.ResearchRunStore(path)
    row = store.start("Topic", "paper")
    before = path.read_bytes()

    def denied(*_a, **_k):
        raise OSError("synthetic disk lock")

    monkeypatch.setattr(research_runs, "atomic_write_text", denied)
    with pytest.raises(OSError):
        store.update(row["run_id"], {"research_state": "failed"}, "Failed")
    assert path.read_bytes() == before and store.list()[0]["research_state"] == "running"
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{broken record", encoding="utf-8")
    unreadable = research_runs.ResearchRunStore(corrupt)
    assert unreadable.storage_error
    with pytest.raises(OSError):
        unreadable.start("New topic", "dossier")
    assert corrupt.read_text(encoding="utf-8") == "{broken record"


def test_ledger_bounds_history_without_evicting_active_runs(tmp_path):
    from orion_core.research_runs import ResearchRunStore
    store = ResearchRunStore(tmp_path / "runs.json", limit=3)
    active = store.start("Still running", "browse")
    for index in range(6):
        row = store.start(f"Finished {index}", "paper")
        store.update(row["run_id"], {"research_state": "completed"}, "Done")
    assert len(store.list()) == 3
    assert store.list(run_id=active["run_id"])[0]["research_state"] == "running"


def test_durable_run_status_open_and_missing_artifact_after_new_dispatcher(tmp_path, monkeypatch):
    from orion_core import file_search, research_runs
    store = research_runs.ResearchRunStore(tmp_path / "runs.json")
    artifact = tmp_path / "report.md"
    artifact.write_text("Evidence", encoding="utf-8")
    row = store.start("Known topic", "dossier")
    store.update(row["run_id"], {"research_state": "completed", "file_paths": [str(artifact)]}, "Completed with evidence")
    dispatcher = Dispatcher()
    dispatcher._research_run_store = research_runs.ResearchRunStore(store.path)
    opened = []
    monkeypatch.setattr(file_search, "open_path", lambda path: opened.append(path))

    async def scenario():
        history = await dispatcher.research_tool({"action": "runs"})
        assert row["run_id"] in history.text
        status = await dispatcher.research_tool({"action": "status", "run_id": row["run_id"]})
        assert status.ok and status.receipt["selected_path"] == str(artifact)
        assert (await dispatcher.research_tool({"action": "open", "run_id": row["run_id"]})).ok
        artifact.unlink()
        missing = await dispatcher.research_tool({"action": "open", "run_id": row["run_id"]})
        assert not missing.ok and missing.receipt["recorded_file_paths"] == [str(artifact)]
        assert missing.receipt["file_paths"] == []

    asyncio.run(scenario())
    assert opened == [str(artifact)]


def test_cancellation_returns_pending_while_thread_settles_then_records_artifact(tmp_path, monkeypatch):
    from orion_core import deep_research, research_docx, research_runs
    monkeypatch.setattr(research_runs, "PUBLICATION_SETTLE_S", 0.01)
    entered, release = threading.Event(), threading.Event()
    path = tmp_path / "late.docx"

    def save(_paper):
        entered.set()
        assert release.wait(3)
        path.write_bytes(b"late publication")
        return path, ""

    async def write(*_a, **_k):
        return object()

    monkeypatch.setattr(research_docx, "write_paper", save)
    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()
    completed = []
    dispatcher.research_completion = lambda _topic, result: completed.append(result)

    async def scenario():
        started = await dispatcher.research_tool({"action": "paper", "topic": "Topic"})
        run_id = started.receipt["run_id"]
        original = dispatcher._background_research_runs[-1]["task"]
        await _wait_entered(entered, original)
        await dispatcher.research_tool({"action": "stop", "run_id": run_id})
        with pytest.raises(asyncio.CancelledError):
            await original
        pending = await dispatcher.research_tool({"action": "status", "run_id": run_id})
        assert pending.receipt["research_state"] == "cancel_requested"
        assert not completed
        release.set()
        await dispatcher._background_research_runs[-1]["publication_task"]
        status = await dispatcher.research_tool({"action": "status", "run_id": run_id})
        assert status.receipt["research_state"] == "cancelled"
        assert status.receipt["file_paths"] == [str(path)]
        rows = await dispatcher.research_tool({"action": "history", "run_id": run_id})
        assert rows.receipt["research_runs"][0]["file_paths"] == [str(path)]

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert len(completed) == 1 and not completed[0].ok


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_director_retains_non_success_outcome_and_cancel_does_not_launch_next(tmp_path, monkeypatch, outcome):
    async def scenario():
        entered, release, second = (asyncio.Event() for _ in range(3))
        agent = ResearchAgent(Bus(), None, None)
        cognition = Cognition()
        director = ResearchDirector(agent.bus, agent, cognition, object())
        monkeypatch.setattr(agent, "RESEARCH_ROOT", tmp_path)
        monkeypatch.setattr(director, "harvest_folder", lambda *_: 0)

        async def run(topic, minutes, folder):
            if topic == "First":
                entered.set()
                await release.wait()
                return ToolResult("Provider failed or no sources", ok=False,
                                  receipt={"research_state": "failed"})
            second.set()
            return ToolResult("Completed")

        monkeypatch.setattr(agent, "_run", run)
        await director.queue_topic("First", 1)
        await entered.wait()
        await director.queue_topic("Second", 1)
        task = agent._active["task"]
        if outcome == "cancelled":
            agent.stop_research()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            await task
        for _ in range(100):
            if cognition.rows["First"]["status"] == outcome:
                break
            await asyncio.sleep(0.005)
        assert cognition.rows["First"]["status"] == outcome
        if outcome == "cancelled":
            assert not second.is_set() and cognition.rows["Second"]["status"] == "queued"
        else:
            await asyncio.wait_for(second.wait(), 0.5)

    asyncio.run(scenario())


def test_unreadable_ledger_prevents_research_operation_start(tmp_path, monkeypatch):
    from orion_core import deep_research, research_runs
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    calls = []

    async def write(*_a, **_k):
        calls.append("provider was contacted")
        return object()

    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()
    dispatcher._research_run_store = research_runs.ResearchRunStore(path)

    async def scenario():
        with pytest.raises(OSError, match="was not started"):
            await dispatcher.research_tool({"action": "paper", "topic": "Topic"})
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert calls == [] and path.read_text(encoding="utf-8") == "{broken"


@pytest.mark.parametrize("kind", ["dossier", "browse"])
def test_cancelled_threaded_dossier_and_browse_publication_retain_actual_output(tmp_path, monkeypatch, kind):
    from orion_core import deep_research, live_research, research_worker
    entered, release = threading.Event(), threading.Event()
    output = tmp_path / ("dossier.md" if kind == "dossier" else "browse")

    def file(*_args):
        entered.set()
        assert release.wait(3)
        if kind == "dossier":
            output.write_text("Partial dossier", encoding="utf-8")
        else:
            output.mkdir()
            (output / "SUMMARY.md").write_text("Saved summary", encoding="utf-8")
        return output

    if kind == "dossier":
        class Agent:
            async def _decompose(self, topic): return ["Question"]
            async def gather_sources(self, question): return [{"title": "Source", "url": "https://example.test"}]
            async def _synthesise(self, *args): return "A note"
            async def _final_summary(self, *args): return "A summary"
        monkeypatch.setattr(research_worker.ResearchWorker, "file", file)
        research = Agent()
    else:
        async def investigate(*_a, **_k):
            return SimpleNamespace(notes=[object()], read=["https://example.test"], seconds=1)
        async def summary(*_a, **_k): return "Summary"
        monkeypatch.setattr(live_research, "LiveResearcher", lambda **_: SimpleNamespace(
            investigate=investigate, summarise=summary, write_folder=file))
        monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(_ask=object()))
        research = SimpleNamespace(_ask=object())
    dispatcher = Dispatcher()
    dispatcher.research = research

    async def scenario():
        waiter = asyncio.create_task(dispatcher.research_tool({"action": kind, "topic": "Topic", "wait": True}))
        await _wait_entered(entered, waiter)
        status = await dispatcher.research_tool({"action": "status", "topic": "Topic"})
        run_id = status.receipt["run_id"]
        await dispatcher.research_tool({"action": "stop", "run_id": run_id})
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        status = await dispatcher.research_tool({"action": "status", "run_id": run_id})
        assert status.receipt["research_state"] == "cancelled"
        assert str(output) in status.receipt["file_paths"]

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("kind,wait", [("paper", False), ("paper", True), ("start", False)])
def test_cancel_during_durable_admission_settles_without_starting_work(tmp_path, monkeypatch, kind, wait):
    from orion_core import deep_research, research_runs
    entered, release = threading.Event(), threading.Event()
    store = research_runs.ResearchRunStore(tmp_path / "runs.json")
    original_start = store.start
    calls = []

    def start(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original_start(*args, **kwargs)

    async def work(*_args, **_kwargs):
        calls.append("provider started")
        return ToolResult("Completed")

    monkeypatch.setattr(store, "start", start)
    monkeypatch.setattr(research_runs, "PUBLICATION_SETTLE_S", 0.01)
    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=work))
    dispatcher = Dispatcher()
    dispatcher._research_run_store = store
    if kind == "start":
        dispatcher.research = ResearchAgent(dispatcher.bus, None, None)
        monkeypatch.setattr(dispatcher.research, "RESEARCH_ROOT", tmp_path / "research")
        monkeypatch.setattr(dispatcher.research, "_run", work)

    async def scenario():
        request = asyncio.create_task(dispatcher.research_tool({"action": kind, "topic": "Topic", "wait": wait}))
        await _wait_entered(entered, request)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        if kind == "start":
            assert dispatcher.research._active["task"].cancelled()
        release.set()
        for _ in range(100):
            rows = store.list()
            if rows and rows[0]["research_state"] == "cancelled":
                break
            await asyncio.sleep(0.005)
        assert rows[0]["research_state"] == "cancelled"
        assert rows[0]["error_category"] == "admission_cancelled"
        assert calls == []

    try:
        asyncio.run(scenario())
    finally:
        release.set()


@pytest.mark.parametrize("ending", ["cancelled", "failed"])
def test_timed_research_reports_one_finish_however_it_ends(tmp_path, ending):
    """Mark XXXX.9: the BRAIN and NEURAL pages show research only while it
    runs, so a cancelled or failed run must clear itself as surely as a
    finished one (a cancellation used to emit nothing on the dashboard)."""
    from orion_core import research_runs

    bus = Bus()
    agent = ResearchAgent(bus, None, None)

    async def decompose(_topic):
        if ending == "failed":
            raise RuntimeError("synthetic planner failure")
        await asyncio.sleep(30)

    agent._decompose = decompose

    async def scenario():
        task = asyncio.create_task(agent._run("Lifecycle topic", 1, tmp_path))
        await asyncio.sleep(0.05)
        if ending == "cancelled":
            assert any(row["topic"] == "Lifecycle topic" for row in research_runs.live_activity())
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    events = [payload for channel, payload in bus.dashboard_event.sent if channel == "research"]
    assert events[0]["status"] == "running" and events[0]["engine"] == "timed"
    assert [e["status"] for e in events[1:]] == [ending]
    assert not any(row["topic"] == "Lifecycle topic" for row in research_runs.live_activity())
