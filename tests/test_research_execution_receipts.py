"""Research means an actual task, a truthful outcome and a retrievable file."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import deep_research, research_docx
from orion_core.data import ToolResult
from orion_core.dispatch_files import FilesDispatchMixin
from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.research import ResearchAgent
from orion_core.research_worker import Dossier, ResearchWorker, Source


class Signal:
    def __init__(self):
        self.sent = []

    def emit(self, *values):
        self.sent.append(values)


def bus():
    return SimpleNamespace(**{name: Signal() for name in (
        "log", "speak_request", "dashboard_event", "banner", "gui_command")})


class Dispatcher(KnowledgeDispatchMixin, FilesDispatchMixin):
    def __init__(self):
        self.bus = bus()
        self.research = object()
        self.research_director = None


def paper(topic="Federal law", *, sourced=True):
    return deep_research.Paper(topic, deep_research.Style.SYSTEMATIC, deep_research.Depth.BRIEF,
        sections=[deep_research.Section(1, "Findings", "brief", "A generated finding", 3)],
        sources=[{"title": "A checked source", "url": "https://example.test/source"}] if sourced else [])


@pytest.mark.parametrize("sourced", [True, False])
def test_dossier_returns_real_output_path_and_distinguishes_unverified_draft(tmp_path, monkeypatch, sourced):
    path = tmp_path / "Federal law dossier.md"
    path.write_text("saved research", encoding="utf-8")
    dossier = Dossier("Federal law", 1.0, path=path, summary="Summary",
                      sources=[Source("Source", "https://example.test/source")] if sourced else [])

    async def investigate(*_a, **_k):
        return dossier

    monkeypatch.setattr(ResearchWorker, "investigate", investigate)
    dispatcher = Dispatcher()
    result = asyncio.run(dispatcher._dossier("Federal law", {}))
    assert result.ok is sourced
    assert result.receipt["file_paths"] == [str(path)]
    assert dispatcher._file_references().resolve("that file") == (True, str(path))
    if not sourced:
        assert "unverified draft" in result.text and "not been completed" in result.text


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_dossier_never_claims_a_missing_file_or_directory_was_saved(tmp_path, monkeypatch, kind):
    path = tmp_path / "not-a-dossier.md"
    if kind == "directory":
        path.mkdir()
    dossier = Dossier("Topic", 1.0, path=path, sources=[Source("Source", "https://example.test")])

    async def investigate(*_a, **_k):
        return dossier

    monkeypatch.setattr(ResearchWorker, "investigate", investigate)
    result = asyncio.run(Dispatcher()._dossier("Topic", {}))
    assert not result.ok and result.receipt["file_paths"] == []
    assert "No saved dossier file could be verified" in result.text


@pytest.mark.parametrize("sourced", [True, False])
def test_paper_receipt_selects_verified_output_and_source_truth(tmp_path, monkeypatch, sourced):
    output = tmp_path / "Federal law paper.docx"
    document = paper(sourced=sourced)

    async def write(*_a, **_k):
        return document

    def save(_paper):
        output.write_bytes(b"synthetic saved document")
        return output, ""

    monkeypatch.setattr(research_docx, "write_paper", save)
    dispatcher = Dispatcher()
    result = asyncio.run(dispatcher._finish_paper(SimpleNamespace(write=write), "Federal law", None, None))
    assert result.ok is sourced and result.receipt["file_paths"] == [str(output)]
    assert dispatcher._file_references().resolve("that file") == (True, str(output))
    assert result.receipt["source_count"] == (1 if sourced else 0)


@pytest.mark.parametrize("failure", ["returned", "raised", "missing"])
def test_paper_filing_failures_never_announce_saved_or_success(tmp_path, monkeypatch, failure):
    async def write(*_a, **_k):
        return paper()

    def fail(_paper):
        if failure == "raised":
            raise OSError("disk unavailable")
        return (tmp_path / "absent.docx" if failure == "missing" else None), "disk unavailable"

    monkeypatch.setattr(research_docx, "write_paper", fail)
    dispatcher = Dispatcher()
    result = asyncio.run(dispatcher._finish_paper(SimpleNamespace(write=write), "Topic", None, None))
    assert not result.ok and result.receipt["file_paths"] == []
    assert "no paper was saved" in result.text
    assert not dispatcher.bus.speak_request.sent


@pytest.mark.parametrize("shape", ["all_failed", "part_failed", "empty_body", "no_sections", "normal", "unsourced"])
def test_saved_paper_completion_requires_all_sections_generated_and_sources(tmp_path, monkeypatch, shape):
    document = paper(sourced=shape != "unsourced")
    broken = deep_research.Section(2, "Unavailable", "brief", "", 0, error="provider failure")
    if shape == "all_failed":
        document.sections = [broken]
    elif shape == "part_failed":
        document.sections.append(broken)
    elif shape == "empty_body":
        document.sections[0].body = "   "
    elif shape == "no_sections":
        document.sections = []
    output = tmp_path / "saved-paper.docx"

    async def write(*_a, **_k):
        return document

    def save(_paper):
        output.write_bytes(b"saved complete or partial artifact")
        return output, ""

    monkeypatch.setattr(research_docx, "write_paper", save)
    dispatcher = Dispatcher()
    result = asyncio.run(dispatcher._finish_paper(SimpleNamespace(write=write), "Topic", None, None))
    complete = shape == "normal"
    assert result.ok is complete
    assert result.receipt["research_state"] == ("completed" if complete else "failed")
    assert result.receipt["selected_path"] == str(output) and output.is_file()
    assert bool(dispatcher.bus.speak_request.sent) is complete
    if not complete:
        assert "not complete" in result.text
        assert "Research finished" not in result.text


def test_detached_paper_failure_returns_actual_lifecycle_and_status_without_duplicate_speech(monkeypatch):
    async def write(*_a, **_k):
        await asyncio.sleep(0)
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()
    started, completed = [], []
    dispatcher.research_started = lambda topic, receipt: started.append((topic, receipt))
    dispatcher.research_completion = lambda topic, result: completed.append((topic, result))

    async def scenario():
        result = await dispatcher._deep_research({"depth": "brief"}, "Federal law")
        assert result.receipt["research_state"] == "running"
        task = dispatcher._background_research_runs[-1]["task"]
        await task
        await asyncio.sleep(0)
        status = await dispatcher.research_tool({"action": "status"})
        assert not status.ok and "provider unavailable" in status.text
        assert status.receipt["research_state"] == "failed"

    asyncio.run(scenario())
    assert len(started) == len(completed) == 1
    assert started[0][1]["run_id"] == completed[0][1].receipt["run_id"]
    assert not dispatcher.bus.speak_request.sent


@pytest.mark.parametrize("callback", ["absent", "broken"])
def test_detached_returned_failure_is_spoken_once_without_a_working_callback(monkeypatch, callback):
    async def write(*_a, **_k):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()
    if callback == "broken":
        dispatcher.research_completion = lambda *_: (_ for _ in ()).throw(RuntimeError("callback failed"))

    async def scenario():
        await dispatcher._deep_research({"depth": "brief"}, "Topic")
        await dispatcher._background_research_runs[-1]["task"]
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert len(dispatcher.bus.speak_request.sent) == 1
    assert "provider unavailable" in dispatcher.bus.speak_request.sent[0][0]


def test_detached_research_stop_cancels_actual_task_and_records_cancelled(monkeypatch):
    async def write(*_a, **_k):
        await asyncio.Event().wait()

    monkeypatch.setattr(deep_research, "DeepResearchEngine", lambda *_: SimpleNamespace(write=write))
    dispatcher = Dispatcher()
    finished = []
    dispatcher.research_completion = lambda topic, result: finished.append(result)

    async def scenario():
        await dispatcher._deep_research({}, "Topic")
        task = dispatcher._background_research_runs[-1]["task"]
        assert (await dispatcher.research_tool({"action": "status"})).receipt["research_state"] == "running"
        await dispatcher.research_tool({"action": "stop"})
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        status = await dispatcher.research_tool({"action": "status"})
        assert not status.ok and status.receipt["research_state"] == "cancelled"

    asyncio.run(scenario())
    assert len(finished) == 1 and finished[0].receipt["research_state"] == "cancelled"


def test_timed_research_failure_is_retained_in_status(tmp_path, monkeypatch):
    agent = ResearchAgent(bus(), router=None, memory=None)

    async def decompose(_topic):
        raise RuntimeError("source gathering failed")

    monkeypatch.setattr(agent, "_make_folder", lambda *_: tmp_path)
    monkeypatch.setattr(agent, "_decompose", decompose)

    async def scenario():
        assert agent.start_research("Topic", minutes=1).ok
        await agent._active["task"]
        status = agent.status()
        assert not status.ok and "source gathering failed" in status.text
        assert status.receipt["research_state"] == "failed"

    asyncio.run(scenario())


def test_timed_research_cancelled_task_is_never_described_as_complete(tmp_path, monkeypatch):
    agent = ResearchAgent(bus(), router=None, memory=None)

    async def decompose(_topic):
        await asyncio.Event().wait()

    monkeypatch.setattr(agent, "_make_folder", lambda *_: tmp_path)
    monkeypatch.setattr(agent, "_decompose", decompose)

    async def scenario():
        agent.start_research("Topic", minutes=1)
        agent.stop_research()
        with pytest.raises(asyncio.CancelledError):
            await agent._active["task"]
        assert not agent.status().ok and "cancelled" in agent.status().text

    asyncio.run(scenario())


def test_zero_source_worker_announcement_is_a_saved_draft_not_completed_research(tmp_path):
    path = tmp_path / "draft.md"
    path.write_text("draft", encoding="utf-8")
    signals = bus()
    dossier = Dossier("Topic", 1.0, path=path)
    asyncio.run(ResearchWorker(bus=signals).announce(dossier))
    announcement = signals.speak_request.sent[0][0]
    assert "unverified draft" in announcement and "research is not complete" in announcement
    assert "finished researching" not in announcement


def test_paper_memory_fallback_uses_real_tiered_api_and_keeps_exact_location(tmp_path):
    remembered = []

    class Memory:
        def remember(self, tier, key, value):
            remembered.append((tier, key, value))

    dispatcher = Dispatcher()
    dispatcher.memory = Memory()
    document = paper()
    document.docx_path = tmp_path / "Federal law paper.docx"
    document.sections = [deep_research.Section(1, "Scope", "brief", "Checked findings", 2)]
    assert "1 section" in dispatcher._file_research(document)
    assert remembered[0][0] == "knowledge"
    assert remembered[0][1].startswith("research_federal_law_")
    assert str(document.docx_path) in remembered[0][2]


@pytest.mark.parametrize("sourced", [True, False])
def test_timed_research_completion_returns_retrievable_summary_with_source_status(tmp_path, monkeypatch, sourced):
    agent = ResearchAgent(bus(), router=None, memory=None)

    async def decompose(_topic):
        return ["A concrete research question?"]

    async def gather(_question):
        return [{"title": "Source", "url": "https://example.test"}] if sourced else []

    async def synthesise(*_):
        return "Checked findings" if sourced else "General knowledge draft"

    async def summary(*_):
        return "Final synthesis"

    monkeypatch.setattr(agent, "_make_folder", lambda *_: tmp_path)
    monkeypatch.setattr(agent, "_decompose", decompose)
    monkeypatch.setattr(agent, "_gather_sources", gather)
    monkeypatch.setattr(agent, "_synthesise", synthesise)
    monkeypatch.setattr(agent, "_final_summary", summary)
    monkeypatch.setattr(agent, "_final_summary_from_notes", summary)
    dispatcher = Dispatcher()
    dispatcher.research = agent
    completed = []
    dispatcher.research_completion = lambda topic, result: completed.append(result)

    async def scenario():
        started = await dispatcher.research_tool({"action": "start", "topic": "Topic", "minutes": 1})
        assert started.receipt["research_state"] == "running"
        await agent._active["task"]
        await asyncio.sleep(0)
        status = await dispatcher.research_tool({"action": "status"})
        assert status.ok is sourced
        assert status.receipt["selected_path"] == str(tmp_path / "SUMMARY.md")
        assert Path(status.receipt["selected_path"]).is_file()

    asyncio.run(scenario())
    assert len(completed) == 1


@pytest.mark.parametrize("completion_hook", [False, True])
def test_awaited_dossier_is_visible_to_status_then_returns_actual_saved_result(tmp_path, monkeypatch, completion_hook):
    dispatcher = Dispatcher()
    started, completed = [], []
    if completion_hook:
        dispatcher.research_started = lambda topic, receipt: started.append(receipt)
        dispatcher.research_completion = lambda topic, result: completed.append(result)
    output = tmp_path / "Federal law dossier.md"

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()

        async def investigate(worker, topic, *, questions):
            entered.set()
            await release.wait()
            output.write_text("Sourced dossier", encoding="utf-8")
            dossier = Dossier(topic, 1.0, path=output,
                              sources=[Source("Source", "https://example.test")])
            if worker.announce_results:
                await worker.announce(dossier)
            return dossier

        monkeypatch.setattr(ResearchWorker, "investigate", investigate)
        result_task = asyncio.create_task(dispatcher.research_tool({"action": "dossier", "topic": "Federal law"}))
        await entered.wait()
        assert not result_task.done(), "dossier lost its awaited return contract"
        status = await dispatcher.research_tool({"action": "status"})
        assert status.receipt["research_state"] == "running"
        assert "still running" in status.text and status.receipt["action"] == "dossier"
        release.set()
        result = await result_task
        assert result.ok and result.receipt["research_state"] == "completed"
        assert result.receipt["selected_path"] == str(output)
        status = await dispatcher.research_tool({"action": "status"})
        assert status.text == result.text and status.receipt == result.receipt

    asyncio.run(scenario())
    if completion_hook:
        assert len(started) == len(completed) == 1
        assert completed[0].receipt["run_id"] == started[0]["run_id"]
        assert not dispatcher.bus.speak_request.sent, "worker duplicated the Live completion announcement"
    else:
        assert len(dispatcher.bus.speak_request.sent) == 1


def test_awaited_dossier_can_be_stopped_and_status_records_actual_cancellation(monkeypatch):
    dispatcher = Dispatcher()
    completed = []
    dispatcher.research_completion = lambda topic, result: completed.append(result)

    async def scenario():
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def investigate(worker, topic, *, questions):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(ResearchWorker, "investigate", investigate)
        task = asyncio.create_task(dispatcher.research_tool({"action": "dossier", "topic": "Topic"}))
        await entered.wait()
        assert (await dispatcher.research_tool({"action": "status"})).receipt["research_state"] == "running"
        stopping = await dispatcher.research_tool({"action": "stop"})
        assert stopping.receipt["research_state"] == "cancelling"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        status = await dispatcher.research_tool({"action": "status"})
        assert not status.ok and status.receipt["research_state"] == "cancelled"

    asyncio.run(scenario())
    assert len(completed) == 1 and completed[0].receipt["research_state"] == "cancelled"


def test_awaited_dossier_exception_becomes_recorded_failed_outcome_once(monkeypatch):
    dispatcher = Dispatcher()
    completed = []
    dispatcher.research_completion = lambda topic, result: completed.append(result)

    async def investigate(*_a, **_k):
        raise RuntimeError("provider connection failed")

    monkeypatch.setattr(ResearchWorker, "investigate", investigate)

    async def scenario():
        result = await dispatcher.research_tool({"action": "dossier", "topic": "Topic"})
        assert not result.ok and result.receipt["research_state"] == "failed"
        assert "provider connection failed" in result.text
        assert (await dispatcher.research_tool({"action": "status"})).receipt == result.receipt

    asyncio.run(scenario())
    assert len(completed) == 1 and not dispatcher.bus.speak_request.sent
