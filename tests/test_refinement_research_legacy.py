"""Legacy direct paper requests preserve prior artifacts and report actual outcomes."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import research
from orion_core.research import ResearchAgent


def agent_with_sources(sources):
    signal = SimpleNamespace(emit=lambda *_: None)
    agent = ResearchAgent(SimpleNamespace(log=signal, dashboard_event=signal), None, None)

    async def gather(_topic):
        return sources

    async def section(topic, heading, _sources):
        await asyncio.sleep(0)
        return f"Actual finding for {topic}: {heading}."

    agent._gather_sources, agent._write_section = gather, section
    return agent


def test_explicit_existing_folder_preserves_paper_sources_and_returns_new_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "BASE_DIR", tmp_path)
    previous = tmp_path / "paper" / "paper.md"
    previous.parent.mkdir()
    previous.write_bytes(b"prior paper")
    sources = tmp_path / "sources.md"
    sources.write_bytes(b"prior sources")
    agent = agent_with_sources([{"title": "Actual source", "url": "https://example.test", "snippet": "evidence"}])
    result = asyncio.run(agent.write_paper("New topic", tmp_path))
    assert result.ok
    assert previous.read_bytes() == b"prior paper" and sources.read_bytes() == b"prior sources"
    selected = Path(result.receipt["selected_path"])
    assert selected.is_file() and selected != previous and "Actual finding" in selected.read_text()
    assert (selected.parent / "sources.md").is_file()


def test_external_explicit_folder_returns_actual_receipt_without_workspace_relative_error(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "BASE_DIR", tmp_path / "different_workspace")
    agent = agent_with_sources([{"title": "Source", "url": "https://example.test"}])
    result = asyncio.run(agent.write_paper("External topic", tmp_path))
    assert result.ok and Path(result.receipt["selected_path"]).is_file()
    assert result.receipt["research_state"] == "completed"


def test_source_zero_legacy_paper_remains_saved_unverified_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(research, "BASE_DIR", tmp_path)
    result = asyncio.run(agent_with_sources([]).write_paper("Offline topic", tmp_path))
    assert not result.ok and result.receipt["research_state"] == "failed"
    path = Path(result.receipt["selected_path"])
    assert path.is_file() and "unverified" in path.read_text().lower()
    assert "draft" in result.text.lower() and "complete" not in result.text.lower()


def test_concurrent_explicit_papers_keep_each_paper_and_its_own_sources(tmp_path):
    agent = agent_with_sources([])

    async def gather(topic):
        await asyncio.sleep(0)
        return [{"title": topic, "url": f"https://example.test/{topic}", "snippet": f"Unique evidence {topic}"}]

    agent._gather_sources = gather

    async def scenario():
        return await asyncio.gather(*(agent.write_paper(f"topic-{i}", str(tmp_path)) for i in range(6)))

    results = asyncio.run(scenario())
    assert all(result.ok for result in results)
    paths = [Path(result.receipt["selected_path"]) for result in results]
    assert len(set(paths)) == 6
    for index, path in enumerate(paths):
        assert f"topic-{index}" in path.read_text()
        source_text = (path.parent / "sources.md").read_text()
        assert f"Unique evidence topic-{index}" in source_text
        assert all(f"Unique evidence topic-{other}" not in source_text for other in range(6) if other != index)
    assert not list(tmp_path.rglob(".paper-*.tmp"))


def test_generation_failure_reports_actual_sources_without_overwriting_prior_files(tmp_path):
    previous = tmp_path / "paper" / "paper.md"
    previous.parent.mkdir()
    previous.write_bytes(b"reviewed original")
    agent = agent_with_sources([{"title": "Source", "url": "https://example.test", "snippet": "Real evidence"}])

    async def fail(*_args):
        raise RuntimeError("synthetic provider refused")

    agent._write_section = fail
    result = asyncio.run(agent.write_paper("Topic", tmp_path))
    assert not result.ok and "provider refused" in result.text
    assert result.receipt["research_state"] == "failed"
    assert previous.read_bytes() == b"reviewed original"
    assert [Path(path).name for path in result.receipt["file_paths"]] == ["sources.md"]
    assert Path(result.receipt["selected_path"]).is_file()


def test_blank_generation_is_not_published_as_complete(tmp_path):
    agent = agent_with_sources([{"title": "Source", "url": "https://example.test"}])

    async def blank(*_args):
        return "   "

    agent._write_section = blank
    result = asyncio.run(agent.write_paper("Topic", tmp_path))
    assert not result.ok and "No usable text" in result.text
    assert not list(tmp_path.rglob("paper.md"))


def test_unverified_draft_is_not_filed_as_completed_memory(tmp_path):
    remembered = []
    agent = agent_with_sources([])
    agent._remember = lambda *args: remembered.append(args)
    result = asyncio.run(agent.write_paper("Topic", tmp_path))
    assert not result.ok and remembered == []


def test_publication_failure_preserves_previous_paper_and_never_remembers_success(tmp_path, monkeypatch):
    from orion_core import research_docx
    previous = tmp_path / "paper" / "paper.md"
    previous.parent.mkdir()
    previous.write_bytes(b"prior output")
    remembered = []
    agent = agent_with_sources([{"title": "Source", "url": "https://example.test"}])
    agent._remember = lambda *args: remembered.append(args)

    def fail(*_args, **_kwargs):
        raise PermissionError("synthetic publication failure")

    monkeypatch.setattr(research_docx, "_publish_new", fail)
    result = asyncio.run(agent.write_paper("Topic", tmp_path))
    assert not result.ok and "publication failure" in result.text
    assert previous.read_bytes() == b"prior output" and remembered == []
    assert not any(Path(path).name == "paper.md" for path in result.receipt["file_paths"])


def test_file_used_as_destination_returns_failure_without_truncation(tmp_path):
    target = tmp_path / "destination"
    target.write_bytes(b"keep this file")
    result = asyncio.run(agent_with_sources([]).write_paper("Topic", target))
    assert not result.ok and result.receipt["file_paths"] == []
    assert target.read_bytes() == b"keep this file"


@pytest.mark.parametrize("kind", ["legacy", "timed"])
def test_cancelled_source_publication_retains_observed_late_source_artifact(tmp_path, kind):
    from orion_core.research_runs import publication_paths
    agent = agent_with_sources([{"title": "Source", "url": "https://example.test"}])
    entered, release = threading.Event(), threading.Event()
    output = []

    def write(folder, _sources):
        path = folder / "sources.md"
        output.append(path)
        entered.set()
        assert release.wait(3)
        path.write_text("Actual late source evidence")

    agent._write_sources = write
    if kind == "timed":
        async def questions(_topic):
            return ["One synthetic question?"]
        agent._decompose = questions
        agent._synthesise = agent._write_section

    async def scenario():
        work = agent.write_paper("Topic", tmp_path) if kind == "legacy" else agent._run("Topic", 1, tmp_path)
        task = asyncio.create_task(work)
        async def _wait_entered():
            while not entered.is_set():
                if task.done():
                    await task
                    raise AssertionError("research task finished before entering _write_sources")
                await asyncio.sleep(0.001)
        await asyncio.wait_for(_wait_entered(), timeout=5.0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(100):
            if output[0].is_file():
                break
            await asyncio.sleep(0.005)
        paths, pending = publication_paths(task)
        assert str(output[0].resolve()) in paths and pending == []

    try:
        asyncio.run(scenario())
    finally:
        release.set()
