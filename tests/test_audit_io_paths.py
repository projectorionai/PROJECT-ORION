"""Behavioural checks for file I/O found by the whole-tree async inventory."""
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest


async def test_research_writes_in_workers_and_preserves_output(tmp_path, monkeypatch):
    import orion_core.research as module
    from orion_core.research import ResearchAgent
    owner = threading.get_ident()
    written = []
    original = Path.write_text
    def write(self, *args, **kwargs):
        written.append(threading.get_ident())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(module, "BASE_DIR", tmp_path)
    signal = SimpleNamespace(emit=lambda *_: None)
    agent = ResearchAgent(SimpleNamespace(log=signal, dashboard_event=signal), None, None)
    async def questions(*_): return ["one question"]
    async def sources(*_): return [{"title": "Evidence", "url": "https://example.com"}]
    async def prose(*_): return "Test synthesis"
    agent._decompose, agent._gather_sources = questions, sources
    agent._synthesise = agent._final_summary_from_notes = agent._write_section = prose
    agent._remember = lambda *_: None
    await agent._run("test", 0.1, tmp_path)
    assert (tmp_path / "SUMMARY.md").read_text(encoding="utf-8").endswith("Test synthesis\n")
    assert (await agent.write_paper("test", tmp_path)).ok
    assert written and all(t != owner for t in written)


async def test_shared_video_bytes_get_independent_temporary_files():
    from orion_core.video_analysis import VideoAnalyser
    barrier = threading.Barrier(2)
    files = []
    owner = threading.get_ident()
    def reader(path):
        assert threading.get_ident() != owner
        files.append(path)
        barrier.wait(timeout=3)
        assert path.read_bytes() == b"same-jpeg-bytes"
        return "visible text"
    analyser = VideoAnalyser.__new__(VideoAnalyser)
    analyser.vision = SimpleNamespace(_ocr_file_text=reader)
    jpeg = b"same-jpeg-bytes"
    assert await asyncio.gather(analyser._describe_with_ocr(jpeg), analyser._describe_with_ocr(jpeg)) == ["visible text"] * 2
    assert len(set(files)) == 2
    assert all(not path.exists() for path in files)


async def test_rehabilitation_reads_source_off_loop(tmp_path, monkeypatch):
    from orion_core.dynamic_loader import ReflectiveModuleLoader
    path = tmp_path / "broken.py"
    path.write_text("def invalid(:")
    owner = threading.get_ident()
    original = Path.read_text
    seen = []
    def read(self, *args, **kwargs):
        seen.append(threading.get_ident())
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    loader = ReflectiveModuleLoader(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    assert not await loader._can_rehabilitate(path)
    assert seen and all(t != owner for t in seen)


@pytest.mark.parametrize("provider_available", [False, True])
async def test_selfrepair_reads_source_off_loop_before_contacting_provider(tmp_path, monkeypatch, provider_available):
    from orion_core.selfrepair import Incident, SelfRepairAgent
    path = tmp_path / "example.py"
    path.write_text("x = 1")
    owner = threading.get_ident()
    seen = []
    # Repairs now preserve original bytes for the reviewed-source digest.
    # Observe both APIs, including any second read used for a source window.
    for method in ("read_text", "read_bytes"):
        original = getattr(Path, method)
        def read(self, *args, _original=original, **kwargs):
            seen.append(threading.get_ident())
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(Path, method, read)
    agent = SelfRepairAgent.__new__(SelfRepairAgent)
    async def generate(*args, **kwargs):
        assert seen and all(t != owner for t in seen), "source read blocked provider's event loop"
        return None, "NOT HERE: no repair required"
    agent.router = SimpleNamespace(has_text_fallback=lambda: provider_available, generate_text=generate)
    result = await agent._generate_repaired_file(Incident("x", "now", "ValueError", "example"), path)
    assert not result.ok
    assert seen and all(t != owner for t in seen)
