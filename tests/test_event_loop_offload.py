"""
Blocking file work stays off the event loop.

Under qasync the asyncio loop IS the Qt GUI thread, so a synchronous walk or
parse inside a coroutine freezes the window and the voice path together.

* ``rewind`` re-reads every conversation transcript on each call; the read,
  the JSON parse and the search now run in a worker thread.
* ``learn_folder`` walked the whole tree on the loop, and only then applied
  its file cap; the walk is now lazy, capped as it goes, and off the loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.dispatch_knowledge import KnowledgeDispatchMixin
from orion_core.learning import LearningService
from orion_core.rewind import RewindTimeline


def _write_transcript(folder: Path) -> None:
    rows = [
        ("2026-08-09T10:00:00+00:00", "user", "Should we ship the exe today?"),
        ("2026-08-09T10:00:20+00:00", "assistant", "Let's ship the exe tomorrow."),
    ]
    (folder / "2026-08-09.jsonl").write_text(
        "\n".join(json.dumps({"at": a, "role": r, "content": c}) for a, r, c in rows) + "\n",
        encoding="utf-8")


class _RecordingTimeline(RewindTimeline):
    threads: list[int] = []

    def _load(self):
        type(self).threads.append(threading.get_ident())
        return super()._load()


class _Host(KnowledgeDispatchMixin):
    def __init__(self, timeline):
        self._rewind = timeline


def test_refresh_rereads_transcripts(tmp_path):
    timeline = RewindTimeline(tmp_path)
    assert timeline.refresh() == 0
    _write_transcript(tmp_path)
    assert timeline.refresh() == 2


def test_rewind_reads_and_searches_off_the_event_loop(tmp_path):
    _write_transcript(tmp_path)
    timeline = _RecordingTimeline(tmp_path)
    _RecordingTimeline.threads = []
    host = _Host(timeline)

    async def flow():
        loop_thread = threading.get_ident()
        result = await host.rewind_tool({"action": "search", "query": "ship exe"})
        return loop_thread, result

    loop_thread, result = asyncio.run(flow())
    assert result.ok and "ship" in result.text
    assert _RecordingTimeline.threads, "the transcripts were never loaded"
    assert loop_thread not in _RecordingTimeline.threads


def test_rewind_picks_up_a_transcript_written_mid_session(tmp_path):
    timeline = RewindTimeline(tmp_path)
    host = _Host(timeline)
    first = asyncio.run(host.rewind_tool({"action": "days"}))
    assert "No conversation history" in first.text
    _write_transcript(tmp_path)
    second = asyncio.run(host.rewind_tool({"action": "days"}))
    assert "2026-08-09" in second.text


def test_collect_files_stops_at_the_cap(tmp_path):
    for i in range(30):
        (tmp_path / f"note{i:02d}.txt").write_text("x", encoding="utf-8")
    found = LearningService._collect_files(tmp_path, False, {".txt"}, 5)
    assert len(found) == 5


def test_collect_files_filters_by_suffix_and_skips_folders(tmp_path):
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\0")
    (tmp_path / "folder.md").mkdir()           # a directory with a document suffix
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.md").write_text("x", encoding="utf-8")
    flat = LearningService._collect_files(tmp_path, False, {".md"}, 100)
    deep = LearningService._collect_files(tmp_path, True, {".md"}, 100)
    assert [p.name for p in flat] == ["keep.md"]
    assert sorted(p.name for p in deep) == ["deep.md", "keep.md"]


def test_collect_files_with_no_allowance_is_empty(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert LearningService._collect_files(tmp_path, True, {".txt"}, 0) == []
