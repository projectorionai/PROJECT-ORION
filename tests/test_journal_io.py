"""Large journals stay cheap to read, and fault storms do not fill the disk."""
from __future__ import annotations

import io
import json
from pathlib import Path

from orion_core.journal_io import tail_lines


def test_tail_handles_unicode_crlf_and_no_final_newline(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_bytes("older\r\nαβ\r\nlast".encode())
    assert tail_lines(path, 2) == ["αβ", "last"]
    assert tail_lines(path, 0) == []
    assert tail_lines(path, -1) == []
    assert tail_lines(tmp_path / "missing", 10) == []


def test_tail_never_reads_more_than_its_budget(monkeypatch):
    class Counted(io.BytesIO):
        bytes_read = 0
        def read(self, size=-1):
            assert size >= 0
            data = super().read(size)
            self.bytes_read += len(data)
            return data
        def close(self):
            pass
    stream = Counted(b"old\n" * 500_000 + b"one\ntwo\n")
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: stream)
    assert tail_lines(Path("log"), 2, max_bytes=128) == ["one", "two"]
    assert stream.bytes_read <= 128


def test_tail_discards_an_oversized_partial_first_line(tmp_path):
    path = tmp_path / "log"
    path.write_bytes(b"x" * 1000 + b"\nlast\n")
    assert tail_lines(path, 5, max_bytes=20) == ["last"]
    path.write_bytes(b"x" * 1000)
    assert tail_lines(path, 5, max_bytes=20) == []


def test_repeated_fault_captures_are_sampled_but_resolutions_are_kept(tmp_path, monkeypatch):
    from orion_core import selfrepair as module
    clock = [10.0]
    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(module, "JOURNAL_PATH", path)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    agent = module.SelfRepairAgent.__new__(module.SelfRepairAgent)
    incident = module.Incident(id="sample", at="now", error_type="ExampleError", message="same error")
    for _ in range(100):
        agent._journal("captured", incident)
    agent._journal("resolved", incident)
    clock[0] += 61.0
    agent._journal("captured", incident)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["captured", "resolved", "captured"]
    assert rows[-1]["suppressed_repeats"] == 99
    assert len(agent.journal_tail(2)) == 2


def test_journal_skips_malformed_rows_and_keeps_distinct_faults(tmp_path, monkeypatch):
    from orion_core import selfrepair as module
    path = tmp_path / "journal.jsonl"
    monkeypatch.setattr(module, "JOURNAL_PATH", path)
    agent = module.SelfRepairAgent.__new__(module.SelfRepairAgent)
    for number in range(150):
        agent._journal("captured", module.Incident(
            id=str(number), at="now", error_type="ExampleError", message=str(number)))
    assert len(agent._journal_repeats) == 128
    with path.open("a") as stream:
        stream.write('[]\nnot-json\n{"event":"last"}\n')
    assert agent.journal_tail(3) == [{"event": "last"}]
