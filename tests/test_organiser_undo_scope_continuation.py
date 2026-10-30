"""Undo receipts cannot move files outside their recorded selection."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orion_core.file_organiser as organiser


@pytest.fixture
def state(tmp_path, monkeypatch):
    root = tmp_path / "selected"
    root.mkdir()
    undo = tmp_path / "journal"
    monkeypatch.setattr(organiser, "UNDO_DIR", undo)
    service = organiser.FileOrganiser(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))
    return service, root, undo / "undo_20261006.json"


@pytest.mark.parametrize("interrupted", [False, True])
def test_outside_undo_path_is_rejected_without_moving_any_bytes(state, tmp_path, interrupted):
    service, root, record = state
    outside = tmp_path / "unrelated.txt"
    outside.write_text("belongs elsewhere", encoding="utf-8")
    entry = {"src": str(root / "note.txt"), "dest": str(outside),
             "identity": list(organiser._identity(outside))}
    data = {"target": str(root), "moves": [] if interrupted else [entry],
            "pending": entry if interrupted else None, "complete": not interrupted}
    record.write_text(json.dumps(data), encoding="utf-8")
    before = record.read_bytes()
    result = service._undo_last()
    assert not result.ok
    assert outside.read_text() == "belongs elsewhere"
    assert not (root / "note.txt").exists()
    assert record.read_bytes() == before


def test_invalid_later_entry_does_not_partially_undo_valid_entry(state, tmp_path):
    service, root, record = state
    moved = root / "Documents" / "valid.txt"
    moved.parent.mkdir()
    moved.write_text("valid bytes", encoding="utf-8")
    outside = tmp_path / "unrelated.txt"
    outside.write_text("outside bytes", encoding="utf-8")
    data = {"target": str(root), "complete": True, "moves": [
        {"src": str(root / "invalid.txt"), "dest": str(outside)},
        {"src": str(root / "valid.txt"), "dest": str(moved)}]}
    record.write_text(json.dumps(data), encoding="utf-8")
    assert not service._undo_last().ok
    assert moved.read_text() == "valid bytes"
    assert outside.read_text() == "outside bytes"
    assert record.is_file()


def test_protected_code_path_in_a_receipt_is_preserved(state, monkeypatch):
    service, root, record = state
    moved = root / "Code" / "protected.py"
    moved.parent.mkdir()
    moved.write_text("synthetic protected code", encoding="utf-8")
    monkeypatch.setattr(organiser, "is_protected_path", lambda path: Path(path).resolve() == moved.resolve())
    record.write_text(json.dumps({"target": str(root), "complete": True, "moves": [
        {"src": str(root / "protected.py"), "dest": str(moved)}]}), encoding="utf-8")
    assert not service._undo_last().ok
    assert moved.read_text() == "synthetic protected code"
    assert record.is_file()
