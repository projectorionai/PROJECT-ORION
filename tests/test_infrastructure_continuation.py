"""Synthetic acceptance reproductions for the implementation continuation."""
from __future__ import annotations

import asyncio
import io
import sys
from datetime import datetime, timezone
import struct
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def london_2026():
    transitions = [int(datetime(2026, month, day, 1, tzinfo=timezone.utc).timestamp())
                   for month, day in ((3, 29), (10, 25))]
    header = b"TZif\x00" + b"\x00" * 15 + struct.pack(">6l", 0, 0, 0, 2, 2, 8)
    records = struct.pack(">2l", *transitions) + bytes((1, 0))
    records += struct.pack(">lbb", 0, 0, 0) + struct.pack(">lbb", 3600, 1, 4)
    return ZoneInfo.from_file(io.BytesIO(header + records + b"GMT\x00BST\x00"), key="UK-2026")


def test_empty_model_response_does_not_replace_a_usable_model(tmp_path):
    from orion_core.model_store import ModelDownloadError, download_verified
    target = tmp_path / "model.bin"
    target.write_bytes(b"previous verified model")
    with pytest.raises(ModelDownloadError):
        download_verified("synthetic", target, opener=lambda *_, **__: io.BytesIO(b""))
    assert target.read_bytes() == b"previous verified model"
    assert list(tmp_path.iterdir()) == [target]


def test_cron_skips_a_missing_spring_wall_time():
    from orion_core.plugin_schedule import parse_cron
    zone = london_2026()
    now = datetime(2026, 3, 29, 0, 45, tzinfo=zone)
    due = parse_cron("30 1 * * *").next_due(now)
    assert due.date().isoformat() == "2026-03-30", due.isoformat()


def test_cron_can_use_the_remaining_autumn_fold():
    from orion_core.plugin_schedule import parse_cron
    zone = london_2026()
    now = datetime(2026, 10, 25, 1, 45, tzinfo=zone, fold=0)
    due = parse_cron("30 1 * * *").next_due(now)
    assert due.timestamp() - now.timestamp() == 45 * 60, due.isoformat()
    assert due.fold == 1


def test_interval_means_elapsed_seconds_across_autumn_transition():
    from orion_core.plugin_schedule import parse_schedule
    zone = london_2026()
    now = datetime(2026, 10, 25, 0, 30, tzinfo=zone)
    due = parse_schedule(interval_seconds=7200).next_due(now)
    assert due.timestamp() - now.timestamp() == 7200
    assert datetime.fromtimestamp(due.timestamp(), zone) == due


def organiser_service(tmp_path, monkeypatch):
    import orion_core.file_organiser as organiser
    monkeypatch.setattr(organiser, "UNDO_DIR", tmp_path / "journal")
    bus = SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None))
    service = organiser.FileOrganiser(bus)
    folder = tmp_path / "Downloads"
    folder.mkdir()
    monkeypatch.setattr(service, "_resolve_target", lambda _folder: folder)
    return service, folder


def test_unreadable_folder_is_not_announced_as_already_tidy(tmp_path, monkeypatch):
    service, folder = organiser_service(tmp_path, monkeypatch)
    original = Path.iterdir
    def unavailable(path):
        if path == folder:
            raise PermissionError("synthetic unreadable folder")
        return original(path)
    monkeypatch.setattr(Path, "iterdir", unavailable)
    result = asyncio.run(service.organise(str(folder)))
    assert not result.ok and "tidy" not in result.text.lower()


def test_failed_moves_are_not_announced_as_successful_organisation(tmp_path, monkeypatch):
    service, folder = organiser_service(tmp_path, monkeypatch)
    target = folder / "note.txt"
    target.write_text("keep these bytes", encoding="utf-8")
    def fail(*_, **__):
        raise PermissionError("synthetic move failure")
    monkeypatch.setattr(service, "_move_unique", fail)
    result = asyncio.run(service.organise(str(folder), apply=True))
    assert not result.ok
    assert target.read_text(encoding="utf-8") == "keep these bytes"


def test_undo_journal_failure_happens_before_any_file_is_moved(tmp_path, monkeypatch):
    import orion_core.file_organiser as organiser
    service, folder = organiser_service(tmp_path, monkeypatch)
    target = folder / "note.txt"
    target.write_text("keep these bytes", encoding="utf-8")
    def fail(*_, **__):
        raise OSError("synthetic journal disk full")
    monkeypatch.setattr(organiser, "atomic_write_text", fail)
    try:
        result = asyncio.run(service.organise(str(folder), apply=True))
    except OSError:
        result = None
    assert target.exists(), "The source moved before a writable recovery journal was established"
    assert result is not None and not result.ok


def test_documents_target_is_not_skipped_as_an_already_sorted_subfolder(tmp_path, monkeypatch):
    service, folder = organiser_service(tmp_path, monkeypatch)
    documents = folder.with_name("Documents")
    folder.rename(documents)
    (documents / "photo.jpg").write_bytes(b"synthetic photo")
    monkeypatch.setattr(service, "_resolve_target", lambda _folder: documents)
    result = asyncio.run(service.preview(str(documents)))
    assert "would move" in result.text.lower(), result.text
    assert (documents / "photo.jpg").exists()
