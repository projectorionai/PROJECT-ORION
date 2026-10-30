"""Readback and recovery receipts must survive ordinary self-repair IO faults."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import atomic_io, selfrepair as module


@pytest.fixture
def environment(tmp_path, monkeypatch):
    package = tmp_path / "package"
    package.mkdir()
    directory = tmp_path / "repairs"
    monkeypatch.setattr(module, "PACKAGE_DIR", package)
    monkeypatch.setattr(module, "SELF_REPAIR_DIR", directory)
    monkeypatch.setattr(module, "JOURNAL_PATH", directory / "journal.jsonl")
    monkeypatch.setattr(module.SelfRepairAgent, "BACKUP_DIR", directory / "backups")
    signal = SimpleNamespace(emit=lambda *_: None)
    agent = module.SelfRepairAgent(SimpleNamespace(log=signal, banner=signal), None, None)
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = module.Incident("repair", "now", "ValueError", "synthetic", file=str(target), repaired_content="VALUE = 2\n")
    return agent, target, value


def test_intervening_source_edit_during_backup_is_preserved(environment, monkeypatch):
    agent, target, value = environment
    original = atomic_io.atomic_write_bytes
    def write(path, data):
        original(path, data)
        if Path(path).parent == agent.BACKUP_DIR:
            target.write_bytes(b"VALUE = 88\n")
    monkeypatch.setattr(atomic_io, "atomic_write_bytes", write)
    assert not agent._apply_repaired(value, target).ok
    assert target.read_bytes() == b"VALUE = 88\n"
    assert not value.applied


def test_altered_valid_source_is_not_reported_as_the_requested_repair(environment, monkeypatch):
    agent, target, value = environment
    original = atomic_io.atomic_write_bytes
    def write(path, data):
        original(path, data)
        if Path(path) == target and data == b"VALUE = 2\n":
            target.write_bytes(b"VALUE = 88\n")
    monkeypatch.setattr(atomic_io, "atomic_write_bytes", write)
    assert not agent._apply_repaired(value, target).ok
    assert target.read_bytes() == b"VALUE = 88\n"
    assert not value.applied


def test_failed_restore_receipt_survives_a_failed_log_observer(environment, monkeypatch):
    agent, target, value = environment
    value.repaired_content = "def broken(:\n"
    original = atomic_io.atomic_write_bytes
    def write(path, data):
        if Path(path) == target and data == b"VALUE = 1\n":
            raise OSError("synthetic restore lock")
        return original(path, data)
    monkeypatch.setattr(atomic_io, "atomic_write_bytes", write)
    def unavailable(*_):
        raise RuntimeError("synthetic log observer failure")
    agent.bus.log = SimpleNamespace(emit=unavailable)
    receipt = agent._apply_repaired(value, target)
    assert not receipt.ok and "restoring the original ALSO failed" in receipt.text
    backup = next(agent.BACKUP_DIR.glob("*.bak"))
    assert str(backup) in receipt.text and backup.read_bytes() == b"VALUE = 1\n"
    assert not value.applied
