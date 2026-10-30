"""Self-repair must preserve distinct originals and intervening source edits."""
from datetime import datetime as RealDatetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import selfrepair as module


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
    bus = SimpleNamespace(log=signal, banner=signal)
    agent = module.SelfRepairAgent(bus, None, None)
    return agent, package


def incident(agent, path, name, corrected):
    value = module.Incident(name, "synthetic", "ValueError", "synthetic fault", file=str(path), repaired_content=corrected)
    agent._incidents[name] = value
    return value


def test_same_second_equal_basenames_keep_distinct_verified_backups(environment, monkeypatch):
    agent, package = environment
    class FrozenDatetime:
        @staticmethod
        def now():
            return RealDatetime(2026, 10, 7, 12)
    monkeypatch.setattr(module, "datetime", FrozenDatetime)
    paths = [package / "one" / "same.py", package / "two" / "same.py"]
    incidents = []
    for index, path in enumerate(paths):
        path.parent.mkdir()
        path.write_bytes(f"VALUE = {index}\n".encode())
        value = incident(agent, path, str(index), "VALUE = 99\n")
        assert agent._apply_repaired(value, path).ok
        incidents.append(value)
    assert incidents[0].backup_path != incidents[1].backup_path
    for index, value in enumerate(incidents):
        assert Path(value.backup_path).read_bytes() == f"VALUE = {index}\n".encode()


async def test_generated_draft_cannot_overwrite_source_changed_during_generation(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    async def generate(*_, **__):
        target.write_bytes(b"VALUE = 88\n")
        return None, "VALUE = 2\n"
    agent.router = SimpleNamespace(has_text_fallback=lambda: True, generate_text=generate)
    value = incident(agent, target, "draft", "")
    assert (await agent._generate_repaired_file(value, target)).ok
    assert not agent._apply_repaired(value, target).ok
    assert target.read_bytes() == b"VALUE = 88\n"
    assert not value.applied


def test_revert_refuses_intervening_source_changes(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = incident(agent, target, "repair", "VALUE = 2\n")
    assert agent._apply_repaired(value, target).ok
    target.write_bytes(b"VALUE = 88\n")
    assert not agent.revert_last().ok
    assert target.read_bytes() == b"VALUE = 88\n"
    assert value.applied


def test_revert_refuses_a_modified_backup(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = incident(agent, target, "repair", "VALUE = 2\n")
    assert agent._apply_repaired(value, target).ok
    Path(value.backup_path).write_bytes(b"VALUE = 88\n")
    assert not agent.revert_last().ok
    assert target.read_bytes() == b"VALUE = 2\n"
    assert value.applied


def test_completed_source_repair_keeps_receipt_when_optional_observer_raises(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = incident(agent, target, "repair", "VALUE = 2\n")
    def unavailable(*_):
        raise RuntimeError("synthetic dashboard observer failure")
    agent.bus.banner = SimpleNamespace(emit=unavailable)
    receipt = agent._apply_repaired(value, target)
    assert receipt.ok and value.applied
    assert target.read_bytes() == b"VALUE = 2\n"
    assert Path(value.backup_path).read_bytes() == b"VALUE = 1\n"


def test_revert_chooses_last_applied_rather_than_last_created_incident(environment, monkeypatch):
    agent, package = environment
    class SequenceDatetime:
        counter = 0
        @classmethod
        def now(cls):
            result = RealDatetime(2026, 10, 7, 12) + timedelta(seconds=cls.counter)
            cls.counter += 1
            return result
    monkeypatch.setattr(module, "datetime", SequenceDatetime)
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    earlier = incident(agent, target, "earlier-created", "VALUE = 3\n")
    later = incident(agent, target, "later-created", "VALUE = 2\n")
    assert agent._apply_repaired(later, target).ok
    assert agent._apply_repaired(earlier, target).ok
    assert agent.revert_last().ok
    assert target.read_bytes() == b"VALUE = 2\n"
    assert not earlier.applied and later.applied


def test_repeated_apply_retains_the_first_original_backup(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\n")
    value = incident(agent, target, "repair", "VALUE = 2\n")
    assert agent._apply_repaired(value, target).ok
    backup = value.backup_path
    assert agent._apply_repaired(value, target).ok
    assert value.backup_path == backup
    assert Path(backup).read_bytes() == b"VALUE = 1\n"
    assert len(list(agent.BACKUP_DIR.glob("*.bak"))) == 1
    assert agent.revert_last().ok
    assert target.read_bytes() == b"VALUE = 1\n"


def test_successful_windows_text_draft_can_be_reverted_by_its_actual_byte_identity(environment):
    agent, package = environment
    target = package / "module.py"
    target.write_bytes(b"VALUE = 1\r\n")
    value = incident(agent, target, "repair", "VALUE = 2\r\n")
    assert agent._apply_repaired(value, target).ok
    assert target.read_bytes() == b"VALUE = 2\r\n"
    assert agent.revert_last().ok
    assert target.read_bytes() == b"VALUE = 1\r\n"
