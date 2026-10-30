"""Synthetic research receipts and file undo must preserve later external work."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orion_core import atomic_io, dispatch_files, undo
from orion_core.dispatch_files import FilesDispatchMixin
from orion_core.research_runs import ResearchRunStore
from orion_core.memory import OrionMemoryMatrix


class Files(FilesDispatchMixin):
    def _resolve_user_path(self, raw):
        return Path(raw)

    def _ensure_write_safe(self, path):
        return None


@pytest.fixture
def files():
    undo.clear()
    yield Files()
    undo.clear()


@pytest.mark.parametrize("action", ["write_text", "append_text"])
@pytest.mark.parametrize("existed", [False, True])
def test_undo_write_refuses_to_destroy_later_edits(files, tmp_path, action, existed):
    path = tmp_path / "notes.txt"
    if existed:
        path.write_bytes(b"earlier original")
    assert files._file_controller_execute({"action": action, "path": str(path), "text": "ORION edit"}).ok
    path.write_bytes(b"later human work that must survive")
    ok, message = undo.undo_last()
    assert not ok, message
    assert path.read_bytes() == b"later human work that must survive"


@pytest.mark.parametrize("change", ["corrupt", "valid", "missing"])
def test_live_research_writer_preserves_external_ledger_changes(tmp_path, change):
    path = tmp_path / "runs.json"
    store = ResearchRunStore(path)
    try:
        run = store.start("Synthetic investigation", "paper")
        changed = b"{broken external ledger"
        if change == "valid":
            data = json.loads(path.read_bytes())
            data["runs"][0]["result"] = "external newer evidence"
            changed = json.dumps(data).encode()
        if change == "missing":
            path.unlink()
        else:
            path.write_bytes(changed)
        with pytest.raises(OSError, match="changed|removed|unavailable"):
            store.update(run["run_id"], {"research_state": "completed"}, "stale writer completion")
        assert not path.exists() if change == "missing" else path.read_bytes() == changed
    finally:
        store.close()


@pytest.mark.parametrize("existed", [False, True])
def test_undo_preserves_foreign_replacement_even_with_identical_bytes(files, tmp_path, existed):
    path = tmp_path / "notes.txt"
    if existed:
        path.write_bytes(b"original")
    files._file_controller_execute({"action": "write_text", "path": str(path), "text": "ORION edit"})
    identity = path.stat().st_ino
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(path.read_bytes())
    atomic_io._replace(replacement, path)
    assert path.stat().st_ino != identity
    ok, _ = undo.undo_last()
    assert not ok and path.read_bytes() == b"ORION edit"


@pytest.mark.parametrize("change", ["missing", "directory", "reparse"])
def test_undo_refuses_missing_or_nonordinary_targets(files, tmp_path, monkeypatch, change):
    path = tmp_path / "notes.txt"
    files._file_controller_execute({"action": "write_text", "path": str(path), "text": "ORION edit"})
    if change == "missing":
        path.unlink()
    elif change == "directory":
        path.unlink()
        path.mkdir()
    else:
        original = Path.lstat

        def lstat(value, *args, **kwargs):
            info = original(value, *args, **kwargs)
            if value == path:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        monkeypatch.setattr(Path, "lstat", lstat)
    ok, _ = undo.undo_last()
    assert not ok
    if change == "directory":
        assert path.is_dir()
    elif change == "reparse":
        assert path.read_bytes() == b"ORION edit"


def test_consecutive_undos_preserve_exact_binary_original(files, tmp_path):
    path = tmp_path / "notes.txt"
    original = b"\xff\x00original\r\n"
    path.write_bytes(original)
    files._file_controller_execute({"action": "write_text", "path": str(path), "text": "new\n"})
    first = path.read_bytes()
    assert first == ("new" + os.linesep).encode()
    files._file_controller_execute({"action": "append_text", "path": str(path), "text": "tail\n"})
    assert undo.undo_last()[0] and path.read_bytes() == first
    assert undo.undo_last()[0] and path.read_bytes() == original


def test_oversized_result_does_not_retain_an_unbounded_undo(files, tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch_files, "UNDO_MAX_BYTES", 16)
    path = tmp_path / "large.txt"
    result = files._file_controller_execute({"action": "write_text", "path": str(path), "text": "x" * 32})
    assert result.ok and "cannot be undone" in result.text
    assert not undo.can_undo() and path.read_bytes() == b"x" * 32


def test_external_edit_before_undo_registration_is_not_adopted(files, tmp_path, monkeypatch):
    path = tmp_path / "notes.txt"
    snapshot = dispatch_files._undo_file_snapshot

    def changed(target):
        target.write_bytes(b"external work during recording")
        return snapshot(target)

    monkeypatch.setattr(dispatch_files, "_undo_file_snapshot", changed)
    result = files._file_controller_execute({"action": "write_text", "path": str(path), "text": "ORION edit"})
    assert result.ok and "cannot be undone" in result.text
    assert not undo.can_undo() and path.read_bytes() == b"external work during recording"


def test_external_edit_refuses_new_research_admission_and_stays_fail_closed(tmp_path):
    path = tmp_path / "runs.json"
    store = ResearchRunStore(path)
    try:
        store.start("First", "paper")
        path.write_bytes(b"external damage")
        with pytest.raises(OSError, match="changed|unavailable"):
            store.start("Must not be admitted", "paper")
        assert path.read_bytes() == b"external damage"
        with pytest.raises(OSError, match="unavailable"):
            store.start("Still not admitted", "paper")
        assert len(store._rows) == 1
    finally:
        store.close()


def test_same_byte_replaced_ledger_is_preserved_not_claimed(tmp_path):
    path = tmp_path / "runs.json"
    store = ResearchRunStore(path)
    try:
        run = store.start("First", "paper")
        contents = path.read_bytes()
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(contents)
        atomic_io._replace(replacement, path)
        with pytest.raises(OSError, match="changed|unavailable"):
            store.update(run["run_id"], {"research_state": "completed"}, "Must not overwrite")
        assert path.read_bytes() == contents
    finally:
        store.close()


def test_readonly_research_observer_keeps_external_damage_and_does_not_hide_it(tmp_path):
    path = tmp_path / "runs.json"
    owner = ResearchRunStore(path)
    reader = None
    try:
        owner.start("First", "paper")
        reader = ResearchRunStore(path)
        path.write_bytes(b"damaged while owner alive")
        with pytest.raises(OSError, match="unavailable"):
            reader.list()
        with pytest.raises(OSError, match="changed|unavailable"):
            owner.start("Refused stale mutation", "paper")
        assert path.read_bytes() == b"damaged while owner alive"
    finally:
        if reader is not None:
            reader.close()
        owner.close()


@pytest.mark.parametrize("contains,selected", [
    ("%", "percent"), ("_", "underscore"), ("\\", "slash"),
    ("normal", "ordinary"), ("café Δ", "unicode"), ("café_50%\\", "mixed"),
])
def test_memory_forget_contains_is_a_literal_substring(tmp_path, contains, selected):
    signal = SimpleNamespace(emit=lambda *_: None)
    memory = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path / "config", SimpleNamespace(log=signal))
    facts = {"percent": "rate 50% today", "underscore": "project_code", "slash": r"C:\notes\draft",
             "ordinary": "normal item", "unicode": "café Δ", "mixed": "café_50%\\ result"}
    try:
        for key, value in facts.items():
            memory.save("notes", key, value)
        expected = {key for key, value in facts.items() if contains in value}
        assert selected in expected
        assert memory.forget(contains=contains) == len(expected)
        assert {row["key_ref"] for row in memory.records()} == set(facts) - expected
    finally:
        memory.close()


def test_memory_forget_prefix_underscore_does_not_match_another_character(tmp_path):
    signal = SimpleNamespace(emit=lambda *_: None)
    memory = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path / "config", SimpleNamespace(log=signal))
    try:
        for key in ("project_one", "project_two", "projectxone", "other"):
            memory.save("notes", key, key)
        assert memory.forget(key_prefix="project_") == 2
        assert {row["key_ref"] for row in memory.records()} == {"projectxone", "other"}
    finally:
        memory.close()


def test_memory_forget_combines_literal_filters_without_affecting_other_categories(tmp_path):
    signal = SimpleNamespace(emit=lambda *_: None)
    memory = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path / "config", SimpleNamespace(log=signal))
    try:
        memory.save("notes", "task_one", "budget 50%")
        memory.save("notes", "taskxtwo", "budget 50%")
        memory.save("notes", "task_three", "budget 500")
        memory.save("projects", "task_one", "budget 50%")
        assert memory.forget(category="notes", key_prefix="task_", contains="%") == 1
        assert {(row["category"], row["key_ref"]) for row in memory.records()} == {
            ("notes", "taskxtwo"), ("notes", "task_three"), ("projects", "task_one")}
        assert len(memory.query("budget")) == 3  # FTS deletion stays in the same transaction.
    finally:
        memory.close()


def test_memory_forget_punctuation_only_prefix_never_wipes_unrelated_facts(tmp_path):
    signal = SimpleNamespace(emit=lambda *_: None)
    memory = OrionMemoryMatrix(tmp_path / "memory.db", tmp_path / "config", SimpleNamespace(log=signal))
    try:
        memory.save("notes", "entry", "keep me")
        assert memory.forget(key_prefix="_") == 0
        assert memory.records()[0]["value"] == "keep me"
    finally:
        memory.close()
