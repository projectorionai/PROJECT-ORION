"""Sorting and undo must preserve colliding files and unfinished recovery."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import orion_core.file_organiser as organiser


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(organiser, "UNDO_DIR", tmp_path / "journal")
    return organiser.FileOrganiser(SimpleNamespace(log=SimpleNamespace(emit=lambda *_: None)))


def make_move(tmp_path, name):
    src = tmp_path / name
    src.write_text(name, encoding="utf-8")
    return organiser.Move(str(src), str(tmp_path / "Documents" / name))


def test_same_second_organisation_keeps_both_undo_records(service, tmp_path, monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 9, 26, 12, 0, 0)
    monkeypatch.setattr(organiser, "datetime", Clock)
    for name in ("one.txt", "two.txt"):
        assert len(service._apply_moves(tmp_path, [make_move(tmp_path, name)])) == 1
    assert len(list(organiser.UNDO_DIR.glob("undo_*.json"))) == 2
    assert service._undo_last().ok
    assert service._undo_last().ok
    assert (tmp_path / "one.txt").read_text() == "one.txt"
    assert (tmp_path / "two.txt").read_text() == "two.txt"


def test_destination_created_after_planning_is_not_overwritten(service, tmp_path):
    move = make_move(tmp_path, "note.txt")
    destination = tmp_path / "Documents" / "note.txt"
    destination.parent.mkdir()
    destination.write_text("belongs to someone else", encoding="utf-8")
    applied = service._apply_moves(tmp_path, [move])
    assert destination.read_text() == "belongs to someone else"
    assert len(applied) == 1 and applied[0].dest != str(destination)
    assert service._undo_last().ok
    assert (tmp_path / "note.txt").read_text() == "note.txt"


def test_missing_undo_source_keeps_journal_for_retry(service, tmp_path):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    moved = tmp_path / "Documents" / "note.txt"
    saved = moved.read_bytes()
    held = tmp_path / "temporarily-held.txt"
    moved.rename(held)
    result = service._undo_last()
    assert not result.ok
    logs = list(organiser.UNDO_DIR.glob("undo_*.json"))
    assert len(logs) == 1 and len(json.loads(logs[0].read_text())["moves"]) == 1
    held.rename(moved)
    assert service._undo_last().ok
    assert (tmp_path / "note.txt").read_bytes() == saved


def test_completed_undo_never_moves_an_unrelated_replacement(service, tmp_path):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    moved = Path(move.dest)
    original = moved.with_name("original-held.txt")
    moved.rename(original)
    moved.write_text("unrelated replacement", encoding="utf-8")

    result = service._undo_last()

    assert not result.ok
    assert moved.read_text() == "unrelated replacement"
    assert original.read_text() == "note.txt"
    assert not Path(move.src).exists()
    logs = list(organiser.UNDO_DIR.glob("undo_*.json"))
    assert len(logs) == 1
    assert len(json.loads(logs[0].read_text())["moves"]) == 1
    # Once the original file is put back, the retained receipt is usable.
    moved.unlink()
    original.rename(moved)
    assert service._undo_last().ok
    assert Path(move.src).read_text() == "note.txt"


def test_completed_undo_preserves_edits_to_the_same_file(service, tmp_path):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.dest).write_text("the user's updated notes", encoding="utf-8")

    assert service._undo_last().ok
    assert Path(move.src).read_text() == "the user's updated notes"


def test_completed_undo_edited_file_can_retry_a_failed_receipt_removal(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.dest).write_text("the user's updated notes", encoding="utf-8")
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    original_unlink = Path.unlink

    def locked_receipt(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic receipt sharing lock")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_receipt)
    assert not service._undo_last().ok
    assert Path(move.src).read_text() == "the user's updated notes"
    monkeypatch.setattr(Path, "unlink", original_unlink)

    assert service._undo_last().ok
    assert not log.exists()
    assert Path(move.src).read_text() == "the user's updated notes"


def test_copied_move_commits_the_destination_file_identity(service, tmp_path, monkeypatch):
    import errno

    move = make_move(tmp_path, "note.txt")
    before = organiser._identity(Path(move.src))

    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "synthetic cross-device move")

    if organiser.os.name == "nt":
        monkeypatch.setattr(organiser.os, "rename", cross_device)
    else:
        monkeypatch.setattr(organiser.os, "link", cross_device)
    service._apply_moves(tmp_path, [move])
    moved = Path(move.dest)
    after = organiser._identity(moved)
    assert before[:2] != after[:2], "the control must copy to a different file identity"
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    committed = json.loads(log.read_text())["moves"][0]

    assert committed["identity"] == list(after)
    assert service._undo_last().ok
    assert Path(move.src).read_text() == "note.txt"


def test_copied_move_replays_its_commit_after_the_final_snapshot_fails(service, tmp_path, monkeypatch):
    import errno

    move = make_move(tmp_path, "note.txt")
    write = organiser.atomic_write_text
    writes = 0

    def fail_final_snapshot(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic final snapshot failure")
        return write(*args, **kwargs)

    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "synthetic cross-device move")

    monkeypatch.setattr(organiser, "atomic_write_text", fail_final_snapshot)
    if organiser.os.name == "nt":
        monkeypatch.setattr(organiser.os, "rename", cross_device)
    else:
        monkeypatch.setattr(organiser.os, "link", cross_device)
    with pytest.raises(OSError, match="final snapshot"):
        service._apply_moves(tmp_path, [move])
    assert not Path(move.src).exists()
    assert Path(move.dest).read_text() == "note.txt"
    monkeypatch.setattr(organiser, "atomic_write_text", write)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "note.txt"
    assert list(organiser.UNDO_DIR.glob("undo_*.json")) == []


@pytest.mark.parametrize("tamper", ["source_path", "destination_path", "source_identity", "destination_identity",
                                  "missing_destination_identity", "null_destination_identity", "null_source_identity"])
def test_copied_commit_still_requires_the_exact_original_intent(service, tmp_path, tamper):
    moved = tmp_path / "Documents" / "note.txt"
    moved.parent.mkdir()
    moved.write_text("the original file", encoding="utf-8")
    destination_identity = list(organiser._identity(moved))
    source_identity = list(destination_identity)
    source_identity[1] += 1  # A synthetic different inode before the copy.
    intent = {"src": str(tmp_path / "note.txt"), "dest": str(moved),
              "identity": source_identity}
    committed = {**intent, "identity": destination_identity,
                 "source_identity": list(source_identity)}
    if tamper == "source_path":
        committed["src"] = str(tmp_path / "different.txt")
    elif tamper == "destination_path":
        committed["dest"] = str(moved.with_name("different.txt"))
    elif tamper == "source_identity":
        committed["source_identity"][1] += 1
    elif tamper == "destination_identity":
        committed["identity"] = ["not a valid file identity"]
    elif tamper == "missing_destination_identity":
        del committed["identity"]
    elif tamper == "null_destination_identity":
        committed["identity"] = None
    else:
        committed["source_identity"] = None
    log = organiser.UNDO_DIR / "undo_20261016.json"
    log.write_text(json.dumps({"target": str(tmp_path), "moves": [],
                               "pending": None, "complete": False}), encoding="utf-8")
    organiser._append_journal_event(log, {"intent": intent})
    organiser._append_journal_event(log, {"commit": committed})

    assert not service._undo_last().ok
    assert moved.read_text() == "the original file"
    assert not Path(intent["src"]).exists()
    assert log.is_file()


def test_an_interrupted_move_is_recovered_from_its_durable_intent(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    original = service._move_unique

    class Interrupted(BaseException):
        pass

    def interrupt_after_move(*args, **kwargs):
        original(*args, **kwargs)
        raise Interrupted()

    monkeypatch.setattr(service, "_move_unique", interrupt_after_move)
    with pytest.raises(Interrupted):
        service._apply_moves(tmp_path, [move])
    assert not Path(move.src).exists() and Path(move.dest).read_text() == "note.txt"
    monkeypatch.setattr(service, "_move_unique", original)
    assert service._undo_last().ok
    assert (tmp_path / "note.txt").read_text() == "note.txt"
    assert list(organiser.UNDO_DIR.glob("undo_*.json")) == []


def test_uncertain_interrupted_move_never_moves_an_unrelated_replacement(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    original = service._move_unique

    class Interrupted(BaseException):
        pass

    def interrupt_after_move(*args, **kwargs):
        original(*args, **kwargs)
        raise Interrupted()

    monkeypatch.setattr(service, "_move_unique", interrupt_after_move)
    with pytest.raises(Interrupted):
        service._apply_moves(tmp_path, [move])
    moved = Path(move.dest)
    moved.write_text("changed after the interrupted move", encoding="utf-8")
    monkeypatch.setattr(service, "_move_unique", original)
    assert not service._undo_last().ok
    assert moved.read_text() == "changed after the interrupted move"
    assert list(organiser.UNDO_DIR.glob("undo_*.json"))


def test_already_restored_identity_allows_retry_after_a_receipt_failure(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    original = Path.unlink

    def temporarily_locked(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic receipt sharing lock")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", temporarily_locked)
    assert not service._undo_last().ok
    assert (tmp_path / "note.txt").read_text() == "note.txt"
    monkeypatch.setattr(Path, "unlink", original)
    assert service._undo_last().ok
    assert not log.exists()


def test_a_truncated_completion_receipt_retains_the_move_intent(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    original = organiser._append_journal_event

    class Interrupted(BaseException):
        pass

    def partial_commit(path, event):
        if "commit" in event:
            with path.with_suffix(".events").open("ab") as stream:
                stream.write(b'{"comm')
            raise Interrupted()
        original(path, event)

    monkeypatch.setattr(organiser, "_append_journal_event", partial_commit)
    with pytest.raises(Interrupted):
        service._apply_moves(tmp_path, [move])
    assert service._undo_last().ok
    assert Path(move.src).read_text() == "note.txt"


def test_relocated_config_keeps_legacy_undo_records_reachable(service, tmp_path, monkeypatch):
    old = organiser.UNDO_DIR
    service._apply_moves(tmp_path, [make_move(tmp_path, "note.txt")])
    config = tmp_path / "new_config"
    active = config / "file_organiser"
    active.mkdir(parents=True)
    monkeypatch.setattr(organiser, "CONFIG_DIR", config)
    monkeypatch.setattr(organiser, "UNDO_DIR", active)
    monkeypatch.setattr(organiser, "LEGACY_UNDO_DIR", old)
    assert service._undo_last().ok
    assert (tmp_path / "note.txt").read_text() == "note.txt"


def test_restore_collision_can_retry_after_receipt_removal_fails(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.src).write_text("new top-level file", encoding="utf-8")
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    unlink = Path.unlink

    def locked_receipt(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic undo receipt sharing lock")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_receipt)
    assert not service._undo_last().ok
    assert Path(move.src).read_text() == "new top-level file"
    restored = tmp_path / "note (1).txt"
    assert restored.read_text() == "note.txt"
    monkeypatch.setattr(Path, "unlink", unlink)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "new top-level file"
    assert restored.read_text() == "note.txt"
    assert not log.exists()
    assert not log.with_suffix(".restore.events").exists()
    assert not (tmp_path / "note (2).txt").exists()


def test_restore_intent_recovers_interruption_after_collision_safe_move(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.src).write_text("new top-level file", encoding="utf-8")
    perform = service._move_unique

    class Interrupted(BaseException):
        pass

    def move_then_interrupt(*args, **kwargs):
        perform(*args, **kwargs)
        raise Interrupted()

    monkeypatch.setattr(service, "_move_unique", move_then_interrupt)
    with pytest.raises(Interrupted):
        service._undo_last()
    monkeypatch.setattr(service, "_move_unique", perform)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "new top-level file"
    assert (tmp_path / "note (1).txt").read_text() == "note.txt"


def test_restore_copied_commit_can_retry_after_receipt_removal_fails(service, tmp_path, monkeypatch):
    import errno

    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    before = organiser._identity(Path(move.dest))
    Path(move.src).write_text("new top-level file", encoding="utf-8")
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    unlink = Path.unlink

    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "synthetic cross-device restore")

    def locked_receipt(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic undo receipt sharing lock")
        return unlink(path, *args, **kwargs)

    if organiser.os.name == "nt":
        monkeypatch.setattr(organiser.os, "rename", cross_device)
    else:
        monkeypatch.setattr(organiser.os, "link", cross_device)
    monkeypatch.setattr(Path, "unlink", locked_receipt)
    assert not service._undo_last().ok
    restored = tmp_path / "note (1).txt"
    assert organiser._identity(restored)[:2] != before[:2]
    monkeypatch.setattr(Path, "unlink", unlink)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "new top-level file"
    assert restored.read_text() == "note.txt"
    assert not log.exists()


def test_restore_partial_retry_keeps_progress_without_rewriting_all_moves(service, tmp_path, monkeypatch):
    moves = [make_move(tmp_path, name) for name in ("one.txt", "two.txt")]
    service._apply_moves(tmp_path, moves)
    Path(moves[0].src).write_text("new top-level file", encoding="utf-8")
    unavailable = Path(moves[1].dest)
    held = unavailable.with_name("temporarily-held.txt")
    unavailable.rename(held)
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    snapshots = []
    write = organiser.atomic_write_text

    def count_snapshot(*args, **kwargs):
        snapshots.append(args[0])
        return write(*args, **kwargs)

    monkeypatch.setattr(organiser, "atomic_write_text", count_snapshot)
    result = service._undo_last()
    assert not result.ok
    assert (tmp_path / "one (1).txt").read_text() == "one.txt"
    assert snapshots == [], "restore progress should append rather than rewrite the full move list"
    assert len(json.loads(log.read_text())["moves"]) == 2
    held.rename(unavailable)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(moves[0].src).read_text() == "new top-level file"
    assert Path(moves[1].src).read_text() == "two.txt"
    assert (tmp_path / "one (1).txt").read_text() == "one.txt"
    assert not log.exists()


def test_restore_committed_progress_does_not_touch_reused_sorted_path(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    unlink = Path.unlink

    def locked_receipt(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic undo receipt sharing lock")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_receipt)
    assert not service._undo_last().ok
    Path(move.dest).write_text("unrelated sorted-path reuse", encoding="utf-8")
    Path(move.src).write_text("the user's later edit", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", unlink)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "the user's later edit"
    assert Path(move.dest).read_text() == "unrelated sorted-path reuse"
    assert not log.exists()


def test_restore_replaced_completed_destination_keeps_receipt_for_review(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    unlink = Path.unlink

    def locked_receipt(path, *args, **kwargs):
        if path == log:
            raise PermissionError("synthetic undo receipt sharing lock")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_receipt)
    assert not service._undo_last().ok
    held = tmp_path / "original-held.txt"
    Path(move.src).rename(held)
    Path(move.src).write_text("replacement restored-path file", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", unlink)

    assert not service._undo_last().ok
    assert Path(move.src).read_text() == "replacement restored-path file"
    assert held.read_text() == "note.txt"
    assert log.is_file()
    Path(move.src).unlink()
    held.rename(move.src)
    assert service._undo_last().ok


@pytest.mark.parametrize("tamper", ["outside_path", "nested_path", "move_mismatch", "intent_identity_missing",
                                  "intent_identity_null", "intent_identity_bool", "commit_mismatch",
                                  "commit_identity_missing", "commit_identity_null", "malformed_line"])
def test_restore_invalid_events_refuse_before_any_file_moves(service, tmp_path, tamper):
    moves = [make_move(tmp_path, name) for name in ("one.txt", "two.txt")]
    service._apply_moves(tmp_path, moves)
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    receipt = json.loads(log.read_text())["moves"][0]
    intent = {"move": receipt, "path": str(tmp_path / "one (1).txt"),
              "identity": list(organiser._identity(Path(receipt["dest"])))}
    commit = {"intent": json.loads(json.dumps(intent)), "identity": list(intent["identity"])}
    events = [{"restore_intent": intent}]
    if tamper == "outside_path":
        intent["path"] = str(tmp_path.parent / "unrelated.txt")
    elif tamper == "nested_path":
        intent["path"] = str(tmp_path / "Other" / "unrelated.txt")
    elif tamper == "move_mismatch":
        intent["move"] = {**receipt, "src": str(tmp_path / "different.txt")}
    elif tamper == "intent_identity_missing":
        del intent["identity"]
    elif tamper == "intent_identity_null":
        intent["identity"] = None
    elif tamper == "intent_identity_bool":
        intent["identity"][1] = True
    elif tamper.startswith("commit_"):
        events.append({"restore_commit": commit})
        if tamper == "commit_mismatch":
            commit["intent"]["path"] = str(tmp_path / "different.txt")
        elif tamper == "commit_identity_missing":
            del commit["identity"]
        else:
            commit["identity"] = None
    sidecar = log.with_suffix(".restore.events")
    sidecar.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    if tamper == "malformed_line":
        with sidecar.open("ab") as stream:
            stream.write(b"invalid complete event\n")
    before = log.read_bytes(), sidecar.read_bytes()

    assert not service._undo_last().ok
    for move in moves:
        assert not Path(move.src).exists()
        assert Path(move.dest).read_text() == Path(move.src).name
    assert (log.read_bytes(), sidecar.read_bytes()) == before


def test_restore_receipt_append_failure_prevents_file_movement(service, tmp_path, monkeypatch):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    sidecar = log.with_suffix(".restore.events")
    open_file = Path.open

    def no_restore_log(path, mode="r", *args, **kwargs):
        if path == sidecar and "a" in mode:
            raise PermissionError("synthetic restore receipt unavailable")
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_restore_log)

    assert not service._undo_last().ok
    assert not Path(move.src).exists()
    assert Path(move.dest).read_text() == "note.txt"
    assert log.is_file()


@pytest.mark.parametrize("interruption", ["write_failure", "truncated_json", "missing_newline"])
def test_restore_interrupted_commit_append_can_resume_same_file(service, tmp_path, monkeypatch, interruption):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.src).write_text("new top-level file", encoding="utf-8")
    append = getattr(organiser, "_append_restore_event", None)

    class Interrupted(BaseException):
        pass

    def interrupt_commit(path, event):
        if "restore_commit" in event:
            if interruption == "write_failure":
                raise OSError("synthetic restore commit failure")
            data = (json.dumps(event) + "\n").encode("utf-8")
            partial = data[:20] if interruption == "truncated_json" else data[:-1]
            with path.with_suffix(".restore.events").open("ab") as stream:
                stream.write(partial)
                stream.flush()
                organiser.os.fsync(stream.fileno())
            raise Interrupted()
        return append(path, event)

    monkeypatch.setattr(organiser, "_append_restore_event", interrupt_commit, raising=False)
    if interruption == "write_failure":
        assert not service._undo_last().ok
    else:
        with pytest.raises(Interrupted):
            service._undo_last()
    assert (tmp_path / "note (1).txt").read_text() == "note.txt"
    monkeypatch.setattr(organiser, "_append_restore_event", append)

    result = service._undo_last()

    assert result.ok, result.text
    assert Path(move.src).read_text() == "new top-level file"
    assert (tmp_path / "note (1).txt").read_text() == "note.txt"
    assert not (tmp_path / "note (2).txt").exists()


def test_restore_uncommitted_cross_device_copy_is_kept_for_review(service, tmp_path, monkeypatch):
    import errno

    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    Path(move.src).write_text("new top-level file", encoding="utf-8")
    append = getattr(organiser, "_append_restore_event", None)

    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "synthetic cross-device restore")

    def fail_commit(path, event):
        if "restore_commit" in event:
            raise OSError("synthetic restore commit failure")
        return append(path, event)

    if organiser.os.name == "nt":
        monkeypatch.setattr(organiser.os, "rename", cross_device)
    else:
        monkeypatch.setattr(organiser.os, "link", cross_device)
    monkeypatch.setattr(organiser, "_append_restore_event", fail_commit, raising=False)
    assert not service._undo_last().ok
    restored = tmp_path / "note (1).txt"
    identity = organiser._identity(restored)
    monkeypatch.setattr(organiser, "_append_restore_event", append)

    assert not service._undo_last().ok
    assert organiser._identity(restored) == identity
    assert restored.read_text() == "note.txt"
    assert Path(move.src).read_text() == "new top-level file"
    assert not (tmp_path / "note (2).txt").exists()
    assert list(organiser.UNDO_DIR.glob("undo_*.json"))


def test_restore_progress_appends_two_receipts_per_file_without_full_snapshots(service, tmp_path, monkeypatch):
    moves = [make_move(tmp_path, f"note{index}.txt") for index in range(16)]
    service._apply_moves(tmp_path, moves)
    snapshots, events = [], []
    append = getattr(organiser, "_append_restore_event", None)
    write = organiser.atomic_write_text

    def count_snapshot(*args, **kwargs):
        snapshots.append(args[0])
        return write(*args, **kwargs)

    def count_event(path, event):
        events.append(event)
        return append(path, event)

    monkeypatch.setattr(organiser, "atomic_write_text", count_snapshot)
    monkeypatch.setattr(organiser, "_append_restore_event", count_event, raising=False)

    assert service._undo_last().ok
    assert snapshots == []
    assert len(events) == 2 * len(moves)
    assert sum("restore_intent" in event for event in events) == len(moves)
    assert sum("restore_commit" in event for event in events) == len(moves)


def test_restore_forged_commit_cannot_complete_while_original_remains_sorted(service, tmp_path):
    move = make_move(tmp_path, "note.txt")
    service._apply_moves(tmp_path, [move])
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("unrelated bytes", encoding="utf-8")
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    receipt = json.loads(log.read_text())["moves"][0]
    intent = {"move": receipt, "path": str(unrelated),
              "identity": list(organiser._identity(Path(move.dest)))}
    commit = {"intent": intent, "identity": list(organiser._identity(unrelated))}
    log.with_suffix(".restore.events").write_text(
        json.dumps({"restore_intent": intent}) + "\n" +
        json.dumps({"restore_commit": commit}) + "\n", encoding="utf-8")

    result = service._undo_last()

    assert not result.ok
    assert not Path(move.src).exists()
    assert Path(move.dest).read_text() == "note.txt"
    assert unrelated.read_text() == "unrelated bytes"
    assert log.is_file()


@pytest.mark.parametrize("failed_event", ["restore_intent", "restore_commit"])
def test_restore_partial_append_failure_stops_before_the_next_file(service, tmp_path, monkeypatch, failed_event):
    moves = [make_move(tmp_path, name) for name in ("one.txt", "two.txt")]
    service._apply_moves(tmp_path, moves)
    log = next(organiser.UNDO_DIR.glob("undo_*.json"))
    append = getattr(organiser, "_append_restore_event", None)
    failed = False

    def partial_append(path, event):
        nonlocal failed
        if failed_event in event and not failed:
            failed = True
            with path.with_suffix(".restore.events").open("ab") as stream:
                stream.write(b'{"' + failed_event.encode("ascii")[:8])
                stream.flush()
                organiser.os.fsync(stream.fileno())
            raise OSError("synthetic partial restore receipt write")
        return append(path, event)

    monkeypatch.setattr(organiser, "_append_restore_event", partial_append, raising=False)

    assert not service._undo_last().ok
    assert not Path(moves[0].src).exists(), "receipt failure must stop before the next file"
    assert Path(moves[0].dest).read_text() == "one.txt"
    assert log.with_suffix(".restore.events").read_bytes().endswith(b'{"' + failed_event.encode("ascii")[:8])
    if failed_event == "restore_intent":
        assert not Path(moves[1].src).exists()
        assert Path(moves[1].dest).read_text() == "two.txt"
    else:
        assert Path(moves[1].src).read_text() == "two.txt"
        assert not Path(moves[1].dest).exists()
    monkeypatch.setattr(organiser, "_append_restore_event", append)

    result = service._undo_last()

    assert result.ok, result.text
    assert [Path(move.src).read_text() for move in moves] == ["one.txt", "two.txt"]
    assert not log.exists()
